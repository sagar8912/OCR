"""PyTorch CRNN (CNN + BiLSTM + CTC) Model for CPU-optimized Optical Character Recognition."""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.dataset import LabelConverter, DEFAULT_CONVERTER


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for Sequence Recognition.

    Architecture:
        1. 7-layer CNN feature extractor with Batch Normalization
        2. Sequence map via adaptive pooling to (B, channels, 1, W_seq)
        3. 2-layer Bidirectional LSTM
        4. Linear Projection layer to vocabulary classes
        5. CTC Loss output (T, B, num_classes)
    """

    def __init__(
        self,
        img_channels: int = 1,
        num_classes: int = 37,  # 36 alphanumeric + 1 CTC blank
        hidden_size: int = 256,
        rnn_dropout: float = 0.2,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1: (B, 1, 32, W) -> (B, 64, 16, W/2)
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: (B, 64, 16, W/2) -> (B, 128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: (B, 128, 8, W/4) -> (B, 256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 4: (B, 256, 8, W/4) -> (B, 256, 4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 5: (B, 256, 4, W/4) -> (B, 512, 4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Block 6: (B, 512, 4, W/4) -> (B, 512, 2, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # Collapse height to 1 while preserving sequence width
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # Recurrent sequence model: BiLSTM
        # In features = 512, bidirectional output = 2 * hidden_size
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=rnn_dropout if rnn_dropout > 0 else 0,
        )

        # Output projection layer
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming/Xavier and set blank bias penalty to avoid blank collapse."""
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        # CRITICAL FIX: Penalize initial blank logit to prevent CTC blank collapse trap
        with torch.no_grad():
            self.classifier.bias.data[0] = -3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Log-probabilities of shape (T, B, num_classes) suitable for CTCLoss.
        """
        # CNN feature extraction: (B, C, H, W) -> (B, 512, H', W')
        features = self.cnn(x)

        # Pool height dimension to 1: (B, 512, 1, W_seq)
        features = self.adaptive_pool(features)

        # Squeeze height: (B, 512, W_seq)
        features = features.squeeze(2)

        # Permute to (W_seq, B, 512) for PyTorch RNN
        # Sequence length T = W_seq
        seq_features = features.permute(2, 0, 1)

        # BiLSTM processing: (T, B, 2 * hidden_size)
        rnn_out, _ = self.rnn(seq_features)

        # Linear projection: (T, B, num_classes)
        logits = self.classifier(rnn_out)

        # CTC requires log probabilities
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        converter: Optional[LabelConverter] = None,
    ) -> List[str]:
        """Greedy CTC inference on a batch of images.

        Args:
            x: Input image tensor (B, C, H, W).
            converter: LabelConverter instance for decoding.

        Returns:
            List of decoded string predictions.
        """
        self.eval()
        converter = converter or DEFAULT_CONVERTER
        log_probs = self.forward(x)  # (T, B, num_classes)

        # Argmax over class dimension -> (T, B)
        best_indices = torch.argmax(log_probs, dim=-1)  # (T, B)
        best_indices = best_indices.transpose(0, 1)     # (B, T)

        results = []
        for seq in best_indices:
            results.append(converter.decode_greedy(seq))
        return results


def build_crnn(
    img_channels: int = 1,
    num_classes: int = 37,
    hidden_size: int = 256,
    pretrained_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> CRNN:
    """Factory function to instantiate and optionally load weights into CRNN model."""
    state_dict = None
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "hidden_size" in state_dict:
            hidden_size = state_dict["hidden_size"]

    model = CRNN(
        img_channels=img_channels,
        num_classes=num_classes,
        hidden_size=hidden_size,
    )
    if state_dict is not None:
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

    model.to(device)
    return model
