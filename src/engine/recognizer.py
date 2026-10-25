from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import cv2
import numpy as np
import torch

from src.training.model import build_crnn, CRNN
from src.training.dataset import LabelConverter, DEFAULT_CONVERTER
from src.engine.preprocessor import ImagePreprocessor
from src.engine.decoder import LexiconDecoder


class TextRecognizer:
    """CRNN-based word text recognizer with Lexicon-constrained decoding optimized for CPU."""

    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        device: str = "cpu",
        converter: Optional[LabelConverter] = None,
        target_height: int = 32,
        target_width: int = 128,
        batch_size: int = 16,
        use_lexicon: bool = True,
        lexicon_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.converter = converter or DEFAULT_CONVERTER
        self.target_height = target_height
        self.target_width = target_width
        self.batch_size = batch_size
        self.use_lexicon = use_lexicon

        self.preprocessor = ImagePreprocessor()
        self.decoder = LexiconDecoder(lexicon_path=lexicon_path, converter=self.converter) if use_lexicon else None

        # Load CRNN model
        self.model: Optional[CRNN] = None
        self._load_model()

    def _load_model(self):
        """Load trained PyTorch CRNN model weights onto specified device."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path}. "
                f"Please train the model first using: python -m src.training.train"
            )
        self.model = build_crnn(pretrained_path=self.model_path, device=self.device)
        self.model.eval()
        if self.decoder:
            self.decoder.set_model(self.model, self.preprocessor)

    @torch.no_grad()
    def recognize_crop(self, crop: np.ndarray, return_details: bool = False) -> Union[str, Dict[str, Any]]:
        """Transcribe text from a single cropped word image."""
        words = self.recognize_batch([crop], return_details=return_details)
        return words[0] if words else ("" if not return_details else {"word": "", "confidence": 0.0})

    @torch.no_grad()
    def recognize_batch(
        self,
        crops: List[np.ndarray],
        return_details: bool = False,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Transcribe text from a list of cropped word images in batches with lexicon decoding.
        
        Args:
            crops: List of BGR numpy image crops.
            return_details: If True, returns dicts with word, confidence, and method.
            
        Returns:
            List of word strings or list of detail dictionaries.
        """
        if not crops:
            return []

        all_results: List[Dict[str, Any]] = []

        for i in range(0, len(crops), self.batch_size):
            chunk = crops[i : i + self.batch_size]

            # Check if any crop in chunk has a wide aspect ratio (w/h >= 3.5)
            has_wide = any((c.shape[1] / max(c.shape[0], 1) >= 3.5) for c in chunk)

            if has_wide:
                # Process crops individually to preserve aspect ratio for long words/identifiers
                for crop in chunk:
                    h, w = crop.shape[:2]
                    ar = w / max(h, 1)
                    target_w = 224 if ar >= 6.0 else (192 if ar >= 4.5 else (160 if ar >= 3.5 else self.target_width))
                    tensor = self.preprocessor.prepare_for_ocr(
                        crop,
                        target_height=self.target_height,
                        target_width=target_w,
                    ).to(self.device)
                    log_probs = self.model(tensor)
                    greedy_word = self.converter.decode_greedy(torch.argmax(log_probs[:, 0], dim=-1))

                    if self.decoder:
                        dec_res = self.decoder.decode(log_probs, raw_text=greedy_word, crop=crop)
                    else:
                        dec_res = {"word": greedy_word, "raw_word": greedy_word, "confidence": 0.75, "method": "GREEDY"}
                    all_results.append(dec_res)
            else:
                # Fast batched tensor processing for standard crops
                tensors = [
                    self.preprocessor.prepare_for_ocr(
                        crop,
                        target_height=self.target_height,
                        target_width=self.target_width,
                    )
                    for crop in chunk
                ]
                batch_tensor = torch.cat(tensors, dim=0).to(self.device)
                log_probs = self.model(batch_tensor)  # (T, B, C)
                best_indices = torch.argmax(log_probs, dim=-1).transpose(0, 1)  # (B, T)

                for b_idx in range(len(chunk)):
                    greedy_word = self.converter.decode_greedy(best_indices[b_idx])
                    if self.decoder:
                        dec_res = self.decoder.decode(log_probs[:, b_idx : b_idx + 1], raw_text=greedy_word, crop=chunk[b_idx])
                    else:
                        dec_res = {"word": greedy_word, "raw_word": greedy_word, "confidence": 0.75, "method": "GREEDY"}
                    all_results.append(dec_res)

        if return_details:
            return all_results
        return [item["word"] for item in all_results]
