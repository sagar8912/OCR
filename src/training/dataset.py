"""Dataset, Vocabulary, and DataLoader implementation for IIIT-5K OCR."""

import os
import re
import random
from typing import List, Tuple, Dict, Optional, Union, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Canonical vocabulary: 0-9 and A-Z
VOCAB_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = 0


class LabelConverter:
    """Encodes text strings to CTC integer labels and decodes model output sequences."""

    def __init__(self, chars: str = VOCAB_CHARS):
        self.chars = chars
        self.blank_idx = BLANK_IDX
        # Character indices start at 1; 0 is reserved for CTC blank
        self.char2idx: Dict[str, int] = {c: i + 1 for i, c in enumerate(chars)}
        self.idx2char: Dict[int, str] = {i + 1: c for i, c in enumerate(chars)}
        self.num_classes = len(chars) + 1  # 36 chars + 1 blank = 37

    def encode(self, text: str) -> List[int]:
        """Convert a text string to a list of integer indices (uppercased, non-vocab ignored)."""
        clean_text = str(text).upper()
        return [self.char2idx[c] for c in clean_text if c in self.char2idx]

    def decode_greedy(self, pred_indices: Union[List[int], np.ndarray, torch.Tensor]) -> str:
        """Perform standard greedy CTC decoding (collapse repeated tokens, remove blanks)."""
        if isinstance(pred_indices, torch.Tensor):
            pred_indices = pred_indices.cpu().numpy()
        decoded_chars: List[str] = []
        prev_idx = -1
        for idx in pred_indices:
            idx = int(idx)
            if idx != prev_idx:
                if idx != self.blank_idx and idx in self.idx2char:
                    decoded_chars.append(self.idx2char[idx])
                prev_idx = idx
        return "".join(decoded_chars)

    def decode_batch(self, preds: torch.Tensor) -> List[str]:
        """Decode a batch of predictions from shape (T, B, C) or (B, T, C)."""
        # If (T, B, C), permute to (B, T, C)
        if preds.dim() == 3 and preds.size(1) != self.num_classes and preds.size(2) == self.num_classes:
            # Already (B, T, C) or (T, B, C)?
            pass
        # Best class indices
        best_indices = torch.argmax(preds, dim=-1)  # shape (T, B) or (B, T)
        if best_indices.size(0) > best_indices.size(1) and preds.size(0) != best_indices.size(1):
            # Probably (T, B), transpose to (B, T)
            best_indices = best_indices.transpose(0, 1)

        results = []
        for seq in best_indices:
            results.append(self.decode_greedy(seq))
        return results


# Global singleton converter instance for reuse
DEFAULT_CONVERTER = LabelConverter()


class IIIT5KDataset(Dataset):
    """PyTorch Dataset for IIIT-5K word images."""

    def __init__(
        self,
        dataframe: Union[pd.DataFrame, str],
        img_root: str = "IIIT5K-Word_V3.0/IIIT5K",
        converter: Optional[LabelConverter] = None,
        target_height: int = 32,
        target_width: int = 128,
        is_train: bool = False,
        cache_in_memory: bool = True,
    ):
        if isinstance(dataframe, str):
            self.df = pd.read_csv(dataframe)
        else:
            self.df = dataframe.copy().reset_index(drop=True)

        self.img_root = img_root
        self.converter = converter or DEFAULT_CONVERTER
        self.target_height = target_height
        self.target_width = target_width
        self.is_train = is_train
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[int, np.ndarray] = {}

        # Filter out rows with empty or non-alphanumeric ground truth
        valid_indices = []
        for idx, row in self.df.iterrows():
            gt = str(row.get("GroundTruth", ""))
            encoded = self.converter.encode(gt)
            if len(encoded) > 0:
                valid_indices.append(idx)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply mild, readable data augmentations during training."""
        # 1. Subtle brightness / contrast adjustment
        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)  # Contrast
            beta = random.uniform(-15, 15)      # Brightness
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        # 2. Slight rotation (-3 to +3 degrees)
        if random.random() < 0.3:
            angle = random.uniform(-3.0, 3.0)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h), borderMode=cv2.BORDER_REPLICATE
            )

        # 3. Mild Gaussian Blur
        if random.random() < 0.2:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Grayscale, aspect-preserving resize to target_height, and pad/resize to target_width."""
        # Ensure single channel grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.is_train:
            gray = self._apply_augmentation(gray)

        # Direct resize to (target_width, target_height) spreads character strokes
        # evenly across all sequence time steps without artificial zero-padding artifacts.
        processed = cv2.resize(gray, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)

        # Normalize to [-1.0, 1.0] range
        normalized = (processed.astype(np.float32) / 127.5) - 1.0
        # Expand channel dimension (1, H, W)
        tensor_img = np.expand_dims(normalized, axis=0)
        return tensor_img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_rel_path = str(row["ImgName"]).replace("\\", "/")

        if self.cache_in_memory and idx in self._cache:
            base_img = self._cache[idx]
        else:
            full_img_path = os.path.join(self.img_root, img_rel_path)
            base_img = cv2.imread(full_img_path)
            if base_img is None:
                base_img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            if self.cache_in_memory:
                self._cache[idx] = base_img

        processed_img = self._preprocess_image(base_img)
        raw_text = str(row["GroundTruth"]).upper()
        encoded_target = self.converter.encode(raw_text)

        return {
            "image": torch.tensor(processed_img, dtype=torch.float32),
            "target": torch.tensor(encoded_target, dtype=torch.long),
            "target_length": len(encoded_target),
            "raw_text": raw_text,
            "img_name": img_rel_path,
        }


def ctc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch items into tensors suitable for PyTorch CTCLoss."""
    images = torch.stack([item["image"] for item in batch], dim=0)  # (B, 1, H, W)
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    targets = torch.cat([item["target"] for item in batch], dim=0)  # 1D flattened targets
    raw_texts = [item["raw_text"] for item in batch]
    img_names = [item["img_name"] for item in batch]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "raw_texts": raw_texts,
        "img_names": img_names,
    }


def get_dataloaders(
    train_csv: str = "traindata.csv",
    test_csv: str = "testdata.csv",
    img_root: str = "IIIT5K-Word_V3.0/IIIT5K",
    batch_size: int = 32,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    # Ensure relative paths resolve properly
    if not os.path.exists(img_root):
        alt_root = os.path.join("..", img_root)
        if os.path.exists(alt_root):
            img_root = alt_root

    full_train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Deterministic reproducible train / val split
    shuffled_df = full_train_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_size = int(len(shuffled_df) * val_ratio)
    val_df = shuffled_df.iloc[:val_size].reset_index(drop=True)
    train_df = shuffled_df.iloc[val_size:].reset_index(drop=True)

    train_dataset = IIIT5KDataset(train_df, img_root=img_root, is_train=True)
    val_dataset = IIIT5KDataset(val_df, img_root=img_root, is_train=False)
    test_dataset = IIIT5KDataset(test_df, img_root=img_root, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
