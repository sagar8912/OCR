"""Image preprocessing module supporting grayscale, scaling/resizing, and sharpening."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch


class ImagePreprocessor:
    """Configurable image preprocessor for OCR ingestion and recognition pipelines."""

    def __init__(
        self,
        grayscale: bool = True,
        resize: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = None,
        sharpen: bool = True,
        denoise: bool = False,
    ):
        """Initialize preprocessing parameters.

        Args:
            grayscale: Whether to convert input image to single-channel grayscale.
            resize: Whether to apply resizing.
            target_size: Optional (width, height) tuple for fixed resizing.
            scale_factor: Optional scaling multiplier (e.g. 2.0 to upscale).
            sharpen: Whether to enhance contrast/sharpness of characters.
            denoise: Whether to apply gentle bilateral filter denoising.
        """
        self.grayscale = grayscale
        self.resize = resize
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.sharpen = sharpen
        self.denoise = denoise

        # 3x3 balanced mild sharpening kernel (avoids halo artifacts on thin digits)
        self._sharpen_kernel = np.array(
            [[0, -0.5, 0],
             [-0.5, 3.0, -0.5],
             [0, -0.5, 0]],
            dtype=np.float32,
        )

    def process(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Process an input image or image path.

        Args:
            image: Image as numpy array (BGR or Gray) or filesystem path.

        Returns:
            Preprocessed numpy array.
        """
        if isinstance(image, (str, Path)):
            path_str = str(image)
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Could not load image from path: {path_str}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 1. Grayscale conversion
        if self.grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Gentle Denoising
        if self.denoise:
            if len(img.shape) == 2:
                img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
            else:
                img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

        # 3. Sharpening
        if self.sharpen:
            img = cv2.filter2D(img, ddepth=-1, kernel=self._sharpen_kernel)

        # 4. Resizing / Scaling
        if self.resize:
            if self.target_size is not None:
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
            elif self.scale_factor is not None and self.scale_factor > 0:
                h, w = img.shape[:2]
                new_w = max(1, int(w * self.scale_factor))
                new_h = max(1, int(h * self.scale_factor))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return img

    def prepare_for_ocr(
        self,
        crop: np.ndarray,
        target_height: int = 32,
        target_width: int = 128,
    ) -> torch.Tensor:
        """Transform a cropped word patch into a normalized PyTorch tensor (1, 1, H, W).

        Aspect-ratio preserving resize with zero padding.
        """
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return torch.zeros((1, 1, target_height, target_width), dtype=torch.float32)

        # Direct resize to (target_width, target_height) consistent with training
        final_img = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Normalize to [-1.0, 1.0]
        norm = (final_img.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor
