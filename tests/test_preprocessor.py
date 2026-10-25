"""Unit tests for ImagePreprocessor."""

import numpy as np
import pytest
import torch

from src.engine.preprocessor import ImagePreprocessor


def test_grayscale_conversion():
    preprocessor = ImagePreprocessor(grayscale=True, sharpen=False, resize=False)
    color_img = np.full((100, 200, 3), 128, dtype=np.uint8)
    processed = preprocessor.process(color_img)
    assert len(processed.shape) == 2
    assert processed.shape == (100, 200)


def test_sharpening():
    preprocessor = ImagePreprocessor(grayscale=False, sharpen=True, resize=False)
    img = np.random.randint(50, 200, (60, 60, 3), dtype=np.uint8)
    sharpened = preprocessor.process(img)
    assert sharpened.shape == img.shape
    assert sharpened.dtype == np.uint8


def test_resizing():
    preprocessor = ImagePreprocessor(
        grayscale=False,
        sharpen=False,
        resize=True,
        target_size=(80, 40),  # width, height
    )
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = preprocessor.process(img)
    assert resized.shape == (40, 80, 3)


def test_prepare_for_ocr_tensor():
    preprocessor = ImagePreprocessor()
    crop = np.full((40, 80, 3), 100, dtype=np.uint8)
    tensor = preprocessor.prepare_for_ocr(crop, target_height=32, target_width=128)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 32, 128)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0
