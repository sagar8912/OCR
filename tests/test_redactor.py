"""Unit tests for TextRedactor."""

import cv2
import numpy as np
import pytest

from src.engine.redactor import TextRedactor


def test_mask_generation():
    redactor = TextRedactor(padding=2)
    shape = (100, 200, 3)
    boxes = [[10, 20, 30, 60]]

    mask = redactor.create_mask(shape, boxes)
    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8
    # Within padded box, mask should be 255
    assert mask[15, 25] == 255
    # Outside box, mask should be 0
    assert mask[0, 0] == 0


def test_inpaint_telea_and_ns():
    canvas = np.full((120, 240, 3), 200, dtype=np.uint8)
    cv2.putText(canvas, "SECRET", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    boxes = [[35, 25, 80, 180]]

    # Telea method
    redactor_telea = TextRedactor(method="telea", inpaint_radius=5)
    redacted_t = redactor_telea.redact(canvas, boxes)
    assert redacted_t.shape == canvas.shape
    assert redacted_t.dtype == np.uint8

    # Navier-Stokes method
    redactor_ns = TextRedactor(method="ns", inpaint_radius=5)
    redacted_ns = redactor_ns.redact(canvas, boxes)
    assert redacted_ns.shape == canvas.shape
    assert redacted_ns.dtype == np.uint8


def test_redact_empty_boxes():
    redactor = TextRedactor()
    canvas = np.full((50, 50, 3), 128, dtype=np.uint8)
    result = redactor.redact(canvas, [])
    assert np.array_equal(result, canvas)
