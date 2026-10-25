"""Unit tests for TextDetector."""

import cv2
import numpy as np
import pytest

from src.engine.detector import TextDetector


def test_detector_returns_valid_format():
    detector = TextDetector()
    canvas = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    boxes = detector.detect(canvas)
    assert len(boxes) >= 1

    for box in boxes:
        assert len(box) == 4
        top, left, bottom, right = box
        assert isinstance(top, int)
        assert isinstance(left, int)
        assert isinstance(bottom, int)
        assert isinstance(right, int)
        assert top <= bottom
        assert left <= right


def test_detector_fallback_on_blank_image():
    detector = TextDetector()
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    boxes = detector.detect(blank)
    assert len(boxes) == 1
    # Fallback to image boundaries
    assert boxes[0] == [0, 0, 100, 100]


def test_reading_order_sorting():
    detector = TextDetector()
    # Box 1: line 2 left
    # Box 2: line 1 right
    # Box 3: line 1 left
    boxes = [
        [150, 20, 180, 80],
        [30, 200, 60, 260],
        [30, 40, 60, 100],
    ]
    sorted_boxes = detector._sort_reading_order(boxes)

    # Line 1 left should come before line 1 right, and line 2 should come last
    assert sorted_boxes[0] == [30, 40, 60, 100]
    assert sorted_boxes[1] == [30, 200, 60, 260]
    assert sorted_boxes[2] == [150, 20, 180, 80]


def test_detector_regression_border_and_grid_suppression():
    """Regression test: verify outer document borders and table lines do not merge into 1 giant box."""
    detector = TextDetector()

    # Create a 600x800 document with outer border and a table grid
    canvas = np.full((600, 800, 3), 255, dtype=np.uint8)

    # 1. Outer document border
    cv2.rectangle(canvas, (10, 10), (790, 590), (0, 0, 0), 2)

    # 2. Table grid lines
    cv2.rectangle(canvas, (50, 250), (750, 500), (0, 0, 0), 2)
    cv2.line(canvas, (50, 320), (750, 320), (0, 0, 0), 2)  # horizontal rule
    cv2.line(canvas, (350, 250), (350, 500), (0, 0, 0), 2)  # vertical divider

    # 3. Add text inside header and table cells
    cv2.putText(canvas, "INVOICE", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, "COMPANY", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "ITEM ONE", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "PRICE 100", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "ITEM TWO", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "PRICE 200", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    boxes = detector.detect(canvas)

    # Must detect multiple distinct regions, NOT one giant box
    assert len(boxes) >= 5, f"Expected at least 5 detected text regions, got {len(boxes)}"

    # Check that no box covers the outer document border
    for top, left, bottom, right in boxes:
        w = right - left
        h = bottom - top
        assert not (w > 0.85 * 800 and h > 0.5 * 600), "Detected box spans entire document border!"


def test_detector_regression_on_invoice_image():
    """Verify on real invoice test image that multiple word regions are detected."""
    import os
    invoice_path = "sample/invoice_test.png"
    if not os.path.exists(invoice_path):
        pytest.skip(f"Test invoice image not found at {invoice_path}")

    image = cv2.imread(invoice_path)
    assert image is not None

    detector = TextDetector()
    boxes = detector.detect(image)

    # Real invoice has 50+ words across header, billing, table, payment details
    assert len(boxes) >= 20, f"Expected >= 20 text boxes on invoice, got {len(boxes)}"

    # Ensure no box spans the full document
    h_img, w_img = image.shape[:2]
    for top, left, bottom, right in boxes:
        w = right - left
        h = bottom - top
        assert not (w > 0.85 * w_img and h > 0.35 * h_img), "Box encompasses document border"


def test_tiny_noise_filtering():
    """Verify that isolated tiny pixel noise (<15px area) is filtered out."""
    detector = TextDetector(min_area=35)
    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)

    # Draw tiny salt/pepper noise specks (2x2 pixels)
    canvas[50:52, 50:52] = 0
    canvas[150:152, 200:202] = 0
    canvas[250:252, 100:102] = 0

    # Draw one valid word "HELLO"
    cv2.putText(canvas, "HELLO", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    boxes = detector.detect(canvas)
    # The tiny specks must not produce separate bounding boxes
    assert len(boxes) == 1
    t, l, b, r = boxes[0]
    assert (b - t) >= 15
    assert (r - l) >= 40


def test_character_to_word_grouping():
    """Verify that characters on the same baseline with close kerning are grouped into a single word."""
    detector = TextDetector()
    canvas = np.full((200, 500, 3), 255, dtype=np.uint8)

    # Draw two separate words with a wide space between them: "INVOICE" and "TOTAL"
    cv2.putText(canvas, "INVOICE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(canvas, "TOTAL", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    boxes = detector.detect(canvas)

    # Must be grouped into exactly 2 words, NOT 12 individual character boxes
    assert len(boxes) == 2
    # Verify both boxes have word-level widths (> 60px)
    for t, l, b, r in boxes:
        assert (r - l) >= 60


