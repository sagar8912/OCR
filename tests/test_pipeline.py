"""Integration tests for OCREngine and output file generation."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.engine.pipeline import OCREngine


def test_ocr_engine_end_to_end(tmp_path):
    engine = OCREngine(model_path="models/best_model.pt")

    # Create dummy white canvas with a black word
    import cv2
    img = np.full((60, 160, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    sample_file = tmp_path / "test_input.png"
    cv2.imwrite(str(sample_file), img)

    # Execute processing
    result = engine.process(str(sample_file), redact=True)

    assert "results" in result
    assert "redacted_image" in result
    assert "latency_ms" in result
    assert len(result["results"]) >= 1

    first_item = result["results"][0]
    assert "word" in first_item
    assert "bounding_box" in first_item
    top, left, bottom, right = first_item["bounding_box"]
    assert top <= bottom
    assert left <= right

    # Test output saving
    saved = engine.save_outputs(result, input_path=sample_file, output_dir=tmp_path)
    json_path = Path(saved["json_path"])
    redacted_path = Path(saved["redacted_image_path"])

    assert json_path.exists()
    assert redacted_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert isinstance(loaded_json, list)
    assert len(loaded_json) == len(result["results"])
