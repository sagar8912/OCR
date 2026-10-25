"""Unified OCR and Text Redaction Pipeline Engine."""

import io
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import cv2
import numpy as np
from PIL import Image

from src.engine.preprocessor import ImagePreprocessor
from src.engine.detector import TextDetector
from src.engine.recognizer import TextRecognizer
from src.engine.redactor import TextRedactor


class OCREngine:
    """Unified OCR and Inpainting-based Text Redaction Inference Engine.

    Coordinates preprocessing, text region detection, CRNN sequence transcription,
    and OpenCV Telea/Navier-Stokes inpainting.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        device: str = "cpu",
        inpaint_method: str = "telea",
        inpaint_radius: int = 5,
        sharpen: bool = True,
    ):
        self.model_path = model_path
        self.device = device

        self.preprocessor = ImagePreprocessor(grayscale=False, sharpen=sharpen)
        self.detector = TextDetector()
        self.recognizer = TextRecognizer(model_path=model_path, device=device)
        self.redactor = TextRedactor(method=inpaint_method, inpaint_radius=inpaint_radius)

    def load_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Load image input into a BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to decode image from path: {path}")
            return img
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from byte buffer.")
            return img
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def process(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        redact: bool = True,
    ) -> Dict[str, Any]:
        """Execute full OCR and redaction pipeline on an image.

        Args:
            image_input: Filepath, bytes, numpy array, or PIL image.
            redact: Whether to generate the inpainting-redacted image.

        Returns:
            Dict containing:
                - 'results': List of dicts with 'word' and 'bounding_box' [top, left, bottom, right]
                - 'redacted_image': Redacted BGR image array (or None)
                - 'original_image': Original BGR image array
                - 'latency_ms': Processing time in milliseconds
        """
        start_time = time.perf_counter()

        # 1. Load image
        original_bgr = self.load_image(image_input)
        h_orig, w_orig = original_bgr.shape[:2]

        # 2. Preprocess image for detection
        preprocessed = self.preprocessor.process(original_bgr)

        # 3. Detect text regions
        bounding_boxes = self.detector.detect(preprocessed)

        # 4. Crop detected words
        crops = []
        valid_boxes = []
        for box in bounding_boxes:
            top, left, bottom, right = box
            # Safe 2-3px padding margin for OCR ingestion without altering official bounding box
            pad_y = 2
            pad_x = 3
            c_top = max(0, min(top - pad_y, h_orig))
            c_bottom = max(0, min(bottom + pad_y, h_orig))
            c_left = max(0, min(left - pad_x, w_orig))
            c_right = max(0, min(right + pad_x, w_orig))

            crop = original_bgr[c_top:c_bottom, c_left:c_right]
            if crop.size > 0:
                crops.append(crop)
                valid_boxes.append([int(top), int(left), int(bottom), int(right)])

        # 5. Run OCR recognition on word crops with details
        if crops:
            recognized_details = self.recognizer.recognize_batch(crops, return_details=True)
        else:
            recognized_details = []

        # 6. Format results according to assignment specification
        word_results = []
        for item, box in zip(recognized_details, valid_boxes):
            entry = {
                "word": item["word"],
                "bounding_box": box,  # [top, left, bottom, right]
                "confidence": item.get("confidence", 0.8),
            }
            if "raw_word" in item:
                entry["raw_word"] = item["raw_word"]
            if "field_type" in item:
                entry["field_type"] = item["field_type"]
            word_results.append(entry)

        # 7. Redact image using inpainting
        redacted_image = None
        if redact and valid_boxes:
            redacted_image = self.redactor.redact(original_bgr, valid_boxes)
        elif redact:
            redacted_image = original_bgr.copy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "results": word_results,
            "redacted_image": redacted_image,
            "original_image": original_bgr,
            "latency_ms": round(elapsed_ms, 2),
        }

    def save_outputs(
        self,
        process_result: Dict[str, Any],
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """Save results to <stem>_text.json and <stem>_redacted.png.

        Args:
            process_result: Output from self.process().
            input_path: Path of the input image to derive file stems.
            output_dir: Optional output directory (defaults to input file directory).

        Returns:
            Dict with paths to saved json and image files.
        """
        input_path = Path(input_path)
        stem = input_path.stem
        target_dir = Path(output_dir) if output_dir else input_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        json_path = target_dir / f"{stem}_text.json"
        redacted_path = target_dir / f"{stem}_redacted.png"

        # 1. Save JSON formatted results
        # [{ "word": "test", "bounding_box": [top, left, bottom, right] }]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(process_result["results"], f, indent=2)

        # 2. Save redacted image
        if process_result.get("redacted_image") is not None:
            cv2.imwrite(str(redacted_path), process_result["redacted_image"])

        return {
            "json_path": str(json_path),
            "redacted_image_path": str(redacted_path),
        }
