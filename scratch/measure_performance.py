"""Benchmark latency for Detection, OCR Recognition, and Full Pipeline."""

import time
import cv2
import torch
import numpy as np

from src.engine.pipeline import OCREngine

def benchmark():
    print("=" * 60)
    print("PHASE 8: PERFORMANCE BENCHMARK (CPU)")
    print("=" * 60)

    engine = OCREngine(model_path="models/best_model.pt", device="cpu")

    images = [
        ("Invoice Image", "sample/invoice_test.png"),
        ("Document Sample", "sample/document_sample.png"),
    ]

    for label, path in images:
        img = cv2.imread(path)
        assert img is not None

        # Warmup
        _ = engine.process(img, redact=False)

        # 1. Detection alone
        t0 = time.perf_counter()
        prep = engine.preprocessor.process(img)
        boxes = engine.detector.detect(prep)
        t_detect = (time.perf_counter() - t0) * 1000.0

        # 2. OCR Recognition alone
        crops = []
        for box in boxes:
            t, l, b, r = box
            crop = img[t:b, l:r]
            if crop.size > 0:
                crops.append(crop)

        t0 = time.perf_counter()
        words = engine.recognizer.recognize_batch(crops)
        t_ocr = (time.perf_counter() - t0) * 1000.0

        # 3. Full pipeline (including inpainting redaction)
        t0 = time.perf_counter()
        res = engine.process(img, redact=True)
        t_total = (time.perf_counter() - t0) * 1000.0

        print(f"\n--- {label} ({path}) ---")
        print(f"  Image dimensions  : {img.shape[1]}x{img.shape[0]}")
        print(f"  Detected regions  : {len(boxes)}")
        print(f"  Detection Latency : {t_detect:.2f} ms")
        print(f"  OCR Batch Latency : {t_ocr:.2f} ms")
        print(f"  Total End-to-End  : {t_total:.2f} ms")

if __name__ == "__main__":
    benchmark()
