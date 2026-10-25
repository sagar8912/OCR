import json
import time
from pathlib import Path
import cv2
import numpy as np

from src.engine.pipeline import OCREngine

def main():
    engine = OCREngine(model_path="models/best_model.pt", device="cpu")
    img_path = Path("sample/invoice_test.png")
    if not img_path.exists():
        print(f"Error: {img_path} not found")
        return

    t0 = time.perf_counter()
    result = engine.process(str(img_path), redact=True)
    total_time = (time.perf_counter() - t0) * 1000.0

    outputs = engine.save_outputs(result, img_path)
    print(f"Saved JSON: {outputs['json_path']}")
    print(f"Saved Redacted Image: {outputs['redacted_image_path']}")

    words_info = result["results"]
    print(f"\nTotal regions detected: {len(words_info)}")
    print(f"Total Pipeline Latency: {total_time:.2f} ms")

    # Target keywords to check
    keywords = [
        "INVOICE", "CONFIDENTIAL", "DESCRIPTION", "QUANTITY",
        "TOTAL", "AMOUNT", "SUBTOTAL", "TAX", "ACCOUNT", "BANK",
        "DATE", "CUSTOMER", "LAPTOP", "MOUSE", "KEYBOARD", "MONITOR"
    ]
    
    print("\n--- DETECTED REGIONS AND OCR PREDICTIONS ---")
    detected_words = []
    for idx, item in enumerate(words_info):
        word = item["word"]
        box = item["bounding_box"]
        detected_words.append(word)
        print(f"[{idx+1:2d}] Box: {box} | OCR: '{word}'")

    print("\n--- KEYWORD RECOGNITION CHECK ---")
    for kw in keywords:
        found = any(kw in w for w in detected_words)
        near = [w for w in detected_words if kw[:4] in w or w in kw and len(w) >= 3]
        status = "FOUND" if found else (f"PARTIAL ({near})" if near else "NOT FOUND")
        print(f"  {kw:<15}: {status}")

if __name__ == "__main__":
    main()
