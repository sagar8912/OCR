import json
import cv2
from pathlib import Path
from src.engine.pipeline import OCREngine

def main():
    engine = OCREngine("models/best_model.pt")
    res = engine.process("sample/invoice_test.png", redact=False)

    print(f"Total results: {len(res['results'])}")
    for i, item in enumerate(res["results"]):
        w = item["word"]
        box = item["bounding_box"]
        ft = item.get("field_type", "UNKNOWN")
        conf = item.get("confidence", 0.0)
        # Print items that match or partially match target keywords
        print(f"[{i:03d}] {w:20s} | Box: {box} | Field: {ft:15s} | Conf: {conf:.2f}")

if __name__ == "__main__":
    main()
