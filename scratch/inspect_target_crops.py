import cv2
import numpy as np
import torch
from src.engine.pipeline import OCREngine

def main():
    engine = OCREngine("models/best_model.pt")
    img = cv2.imread("sample/invoice_test.png")

    boxes = engine.detector.detect(img)
    print(f"Total detected boxes: {len(boxes)}")

    targets = [
        "TECHNOLOGIES", "GURUGRAM", "HARYANA", "122001",
        "INV", "CUST", "1,500", "12,000", "24,000", "22,950", "9876543210", "HDFC"
    ]

    for b in boxes:
        top, left, bot, right = b
        crop = img[max(0, top):bot, max(0, left):right]
        if crop.size == 0:
            continue

        det_result = engine.recognizer.recognize_crop(crop, return_details=True)
        final_word = det_result["word"]
        raw_pred = det_result.get("raw_word", "")
        field_type = det_result.get("field_type")
        conf = det_result.get("confidence", 0.0)

        # Check if any target is in or close to raw_pred or final_word
        print(f"Box: [{top:3d}, {left:3d}, {bot:3d}, {right:3d}] (w={right-left:3d}, h={bot-top:2d}) | Raw: '{raw_pred:16s}' | Final: '{final_word:16s}' | Field: {field_type:15s} | Conf: {conf:.2f}")

if __name__ == "__main__":
    main()
