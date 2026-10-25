import cv2
import torch
from src.engine.pipeline import OCREngine
from src.engine.decoder import LexiconDecoder, analyze_crop_topology

def main():
    engine = OCREngine("models/best_model.pt")
    img = cv2.imread("sample/invoice_test.png")

    # Currency crops coordinates
    currency_boxes = [
        ("45,000", [355, 699, 371, 754]),
        ("90,000", [355, 919, 371, 974]),
        ("1,200",  [389, 711, 406, 754]),
        ("6,000",  [389, 929, 406, 974]),
        ("1,500",  [423, 711, 440, 754]),
        ("7,500",  [423, 930, 440, 974]),
        ("12,000", [457, 701, 474, 754]),
        ("24,000", [457, 919, 474, 974]),
        ("127,500",[491, 912, 508, 974]),
        ("22,950", [525, 920, 542, 974]),
        ("150,450",[561, 907, 578, 974]),
    ]

    for expected, b in currency_boxes:
        crop = img[b[0]:b[2], b[1]:b[3]]
        topo = analyze_crop_topology(crop)
        res = engine.recognizer.recognize_crop(crop, return_details=True)
        print(f"Target: {expected:8s} | Box: {b} | Has Comma: {topo['has_comma']} | Commas: {topo['commas']} | OCR Pred: '{res['word']}' (raw: '{res.get('raw_word')}') | Field: {res.get('field_type')} | Conf: {res.get('confidence')}")

if __name__ == "__main__":
    main()
