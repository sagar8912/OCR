import cv2
import torch
from src.engine.recognizer import TextRecognizer
from src.engine.detector import TextDetector
from src.engine.decoder import LexiconDecoder

def main():
    rec = TextRecognizer(model_path="models/best_model.pt")
    dec = LexiconDecoder()
    det = TextDetector()

    inv = cv2.imread("sample/invoice_test.png")
    gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    boxes = det.detect(gray)
    crops = [inv[b[0]:b[2], b[1]:b[3]] for b in boxes]

    print(f"Detected {len(crops)} crops. Running recognition + lexicon decoding...")
    corrected_list = []
    
    for i in range(0, len(crops), 16):
        chunk = crops[i : i + 16]
        tensors = [rec.preprocessor.prepare_for_ocr(c) for c in chunk]
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            lp = rec.model(batch)
            preds = rec.model.predict(batch)
            for b_idx in range(len(chunk)):
                raw = preds[b_idx]
                res = dec.decode(lp[:, b_idx : b_idx + 1], raw_text=raw)
                corrected_list.append((raw, res["word"], res["confidence"], res["method"]))
                if raw != res["word"]:
                    w_corr = res["word"]
                    conf = res["confidence"]
                    m = res["method"]
                    print(f"[{i + b_idx:2d}] Raw: {raw:<14} -> Corrected: {w_corr:<14} (conf: {conf:.2f}, {m})")

    # Check key words
    print("\n--- KEY INVOICE WORDS CHECK ---")
    final_words = [item[1] for item in corrected_list]
    keywords = [
        "INVOICE", "CONFIDENTIAL", "DESCRIPTION", "QUANTITY", "UNIT", "PRICE",
        "AMOUNT", "LAPTOP", "DELL", "INSPIRON", "WIRELESS", "MOUSE", "KEYBOARD",
        "MONITOR", "INCH", "SUBTOTAL", "TAX", "TOTAL", "BANK", "NAME", "HDFC"
    ]
    for kw in keywords:
        found = kw in final_words
        near = [w for w in final_words if kw in w or w in kw]
        status = "FOUND" if found else (f"PARTIAL: {near}" if near else "MISSING")
        print(f"  {kw:<15}: {status}")

if __name__ == "__main__":
    main()
