import cv2
import numpy as np
import torch
from src.engine.pipeline import OCREngine
from src.engine.decoder import analyze_crop_topology

engine = OCREngine("models/best_model.pt")
img = cv2.imread("sample/invoice_test.png")

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

def decode_subpatch(patch):
    if patch.shape[1] < 3:
        return ""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch.copy()
    pw = patch.shape[1]
    # Optimal width for patch based on character aspect ratio: ~24px per character
    target_w = max(48, min(128, int(pw * 2.2)))
    resized = cv2.resize(gray, (target_w, 32), interpolation=cv2.INTER_AREA)
    norm = (resized.astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(engine.recognizer.device)
    with torch.no_grad():
        lp = engine.recognizer.model(tensor)
        s, conf = engine.recognizer.decoder._decode_numeric_tensor(lp)
        return s

for expected, b in currency_boxes:
    crop = img[b[0]:b[2], b[1]:b[3]]
    # Otsu topology
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_otsu)
    commas = []
    h, w = gray.shape
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area < 4:
            continue
        if (y >= 0.45 * h) and (ch <= 0.45 * h) and (cw <= 7) and (area <= 35):
            commas.append(int(x + cw // 2))
    commas.sort()

    if commas:
        parts = []
        last_x = 0
        for cx in commas:
            p = crop[:, last_x : max(last_x, cx - 1)]
            parts.append(decode_subpatch(p))
            last_x = cx + 3
        p_last = crop[:, min(w, last_x) :]
        parts.append(decode_subpatch(p_last))
        pred = ",".join([p for p in parts if p])
    else:
        pred = decode_subpatch(crop)
    
    print(f"Target: {expected:8s} | Commas: {commas} | Pred: {pred:8s} | Match: {expected == pred}")
