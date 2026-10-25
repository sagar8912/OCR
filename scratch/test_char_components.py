import cv2
import numpy as np
import torch
from src.engine.pipeline import OCREngine

engine = OCREngine("models/best_model.pt")
img = cv2.imread("sample/invoice_test.png")
crop = img[561:578, 907:974]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
_, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_otsu)

comps = []
for i in range(1, num_labels):
    x, y, w, h, a = stats[i]
    if a >= 5:
        comps.append((x, y, w, h, a))

comps.sort(key=lambda c: c[0])
print(f"Total components found in 150,450: {len(comps)}")

for i, (x, y, w, h, a) in enumerate(comps):
    # Check if comma: baseline and small
    if y >= 8 and h <= 8 and w <= 5:
        print(f"[{i}] Comma detected at x={x}!")
        continue
    # Digit patch
    char_patch = crop[:, max(0, x-1) : min(crop.shape[1], x+w+1)]
    # pad to square-ish then resize to (32, 32) or (64, 32)
    ch_h, ch_w = char_patch.shape[:2]
    pad_h = np.zeros((32, 64, 3), dtype=np.uint8) + 255
    res_ch = cv2.resize(char_patch, (int(ch_w * (24.0 / max(1, ch_h))), 24))
    r_h, r_w = res_ch.shape[:2]
    start_x = (64 - r_w) // 2
    start_y = (32 - r_h) // 2
    pad_h[start_y:start_y+r_h, start_x:start_x+r_w] = res_ch
    
    with torch.no_grad():
        tensor = engine.preprocessor.prepare_for_ocr(pad_h, target_width=128).to(engine.recognizer.device)
        lp = engine.recognizer.model(tensor)
        raw = engine.recognizer.converter.decode_greedy(torch.argmax(lp[:, 0], dim=-1))
    print(f"[{i}] Comp at x={x} (w={w}, h={h}): raw='{raw}'")
