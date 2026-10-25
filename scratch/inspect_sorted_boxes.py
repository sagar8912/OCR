"""Inspect boxes from ratio=0.45, kw=12."""
import cv2
from scratch.test_perfect_grouping import run_grouping_test

boxes = run_grouping_test(min_space_ratio=0.45, kw_val=12)
img = cv2.imread("sample/invoice_test.png")

# Sort in reading order
boxes_with_centers = []
for b in boxes:
    y_center = (b[0] + b[2]) / 2.0
    height = b[2] - b[0]
    boxes_with_centers.append((y_center, height, b))

lines = []
for yc, bh, b in sorted(boxes_with_centers, key=lambda x: x[0]):
    placed = False
    for line in lines:
        if abs(yc - line["yc"]) < max(bh, line["h"]) * 0.5:
            line["boxes"].append(b)
            placed = True
            break
    if not placed:
        lines.append({"yc": yc, "h": bh, "boxes": [b]})

sorted_boxes = []
for line in sorted(lines, key=lambda l: l["yc"]):
    line_boxes = sorted(line["boxes"], key=lambda b: b[1])
    sorted_boxes.extend(line_boxes)

print(f"\nFinal sorted boxes count: {len(sorted_boxes)}")
from src.engine.recognizer import TextRecognizer
recognizer = TextRecognizer(model_path="models/best_model.pt", device="cpu")

for idx, (t, l, b, r) in enumerate(sorted_boxes):
    crop = img[t:b, l:r]
    w = r - l
    h = b - t
    pred = recognizer.recognize_crop(crop)
    print(f"Box {idx:2d}: [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d}) -> '{pred}'")
