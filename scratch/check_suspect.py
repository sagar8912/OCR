"""Inspect suspect line/fragment boxes."""
import json
import cv2

with open("sample/invoice_test_text.json") as f:
    detections = json.load(f)

for i, d in enumerate(detections):
    t, l, b, r = d["bounding_box"]
    w = r - l
    h = b - t
    aspect = w / max(1, h)
    if h <= 15 or w <= 12 or aspect > 10:
        print(f"Suspect line/fragment box {i:2d}: [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d}, aspect={aspect:.2f}) -> '{d['word']}'")
