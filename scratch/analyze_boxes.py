"""Analyze all detected regions on invoice_test.png."""
import json
import cv2

img = cv2.imread("sample/invoice_test.png")
with open("sample/invoice_test_text.json") as f:
    detections = json.load(f)

print(f"Total detections: {len(detections)}")
for i, d in enumerate(detections):
    t, l, b, r = d["bounding_box"]
    w = r - l
    h = b - t
    crop = img[t:b, l:r]
    # mean color
    mean_val = crop.mean() if crop.size > 0 else 0
    print(f"Box {i:2d}: [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d}, mean={mean_val:.1f}) -> Word: '{d['word']}'")
