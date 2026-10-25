"""Inspect all boxes on invoice_test.png."""
import cv2
import json

img = cv2.imread("sample/invoice_test.png")
from src.engine.detector import TextDetector
detector = TextDetector()
boxes = detector.detect(img)

print(f"Total boxes: {len(boxes)}")
for i, (t, l, b, r) in enumerate(boxes):
    crop = img[t:b, l:r]
    # check if crop is basically a horizontal line or vertical line
    h, w = crop.shape[:2]
    # Save crops of interest
    cv2.imwrite(f"scratch/crop_{i:02d}.png", crop)
    print(f"Box {i:2d}: [t={t:3d}, l={l:3d}, b={b:3d}, r={r:3d}] ({w:3d}x{h:2d})")
