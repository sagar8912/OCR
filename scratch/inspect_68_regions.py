"""Inspect all 68 regions in invoice."""

import cv2
from scratch.test_all_phases import RobustTextDetector

detector = RobustTextDetector()
img = cv2.imread("sample/invoice_test.png")
boxes = detector.detect(img)

print(f"Total detected regions: {len(boxes)}")
for i, (t, l, b, r) in enumerate(boxes):
    w = r - l
    h = b - t
    aspect = w / max(1, h)
    print(f"Region {i:2d}: [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d}, aspect={aspect:.2f})")
