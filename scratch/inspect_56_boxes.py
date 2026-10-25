"""Inspect the 56 detected boxes on invoice_test.png."""
import cv2
from scratch.refined_detector_class import TextDetector

detector = TextDetector()
img = cv2.imread("sample/invoice_test.png")
boxes = detector.detect(img)

print(f"Total boxes: {len(boxes)}")
for i, (t, l, b, r) in enumerate(boxes):
    w = r - l
    h = b - t
    print(f"Box {i:2d}: [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d})")
