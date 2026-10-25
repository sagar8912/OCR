"""Inspect the 121 boxes from refined detector."""
import cv2
from scratch.test_refined_detector import run_refined_detector

boxes = run_refined_detector("sample/invoice_test.png", debug=False)
img = cv2.imread("sample/invoice_test.png")

print(f"\nInspecting sample boxes:")
for i, (t, l, b, r) in enumerate(boxes):
    w = r - l
    h = b - t
    crop = img[t:b, l:r]
    # let's print if width < 15 or aspect < 1.0
    if i < 30 or (w < 15 and h > 10):
        print(f"Box {i:3d}: [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d})")
