"""Check redaction performance and visual quality."""
import cv2
import numpy as np
from src.engine.detector import TextDetector
from src.engine.redactor import TextRedactor

img = cv2.imread("sample/invoice_test.png")
detector = TextDetector()
redactor = TextRedactor(method="telea", inpaint_radius=4)

boxes = detector.detect(img)
redacted = redactor.redact(img, boxes)

cv2.imwrite("sample/diagnostic_phase4_redacted.png", redacted)
print(f"Redaction completed on {len(boxes)} boxes. Saved to sample/diagnostic_phase4_redacted.png")
