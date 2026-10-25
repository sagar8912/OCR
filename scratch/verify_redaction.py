"""Verify redaction quality on invoice_test_redacted.png."""
import cv2
import numpy as np
import json

orig = cv2.imread("sample/invoice_test.png")
redacted = cv2.imread("sample/invoice_test_redacted.png")

with open("sample/invoice_test_text.json") as f:
    detections = json.load(f)

print(f"Original image shape: {orig.shape}")
print(f"Redacted image shape: {redacted.shape}")

# Calculate pixel changes
diff = cv2.absdiff(orig, redacted)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
changed_pixels = np.sum(diff_gray > 10)
total_pixels = orig.shape[0] * orig.shape[1]

print(f"Changed pixels in redacted image: {changed_pixels} ({changed_pixels / total_pixels * 100:.2f}%)")

# Check inside boxes vs outside boxes
inside_box_mask = np.zeros(orig.shape[:2], dtype=np.uint8)
for d in detections:
    t, l, b, r = d["bounding_box"]
    inside_box_mask[t:b, l:r] = 255

inside_changes = np.sum((diff_gray > 10) & (inside_box_mask > 0))
outside_changes = np.sum((diff_gray > 10) & (inside_box_mask == 0))

print(f"Changes strictly inside text regions: {inside_changes} ({inside_changes / np.sum(inside_box_mask > 0) * 100:.2f}% of text area modified)")
print(f"Unintended changes outside text regions: {outside_changes} ({outside_changes / np.sum(inside_box_mask == 0) * 100:.3f}% of non-text area)")

# Check standard deviation inside major boxes in redacted image to ensure text edges were removed
for name, t, l, b, r in [
    ("INVOICE", 23, 741, 58, 918),
    ("CONFIDENTIAL", 222, 718, 248, 961),
    ("Description", 318, 263, 336, 360),
]:
    orig_crop = orig[t:b, l:r]
    red_crop = redacted[t:b, l:r]
    orig_std = orig_crop.std()
    red_std = red_crop.std()
    print(f"Region '{name}' [t={t}, l={l}, b={b}, r={r}]:")
    print(f"   Original texture std: {orig_std:.2f} (high contrast text strokes)")
    print(f"   Redacted texture std: {red_std:.2f} (smooth inpainting fill)")
