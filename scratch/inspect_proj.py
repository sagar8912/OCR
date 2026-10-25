"""Inspect col_proj for Laptop Dell Inspiron."""
import cv2
import numpy as np

img = cv2.imread("sample/invoice_test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.magnitude(grad_x, grad_y)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

sub = binary[353:372, 174:333]
col_proj = np.sum(sub > 0, axis=0)
print("Col projection values:")
print(list(col_proj))
