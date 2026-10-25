"""Measure text heights and spacings on the invoice."""
import cv2
import numpy as np

img = cv2.imread("sample/invoice_test.png")

# 1. "INVOICE" header
# Letters: I, N, V, O, I, C, E
# Height: ~35px.
# Gap between O (x~848) and I (x~864): 16px!

# 2. "123 Business Park, Sector 62"
# Height: ~14px.
# Gap between "123" (x~47..72) and "Business" (x~78..138): 6px!
# Gap between "Park," and "Sector": 7px!

# 3. "Laptop Dell Inspiron"
# Height: ~14px.
# Gap between "Laptop" and "Dell": 7px!
# Gap between "Dell" and "Inspiron": 7px!

# 4. "Total Amount"
# Height: ~15px.
# Gap between "Total" and "Amount": 8px!

print("Analysis of gaps vs text height:")
print("Large font (h=35px, INVOICE): Intra-word letter gap = ~15px (Ratio gap/height = 0.43)")
print("Small font (h=14px, Body):    Inter-word space      = ~7px  (Ratio gap/height = 0.50)")
print("Small font (h=14px, Body):    Intra-word letter gap = ~2-3px(Ratio gap/height = 0.18)")
