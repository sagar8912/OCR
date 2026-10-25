"""Test clean individual word crops from the invoice."""
import cv2
import torch
from src.engine.recognizer import TextRecognizer

recognizer = TextRecognizer(model_path="models/best_model.pt", device="cpu")
img = cv2.imread("sample/invoice_test.png")

# Let's manually crop true words from the invoice and see what the model predicts:
# 1. "INVOICE"
# The header INVOICE is roughly at y: 22..62, x: 710..925
invoice_crop = img[22:62, 715:925]
cv2.imwrite("scratch/true_crop_invoice.png", invoice_crop)

# 2. "DESCRIPTION"
# Header at y: 318..336, x: 255..365
desc_crop = img[315:338, 250:365]
cv2.imwrite("scratch/true_crop_desc.png", desc_crop)

# 3. "QUANTITY"
# y: 318..336, x: 485..575
qty_crop = img[315:338, 485:575]
cv2.imwrite("scratch/true_crop_qty.png", qty_crop)

# 4. "CONFIDENTIAL" (red text inside box)
# y: 320..348, x: 695..940
conf_crop = img[320:348, 695:940] # wait, confidential is at y: 320 or 220?
# Let's check where CONFIDENTIAL is:

# Let's find CONFIDENTIAL:
# In previous boxes: Box 21: [t=222, l=718, b=248, r=961]
conf_crop = img[220:250, 695:945]
cv2.imwrite("scratch/true_crop_conf.png", conf_crop)

# 5. "TOTAL"
# y: 556..577, x: 610..680
total_crop = img[555:578, 610:680]
cv2.imwrite("scratch/true_crop_total.png", total_crop)

# 6. "AMOUNT"
# y: 556..577, x: 685..775
amount_crop = img[555:578, 685:780]
cv2.imwrite("scratch/true_crop_amount.png", amount_crop)

# 7. "SUBTOTAL"
subtotal_crop = img[488:508, 610:705]
cv2.imwrite("scratch/true_crop_subtotal.png", subtotal_crop)

crops = {
    "INVOICE": invoice_crop,
    "DESCRIPTION": desc_crop,
    "QUANTITY": qty_crop,
    "CONFIDENTIAL": conf_crop,
    "TOTAL": total_crop,
    "AMOUNT": amount_crop,
    "SUBTOTAL": subtotal_crop,
}

print("Testing True Clean Word Crops from Invoice:")
for name, crop in crops.items():
    pred = recognizer.recognize_crop(crop)
    print(f"Word: {name:15s} | Crop shape: {crop.shape[:2]} | Pred: '{pred}'")
