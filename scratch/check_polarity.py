"""Check polarity (light-on-dark vs dark-on-light) of IIIT-5K vs invoice crops."""
import os
import cv2
import numpy as np
import pandas as pd

df = pd.read_csv("testdata.csv")

print("IIIT-5K Samples background/foreground polarity:")
for i in [0, 50, 100, 150, 200]:
    row = df.iloc[i]
    p = os.path.join("IIIT5K-Word_V3.0/IIIT5K", row["ImgName"])
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # Check border pixels vs center pixels
    border_mean = (np.mean(img[0, :]) + np.mean(img[-1, :]) + np.mean(img[:, 0]) + np.mean(img[:, -1])) / 4.0
    center_mean = np.mean(img[img.shape[0]//4 : 3*img.shape[0]//4, img.shape[1]//4 : 3*img.shape[1]//4])
    print(f"Sample {i:3d} ({row['GroundTruth']}): border_mean={border_mean:.1f}, center_mean={center_mean:.1f}, overall={np.mean(img):.1f}")

print("\nInvoice crops polarity:")
invoice_img = cv2.imread("sample/invoice_test.png", cv2.IMREAD_GRAYSCALE)
for name, y1, y2, x1, x2 in [
    ("INVOICE", 22, 62, 715, 925),
    ("DESCRIPTION", 315, 338, 250, 365),
    ("QUANTITY", 315, 338, 485, 575),
    ("TOTAL", 555, 578, 610, 680),
]:
    crop = invoice_img[y1:y2, x1:x2]
    border_mean = (np.mean(crop[0, :]) + np.mean(crop[-1, :]) + np.mean(crop[:, 0]) + np.mean(crop[:, -1])) / 4.0
    center_mean = np.mean(crop[crop.shape[0]//4 : 3*crop.shape[0]//4, crop.shape[1]//4 : 3*crop.shape[1]//4])
    print(f"Invoice {name}: border_mean={border_mean:.1f}, center_mean={center_mean:.1f}, overall={np.mean(crop):.1f}")
