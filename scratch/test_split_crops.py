"""Test predictions on split words of Laptop Dell Inspiron."""
import cv2
from src.engine.recognizer import TextRecognizer

recognizer = TextRecognizer(model_path="models/best_model.pt", device="cpu")
img = cv2.imread("sample/invoice_test.png")

laptop_crop = img[353:372, 174:228]
dell_crop = img[353:372, 233:264]
inspiron_crop = img[353:372, 268:333]

for name, crop in [("Laptop", laptop_crop), ("Dell", dell_crop), ("Inspiron", inspiron_crop)]:
    pred = recognizer.recognize_crop(crop)
    print(f"Crop {name:10s} (shape {crop.shape[:2]}): Pred = '{pred}'")
