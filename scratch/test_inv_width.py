import cv2
import torch
from src.engine.recognizer import TextRecognizer

rec = TextRecognizer("models/best_model.pt")
crop = cv2.imread("scratch/crop_inv_full.png")
print("Crop shape:", crop.shape)

with torch.no_grad():
    for target_w in [128, 160, 192, 224, 256, 288, 320]:
        tensor = rec.preprocessor.prepare_for_ocr(crop, target_height=32, target_width=target_w)
        logits = rec.model(tensor)
        raw = rec.converter.decode_greedy(logits)[0]
        print(f"Width {target_w}: raw='{raw}'")
