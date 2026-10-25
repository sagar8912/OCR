"""Diagnostic script for Phase 2: Recognizer investigation."""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from src.engine.recognizer import TextRecognizer
from src.training.dataset import DEFAULT_CONVERTER
from src.training.model import build_crnn

def test_recognizer_diagnostic():
    print("=" * 60)
    print("PHASE 2: RECOGNIZER DIAGNOSTIC")
    print("=" * 60)

    # 1. Inspect loaded checkpoint
    ckpt_path = "models/best_model.pt"
    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {list(state.keys())}")
    if "epoch" in state:
        print(f"Trained epoch: {state.get('epoch')}, val_loss: {state.get('val_loss')}")
    
    # 2. Check character vocabulary and CTC blank
    print(f"\nCharacter vocabulary size: {len(DEFAULT_CONVERTER.chars)}")
    print(f"Characters: {DEFAULT_CONVERTER.chars}")
    print(f"Blank index: {DEFAULT_CONVERTER.blank_idx}")
    print(f"Num classes: {DEFAULT_CONVERTER.num_classes}")

    # 3. Initialize recognizer
    recognizer = TextRecognizer(model_path=ckpt_path, device="cpu")
    print(f"Model is in eval mode: {not recognizer.model.training}")

    # 4. Test on 20 known crops from IIIT-5K test set
    print("\n--- Testing on 20 known IIIT-5K Test Images ---")
    df = pd.read_csv("testdata.csv")
    indices = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500]

    correct = 0
    total = len(indices)

    for idx in indices:
        row = df.iloc[idx]
        img_rel_path = row["ImgName"]
        gt_text = str(row["GroundTruth"]).upper()

        full_img_path = os.path.join("IIIT5K-Word_V3.0/IIIT5K", img_rel_path)
        img = cv2.imread(full_img_path)
        
        # Test recognizer
        pred_text = recognizer.recognize_crop(img)
        is_match = (pred_text == gt_text)
        if is_match:
            correct += 1
        
        print(f"[{idx:4d}] GT: {gt_text:15s} | Pred: {pred_text:15s} | Match: {is_match}")

    print(f"\nIIIT-5K 20-Sample Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # 5. Now test on the Invoice crops
    print("\n--- Testing on Invoice Crops ---")
    invoice_img = cv2.imread("sample/invoice_test.png")
    from src.engine.detector import TextDetector
    detector = TextDetector()
    boxes = detector.detect(invoice_img)

    for i, box in enumerate(boxes):
        top, left, bottom, right = box
        crop = invoice_img[top:bottom, left:right]
        pred = recognizer.recognize_crop(crop)
        w = right - left
        h = bottom - top
        print(f"Box {i:2d} ({w:3d}x{h:2d}) [t={top:3d}, l={left:3d}, b={bottom:3d}, r={right:3d}]: Pred = '{pred}'")

if __name__ == "__main__":
    test_recognizer_diagnostic()
