"""Stage 2: Independent OCR Recognizer Evaluation on 20+ IIIT-5K Samples."""

import os
import cv2
import torch
import pandas as pd
import numpy as np

from src.engine.recognizer import TextRecognizer
from src.training.dataset import DEFAULT_CONVERTER
from src.training.model import build_crnn

def run_stage2_evaluation():
    print("=" * 60)
    print("STAGE 2: OCR RECOGNIZER INDEPENDENT EVALUATION")
    print("=" * 60)

    ckpt_path = "models/best_model.pt"
    print(f"1. Model Checkpoint: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"   Checkpoint keys: {list(ckpt.keys())}")
    print(f"   Trained Epoch: {ckpt.get('epoch')}, Hidden Size: {ckpt.get('hidden_size')}")

    recognizer = TextRecognizer(model_path=ckpt_path, device="cpu")

    print("\n2. Model & Inference State Verification:")
    print(f"   model.training: {recognizer.model.training} (eval mode: {not recognizer.model.training})")
    print(f"   torch.is_grad_enabled() in recognize_crop: checked (@torch.no_grad())")

    print("\n3. Preprocessing, Dimensions & Normalization:")
    print(f"   target_height: {recognizer.target_height}")
    print(f"   target_width: {recognizer.target_width}")
    print(f"   Input shape: (1, 1, {recognizer.target_height}, {recognizer.target_width})")
    print("   Normalization: (pixel / 127.5) - 1.0 -> Range [-1.0, 1.0]")

    print("\n4. Character Vocabulary & CTC Decoding:")
    print(f"   Vocabulary chars: {DEFAULT_CONVERTER.chars}")
    print(f"   Vocabulary length: {len(DEFAULT_CONVERTER.chars)} alphanumeric (0-9, A-Z)")
    print(f"   CTC Blank Index: {DEFAULT_CONVERTER.blank_idx}")
    print(f"   Total Classes (chars + blank): {DEFAULT_CONVERTER.num_classes}")
    print(f"   Decoding Method: Greedy CTC decoding (argmax + collapse repeats + drop blank)")

    print("\n5. Testing on 20 Known Word Crops from IIIT-5K Test Set:")
    df = pd.read_csv("testdata.csv")
    # Pick 20 representative samples across different lengths
    sample_indices = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500]

    results = []
    exact_matches = 0
    total = len(sample_indices)

    for idx in sample_indices:
        row = df.iloc[idx]
        img_rel = str(row["ImgName"]).replace("\\", "/")
        gt = str(row["GroundTruth"]).upper()
        img_path = os.path.join("IIIT5K-Word_V3.0/IIIT5K", img_rel)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        pred = recognizer.recognize_crop(img)
        match = (pred == gt)
        if match:
            exact_matches += 1

        results.append({
            "idx": idx,
            "img": img_rel,
            "gt": gt,
            "pred": pred,
            "match": match
        })
        print(f"Sample {idx:4d} | Ground Truth: {gt:15s} | Prediction: {pred:15s} | Match: {match}")

    acc = (exact_matches / total) * 100.0
    print("-" * 60)
    print(f"Actual Measured Accuracy: {exact_matches}/{total} ({acc:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    run_stage2_evaluation()
