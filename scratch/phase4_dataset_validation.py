"""Phase 4: Dataset Validation Script (50 Random Samples)."""

import os
import random
import pandas as pd
from src.training.dataset import DEFAULT_CONVERTER

def validate_dataset(csv_path="traindata.csv", n_samples=50):
    print("=" * 60)
    print(f"PHASE 4: DATASET VALIDATION on {csv_path} ({n_samples} samples)")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Total rows in {csv_path}: {total_rows}")

    # Randomly select 50 indices
    random.seed(42)
    sample_indices = random.sample(range(total_rows), n_samples)

    mismatches = 0
    missing_images = 0

    for idx in sample_indices:
        row = df.iloc[idx]
        img_name = str(row["ImgName"]).replace("\\", "/")
        gt_raw = str(row["GroundTruth"])
        gt_clean = "".join([c for c in gt_raw.upper() if c in DEFAULT_CONVERTER.char2idx])

        # Verify image file exists
        full_path = os.path.join("IIIT5K-Word_V3.0/IIIT5K", img_name)
        if not os.path.exists(full_path):
            print(f"[FAIL] Missing image: {full_path}")
            missing_images += 1
            continue

        # Encode
        encoded = DEFAULT_CONVERTER.encode(gt_clean)
        # Decode
        decoded = DEFAULT_CONVERTER.decode_greedy(encoded)

        if decoded != gt_clean:
            print(f"[MISMATCH] Row {idx}: GT='{gt_clean}' != Decoded='{decoded}' (encoded={encoded})")
            mismatches += 1
        else:
            # Print sample for report
            if len(encoded) > 0 and mismatches == 0 and sample_indices.index(idx) < 10:
                print(f"Sample {idx:4d} | File: {img_name:25s} | GT: {gt_clean:12s} | Encoded: {str(encoded):25s} | Decoded: {decoded:12s} [OK]")

    print("-" * 60)
    print(f"Dataset Validation Result: {n_samples - mismatches - missing_images}/{n_samples} verified 100% correct.")
    print(f"Mismatches: {mismatches}, Missing images: {missing_images}")
    print("=" * 60)
    assert mismatches == 0 and missing_images == 0, "Dataset validation failed!"

if __name__ == "__main__":
    validate_dataset("traindata.csv", 50)
    validate_dataset("testdata.csv", 50)
