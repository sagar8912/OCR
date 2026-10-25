"""Comprehensive Diagnostic Script for OCR Model, Preprocessing, Dimensions, CTC Loss, and Overfitting."""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.dataset import IIIT5KDataset, DEFAULT_CONVERTER, ctc_collate_fn
from src.training.model import build_crnn, CRNN
from src.training.evaluate import calculate_metrics


def run_diagnostics():
    print("=" * 60)
    print("INVESTIGATION STEP 2 & 3: SAMPLES & LABEL VERIFICATION")
    print("=" * 60)
    train_df = pd.read_csv("traindata.csv")
    ds = IIIT5KDataset(train_df, img_root="IIIT5K-Word_V3.0/IIIT5K", is_train=False)

    np.random.seed(42)
    sample_indices = np.random.choice(len(ds), 20, replace=False)
    for idx in sample_indices:
        item = ds[int(idx)]
        gt = item["raw_text"]
        enc = item["target"].tolist()
        # Decode without CTC collapse just to check 1-to-1 index mapping
        char_map_dec = "".join([DEFAULT_CONVERTER.idx2char.get(i, "?") for i in enc])
        # CTC greedy decode
        ctc_dec = DEFAULT_CONVERTER.decode_greedy(enc)
        print(f"Idx {idx:4d} | File: {item['img_name']:<18} | GT: {gt:<12} | Enc: {enc[:6]}... | Dec: {ctc_dec:<12} | Direct: {char_map_dec}")

    print("\n" + "=" * 60)
    print("INVESTIGATION STEP 4: VOCABULARY & BLANK TOKEN")
    print("=" * 60)
    print(f"Converter chars ({len(DEFAULT_CONVERTER.chars)}): {DEFAULT_CONVERTER.chars}")
    print(f"Blank idx: {DEFAULT_CONVERTER.blank_idx}")
    print(f"Num classes: {DEFAULT_CONVERTER.num_classes}")
    assert DEFAULT_CONVERTER.blank_idx == 0
    assert DEFAULT_CONVERTER.num_classes == 37

    print("\n" + "=" * 60)
    print("INVESTIGATION STEP 5: TENSOR DIMENSIONS THROUGH CRNN")
    print("=" * 60)
    model = build_crnn(hidden_size=128)
    model.eval()

    dummy_input = torch.randn(2, 1, 32, 128)
    print(f"1. Input shape: {dummy_input.shape}")

    # Inspect through intermediate layers
    feat = model.cnn(dummy_input)
    print(f"2. CNN output shape (B, C, H', W'): {feat.shape}")

    pooled = model.adaptive_pool(feat)
    print(f"3. Adaptive pool shape (B, C, 1, W_seq): {pooled.shape}")

    squeezed = pooled.squeeze(2)
    print(f"4. Squeezed shape (B, C, W_seq): {squeezed.shape}")

    permuted = squeezed.permute(2, 0, 1)
    print(f"5. Sequence into BiLSTM (T, B, C): {permuted.shape} (T = {permuted.shape[0]})")

    rnn_out, _ = model.rnn(permuted)
    print(f"6. BiLSTM output (T, B, 2*hidden): {rnn_out.shape}")

    logits = model.classifier(rnn_out)
    print(f"7. Classifier logits (T, B, num_classes): {logits.shape}")

    log_probs = model(dummy_input)
    print(f"8. Full model forward log_probs (T, B, num_classes): {log_probs.shape}")

    print("\n" + "=" * 60)
    print("INVESTIGATION STEP 6: CTC LOSS ON SYNTHETIC FORWARD")
    print("=" * 60)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    batch = ctc_collate_fn([ds[i] for i in range(4)])
    imgs = batch["images"]
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]

    log_probs = model(imgs)
    T, B, C = log_probs.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)

    print(f"T={T}, B={B}, C={C}")
    print(f"input_lengths: {input_lengths.tolist()}")
    print(f"target_lengths: {target_lengths.tolist()}")
    print(f"targets (len={len(targets)}): {targets.tolist()}")

    # Check validity condition: for every item, target_length <= input_length
    for i, tlen in enumerate(target_lengths.tolist()):
        print(f"  Item {i}: input_length {T} >= target_length {tlen}: {T >= tlen}")
        assert T >= tlen, f"CTC violation! Target length {tlen} > input length {T}"

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    print(f"Calculated CTC Loss on untrained model: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("INVESTIGATION STEP 7: OVERFITTING TEST ON 20 SAMPLES")
    print("=" * 60)
    # Extract 20 fixed samples
    subset_samples = [ds[i] for i in range(20)]
    subset_batch = ctc_collate_fn(subset_samples)

    overfit_model = build_crnn(hidden_size=128)
    overfit_model.train()
    optimizer = optim.Adam(overfit_model.parameters(), lr=0.001)

    s_imgs = subset_batch["images"]
    s_targets = subset_batch["targets"]
    s_target_lens = subset_batch["target_lengths"]
    s_gts = subset_batch["raw_texts"]
    B_sub = s_imgs.size(0)

    print(f"Training on 20 samples for 80 iterations to test memorization capacity...")
    for step in range(1, 81):
        optimizer.zero_grad()
        l_probs = overfit_model(s_imgs)
        T_sub = l_probs.size(0)
        inp_lens = torch.full((B_sub,), T_sub, dtype=torch.long)
        s_loss = criterion(l_probs, s_targets, inp_lens, s_target_lens)
        s_loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            overfit_model.eval()
            with torch.no_grad():
                preds = overfit_model.predict(s_imgs)
            metrics = calculate_metrics(preds, s_gts)
            overfit_model.train()
            print(f"  Step {step:2d} | CTC Loss: {s_loss.item():.4f} | Word Acc: {metrics['word_accuracy_pct']}% | CER: {metrics['character_error_rate_pct']}%")

    print("\nSample predictions from overfitted model:")
    overfit_model.eval()
    with torch.no_grad():
        final_preds = overfit_model.predict(s_imgs)
    for i in range(5):
        print(f"  Sample {i+1} | GT: {s_gts[i]:<12} | Pred: {final_preds[i]}")


if __name__ == "__main__":
    run_diagnostics()
