"""Phase 5: Overfit Test on Tiny Subset (20 images) with unbuffered output."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from src.training.dataset import IIIT5KDataset, DEFAULT_CONVERTER, ctc_collate_fn
from src.training.model import build_crnn

def run_overfit_test(n_samples=20, max_epochs=70, target_acc=0.90):
    print("=" * 60, flush=True)
    print(f"PHASE 5: OVERFIT TEST on {n_samples} samples", flush=True)
    print("=" * 60, flush=True)

    df = pd.read_csv("traindata.csv").head(n_samples)
    dataset = IIIT5KDataset(
        dataframe=df,
        img_root="IIIT5K-Word_V3.0/IIIT5K",
        converter=DEFAULT_CONVERTER,
        is_train=False,
    )
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False, collate_fn=ctc_collate_fn)

    model = build_crnn(
        img_channels=1,
        num_classes=DEFAULT_CONVERTER.num_classes,
        hidden_size=256,
        device="cpu",
    )
    model.train()

    criterion = nn.CTCLoss(blank=DEFAULT_CONVERTER.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    print(f"Model instantiated. Total parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Starting overfit loop for up to {max_epochs} epochs...", flush=True)

    start_time = time.time()
    best_acc = 0.0

    batch = next(iter(dataloader))
    images = batch["images"]
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]
    raw_texts = batch["raw_texts"]
    batch_size = images.size(0)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        optimizer.zero_grad()
        log_probs = model(images)  # (T, B, C)
        T = log_probs.size(0)
        input_lengths = torch.full((batch_size,), T, dtype=torch.long)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == max_epochs:
            model.eval()
            with torch.no_grad():
                preds = model.predict(images, converter=DEFAULT_CONVERTER)
                correct = sum(1 for p, g in zip(preds, raw_texts) if p == g)
                acc = correct / len(raw_texts)
                if acc > best_acc:
                    best_acc = acc

            print(f"Epoch {epoch:2d}/{max_epochs:2d} | Loss: {loss.item():.4f} | Word Acc: {correct}/{len(raw_texts)} ({acc*100:.1f}%) | Time: {time.time()-t0:.2f}s", flush=True)
            model.train()

            if acc >= target_acc:
                print(f"\n[SUCCESS] Target accuracy {target_acc*100:.0f}% reached at epoch {epoch}!", flush=True)
                break

    elapsed = time.time() - start_time
    print("-" * 60, flush=True)
    print(f"Overfit test finished in {elapsed:.2f}s. Best Acc: {best_acc*100:.1f}%", flush=True)

    model.eval()
    with torch.no_grad():
        preds = model.predict(images, converter=DEFAULT_CONVERTER)
        print("\nPredictions vs Ground Truth:", flush=True)
        for idx, (p, g) in enumerate(zip(preds, raw_texts)):
            print(f"[{idx:2d}] GT: {g:15s} | Pred: {p:15s} | Match: {p == g}", flush=True)

    print("=" * 60, flush=True)
    assert best_acc >= target_acc, f"Overfit test failed! Best acc was {best_acc*100:.1f}%"

if __name__ == "__main__":
    run_overfit_test(n_samples=20, max_epochs=70, target_acc=0.90)
