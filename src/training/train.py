"""Training pipeline for CRNN OCR model using CTC loss."""

import os
import time
import json
import argparse
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.dataset import get_dataloaders, DEFAULT_CONVERTER
from src.training.model import build_crnn, CRNN
from src.training.evaluate import evaluate_model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CTCLoss,
    device: str = "cpu",
    max_batches: int = 0,
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    batch_count = 0

    for i, batch in enumerate(dataloader):
        if max_batches > 0 and i >= max_batches:
            break

        images = batch["images"].to(device)           # (B, 1, H, W)
        targets = batch["targets"].to(device)         # (total_target_length,)
        target_lengths = batch["target_lengths"].to(device)  # (B,)

        optimizer.zero_grad()

        # Forward pass: outputs log_probs of shape (T, B, num_classes)
        log_probs = model(images)
        seq_len, batch_size, _ = log_probs.size()

        # Input lengths for CTC are all equal to sequence length T
        input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: CTC loss is NaN or Inf at batch {i}, skipping update.")
            continue

        loss.backward()
        # Gradient clipping prevents exploding gradients in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    learning_rate: float = 0.0005,
    model_save_path: str = "models/best_model.pt",
    history_save_path: str = "reports/training_history.json",
    hidden_size: int = 256,
    patience: int = 5,
    device: str = "cpu",
    smoke_test: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end training and validation loop."""
    print(f"\nInitializing CRNN Model (hidden_size={hidden_size}) on device={device}...")
    torch.set_num_threads(min(os.cpu_count() or 4, 8))
    model = build_crnn(hidden_size=hidden_size, device=device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    start_epoch = 1
    if resume and os.path.exists(model_save_path):
        try:
            ckpt = torch.load(model_save_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                if "hidden_size" in ckpt and ckpt["hidden_size"] != hidden_size:
                    hidden_size = ckpt["hidden_size"]
                    model = build_crnn(hidden_size=hidden_size, device=device)
                    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.5, patience=2
                    )
                model.load_state_dict(ckpt["model_state_dict"])
                best_val_acc = ckpt.get("best_val_acc", -1.0)
                best_val_loss = ckpt.get("val_metrics", {}).get("val_loss", float("inf"))
                start_epoch = ckpt.get("epoch", 0) + 1
                print(f"Resumed from {model_save_path} at epoch {start_epoch - 1} (previous best val acc: {best_val_acc:.2f}%)")
        except Exception as e:
            print(f"Warning: Could not resume from {model_save_path} ({e}), starting from scratch.")

    epochs_no_improve = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_word_acc": [],
        "val_cer": [],
        "learning_rate": [],
    }

    if smoke_test:
        print("[SMOKE TEST MODE] Running 1 epoch with 3 batches only...")
        epochs = 1

    end_epoch = start_epoch + epochs - 1
    print(f"Starting training for {epochs} epochs (epochs {start_epoch} to {end_epoch})...")
    start_total_time = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()
        max_batches = 3 if smoke_test else 0

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
        )

        # Validation evaluation
        val_metrics, _, _ = evaluate_model(model, val_loader, device=device, criterion=criterion)
        val_acc = val_metrics["word_accuracy_pct"]
        val_cer = val_metrics["character_error_rate_pct"]
        val_loss = val_metrics.get("val_loss", 0.0)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:2d}/{end_epoch:2d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val CER: {val_cer:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(val_loss)
        history["val_word_acc"].append(round(val_acc, 2))
        history["val_cer"].append(round(val_cer, 2))
        history["learning_rate"].append(current_lr)

        # Learning rate schedule step on validation loss
        scheduler.step(val_loss)

        # Checkpoint best model based on validation word accuracy or validation loss
        is_best = (val_acc > best_val_acc) or (abs(val_acc - best_val_acc) < 1e-4 and val_loss < best_val_loss)
        if is_best:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "hidden_size": hidden_size,
                },
                model_save_path,
            )
            print(f"  --> Saved new best checkpoint to {model_save_path} (Val Acc: {best_val_acc:.2f}%, Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and not smoke_test:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    total_duration = time.time() - start_total_time
    print(f"\nTraining completed in {total_duration:.1f}s. Best Val Acc: {best_val_acc:.2f}%")

    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_save_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train CRNN OCR Model on IIIT-5K")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="BiLSTM hidden size")
    parser.add_argument("--model-path", type=str, default="models/best_model.pt", help="Path to save best model")
    parser.add_argument("--history-path", type=str, default="reports/training_history.json", help="Path to save history")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-epoch smoke test")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_loader, val_loader, _ = get_dataloaders(
        train_csv="traindata.csv",
        test_csv="testdata.csv",
        img_root="IIIT5K-Word_V3.0/IIIT5K",
        batch_size=args.batch_size,
    )

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_save_path=args.model_path,
        history_save_path=args.history_path,
        hidden_size=args.hidden_size,
        patience=args.patience,
        device=args.device,
        smoke_test=args.smoke_test,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
