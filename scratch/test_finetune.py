import os
import time
import torch

torch.set_num_threads(4)

from src.training.dataset import get_dataloaders
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate_model
from src.training.model import build_crnn

def main():
    train_loader, val_loader, _ = get_dataloaders(batch_size=64)
    # Warm up dataset memory cache
    for _ in train_loader:
        pass
    for _ in val_loader:
        pass

    model = build_crnn(pretrained_path="models/best_model.pt", device="cpu")
    crit = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    print("Evaluating baseline on validation set...")
    metrics, _, _ = evaluate_model(model, val_loader)
    print(f"Start Val Acc: {metrics['word_accuracy_pct']}%, Val CER: {metrics['character_error_rate_pct']}%", flush=True)

    for ep in range(16, 21):
        t0 = time.time()
        t_loss = train_one_epoch(model, train_loader, opt, crit)
        dt = time.time() - t0
        v_metrics, _, _ = evaluate_model(model, val_loader)
        print(f"Epoch {ep} ({dt:.1f}s) | Train Loss: {t_loss:.4f} | Val Acc: {v_metrics['word_accuracy_pct']}% | CER: {v_metrics['character_error_rate_pct']}%", flush=True)

if __name__ == "__main__":
    main()
