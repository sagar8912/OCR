"""Evaluation metrics and evaluation script for the OCR model."""

import json
import os
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional

import torch
from rapidfuzz.distance import Levenshtein

from src.training.dataset import (
    get_dataloaders,
    DEFAULT_CONVERTER,
    LabelConverter,
)
from src.training.model import build_crnn, CRNN


def compute_alignment_stats(pred: str, gt: str) -> Tuple[int, int, int, int]:
    """Compute matches, substitutions, insertions, deletions via Levenshtein edit ops.

    Returns:
        (matches, substitutions, insertions, deletions)
    """
    edit_ops = Levenshtein.editops(gt, pred)
    subs = sum(1 for op in edit_ops if op.tag == "replace")
    deletes = sum(1 for op in edit_ops if op.tag == "delete")
    inserts = sum(1 for op in edit_ops if op.tag == "insert")
    matches = len(gt) - subs - deletes
    return matches, subs, inserts, deletes


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str],
    inference_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """Compute Word Accuracy, Character Error Rate (CER), Precision, Recall, and Latency."""
    total_samples = len(ground_truths)
    if total_samples == 0:
        return {}

    exact_matches = 0
    total_edit_distance = 0
    total_gt_chars = 0
    total_pred_chars = 0

    total_char_matches = 0
    total_char_subs = 0
    total_char_inserts = 0
    total_char_deletes = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_clean = str(pred).strip().upper()
        gt_clean = str(gt).strip().upper()

        # Word-level exact match
        if pred_clean == gt_clean:
            exact_matches += 1

        dist = Levenshtein.distance(pred_clean, gt_clean)
        total_edit_distance += dist
        total_gt_chars += len(gt_clean)
        total_pred_chars += len(pred_clean)

        matches, subs, inserts, deletes = compute_alignment_stats(pred_clean, gt_clean)
        total_char_matches += matches
        total_char_subs += subs
        total_char_inserts += inserts
        total_char_deletes += deletes

    # Exact match word accuracy
    word_accuracy = (exact_matches / total_samples) * 100.0

    # Character Error Rate (CER) = Total Edit Distance / Total Ground Truth Characters
    cer = (total_edit_distance / max(total_gt_chars, 1)) * 100.0

    # Average Edit Distance per sample
    avg_edit_distance = total_edit_distance / total_samples

    # Character-level Precision and Recall
    # Precision = correctly matched chars / total predicted chars
    char_precision = (
        (total_char_matches / max(total_char_matches + total_char_subs + total_char_inserts, 1))
        * 100.0
    )
    # Recall = correctly matched chars / total ground truth chars
    char_recall = (
        (total_char_matches / max(total_char_matches + total_char_subs + total_char_deletes, 1))
        * 100.0
    )
    # F1 Score
    if char_precision + char_recall > 0:
        char_f1 = (2 * char_precision * char_recall) / (char_precision + char_recall)
    else:
        char_f1 = 0.0

    # Latency per sample in milliseconds
    avg_latency_ms = (inference_time_sec / total_samples) * 1000.0 if inference_time_sec > 0 else 0.0

    return {
        "num_samples": total_samples,
        "exact_matches": exact_matches,
        "word_accuracy_pct": round(word_accuracy, 2),
        "total_edit_distance": total_edit_distance,
        "avg_edit_distance": round(avg_edit_distance, 4),
        "character_error_rate_pct": round(cer, 2),
        "character_precision_pct": round(char_precision, 2),
        "character_recall_pct": round(char_recall, 2),
        "character_f1_pct": round(char_f1, 2),
        "total_inference_time_sec": round(inference_time_sec, 4),
        "avg_latency_ms_per_sample": round(avg_latency_ms, 2),
    }


def evaluate_model(
    model: CRNN,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    converter: LabelConverter = DEFAULT_CONVERTER,
    criterion: Optional[torch.nn.CTCLoss] = None,
    decoder: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Evaluate model on a DataLoader using CPU/device inference."""
    model.eval()
    model.to(device)

    all_predictions: List[str] = []
    all_ground_truths: List[str] = []
    total_loss = 0.0
    loss_batches = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            raw_texts = batch["raw_texts"]

            log_probs = model(images)
            if criterion is not None and "targets" in batch:
                seq_len, b_size, _ = log_probs.size()
                inp_lens = torch.full((b_size,), seq_len, dtype=torch.long, device=device)
                targets = batch["targets"].to(device)
                tgt_lens = batch["target_lengths"].to(device)
                batch_loss = criterion(log_probs, targets, inp_lens, tgt_lens)
                if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                    total_loss += batch_loss.item()
                    loss_batches += 1

            best_indices = torch.argmax(log_probs, dim=-1).transpose(0, 1)
            b_size = images.size(0)
            for b in range(b_size):
                greedy = converter.decode_greedy(best_indices[b])
                if decoder is not None:
                    final_word = decoder.decode(log_probs[:, b : b + 1], raw_text=greedy)["word"]
                else:
                    final_word = greedy
                all_predictions.append(final_word)
            all_ground_truths.extend(raw_texts)

    elapsed_time = time.perf_counter() - start_time
    metrics = calculate_metrics(all_predictions, all_ground_truths, elapsed_time)
    if criterion is not None and loss_batches > 0:
        metrics["val_loss"] = round(total_loss / loss_batches, 4)
    return metrics, all_predictions, all_ground_truths


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR model on IIIT-5K test set")
    parser.add_argument("--model-path", type=str, default="models/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--test-csv", type=str, default="testdata.csv", help="Path to test CSV")
    parser.add_argument("--img-root", type=str, default="IIIT5K-Word_V3.0/IIIT5K", help="Root folder for images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--report-path", type=str, default="reports/evaluation.json", help="Path to save JSON metrics")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--no-lexicon", action="store_true", help="Disable lexicon-constrained decoding")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} onto {args.device}...")
    model = build_crnn(pretrained_path=args.model_path, device=args.device)

    decoder = None
    if not args.no_lexicon:
        from src.engine.decoder import LexiconDecoder
        print("Initializing LexiconDecoder for beam search & lexicon scoring...")
        decoder = LexiconDecoder()

    print(f"Loading test dataset from {args.test_csv}...")
    from src.training.dataset import IIIT5KDataset, ctc_collate_fn
    from torch.utils.data import DataLoader

    test_dataset = IIIT5KDataset(args.test_csv, img_root=args.img_root, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    print(f"Evaluating {len(test_dataset)} test samples...")
    metrics, preds, gts = evaluate_model(model, test_loader, device=args.device, decoder=decoder)

    print("\n========== EVALUATION RESULTS ==========")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("========================================")

    # Show first 10 sample predictions vs ground truths
    print("\nSample Predictions:")
    for i in range(min(10, len(preds))):
        print(f"  Sample {i + 1:2d} | GT: {gts[i]:<15} | Pred: {preds[i]}")

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nEvaluation metrics successfully written to: {args.report_path}")


if __name__ == "__main__":
    main()
