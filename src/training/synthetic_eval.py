"""Synthetic Numeric and Alphanumeric Benchmark and Evaluation Suite.

Evaluates OCR performance across distinct field types:
  1. WORD OCR (Exact match, CER)
  2. NUMERIC OCR (Exact numeric accuracy, digit accuracy, numeric CER)
  3. IDENTIFIER OCR (Exact identifier accuracy, character accuracy)
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein

from src.engine.pipeline import OCREngine
from src.engine.decoder import LexiconDecoder, FieldType, classify_field_type
from src.training.dataset import DEFAULT_CONVERTER


def render_synthetic_image(text: str, width: int = 128, height: int = 32) -> np.ndarray:
    """Render a text string onto a clean RGB image patch mimicking document typography."""
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    w = max(width, len(text) * 14 + 24)
    img = Image.new("RGB", (w, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 4), text, fill=(0, 0, 0), font=font)
    return np.array(img)


def generate_benchmark_samples() -> List[Dict[str, str]]:
    """Generate representative benchmark samples across field categories."""
    samples = []

    # 1. Pure Numbers
    pure_numbers = [
        "123", "4567", "9876543210", "122001", "24", "2026", "500081",
        "46", "13", "62", "1", "2", "3", "4", "5", "8", "90", "100"
    ]
    for n in pure_numbers:
        samples.append({"text": n, "category": "NUMERIC"})

    # 2. Formatted Numbers / Currency
    currency_numbers = [
        "45,000", "90,000", "1,200", "6,000", "1,500", "7,500",
        "12,000", "24,000", "127,500", "22,950", "150,450", "1,000,000",
        "35,200", "18,400", "5,500"
    ]
    for c in currency_numbers:
        samples.append({"text": c, "category": "NUMERIC"})

    # 3. Identifiers
    identifiers = [
        "INV-2024-0156", "INV-2023-8901", "CUST-7890", "CUST-1245",
        "HDFC0001234", "SBIN0004567", "ICIC0009876", "UTIB0001122",
        "PO-99120", "TXN-88410", "BATCH-2026"
    ]
    for i in identifiers:
        samples.append({"text": i, "category": "IDENTIFIER"})

    # 4. Standard English Words
    words = [
        "INVOICE", "CONFIDENTIAL", "DESCRIPTION", "QUANTITY", "SUBTOTAL",
        "TOTAL", "AMOUNT", "PRICE", "UNIT", "LAPTOP", "DELL", "INSPIRON",
        "WIRELESS", "MOUSE", "KEYBOARD", "MONITOR", "BUSINESS", "PARK"
    ]
    for w in words:
        samples.append({"text": w, "category": "WORD"})

    return samples


def evaluate_numeric_and_alphanumeric(engine: OCREngine) -> Dict[str, Any]:
    """Run full evaluation comparing Before (Raw Greedy CTC) vs After (Field-Aware Decoders)."""
    samples = generate_benchmark_samples()

    results_before = {"NUMERIC": [], "IDENTIFIER": [], "WORD": []}
    results_after = {"NUMERIC": [], "IDENTIFIER": [], "WORD": []}

    conv = engine.recognizer.converter

    for s in samples:
        target = s["text"]
        cat = s["category"]

        # Render image
        patch = render_synthetic_image(target)

        # Baseline Before: Raw Greedy CTC
        tensor = engine.preprocessor.prepare_for_ocr(patch, target_width=128)
        with torch.no_grad():
            lp = engine.recognizer.model(tensor)
            greedy_pred = conv.decode_greedy(torch.argmax(lp[:, 0], dim=-1).numpy())

        # After: Field-Aware Specialized Recognition
        after_detail = engine.recognizer.recognize_crop(patch, return_details=True)
        after_pred = after_detail["word"] if isinstance(after_detail, dict) else after_detail

        # Record Before
        dist_b = Levenshtein.distance(target, greedy_pred)
        exact_b = (target == greedy_pred)
        results_before[cat].append({
            "target": target,
            "pred": greedy_pred,
            "exact": exact_b,
            "dist": dist_b,
            "len": len(target),
        })

        # Record After
        dist_a = Levenshtein.distance(target, after_pred)
        exact_a = (target == after_pred)
        results_after[cat].append({
            "target": target,
            "pred": after_pred,
            "exact": exact_a,
            "dist": dist_a,
            "len": len(target),
        })

    def calc_metrics(res_list):
        total = len(res_list)
        exact_count = sum(1 for r in res_list if r["exact"])
        total_dist = sum(r["dist"] for r in res_list)
        total_len = sum(r["len"] for r in res_list)
        word_acc = (exact_count / max(total, 1)) * 100.0
        cer = (total_dist / max(total_len, 1)) * 100.0
        char_acc = max(0.0, 100.0 - cer)
        return {
            "total_samples": total,
            "exact_matches": exact_count,
            "exact_accuracy": round(word_acc, 2),
            "cer": round(cer, 2),
            "char_accuracy": round(char_acc, 2),
            "avg_edit_distance": round(total_dist / max(total, 1), 3),
        }

    report = {
        "WORD_OCR": {
            "before": calc_metrics(results_before["WORD"]),
            "after": calc_metrics(results_after["WORD"]),
        },
        "NUMERIC_OCR": {
            "before": calc_metrics(results_before["NUMERIC"]),
            "after": calc_metrics(results_after["NUMERIC"]),
        },
        "IDENTIFIER_OCR": {
            "before": calc_metrics(results_before["IDENTIFIER"]),
            "after": calc_metrics(results_after["IDENTIFIER"]),
        },
    }

    return report


if __name__ == "__main__":
    engine = OCREngine(model_path="models/best_model.pt", device="cpu")
    print("Running Synthetic Numeric and Alphanumeric Benchmark...")
    report = evaluate_numeric_and_alphanumeric(engine)

    out_path = Path("reports/evaluation_numeric_alphanumeric.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    for cat, data in report.items():
        print(f"\n--- {cat} ---")
        b = data["before"]
        a = data["after"]
        print(f"  Exact Accuracy: {b['exact_accuracy']}%  -->  {a['exact_accuracy']}% (+{a['exact_accuracy'] - b['exact_accuracy']:.2f}%)")
        print(f"  CER           : {b['cer']}%  -->  {a['cer']}% ({a['cer'] - b['cer']:.2f}%)")
        print(f"  Char Accuracy : {b['char_accuracy']}%  -->  {a['char_accuracy']}%")
    print(f"\nSaved detailed metrics report to: {out_path}")
