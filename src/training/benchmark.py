"""CPU Optimization Benchmarking Suite.

Measures and compares:
- Model size (FP32 vs INT8 Quantized)
- Single-item and batched inference latencies
- Throughput (samples/sec)
- Pipeline stage breakdown (preprocess, detect, recognize, redact)
- Accuracy retention after quantization
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import cv2
import torch
import torch.nn as nn

from src.training.model import build_crnn, CRNN
from src.training.dataset import get_dataloaders, DEFAULT_CONVERTER
from src.training.evaluate import calculate_metrics
from src.engine.pipeline import OCREngine


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate in-memory parameter footprint in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def benchmark_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = 10,
    test_runs: int = 50,
) -> Dict[str, float]:
    """Measure single-forward-pass latency and throughput on CPU."""
    model.eval()

    # Warmup runs to populate CPU instruction caches
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Timed benchmark runs
    latencies = []
    with torch.no_grad():
        for _ in range(test_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    batch_size = input_tensor.size(0)
    avg_latency = sum(latencies) / len(latencies)
    throughput = (batch_size * 1000.0) / avg_latency if avg_latency > 0 else 0.0

    return {
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "throughput_samples_per_sec": round(throughput, 2),
    }


def run_benchmark(
    model_path: str = "models/best_model.pt",
    test_csv: str = "testdata.csv",
    num_eval_samples: int = 200,
    output_path: str = "reports/cpu_benchmark.json",
) -> Dict[str, Any]:
    """Run full CPU benchmark comparing Baseline FP32 vs Optimized/Quantized INT8."""
    print("=" * 60)
    print("CPU OCR INFERENCE BENCHMARK")
    print("=" * 60)

    # Ensure CPU threading is optimal
    num_threads = min(os.cpu_count() or 4, 8)
    torch.set_num_threads(num_threads)
    print(f"Configured PyTorch CPU threads: {torch.get_num_threads()} (System CPU count: {os.cpu_count()})")

    # 1. Load Baseline Model (FP32)
    print(f"Loading baseline FP32 model from {model_path}...")
    baseline_model = build_crnn(pretrained_path=model_path, device="cpu")
    baseline_model.eval()

    # 2. Dynamic INT8 Quantization
    print("Applying dynamic INT8 quantization on Linear and LSTM layers...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            baseline_model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        quantized_model.eval()
        has_quant = True
    except Exception as e:
        print(f"Quantization failed: {e}")
        quantized_model = baseline_model
        has_quant = False

    # 3. Model Sizes
    checkpoint_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    baseline_mem_mb = get_model_size_mb(baseline_model)

    # Save temp quantized model to measure disk footprint
    temp_quant_path = "models/temp_quant.pt"
    torch.save(quantized_model.state_dict(), temp_quant_path)
    quant_disk_mb = os.path.getsize(temp_quant_path) / (1024 * 1024)
    if os.path.exists(temp_quant_path):
        os.remove(temp_quant_path)

    print(f"Baseline checkpoint size : {checkpoint_size_mb:.2f} MB")
    print(f"Quantized model size     : {quant_disk_mb:.2f} MB (Reduction: {(1 - quant_disk_mb/checkpoint_size_mb)*100:.1f}%)")

    # 4. Latency: Batch size 1 (Interactive / Realtime latency)
    single_input = torch.randn(1, 1, 32, 128)
    print("\nMeasuring single-item latency (Batch Size = 1)...")
    base_single = benchmark_latency(baseline_model, single_input)
    quant_single = benchmark_latency(quantized_model, single_input) if has_quant else base_single
    print(f"  Baseline FP32   : {base_single['avg_latency_ms']} ms/item ({base_single['throughput_samples_per_sec']} items/s)")
    print(f"  Quantized INT8  : {quant_single['avg_latency_ms']} ms/item ({quant_single['throughput_samples_per_sec']} items/s)")

    # 5. Latency: Batched (Batch Size = 16)
    batch_input = torch.randn(16, 1, 32, 128)
    print("\nMeasuring batched latency (Batch Size = 16)...")
    base_batch = benchmark_latency(baseline_model, batch_input)
    quant_batch = benchmark_latency(quantized_model, batch_input) if has_quant else base_batch
    print(f"  Baseline FP32   : {base_batch['avg_latency_ms']} ms/batch ({base_batch['throughput_samples_per_sec']} items/s)")
    print(f"  Quantized INT8  : {quant_batch['avg_latency_ms']} ms/batch ({quant_batch['throughput_samples_per_sec']} items/s)")

    # 6. Accuracy Impact on a test subset
    print(f"\nEvaluating accuracy retention on {num_eval_samples} test samples...")
    from src.training.dataset import IIIT5KDataset, ctc_collate_fn
    from torch.utils.data import DataLoader

    test_ds = IIIT5KDataset(test_csv, is_train=False)
    sub_indices = list(range(min(num_eval_samples, len(test_ds))))
    sub_loader = DataLoader(
        torch.utils.data.Subset(test_ds, sub_indices),
        batch_size=16,
        collate_fn=ctc_collate_fn,
    )

    base_preds, quant_preds, gts = [], [], []
    with torch.no_grad():
        for batch in sub_loader:
            imgs = batch["images"]
            gts.extend(batch["raw_texts"])
            base_preds.extend(baseline_model.predict(imgs))
            quant_preds.extend(quantized_model.predict(imgs))

    base_metrics = calculate_metrics(base_preds, gts)
    quant_metrics = calculate_metrics(quant_preds, gts)
    print(f"  Baseline Accuracy  : {base_metrics['word_accuracy_pct']}% | CER: {base_metrics['character_error_rate_pct']}%")
    print(f"  Quantized Accuracy : {quant_metrics['word_accuracy_pct']}% | CER: {quant_metrics['character_error_rate_pct']}%")

    # 7. End-to-end Pipeline Stage Breakdown
    print("\nMeasuring end-to-end pipeline stage latencies on sample image...")
    engine = OCREngine(model_path=model_path)
    sample_doc = "sample/document_sample.png"
    if os.path.exists(sample_doc):
        img = engine.load_image(sample_doc)

        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        prep = engine.preprocessor.process(img)
        t_prep = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Detection
        t0 = time.perf_counter()
        boxes = engine.detector.detect(prep)
        t_det = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Recognition
        crops = [img[b[0]:b[2], b[1]:b[3]] for b in boxes]
        t0 = time.perf_counter()
        _ = engine.recognizer.recognize_batch(crops)
        t_rec = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Inpainting / Redaction
        t0 = time.perf_counter()
        _ = engine.redactor.redact(img, boxes)
        t_red = (time.perf_counter() - t0) * 1000.0

        total_pipe = t_prep + t_det + t_rec + t_red
        pipeline_breakdown = {
            "preprocessing_ms": round(t_prep, 2),
            "detection_ms": round(t_det, 2),
            "recognition_ms": round(t_rec, 2),
            "redaction_ms": round(t_red, 2),
            "total_ms": round(total_pipe, 2),
            "detected_regions": len(boxes),
        }
        print(f"  Preprocessing : {t_prep:.2f} ms")
        print(f"  Detection     : {t_det:.2f} ms")
        print(f"  Recognition   : {t_rec:.2f} ms")
        print(f"  Redaction     : {t_red:.2f} ms")
        print(f"  Total Pipeline: {total_pipe:.2f} ms")
    else:
        pipeline_breakdown = {}

    benchmark_report = {
        "hardware": {
            "device": "cpu",
            "cpu_cores": os.cpu_count(),
            "pytorch_threads": torch.get_num_threads(),
        },
        "model_footprint": {
            "baseline_checkpoint_mb": round(checkpoint_size_mb, 2),
            "quantized_checkpoint_mb": round(quant_disk_mb, 2),
            "size_reduction_pct": round((1 - quant_disk_mb / checkpoint_size_mb) * 100, 1),
        },
        "single_item_batch_1": {
            "baseline_ms": base_single["avg_latency_ms"],
            "quantized_ms": quant_single["avg_latency_ms"],
            "speedup_ratio": round(base_single["avg_latency_ms"] / max(quant_single["avg_latency_ms"], 0.001), 2),
            "baseline_throughput_items_per_sec": base_single["throughput_samples_per_sec"],
            "quantized_throughput_items_per_sec": quant_single["throughput_samples_per_sec"],
        },
        "batched_batch_16": {
            "baseline_batch_ms": base_batch["avg_latency_ms"],
            "quantized_batch_ms": quant_batch["avg_latency_ms"],
            "baseline_per_item_ms": round(base_batch["avg_latency_ms"] / 16.0, 2),
            "quantized_per_item_ms": round(quant_batch["avg_latency_ms"] / 16.0, 2),
            "throughput_items_per_sec": quant_batch["throughput_samples_per_sec"],
        },
        "accuracy_comparison": {
            "eval_samples": len(gts),
            "baseline": base_metrics,
            "quantized": quant_metrics,
        },
        "pipeline_breakdown": pipeline_breakdown,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=4)
    print(f"\nBenchmark report successfully written to: {output_path}")

    return benchmark_report


def main():
    parser = argparse.ArgumentParser(description="Run CPU OCR Benchmark")
    parser.add_argument("--model-path", type=str, default="models/best_model.pt")
    parser.add_argument("--test-csv", type=str, default="testdata.csv")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="reports/cpu_benchmark.json")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        test_csv=args.test_csv,
        num_eval_samples=args.samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
