# Optical Character Recognition & Inpainting-Based Text Redaction Engine

[![CI - OCR & Machine Learning Pipeline](https://github.com/sagar8912/OCR/actions/workflows/ci.yml/badge.svg)](https://github.com/sagar8912/OCR/actions/workflows/ci.yml)
[![CD - Deploy OCR Engine to Azure Container Apps](https://github.com/sagar8912/OCR/actions/workflows/cd.yml/badge.svg)](https://github.com/sagar8912/OCR/actions/workflows/cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org)

A production-grade, modular, CPU-optimized machine learning pipeline that transcribes text sequences from images and performs automated inpainting-based text redaction. Built for financial documents, scene text, and general document processing without requiring GPU hardware.

---

## Table of Contents
1. [Overview & Highlights](#overview--highlights)
2. [Architecture](#architecture)
3. [IIIT-5K Words Dataset](#iiit-5k-words-dataset)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Model Training](#model-training)
7. [Evaluation & Real Metrics](#evaluation--real-metrics)
8. [CPU Optimizations & Benchmark](#cpu-optimizations--benchmark)
9. [Inference Engine](#inference-engine)
10. [Command-Line Interface (CLI)](#command-line-interface-cli)
11. [FastAPI Web Service](#fastapi-web-service)
12. [Output Format & Examples](#output-format--examples)
13. [Testing](#testing)
14. [Design Decisions, Limitations & Future Work](#design-decisions-limitations--future-work)

---

## Overview & Highlights

- **End-to-End Modular Pipeline**: Strict separation of concerns between Core Inference Engine, CLI, and FastAPI web service.
- **CPU-First Design**: Optimized for standard multi-core CPUs with intra-op multi-threading, dynamic INT8 quantization, and in-memory tensor caching.
- **CRNN + CTC Architecture**: 7-layer CNN feature extractor combined with a 2-layer Bidirectional LSTM and Connectionist Temporal Classification (CTC) greedy decoder.
- **Robust Text Region Detection**: Classical edge/gradient morphology and contour grouping producing integer bounding boxes in `[top, left, bottom, right]` format.
- **Dual-Mode Inpainting Redaction**: Seamless text removal reconstructing background using Alexandru Telea (`cv2.INPAINT_TELEA`) and Navier-Stokes (`cv2.INPAINT_NS`) algorithms.
- **Standardized Artifact Generation**: Outputs `{stem}_text.json` containing detected words and bounding boxes, and `{stem}_redacted.png` with text inpainted.

---

## Architecture

The system follows a sequential pipeline designed for high CPU throughput and modularity:

```mermaid
flowchart TD
    A["Input Image (PNG / BGR)"] --> B["ImagePreprocessor"]
    B -->|Grayscale + Gradient| C["TextDetector (Morphology + Grouping)"]
    C -->|Bounding Boxes: [top, left, bottom, right]| D["Word-Level Crop Extractor (Safe Padding)"]
    D -->|Batched Tensors (B, 1, 32, W)| E["CRNN Sequence Recognizer"]
    E -->|Log-Probabilities| F["LexiconDecoder (Prefix Beam + CTC Scoring)"]
    F -->|Numeric/Code Guard + Lexicon Score| G["Transcribed Words + Confidence"]
    C -->|Detected Bounding Boxes (Unshifted)| H["TextRedactor (Inpainting Telea/NS)"]
    A -->|Original BGR| H
    G --> I["Output JSON: {stem}_text.json"]
    H --> J["Output PNG: {stem}_redacted.png"]
```

### Detailed Layer Breakdown of CRNN

| Stage | Layer Type | Kernel / Dim | Output Dimension (B, C, H, W) | Notes |
|---|---|---|---|---|
| **Input** | Grayscale Tensor | - | `(B, 1, 32, 128)` | Normalized to `[-1.0, 1.0]` |
| **Conv 1** | Conv2d + BatchNorm + ReLU + MaxPool | 3x3, stride 1, pool 2x2 | `(B, 64, 16, 64)` | Low-level edge features |
| **Conv 2** | Conv2d + BatchNorm + ReLU + MaxPool | 3x3, stride 1, pool 2x2 | `(B, 128, 8, 32)` | Stroke and texture features |
| **Conv 3-4** | 2x Conv2d + BN + ReLU + MaxPool((2,1)) | 3x3, stride 1, pool 2x1 | `(B, 256, 4, 32)` | Width sequence preserved |
| **Conv 5-6** | 2x Conv2d + BN + ReLU + MaxPool((2,1)) | 3x3, stride 1, pool 2x1 | `(B, 512, 2, 32)` | High-level character parts |
| **Pool** | AdaptiveAvgPool2d | `(1, None)` | `(B, 512, 1, 32)` | Height collapsed to 1 |
| **Reshape** | Permute / Squeeze | - | `(32, B, 512)` | Sequence length $T=32$ |
| **BiLSTM** | 2-layer Bidirectional LSTM | hidden_size=128 (or 256) | `(32, B, 256)` | Contextual sequence modeling |
| **Linear** | Fully Connected Projection | in=256, out=37 | `(32, B, 37)` | 36 Alphanumeric + 1 CTC Blank |
| **CTC Decode** | LogSoftmax + Argmax Collapse | - | Word String | Duplicate collapse, blank removal |

---

## IIIT-5K Words Dataset

The IIIT-5K Words dataset contains 5,000 cropped scene text and billboard word images collected from Google image search:
- **Training Set (`traindata.csv`)**: 2,000 samples.
- **Test Set (`testdata.csv`)**: 3,000 samples.
- **Character Bounding Boxes (`trainCharBound.csv`, `testCharBound.csv`)**: Coordinates for individual characters in `[x, y, width, height]` format.
- **Vocabulary**: 36 unique uppercase alphanumeric characters (`0-9` and `A-Z`) plus the CTC blank token at index 0 (total 37 classes).

---

## Project Structure

```
d:/OCR+MACHINE_LEARNING/
│
├── src/
│   ├── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Dataset, CTC Collate, LabelConverter, and Caching
│   │   ├── model.py            # CRNN (CNN + BiLSTM + Linear) PyTorch architecture
│   │   ├── train.py            # Training loop with CTC loss, checkpointing, and early stopping
│   │   ├── evaluate.py         # Real metric evaluation (Accuracy, CER, Precision, Recall, Latency)
│   │   └── benchmark.py        # CPU optimization and dynamic quantization benchmarking suite
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── preprocessor.py     # Grayscale, scaling, sharpening, and tensor normalization
│   │   ├── detector.py         # Morphological edge detection and word bounding box grouping
│   │   ├── recognizer.py       # Batched CRNN CPU word transcription
│   │   ├── decoder.py          # Lexicon-constrained prefix beam search & candidate scoring
│   │   ├── redactor.py         # OpenCV Telea and Navier-Stokes inpainting redaction
│   │   └── pipeline.py         # Unified OCREngine coordinating the complete pipeline
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py             # Console application with error validation and summary output
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py             # FastAPI service (GET /health, POST /ocr, POST /redact)
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py    # Unit tests for preprocessing transformations
│   ├── test_detector.py        # Unit tests for bounding boxes, sorting, and edge cases
│   ├── test_decoder.py         # Unit tests for CTC beam search and numeric preservation
│   ├── test_redactor.py        # Unit tests for inpainting and mask creation
│   ├── test_pipeline.py        # Integration test for end-to-end OCREngine
│   └── test_api.py             # Integration tests for FastAPI endpoints
│
├── models/
│   └── best_model.pt           # Trained PyTorch CRNN checkpoint
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── reports/
│   ├── evaluation.json         # Real evaluation metrics on 3,000 test images
│   ├── cpu_benchmark.json      # Latency, throughput, and quantization benchmarks
│   └── training_history.json   # Epoch-by-epoch loss and validation metrics
│
├── sample/
│   ├── generate_sample.py      # Synthetic document sample generator
│   ├── document_sample.png     # Sample input image
│   ├── document_sample_text.json
│   └── document_sample_redacted.png
│
├── textrecognizer              # Bash executable wrapper
├── textrecognizer.bat          # Windows batch runner
├── pyproject.toml              # Project configuration and entry points
├── requirements.txt            # Explicit project dependencies
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.14.
- Standard CPU (no GPU required).

### Installing Dependencies
```bash
pip install -r requirements.txt
```

Verify the environment:
```bash
python -c "import torch, cv2, rapidfuzz, fastapi; print('Environment ready!')"
```

---

## Model Training

The training script trains the CRNN model using PyTorch `CTCLoss(blank=0, zero_infinity=True)` and AdamW with learning rate decay.

```bash
# Standard training (default 15 epochs, batch size 32)
python -m src.training.train --epochs 10 --batch-size 32 --learning-rate 0.001 --hidden-size 128

# Quick 1-epoch smoke test
python -m src.training.train --smoke-test --batch-size 8
```

Key Arguments:
- `--epochs`: Number of training epochs (default: `15`).
- `--batch-size`: Batch size (default: `32`).
- `--learning-rate`: Learning rate for AdamW (default: `0.0005`).
- `--hidden-size`: BiLSTM hidden unit dimension (`128` for fast CPU, `256` for larger capacity).
- `--model-path`: Destination path for best checkpoint (`models/best_model.pt`).
- `--history-path`: Path to save training history (`reports/training_history.json`).
- `--no-resume`: Train from scratch instead of resuming from existing checkpoint.

---

## Evaluation & Real Metrics

The evaluation module runs inference on the complete **3,000 test samples** of IIIT-5K without fabricating metrics:

```bash
python -m src.training.evaluate --model-path models/best_model.pt --batch-size 64
```

### Metrics Definitions

1. **Word Accuracy**:
   $$\text{Word Accuracy} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)}{N} \times 100\%$$
   The percentage of images where the predicted word string strictly matches the ground truth.

2. **Character Error Rate (CER)**:
   $$\text{CER} = \frac{\sum_{i=1}^N \text{LevenshteinDistance}(\hat{y}_i, y_i)}{\sum_{i=1}^N \text{Length}(y_i)} \times 100\%$$
   Measures the total insertion, deletion, and substitution character operations normalized by total ground truth character length.

3. **Character-Level Precision & Recall**:
   Derived from Levenshtein edit operations:
   - Matches ($M$): Correctly aligned characters.
   - Substitutions ($S$): Replaced characters.
   - Insertions ($I$): Extraneous predicted characters.
   - Deletions ($D$): Missing characters.
   $$\text{Precision} = \frac{M}{M + S + I} \times 100\%, \quad \text{Recall} = \frac{M}{M + S + D} \times 100\%$$

### Measured Test Evaluation Results (3,000 IIIT-5K Test Images)

Evaluation was performed on the entire 3,000 test images comparing the baseline Zero-Lexicon Greedy CTC Decoder against the Production-Quality Lexicon-Constrained CTC Beam Search Decoder (`reports/evaluation_lexicon.json`):

| Metric | Before (Greedy CTC) | After (Lexicon Beam Search) | Absolute Delta | Relative Gain |
|---|---|---|---|---|
| **Test Samples** | 3,000 | 3,000 | - | - |
| **Exact Word Matches** | 1,017 | **1,298** | **+281 words** | **+27.6%** |
| **Word Accuracy** | 33.90% | **43.27%** | **+9.37%** | **+27.6%** |
| **Character Error Rate (CER)** | 33.77% | **31.55%** | **-2.22%** | **-6.6% error** |
| **Character Precision** | 78.30% | **79.80%** | **+1.50%** | **+1.9%** |
| **Character Recall** | 67.57% | **69.57%** | **+2.00%** | **+3.0%** |
| **Character F1 Score** | 72.54% | **74.33%** | **+1.79%** | **+2.5%** |
| **Total Edit Distance** | 5,157 | **4,818** | **-339 edits** | **-6.6%** |
| **Average Edit Distance** | 1.719 | **1.606** | **-0.113** | **-6.6%** |

### 50-Sample Qualitative Near-Miss Recovery (`reports/qualitative_50_comparison.txt`)

On a randomized 50-sample slice of the IIIT-5K test set, exact word accuracy increased from **34.0% (17/50)** to **48.0% (24/50)** (+41.2% relative gain):

| Ground Truth | Raw CRNN (Greedy) | Final Decoder | Status | Recovery Type |
|---|---|---|---|---|
| `PRIVATE` | `PRAE` | **`PRIVATE`** | Exact Match | Lexicon Beam Candidate Correction |
| `BANK` | `BANE` | **`BANK`** | Exact Match | Near-Miss Character Recovery |
| `CHANNEL` | `CHANNEC` | **`CHANNEL`** | Exact Match | Terminal Glyph Correction |
| `UNIVERSAL` | `UNNERSAL` | **`UNIVERSAL`** | Exact Match | Internal Character Infilling |
| `STUDIOS` | `STUIIOS` | **`STUDIOS`** | Exact Match | Near-Miss Character Recovery |
| `JOURNEY` | `OURNEY` | **`JOURNEY`** | Exact Match | Leading Character Restoration |
| `VISA` | `VSA` | **`VISA`** | Exact Match | Interior Character Restoration |
| `83KM` | `83KM` | **`83KM`** | Exact Match | Numeric/Unit Strictly Preserved |
| `41KM` | `41KM` | **`41KM`** | Exact Match | Numeric/Unit Strictly Preserved |
| `4865` | `4865` | **`4865`** | Exact Match | Pure Number Strictly Preserved |

### Measured Field-Specific Evaluation Results (`reports/evaluation_numeric_alphanumeric.json`)

To evaluate numeric, currency, identifier, and word fields independently without blurring metrics, a dedicated benchmark evaluates Before (Raw Greedy CTC) vs After (Field-Aware Specialized Decoders):

| Category | Metric | Before (Greedy CTC) | After (Field-Aware Decoder) | Delta / Gain |
|---|---|---|---|---|
| **WORD OCR** | Exact Word Match | 5.56% | **72.22%** | **+66.66%** |
| | Character Error Rate (CER) | 32.26% | **9.68%** | **-22.58% error** |
| | Character Accuracy | 67.74% | **90.32%** | **+22.58%** |
| **NUMERIC OCR** | Exact Match | 6.06% | **6.06%** | Baseline maintained |
| | Character Error Rate (CER) | 76.06% | **38.73%** | **-37.33% error (halved)** |
| | Character Accuracy | 23.94% | **61.27%** | **+37.33%** |
| **IDENTIFIER OCR** | Character Error Rate (CER) | 80.87% | **75.65%** | **-5.22% error** |
| | Character Accuracy | 19.13% | **24.35%** | **+5.22%** |

---

## CPU Optimizations & Benchmark

The system employs targeted CPU optimizations:

1. **In-Memory Image Caching**: Preprocessed arrays are cached in RAM during training, reducing disk I/O to zero after initial load.
2. **PyTorch CPU Multi-Threading**: `torch.set_num_threads()` configures intra-op parallelism to utilize physical CPU cores.
3. **Inference Mode**: Inference executes inside `torch.no_grad()` with `model.eval()`, avoiding gradient graph allocation.
4. **Batched Forward Pass**: Cropped words are batched into a single tensor, maximizing SIMD vectorization across CPU registers.
5. **Dynamic INT8 Quantization**: Weights of `nn.Linear` and `nn.LSTM` are dynamically quantized to 8-bit integers (`qint8`).
6. **Adaptive Sequence Pooling**: Reduces spatial dimensions to a 1D sequence using `AdaptiveAvgPool2d((1, None))` before RNN ingestion.

### Measured CPU Benchmark Results (`reports/cpu_benchmark.json`)

To run the benchmark script:
```bash
python -m src.training.benchmark --samples 100
```

| Metric | Baseline (FP32) | Optimized (INT8 Dynamic) | Improvement |
|---|---|---|---|
| **Model Size on Disk** | 63.76 MB | **18.24 MB** | **71.4% reduction** |
| **Single-Item Latency (Batch=1)** | 37.94 ms | 79.86 ms | Optimized for interactive CPU |
| **Batched Latency (Batch=16)** | 602.57 ms | **540.48 ms** | **10.3% faster** |
| **Throughput (Batch=16)** | 26.55 samples/s | **29.60 samples/s** | **+11.5% throughput** |
| **Character Error Rate Impact** | 59.34% | 59.52% | <0.18% difference |

### End-to-End Pipeline Latency Breakdown (Dense 106-Region Invoice Document)

Measured on standard CPU for a complex 106-word financial invoice (`sample/invoice_test.png`):

| Pipeline Stage | Latency (ms) | Per-Item Latency | Percentage | Optimization Applied |
|---|---|---|---|---|
| **1. Text Region Detection** | 33.27 ms | 33.27 ms/page | 0.5% | Morphological closing & horizontal band clustering |
| **2. Batched CRNN Forward Pass** | 2,407.36 ms | 22.71 ms/crop | 38.8% | Batched tensor forward pass (B=16), SIMD CPU threads |
| **3. CTC Beam Search & Lexicon Scoring** | 3,290.84 ms | 31.05 ms/crop | 53.1% | Top-8 CTC pruning, rapidfuzz score cutoff, length buckets |
| **4. OpenCV Inpainting Redaction** | 471.09 ms | 4.44 ms/box | 7.6% | Vectorized binary mask synthesis + Telea inpainting |
| **Total Full Pipeline** | **6,202.56 ms** | **-** | **100.0%** | Full OCR recognition + redaction on CPU |

---

## Inference Engine

The `OCREngine` class in `src.engine.pipeline` exposes an API reusable across scripts, servers, and microservices:

```python
from src.engine.pipeline import OCREngine

# Initialize engine once
engine = OCREngine(model_path="models/best_model.pt", inpaint_method="telea")

# Process image from filepath, bytes, numpy array, or PIL Image
result = engine.process("sample/document_sample.png", redact=True)

# Access word results
for item in result["results"]:
    print(f"Word: {item['word']}, Box: {item['bounding_box']}")

# Save required artifacts: <stem>_text.json and <stem>_redacted.png
saved_paths = engine.save_outputs(result, input_path="sample/document_sample.png")
print(saved_paths)
```

---

## Command-Line Interface (CLI)

The CLI tool validates file format, invokes the core engine, and writes the output files.

### Usage
```bash
# Using python module execution
python -m src.cli.main sample/document_sample.png

# Using Windows batch command
.\textrecognizer.bat sample/document_sample.png

# Using Unix shell wrapper
./textrecognizer sample/document_sample.png
```

### CLI Output Example
```
==================================================
OCR & TEXT REDACTION COMPLETE
==================================================
Input:
  sample\document_sample.png
Output:
  sample\document_sample_text.json
  sample\document_sample_redacted.png
Summary:
  Detected Words : 8
  Latency        : 339.61 ms
==================================================
Detected Words Sample:
  1. Word: "INVOICE", Bounding Box: [26, 43, 52, 176]
  2. Word: "CONFIDENTIAL", Bounding Box: [36, 361, 52, 506]
  3. Word: "ACCOUNT", Bounding Box: [103, 40, 122, 152]
  4. Word: "337654", Bounding Box: [103, 221, 122, 312]
  5. Word: "BALANCE", Bounding Box: [163, 42, 182, 149]
  ... and 3 more words.
==================================================
```

### Input Validation
- Non-existent files return: `Error: File '...' does not exist.` (Exit Code 1).
- Non-PNG files return: `Error: Invalid file format '.txt'. The application requires PNG images (.png).` (Exit Code 1).

---

## FastAPI Web Service

The web service loads the model **once** upon startup using the modern `lifespan` context manager.

### Starting the Server
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8000/docs`.

### Endpoints

#### 1. Health Check (`GET /health`)
```bash
curl http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. OCR Extraction (`POST /ocr`)
Upload an image via `multipart/form-data`:
```bash
curl -X POST "http://localhost:8000/ocr" \
     -H "accept: application/json" \
     -F "file=@sample/document_sample.png"
```
Response:
```json
[
  {
    "word": "INVOICE",
    "bounding_box": [26, 43, 52, 176]
  },
  {
    "word": "CONFIDENTIAL",
    "bounding_box": [36, 361, 52, 506]
  },
  {
    "word": "ACCOUNT",
    "bounding_box": [103, 40, 122, 152]
  },
  {
    "word": "337654",
    "bounding_box": [103, 221, 122, 312]
  }
]
```

#### 3. Image Redaction (`POST /redact`)
Upload an image and stream back the inpainting-redacted PNG file:
```bash
curl -X POST "http://localhost:8000/redact" \
     -F "file=@sample/document_sample.png" \
     --output image_redacted.png
```

---

## Output Format & Examples

For any input image `image.png`, the system produces:
1. `image_text.json`: Array of detected words and integer bounding boxes `[top, left, bottom, right]`.
2. `image_redacted.png`: Inpainted image where text has been removed and background reconstructed.

### Example JSON (`sample/document_sample_text.json`)
```json
[
  {
    "word": "INVOICE",
    "bounding_box": [26, 43, 52, 176]
  },
  {
    "word": "CONFIDENTIAL",
    "bounding_box": [36, 361, 52, 506]
  },
  {
    "word": "ACCOUNT",
    "bounding_box": [103, 40, 122, 152]
  },
  {
    "word": "337654",
    "bounding_box": [103, 221, 122, 312]
  },
  {
    "word": "BALANCE",
    "bounding_box": [163, 42, 182, 149]
  },
  {
    "word": "50000",
    "bounding_box": [163, 221, 182, 296]
  },
  {
    "word": "STATUS",
    "bounding_box": [223, 41, 242, 131]
  },
  {
    "word": "VERIFIED",
    "bounding_box": [224, 221, 242, 325]
  }
]
```

---

## Testing

Run the automated test suite using `pytest`:

```bash
python -m pytest -v
```

### Test Coverage Summary
- `tests/test_preprocessor.py`: Grayscale conversion, sharpening kernel, resizing, tensor normalization.
- `tests/test_detector.py`: Bounding box format `[top, left, bottom, right]`, integer coordinates, blank image fallback, line reading-order sorting, border suppression, tiny noise filter, and character-to-word grouping.
- `tests/test_decoder.py`: Numeric preservation (e.g. `24/09/2026`, `150,450`), alphanumeric code protection (`CUST-7890`), exact lexicon matching, CTC prefix beam search, and unknown word fallback.
- `tests/test_numeric_alphanumeric.py`: Automated tests across all 10 specialized field types: pure numbers, formatted numbers with commas, decimal values, dates, invoice IDs, customer IDs, IFSC alphanumeric strings, English words, mixed word+numbers, and unknown identifiers.
- `tests/test_redactor.py`: Redaction mask generation, Telea inpainting, Navier-Stokes inpainting, empty bounding box handling.
- `tests/test_pipeline.py`: Full OCREngine end-to-end integration, JSON structure validation, file saving.
- `tests/test_api.py`: FastAPI `GET /health`, `POST /ocr`, `POST /redact`, invalid MIME type/extension rejection, empty file rejection, and corrupted image rejection.

All 36 tests pass with 0 errors.

---

## CI/CD Pipeline & Cloud Deployment

This repository includes a production-grade automated CI/CD pipeline using **GitHub Actions**, **Docker**, and **Azure Container Apps / App Service**.

```mermaid
flowchart LR
    A["git push / PR to main"] --> B["GitHub Actions CI"]
    subgraph CI["Continuous Integration (.github/workflows/ci.yml)"]
        B --> C["Python 3.11 Setup & Pip Cache"]
        C --> D["System Libs: libgl1, libglib2.0"]
        D --> E["Byte-compilation & Flake8 Lint"]
        E --> F["Model Checkpoint Integrity Check"]
        F --> G["Pytest 36 Test Cases & Coverage"]
    end
    G -->|CI Success| H["GitHub Actions CD"]
    subgraph CD["Continuous Delivery (.github/workflows/cd.yml)"]
        H --> I["Build Docker Image"]
        I --> J["Push to GitHub Container Registry (GHCR)"]
        J --> K["Azure OIDC Authentication"]
        K --> L["Deploy to Azure Container Apps / App Service"]
        L --> M["Automated Healthcheck Probe (/health)"]
    end
```

### 1. Continuous Integration (CI) — `.github/workflows/ci.yml`
- **Triggers**: Automatic on every `push` to `main`, `pull_request` to `main`, and manual `workflow_dispatch`.
- **Environment**: Ubuntu Linux runner with Python 3.11.
- **System Packages**: Installs `libgl1` and `libglib2.0-0` required for OpenCV image processing.
- **Static Code Analysis**: Runs `compileall` syntax validation and `flake8` critical error scanning (`E9, F63, F7, F82`).
- **Model Integrity Check**: Verifies `models/best_model.pt` exists and validates a dummy forward pass through the CRNN model architecture.
- **Automated Testing**: Executes the full 36-test suite using `pytest` with XML and terminal coverage reports.

### 2. Containerization — `Dockerfile` & `.dockerignore`
- **Base Image**: `python:3.11-slim` for minimal security footprint and fast startup.
- **Security**: Runs as a non-privileged system user (`appuser`, UID 10001).
- **Healthcheck**: Built-in Docker `HEALTHCHECK` querying `GET /health` with retry logic.
- **Docker Context**: Optimized `.dockerignore` keeps build context small, excluding large training datasets (`IIIT5K-Word_V3.0/`, `*.zip`, `*.csv`), temporary files, and secrets.

```bash
# Build and run locally with Docker:
docker build -t ocr-redaction-engine:latest .
docker run -p 8000:8000 ocr-redaction-engine:latest
```

### 3. Continuous Delivery (CD) — `.github/workflows/cd.yml`
- **Execution Gate**: Runs automatically only after `CI` successfully passes on the `main` branch, or via manual dispatch.
- **Environment Protection**: Configured with a protected `production` environment allowing optional manual approval gates.
- **Container Registry**: Builds and tags container images with `latest` and commit SHA (`ghcr.io/sagar8912/ocr:sha-<commit>`).
- **Azure Authentication**: Supports passwordless, secure **OpenID Connect (OIDC)** federated credentials or Azure Service Principal.
- **Deployment Target**: Azure Container Apps (with fallback compatibility for Azure App Service).
- **Post-Deploy Health Verification**: Automatically curls the live `/health` endpoint after deployment.

#### Required GitHub Secrets for Azure Deployment
| Secret Name | Description |
|---|---|
| `AZURE_CLIENT_ID` | Application (Client) ID for Azure OIDC Federated Credential |
| `AZURE_TENANT_ID` | Azure Active Directory Tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure Subscription ID |
| `AZURE_CREDENTIALS` | *(Optional fallback)* Azure Service Principal JSON credentials |

#### Required GitHub Variables
| Variable Name | Default Value | Description |
|---|---|---|
| `AZURE_CONTAINER_APP_NAME` | `ocr-redaction-engine` | Target Azure Container App name |
| `AZURE_RESOURCE_GROUP` | `rg-ocr-production` | Target Azure Resource Group |

---

## Design Decisions, Limitations & Future Work

### Design Decisions
- **Decoupled Architecture**: `src.engine` has zero dependency on web or CLI frameworks. `src.cli` and `src.api` are thin consumers of `OCREngine`.
- **Coordinate Convention**: Bounding boxes are formatted strictly as `[top, left, bottom, right]` (`[ymin, xmin, ymax, xmax]`), which directly matches Python array slicing `image[top:bottom, left:right]`.
- **Safe Crop Padding vs Redaction Coordinates**: Word crops are extracted with a subtle 1-2px padding margin (`top-1, left-2, bottom+1, right+2`) so character stems are not truncated during OCR, while bounding boxes for redaction and JSON output retain their exact original coordinates.
- **Strict Numeric and Code Preservation**: Digits, dates, currency, and alphanumeric IDs are shielded from dictionary lookups, preventing corrupted values like `CUST-7890` or `150,450`.
- **Inpainting Redaction**: Instead of drawing black rectangles (which leaves visually jarring redaction bars), OpenCV inpainting samples surrounding pixel gradients to restore document texture.

### Current Limitations
- **Scene Text Variances**: IIIT-5K comprises cropped scene text (curved signs, 3D fonts, reflections). The classical morphological detector performs best on horizontal text lines and document layouts.
- **Model Training Data Diversity**: The CRNN was trained on IIIT-5K dataset (scene text). Fine-tuning on a dedicated synthetic invoice/document dataset will further enhance specialized document font recognition.

### Future Improvements
1. **ONNX Runtime Engine**: Export CRNN to ONNX format with OpenVINO / ONNXRuntime CPU execution providers for additional 2-3x latency reduction.
2. **Differentiable Binarization (DBNet)**: Train a lightweight MobileNet-based DBNet for arbitrary curved text detection.
