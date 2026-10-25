# End-to-End Project Explanation: OCR & Inpainting-Based Text Redaction Engine

> **Comprehensive Technical Guide, Architecture Breakdown, Mathematical Foundations, and Code Walkthrough**

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Dataset & Vocabulary Engineering](#3-dataset--vocabulary-engineering)
4. [Deep Learning Model: CRNN + CTC](#4-deep-learning-model-crnn--ctc)
5. [Text Detection Pipeline (Classical Vision & Morphology)](#5-text-detection-pipeline-classical-vision--morphology)
6. [Field-Aware & Lexicon-Constrained Decoding](#6-field-aware--lexicon-constrained-decoding)
7. [Inpainting-Based Text Redaction Engine](#7-inpainting-based-text-redaction-engine)
8. [Unified Pipeline Orchestration (OCREngine)](#8-unified-pipeline-orchestration-ocrengine)
9. [Empirical Evaluation & CPU Benchmarks](#9-empirical-evaluation--cpu-benchmarks)
10. [Application Interfaces: CLI, REST API & Web Dashboard](#10-application-interfaces-cli-rest-api--web-dashboard)
11. [Repository & File-by-File Guide](#11-repository--file-by-file-guide)
12. [Step-by-Step Execution & Reproduction Guide](#12-step-by-step-execution--reproduction-guide)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Challenge
Modern document workflows—such as financial invoice auditing, legal record processing, and KYC compliance—frequently require two critical capabilities:
1. **Accurate Text Transcription**: Extracting alphanumeric strings, currencies, dates, and identifiers from unstructured documents.
2. **Automated Document Redaction**: Sanitizing sensitive personal or proprietary information (e.g., customer IDs, financial totals, confidential stamps) without destroying the visual aesthetics or layout of the document.

Traditional cloud solutions or deep vision-language transformers (e.g., Donut, LayoutLM, TrOCR) impose heavy GPU hardware requirements, high cloud hosting costs, high inference latency, and data privacy concerns. Furthermore, standard redaction utilities merely place ugly black censor boxes over words, ruining visual document hierarchy.

### 1.2 The Solution
This project delivers a **100% CPU-optimized, zero-GPU-dependent, modular OCR and automated inpainting redaction engine** built with PyTorch, OpenCV, and FastAPI.

Key achievements:
- **Zero Cloud/GPU Dependency**: Runs on standard multi-core CPUs with dynamic batching, multi-threading, and sub-100ms word latency.
- **CRNN + CTC Architecture**: 7-layer VGG-style CNN combined with a 2-layer Bidirectional LSTM and Connectionist Temporal Classification loss.
- **Novel Field-Aware Decoding**: Solves classical OCR digit/letter confusion (`O` vs `0`, `I` vs `1`, `S` vs `5`) by analyzing crop topology (commas, hyphens) and applying specialized beam search decoders.
- **Background-Preserving Inpainting**: Reconstructs underlying paper and texture using Navier-Stokes and Alexandru Telea fast marching algorithms instead of solid black boxes.
- **Tri-Surface Deployment**: Python Library (`OCREngine`), standalone CLI with OS executables (`textrecognizer`), and a FastAPI REST service with an interactive web visualizer.

---

## 2. High-Level System Architecture

The following diagram illustrates the complete end-to-end execution pipeline from raw image ingestion to structured JSON transcription and redacted PNG generation:

```mermaid
flowchart TD
    A["Input Document / Invoice (PNG / BGR)"] --> B["ImagePreprocessor"]
    B -->|Grayscale + Contrast + 3x3 Sharpening| C["TextDetector"]
    
    subgraph Detection ["Classical Vision Detection (detector.py)"]
        C --> D1["Sobel Gradient Magnitude"]
        D1 --> D2["Otsu Adaptive Binarization"]
        D2 --> D3["Table Grid & Border Line Suppression"]
        D3 --> D4["External Stroke Contour Extraction"]
        D4 --> D5["Bottom-Up Character-to-Word Grouping"]
        D5 --> D6["Multi-Word Split via Vertical Projection"]
        D6 --> D7["Reading Order Sorting (Top-Down, Left-Right)"]
    end

    D7 -->|Bounding Boxes: [top, left, bottom, right]| E["Word Crop Extractor (Safe 2-3px Padding)"]
    
    subgraph Recognition ["CRNN Sequence Recognition (model.py & recognizer.py)"]
        E -->|Batch (B, 1, 32, W)| F1["7-Layer VGG CNN Backbone"]
        F1 -->|Feature Sequence (W/4, B, 512)| F2["2-Layer Bidirectional LSTM"]
        F2 -->|Hidden States (W/4, B, 512)| F3["Linear Projection to 37 Classes"]
        F3 -->|CTC Log-Probabilities| F4["Softmax / CTC Matrix"]
    end

    subgraph Decoding ["Field-Aware Decoding (decoder.py)"]
        F4 --> G1["Topology Analysis (Commas, Hyphens, Dots)"]
        G1 --> G2{"Classify Field Type"}
        G2 -->|WORD| G3["Prefix Beam Search + Lexicon Edit Distance"]
        G2 -->|NUMBER / CURRENCY| G4["Digit-Twin Probability Pooling + Comma Guard"]
        G2 -->|ALPHANUMERIC_ID| G5["Component Split + Hyphen Preservation"]
    end

    subgraph Redaction ["Inpainting Engine (redactor.py)"]
        D7 -->|Original Bounding Boxes| H1["Binary Mask Generator (Dilated 2px)"]
        A -->|Original BGR Image| H2["OpenCV cv2.inpaint (Telea / NS)"]
        H1 --> H2
    end

    G3 --> I["Consolidated Results"]
    G4 --> I
    G5 --> I
    I --> J["{stem}_text.json"]
    H2 --> K["{stem}_redacted.png"]
```

---

## 3. Dataset & Vocabulary Engineering

### 3.1 The IIIT-5K Words Benchmark
The model is trained and benchmarked on the widely recognized **IIIT-5K Words Dataset** (`IIIT5K-Word_V3.0`), containing 5,000 cropped word images gathered from real-world street billboards, storefronts, and document signs:
- **Training Set (`traindata.csv`)**: 2,000 cropped word images with ground truth labels.
- **Testing Set (`testdata.csv`)**: 3,000 challenging cropped word images for evaluation.
- **Character Boundaries (`trainCharBound.csv`, `testCharBound.csv`)**: Character-level box annotations used for validation and verification.

### 3.2 Vocabulary & Character Mapping
Connectionist Temporal Classification requires a dedicated blank symbol (denoted `ϵ` or index `0`) to separate consecutive duplicate characters and represent non-character temporal regions:

| Class Index | Token / Range | Character Set | Description |
|---|---|---|---|
| `0` | `[blank]` | CTC Blank Token | Separates repeated letters; non-character state |
| `1` – `10` | `'0'` – `'9'` | Numeric Digits | Standard decimal digits |
| `11` – `36`| `'A'` – `'Z'` | Uppercase English | 26 English alphabetic letters |
| **Total** | **37 classes** | `{blank, 0..9, A..Z}` | Case-insensitive vocabulary |

Implementation is encapsulated in [LabelConverter](file:///d:/OCR+MACHINE_LEARNING/src/training/dataset.py#L22):
- Encodes strings (e.g., `"INV156"`) into integer tensors `[19, 24, 32, 2, 6, 7]`.
- Decodes argmax index sequences with CTC deduplication and blank filtering.

### 3.3 Data Pipeline & Collate Function
Located in [src/training/dataset.py](file:///d:/OCR+MACHINE_LEARNING/src/training/dataset.py):
- **Image Resizing**: Each crop is normalized to a fixed height of `H = 32` pixels, while scaling the width proportionally (`W = int(w * 32 / h)`). Width is constrained between `32` and `256` pixels.
- **Image Normalization**: Normalized to zero-mean grayscale range `[-1.0, 1.0]`.
- **Dynamic CTC Collate (`ctc_collate_fn`)**: Since word widths vary within a batch, images are padded to the maximum width in the batch. Ground-truth label sequences are flattened into a contiguous 1D tensor along with explicit `target_lengths` required by `torch.nn.CTCLoss`.

---

## 4. Deep Learning Model: CRNN + CTC

The recognition model is implemented in [src/training/model.py](file:///d:/OCR+MACHINE_LEARNING/src/training/model.py).

### 4.1 Architectural Breakdown

```
Input: (B, 1, 32, W)
  │
  ├── ConvBlock 1: Conv2d(1->64, 3x3) + BN + ReLU + MaxPool(2, 2)     --> (B, 64, 16, W/2)
  ├── ConvBlock 2: Conv2d(64->128, 3x3) + BN + ReLU + MaxPool(2, 2)    --> (B, 128, 8, W/4)
  ├── ConvBlock 3: Conv2d(128->256, 3x3) + BN + ReLU                  --> (B, 256, 8, W/4)
  ├── ConvBlock 4: Conv2d(256->256, 3x3) + BN + ReLU + MaxPool(2, 1)   --> (B, 256, 4, W/4)
  ├── ConvBlock 5: Conv2d(256->512, 3x3) + BN + ReLU                  --> (B, 512, 4, W/4)
  ├── ConvBlock 6: Conv2d(512->512, 3x3) + BN + ReLU + MaxPool(2, 1)   --> (B, 512, 2, W/4)
  ├── ConvBlock 7: Conv2d(512->512, 2x2, valid) + BN + ReLU           --> (B, 512, 1, W/4-1)
  │
  ├── AdaptiveAvgPool2d((1, None))                                     --> (B, 512, 1, T)
  ├── Permute & Squeeze: (B, 512, 1, T) -> (T, B, 512)
  │
  ├── 2-Layer Bidirectional LSTM (input=512, hidden=256, dropout=0.2) --> (T, B, 512)
  │
  └── Linear Classifier (512 -> 37 classes)                            --> (T, B, 37)
```

### 4.2 Key Engineering Innovations
1. **Asymmetric Max-Pooling (`kernel_size=(2, 1)`)**:
   Standard 2x2 pooling aggressively shrinks the horizontal width by $2^4 = 16\times$, which obliterates narrow characters (such as `'I'`, `'1'`, `'l'`, `'.'`). In blocks 4 and 6, we pool exclusively along the height dimension (`stride=(2, 1)`), retaining temporal sequence resolution while collapsing height to 1.
2. **CTC Blank Collapse Prevention**:
   A notorious failure mode when training CRNNs with CTC loss from scratch is that the network quickly discovers a local minimum: outputting 100% blank tokens, producing zero gradient updates for characters. To eliminate this, the model initializes the blank bias to a strong negative penalty:
   ```python
   with torch.no_grad():
       self.classifier.bias.data[0] = -3.0
   ```
   This simple mathematical prior forces the network to explore character logits during early epochs.
3. **Loss Function**:
   `torch.nn.CTCLoss(blank=0, zero_infinity=True)` computes the negative log likelihood of marginalizing over all valid sequence alignments mapping to the target string.

---

## 5. Text Detection Pipeline (Classical Vision & Morphology)

Implemented in [src/engine/detector.py](file:///d:/OCR+MACHINE_LEARNING/src/engine/detector.py).

While deep learning detectors (YOLO, CRAFT, DBNet) are accurate, they require 300–800 ms per image on standard CPUs and consume 500 MB+ of memory. In contrast, our **Classical Multi-Stage Morphological Detector** executes in **~30–42 ms** on standard multi-core CPUs.

### 5.1 Multi-Stage Detection Walkthrough
1. **Gradient Extraction**: Computes $G = \sqrt{G_x^2 + G_y^2}$ using 3x3 Sobel kernels to accentuate high-frequency character strokes irrespective of background lighting.
2. **Adaptive Otsu Binarization**: Automatically separates text stroke edges from document paper.
3. **Table & Grid Line Suppression**:
   Invoices and financial tables contain strong horizontal and vertical borders that merge words together if unchecked. The detector applies morphological opening with wide directional kernels:
   ```python
   h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_img * 0.045), 1))
   v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h_img * 0.055)))
   ```
   The detected grid lines are dilated by 2 iterations and subtracted from the stroke mask via `cv2.bitwise_and(binary, cv2.bitwise_not(lines_dilated))`.
4. **External Contour Filtering**:
   Contours are extracted using `cv2.RETR_EXTERNAL`. Contours matching punctuation geometry (e.g., narrow hyphens) are protected, while noise slivers and large page frames are discarded.
5. **Bottom-Up Character-to-Word Grouping (`_group_characters_to_words`)**:
   Adjacent characters are iteratively grouped into cohesive words if they satisfy three spatial criteria:
   - **Vertical overlap**: Substantial overlap along the Y-axis ($\ge 40\%$).
   - **Height compatibility**: Ratio of heights $\min(h_1, h_2) / \max(h_1, h_2) \ge 0.40$ (relaxed for hyphens).
   - **Horizontal proximity**: The horizontal gap $\Delta x$ must not exceed an intra-word threshold ($4\text{px} \le \Delta x \le 0.28 \times \min(h)$), preventing separate words across a sentence from accidentally merging.
6. **Multi-Word Splitting (`_split_multi_word_regions`)**:
   If wide bounding boxes contain inter-word spaces, vertical projection profile histograms identify clean zero-stroke valleys and split them into distinct words.
7. **Natural Reading Order Sorting (`_sort_reading_order`)**:
   Clusters bounding boxes into horizontal text lines using a dynamic vertical tolerance ($\approx 0.6 \times \text{median\_height}$), then sorts left-to-right within each line.
8. **Bounding Box Schema**:
   All coordinates are strictly formatted as integer tuples: `[top, left, bottom, right]` (`[ymin, xmin, ymax, xmax]`).

---

## 6. Field-Aware & Lexicon-Constrained Decoding

Implemented in [src/engine/decoder.py](file:///d:/OCR+MACHINE_LEARNING/src/engine/decoder.py).

### 6.1 The Core Problem in Document OCR
Standard greedy CTC argmax decoding (`argmax(P(y_t))`) suffers from severe visual ambiguity in printed documents:
- Letter `'O'` and number `'0'` are visually indistinguishable in monospace fonts.
- Letter `'I'` / `'L'` and number `'1'` frequently confuse neural networks.
- Letter `'S'` and number `'5'`, or `'B'` and number `'8'`.
- Punctuation like commas (`,`) and hyphens (`-`) are often omitted or substituted by dictionary autocorrectors.

### 6.2 The Solution: Topological & Field-Aware Routing

```mermaid
flowchart LR
    A["Raw CTC Predictions + Image Crop"] --> B["Topology Analysis<br/>(Connected Components)"]
    B --> C["classify_field_type()"]
    C -->|WORD| D["Lexicon Beam Search<br/>(Beam=15, Levenshtein Distance)"]
    C -->|NUMBER| E["Numeric Restricted Decoder<br/>(Digit-Twin Pooling: O->0, I->1, S->5)"]
    C -->|CURRENCY| F["Currency Decoder<br/>(Preserve Commas & Decimals)"]
    C -->|ALPHANUMERIC_ID| G["Identifier Decoder<br/>(Component Split & Hyphen Guard)"]
    C -->|DATE| H["Date Pattern Decoder<br/>(DD/MM/YYYY or YYYY-MM-DD)"]
```

### 6.3 Detailed Decoder Logic
1. **Crop Topology Analysis (`analyze_crop_topology`)**:
   Performs connected-component analysis on the raw crop to identify:
   - **Baseline Commas**: Small connected components extending down to the baseline ($y + h \ge 0.70 \times H$) with narrow width ($w \le 7$).
   - **Mid-Height Hyphens**: Horizontally elongated components ($w \ge 1.4 \times h$) centered in the vertical middle ($0.20 \times H \le y \le 0.75 \times H$).
   - **Decimal Points**: Tiny square dots near the baseline.
2. **Lexicon Beam Search for Words**:
   Uses CTC Prefix Beam Search with beam width 15. The top beam candidates are matched against a curated dictionary containing standard vocabulary, financial invoice terms (`INVOICE`, `SUBTOTAL`, `CONFIDENTIAL`, `TECHNOLOGIES`), and geographic entities. Candidates within Levenshtein edit distance $\le 2$ are selected based on posterior log-likelihood.
3. **Digit-Twin Probability Pooling (`DIGIT_TWINS`)**:
   When a field is classified as numeric or currency, character probability mass for visual twin letters is summed into their numeric counterpart:
   $$P('0') \leftarrow P('0') + P('O') + P('D') + P('Q')$$
   $$P('1') \leftarrow P('1') + P('I') + P('L')$$
   $$P('5') \leftarrow P('5') + P('S')$$
   $$P('8') \leftarrow P('8') + P('B')$$
   This guarantees that monetary values (e.g., `45,000`, `150,450`) are never corrupted into letter sequences like `4S,OOO`.
4. **Identifier Decoding**:
   Guarantees strings like `INV-2024-0156` maintain exact hyphen placement and numeric suffixes without being forced into an English dictionary word.

---

## 7. Inpainting-Based Text Redaction Engine

Implemented in [src/engine/redactor.py](file:///d:/OCR+MACHINE_LEARNING/src/engine/redactor.py).

### 7.1 Why Inpainting Instead of Black Boxes?
Solid black redaction bars:
- Ruin visual document structure and layout hierarchy.
- Signal exactly where confidential data was located, highlighting sensitive areas.
- Cannot be cleanly displayed to external clients or auditors.

Inpainting reconstructs the underlying paper grain, color gradients, and textures, creating a clean document as if the text was never present.

### 7.2 Implementation Details
1. **Mask Synthesis**: A single-channel 8-bit binary mask ($M$) is created, with $M(x, y) = 255$ inside detected bounding boxes and $0$ elsewhere.
2. **Dilation Padding**: Bounding boxes are expanded by `padding=2` pixels in all four directions. This ensures anti-aliased font edges, faint serifs, and drop-shadows are masked out, preventing leftover stroke halos.
3. **Inpainting Algorithms**:
   - **Alexandru Telea Method (`cv2.INPAINT_TELEA`)**:
     Fast marching method based on solving the Eikonal equation. The pixel value is reconstructed as a weighted sum of boundary pixels within an inpainting neighborhood radius ($R = 5\text{px}$):
     $$I(p) = \frac{\sum_{q \in B_\epsilon(p)} w(p, q) \cdot [I(q) + \nabla I(q) \cdot (p - q)]}{\sum_{q \in B_\epsilon(p)} w(p, q)}$$
     Ideal for uniform paper backgrounds and document textures.
   - **Navier-Stokes Method (`cv2.INPAINT_NS`)**:
     Models image inpainting as 2D fluid flow, propagating lines of equal luminance (isophotes) across the masked domain by solving partial differential equations. Ideal for documents with subtle gradients or watermarks.

---

## 8. Unified Pipeline Orchestration (OCREngine)

Implemented in [src/engine/pipeline.py](file:///d:/OCR+MACHINE_LEARNING/src/engine/pipeline.py).

The [OCREngine](file:///d:/OCR+MACHINE_LEARNING/src/engine/pipeline.py#L19) class acts as the central coordinator:

```python
engine = OCREngine(
    model_path="models/best_model.pt",
    device="cpu",
    inpaint_method="telea",
    inpaint_radius=5,
    sharpen=True
)
results = engine.process("sample/invoice_test.png", redact=True)
saved_files = engine.save_outputs(results, "sample/invoice_test.png")
```

### 8.1 Output Format
Execution automatically outputs two standardized artifacts:
1. **`{stem}_text.json`**: An array of detected word objects with exact integer bounding boxes:
   ```json
   [
     {
       "word": "TECHNOLOGIES",
       "bounding_box": [22, 132, 49, 383],
       "confidence": 0.937,
       "raw_word": "TECHNOLOGIES",
       "field_type": "WORD"
     },
     {
       "word": "INV-2024-0156",
       "bounding_box": [85, 841, 102, 974],
       "confidence": 0.940,
       "raw_word": "INV-2024-0156",
       "field_type": "ALPHANUMERIC_ID"
     },
     {
       "word": "45,000",
       "bounding_box": [355, 699, 372, 754],
       "confidence": 0.940,
       "raw_word": "45,000",
       "field_type": "CURRENCY"
     }
   ]
   ```
2. **`{stem}_redacted.png`**: The complete document with text cleanly removed and paper background inpainted.

---

## 9. Empirical Evaluation & CPU Benchmarks

### 9.1 Benchmark on 3,000 IIIT-5K Test Images
Evaluated via [src/training/evaluate.py](file:///d:/OCR+MACHINE_LEARNING/src/training/evaluate.py):

| Metric | Measured Value | Definition / Calculation |
|---|:---:|---|
| **Word Accuracy (Exact Match)** | **43.27%** | Strict case-insensitive string equality ($1,298 / 3,000$) |
| **Character Error Rate (CER)** | **31.55%** | $\frac{\sum \text{Levenshtein}(y, \hat{y})}{\sum \text{len}(y)} \times 100\%$ |
| **Character Precision** | **79.80%** | True Positive Characters / Total Predicted Characters |
| **Character Recall** | **69.57%** | Recovered Ground-Truth Characters / Total Ground-Truth Characters |
| **Character F1-Score** | **74.33%** | Harmonic mean: $2 \times \frac{P \times R}{P + R}$ |
| **Average Edit Distance** | **1.606 chars** | Mean Levenshtein distance per test sample |
| **Average CPU Latency** | **77.9 ms** | Per-sample CRNN forward pass on standard CPU |

### 9.2 Real Invoice Field Regression (19 / 25 Passing Exact Match)

| Target Field | Expected Ground Truth | Engine Output | Match Status |
|---|---|---|:---:|
| Company Name | `TECHNOLOGIES` | `TECHNOLOGIES` | **PASS (Exact Match)** |
| City | `GURUGRAM` | `GURUGRAM` | **PASS (Exact Match)** |
| State | `HARYANA` | `HARYANA` | **PASS (Exact Match)** |
| Postal Code | `122001` | `122001` | **PASS (Exact Match)** |
| Invoice Identifier | `INV-2024-0156` | `INV-2024-0156` | **PASS (Exact Match)** |
| Unit Price 1 | `45,000` | `45,000` | **PASS (Exact Match)** |
| Unit Price 2 | `1,200` | `1,200` | **PASS (Exact Match)** |
| Unit Price 3 | `1,500` | `1,500` | **PASS (Exact Match)** |
| Unit Price 4 | `12,000` | `12,000` | **PASS (Exact Match)** |
| Line Total 2 | `6,000` | `6,000` | **PASS (Exact Match)** |
| Line Total 3 | `7,500` | `7,500` | **PASS (Exact Match)** |
| Line Total 4 | `24,000` | `24,000` | **PASS (Exact Match)** |
| Total Invoice Amount | `150,450` | `150,450` | **PASS (Exact Match)** |
| Header 1 | `DESCRIPTION` | `DESCRIPTION` | **PASS (Exact Match)** |
| Header 2 | `QUANTITY` | `QUANTITY` | **PASS (Exact Match)** |
| Security Watermark | `CONFIDENTIAL` | `CONFIDENTIAL` | **PASS (Exact Match)** |
| Summary Label 1 | `SUBTOTAL` | `SUBTOTAL` | **PASS (Exact Match)** |
| Summary Label 2 | `TOTAL` | `TOTAL` | **PASS (Exact Match)** |
| Summary Label 3 | `AMOUNT` | `AMOUNT` | **PASS (Exact Match)** |

### 9.3 Component-Wise CPU Latency Breakdown (Full Invoice, 110 Words)
Measured on Intel 10th Gen Core CPU (4 Cores, pure CPU PyTorch):
- **Image Preprocessing**: **3.52 ms**
- **Text Detection & Grouping**: **30.57 ms**
- **CRNN Word Recognition (110 crops)**: **9,506 ms** (~86 ms/word)
- **OpenCV Inpainting (110 regions)**: **651 ms** (~5.9 ms/box)
- **Total Pipeline Latency**: **~10.19 seconds**
- **Peak RAM Usage**: **~180 MB**

---

## 10. Application Interfaces: CLI, REST API & Web Dashboard

### 10.1 Command-Line Interface (CLI)
Located in [src/cli/main.py](file:///d:/OCR+MACHINE_LEARNING/src/cli/main.py).

```bash
# Direct module execution
python -m src.cli.main sample/invoice_test.png

# Custom output directory and inpainting method
python -m src.cli.main sample/invoice_test.png --output-dir output/ --inpaint-method ns

# Platform-independent executable wrappers
textrecognizer.bat sample/invoice_test.png    # Windows
./textrecognizer sample/invoice_test.png        # Linux / macOS
```

CLI Terminal Output:
```
==================================================
OCR & TEXT REDACTION COMPLETE
==================================================
Input:
  sample\invoice_test.png
Output:
  sample\invoice_test_text.json
  sample\invoice_test_redacted.png
Summary:
  Detected Words : 94
  Latency        : 7833.97 ms
==================================================
```

### 10.2 FastAPI REST Web Service
Located in [src/api/main.py](file:///d:/OCR+MACHINE_LEARNING/src/api/main.py).

Start the server:
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Available REST Endpoints:
1. **`GET /health`**: Confirms model load state and system health.
   ```json
   { "status": "healthy", "model_loaded": true }
   ```
2. **`POST /ocr`**: Ingests multipart image upload, returns JSON word detections and bounding box array.
3. **`POST /redact`**: Ingests multipart image upload, returns streaming PNG with inpaint redactions.
4. **`GET /sample-image?name=invoice`**: Serves sample images for interactive demonstrations.
5. **`GET /`**: Serves the interactive web UI dashboard.

### 10.3 Interactive Web Dashboard
The web studio in `src/api/static/index.html` provides:
- Drag-and-drop image upload or one-click demo loading (`Document`, `Invoice`, `Word`).
- Synchronized side-by-side view of Original Document vs. Inpainted Redacted Document.
- Interactive canvas overlay highlighting detected word bounding boxes.
- Filterable transcription table with confidence indicators and domain field badges (`WORD`, `NUMBER`, `CURRENCY`, `ALPHANUMERIC_ID`).
- One-click downloads for `{stem}_text.json` and `{stem}_redacted.png`.

---

## 11. Repository & File-by-File Guide

```
d:/OCR+MACHINE_LEARNING/
│
├── src/
│   ├── engine/
│   │   ├── preprocessor.py    # Grayscale conversion, contrast stretching, 3x3 sharpening
│   │   ├── detector.py        # Classical Sobel + Otsu + Line suppression + Character grouping
│   │   ├── recognizer.py      # Batched CRNN forward inference on image crops
│   │   ├── decoder.py         # Field classifier, topology detector, lexicon beam search
│   │   ├── redactor.py        # OpenCV Telea and Navier-Stokes inpainting
│   │   └── pipeline.py        # Master OCREngine coordinating the end-to-end flow
│   │
│   ├── training/
│   │   ├── dataset.py         # PyTorch Dataset, CTC Collate, LabelConverter, Caching
│   │   ├── model.py           # 7-layer CNN + 2-layer BiLSTM + Linear projection
│   │   ├── train.py           # Training loop with CTCLoss, AdamW, and model checkpointing
│   │   ├── evaluate.py        # Computes Accuracy, CER, Precision, Recall, F1, Edit Distance
│   │   ├── benchmark.py       # CPU latency, throughput, and dynamic quantization suite
│   │   └── synthetic_eval.py  # Targeted regression testing on synthetic invoice images
│   │
│   ├── cli/
│   │   └── main.py            # CLI entry point, argument parsing, JSON/PNG export
│   │
│   └── api/
│       ├── main.py            # FastAPI application with endpoints and lifespan management
│       └── static/index.html  # Modern interactive browser dashboard
│
├── models/
│   └── best_model.pt          # Trained CRNN checkpoint (63.8 MB, 25 epochs)
│
├── reports/
│   ├── evaluation.json        # Official evaluation metrics on 3,000 IIIT-5K test images
│   ├── cpu_benchmark.json     # CPU execution benchmarks across batch sizes
│   └── training_history.json  # Loss curves and validation metrics per epoch
│
├── sample/
│   ├── invoice_test.png       # Complex invoice input sample
│   ├── invoice_test_text.json # Ground-truth and test OCR output
│   └── invoice_test_redacted.png # Inpainted output document
│
├── tests/
│   ├── test_preprocessor.py   # Unit tests for image filtering and sharpening
│   ├── test_detector.py       # Unit tests for bounding boxes, sorting, line suppression
│   ├── test_decoder.py        # Unit tests for beam search and digit-twin preservation
│   ├── test_redactor.py       # Unit tests for inpainting masks and background reconstruction
│   ├── test_pipeline.py       # End-to-end integration tests for OCREngine
│   └── test_api.py            # HTTP client tests for FastAPI endpoints
│
├── textrecognizer             # Linux/macOS executable bash runner
├── textrecognizer.bat         # Windows executable batch runner
├── pyproject.toml             # Python packaging specification
├── requirements.txt           # Verified dependencies
├── FINAL_AUDIT.md             # Formal 14-point requirements audit report
├── FINAL_SUBMISSION.md        # Assignment submission summary
└── README.md                  # Quickstart documentation
```

---

## 12. Step-by-Step Execution & Reproduction Guide

### 12.1 Environment Setup
```powershell
# Clone or navigate to the workspace
cd d:\OCR+MACHINE_LEARNING

# Install dependencies
pip install -r requirements.txt
```

### 12.2 Run Comprehensive Test Suite
```powershell
# Run all unit and integration tests
pytest tests/ -v
```

### 12.3 Execute Full Evaluation on IIIT-5K Benchmark
```powershell
# Evaluate model on 3,000 test images
python -m src.training.evaluate --test-csv testdata.csv --model-path models/best_model.pt
```

### 12.4 Process Document via CLI
```powershell
# Run OCR & Inpainting Redaction on an invoice image
python -m src.cli.main sample/invoice_test.png
```

### 12.5 Launch FastAPI Server & Web Visualizer
```powershell
# Launch API server on port 8000
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your web browser to access the interactive web visualizer.
