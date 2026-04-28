<p align="center">
  <h1 align="center">✍️ Handwritten Text Recognition (HTR)</h1>
  <p align="center">
    A deep learning pipeline that transcribes full pages of handwriting into digital text using a CRNN architecture with CTC decoding.
  </p>
  <p align="center">
    <a href="#quickstart"><strong>Quickstart</strong></a> · <a href="#architecture"><strong>Architecture</strong></a> · <a href="#training"><strong>Training</strong></a> · <a href="#fine-tuning"><strong>Fine-tuning</strong></a> · <a href="#results"><strong>Results</strong></a>
  </p>
</p>

---

## Overview

This project implements an end-to-end **Handwritten Text Recognition** system that takes a full page image of handwritten text and outputs structured digital text. The pipeline handles everything from preprocessing and line segmentation to character-level recognition, with support for fine-tuning on custom handwriting styles.

### Key Features

- 🏗️ **CRNN Architecture** — 7-layer CNN backbone + 2-layer BiLSTM + CTC decoder
- 📄 **Full Page Processing** — Automatic header detection, line segmentation, and paragraph grouping
- 🎯 **12.74% CER** on the CVL test set (best model)
- 🔧 **Fine-tuning Pipeline** — Adapt the model to new handwriting styles (phone photos, screenshots, etc.)
- 📊 **Dual Output** — Plain text (`.txt`) and structured JSON with bounding box metadata
- ⚡ **Mixed Precision Training** — AMP + GradScaler for fast GPU training
- 🔄 **11 Data Augmentations** — Rotation, perspective warp, blur, noise, JPEG compression, and more

---

## Quickstart

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/<your-username>/handwritten-text-recognition.git
cd handwritten-text-recognition

pip install -r requirements.txt
```

**Dependencies:**
| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Deep learning framework |
| `opencv-python` | Image preprocessing & segmentation |
| `Pillow` | Image loading & augmentations |
| `numpy` | Numerical operations |
| `scipy` | Signal processing (projection smoothing) |
| `Levenshtein` | CER/WER metric computation |

### Inference

Run the model on a handwritten page image:

```bash
python inference.py --image path/to/handwritten_page.png
```

Use a fine-tuned checkpoint:

```bash
python inference.py --image path/to/photo.png --checkpoint checkpoints/best_finetuned.pth
```

**Output files:**
- `output.txt` — Plain text transcription
- `output.json` — Structured output with paragraph/line grouping and bounding boxes

---

## Architecture

```
Full Page Image
    │
    ├─ Preprocessing ──────── Grayscale → Adaptive Threshold → Denoise → Upscale (if low-res)
    │
    ├─ Header Detection ───── Horizontal projection → Find printed header → Crop handwritten region
    │
    ├─ Line Segmentation ──── Smoothed horizontal projection → Valley detection → Line extraction
    │
    ├─ Paragraph Detection ── Vertical gap analysis → Group lines by proximity
    │
    └─ Recognition (CRNN) ─── For each line:
         │
         ├─ CNN Backbone (7 layers)
         │   Conv(1→64) → Conv(64→128) → Conv(128→256) → Conv(256→256)
         │   → Conv(256→512) → Conv(512→512) → Conv(512→512)
         │   with BatchNorm, ReLU, and strategic MaxPooling
         │
         ├─ Adaptive Pooling ── Collapse height to 1 → (B, 512, W')
         │
         ├─ BiLSTM (2 layers) ── 256 hidden units each direction → 512-dim output
         │
         └─ CTC Decoder ── Greedy decoding with blank collapse
```

### Model Summary

| Component | Details |
|---|---|
| Input size | `1 × 64 × 1024` (grayscale, height-normalized, width-padded) |
| CNN | 7 conv layers, 512 output channels, BatchNorm at layers 3, 5, 7 |
| Pooling | `AdaptiveAvgPool2d((1, None))` to collapse spatial height |
| RNN | 2 × BiLSTM, 256 hidden units → 512-dim per timestep |
| Dropout | 0.3 between LSTM layers |
| Output | `Linear(512, num_classes)` with CTC blank at index 0 |

---

## Dataset

The model is trained on the [CVL Handwriting Database](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/) (version 1.1).

### Data Organization

```
cvl-database-1-1/
├── trainset/
│   ├── words/          # Individual word crops with labels encoded in filenames
│   │   └── <writerID>/
│   │       └── <writerID>-<pageID>-<lineID>-<wordIdx>-<label>.tif
│   └── lines/          # Full line images
│       └── <writerID>/
│           └── <writerID>-<pageID>-<lineID>.tif
└── testset/
    ├── words/
    └── lines/
```

### Label Reconstruction

Word-level labels are reconstructed into line-level labels by:
1. Parsing filenames to extract `(writer, page, line, wordIdx, label)`
2. Sorting words by `wordIdx` within each line
3. Joining with spaces to form the full line text
4. Matching to the corresponding line image

> **Note:** German pages (page ID `6`) are automatically filtered out — only English text is used for training.

---

## Training

### Train from Scratch

```bash
python train.py --data_dir cvl-database-1-1 --epochs 100 --batch_size 16
```

### Resume Training

```bash
python train.py --data_dir cvl-database-1-1 --epochs 100 --resume
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss | CTCLoss (blank=0, zero_infinity=True) |
| Gradient clipping | Max norm 5.0 |
| Mixed precision | AMP with GradScaler |
| Early stopping | 15 epochs without improvement |
| Train/Val split | 90/10 by writer (prevents data leakage) |

### Data Augmentations (11 types)

Applied with 5× multiplier during training:

| Augmentation | Probability | Purpose |
|---|---|---|
| Rotation (±3°) | 50% | Slight skew variation |
| Affine shear (±3°) | 50% | Writing slant |
| Resolution degradation (30–70%) | 40% | Low-res phone captures |
| Perspective warp | 30% | Angled phone photos |
| Gaussian blur (σ 0.3–2.0) | 40% | Defocus simulation |
| Brightness (0.6–1.4×) | 50% | Lighting variation |
| Contrast (0.5–1.5×) | 50% | Dynamic range |
| Uneven lighting gradient | 30% | Phone flash / shadows |
| Gaussian noise (σ 3–15) | 30% | Sensor noise |
| JPEG compression (q15–50) | 30% | Compression artifacts |
| Erosion / Dilation | 20% | Pen thickness variation |

---

## Fine-tuning

Adapt the pre-trained model to new handwriting styles or image conditions (phone photos, screenshots, etc.).

### Step 1: Prepare Fine-tuning Data

**Option A — Auto-degraded CVL crops** (no manual work):
```bash
python prepare_finetune.py --auto --data_dir cvl-database-1-1
```
Generates degraded line crops at multiple resolution scales (15%–35%) with JPEG artifacts to simulate real-world capture conditions.

**Option B — Phone/screenshot photos** (manual labeling required):
```bash
# Place your photos in finetune_data/raw_photos/
python prepare_finetune.py --phone
```
Extracts individual lines from photos → edit `finetune_data/phone_lines/labels.txt` to add ground truth text.

### Step 2: Run Fine-tuning

```bash
python finetune.py --epochs 20 --lr 1e-4 --checkpoint checkpoints/best_model.pth
```

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 20 | Fine-tuning epochs |
| `--lr` | 1e-4 | Learning rate (lower than initial training) |
| `--num_augments` | 3 | Augmentation multiplier |
| `--checkpoint` | `checkpoints/best_model.pth` | Base model to fine-tune |

The fine-tuned model is saved to `checkpoints/best_finetuned.pth`.

---

## Evaluation

Run evaluation on the CVL test set:

```bash
python test.py --checkpoint checkpoints/best_model.pth --data_dir cvl-database-1-1
```

Produces per-line predictions vs. ground truth in `test_results.txt`.

---

## Results

### Test Set Performance

| Metric | Score |
|---|---|
| **Character Error Rate (CER)** | **15.33%** |
| **Word Error Rate (WER)** | **29.06%** |

### Example Predictions

| Ground Truth | Prediction | CER |
|---|---|---|
| `Mailüfterl is an Austrian nickname for the first` | `Mailüfterl is an Austrian nicknam for the first` | 2.1% |
| `Imagine a vast sheet of paper on which straight Lines` | `Imagine a vast sheet of paper on which straight Lines` | 0.0% |
| `Disdaining fortune with his brandish'd steel` | `Disdaining fortune with his brandish'd steel` | 0.0% |
| `variety in a state of nature` | `variety in a stete of nature` | 3.6% |

---

## Project Structure

```
.
├── model.py                # CRNN architecture (CNN + BiLSTM + Linear)
├── dataset.py              # CVL dataset loader, charset builder, augmentations
├── train.py                # Training loop with validation, mixed precision, checkpointing
├── test.py                 # Test set evaluation with per-line CER/WER
├── inference.py            # Full-page inference pipeline (preprocess → segment → recognize)
├── finetune.py             # Fine-tuning on degraded/phone data
├── prepare_finetune.py     # Data preparation for fine-tuning (auto-degraded + phone lines)
├── requirements.txt        # Python dependencies
├── HTR_Full_Architecture.md # Detailed architecture documentation
├── checkpoints/            # Model checkpoints
│   ├── best_model.pth          # Best base model (by val CER)
│   ├── best_finetuned.pth      # Best fine-tuned model
│   └── last_checkpoint.pth     # Latest checkpoint (for resuming)
├── cvl-database-1-1/       # CVL Handwriting Database
└── finetune_data/          # Fine-tuning data
    ├── auto_degraded/          # Degraded CVL crops + labels.txt
    ├── phone_lines/            # Extracted phone photo lines + labels.txt
    └── raw_photos/             # Raw phone/screenshot photos (input)
```

---

## Output Format

### Plain Text (`output.txt`)

Paragraphs separated by blank lines:

```
Imagine a vast sheet of paper on which straight Lines
Triangles Squares Pentagons Hexagons and other figures

And fortune on his damned quarrel smiling
Show'd like a rebel's whore but all's too weak
```

### Structured JSON (`output.json`)

```json
{
  "paragraphs": [
    {
      "paragraph_id": 1,
      "lines": [
        {
          "line_id": 1,
          "text": "Imagine a vast sheet of paper on which straight Lines",
          "bbox": [0, 120, 2480, 195]
        }
      ],
      "full_text": "Imagine a vast sheet of paper..."
    }
  ]
}
```

---

## License

This project is for research and educational purposes. The CVL Handwriting Database is subject to its own [license terms](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/).

---

## Acknowledgments

- **CVL Handwriting Database** — Vienna University of Technology
- **CRNN Architecture** — Shi, Bai, Yao (2015): *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*
- **CTC Loss** — Graves et al. (2006): *Connectionist Temporal Classification*
