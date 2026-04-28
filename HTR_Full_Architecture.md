# Handwritten Text Recognition (HTR) — Full Implementation Architecture

---

## System Overview

A two-stage pipeline that answers two questions:
- **WHERE** is the text? → Segmentation stage
- **WHAT** does the text say? → OCR/Recognition stage

```
Mobile Image → Preprocessing → Segmentation → OCR Model → Post-processing → Digital Text
```

---

## Stage 1 — Preprocessing

**Goal:** Clean and normalize the raw image before any text detection.

### Steps

| Step | Method | Tool |
|---|---|---|
| Resize | Normalize to fixed height (e.g. 1024px) | OpenCV |
| Grayscale | Convert RGB → single channel | OpenCV `cvtColor` |
| Binarization | Otsu's thresholding | OpenCV `threshold` |
| Deskew | Detect and correct rotation angle | OpenCV Hough lines |
| Noise removal | Median blur / morphological ops | OpenCV |

### Code Skeleton

```python
import cv2
import numpy as np

def preprocess(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, 3)
    return denoised
```

---

## Stage 2 — Text Segmentation (WHERE)

**Goal:** Locate and crop each word/line region from the preprocessed image.

### Sub-steps

#### 2a. Line Detection
- Compute **horizontal projection profile** (sum pixel values per row)
- Find valleys = line separators
- Extract horizontal strips (one per text line)

#### 2b. Word Segmentation
- For each line strip, compute **vertical projection profile**
- Find valleys = word separators
- Extract word bounding boxes

#### 2c. Bounding Box Crop
- Use `cv2.findContours` or `cv2.connectedComponentsWithStats`
- Add a small padding margin (e.g. 4–8px) around each crop
- Resize all crops to a fixed input size (e.g. 32×128px)

### Code Skeleton

```python
def segment_words(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    word_crops = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # filter noise
            crop = binary_image[y-4:y+h+4, x-4:x+w+4]
            crop_resized = cv2.resize(crop, (128, 32))
            word_crops.append((crop_resized, (x, y, w, h)))
    return sorted(word_crops, key=lambda c: (c[1][1], c[1][0]))  # sort top-left to bottom-right
```

---

## Stage 3 — OCR Model (WHAT)

**Goal:** Read each cropped word image and output a character string.

### Architecture: CRNN (CNN + BiLSTM + CTC)

```
Word Crop (32×128) → CNN Backbone → Sequence Model (BiLSTM) → CTC Decoder → Text String
```

#### 3a. CNN Backbone (Feature Extractor)
- Model: ResNet-18 or MobileNetV3 (lightweight for mobile)
- Input: 32×128 grayscale image
- Output: Feature map of shape (W', C) — a sequence of column features

#### 3b. Sequence Model (BiLSTM)
- Takes the feature sequence from CNN
- 2-layer Bidirectional LSTM (256 hidden units)
- Captures left-to-right and right-to-left context
- Output: Per-timestep probability distribution over characters

#### 3c. CTC Decoder
- Loss: Connectionist Temporal Classification (CTC)
- Decoding: Greedy or Beam Search
- Handles variable-length outputs without explicit segmentation of characters

### Model Code Skeleton

```python
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
        )
        # BiLSTM
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
        # Output projection
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                         # (B, C, H, W)
        feat = feat.squeeze(2).permute(0, 2, 1)    # (B, W, C)
        out, _ = self.rnn(feat)                    # (B, W, 512)
        return self.fc(out)                        # (B, W, num_classes)
```

---

## Training Pipeline

### Dataset
- **IAM Handwriting Database** (primary benchmark)
- Custom mobile-captured notes (real-world generalization)
- Synthetically generated handwriting (optional augmentation)

### Data Augmentation (to simulate real-world imperfect crops)

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=3),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomAffine(degrees=0, shear=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

### Loss & Optimizer

```python
criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

### Training Loop (simplified)

```python
for epoch in range(num_epochs):
    for images, labels, label_lengths in dataloader:
        outputs = model(images)                         # (B, T, C)
        log_probs = outputs.permute(1, 0, 2).log_softmax(2)
        input_lengths = torch.full((batch_size,), T, dtype=torch.long)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Stage 4 — Post-processing

**Goal:** Assemble raw OCR outputs into clean, structured text.

### Steps

| Step | Method | Tool |
|---|---|---|
| Word assembly | Concatenate crops in reading order | Custom sort logic |
| Line reconstruction | Group words by Y-coordinate proximity | NumPy clustering |
| Spell correction | Dictionary + context correction | `pyspellchecker` / LLM API |
| Punctuation restoration | Rule-based or seq2seq model | Optional |
| Output formatting | Build paragraph structure | String processing |

### Code Skeleton

```python
from spellchecker import SpellChecker

spell = SpellChecker()

def post_process(word_list):
    corrected = []
    for word in word_list:
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return ' '.join(corrected)
```

---

## Output Layer

The final output can be delivered in multiple formats:

- `.txt` — plain digital text
- `.json` — structured with bounding box metadata per word
- `.pdf` — searchable PDF overlay on original image

```json
{
  "lines": [
    {
      "line_id": 1,
      "text": "This is a handwritten note",
      "words": [
        {"word": "This", "bbox": [50, 20, 80, 40]},
        {"word": "is", "bbox": [140, 20, 60, 40]}
      ]
    }
  ]
}
```

---

## Serving Architecture (API)

```
Client (Mobile App / Web) → FastAPI endpoint → Preprocessing → Segmentation → CRNN Inference → Post-processing → JSON Response
```

### FastAPI Endpoint Skeleton

```python
from fastapi import FastAPI, UploadFile
import numpy as np
import cv2

app = FastAPI()

@app.post("/ocr")
async def run_ocr(file: UploadFile):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    preprocessed = preprocess(img)
    crops = segment_words(preprocessed)
    results = [model_inference(crop) for crop, _ in crops]
    text = post_process(results)
    return {"text": text}
```

---

## Tech Stack Summary

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Image processing | OpenCV |
| Deep learning | PyTorch |
| OCR model | CRNN (CNN + BiLSTM + CTC) |
| Training data | IAM dataset + custom data |
| Spell correction | pyspellchecker |
| API serving | FastAPI |
| Deployment | Docker + uvicorn |

---

## Future Upgrade Path

| Current | Upgrade | Benefit |
|---|---|---|
| Classical segmentation | Deep learning (CRAFT / DBNet) | Better on cursive, overlapping text |
| CRNN | TrOCR / Donut (Transformer) | Higher accuracy, handles context |
| Rule-based post-processing | LLM-based correction | Grammar-aware, contextual fixes |
| Word-level segmentation | Layout detection (LayoutLMv3) | Paragraph and document structure |

---

## Project Key Insight

> **Segmentation** answers *WHERE* the text is.  
> **The model** answers *WHAT* the text says.  
> These concerns are intentionally separated for modularity and independent upgradeability.
