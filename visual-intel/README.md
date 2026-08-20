# Visual Intelligence Pipeline — OCR & Object Detection

A small computer-vision toolkit that ingests raw images and extracts
machine-readable information from them, using two independent pre-trained
recognition paths:

- **Path 1 — OCR**: `pytesseract` (Tesseract engine) extracts text strings
  from documents/scans.
- **Path 2 — Object Detection**: a pre-trained MobileNet-SSD network
  (via OpenCV's `cv2.dnn` module) locates and labels objects with
  bounding boxes.

Both paths share a common pre-processing stage (`src/preprocessing.py`)
and both filter their output by a confidence threshold (default **80%**)
before drawing anything — low-confidence guesses are logged but not
rendered, so the visual output only shows what the model is actually
sure about.

## Project structure

```
visual-intel/
├── src/
│   ├── preprocessing.py             # grayscale, blur, adaptive threshold, deskew, blob builder
│   ├── ocr_pipeline.py              # Path 1: OCR
│   └── object_detection_pipeline.py # Path 2: Object detection
├── samples/
│   └── sample_invoice.png           # synthetic test image (generated, included)
├── outputs/                         # annotated images + JSON reports land here
├── models/                          # place MobileNet-SSD weights here (see below)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need the Tesseract OCR binary installed system-wide (pytesseract
is just a Python wrapper around it):

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
```

## Running the OCR pipeline (Path 1)

Already tested and working against the included sample image:

```bash
cd src
python3 ocr_pipeline.py --image ../samples/sample_invoice.png --psm 6
```

`--psm` (Page Segmentation Mode) controls how Tesseract interprets layout:
- `3` — fully automatic, mixed layouts
- `6` — a single uniform block of text (best for the sample invoice)
- `7` — a single text line (plates, headers)
- `11` — sparse/scattered text

Outputs land in `outputs/`:
- `<name>_ocr_annotated.png` — original image with green boxes + text over 80% confidence
- `<name>_ocr_preprocessed.png` — the binarized/deskewed intermediate image
- `<name>_ocr_report.json` — every detection with its confidence score, split into accepted/rejected

## Running the object detection pipeline (Path 2)

This path needs two pre-trained model files that are too large to embed
here — download them once and drop them into `models/`:

- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`

Both are freely available from the standard `chuanqi305/MobileNet-SSD`
GitHub repository (search that repo name — it's the canonical source
most OpenCV tutorials reference). Once placed in `models/`:

```bash
cd src
python3 object_detection_pipeline.py --image ../samples/your_photo.jpg
```

Outputs land in `outputs/`:
- `<name>_detections_annotated.png` — labeled bounding boxes over 80% confidence
- `<name>_detections_report.json` — every detection (label, confidence, box) accepted/rejected

## How it satisfies each requirement

| Requirement | Where |
|---|---|
| Use a pre-trained model / library | `pytesseract` (Tesseract engine) and MobileNet-SSD via `cv2.dnn` |
| Pre-processing integrity | `preprocessing.py`: grayscale → Gaussian blur → Otsu + adaptive threshold → deskew |
| Perform recognition on sample input | `ocr_pipeline.py` / `object_detection_pipeline.py` |
| Confidence-based accuracy filtering (≥80%) | Both pipelines split results into `accepted` (≥80%) vs `rejected` and only draw/report accepted ones prominently |
| Clear visual output | Annotated PNGs with bounding boxes + labels, plus structured JSON reports |

## Notes on adapting this for your own submission

I built and tested the OCR path against the included synthetic sample —
you saw the actual output. The object-detection path is fully written
and correct but needs the model weight files downloaded on your machine
(I don't have network access in this sandbox to fetch them for you).
Before you submit this anywhere, run it yourself, read through
`preprocessing.py` and both pipeline files, and make sure you can explain
what each stage does — that's the actual skill being assessed here, and
you'll want to own it if anyone asks you about it.
