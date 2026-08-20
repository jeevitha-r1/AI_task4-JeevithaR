"""
ocr_pipeline.py
----------------
Path 1: Optical Character Recognition.

Ingests a raw image, runs it through pre-processing, extracts text with
Tesseract (via pytesseract), filters detections by confidence, and produces
a clean annotated visual output plus a machine-readable JSON report.

Usage:
    python3 ocr_pipeline.py --image samples/sample_invoice.png --psm 6
"""

import argparse
import json
import os

import cv2
import pytesseract
from pytesseract import Output

from preprocessing import preprocess_for_ocr

CONFIDENCE_THRESHOLD = 80.0  # percent


def run_ocr(image_path: str, psm: int = 6, conf_threshold: float = CONFIDENCE_THRESHOLD):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")

    processed = preprocess_for_ocr(original)

    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(processed, config=config, output_type=Output.DICT)

    annotated = original.copy()
    accepted, rejected = [], []

    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text:
            continue

        record = {"text": text, "confidence": round(conf, 2)}

        if conf >= conf_threshold:
            accepted.append(record)
            x, y, w, h = (data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i])
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            label = f"{text} ({conf:.0f}%)"
            cv2.putText(annotated, label, (x, max(y - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        else:
            rejected.append(record)

    full_text = " ".join(r["text"] for r in accepted)
    return annotated, processed, accepted, rejected, full_text


def main():
    parser = argparse.ArgumentParser(description="OCR recognition pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--psm", type=int, default=6,
                         help="Tesseract Page Segmentation Mode (3, 6, 7, 11...)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                         help="Minimum confidence percentage to accept a detection")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    annotated, processed, accepted, rejected, full_text = run_ocr(
        args.image, psm=args.psm, conf_threshold=args.conf
    )

    base = os.path.splitext(os.path.basename(args.image))[0]
    annotated_path = os.path.join(args.outdir, f"{base}_ocr_annotated.png")
    processed_path = os.path.join(args.outdir, f"{base}_ocr_preprocessed.png")
    report_path = os.path.join(args.outdir, f"{base}_ocr_report.json")

    cv2.imwrite(annotated_path, annotated)
    cv2.imwrite(processed_path, processed)

    report = {
        "source_image": args.image,
        "psm_mode": args.psm,
        "confidence_threshold": args.conf,
        "accepted_detections": accepted,
        "rejected_detections": rejected,
        "full_text": full_text,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("OCR PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Accepted (>= {args.conf}% confidence): {len(accepted)} strings")
    print(f"Rejected (below threshold):            {len(rejected)} strings")
    print("-" * 60)
    print("Extracted text:")
    print(full_text if full_text else "(no text passed the confidence filter)")
    print("-" * 60)
    print(f"Annotated image : {annotated_path}")
    print(f"Preprocessed img: {processed_path}")
    print(f"JSON report      : {report_path}")


if __name__ == "__main__":
    main()
