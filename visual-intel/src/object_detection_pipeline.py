"""
object_detection_pipeline.py
------------------------------
Path 2: Object Detection with a pre-trained MobileNet-SSD (Caffe) model.

Ingests a raw image, builds a 4D blob (mean-subtracted, resized to
300x300), runs a forward pass through the network, filters detections
by confidence, scales normalized coordinates back to pixel space, and
draws labeled bounding boxes.

Requires two model files (not included — see README for download links,
since they're large binary weights):
    models/MobileNetSSD_deploy.prototxt
    models/MobileNetSSD_deploy.caffemodel

Usage:
    python3 object_detection_pipeline.py --image samples/sample_scene.jpg
"""

import argparse
import json
import os

import cv2
import numpy as np

from preprocessing import build_blob

CONFIDENCE_THRESHOLD = 0.80

# Class labels the standard MobileNet-SSD (VOC-trained) model was trained on
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")


def load_network(prototxt_path: str, model_path: str):
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model files not found. Download MobileNetSSD_deploy.prototxt and "
            "MobileNetSSD_deploy.caffemodel into the models/ folder "
            "(see README.md for links)."
        )
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


def run_detection(image_path: str, net, conf_threshold: float = CONFIDENCE_THRESHOLD):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")

    (h, w) = image.shape[:2]
    blob = build_blob(image)

    net.setInput(blob)
    detections = net.forward()

    annotated = image.copy()
    accepted, rejected = [], []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        label_name = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)

        # Normalized coordinates -> pixel space
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        record = {
            "label": label_name,
            "confidence": round(confidence, 4),
            "box_xywh": [int(startX), int(startY), int(endX - startX), int(endY - startY)],
        }

        if confidence >= conf_threshold:
            accepted.append(record)
            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)
            text = f"{label_name}: {confidence * 100:.1f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 15
            cv2.putText(annotated, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            rejected.append(record)

    return annotated, accepted, rejected


def main():
    parser = argparse.ArgumentParser(description="Object detection pipeline (MobileNet-SSD)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prototxt", default="models/MobileNetSSD_deploy.prototxt")
    parser.add_argument("--model", default="models/MobileNetSSD_deploy.caffemodel")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                         help="Minimum confidence (0-1) to accept a detection")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    net = load_network(args.prototxt, args.model)
    annotated, accepted, rejected = run_detection(args.image, net, conf_threshold=args.conf)

    base = os.path.splitext(os.path.basename(args.image))[0]
    annotated_path = os.path.join(args.outdir, f"{base}_detections_annotated.png")
    report_path = os.path.join(args.outdir, f"{base}_detections_report.json")

    cv2.imwrite(annotated_path, annotated)
    with open(report_path, "w") as f:
        json.dump({
            "source_image": args.image,
            "confidence_threshold": args.conf,
            "accepted_detections": accepted,
            "rejected_detections": rejected,
        }, f, indent=2)

    print("=" * 60)
    print("OBJECT DETECTION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Accepted (>= {args.conf * 100:.0f}% confidence): {len(accepted)} objects")
    for r in accepted:
        print(f"  - {r['label']} @ {r['confidence'] * 100:.1f}% box={r['box_xywh']}")
    print(f"Rejected (below threshold): {len(rejected)} objects")
    print("-" * 60)
    print(f"Annotated image: {annotated_path}")
    print(f"JSON report     : {report_path}")


if __name__ == "__main__":
    main()
