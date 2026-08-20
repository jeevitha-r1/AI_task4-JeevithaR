"""
preprocessing.py
-----------------
Shared image pre-processing utilities used by both the OCR pipeline
and the object-detection pipeline.

Pipeline stages:
    1. Grayscale conversion   -> collapses the 3-channel image to 1 channel
    2. Gaussian blur          -> suppresses sensor / compression noise
    3. Adaptive thresholding  -> binarizes the image so text/edges pop
    4. Deskew (OCR only)      -> straightens rotated text using the
                                  minimum-area bounding rectangle of
                                  all "ink" pixels
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply a Gaussian blur to smooth out micro-noise before thresholding."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Binarize the image using Otsu's method (auto-computed global threshold)
    combined with a local adaptive fallback for uneven lighting.
    """
    # Otsu global threshold — good for fairly even lighting
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive local threshold — good for shadows / uneven lighting
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # Combine: trust Otsu unless the local window disagrees strongly
    combined = cv2.bitwise_and(otsu, adaptive)
    return combined


def deskew(binary_img: np.ndarray) -> np.ndarray:
    """
    Detect the dominant rotation angle of foreground pixels and rotate
    the image back to a horizontal baseline. Falls back to the original
    image if there isn't enough foreground signal to compute an angle.
    """
    coords = np.column_stack(np.where(binary_img < 255))
    if coords.shape[0] < 20:
        return binary_img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return binary_img

    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Full OCR pre-processing chain: grayscale -> blur -> threshold -> deskew."""
    gray = to_grayscale(image)
    blurred = denoise(gray, kernel_size=3)
    binary = adaptive_threshold(blurred)
    straightened = deskew(binary)
    return straightened


def build_blob(image: np.ndarray, size=(300, 300), scale=1.0 / 127.5,
                mean=(127.5, 127.5, 127.5)) -> np.ndarray:
    """
    Construct a 4D blob for a cv2.dnn network (used by the object-detection
    pipeline): resizes, scales, and mean-subtracts the image.
    """
    return cv2.dnn.blobFromImage(image, scale, size, mean, swapRB=True, crop=False)
