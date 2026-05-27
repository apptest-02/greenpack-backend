"""
Greenpack Pro — Image Pre-processing Service
PDF rasterization, deskew, denoise, DPI normalization
"""
import cv2
import numpy as np
from pathlib import Path
import logging
import tempfile
import shutil

log = logging.getLogger(__name__)


def rasterize_pdf(pdf_path: Path, dpi: int = 300) -> Path:
    """
    Convert PDF first page to PNG at specified DPI.
    Uses pdf2image (Poppler) with pypdfium2 as fallback.
    Returns path to generated PNG.
    """
    out_path = pdf_path.parent / f"{pdf_path.stem}_rasterized.png"

    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=1,
            last_page=1,
            fmt="png",
        )
        if images:
            images[0].save(str(out_path), "PNG")
            log.info(f"Rasterized PDF via pdf2image: {out_path}")
            return out_path
    except Exception as e:
        log.warning(f"pdf2image failed ({e}), trying pypdfium2")

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        page = pdf[0]
        scale = dpi / 72  # 72 = default PDF DPI
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        pil_image.save(str(out_path), "PNG")
        pdf.close()
        log.info(f"Rasterized PDF via pypdfium2: {out_path}")
        return out_path
    except Exception as e:
        raise RuntimeError(f"Cannot rasterize PDF {pdf_path}: {e}")


def preprocess_image(image_path: Path) -> Path:
    """
    Preprocess a scanned image:
    1. Deskew (correct rotation)
    2. Denoise
    3. Normalize to expected DPI range
    Returns path to preprocessed image (saved alongside original)
    """
    out_path = image_path.parent / f"{image_path.stem}_preprocessed.png"

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Convert to grayscale for processing operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Deskew ─────────────────────────────────────────────────────────────────
    img = _deskew(img, gray)

    # ── Denoise (light — preserve label detail) ────────────────────────────────
    img = cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

    # ── Save preprocessed result ───────────────────────────────────────────────
    cv2.imwrite(str(out_path), img)
    log.info(f"Preprocessed image: {out_path} ({img.shape[1]}×{img.shape[0]})")
    return out_path


def _deskew(img: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Detect and correct document skew using Hough line transform"""
    try:
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return img

        # Calculate median angle of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:  # Only near-horizontal lines
                    angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)

        # Only correct if skew is significant (> 0.3 degrees)
        if abs(median_angle) < 0.3:
            return img

        # Rotate image to correct skew
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(
            img, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        log.debug(f"Deskewed by {median_angle:.2f} degrees")
        return corrected

    except Exception as e:
        log.warning(f"Deskew failed: {e}, returning original")
        return img


def normalize_resolution(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize image to match target dimensions for comparison"""
    h, w = img.shape[:2]
    if w == target_width and h == target_height:
        return img
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
