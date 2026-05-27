"""
Greenpack Pro — OCR Service
Dual OCR engine: EasyOCR (primary) + Tesseract (secondary)
Singleton pattern to load model once and reuse.
"""
import difflib
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_ocr_reader():
    """
    Load EasyOCR model ONCE. Cached singleton — never create multiple instances.
    First load: ~8-12 seconds, ~2GB RAM. Subsequent calls: instant.
    """
    log.info("Loading EasyOCR model (first time — this takes ~10 seconds)...")
    import easyocr
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=settings.easyocr_model_dir,
        download_enabled=settings.easyocr_download_enabled,
        verbose=False,
    )
    log.info("EasyOCR model loaded successfully")
    return reader


def _configure_tesseract():
    """Set Tesseract executable path"""
    try:
        import pytesseract
        tesseract_path = Path(settings.tesseract_path)
        if tesseract_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
            log.debug(f"Tesseract configured: {tesseract_path}")
        return pytesseract
    except ImportError:
        log.warning("pytesseract not available")
        return None


def run_dual_ocr(master_path: str, scan_path: str) -> dict:
    """
    Run EasyOCR on both master and scan images.
    Returns dict with extracted text regions from both images.
    """
    try:
        reader = get_ocr_reader()
    except Exception as e:
        log.error(f"Cannot load OCR model: {e}")
        return {"master_regions": [], "scan_regions": [], "errors": [], "engine": "none"}

    min_conf = settings.ocr_min_confidence

    def extract_regions(image_path: str) -> list[dict]:
        try:
            results = reader.readtext(
                image_path,
                detail=1,
                paragraph=False,
                batch_size=4,
            )
            regions = []
            for bbox, text, conf in results:
                if conf >= min_conf and text.strip():
                    # Normalize bbox to dict format
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    regions.append({
                        "text": text.strip(),
                        "confidence": round(float(conf), 3),
                        "bbox": {
                            "x": int(min(x_coords)),
                            "y": int(min(y_coords)),
                            "w": int(max(x_coords) - min(x_coords)),
                            "h": int(max(y_coords) - min(y_coords)),
                        },
                    })
            return regions
        except Exception as e:
            log.error(f"EasyOCR extraction failed for {image_path}: {e}")
            return []

    master_regions = extract_regions(master_path)
    scan_regions = extract_regions(scan_path)

    log.info(
        f"OCR complete: master={len(master_regions)} regions, "
        f"scan={len(scan_regions)} regions"
    )

    return {
        "master_regions": master_regions,
        "scan_regions": scan_regions,
        "errors": [],
        "timeout": False,
        "engine": "easyocr",
    }


def diff_text_regions(
    master_regions: list[dict],
    scan_regions: list[dict],
) -> list[dict]:
    """
    Compare text regions between master and scan using character-level diff.
    Returns list of text errors with location, type, and content.
    """
    errors = []

    for m_region in master_regions:
        # Find closest scan region by spatial proximity
        closest = _find_closest_region(m_region["bbox"], scan_regions)
        if closest is None:
            # Region in master but missing in scan
            errors.append({
                "type": "DELETE",
                "region_bbox": m_region["bbox"],
                "master_text": m_region["text"],
                "scan_text": "",
                "description": f"Text missing in scan: '{m_region['text']}'",
                "severity": "high",
            })
            continue

        master_text = m_region["text"]
        scan_text = closest["text"]

        if master_text == scan_text:
            continue  # Perfect match

        # Character-level diff
        matcher = difflib.SequenceMatcher(None, master_text, scan_text, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            master_substr = master_text[i1:i2]
            scan_substr = scan_text[j1:j2]

            severity = _assess_severity(tag, master_substr, scan_substr)

            errors.append({
                "type": tag.upper(),
                "region_bbox": m_region["bbox"],
                "master_text": master_text,
                "scan_text": scan_text,
                "master_char": master_substr,
                "scan_char": scan_substr,
                "description": _describe_error(tag, master_substr, scan_substr),
                "severity": severity,
            })

    return errors


def _find_closest_region(
    master_bbox: dict,
    scan_regions: list[dict],
    max_distance: float = 150.0,
) -> Optional[dict]:
    """Find the scan region closest to the given master bbox"""
    if not scan_regions:
        return None

    m_cx = master_bbox["x"] + master_bbox["w"] / 2
    m_cy = master_bbox["y"] + master_bbox["h"] / 2

    closest = None
    min_dist = float("inf")

    for region in scan_regions:
        s_cx = region["bbox"]["x"] + region["bbox"]["w"] / 2
        s_cy = region["bbox"]["y"] + region["bbox"]["h"] / 2
        dist = ((m_cx - s_cx) ** 2 + (m_cy - s_cy) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = region

    return closest if min_dist <= max_distance else None


def _assess_severity(tag: str, master_char: str, scan_char: str) -> str:
    """Assess error severity: high / medium / low"""
    if tag == "replace":
        # Digit/number errors are always high severity
        if any(c.isdigit() for c in master_char + scan_char):
            return "high"
        return "medium"
    elif tag == "delete":
        return "high"
    elif tag == "insert":
        return "low"
    return "medium"


def _describe_error(tag: str, master_char: str, scan_char: str) -> str:
    """Human-readable error description"""
    if tag == "replace":
        return f"Character mismatch: '{master_char}' printed as '{scan_char}'"
    elif tag == "delete":
        return f"Missing text: '{master_char}'"
    elif tag == "insert":
        return f"Extra text: '{scan_char}'"
    return f"Text difference: master='{master_char}' scan='{scan_char}'"
