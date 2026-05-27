"""
Greenpack Pro — Annotated Image Generator
Creates side-by-side comparison with colored error overlays
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# Annotation colors (BGR for OpenCV)
COLOR_OCR = (0, 0, 220)       # Red for OCR errors
COLOR_COLOR = (220, 120, 0)   # Orange for color zones
COLOR_DEFECT = (0, 165, 255)  # Amber for defects
COLOR_BARCODE_OK = (50, 180, 50)   # Green for valid barcodes
COLOR_BARCODE_FAIL = (0, 0, 220)  # Red for failed barcodes
COLOR_LABEL = (255, 255, 255)      # White for labels


def create_annotated_image(
    master: np.ndarray,
    aligned_scan: np.ndarray,
    text_errors: list,
    color_zones: list,
    defects: list,
    barcode_results: list,
    job_id: str,
) -> Optional[Path]:
    """
    Create side-by-side annotated comparison image.
    Left: master with expected zones. Right: scan with error overlays.
    Returns path to saved PNG.
    """
    try:
        h, w = master.shape[:2]

        # Create scan copy for annotation
        scan_annotated = aligned_scan.copy()
        master_annotated = master.copy()

        # ── Draw OCR error boxes on scan ──────────────────────────────────────
        for err in text_errors:
            bbox = err.get("region_bbox", {})
            if bbox:
                x, y, bw, bh = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 50), bbox.get("h", 20)
                # Ensure bbox is within image bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                x2 = min(x + bw, w - 1)
                y2 = min(y + bh, h - 1)
                thickness = 3 if err.get("severity") == "high" else 2
                cv2.rectangle(scan_annotated, (x, y), (x2, y2), COLOR_OCR, thickness)
                # Label
                label = f"OCR: {err.get('type', '')}"
                cv2.putText(scan_annotated, label, (x, max(y - 5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OCR, 1, cv2.LINE_AA)

        # ── Draw defect boxes on scan ──────────────────────────────────────────
        for defect in defects:
            bbox = defect.get("bbox", {})
            if bbox:
                x, y, bw, bh = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 20), bbox.get("h", 20)
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                x2 = min(x + bw, w - 1)
                y2 = min(y + bh, h - 1)
                cv2.rectangle(scan_annotated, (x, y), (x2, y2), COLOR_DEFECT, 2)
                label = f"Defect: {defect.get('type', '')}"
                cv2.putText(scan_annotated, label, (x, max(y - 5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_DEFECT, 1, cv2.LINE_AA)

        # ── Draw barcode boxes on scan ─────────────────────────────────────────
        for bc in barcode_results:
            bbox = bc.get("bbox", {})
            if bbox:
                x, y, bw, bh = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 50), bbox.get("h", 20)
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                x2 = min(x + bw, w - 1)
                y2 = min(y + bh, h - 1)
                color = COLOR_BARCODE_OK if bc.get("pass") else COLOR_BARCODE_FAIL
                cv2.rectangle(scan_annotated, (x, y), (x2, y2), color, 2)
                label = f"{bc.get('type', '')}: {bc.get('quality_grade', '?')}"
                cv2.putText(scan_annotated, label, (x, max(y - 5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # ── Add legend ─────────────────────────────────────────────────────────
        legend_h = 40
        legend = np.zeros((legend_h, w * 2 + 10, 3), dtype=np.uint8)
        legend[:] = (30, 30, 30)

        legend_items = [
            ("OCR Error", COLOR_OCR),
            ("Defect", COLOR_DEFECT),
            ("Barcode OK", COLOR_BARCODE_OK),
            ("Barcode Fail", COLOR_BARCODE_FAIL),
        ]
        x_pos = 10
        for name, color in legend_items:
            cv2.rectangle(legend, (x_pos, 10), (x_pos + 15, 25), color, -1)
            cv2.putText(legend, name, (x_pos + 20, 21),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            x_pos += 120

        # ── Labels ─────────────────────────────────────────────────────────────
        label_h = 30
        label_bar = np.zeros((label_h, w * 2 + 10, 3), dtype=np.uint8)
        label_bar[:] = (13, 27, 42)  # NAVY

        cv2.putText(label_bar, "MASTER (Original PDF)", (w // 2 - 80, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 194, 203), 1, cv2.LINE_AA)
        cv2.putText(label_bar, "SCAN (Printed Label) + Annotations", (w + 10 + w // 2 - 110, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 194, 203), 1, cv2.LINE_AA)

        # ── Combine side by side ───────────────────────────────────────────────
        separator = np.full((h, 10, 3), (100, 100, 100), dtype=np.uint8)
        comparison = np.hstack([master_annotated, separator, scan_annotated])

        # Stack: label bar + comparison + legend
        final = np.vstack([label_bar, comparison, legend])

        # Save
        out_dir = Path(settings.reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job_id}_annotated.jpg"
        cv2.imwrite(str(out_path), final, [cv2.IMWRITE_JPEG_QUALITY, 92])

        log.info(f"Annotated image saved: {out_path}")
        return out_path

    except Exception as e:
        log.error(f"Annotated image creation failed: {e}")
        return None
