"""
Greenpack Pro — SSIM Defect Detection Service
Structural Similarity Index for print defect detection.
Classifies smears, voids, banding, and missing elements.
"""
import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


def detect_defects(
    master: np.ndarray,
    scan: np.ndarray,
    threshold: float = 0.75,
    min_defect_area: int = 50,
) -> dict:
    """
    Detect print defects using Structural Similarity Index (SSIM).

    Args:
        master: Master label image (BGR)
        scan: Aligned scan image (BGR)
        threshold: SSIM minimum (below = defect). Default 0.75.
        min_defect_area: Minimum contour area to count as defect (pixels)

    Returns:
        dict with ssim_score, defects list, pass/fail, diff_image
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        h, w = master.shape[:2]
        if scan.shape[:2] != (h, w):
            scan = cv2.resize(scan, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Convert to grayscale for SSIM
        gray_m = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, diff = ssim(gray_m, gray_s, full=True, data_range=255)

        # Convert diff to uint8 (1.0=same → 0, 0.0=different → 255)
        diff_u8 = ((1.0 - diff) * 255).astype(np.uint8)

        # Threshold to find defect regions
        _, thresh = cv2.threshold(diff_u8, 50, 255, cv2.THRESH_BINARY)

        # Morphological ops to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours = defect regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_defect_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / ch if ch > 0 else 1.0

            # Classify defect by shape
            if aspect_ratio > 6:
                defect_type = "banding"      # Wide horizontal band
            elif area < 300:
                defect_type = "void"          # Small missing ink
            elif area < 2000:
                defect_type = "smear"         # Medium irregular
            else:
                defect_type = "missing_element"  # Large area

            severity = (
                "critical" if area > 5000
                else "high" if area > 2000
                else "medium" if area > 500
                else "low"
            )

            defects.append({
                "type": defect_type,
                "severity": severity,
                "bbox": {"x": x, "y": y, "w": cw, "h": ch},
                "area_pixels": int(area),
                "aspect_ratio": round(aspect_ratio, 2),
            })

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        defects.sort(key=lambda d: severity_order.get(d["severity"], 4))

        # Create diff visualization
        diff_vis = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)

        log.info(
            f"SSIM analysis: score={score:.4f} defects={len(defects)} pass={score >= threshold}"
        )

        return {
            "ssim_score": round(float(score), 4),
            "pass": float(score) >= threshold,
            "threshold": threshold,
            "defects": defects,
            "defect_count": len(defects),
            "diff_image": diff_vis,
        }

    except ImportError:
        log.warning("scikit-image not available, using simplified SSIM")
        return _simplified_ssim(master, scan, threshold)
    except Exception as e:
        log.error(f"SSIM analysis failed: {e}")
        return {
            "ssim_score": 0.0,
            "pass": False,
            "defects": [],
            "error": str(e),
        }


def _simplified_ssim(
    master: np.ndarray,
    scan: np.ndarray,
    threshold: float,
) -> dict:
    """Fallback when scikit-image unavailable — basic pixel diff"""
    h, w = master.shape[:2]
    if scan.shape[:2] != (h, w):
        scan = cv2.resize(scan, (w, h))

    diff = cv2.absdiff(master, scan)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mean_diff = float(np.mean(diff_gray))
    score = max(0.0, 1.0 - mean_diff / 255.0)

    _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        defects.append({
            "type": "unknown",
            "severity": "medium",
            "bbox": {"x": x, "y": y, "w": cw, "h": ch},
            "area_pixels": int(area),
        })

    return {
        "ssim_score": round(score, 4),
        "pass": score >= threshold,
        "defects": defects,
        "engine": "simplified",
    }
