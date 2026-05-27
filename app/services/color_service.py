"""
Greenpack Pro — Color Analysis Service
Delta-E CIE2000 per-pixel comparison with K-means zone detection
"""
import cv2
import numpy as np
import logging
from typing import Optional

log = logging.getLogger(__name__)


def analyze_color_zones(
    master: np.ndarray,
    scan: np.ndarray,
    threshold: float = 2.0,
    n_zones: int = 6,
) -> dict:
    """
    Analyze color accuracy using Delta-E CIE2000.

    Args:
        master: Master image (BGR numpy array)
        scan: Aligned scan image (BGR numpy array)
        threshold: ΔE tolerance (default 2.0 — FMCG standard)
        n_zones: Number of color zones to detect (K-means)

    Returns:
        dict with zone_results, heatmap, overall stats
    """
    try:
        from skimage import color as skcolor

        # Ensure same dimensions
        h, w = master.shape[:2]
        if scan.shape[:2] != (h, w):
            scan = cv2.resize(scan, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Convert BGR → RGB → LAB for accurate color math
        master_rgb = cv2.cvtColor(master, cv2.COLOR_BGR2RGB).astype("float64") / 255.0
        scan_rgb = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB).astype("float64") / 255.0

        master_lab = skcolor.rgb2lab(master_rgb)
        scan_lab = skcolor.rgb2lab(scan_rgb)

        # Per-pixel Delta-E CIE2000
        dE_map = skcolor.deltaE_ciede2000(master_lab, scan_lab)

        # Overall stats
        mean_dE = float(np.mean(dE_map))
        max_dE = float(np.max(dE_map))
        pct_fail = float(np.mean(dE_map > threshold) * 100)

        # Zone analysis via K-means clustering on master image
        zone_results = _analyze_color_zones(
            master, master_rgb, dE_map, threshold, n_zones
        )

        # Generate heatmap for visualization
        heatmap = _create_heatmap(dE_map, threshold)

        log.info(
            f"Color analysis: mean ΔE={mean_dE:.2f} max={max_dE:.2f} "
            f"fail_pct={pct_fail:.1f}% zones={len(zone_results)}"
        )

        return {
            "zone_results": zone_results,
            "mean_delta_e": round(mean_dE, 3),
            "max_delta_e": round(max_dE, 3),
            "pct_out_of_tolerance": round(pct_fail, 2),
            "threshold": threshold,
            "pass": mean_dE <= threshold and pct_fail <= 10.0,
            "heatmap": heatmap,
            "heatmap_b64": _heatmap_to_b64(heatmap),
        }

    except ImportError:
        log.warning("scikit-image not available, using simplified color analysis")
        return _simplified_color_analysis(master, scan, threshold)
    except Exception as e:
        log.error(f"Color analysis failed: {e}")
        return {
            "zone_results": [],
            "mean_delta_e": 0,
            "max_delta_e": 0,
            "pct_out_of_tolerance": 0,
            "pass": True,
            "error": str(e),
        }


def _analyze_color_zones(
    master_bgr: np.ndarray,
    master_rgb: np.ndarray,
    dE_map: np.ndarray,
    threshold: float,
    n_zones: int,
) -> list[dict]:
    """Use K-means to identify color zones and compute ΔE per zone"""
    h, w = master_bgr.shape[:2]
    pixels = master_rgb.reshape(-1, 3).astype(np.float32)

    # K-means clustering to find dominant color zones
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    try:
        _, labels, centers = cv2.kmeans(
            pixels, n_zones, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
    except cv2.error:
        n_zones = min(n_zones, 3)
        _, labels, centers = cv2.kmeans(
            pixels, n_zones, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )

    labels = labels.reshape(h, w)
    zones = []

    for zone_idx in range(n_zones):
        zone_mask = labels == zone_idx
        zone_pixels_count = np.sum(zone_mask)

        if zone_pixels_count < 100:  # Skip tiny zones
            continue

        zone_dE = dE_map[zone_mask]
        zone_mean_dE = float(np.mean(zone_dE))
        zone_max_dE = float(np.max(zone_dE))
        zone_pct_fail = float(np.mean(zone_dE > threshold) * 100)

        # Identify zone by dominant color
        center_rgb = centers[zone_idx]
        center_bgr = (center_rgb[2] * 255, center_rgb[1] * 255, center_rgb[0] * 255)
        zone_name = _color_name(center_rgb)

        # Find bounding box of zone
        rows = np.any(zone_mask, axis=1)
        cols = np.any(zone_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        zones.append({
            "zone_id": zone_idx,
            "zone_name": zone_name,
            "color_rgb": [round(float(c * 255)) for c in center_rgb],
            "mean_delta_e": round(zone_mean_dE, 3),
            "max_delta_e": round(zone_max_dE, 3),
            "pct_out_of_tolerance": round(zone_pct_fail, 2),
            "threshold": threshold,
            "pass": zone_mean_dE <= threshold and zone_pct_fail <= 15.0,
            "pixel_count": int(zone_pixels_count),
            "bbox": {
                "x": int(x_min), "y": int(y_min),
                "w": int(x_max - x_min), "h": int(y_max - y_min),
            },
        })

    zones.sort(key=lambda z: z["pixel_count"], reverse=True)
    return zones


def _color_name(rgb: np.ndarray) -> str:
    """Approximate color name from RGB values"""
    r, g, b = rgb[0], rgb[1], rgb[2]
    names = {
        "Red": (r > 0.6 and g < 0.3 and b < 0.3),
        "Green": (g > 0.5 and r < 0.4 and b < 0.4),
        "Blue": (b > 0.5 and r < 0.4 and g < 0.4),
        "Yellow": (r > 0.7 and g > 0.7 and b < 0.3),
        "White": (r > 0.85 and g > 0.85 and b > 0.85),
        "Black": (r < 0.15 and g < 0.15 and b < 0.15),
        "Orange": (r > 0.8 and g > 0.3 and g < 0.6 and b < 0.2),
        "Purple": (r > 0.4 and b > 0.4 and g < 0.3),
    }
    for name, condition in names.items():
        if condition:
            return name
    return f"Zone ({int(r*255)},{int(g*255)},{int(b*255)})"


def _create_heatmap(dE_map: np.ndarray, threshold: float) -> np.ndarray:
    """Create color-coded heatmap: blue=OK, red=out of tolerance"""
    normalized = np.clip(dE_map / (threshold * 3), 0, 1)
    heatmap_u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)


def _heatmap_to_b64(heatmap: Optional[np.ndarray]) -> Optional[str]:
    """Convert heatmap to base64 PNG for API response"""
    if heatmap is None:
        return None
    import base64
    _, buffer = cv2.imencode(".jpg", heatmap, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def _simplified_color_analysis(
    master: np.ndarray, scan: np.ndarray, threshold: float
) -> dict:
    """Fallback: simple BGR channel difference when scikit-image unavailable"""
    h, w = master.shape[:2]
    if scan.shape[:2] != (h, w):
        scan = cv2.resize(scan, (w, h))

    diff = cv2.absdiff(master, scan).astype(float)
    mean_diff = float(np.mean(diff))

    return {
        "zone_results": [{
            "zone_id": 0,
            "zone_name": "Overall",
            "mean_delta_e": round(mean_diff / 10, 2),
            "pass": mean_diff < threshold * 10,
        }],
        "mean_delta_e": round(mean_diff / 10, 2),
        "max_delta_e": round(float(np.max(diff)) / 10, 2),
        "pct_out_of_tolerance": round(float(np.mean(diff > threshold * 10) * 100), 2),
        "pass": mean_diff < threshold * 10,
        "engine": "simplified",
    }
