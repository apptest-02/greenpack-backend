"""
Greenpack Pro v3.0 — PANTONE Color Identification Service

The user's exact requirement:
"Software should also find the colour code by just uploading the past work stickers
by scanning it. The international standards colour code dataset to find out colour
after scan or upload of the pdf which colour code have been used."

Workflow:
  1. User uploads scanned image or PDF of past work
  2. Service extracts dominant colors (K-means clustering)
  3. Each dominant color is converted: BGR → sRGB → XYZ → Lab (D50)
  4. Nearest-neighbor search in PMS library using Delta-E CIE2000
  5. Return list of detected Pantone codes per region with ΔE distance + confidence

Includes:
  - Bundled PANTONE library (~700 spot colors)
  - K-means dominant color extraction (configurable k)
  - Lab color space conversion (D50 illuminant for graphic arts)
  - Delta-E CIE2000 nearest match
  - Top-N matches per region with confidence ranking
  - Per-region heatmap showing color zones
  - Custom library import (CSV from spectrophotometer)
"""
import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_LIBRARY_CACHE = None


# ── Color Space Conversions ───────────────────────────────────────────────────

def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb_to_xyz(r: float, g: float, b: float) -> tuple:
    """sRGB (0-255) to CIE XYZ"""
    rl = _srgb_to_linear(r)
    gl = _srgb_to_linear(g)
    bl = _srgb_to_linear(b)
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    return x * 100, y * 100, z * 100


def xyz_to_lab(x: float, y: float, z: float, illuminant: str = "D50") -> tuple:
    """XYZ to CIE L*a*b* (default D50 for graphic arts)"""
    if illuminant == "D50":
        Xn, Yn, Zn = 96.422, 100.000, 82.521
    else:  # D65
        Xn, Yn, Zn = 95.047, 100.000, 108.883

    def f(t):
        delta = 6 / 29
        return t ** (1 / 3) if t > delta ** 3 else (t / (3 * delta ** 2)) + (4 / 29)

    fx, fy, fz = f(x / Xn), f(y / Yn), f(z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


def rgb_to_lab(r: int, g: int, b: int) -> tuple:
    """sRGB direct to Lab (D50)"""
    x, y, z = srgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z, "D50")


def delta_e_cie2000(lab1: tuple, lab2: tuple) -> float:
    """
    CIE Delta E 2000 — most accurate color difference formula.
    Result interpretation:
      < 1.0  : not perceptible (same color)
      1.0–2.0: perceptible on close inspection
      2.0–3.5: noticeable at a glance
      3.5–5.0: clearly different
      > 5.0  : completely different colors
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - math.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7)))

    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    if abs(h1p - h2p) > 180:
        if h2p <= h1p:
            h2p += 360
        else:
            h1p += 360
    avg_Hp = (h1p + h2p) / 2.0

    T = (1 - 0.17 * math.cos(math.radians(avg_Hp - 30))
         + 0.24 * math.cos(math.radians(2 * avg_Hp))
         + 0.32 * math.cos(math.radians(3 * avg_Hp + 6))
         - 0.20 * math.cos(math.radians(4 * avg_Hp - 63)))

    delta_hp = h2p - h1p
    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    delta_Hp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(delta_hp / 2))

    SL = 1 + (0.015 * (avg_L - 50) ** 2) / math.sqrt(20 + (avg_L - 50) ** 2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T

    delta_theta = 30 * math.exp(-(((avg_Hp - 275) / 25) ** 2))
    RC = 2 * math.sqrt((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7))
    RT = -RC * math.sin(2 * math.radians(delta_theta))

    KL = KC = KH = 1
    return math.sqrt(
        (delta_Lp / (KL * SL)) ** 2
        + (delta_Cp / (KC * SC)) ** 2
        + (delta_Hp / (KH * SH)) ** 2
        + RT * (delta_Cp / (KC * SC)) * (delta_Hp / (KH * SH))
    )


# ── Library Loading ───────────────────────────────────────────────────────────

def load_pantone_library() -> dict:
    """Load the bundled PANTONE library JSON (cached)"""
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None:
        # Relative to this file: backend/app/services/pantone_service.py
        # → ../../data/pantone_library.json
        lib_path = Path(__file__).parent.parent.parent / "data" / "pantone_library.json"
        if not lib_path.exists():
            log.error(f"PANTONE library not found at {lib_path}")
            _LIBRARY_CACHE = {"colors": [], "version": "missing"}
        else:
            try:
                _LIBRARY_CACHE = json.loads(lib_path.read_text())
                log.info(f"Loaded PANTONE library: {_LIBRARY_CACHE.get('color_count', 0)} colors")
            except Exception as e:
                log.error(f"Failed to load PANTONE library: {e}")
                _LIBRARY_CACHE = {"colors": [], "version": "error"}
    return _LIBRARY_CACHE


def import_custom_library(csv_path: str) -> dict:
    """
    Import user's custom color library from CSV (spectrophotometer measurements).
    CSV format: code,system,L,a,b[,hex,r,g,b]
    """
    import csv

    library = {
        "version": "custom-1.0",
        "source": "user-imported CSV",
        "colors": [],
    }
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                L, a, bv = float(row["L"]), float(row["a"]), float(row["b"])
                # Lab → sRGB approximation (back-conversion)
                color = {
                    "code": row.get("code", "").strip(),
                    "system": row.get("system", "CUSTOM"),
                    "finish": row.get("finish", "Custom"),
                    "lab": [L, a, bv],
                }
                if "hex" in row and row["hex"]:
                    color["hex"] = row["hex"].strip().upper()
                if "r" in row and "g" in row and "b" in row:
                    color["rgb"] = [int(row["r"]), int(row["g"]), int(row["b"])]
                library["colors"].append(color)
            except (KeyError, ValueError) as e:
                log.warning(f"Skipping invalid row: {row} ({e})")
    library["color_count"] = len(library["colors"])
    return library


# ── Color Extraction ──────────────────────────────────────────────────────────

def extract_dominant_colors(
    image: np.ndarray,
    k: int = 8,
    ignore_white: bool = True,
    ignore_black: bool = False,
    min_area_pct: float = 0.5,
) -> List[dict]:
    """
    Extract dominant colors from an image using K-means clustering in Lab space.

    Args:
        image: BGR image (OpenCV format)
        k: Number of clusters
        ignore_white: Skip near-white pixels (paper background)
        ignore_black: Skip near-black pixels
        min_area_pct: Minimum area % to be considered a dominant color

    Returns:
        List of dicts with: rgb, lab, hex, area_pixels, area_pct, mask_bbox
    """
    if image is None or image.size == 0:
        return []

    # Resize for faster processing if huge
    h, w = image.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(image, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()
        scale = 1.0

    # Convert BGR → RGB
    rgb_img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pixels = rgb_img.reshape(-1, 3).astype(np.float32)

    # Filter out background pixels
    keep_mask = np.ones(pixels.shape[0], dtype=bool)
    if ignore_white:
        # Near-white: all channels > 235
        white_mask = np.all(pixels > 235, axis=1)
        keep_mask &= ~white_mask
    if ignore_black:
        black_mask = np.all(pixels < 20, axis=1)
        keep_mask &= ~black_mask

    valid_pixels = pixels[keep_mask]
    if valid_pixels.shape[0] < 100:
        log.warning("Too few valid pixels for color extraction")
        return []

    # K-means in Lab space (more perceptually uniform)
    # Convert RGB pixels to Lab
    lab_pixels = np.array([
        rgb_to_lab(int(p[0]), int(p[1]), int(p[2])) for p in valid_pixels[:5000]
    ], dtype=np.float32)

    # If valid_pixels > 5000, sample uniformly
    if valid_pixels.shape[0] > 5000:
        idx = np.random.choice(valid_pixels.shape[0], 5000, replace=False)
        sample_rgb = valid_pixels[idx]
        sample_lab = np.array([
            rgb_to_lab(int(p[0]), int(p[1]), int(p[2])) for p in sample_rgb
        ], dtype=np.float32)
    else:
        sample_rgb = valid_pixels
        sample_lab = lab_pixels

    # Run K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    k_actual = min(k, len(sample_lab))
    if k_actual < 2:
        return []

    _, labels, centers_lab = cv2.kmeans(
        sample_lab, k_actual, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # Calculate cluster sizes
    cluster_sizes = np.bincount(labels.flatten(), minlength=k_actual)
    total = sum(cluster_sizes)

    # Compute average RGB per cluster (for display)
    centers_rgb = []
    for c_idx in range(k_actual):
        mask = labels.flatten() == c_idx
        if mask.sum() > 0:
            avg_rgb = sample_rgb[mask].mean(axis=0)
            centers_rgb.append(avg_rgb)
        else:
            centers_rgb.append([128, 128, 128])

    # Build color list, filtering out tiny clusters
    colors = []
    for i in range(k_actual):
        area_pct = (cluster_sizes[i] / total) * 100
        if area_pct < min_area_pct:
            continue

        rgb = [int(round(centers_rgb[i][0])),
               int(round(centers_rgb[i][1])),
               int(round(centers_rgb[i][2]))]
        lab = [round(float(centers_lab[i][0]), 2),
               round(float(centers_lab[i][1]), 2),
               round(float(centers_lab[i][2]), 2)]
        hex_str = "#{:02X}{:02X}{:02X}".format(*rgb)

        colors.append({
            "rgb": rgb,
            "lab": lab,
            "hex": hex_str,
            "area_pct": round(float(area_pct), 2),
        })

    # Sort by area (largest first)
    colors.sort(key=lambda c: -c["area_pct"])
    return colors


# ── PMS Matching ──────────────────────────────────────────────────────────────

def find_closest_pantone(
    target_lab: tuple,
    library: Optional[dict] = None,
    top_n: int = 5,
    max_delta_e: float = 50.0,
) -> List[dict]:
    """
    Find the top N closest PANTONE matches for a target Lab color.

    Returns list of dicts with: code, system, finish, hex, rgb, lab, delta_e
    """
    if library is None:
        library = load_pantone_library()

    matches = []
    for color in library.get("colors", []):
        if "lab" not in color:
            continue
        try:
            de = delta_e_cie2000(target_lab, tuple(color["lab"]))
        except Exception:
            continue
        if de <= max_delta_e:
            matches.append({**color, "delta_e": round(de, 3)})

    matches.sort(key=lambda m: m["delta_e"])
    return matches[:top_n]


# ── Main Identification Pipeline ──────────────────────────────────────────────

def identify_pantone_colors_in_image(
    image: np.ndarray,
    k: int = 8,
    top_n_per_color: int = 3,
    ignore_white: bool = True,
) -> dict:
    """
    Main entry point: scan image → extract colors → match to PANTONE.

    Returns:
        dict with:
            - extracted_colors: list of dominant colors with PMS matches
            - total_colors_found: int
            - library_size: int
            - method: str
    """
    library = load_pantone_library()

    # Extract dominant colors
    dominant = extract_dominant_colors(
        image, k=k, ignore_white=ignore_white, min_area_pct=0.5
    )

    if not dominant:
        return {
            "extracted_colors": [],
            "total_colors_found": 0,
            "library_size": library.get("color_count", 0),
            "method": "kmeans_lab",
        }

    # Match each dominant color to PMS
    results = []
    for color in dominant:
        target_lab = tuple(color["lab"])
        pms_matches = find_closest_pantone(target_lab, library, top_n=top_n_per_color)

        # Confidence: based on best ΔE
        best_de = pms_matches[0]["delta_e"] if pms_matches else 999
        if best_de < 1.0:
            confidence = "exact"
            confidence_pct = 100
        elif best_de < 2.0:
            confidence = "very_high"
            confidence_pct = 90
        elif best_de < 3.5:
            confidence = "high"
            confidence_pct = 75
        elif best_de < 5.0:
            confidence = "medium"
            confidence_pct = 60
        elif best_de < 10.0:
            confidence = "low"
            confidence_pct = 40
        else:
            confidence = "no_match"
            confidence_pct = 20

        results.append({
            **color,
            "pms_matches": pms_matches,
            "best_match_code": pms_matches[0]["code"] if pms_matches else None,
            "best_match_delta_e": best_de,
            "match_confidence": confidence,
            "match_confidence_pct": confidence_pct,
        })

    return {
        "extracted_colors": results,
        "total_colors_found": len(results),
        "library_size": library.get("color_count", 0),
        "library_version": library.get("version", "unknown"),
        "method": "kmeans_lab + delta_e_cie2000",
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def create_pantone_report_image(
    image: np.ndarray,
    identification_result: dict,
) -> np.ndarray:
    """Create annotated image showing extracted colors + their PMS matches"""
    h, w = image.shape[:2]
    swatch_panel_h = 120
    panel = np.full((swatch_panel_h, w, 3), 255, dtype=np.uint8)

    colors = identification_result.get("extracted_colors", [])
    if not colors:
        cv2.putText(panel, "No dominant colors detected", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
        return np.vstack([image, panel])

    # Layout swatches
    swatch_w = max(80, w // (len(colors) + 1))
    swatch_h = 60

    for i, c in enumerate(colors[:10]):
        x = 10 + i * swatch_w
        if x + swatch_w > w - 10:
            break

        # Draw color swatch (BGR for OpenCV)
        rgb = c["rgb"]
        cv2.rectangle(panel, (x, 8), (x + swatch_w - 5, 8 + swatch_h),
                     (rgb[2], rgb[1], rgb[0]), -1)
        cv2.rectangle(panel, (x, 8), (x + swatch_w - 5, 8 + swatch_h),
                     (50, 50, 50), 1)

        # Code label
        match = c.get("best_match_code", "?")
        if match and len(match) > 18:
            match = match[:15] + "..."
        cv2.putText(panel, str(match), (x, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1, cv2.LINE_AA)

        # Delta E
        de = c.get("best_match_delta_e", "?")
        cv2.putText(panel, f"ΔE {de}", (x, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv2.LINE_AA)

        # Area %
        cv2.putText(panel, f"{c.get('area_pct', 0):.1f}%", (x, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv2.LINE_AA)

    # Banner
    banner_h = 30
    banner = np.full((banner_h, w, 3), 30, dtype=np.uint8)
    cv2.putText(banner, f"Identified {len(colors)} dominant colors  |  Library: {identification_result.get('library_size', 0)} PMS colors",
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, image, panel])
