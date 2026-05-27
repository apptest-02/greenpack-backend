"""
Greenpack Pro v3.0 — Icon / Logo Size Verification

Detects graphic elements (logos, icons, certification marks like 'CE', 'FSC',
'Recycle' symbols, etc.) and compares dimensions between final design and trial proof.

Method:
  1. Find connected components in both images (after binarization)
  2. Filter by size (likely icons are 0.5%-15% of label area)
  3. Match icons between images using shape descriptors + position
  4. Flag any mismatch in width or height beyond tolerance
"""
import logging
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def detect_graphic_elements(image: np.ndarray) -> List[dict]:
    """Detect potential icon/logo regions in an image"""
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Multi-method detection (handles various icon styles)
    candidates = []

    # Method 1: Adaptive threshold for filled icons
    binary_a = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    candidates.append(binary_a)

    # Method 2: Otsu for high-contrast icons
    _, binary_o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(binary_o)

    # Method 3: Edges for outline icons
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    candidates.append(closed)

    image_area = h * w
    min_icon_area = max(image_area * 0.001, 100)   # 0.1% min
    max_icon_area = image_area * 0.15              # 15% max

    all_icons = []
    for binary in candidates:
        # Close gaps within icons
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_icon_area or area > max_icon_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter very flat shapes (likely text, not icons)
            aspect = cw / ch if ch > 0 else 0
            if aspect > 8 or aspect < 0.125:
                continue

            # Get shape moments for matching
            try:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                else:
                    cx, cy = x + cw / 2, y + ch / 2
                hu_moments = cv2.HuMoments(m).flatten()
            except Exception:
                continue

            # Fill ratio
            fill_ratio = area / (cw * ch) if (cw * ch) > 0 else 0
            if fill_ratio < 0.15:  # Too sparse to be an icon
                continue

            all_icons.append({
                "bbox": {"x": x, "y": y, "w": cw, "h": ch},
                "center": (cx, cy),
                "area": area,
                "aspect": aspect,
                "fill_ratio": round(fill_ratio, 3),
                "hu_moments": [float(h) for h in hu_moments],
            })

    # NMS to dedupe across methods
    if not all_icons:
        return []

    all_icons.sort(key=lambda i: -i["area"])
    keep = []
    for icon in all_icons:
        bx, by = icon["bbox"]["x"], icon["bbox"]["y"]
        bw, bh = icon["bbox"]["w"], icon["bbox"]["h"]
        overlap = False
        for kept in keep:
            kx, ky = kept["bbox"]["x"], kept["bbox"]["y"]
            kw, kh = kept["bbox"]["w"], kept["bbox"]["h"]
            ox = max(bx, kx)
            oy = max(by, ky)
            ox2 = min(bx + bw, kx + kw)
            oy2 = min(by + bh, ky + kh)
            if ox2 > ox and oy2 > oy:
                inter = (ox2 - ox) * (oy2 - oy)
                union = bw * bh + kw * kh - inter
                if inter / union > 0.5:
                    overlap = True
                    break
        if not overlap:
            keep.append(icon)
    return keep


def match_icons(final_icons: List[dict], trial_icons: List[dict],
                position_tolerance_px: float = 30) -> List[Tuple[dict, dict]]:
    """
    Pair up icons in the final design with their counterparts in the trial.
    Match by position similarity + shape similarity (Hu moments).
    """
    matches = []
    used_trial = set()

    for f_icon in final_icons:
        fx, fy = f_icon["center"]
        best_idx = None
        best_score = float("inf")

        for i, t_icon in enumerate(trial_icons):
            if i in used_trial:
                continue
            tx, ty = t_icon["center"]
            pos_dist = np.sqrt((fx - tx) ** 2 + (fy - ty) ** 2)

            if pos_dist > position_tolerance_px * 5:  # Way off
                continue

            # Shape similarity: log-Hu distance
            shape_dist = sum(
                abs(np.sign(a) * np.log10(abs(a) + 1e-10)
                    - np.sign(b) * np.log10(abs(b) + 1e-10))
                for a, b in zip(f_icon["hu_moments"], t_icon["hu_moments"])
            )

            # Combined score (lower is better)
            score = pos_dist + shape_dist * 50
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            matches.append((f_icon, trial_icons[best_idx]))
            used_trial.add(best_idx)

    return matches


def compare_icon_sizes(final_img: np.ndarray, trial_img: np.ndarray,
                        size_tolerance_pct: float = 10.0) -> dict:
    """
    Main entry point: compare icon dimensions between final design and trial.

    Returns dict with: total_checked, matches, mismatches, mismatch_details
    """
    try:
        final_icons = detect_graphic_elements(final_img)
        trial_icons = detect_graphic_elements(trial_img)

        matches = match_icons(final_icons, trial_icons)
        mismatch_details = []

        for f_icon, t_icon in matches:
            f_w = f_icon["bbox"]["w"]
            f_h = f_icon["bbox"]["h"]
            t_w = t_icon["bbox"]["w"]
            t_h = t_icon["bbox"]["h"]

            w_diff_pct = abs(f_w - t_w) / f_w * 100 if f_w > 0 else 0
            h_diff_pct = abs(f_h - t_h) / f_h * 100 if f_h > 0 else 0

            if w_diff_pct > size_tolerance_pct or h_diff_pct > size_tolerance_pct:
                mismatch_details.append({
                    "position": f"({int(f_icon['center'][0])}, {int(f_icon['center'][1])})",
                    "final_size": f"{f_w}×{f_h}px",
                    "trial_size": f"{t_w}×{t_h}px",
                    "width_diff_pct": round(w_diff_pct, 1),
                    "height_diff_pct": round(h_diff_pct, 1),
                    "description": f"Icon {f_w}×{f_h}px on final → {t_w}×{t_h}px on trial "
                                   f"(W diff {w_diff_pct:.1f}%, H diff {h_diff_pct:.1f}%)",
                    "final_bbox": f_icon["bbox"],
                    "trial_bbox": t_icon["bbox"],
                })

        unmatched_final = len(final_icons) - len(matches)

        return {
            "total_in_final": len(final_icons),
            "total_in_trial": len(trial_icons),
            "total_checked": len(matches),
            "matches": len(matches),
            "mismatches": len(mismatch_details),
            "missing_in_trial": max(0, unmatched_final),
            "mismatch_details": mismatch_details,
            "tolerance_pct": size_tolerance_pct,
        }
    except Exception as e:
        log.exception(f"Icon size comparison failed: {e}")
        return {
            "total_checked": 0,
            "matches": 0,
            "mismatches": 0,
            "mismatch_details": [],
            "error": str(e),
        }
