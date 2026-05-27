"""
Greenpack Pro v2.0 — Multi-Up Label Detection Engine

This is the CORE module that handles the user's main requirement:
  "Put a cut-out of the roll in scanner (1-15 labels) and the software detects
   each label, compares it to the master, and returns per-label pass/fail."

Algorithm:
  1. Enhance scan (adaptive threshold + morphology to find label boundaries)
  2. Detect all label regions using contour detection + template matching
  3. Determine imposition (N×M grid or irregular)
  4. Validate label count against expected
  5. Crop each label region
  6. Align each crop to master
  7. Run full inspection pipeline per label
  8. Aggregate results with per-label status

Handles:
  - Printed labels on white liner
  - Clear/transparent labels (polarity detection)
  - Opaque labels on any background
  - Irregular positioning (not perfectly aligned)
  - 1 to 15+ labels in any arrangement
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def detect_and_extract_labels(
    scan_image: np.ndarray,
    master_image: np.ndarray,
    expected_count: Optional[int] = None,
    min_match_score: float = 0.55,
) -> dict:
    """
    Detect all label instances in a scanned multi-up sheet.

    Args:
        scan_image: Scanned image (BGR) containing 1..N labels
        master_image: Master label image (BGR) - single label
        expected_count: Expected number of labels (optional — enables missing detection)
        min_match_score: Minimum template-match score to count as label (0-1)

    Returns:
        dict with keys:
          - method: "template_match" | "contour" | "single"
          - labels: list of dict, each with:
                   {"bbox", "crop", "row", "col", "score"}
          - count_found: int
          - count_expected: int (or None)
          - missing_labels: int
          - imposition: "rows × cols" string
          - confidence: overall detection confidence (0-1)
          - debug_visualization: annotated scan (for UI)
    """
    log.info(
        f"Multi-up detection starting: scan={scan_image.shape[:2]}, "
        f"master={master_image.shape[:2]}, expected={expected_count}"
    )

    # Step 1: Try template-matching-based detection (most accurate)
    labels, score_map = _detect_via_template_matching(
        scan_image, master_image, min_match_score
    )
    method = "template_match"

    # Step 2: Fallback to contour-based detection if template matching gave too few
    if expected_count is not None and len(labels) < expected_count * 0.6:
        log.info(f"Template match found {len(labels)}, expected ~{expected_count}. Trying contour detection.")
        contour_labels = _detect_via_contours(scan_image, master_image)
        if len(contour_labels) > len(labels):
            labels = contour_labels
            method = "contour"

    # Step 3: If still nothing, treat whole scan as single label
    if not labels:
        log.warning("No labels detected, treating whole scan as a single label")
        h, w = scan_image.shape[:2]
        labels = [{
            "bbox": {"x": 0, "y": 0, "w": w, "h": h},
            "crop": scan_image.copy(),
            "row": 0, "col": 0, "score": 0.5,
        }]
        method = "single"

    # Step 4: Order labels by row, then column
    labels = _order_labels_grid(labels)

    # Step 5: Compute imposition (rows × cols)
    rows, cols = _infer_grid(labels)

    # Step 6: Missing label detection
    missing = 0
    if expected_count is not None:
        missing = max(0, expected_count - len(labels))
        if len(labels) > expected_count:
            log.warning(f"Found MORE labels ({len(labels)}) than expected ({expected_count})")

    # Step 7: Overall confidence
    if labels:
        avg_score = np.mean([lb["score"] for lb in labels])
    else:
        avg_score = 0.0

    # Step 8: Build debug visualization
    viz = _create_detection_viz(scan_image, labels, missing)

    result = {
        "method": method,
        "labels": labels,
        "count_found": len(labels),
        "count_expected": expected_count,
        "missing_labels": missing,
        "imposition": f"{rows} × {cols}" if rows * cols else "irregular",
        "confidence": float(avg_score),
        "debug_visualization": viz,
    }
    log.info(
        f"Detection complete: {len(labels)} labels found "
        f"({rows}×{cols}), missing={missing}, avg_score={avg_score:.2f}"
    )
    return result


# ── Method 1: Template Matching with NMS ──────────────────────────────────────

def _detect_via_template_matching(
    scan: np.ndarray,
    master: np.ndarray,
    min_score: float = 0.55,
) -> Tuple[List[dict], np.ndarray]:
    """
    Detect label locations using multi-scale template matching + Non-Max Suppression.
    Most accurate when master and scanned labels are similar scale.
    """
    # Convert both to grayscale for matching
    scan_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)

    # Equalize histograms for lighting invariance
    scan_gray = cv2.equalizeHist(scan_gray)
    master_gray = cv2.equalizeHist(master_gray)

    mh, mw = master_gray.shape
    sh, sw = scan_gray.shape

    # Sanity check: master must fit inside scan
    if mh >= sh or mw >= sw:
        # Master larger than scan - resize master to fit
        scale = min(sh * 0.9 / mh, sw * 0.9 / mw)
        nh, nw = int(mh * scale), int(mw * scale)
        master_gray = cv2.resize(master_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        mh, mw = nh, nw

    # Multi-scale template matching (labels may be slightly different size)
    best_scale = 1.0
    best_score = -1
    best_locations = []

    for scale in [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]:
        nh = max(20, int(mh * scale))
        nw = max(20, int(mw * scale))

        if nh >= sh or nw >= sw:
            continue

        scaled_template = cv2.resize(master_gray, (nw, nh), interpolation=cv2.INTER_AREA)

        try:
            result = cv2.matchTemplate(scan_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        except cv2.error as e:
            log.debug(f"matchTemplate failed at scale {scale}: {e}")
            continue

        # Find all locations above threshold
        locations = np.where(result >= min_score)
        scores = result[locations]

        if len(scores) > 0 and scores.max() > best_score:
            best_score = scores.max()
            best_scale = scale
            best_locations = list(zip(locations[1], locations[0], scores))  # (x, y, score)
            best_template_size = (nw, nh)

    if not best_locations:
        log.debug(f"Template matching found no matches above {min_score}")
        return [], np.zeros((1, 1))

    # Sort by score (highest first)
    best_locations.sort(key=lambda t: -t[2])

    # Non-Maximum Suppression to eliminate overlapping detections
    nw, nh = best_template_size
    bboxes = [(x, y, x + nw, y + nh, score) for x, y, score in best_locations]
    bboxes = _non_max_suppression(bboxes, overlap_threshold=0.3)

    # Extract crops
    labels = []
    for x1, y1, x2, y2, score in bboxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(sw, int(x2)), min(sh, int(y2))
        crop = scan[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        labels.append({
            "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            "crop": crop,
            "row": 0, "col": 0,
            "score": float(score),
        })

    log.info(f"Template matching: {len(labels)} labels at scale {best_scale:.2f}")
    return labels, np.zeros((1, 1))


# ── Method 2: Contour-Based Detection ─────────────────────────────────────────

def _detect_via_contours(scan: np.ndarray, master: np.ndarray) -> List[dict]:
    """
    Detect labels by finding rectangular contours in the scan.
    Works well for labels on high-contrast liner (white paper on dark, or vice versa).
    """
    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # Try multiple threshold methods to handle different polarities
    candidates = []

    # Method A: Adaptive threshold (handles shadows)
    binary_a = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
    )
    candidates.append(binary_a)

    # Method B: Otsu + inverted (for dark labels on light background)
    _, binary_b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(binary_b)
    candidates.append(cv2.bitwise_not(binary_b))

    # Method C: Edge-based
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    candidates.append(edges_closed)

    # Target label size (from master)
    mh, mw = master.shape[:2]
    target_area = mw * mh
    min_area = target_area * 0.3
    max_area = target_area * 3.0

    # Minimum fraction of scan area a single label should be
    scan_area = scan.shape[0] * scan.shape[1]
    min_area = max(min_area, scan_area * 0.005)
    max_area = min(max_area, scan_area * 0.5)

    all_bboxes = []

    for binary in candidates:
        # Morph to close gaps inside labels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Filter extreme aspect ratios (not label-like)
            aspect = w / h if h > 0 else 0
            master_aspect = mw / mh if mh > 0 else 1
            if aspect < master_aspect * 0.5 or aspect > master_aspect * 2.0:
                continue

            # Check fill ratio (reject sparse contours)
            fill = area / (w * h)
            if fill < 0.4:
                continue

            all_bboxes.append((x, y, x + w, y + h, fill))

    # NMS to dedupe across methods
    all_bboxes = _non_max_suppression(all_bboxes, overlap_threshold=0.3)

    labels = []
    for x1, y1, x2, y2, score in all_bboxes:
        crop = scan[int(y1):int(y2), int(x1):int(x2)].copy()
        if crop.size == 0:
            continue
        labels.append({
            "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
            "crop": crop,
            "row": 0, "col": 0,
            "score": float(score),
        })

    log.info(f"Contour detection: {len(labels)} candidate labels")
    return labels


# ── Helpers ────────────────────────────────────────────────────────────────────

def _non_max_suppression(
    bboxes: List[Tuple[float, float, float, float, float]],
    overlap_threshold: float = 0.3,
) -> List[Tuple[float, float, float, float, float]]:
    """Suppress overlapping detections, keeping the highest-score bbox"""
    if not bboxes:
        return []

    boxes = np.array([[b[0], b[1], b[2], b[3]] for b in bboxes], dtype=np.float32)
    scores = np.array([b[4] for b in bboxes], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)
        order = order[1:][iou < overlap_threshold]

    return [bboxes[i] for i in keep]


def _order_labels_grid(labels: List[dict]) -> List[dict]:
    """
    Sort labels by row, then column. Assign row/col indices.
    Uses y-clustering to group into rows.
    """
    if not labels:
        return labels

    # Get center y for each label
    centers = [(i, lb["bbox"]["y"] + lb["bbox"]["h"] / 2) for i, lb in enumerate(labels)]

    # Average label height for row threshold
    avg_h = np.mean([lb["bbox"]["h"] for lb in labels])
    row_thresh = avg_h * 0.5

    # Sort by y
    centers.sort(key=lambda t: t[1])

    # Cluster into rows
    rows = []
    current_row = [centers[0]]
    for idx, cy in centers[1:]:
        if abs(cy - current_row[-1][1]) < row_thresh:
            current_row.append((idx, cy))
        else:
            rows.append(current_row)
            current_row = [(idx, cy)]
    rows.append(current_row)

    # Within each row, sort by x
    ordered = []
    for row_idx, row in enumerate(rows):
        items = [(i, labels[i]["bbox"]["x"]) for i, _ in row]
        items.sort(key=lambda t: t[1])
        for col_idx, (i, _) in enumerate(items):
            lb = labels[i].copy()
            lb["row"] = row_idx
            lb["col"] = col_idx
            ordered.append(lb)

    return ordered


def _infer_grid(labels: List[dict]) -> Tuple[int, int]:
    """Infer imposition rows × cols from ordered labels"""
    if not labels:
        return 0, 0
    rows = max(lb["row"] for lb in labels) + 1
    cols = 0
    for r in range(rows):
        row_cols = sum(1 for lb in labels if lb["row"] == r)
        cols = max(cols, row_cols)
    return rows, cols


def _create_detection_viz(
    scan: np.ndarray, labels: List[dict], missing: int = 0
) -> np.ndarray:
    """Draw detection overlay for UI preview"""
    viz = scan.copy()

    # Colors per row (cycle)
    row_colors = [
        (0, 255, 128), (0, 128, 255), (255, 128, 0),
        (255, 0, 128), (128, 255, 0), (128, 0, 255),
    ]

    for lb in labels:
        bbox = lb["bbox"]
        color = row_colors[lb["row"] % len(row_colors)]
        x, y = bbox["x"], bbox["y"]
        x2, y2 = x + bbox["w"], y + bbox["h"]
        cv2.rectangle(viz, (x, y), (x2, y2), color, 3)

        # Label number
        label_id = f"{lb['row']+1}-{lb['col']+1}"
        cv2.putText(viz, label_id, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Score
        score_text = f"{lb['score']:.2f}"
        cv2.putText(viz, score_text, (x + 10, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Banner showing counts
    banner_h = 40
    banner = np.zeros((banner_h, viz.shape[1], 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    text = f"DETECTED: {len(labels)} labels"
    if missing > 0:
        text += f"   |   MISSING: {missing}"
        banner[:] = (30, 30, 150)
    cv2.putText(banner, text, (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return np.vstack([banner, viz])


# ── Transparent/Clear Label Support ─────────────────────────────────────────

def detect_clear_labels(scan: np.ndarray, master: np.ndarray) -> List[dict]:
    """
    Detect transparent / clear-on-clear labels which are visible only via
    light refraction on their die-cut edges.

    Uses edge density maps rather than intensity.
    """
    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # Edge detection catches the die-cut boundary even on clear labels
    edges = cv2.Canny(gray, 15, 60, apertureSize=3)

    # Dilate to connect edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Close small gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mh, mw = master.shape[:2]
    target_area = mw * mh

    labels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < target_area * 0.3 or area > target_area * 3.0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        crop = scan[y:y+h, x:x+w].copy()
        labels.append({
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "crop": crop,
            "row": 0, "col": 0,
            "score": 0.7,  # Medium confidence for clear labels
            "is_clear": True,
        })

    return _order_labels_grid(labels)
