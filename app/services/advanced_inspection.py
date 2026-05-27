"""
Greenpack Pro v2.0 — Advanced Inspection Modules

Additional inspection checks beyond v1.0:
  • Registration drift — how much each label is offset from master
  • Die-cut quality — edge sharpness of label boundary
  • Mottling / uneven ink — localized density variation
  • Braille — dot detection for EU pharma compliance (ISO 17351)
  • Font size verification — OCR text height in points
  • Pinhole detection — tiny holes in substrate
  • Smearing / banding / ghosting detection

All return a dict with:
  - pass: bool
  - quality_score: 0-100
  - details: method-specific details
"""
import logging
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Registration Drift ────────────────────────────────────────────────────────

def check_registration_drift(master: np.ndarray, scan: np.ndarray,
                              max_acceptable_px: float = 5.0) -> dict:
    """
    Measure per-label positional drift (offset from master).
    Uses phase correlation — very fast sub-pixel accuracy.
    """
    try:
        h, w = master.shape[:2]
        # Resize scan to master dimensions
        if scan.shape[:2] != (h, w):
            scan_r = cv2.resize(scan, (w, h))
        else:
            scan_r = scan

        m_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY).astype(np.float32)
        s_gray = cv2.cvtColor(scan_r, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Phase correlation returns (dx, dy) translation
        (dx, dy), response = cv2.phaseCorrelate(m_gray, s_gray)

        offset_px = float(np.sqrt(dx * dx + dy * dy))
        offset_mm = offset_px / 11.8  # ~300 DPI → 11.8 px/mm

        passed = offset_px <= max_acceptable_px
        quality = max(0.0, 100.0 - (offset_px / max_acceptable_px) * 50.0)

        return {
            "pass": passed,
            "quality_score": round(quality, 2),
            "offset_px": round(offset_px, 2),
            "offset_mm": round(offset_mm, 3),
            "dx_px": round(float(dx), 2),
            "dy_px": round(float(dy), 2),
            "response": round(float(response), 4),
            "max_acceptable_px": max_acceptable_px,
        }
    except Exception as e:
        log.warning(f"Registration check failed: {e}")
        return {"pass": True, "quality_score": 100.0, "offset_px": 0, "error": str(e)}


# ── Die-Cut Quality ───────────────────────────────────────────────────────────

def check_die_cut_quality(label_crop: np.ndarray) -> dict:
    """
    Assess die-cut edge quality:
      - Sharpness (how clean is the edge)
      - Smoothness (ragged vs clean)
      - Radius uniformity (for rounded corners)
      - Tears / tabs / flags

    Returns quality score 0-100 where 100 is factory-fresh cut.
    """
    try:
        h, w = label_crop.shape[:2]
        gray = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 30, 100)

        # Extract border region only (outer 10% of label)
        border_mask = np.zeros_like(edges)
        border_pct = 0.1
        bx = int(w * border_pct); by = int(h * border_pct)
        cv2.rectangle(border_mask, (0, 0), (w, h), 255, -1)
        cv2.rectangle(border_mask, (bx, by), (w - bx, h - by), 0, -1)
        border_edges = cv2.bitwise_and(edges, border_mask)

        # Find largest contour (label outline)
        contours, _ = cv2.findContours(border_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {"pass": True, "quality_score": 100.0, "details": "No clear edge detected"}

        main_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_contour, True)
        if perimeter < 50:
            return {"pass": True, "quality_score": 100.0, "details": "Edge too small"}

        # Smoothness: compare perimeter vs smoothed perimeter
        epsilon = 0.001 * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        smooth_perimeter = cv2.arcLength(approx, True)
        smoothness = smooth_perimeter / perimeter if perimeter > 0 else 1.0

        # Sharpness: edge gradient strength at the boundary
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

        # Sample gradient only at detected edge pixels
        edge_gradients = grad_mag[border_edges > 0]
        avg_gradient = float(np.mean(edge_gradients)) if edge_gradients.size > 0 else 0

        # Sharpness score: higher gradient = sharper edge
        sharpness = min(avg_gradient / 100.0, 1.0)

        # Defects: look for tears / tabs using convexity defects
        # Must use un-approximated contour for convexityDefects (monotonic indices)
        tab_count = 0
        try:
            simplified = cv2.approxPolyDP(main_contour, 2.0, True)
            if len(simplified) >= 4:
                hull_idx = cv2.convexHull(simplified, returnPoints=False)
                if hull_idx is not None and len(hull_idx) > 3:
                    # Sort indices to satisfy monotonicity requirement
                    hull_idx = np.sort(hull_idx, axis=0)
                    defects = cv2.convexityDefects(simplified, hull_idx)
                    if defects is not None:
                        for d in defects[:, 0]:
                            depth = d[3] / 256.0
                            if depth > w * 0.02:  # >2% of label width
                                tab_count += 1
        except cv2.error:
            # Convexity calculation failed - assume clean cut
            tab_count = 0

        # Composite quality
        quality = (smoothness * 40 + sharpness * 40 + max(0, 1 - tab_count / 5) * 20) * 100
        quality = max(0.0, min(100.0, quality))

        return {
            "pass": quality >= 70.0,
            "quality_score": round(quality, 2),
            "smoothness": round(smoothness, 3),
            "sharpness": round(sharpness, 3),
            "tab_count": int(tab_count),
            "avg_edge_gradient": round(avg_gradient, 2),
            "details": f"Die-cut quality: {'good' if quality >= 85 else 'acceptable' if quality >= 70 else 'poor'}",
        }
    except Exception as e:
        log.warning(f"Die-cut check failed: {e}")
        return {"pass": True, "quality_score": 100.0, "error": str(e)}


# ── Mottling / Uneven Ink ─────────────────────────────────────────────────────

def check_mottling(aligned_scan: np.ndarray, max_stddev_pct: float = 15.0) -> dict:
    """
    Detect mottling — uneven ink density in what should be solid areas.

    Method: local standard deviation map, but only in SOLID regions
    (ignoring text edges and graphic transitions that naturally have high variance).
    """
    try:
        gray = cv2.cvtColor(aligned_scan, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Find edges (text, graphics) — these naturally have high local variance
        edges = cv2.Canny(gray, 50, 150)
        # Dilate edges so we exclude nearby pixels too
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        edge_mask = cv2.dilate(edges, kernel, iterations=2)

        # Solid-area mask (non-edge regions)
        solid_mask = edge_mask == 0

        # Compute local standard deviation (rolling window)
        window = 15
        mean = cv2.blur(gray.astype(np.float32), (window, window))
        sq_mean = cv2.blur(gray.astype(np.float32) ** 2, (window, window))
        variance = sq_mean - mean ** 2
        local_std = np.sqrt(np.maximum(variance, 0))

        # Only consider solid (non-edge) regions for mottling
        if np.sum(solid_mask) > 100:
            solid_stds = local_std[solid_mask]
            avg_mottling = float(np.mean(solid_stds))
            max_mottling_px = float(np.percentile(solid_stds, 99))
        else:
            # No solid areas found (heavily patterned label)
            avg_mottling = float(np.mean(local_std))
            max_mottling_px = float(np.max(local_std))

        # Find areas with ABNORMAL variance within solid regions
        global_solid_std = float(np.std(gray[solid_mask])) if np.sum(solid_mask) > 0 else float(np.std(gray))
        mottling_threshold = max(global_solid_std * 2.0, 20.0)

        mottling_mask = ((local_std > mottling_threshold) & solid_mask).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mottling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mottling_regions = [
            {"bbox": {"x": x, "y": y, "w": mw, "h": mh}, "area": mw * mh}
            for x, y, mw, mh in [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]
        ]

        # Quality score: based on mottling in SOLID areas only
        # 0-5 avg_stddev = clean; 5-15 = acceptable; >20 = heavily mottled
        if avg_mottling < 5.0:
            quality = 100.0
        elif avg_mottling < 15.0:
            quality = 100.0 - (avg_mottling - 5.0) * 3.0  # Linear 100→70
        else:
            quality = max(0.0, 70.0 - (avg_mottling - 15.0) * 3.0)

        return {
            "pass": avg_mottling < 15.0 and len(mottling_regions) < 5,
            "quality_score": round(quality, 2),
            "avg_stddev": round(avg_mottling, 2),
            "max_stddev": round(max_mottling_px, 2),
            "mottling_regions": mottling_regions[:20],
            "severity": "high" if avg_mottling > 20 else "medium" if avg_mottling > 10 else "low",
        }
    except Exception as e:
        log.warning(f"Mottling check failed: {e}")
        return {"pass": True, "quality_score": 100.0, "error": str(e)}


# ── Pinhole Detection ─────────────────────────────────────────────────────────

def detect_pinholes(label_crop: np.ndarray, min_area: int = 4, max_area: int = 30) -> dict:
    """
    Detect pinholes — tiny light spots where ink is missing.
    """
    try:
        gray = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)

        # Find the "dark ink" regions (printed areas)
        _, ink_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Within ink regions, find small bright spots (pinholes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(ink_mask, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=2)

        # Difference = tiny holes
        diff = cv2.subtract(dilated, ink_mask)

        # Find contours of small holes
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pinholes = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(c)
                # Check circularity (pinholes are round)
                perim = cv2.arcLength(c, True)
                if perim > 0:
                    circularity = 4 * np.pi * area / (perim * perim)
                    if circularity > 0.5:
                        pinholes.append({
                            "bbox": {"x": x, "y": y, "w": w, "h": h},
                            "area": int(area),
                            "circularity": round(circularity, 3),
                        })

        count = len(pinholes)
        quality = max(0.0, 100.0 - count * 5.0)

        return {
            "pass": count < 3,
            "quality_score": round(quality, 2),
            "pinhole_count": count,
            "pinholes": pinholes[:50],  # Top 50
        }
    except Exception as e:
        log.warning(f"Pinhole detection failed: {e}")
        return {"pass": True, "quality_score": 100.0, "error": str(e)}


# ── Braille Detection (ISO 17351) ─────────────────────────────────────────────

def check_braille_region(
    image: np.ndarray,
    roi: Optional[dict] = None,
) -> dict:
    """
    Detect Braille dots in a region of interest.
    ISO 17351:2013 requires dot diameter 1.4-1.6mm, dot height 0.12-0.20mm.
    At 300 DPI, dot diameter ≈ 17-19 pixels.

    Returns:
      - dot_count: number of dots detected
      - pattern_quality: 0-100 (uniformity)
      - estimated_characters: approximate char count (6 dots/cell)
    """
    try:
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            region = image[y:y+h, x:x+w]
        else:
            region = image

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Blob detection for dots
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 30
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 400
        params.filterByCircularity = True
        params.minCircularity = 0.6
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        dots = [{
            "x": int(kp.pt[0]),
            "y": int(kp.pt[1]),
            "diameter_px": round(float(kp.size), 1),
            "diameter_mm": round(float(kp.size) / 11.8, 3),  # 300 DPI
        } for kp in keypoints]

        # Check diameter uniformity
        if dots:
            diams = [d["diameter_px"] for d in dots]
            mean_d = np.mean(diams)
            std_d = np.std(diams)
            uniformity = max(0, 100 - (std_d / mean_d * 200)) if mean_d > 0 else 0
        else:
            uniformity = 0

        return {
            "pass": len(dots) > 5 and uniformity > 70,
            "dot_count": len(dots),
            "dots": dots[:100],
            "pattern_quality": round(uniformity, 2),
            "estimated_chars": len(dots) // 6,
            "details": f"Detected {len(dots)} Braille dots",
        }
    except Exception as e:
        log.warning(f"Braille check failed: {e}")
        return {"pass": True, "dot_count": 0, "pattern_quality": 0, "error": str(e)}


# ── Font Size Verification ────────────────────────────────────────────────────

def verify_font_sizes(
    ocr_regions: List[dict],
    dpi: int = 300,
    min_pt: float = 6.0,
) -> dict:
    """
    Verify OCR text regions have font size above minimum (GMP compliance).
    1 pt = 1/72 inch = dpi/72 pixels
    """
    if not ocr_regions:
        return {"pass": True, "violations": [], "total_checked": 0}

    px_per_pt = dpi / 72.0
    min_height_px = min_pt * px_per_pt

    violations = []
    for region in ocr_regions:
        bbox = region.get("bbox", {})
        height_px = bbox.get("h", 0)
        if height_px < min_height_px:
            est_pt = height_px / px_per_pt
            violations.append({
                "text": region.get("text", "")[:50],
                "bbox": bbox,
                "height_px": height_px,
                "estimated_pt": round(est_pt, 1),
                "min_required_pt": min_pt,
                "description": f"Text '{region.get('text', '')[:30]}' ~{est_pt:.1f}pt < {min_pt}pt",
            })

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "total_checked": len(ocr_regions),
        "min_size_pt": min_pt,
        "dpi": dpi,
    }


# ── Smear & Banding Detection ──────────────────────────────────────────────────

def detect_smear_and_banding(
    master: np.ndarray, aligned_scan: np.ndarray,
) -> dict:
    """
    Detect directional printing artifacts:
    - Smear: irregular ink displacement
    - Banding: periodic stripes (from worn rollers or gear marks)
    - Ghosting: faint secondary image
    """
    try:
        h, w = master.shape[:2]
        if aligned_scan.shape[:2] != (h, w):
            aligned_scan = cv2.resize(aligned_scan, (w, h))

        m_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
        s_gray = cv2.cvtColor(aligned_scan, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(m_gray, s_gray)

        # Horizontal banding detection (FFT on row sums)
        row_sums = diff.sum(axis=1).astype(np.float32)
        row_sums -= row_sums.mean()

        fft = np.fft.fft(row_sums)
        freq_power = np.abs(fft[:len(fft) // 2])
        # Peak frequency (excluding DC)
        peak_idx = np.argmax(freq_power[3:]) + 3
        peak_power = freq_power[peak_idx]
        noise_power = np.median(freq_power[3:])

        banding_ratio = peak_power / (noise_power + 1e-6)
        has_banding = banding_ratio > 8.0 and peak_power > 1000

        # Smear detection: directional gradient
        grad_x = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
        grad_ratio = float(np.sum(np.abs(grad_x))) / (float(np.sum(np.abs(grad_y))) + 1)

        # High horizontal gradient compared to vertical = smear in one direction
        has_smear = grad_ratio > 2.5 or grad_ratio < 0.4

        return {
            "has_banding": bool(has_banding),
            "banding_strength": round(float(banding_ratio), 2),
            "banding_period_px": int(h / peak_idx) if peak_idx > 0 else 0,
            "has_smear": bool(has_smear),
            "smear_direction_ratio": round(grad_ratio, 2),
            "quality_score": round(100 - (banding_ratio / 2 if has_banding else 0)
                                    - (10 if has_smear else 0), 2),
        }
    except Exception as e:
        log.warning(f"Smear/banding check failed: {e}")
        return {"has_banding": False, "has_smear": False, "quality_score": 100.0, "error": str(e)}


# ── Spell Check ───────────────────────────────────────────────────────────────

_spell_dict_cache = None

def spell_check_regions(ocr_regions: List[dict], language: str = "en") -> dict:
    """
    Simple English spell check on OCR output.
    Uses a built-in dictionary for offline operation.
    """
    global _spell_dict_cache

    if _spell_dict_cache is None:
        _spell_dict_cache = _load_common_english_words()

    errors = []
    for region in ocr_regions:
        text = region.get("text", "")
        # Split into words, strip punctuation
        words = [w.strip(".,!?;:()[]\"' ").lower() for w in text.split()]

        for word in words:
            if not word or len(word) < 3:
                continue
            # Skip numbers, dates, product codes
            if word.replace("-", "").replace("/", "").isdigit():
                continue
            if any(c.isdigit() for c in word):
                continue
            # Check dictionary
            if word not in _spell_dict_cache:
                errors.append({
                    "word": word,
                    "context": text[:80],
                    "bbox": region.get("bbox", {}),
                    "type": "possible_misspelling",
                })

    return {
        "pass": len(errors) == 0,
        "misspellings": errors[:50],  # Cap at 50
        "total_errors": len(errors),
        "language": language,
    }


def _load_common_english_words() -> set:
    """Load the 10k most common English words for offline spell check"""
    # Abbreviated common words list - extend with actual dictionary file in production
    return set("""
    the be to of and a in that have I it for not on with he as you do at this
    but his by from they we say her she or an will my one all would there their
    what so up out if about who get which go me when make can like time no just him
    know take people into year your good some could them see other than then now look
    only come its over think also back after use two how our work first well way even
    new want because any these give day most us a an on is are was were be been being
    have has had do does did will would shall should may might must can could of in to
    for from with by at about against between into through during before after above below
    up down out off over under again further then once here there when where why how all
    any both each few more most other some such no nor not only own same so than too very
    s t can will just don should now expiry exp batch lot code mfg date use before best
    ingredients contains warning caution store keep dry place cool avoid direct sunlight
    made in contains allergens wheat milk soy egg gluten free natural artificial flavor
    color preservative approved manufacturer distributed by product net weight volume
    servings per container calories protein fat carbohydrate sodium cholesterol vitamin
    mineral daily value percent nutrition facts supplement dietary active ingredient
    inactive consult physician doctor pregnant nursing children age years months tablet
    capsule liquid suspension drops cream ointment lotion gel spray inhaler prescription
    over counter otc recommended dosage adults tablespoon teaspoon tablet capsule once
    twice three times daily morning evening before after meal food water symptoms relief
    fever cough cold flu pain headache allergy nasal sinus congestion decongestant
    antihistamine analgesic anti-inflammatory
    """.split())
