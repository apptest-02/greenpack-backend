"""
Greenpack Pro — Image Alignment Service
ORB keypoint matching + RANSAC homography.
Fallback to template matching if ORB fails.
"""
import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


def align_images(
    master: np.ndarray,
    scan: np.ndarray,
    min_good_matches: int = 15,
) -> tuple[np.ndarray, float]:
    """
    Align scanned label image to master coordinate space.

    Args:
        master: Master label image (numpy array BGR)
        scan: Scanned print image (numpy array BGR)
        min_good_matches: Minimum feature matches required for homography

    Returns:
        (aligned_scan, confidence_0_to_1)

    Raises:
        ValueError: If alignment completely fails
    """
    h_master, w_master = master.shape[:2]

    # Convert to grayscale for feature detection
    gray_master = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    gray_scan = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # ── Method 1: ORB Feature Matching ────────────────────────────────────────
    try:
        aligned, confidence = _orb_align(
            master, scan, gray_master, gray_scan, min_good_matches
        )
        if confidence > 0.15:
            log.debug(f"ORB alignment success: confidence={confidence:.3f}")
            return aligned, confidence
        log.warning(f"ORB alignment low confidence: {confidence:.3f}, trying SIFT")
    except Exception as e:
        log.warning(f"ORB alignment failed: {e}")

    # ── Method 2: SIFT Feature Matching (better quality, slower) ──────────────
    try:
        aligned, confidence = _sift_align(
            master, scan, gray_master, gray_scan, min_good_matches
        )
        if confidence > 0.15:
            log.debug(f"SIFT alignment success: confidence={confidence:.3f}")
            return aligned, confidence
        log.warning(f"SIFT alignment low confidence: {confidence:.3f}")
    except Exception as e:
        log.warning(f"SIFT alignment failed: {e}")

    # ── Method 3: Template Matching Fallback ───────────────────────────────────
    try:
        aligned, confidence = _template_match_align(master, scan, gray_master, gray_scan)
        log.info(f"Template match fallback: confidence={confidence:.3f}")
        return aligned, confidence
    except Exception as e:
        log.error(f"All alignment methods failed: {e}")

    # Last resort: resize scan to master dimensions
    log.warning("Using direct resize as last resort")
    resized = cv2.resize(scan, (w_master, h_master), interpolation=cv2.INTER_LANCZOS4)
    return resized, 0.05


def _orb_align(
    master, scan, gray_master, gray_scan, min_matches
) -> tuple[np.ndarray, float]:
    """ORB-based alignment"""
    orb = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
    )

    kp_m, des_m = orb.detectAndCompute(gray_master, None)
    kp_s, des_s = orb.detectAndCompute(gray_scan, None)

    if des_m is None or des_s is None or len(kp_m) < 10 or len(kp_s) < 10:
        raise ValueError("Insufficient ORB keypoints detected")

    # KNN matching with Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    all_matches = bf.knnMatch(des_s, des_m, k=2)
    good = [m for m, n in all_matches if m.distance < 0.75 * n.distance]

    if len(good) < min_matches:
        raise ValueError(f"Insufficient good ORB matches: {len(good)} < {min_matches}")

    src_pts = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography computation failed")

    h, w = master.shape[:2]
    aligned = cv2.warpPerspective(scan, H, (w, h), flags=cv2.INTER_LANCZOS4)

    inliers = np.sum(mask)
    confidence = min(inliers / 100.0, 1.0)
    return aligned, float(confidence)


def _sift_align(
    master, scan, gray_master, gray_scan, min_matches
) -> tuple[np.ndarray, float]:
    """SIFT-based alignment (higher quality than ORB)"""
    sift = cv2.SIFT_create(nfeatures=3000)

    kp_m, des_m = sift.detectAndCompute(gray_master, None)
    kp_s, des_s = sift.detectAndCompute(gray_scan, None)

    if des_m is None or des_s is None:
        raise ValueError("SIFT descriptor extraction failed")

    flann_params = dict(algorithm=1, trees=5)
    flann = cv2.FlannBasedMatcher(flann_params, {"checks": 50})
    all_matches = flann.knnMatch(des_s, des_m, k=2)
    good = [m for m, n in all_matches if m.distance < 0.75 * n.distance]

    if len(good) < min_matches:
        raise ValueError(f"Insufficient SIFT matches: {len(good)}")

    src_pts = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if H is None:
        raise ValueError("SIFT homography failed")

    h, w = master.shape[:2]
    aligned = cv2.warpPerspective(scan, H, (w, h), flags=cv2.INTER_LANCZOS4)
    confidence = min(np.sum(mask) / 80.0, 1.0)
    return aligned, float(confidence)


def _template_match_align(
    master, scan, gray_master, gray_scan
) -> tuple[np.ndarray, float]:
    """Template matching as last resort for simple alignment"""
    h_m, w_m = master.shape[:2]

    # Resize scan to master dimensions first
    resized_scan = cv2.resize(gray_scan, (w_m, h_m))

    result = cv2.matchTemplate(resized_scan, gray_master, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    # Use resized scan as-is (template matching gives similarity but not geometric transform)
    aligned_color = cv2.resize(scan, (w_m, h_m), interpolation=cv2.INTER_LANCZOS4)
    return aligned_color, float(max_val * 0.5)  # Scale down confidence
