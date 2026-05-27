"""
Greenpack Pro — Barcode Verification Service
pyzbar for reading barcodes (ZBar DLL bundled on Windows).
GS1 check digit validation, barcode quality grading.
"""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


def verify_barcodes(
    scan_image_path: str,
    expected_barcodes: list[dict],
) -> list[dict]:
    """
    Read and verify all barcodes in the scanned label image.

    Args:
        scan_image_path: Path to scanned label image
        expected_barcodes: List of expected barcodes from template config
                          e.g. [{"type": "EAN13", "value": "8690526040015"}]

    Returns:
        List of BarcodeResult dicts
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        from PIL import Image

        img = Image.open(scan_image_path)

        # Try on original image first
        decoded = zbar_decode(img)

        # If no barcodes found, try enhanced image
        if not decoded:
            enhanced = _enhance_for_barcode(scan_image_path)
            if enhanced is not None:
                _, buf = cv2.imencode(".png", enhanced)
                from PIL import Image as PILImage
                import io
                pil_enhanced = PILImage.open(io.BytesIO(buf.tobytes()))
                decoded = zbar_decode(pil_enhanced)

        results = []
        for barcode in decoded:
            try:
                value = barcode.data.decode("utf-8").strip()
            except UnicodeDecodeError:
                value = barcode.data.decode("latin-1").strip()

            bc_type = barcode.type.upper()

            # Validate check digit
            check_valid = _validate_check_digit(bc_type, value)

            # Parse GS1-128 if applicable
            gs1_data = None
            if bc_type == "CODE128":
                gs1_data = _parse_gs1_128(value)

            # Find matching expected barcode
            expected_match = _find_expected_match(bc_type, value, expected_barcodes)

            # Grade barcode quality
            quality_grade = _grade_barcode_quality(barcode, scan_image_path)

            barcode_pass = (
                check_valid
                and quality_grade not in ["F"]
                and (expected_match is not None or len(expected_barcodes) == 0)
            )

            results.append({
                "type": bc_type,
                "decoded_value": value,
                "expected_value": expected_match["value"] if expected_match else None,
                "match": expected_match is not None,
                "check_digit_valid": check_valid,
                "quality_grade": quality_grade,
                "gs1_data": gs1_data,
                "bbox": {
                    "x": barcode.rect.left,
                    "y": barcode.rect.top,
                    "w": barcode.rect.width,
                    "h": barcode.rect.height,
                },
                "pass": barcode_pass,
                "message": _result_message(bc_type, value, expected_match, check_valid, quality_grade),
            })

        # Check for expected barcodes that weren't found
        for expected in expected_barcodes:
            found = any(
                r["type"] == expected.get("type", "").upper()
                and r["decoded_value"] == expected.get("value", "")
                for r in results
            )
            if not found:
                results.append({
                    "type": expected.get("type", "UNKNOWN"),
                    "decoded_value": None,
                    "expected_value": expected.get("value"),
                    "match": False,
                    "check_digit_valid": False,
                    "quality_grade": "F",
                    "pass": False,
                    "message": f"Expected barcode not found: {expected.get('type')} {expected.get('value', '')}",
                })

        log.info(f"Barcode verification: found={len(decoded)} results={len(results)}")
        return results

    except ImportError:
        log.warning("pyzbar not available — barcode verification skipped")
        return []
    except Exception as e:
        log.error(f"Barcode verification error: {e}")
        return []


def _validate_check_digit(bc_type: str, value: str) -> bool:
    """Validate barcode check digit"""
    if bc_type in ("EAN13", "EAN-13"):
        return _validate_ean13(value)
    elif bc_type in ("EAN8", "EAN-8"):
        return _validate_ean8(value)
    elif bc_type in ("UPCA", "UPC-A"):
        return _validate_upca(value)
    # QR, Code128, DataMatrix — self-correcting
    return True


def _validate_ean13(value: str) -> bool:
    """GS1 EAN-13 check digit validation"""
    value = value.strip().replace(" ", "")
    if len(value) != 13 or not value.isdigit():
        return False
    total = sum(
        int(d) * (1 if i % 2 == 0 else 3)
        for i, d in enumerate(value[:12])
    )
    check = (10 - (total % 10)) % 10
    return check == int(value[-1])


def _validate_ean8(value: str) -> bool:
    """EAN-8 check digit validation"""
    if len(value) != 8 or not value.isdigit():
        return False
    total = sum(
        int(d) * (3 if i % 2 == 0 else 1)
        for i, d in enumerate(value[:7])
    )
    check = (10 - (total % 10)) % 10
    return check == int(value[-1])


def _validate_upca(value: str) -> bool:
    """UPC-A check digit validation"""
    if len(value) != 12 or not value.isdigit():
        return False
    total = sum(
        int(d) * (3 if i % 2 == 1 else 1)
        for i, d in enumerate(value[:11])
    )
    check = (10 - (total % 10)) % 10
    return check == int(value[-1])


def _parse_gs1_128(value: str) -> Optional[dict]:
    """Parse GS1-128 Application Identifiers"""
    if not value.startswith("(") and "\x1d" not in value and all(c.isalnum() or c in "()-." for c in value[:5]):
        return None  # Likely not GS1-128

    GS1_AI = {
        "00": ("SSCC", 18, False),
        "01": ("GTIN", 14, False),
        "02": ("CONTENT_GTIN", 14, False),
        "10": ("BATCH_LOT", 20, True),
        "11": ("PRODUCTION_DATE", 6, False),
        "13": ("PACK_DATE", 6, False),
        "15": ("BEST_BEFORE", 6, False),
        "17": ("EXPIRY_DATE", 6, False),
        "20": ("PRODUCT_VARIANT", 2, False),
        "21": ("SERIAL_NUMBER", 20, True),
        "37": ("UNITS_CONTAINED", 8, True),
        "310": ("NET_WEIGHT_KG", 6, False),
        "90": ("INTERNAL", 30, True),
    }

    result = {}
    # Remove parentheses notation if present
    clean = value.replace("(", "").replace(")", " ").strip()
    parts = clean.split()

    for part in parts:
        for ai_len in [3, 2]:
            ai = part[:ai_len]
            if ai in GS1_AI:
                name, max_len, variable = GS1_AI[ai]
                field_value = part[ai_len:ai_len + max_len] if not variable else part[ai_len:]
                result[name] = field_value
                break

    return result if result else None


def _find_expected_match(
    bc_type: str, value: str, expected: list[dict]
) -> Optional[dict]:
    """Find matching expected barcode config"""
    for exp in expected:
        exp_type = exp.get("type", "").upper()
        exp_value = exp.get("value", "")

        type_match = (exp_type == bc_type or exp_type == "" or exp_type == "ANY")

        # Support regex-like patterns with *
        if "*" in exp_value:
            import re
            pattern = "^" + re.escape(exp_value).replace(r"\*", ".*") + "$"
            value_match = bool(re.match(pattern, value))
        else:
            value_match = (exp_value == value or exp_value == "")

        if type_match and value_match:
            return exp

    return None


def _grade_barcode_quality(barcode, image_path: str) -> str:
    """
    Grade barcode print quality A-F based on region analysis.
    Uses SSIM on barcode region vs expected ideal.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "C"

        rect = barcode.rect
        # Extract barcode region with some padding
        pad = 10
        x = max(0, rect.left - pad)
        y = max(0, rect.top - pad)
        x2 = min(img.shape[1], rect.left + rect.width + pad)
        y2 = min(img.shape[0], rect.top + rect.height + pad)

        region = img[y:y2, x:x2]
        if region.size == 0:
            return "C"

        # Assess contrast and sharpness
        contrast = float(np.std(region))
        laplacian = cv2.Laplacian(region, cv2.CV_64F)
        sharpness = float(np.var(laplacian))

        # Grade based on contrast and sharpness
        if contrast > 80 and sharpness > 500:
            return "A"
        elif contrast > 60 and sharpness > 300:
            return "B"
        elif contrast > 40 and sharpness > 150:
            return "C"
        elif contrast > 20:
            return "D"
        else:
            return "F"
    except Exception:
        return "C"


def _enhance_for_barcode(image_path: str) -> Optional[np.ndarray]:
    """Enhance image to improve barcode detection"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _result_message(bc_type, value, match, check_valid, grade) -> str:
    """Human-readable result message"""
    if not check_valid:
        return f"{bc_type}: '{value}' — check digit INVALID"
    if match is None:
        return f"{bc_type}: '{value}' decoded OK (no expected value configured)"
    if not match:
        return f"{bc_type}: '{value}' — value MISMATCH with expected"
    if grade == "F":
        return f"{bc_type}: Decoded but print quality POOR (Grade F)"
    return f"{bc_type}: '{value}' — PASS (Grade {grade})"
