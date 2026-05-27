"""
Greenpack Pro v3.0 — Expiry Date Validation

Detects expiry/best-before dates in label text and verifies:
  - Format consistency (e.g. matches expected pattern)
  - Date validity (not in the past, reasonable future)
  - Date matches between final design and trial proof
  - Manufacturing date < expiry date (logical consistency)

Supports common formats:
  - DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, YYYY-MM-DD
  - DD/MM/YY, MM/YY
  - DD MMM YYYY (e.g., "15 Jan 2026")
  - MMM YYYY (e.g., "Dec 2026")
  - Numeric-only (e.g., "20251231" YYYYMMDD)
  - Arabic-numeral dates as well (for MENA market)

Plus prefix labels:
  - EXP, EXPIRY, EXPIRES, USE BY, USE BEFORE, BEST BEFORE, BB,
    BEFORE END OF, MFG, MANUFACTURED, BATCH
"""
import logging
import re
from datetime import datetime, timedelta
from typing import List

log = logging.getLogger(__name__)


# ── Date Pattern Library ──────────────────────────────────────────────────────

DATE_LABEL_KEYWORDS = [
    "expiry", "exp", "expires", "expire",
    "use by", "use before", "best before", "bb", "bbe",
    "before end of", "bes",
    "mfg", "manufactured", "manuf", "made on", "production date",
    "best by", "lot", "batch", "lot/batch",
]

# Each pattern: (regex, format_string for strptime, description)
DATE_PATTERNS = [
    # YYYY-MM-DD
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y-%m-%d", "YYYY-MM-DD"),
    # YYYY/MM/DD
    (r"\b(\d{4})/(\d{2})/(\d{2})\b", "%Y/%m/%d", "YYYY/MM/DD"),
    # DD/MM/YYYY (most common in MENA/Europe)
    (r"\b(\d{2})/(\d{2})/(\d{4})\b", "%d/%m/%Y", "DD/MM/YYYY"),
    # DD-MM-YYYY
    (r"\b(\d{2})-(\d{2})-(\d{4})\b", "%d-%m-%Y", "DD-MM-YYYY"),
    # DD.MM.YYYY
    (r"\b(\d{2})\.(\d{2})\.(\d{4})\b", "%d.%m.%Y", "DD.MM.YYYY"),
    # DD/MM/YY
    (r"\b(\d{2})/(\d{2})/(\d{2})\b", "%d/%m/%y", "DD/MM/YY"),
    # MM/YYYY
    (r"\b(\d{2})/(\d{4})\b", "%m/%Y", "MM/YYYY"),
    # MM-YYYY
    (r"\b(\d{2})-(\d{4})\b", "%m-%Y", "MM-YYYY"),
    # MM/YY
    (r"\b(\d{2})/(\d{2})\b", "%m/%y", "MM/YY"),
    # DD MMM YYYY (e.g., 15 JAN 2026)
    (r"\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})\b",
     "%d %b %Y", "DD MMM YYYY"),
    # MMM YYYY (e.g., DEC 2026)
    (r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})\b",
     "%b %Y", "MMM YYYY"),
    # YYYYMMDD
    (r"\b(\d{8})\b", "%Y%m%d", "YYYYMMDD"),
]


def find_dates_in_text(text: str) -> List[dict]:
    """Find all date strings in a text"""
    if not text:
        return []

    found = []
    text_upper = text.upper()
    seen_positions = set()

    for pattern, fmt, fmt_desc in DATE_PATTERNS:
        for match in re.finditer(pattern, text_upper):
            start = match.start()
            if start in seen_positions:
                continue
            seen_positions.add(start)

            raw = match.group(0)
            try:
                dt = datetime.strptime(raw, fmt)
                # Sanity check: year 1990-2100
                if dt.year < 1990 or dt.year > 2100:
                    continue

                # Check label context (50 chars before)
                context = text_upper[max(0, start - 50):start]
                label_type = _detect_label_type(context)

                found.append({
                    "raw": raw,
                    "datetime": dt,
                    "iso": dt.strftime("%Y-%m-%d"),
                    "format": fmt_desc,
                    "label_type": label_type,
                    "context": context.strip()[-30:],
                })
            except ValueError:
                continue

    return found


def _detect_label_type(context: str) -> str:
    """Detect what kind of date is being labeled (expiry vs mfg vs unknown)"""
    ctx = context.lower()
    if any(kw in ctx for kw in ["exp", "expiry", "expires", "use by",
                                  "best before", "bbe", "bb"]):
        return "expiry"
    if any(kw in ctx for kw in ["mfg", "manufactured", "made on",
                                  "production", "manuf"]):
        return "manufacturing"
    if any(kw in ctx for kw in ["lot", "batch"]):
        return "batch"
    return "unknown"


# ── Validation ────────────────────────────────────────────────────────────────

def validate_expiry_dates(trial_text: str, final_text: str = None) -> dict:
    """
    Validate dates in trial text.

    Args:
        trial_text: Combined OCR text from the trial proof
        final_text: Combined OCR text from the final design (for cross-check)

    Returns:
        dict with:
          - dates_found_trial: list of detected dates
          - dates_found_final: list of detected dates
          - dates_match: bool — do trial and final have same dates?
          - format_valid: bool
          - dates_in_past: list of dates that are already in the past
          - dates_in_past_count: int
          - logical_issues: list of e.g. mfg > expiry
          - all_passed: bool
    """
    today = datetime.now()
    trial_dates = find_dates_in_text(trial_text or "")
    final_dates = find_dates_in_text(final_text or "") if final_text else []

    # Check for past dates (critical)
    expired_dates = [d for d in trial_dates if d["datetime"] < today
                     and d["label_type"] in ("expiry", "unknown")]

    # Check that mfg date < expiry date (logical consistency)
    mfg_dates = [d for d in trial_dates if d["label_type"] == "manufacturing"]
    exp_dates = [d for d in trial_dates if d["label_type"] == "expiry"]

    logical_issues = []
    if mfg_dates and exp_dates:
        latest_mfg = max(mfg_dates, key=lambda d: d["datetime"])
        earliest_exp = min(exp_dates, key=lambda d: d["datetime"])
        if latest_mfg["datetime"] >= earliest_exp["datetime"]:
            logical_issues.append({
                "type": "mfg_after_expiry",
                "description": f"Manufacturing date {latest_mfg['iso']} is on/after expiry {earliest_exp['iso']}",
            })

    # Check that expiry is reasonable future (within 5 years)
    far_future_threshold = today + timedelta(days=365 * 10)
    for d in exp_dates:
        if d["datetime"] > far_future_threshold:
            logical_issues.append({
                "type": "expiry_far_future",
                "description": f"Expiry {d['iso']} is more than 10 years in future (typo?)",
            })

    # Cross-check trial vs final
    dates_match = True
    date_diffs = []
    if final_dates and trial_dates:
        # For each labeled type, compare
        for label_type in ["expiry", "manufacturing"]:
            f_typed = [d for d in final_dates if d["label_type"] == label_type]
            t_typed = [d for d in trial_dates if d["label_type"] == label_type]
            if f_typed and t_typed:
                # Compare first of each
                f_d = f_typed[0]["datetime"]
                t_d = t_typed[0]["datetime"]
                if f_d != t_d:
                    dates_match = False
                    date_diffs.append({
                        "label_type": label_type,
                        "final_date": f_typed[0]["iso"],
                        "trial_date": t_typed[0]["iso"],
                        "description": f"{label_type.title()} date mismatch: "
                                       f"final shows {f_typed[0]['iso']} but trial shows {t_typed[0]['iso']}",
                    })

    # Format consistency: all dates use the same format?
    if trial_dates:
        formats_used = set(d["format"] for d in trial_dates)
        format_consistent = len(formats_used) == 1
    else:
        format_consistent = True

    return {
        "dates_found_trial": [
            {k: v.strftime("%Y-%m-%d %H:%M") if isinstance(v, datetime) else v
             for k, v in d.items()} for d in trial_dates
        ],
        "dates_found_final": [
            {k: v.strftime("%Y-%m-%d %H:%M") if isinstance(v, datetime) else v
             for k, v in d.items()} for d in final_dates
        ],
        "trial_date_count": len(trial_dates),
        "final_date_count": len(final_dates),
        "dates_match": dates_match,
        "date_differences": date_diffs,
        "format_valid": format_consistent,
        "formats_used": list(set(d["format"] for d in trial_dates)) if trial_dates else [],
        "expired_dates": [d["iso"] for d in expired_dates],
        "dates_in_past_count": len(expired_dates),
        "logical_issues": logical_issues,
        "all_passed": (
            len(expired_dates) == 0
            and len(logical_issues) == 0
            and format_consistent
            and dates_match
        ),
    }
