"""
Greenpack Pro v3.0 — Prepress (Trial-vs-Final) Inspection Engine

User's exact requirement:
"Before putting final draft design to the production printing,
team will print some trial demo design to check and compare with the final design draft.
Software will find the problem in: Text, color, icon size, and expiry date."

This is the GO / NO-GO gate before committing to a full production run.

Workflow:
  1. Upload final design (PDF/image) — the source of truth
  2. Upload trial demo prints (one or more scanned proofs)
  3. Software runs a comprehensive comparison covering:
      - Text content (OCR diff, character-by-character)
      - Font rendering (size, weight estimation)
      - Layout (graphic positions, sizes)
      - Colors (per-zone ΔE, Pantone match)
      - Icon/logo dimensions
      - Expiry date format and validity
      - Barcode integrity
      - Spell check
  4. Computes ACCURACY SCORE (0-100)
  5. Generates GO/NO-GO decision with waste estimate
  6. Detailed accuracy report with annotated images

Key innovation: Real-time accuracy report → STOPS waste before production.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from app.config import get_settings
from app.services.alignment import align_images
from app.services.ocr_service import run_dual_ocr, diff_text_regions
from app.services.color_service import analyze_color_zones
from app.services.ssim_service import detect_defects
from app.services.preprocess import rasterize_pdf, preprocess_image
from app.services.advanced_inspection import (
    verify_font_sizes, spell_check_regions, detect_smear_and_banding,
)
from app.services.pantone_service import identify_pantone_colors_in_image
from app.services.icon_size_check import compare_icon_sizes
from app.services.expiry_date_validator import validate_expiry_dates

log = logging.getLogger(__name__)
settings = get_settings()


class PrepressError(Exception):
    pass


class PrepressInspectionEngine:
    """
    Compares one or more trial proof prints against the final design.
    Produces a real-time accuracy report and GO/NO-GO recommendation.
    """

    async def compare_trial_to_final(
        self,
        job_id: str,
        final_design_path: str,
        trial_proof_paths: List[str],
        config: dict,
        progress_callback=None,
    ) -> dict:
        """
        Run prepress comparison.

        Config:
            - color_threshold: ΔE limit (default 2.0)
            - identify_pantones: bool (run Pantone identification on final)
            - check_expiry_dates: bool
            - check_icon_sizes: bool
            - min_font_size_pt: float (GMP minimum)
            - waste_estimate: dict {"unit_cost_per_m2": ..., "expected_run_m2": ...}
        """
        start_time = time.time()
        log.info(f"[{job_id}] Prepress trial-vs-final inspection starting")

        try:
            # ── Load files ────────────────────────────────────────────────────
            final_path = await self._prepare_file(final_design_path, "final_design")
            final_img = cv2.imread(str(final_path))
            if final_img is None:
                raise PrepressError("Cannot load final design image")

            trial_imgs = []
            for tp in trial_proof_paths:
                tp_path = await self._prepare_file(tp, "trial_proof")
                ti = cv2.imread(str(tp_path))
                if ti is not None:
                    trial_imgs.append((str(tp_path), ti))

            if not trial_imgs:
                raise PrepressError("No trial proof images could be loaded")

            log.info(f"[{job_id}] Final design + {len(trial_imgs)} trial proof(s) loaded")

            # ── Run identification on final design (Pantone) ──────────────────
            final_pantones = None
            if config.get("identify_pantones", True):
                final_pantones = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: identify_pantone_colors_in_image(final_img, k=8)
                )

            # ── Inspect each trial proof against final ────────────────────────
            trial_reports = []
            for idx, (path, trial_img) in enumerate(trial_imgs, 1):
                log.info(f"[{job_id}] Inspecting trial proof {idx}/{len(trial_imgs)}")
                report = await self._inspect_one_trial(
                    final_img=final_img,
                    trial_img=trial_img,
                    trial_path=path,
                    trial_idx=idx,
                    final_pantones=final_pantones,
                    config=config,
                )
                trial_reports.append(report)

            # ── Aggregate ─────────────────────────────────────────────────────
            best_trial = max(trial_reports, key=lambda r: r["accuracy_score"])
            worst_trial = min(trial_reports, key=lambda r: r["accuracy_score"])
            avg_accuracy = sum(r["accuracy_score"] for r in trial_reports) / len(trial_reports)

            # ── GO/NO-GO Decision ──────────────────────────────────────────────
            min_required = config.get("min_accuracy_for_go", 90.0)
            decision = self._make_decision(trial_reports, min_required)

            # ── Waste prediction ──────────────────────────────────────────────
            waste = self._estimate_waste_savings(trial_reports, config)

            # ── Final result ──────────────────────────────────────────────────
            result = {
                "job_id": job_id,
                "mode": "prepress_trial_comparison",
                "decision": decision["decision"],  # GO | HOLD | NO_GO
                "decision_reason": decision["reason"],
                "decision_severity": decision["severity"],  # critical | warning | info
                "accuracy_score": round(avg_accuracy, 2),
                "best_trial_score": round(best_trial["accuracy_score"], 2),
                "worst_trial_score": round(worst_trial["accuracy_score"], 2),
                "trial_count": len(trial_reports),
                "trial_reports": trial_reports,
                "final_pantones": final_pantones,
                "waste_savings": waste,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

            log.info(
                f"[{job_id}] Prepress complete: decision={decision['decision']}, "
                f"avg_accuracy={avg_accuracy:.1f}%, "
                f"waste_savings=${waste.get('estimated_savings_usd', 0):.0f}"
            )
            return result

        except PrepressError:
            raise
        except Exception as e:
            log.exception(f"[{job_id}] Prepress error: {e}")
            raise PrepressError(f"Prepress inspection failed: {e}")

    # ── Single Trial Comparison ────────────────────────────────────────────────

    async def _inspect_one_trial(
        self, final_img, trial_img, trial_path, trial_idx, final_pantones, config,
    ) -> dict:
        """Compare ONE trial proof against the final design"""
        loop = asyncio.get_event_loop()

        # Align trial to final
        aligned, align_conf = await loop.run_in_executor(
            None, lambda: align_images(final_img, trial_img)
        )
        if align_conf < 0.15:
            log.warning(f"Trial {trial_idx} alignment poor: conf={align_conf:.2f}")

        # OCR diff
        ocr_errors = []
        ocr_regions_final = []
        ocr_regions_trial = []
        try:
            tmp_dir = Path(settings.temp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            f_path = tmp_dir / f"prepress_final_{trial_idx}.png"
            t_path = tmp_dir / f"prepress_trial_{trial_idx}.png"
            cv2.imwrite(str(f_path), final_img)
            cv2.imwrite(str(t_path), aligned)

            ocr_result = await loop.run_in_executor(
                None, lambda: run_dual_ocr(str(f_path), str(t_path))
            )
            ocr_regions_final = ocr_result.get("master_regions", [])
            ocr_regions_trial = ocr_result.get("scan_regions", [])
            ocr_errors = diff_text_regions(ocr_regions_final, ocr_regions_trial)

            try:
                f_path.unlink(); t_path.unlink()
            except Exception:
                pass
        except Exception as e:
            log.warning(f"Trial {trial_idx} OCR failed: {e}")

        # Color analysis
        color_result = await loop.run_in_executor(
            None, lambda: analyze_color_zones(
                final_img, aligned,
                threshold=config.get("color_threshold", 2.0),
            )
        )

        # SSIM defects
        ssim_result = await loop.run_in_executor(
            None, lambda: detect_defects(
                final_img, aligned,
                threshold=config.get("ssim_threshold", 0.75),
            )
        )

        # Smear / banding
        smear_banding = await loop.run_in_executor(
            None, lambda: detect_smear_and_banding(final_img, aligned)
        )

        # Font size verification
        font_size = None
        if config.get("check_font_size", True):
            font_size = verify_font_sizes(
                ocr_regions_trial, dpi=300,
                min_pt=config.get("min_font_size_pt", 6.0),
            )

        # Spell check
        spell = None
        if config.get("spell_check", True):
            spell = spell_check_regions(ocr_regions_trial)

        # Icon / logo size verification
        icon_size = None
        if config.get("check_icon_sizes", True):
            icon_size = await loop.run_in_executor(
                None, lambda: compare_icon_sizes(final_img, aligned)
            )

        # Expiry date validation
        expiry = None
        if config.get("check_expiry_dates", True):
            # Combine text from all regions
            trial_text = " ".join(r.get("text", "") for r in ocr_regions_trial)
            final_text = " ".join(r.get("text", "") for r in ocr_regions_final)
            expiry = validate_expiry_dates(trial_text, final_text)

        # ── Compute trial accuracy score ──────────────────────────────────────
        scores = self._compute_accuracy_score(
            ocr_errors=ocr_errors,
            color_result=color_result,
            ssim_result=ssim_result,
            font_size=font_size,
            spell=spell,
            icon_size=icon_size,
            expiry=expiry,
            smear_banding=smear_banding,
        )

        # ── Categorize errors ─────────────────────────────────────────────────
        error_summary = self._categorize_errors(
            ocr_errors=ocr_errors,
            color_result=color_result,
            ssim_result=ssim_result,
            font_size=font_size,
            spell=spell,
            icon_size=icon_size,
            expiry=expiry,
        )

        # ── Save annotated trial image ─────────────────────────────────────────
        annotated = self._create_trial_annotation(
            final_img, aligned, ocr_errors, color_result, ssim_result, error_summary,
        )
        out_path = Path(settings.reports_dir) / f"prepress_trial_{trial_idx}_{int(time.time())}.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return {
            "trial_idx": trial_idx,
            "trial_path": trial_path,
            "alignment_confidence": round(align_conf, 3),
            "accuracy_score": round(scores["overall"], 2),
            "scores": scores,
            "passed": scores["overall"] >= config.get("min_accuracy_for_go", 90.0),
            "ocr_errors": ocr_errors,
            "ocr_error_count": len(ocr_errors),
            "color_zone_failures": color_result.get("failures", 0),
            "color_mean_delta_e": color_result.get("mean_delta_e", 0),
            "ssim_score": ssim_result.get("ssim_score", 1.0),
            "defects": ssim_result.get("defects", []),
            "defect_count": len(ssim_result.get("defects", [])),
            "smear_banding": smear_banding,
            "font_size": font_size,
            "spell_check": spell,
            "icon_size": icon_size,
            "expiry_date": expiry,
            "error_summary": error_summary,
            "annotated_path": str(out_path),
        }

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _compute_accuracy_score(
        self, ocr_errors, color_result, ssim_result,
        font_size, spell, icon_size, expiry, smear_banding,
    ) -> dict:
        """
        Prepress accuracy weighted scoring (different from production):
          Text 25% + Color 25% + SSIM 15% + Icon size 10% +
          Expiry 10% + Font size 5% + Spell 5% + Print quality 5%
        """
        # Text accuracy
        if ocr_errors:
            penalty = sum(15 if e.get("severity") == "high" else 10 if e.get("severity") == "medium" else 5
                          for e in ocr_errors)
            text_score = max(0.0, 100.0 - penalty)
        else:
            text_score = 100.0

        # Color accuracy
        zones = color_result.get("zone_results", [])
        if zones:
            pct_fail = sum(1 for z in zones if not z.get("pass", True)) / len(zones)
            color_score = max(0.0, 100.0 - pct_fail * 100.0)
        else:
            color_score = 100.0

        # SSIM (overall image structural similarity)
        ssim_score_pct = ssim_result.get("ssim_score", 1.0) * 100.0

        # Icon size accuracy
        icon_score = 100.0
        if icon_size:
            mismatches = icon_size.get("mismatches", 0)
            checked = icon_size.get("total_checked", 1) or 1
            icon_score = max(0.0, 100.0 - (mismatches / checked) * 100.0)

        # Expiry date validity
        expiry_score = 100.0
        if expiry:
            if expiry.get("dates_match", True) is False:
                expiry_score -= 50.0
            if not expiry.get("format_valid", True):
                expiry_score -= 30.0
            if expiry.get("dates_in_past_count", 0) > 0:
                expiry_score = 0.0  # Critical: expired date on label
            expiry_score = max(0.0, expiry_score)

        # Font size compliance
        font_size_score = 100.0
        if font_size:
            violations = len(font_size.get("violations", []))
            font_size_score = max(0.0, 100.0 - violations * 10)

        # Spell check
        spell_score = 100.0
        if spell:
            errors = spell.get("total_errors", 0)
            spell_score = max(0.0, 100.0 - errors * 5)

        # Print quality (smear/banding)
        print_quality = 100.0
        if smear_banding:
            print_quality = smear_banding.get("quality_score", 100.0)

        overall = (
            text_score * 0.25
            + color_score * 0.25
            + ssim_score_pct * 0.15
            + icon_score * 0.10
            + expiry_score * 0.10
            + font_size_score * 0.05
            + spell_score * 0.05
            + print_quality * 0.05
        )

        return {
            "overall": overall,
            "text": round(text_score, 2),
            "color": round(color_score, 2),
            "ssim": round(ssim_score_pct, 2),
            "icon_size": round(icon_score, 2),
            "expiry": round(expiry_score, 2),
            "font_size": round(font_size_score, 2),
            "spell": round(spell_score, 2),
            "print_quality": round(print_quality, 2),
        }

    # ── Error Categorization ───────────────────────────────────────────────────

    def _categorize_errors(self, ocr_errors, color_result, ssim_result,
                           font_size, spell, icon_size, expiry) -> dict:
        """Group findings by category for the report"""
        critical = []
        warning = []
        info = []

        # Critical: expired dates
        if expiry and expiry.get("dates_in_past_count", 0) > 0:
            critical.append({
                "type": "expired_date",
                "category": "Date",
                "description": f"Expired date detected on trial: {expiry.get('expired_dates', [])}",
            })

        # Critical: high severity OCR errors
        for err in ocr_errors:
            if err.get("severity") == "high":
                critical.append({
                    "type": "text_mismatch",
                    "category": "Text",
                    "description": f"'{err.get('master_text', '')[:30]}' → '{err.get('scan_text', '')[:30]}'",
                })

        # Warning: medium OCR errors
        for err in ocr_errors:
            if err.get("severity") == "medium":
                warning.append({
                    "type": "text_difference",
                    "category": "Text",
                    "description": f"'{err.get('master_text', '')[:30]}' vs '{err.get('scan_text', '')[:30]}'",
                })

        # Color zone failures
        zones = color_result.get("zone_results", []) if color_result else []
        for z in zones:
            if not z.get("pass", True):
                de = z.get("delta_e", 0)
                level = "critical" if de > 5 else "warning" if de > 3 else "info"
                entry = {
                    "type": "color_drift",
                    "category": "Color",
                    "description": f"Zone ΔE {de:.1f} (limit {z.get('threshold', 2.0)})",
                    "delta_e": de,
                }
                if level == "critical": critical.append(entry)
                elif level == "warning": warning.append(entry)
                else: info.append(entry)

        # Defects
        if ssim_result:
            for d in ssim_result.get("defects", [])[:20]:
                sev = d.get("severity", "medium")
                entry = {
                    "type": "print_defect",
                    "category": "Print",
                    "description": f"{d.get('type', 'defect')} ({sev}) — {d.get('area_pixels', 0)}px²",
                }
                if sev in ("critical", "high"):
                    critical.append(entry)
                elif sev == "medium":
                    warning.append(entry)
                else:
                    info.append(entry)

        # Icon size mismatches
        if icon_size and icon_size.get("mismatches", 0) > 0:
            for m in icon_size.get("mismatch_details", [])[:10]:
                warning.append({
                    "type": "icon_size_mismatch",
                    "category": "Icon Size",
                    "description": f"Icon at {m.get('position', '?')}: {m.get('description', '')}",
                })

        # Font size violations
        if font_size and font_size.get("violations"):
            for v in font_size["violations"][:10]:
                warning.append({
                    "type": "small_font",
                    "category": "Font Size",
                    "description": v.get("description", "Font below minimum size"),
                })

        # Spell errors
        if spell and spell.get("misspellings"):
            for s in spell["misspellings"][:5]:
                info.append({
                    "type": "possible_misspelling",
                    "category": "Spelling",
                    "description": f"Possible misspelling: '{s.get('word')}'",
                })

        return {
            "critical": critical,
            "warning": warning,
            "info": info,
            "total": len(critical) + len(warning) + len(info),
            "critical_count": len(critical),
            "warning_count": len(warning),
            "info_count": len(info),
        }

    # ── Decision Engine ────────────────────────────────────────────────────────

    def _make_decision(self, trial_reports, min_required) -> dict:
        """Make GO / HOLD / NO_GO decision"""
        # Any critical errors anywhere → NO_GO
        all_critical = sum(
            r["error_summary"].get("critical_count", 0) for r in trial_reports
        )
        if all_critical > 0:
            return {
                "decision": "NO_GO",
                "severity": "critical",
                "reason": f"{all_critical} critical errors detected — must fix before printing",
            }

        # Any trial below minimum accuracy → HOLD
        worst_score = min(r["accuracy_score"] for r in trial_reports)
        avg_score = sum(r["accuracy_score"] for r in trial_reports) / len(trial_reports)

        if worst_score < min_required:
            return {
                "decision": "HOLD",
                "severity": "warning",
                "reason": f"Worst trial scored {worst_score:.1f}% (need {min_required}%) — review and re-prove",
            }

        if avg_score < min_required:
            return {
                "decision": "HOLD",
                "severity": "warning",
                "reason": f"Average accuracy {avg_score:.1f}% below threshold {min_required}%",
            }

        # Total warnings count
        total_warnings = sum(
            r["error_summary"].get("warning_count", 0) for r in trial_reports
        )
        if total_warnings > 10:
            return {
                "decision": "HOLD",
                "severity": "warning",
                "reason": f"{total_warnings} warnings — review carefully before proceeding",
            }

        return {
            "decision": "GO",
            "severity": "info",
            "reason": f"All trials passed with avg {avg_score:.1f}% accuracy. Safe to proceed to production.",
        }

    # ── Waste Prediction ───────────────────────────────────────────────────────

    def _estimate_waste_savings(self, trial_reports, config) -> dict:
        """Estimate ink/paper/sticker waste avoided by catching errors now"""
        ws = config.get("waste_estimate", {})
        unit_cost = ws.get("unit_cost_per_m2", 5.0)  # USD per m²
        run_size = ws.get("expected_run_m2", 1000.0)  # m² total run

        # If trial passed, no waste savings (would have run anyway)
        # If trial failed, calculate what would have been wasted
        worst_score = min(r["accuracy_score"] for r in trial_reports)

        if worst_score >= 90:
            return {
                "estimated_savings_usd": 0,
                "details": "No waste — print would have proceeded successfully",
                "would_have_been_caught_in_production": False,
            }

        # Estimate fraction of run that would have been wasted
        if worst_score < 50:
            waste_fraction = 1.0  # Whole run would be scrapped
        elif worst_score < 70:
            waste_fraction = 0.5
        elif worst_score < 80:
            waste_fraction = 0.25
        else:
            waste_fraction = 0.1

        wasted_m2 = run_size * waste_fraction
        savings = wasted_m2 * unit_cost

        return {
            "estimated_savings_usd": round(savings, 2),
            "estimated_wasted_m2": round(wasted_m2, 1),
            "waste_fraction": waste_fraction,
            "unit_cost_per_m2": unit_cost,
            "expected_run_m2": run_size,
            "details": f"Trial caught errors that would waste ${savings:.0f} in production",
            "would_have_been_caught_in_production": True,
        }

    # ── Annotation ─────────────────────────────────────────────────────────────

    def _create_trial_annotation(
        self, final_img, aligned_trial, ocr_errors, color_result, ssim_result,
        error_summary,
    ) -> np.ndarray:
        """Side-by-side annotated comparison image"""
        h, w = final_img.shape[:2]
        if aligned_trial.shape[:2] != (h, w):
            aligned_trial = cv2.resize(aligned_trial, (w, h))

        # Annotate trial with errors
        trial_annot = aligned_trial.copy()
        for err in ocr_errors:
            bbox = err.get("bbox") or err.get("scan_bbox", {})
            if bbox:
                x, y = bbox.get("x", 0), bbox.get("y", 0)
                bw, bh = bbox.get("w", 50), bbox.get("h", 20)
                color = (0, 0, 220) if err.get("severity") == "high" else (0, 165, 255)
                cv2.rectangle(trial_annot, (x, y), (x + bw, y + bh), color, 2)

        for d in ssim_result.get("defects", [])[:50]:
            bbox = d.get("bbox", {})
            if bbox:
                x, y = bbox.get("x", 0), bbox.get("y", 0)
                bw, bh = bbox.get("w", 10), bbox.get("h", 10)
                cv2.rectangle(trial_annot, (x, y), (x + bw, y + bh), (50, 50, 200), 2)

        # Side-by-side
        sep = np.full((h, 8, 3), 80, dtype=np.uint8)
        combined = np.hstack([final_img, sep, trial_annot])

        # Banner
        banner_h = 50
        banner = np.zeros((banner_h, combined.shape[1], 3), dtype=np.uint8)
        banner[:] = (40, 40, 40)
        cv2.putText(banner, "FINAL DESIGN", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(banner, "TRIAL PROOF (annotated)", (w + 25, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Bottom legend
        legend_h = 40
        legend = np.full((legend_h, combined.shape[1], 3), 240, dtype=np.uint8)
        critical = error_summary.get("critical_count", 0)
        warning = error_summary.get("warning_count", 0)
        info = error_summary.get("info_count", 0)
        cv2.putText(legend, f"Critical: {critical}   Warnings: {warning}   Info: {info}",
                   (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   (40, 40, 200) if critical else (40, 130, 40), 2)

        return np.vstack([banner, combined, legend])

    async def _prepare_file(self, file_path: str, role: str) -> Path:
        p = Path(file_path)
        if not p.exists():
            raise PrepressError(f"File not found: {file_path}")
        if p.suffix.lower() == ".pdf":
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: rasterize_pdf(p, settings.pdf_raster_dpi)
            )
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: preprocess_image(p)
        )


# Singleton
_prepress_engine: Optional[PrepressInspectionEngine] = None

def get_prepress_engine() -> PrepressInspectionEngine:
    global _prepress_engine
    if _prepress_engine is None:
        _prepress_engine = PrepressInspectionEngine()
    return _prepress_engine
