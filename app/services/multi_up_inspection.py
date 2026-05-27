"""
Greenpack Pro v2.0 — Multi-Up Inspection Engine

Extension of InspectionEngine that processes multi-label sheets.

Workflow:
  1. Detect all N labels in scan (multi_up_detection)
  2. For each label crop:
      - Align to master
      - Run OCR diff
      - Run color ΔE
      - Run SSIM defect detection
      - Run barcode verification
      - Run registration check
      - Run die-cut edge check
      - Run mottling / uneven ink check
      - Compute per-label score
  3. Aggregate sheet-level result:
      - Total labels found vs expected
      - Per-label pass/fail
      - Sheet pass-rate
      - Defect map visualization

This is the core of the user's request.
"""
import asyncio
import gc
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings
from app.services.alignment import align_images
from app.services.ocr_service import run_dual_ocr, diff_text_regions
from app.services.color_service import analyze_color_zones
from app.services.ssim_service import detect_defects
from app.services.barcode_service import verify_barcodes
from app.services.preprocess import rasterize_pdf, preprocess_image
from app.services.multi_up_detection import detect_and_extract_labels, detect_clear_labels
from app.services.advanced_inspection import (
    check_registration_drift, check_die_cut_quality, check_mottling,
    check_braille_region, verify_font_sizes,
)

log = logging.getLogger(__name__)
settings = get_settings()


class MultiUpInspectionError(Exception):
    pass


class MultiUpInspectionEngine:
    """Main inspection engine for multi-label sheets"""

    async def inspect_sheet(
        self,
        job_id: str,
        master_path: str,
        scan_path: str,
        config: dict,
        progress_callback=None,
    ) -> dict:
        """
        Run full multi-up inspection.

        Config can include:
          - expected_count: int (for missing-label detection)
          - is_transparent: bool (clear/transparent labels)
          - color_threshold: float (ΔE, default 2.0)
          - ssim_threshold: float (default 0.75)
          - check_braille: bool
          - check_font_size: bool
          - min_font_size_pt: float (default 6.0)
          - barcode_rules: list
          - spell_check_language: str (default "en")
        """
        start_time = time.time()
        log.info(f"[{job_id}] Multi-up inspection starting")

        def progress(step: int, total: int, msg: str):
            log.info(f"[{job_id}] [{step}/{total}] {msg}")
            if progress_callback:
                try:
                    asyncio.create_task(progress_callback(job_id, step, total, msg))
                except Exception:
                    pass

        try:
            # ── Step 1: Prepare files ─────────────────────────────────────────
            progress(1, 10, "Loading master and scan")
            master_path = await self._prepare_file(master_path, "master")
            scan_path = await self._prepare_file(scan_path, "scan")

            master_img = cv2.imread(str(master_path))
            scan_img = cv2.imread(str(scan_path))

            if master_img is None or scan_img is None:
                raise MultiUpInspectionError("Cannot load master or scan image")

            # ── Step 2: Detect all labels ─────────────────────────────────────
            progress(2, 10, "Detecting individual labels on sheet")
            if config.get("is_transparent", False):
                labels_meta = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: {
                        "method": "clear_label",
                        "labels": detect_clear_labels(scan_img, master_img),
                        "count_found": 0, "count_expected": config.get("expected_count"),
                        "missing_labels": 0, "confidence": 0.7,
                        "imposition": "clear",
                        "debug_visualization": None,
                    }
                )
                labels_meta["count_found"] = len(labels_meta["labels"])
            else:
                labels_meta = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: detect_and_extract_labels(
                        scan_img, master_img,
                        expected_count=config.get("expected_count"),
                    ),
                )

            log.info(
                f"[{job_id}] Detected {labels_meta['count_found']} labels "
                f"({labels_meta['imposition']})"
            )

            if labels_meta["count_found"] == 0:
                raise MultiUpInspectionError(
                    "No labels detected in scan. Check that the scan contains labels and "
                    "is not blank, or that the master is not too different."
                )

            # ── Step 3: Per-label inspection ──────────────────────────────────
            per_label_results = []
            all_defects = []
            total_labels = len(labels_meta["labels"])

            for idx, label_info in enumerate(labels_meta["labels"], 1):
                progress(3 + int(idx * 4 / total_labels),
                         10, f"Inspecting label {idx}/{total_labels}")

                result = await self._inspect_single_label(
                    label_info=label_info,
                    master_img=master_img,
                    config=config,
                    label_idx=idx,
                )
                per_label_results.append(result)

                # Add defects to sheet-level list for visualization
                bbox = label_info["bbox"]
                for defect in result.get("defects", []):
                    # Transform defect bbox from label-local to sheet coords
                    d_copy = defect.copy()
                    d_bbox = d_copy.get("bbox", {})
                    d_copy["bbox"] = {
                        "x": d_bbox.get("x", 0) + bbox["x"],
                        "y": d_bbox.get("y", 0) + bbox["y"],
                        "w": d_bbox.get("w", 0),
                        "h": d_bbox.get("h", 0),
                    }
                    d_copy["label_id"] = f"{label_info['row']+1}-{label_info['col']+1}"
                    all_defects.append(d_copy)

            # ── Step 4: Aggregate sheet-level result ──────────────────────────
            progress(8, 10, "Aggregating results")
            passed = sum(1 for r in per_label_results if r["pass_fail"])
            failed = len(per_label_results) - passed
            pass_rate = (passed / len(per_label_results) * 100) if per_label_results else 0

            avg_score = np.mean([r["overall_score"] for r in per_label_results])

            # ── Step 5: Missing label penalty ─────────────────────────────────
            expected = config.get("expected_count")
            missing = labels_meta["missing_labels"]
            if expected and missing > 0:
                # Pass rate drops proportionally to missing labels
                total_expected = expected
                effective_passed = passed
                pass_rate = (effective_passed / total_expected * 100)
                log.info(
                    f"[{job_id}] Missing labels penalty: {missing}/{expected} missing, "
                    f"effective pass rate: {pass_rate:.1f}%"
                )

            # ── Step 6: Sheet-level pass/fail ─────────────────────────────────
            # Sheet passes only if:
            #   - All labels pass
            #   - No labels missing (if expected_count given)
            #   - Pass rate >= 95%
            sheet_pass = (
                failed == 0
                and missing == 0
                and pass_rate >= 95.0
            )

            # ── Step 7: Build annotated sheet visualization ───────────────────
            progress(9, 10, "Creating annotated visualization")
            sheet_viz = self._create_sheet_annotation(
                scan_img, labels_meta, per_label_results, all_defects
            )

            # Save annotated image
            out_dir = Path(settings.reports_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            sheet_viz_path = out_dir / f"{job_id}_sheet_annotated.jpg"
            cv2.imwrite(str(sheet_viz_path), sheet_viz, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # ── Step 8: Final assembly ────────────────────────────────────────
            progress(10, 10, "Complete")
            processing_ms = int((time.time() - start_time) * 1000)

            final_result = {
                "job_id": job_id,
                "mode": "multi_up",
                "sheet_pass": sheet_pass,
                "overall_score": round(float(avg_score), 2),
                "labels_found": labels_meta["count_found"],
                "labels_expected": labels_meta["count_expected"],
                "labels_missing": missing,
                "labels_passed": passed,
                "labels_failed": failed,
                "pass_rate": round(pass_rate, 2),
                "imposition": labels_meta["imposition"],
                "detection_method": labels_meta["method"],
                "detection_confidence": round(labels_meta["confidence"], 3),
                "per_label_results": per_label_results,
                "all_defects": all_defects,
                "sheet_annotated_path": str(sheet_viz_path),
                "processing_time_ms": processing_ms,
            }

            log.info(
                f"[{job_id}] Multi-up complete: {passed}/{total_labels} passed, "
                f"{missing} missing, {failed} failed, {processing_ms}ms"
            )
            return final_result

        except MultiUpInspectionError:
            raise
        except Exception as e:
            log.exception(f"[{job_id}] Multi-up inspection error: {e}")
            raise MultiUpInspectionError(f"Multi-up inspection failed: {e}")
        finally:
            gc.collect()

    # ── Per-Label Processing ───────────────────────────────────────────────────

    async def _inspect_single_label(
        self,
        label_info: dict,
        master_img: np.ndarray,
        config: dict,
        label_idx: int,
    ) -> dict:
        """Run complete inspection on a single detected label crop"""
        label_crop = label_info["crop"]
        label_id = f"{label_info['row']+1}-{label_info['col']+1}"

        # Align crop to master
        try:
            aligned, align_conf = align_images(master_img, label_crop)
        except Exception as e:
            log.warning(f"Label {label_id} alignment failed: {e}")
            aligned = cv2.resize(label_crop, (master_img.shape[1], master_img.shape[0]))
            align_conf = 0.1

        # OCR
        ocr_errors = []
        if align_conf > 0.15:
            try:
                # Save crops temporarily for OCR service
                temp_master = Path(settings.temp_dir) / f"label_master_{label_idx}.png"
                temp_scan = Path(settings.temp_dir) / f"label_scan_{label_idx}.png"
                temp_master.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(temp_master), master_img)
                cv2.imwrite(str(temp_scan), aligned)

                ocr_result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: run_dual_ocr(str(temp_master), str(temp_scan))
                )
                ocr_errors = diff_text_regions(
                    ocr_result.get("master_regions", []),
                    ocr_result.get("scan_regions", []),
                )

                # Cleanup
                try:
                    temp_master.unlink()
                    temp_scan.unlink()
                except Exception:
                    pass
            except Exception as e:
                log.warning(f"Label {label_id} OCR failed: {e}")

        # Color analysis
        color_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: analyze_color_zones(
                master_img, aligned,
                threshold=config.get("color_threshold", 2.0),
            )
        )

        # SSIM defect detection
        ssim_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: detect_defects(
                master_img, aligned,
                threshold=config.get("ssim_threshold", 0.75),
            )
        )

        # Advanced checks
        registration = check_registration_drift(master_img, aligned)
        die_cut = check_die_cut_quality(label_crop)
        mottling = check_mottling(aligned)

        # Barcode
        barcode_results = []
        if config.get("barcode_rules"):
            try:
                temp_crop = Path(settings.temp_dir) / f"label_bc_{label_idx}.png"
                temp_crop.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(temp_crop), label_crop)
                barcode_results = verify_barcodes(str(temp_crop), config["barcode_rules"])
                try: temp_crop.unlink()
                except: pass
            except Exception as e:
                log.warning(f"Label {label_id} barcode failed: {e}")

        # Compute per-label score
        scores = self._compute_label_scores(
            ocr_errors=ocr_errors,
            color_result=color_result,
            ssim_result=ssim_result,
            barcode_results=barcode_results,
            registration=registration,
            die_cut=die_cut,
            mottling=mottling,
        )

        return {
            "label_id": label_id,
            "row": label_info["row"],
            "col": label_info["col"],
            "bbox": label_info["bbox"],
            "overall_score": round(scores["overall"], 2),
            "pass_fail": scores["overall"] >= 75.0,
            "alignment_confidence": round(align_conf, 3),
            "detection_score": round(label_info.get("score", 0), 3),
            "ocr_score": round(scores["ocr"], 2),
            "color_score": round(scores["color"], 2),
            "ssim_score": round(scores["ssim"], 2),
            "ssim_raw": ssim_result.get("ssim_score", 0),
            "barcode_score": round(scores["barcode"], 2),
            "registration_score": round(scores["registration"], 2),
            "die_cut_score": round(scores["die_cut"], 2),
            "mottling_score": round(scores["mottling"], 2),
            "ocr_errors": ocr_errors,
            "color_results": color_result.get("zone_results", []),
            "barcode_results": barcode_results,
            "defects": ssim_result.get("defects", []),
            "registration": registration,
            "die_cut": die_cut,
            "mottling": mottling,
        }

    def _compute_label_scores(
        self, ocr_errors, color_result, ssim_result,
        barcode_results, registration, die_cut, mottling,
    ) -> dict:
        """
        Per-label weighted scoring:
          OCR 25% + Color 20% + SSIM 20% + Barcode 10% + Registration 10% + Die-cut 10% + Mottling 5%
        """
        # OCR
        penalty = 0.0
        for err in ocr_errors:
            sev = err.get("severity", "medium")
            if sev == "high": penalty += 15.0
            elif sev == "medium": penalty += 10.0
            else: penalty += 5.0
        ocr = max(0.0, 100.0 - penalty)

        # Color
        zones = color_result.get("zone_results", [])
        if zones:
            pct_fail = sum(1 for z in zones if not z.get("pass", True)) / len(zones)
            color = max(0.0, 100.0 - pct_fail * 100.0)
        else:
            color = 100.0

        # SSIM
        ssim = ssim_result.get("ssim_score", 1.0) * 100.0

        # Barcode
        if barcode_results:
            passed = sum(1 for b in barcode_results if b.get("pass"))
            barcode = (passed / len(barcode_results)) * 100.0
        else:
            barcode = 100.0

        # Registration (position offset in px)
        reg_offset = registration.get("offset_px", 0)
        registration_score = max(0.0, 100.0 - reg_offset * 2.0)

        # Die-cut quality
        die_cut_score = die_cut.get("quality_score", 100.0)

        # Mottling / uneven ink
        mottling_score = mottling.get("quality_score", 100.0)

        overall = (
            ocr * 0.25 + color * 0.20 + ssim * 0.20 +
            barcode * 0.10 + registration_score * 0.10 +
            die_cut_score * 0.10 + mottling_score * 0.05
        )

        return {
            "overall": overall,
            "ocr": ocr, "color": color, "ssim": ssim,
            "barcode": barcode, "registration": registration_score,
            "die_cut": die_cut_score, "mottling": mottling_score,
        }

    # ── Utilities ──────────────────────────────────────────────────────────────

    async def _prepare_file(self, file_path: str, role: str) -> Path:
        p = Path(file_path)
        if not p.exists():
            raise MultiUpInspectionError(f"File not found: {file_path}")
        if p.suffix.lower() == ".pdf":
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: rasterize_pdf(p, settings.pdf_raster_dpi)
            )
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: preprocess_image(p)
        )

    def _create_sheet_annotation(
        self, scan_img, labels_meta, per_label_results, all_defects
    ) -> np.ndarray:
        """Create annotated sheet image with per-label pass/fail + defects"""
        viz = scan_img.copy()
        h, w = viz.shape[:2]

        # Draw each label with pass/fail color
        for label_info, result in zip(labels_meta["labels"], per_label_results):
            bbox = label_info["bbox"]
            x, y, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            if result["pass_fail"]:
                color = (50, 200, 50)  # Green
                badge = "PASS"
            else:
                color = (50, 50, 220)  # Red
                badge = "FAIL"

            # Thick colored border
            cv2.rectangle(viz, (x, y), (x + bw, y + bh), color, 4)

            # Score badge
            label_id = result["label_id"]
            score = result["overall_score"]
            text = f"{label_id} | {score:.0f} | {badge}"

            # Text background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(viz, (x, y), (x + tw + 16, y + th + 16), color, -1)
            cv2.putText(viz, text, (x + 8, y + th + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Overlay all defects as semi-transparent red dots
        for defect in all_defects:
            bbox = defect.get("bbox", {})
            if bbox:
                center = (bbox.get("x", 0) + bbox.get("w", 0) // 2,
                          bbox.get("y", 0) + bbox.get("h", 0) // 2)
                cv2.circle(viz, center, 6, (0, 0, 255), -1)
                cv2.circle(viz, center, 8, (255, 255, 255), 1)

        # Banner
        banner_h = 60
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        passed = sum(1 for r in per_label_results if r["pass_fail"])
        total = len(per_label_results)
        expected = labels_meta.get("count_expected")

        sheet_pass = (
            passed == total
            and labels_meta["missing_labels"] == 0
            and total > 0
        )
        banner_color = (30, 150, 30) if sheet_pass else (30, 30, 180)
        banner[:] = banner_color

        text1 = f"SHEET: {passed}/{total} PASS"
        if expected:
            text1 += f" (expected {expected}, missing {labels_meta['missing_labels']})"

        cv2.putText(banner, text1, (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        text2 = f"Imposition: {labels_meta['imposition']}   Defects: {len(all_defects)}   Method: {labels_meta['method']}"
        cv2.putText(banner, text2, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        return np.vstack([banner, viz])


# Singleton
_multi_engine: Optional[MultiUpInspectionEngine] = None

def get_multi_engine() -> MultiUpInspectionEngine:
    global _multi_engine
    if _multi_engine is None:
        _multi_engine = MultiUpInspectionEngine()
    return _multi_engine
