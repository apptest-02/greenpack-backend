"""
Greenpack Pro — Core Inspection Engine
Orchestrates the full 10-step inspection pipeline:
PDF rasterize → align → OCR → color → SSIM → barcode → score → report
"""
import asyncio
import gc
import logging
import time
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.services.alignment import align_images
from app.services.ocr_service import run_dual_ocr, diff_text_regions
from app.services.color_service import analyze_color_zones
from app.services.ssim_service import detect_defects
from app.services.barcode_service import verify_barcodes
from app.services.report_service import generate_pdf_report, generate_excel_report
from app.services.preprocess import rasterize_pdf, preprocess_image
from app.services.annotator import create_annotated_image
from app.services.webhook_service import post_webhook

import cv2
import numpy as np

log = logging.getLogger(__name__)
settings = get_settings()


class InspectionError(Exception):
    """Raised when inspection cannot complete"""
    pass


class InspectionEngine:
    """
    Main inspection engine. Call run_inspection() to process a job.
    Memory: large numpy arrays are explicitly deleted after each job.
    EasyOCR model is loaded once (singleton) and reused.
    """

    async def run_inspection(
        self,
        job_id: str,
        master_path: str,
        scan_path: str,
        config: dict,
        progress_callback=None,
    ) -> dict:
        """
        Run full inspection pipeline.
        Returns result dict with all scores and error lists.
        """
        start_time = time.time()
        log.info(f"[{job_id}] Starting inspection pipeline")

        def progress(step: int, name: str, detail: str = ""):
            log.info(f"[{job_id}] Step {step}/10: {name} {detail}")
            if progress_callback:
                asyncio.create_task(
                    progress_callback(job_id, step, name, detail)
                )

        master_img = None
        scan_img = None
        aligned_img = None

        try:
            # ── Step 1: File Pre-processing ────────────────────────────────────
            progress(1, "Pre-processing", "Rasterizing and normalizing images")
            master_path = await self._prepare_file(master_path, "master")
            scan_path = await self._prepare_file(scan_path, "scan")

            # ── Step 2: Load Images ────────────────────────────────────────────
            progress(2, "Loading images", "Reading into memory")
            master_img = cv2.imread(str(master_path))
            scan_img = cv2.imread(str(scan_path))

            if master_img is None:
                raise InspectionError(f"Cannot read master image: {master_path}")
            if scan_img is None:
                raise InspectionError(f"Cannot read scan image: {scan_path}")

            # ── Step 3: Image Alignment ────────────────────────────────────────
            progress(3, "Aligning images", "ORB keypoint matching + RANSAC")
            aligned_img, alignment_confidence = align_images(master_img, scan_img)
            log.info(f"[{job_id}] Alignment confidence: {alignment_confidence:.3f}")

            if alignment_confidence < 0.20:
                raise InspectionError(
                    "Cannot align images — verify correct label was scanned. "
                    "The master and scan appear to be different labels."
                )

            # ── Step 4: OCR Text Extraction ────────────────────────────────────
            progress(4, "OCR text extraction", "EasyOCR + Tesseract dual engine")
            try:
                ocr_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: run_dual_ocr(str(master_path), str(scan_path))
                    ),
                    timeout=settings.ocr_timeout_seconds,
                )
                ocr_timeout = False
            except asyncio.TimeoutError:
                log.warning(f"[{job_id}] OCR timed out after {settings.ocr_timeout_seconds}s")
                ocr_result = {"master_regions": [], "scan_regions": [], "errors": [], "timeout": True}
                ocr_timeout = True

            # ── Step 5: Text Comparison ────────────────────────────────────────
            progress(5, "Comparing text", "Character-level diff analysis")
            if not ocr_timeout:
                text_errors = diff_text_regions(
                    ocr_result["master_regions"],
                    ocr_result["scan_regions"],
                )
            else:
                text_errors = []

            # ── Step 6: Color Analysis ─────────────────────────────────────────
            progress(6, "Color analysis", f"Delta-E CIE2000 (threshold: ΔE {config.get('color_threshold', 2.0)})")
            color_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analyze_color_zones(
                    master_img,
                    aligned_img,
                    threshold=config.get("color_threshold", settings.default_color_tolerance_de),
                ),
            )

            # ── Step 7: Defect Detection ───────────────────────────────────────
            progress(7, "Defect detection", f"SSIM analysis (threshold: {config.get('ssim_threshold', 0.75)})")
            ssim_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: detect_defects(
                    master_img,
                    aligned_img,
                    threshold=config.get("ssim_threshold", settings.default_ssim_threshold),
                ),
            )

            # ── Step 8: Barcode Verification ───────────────────────────────────
            progress(8, "Barcode verification", "pyzbar EAN/QR/Code128 + GS1 check digit")
            barcode_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: verify_barcodes(
                    str(scan_path),
                    config.get("barcode_rules", []),
                ),
            )

            # ── Step 9: Score Calculation ──────────────────────────────────────
            progress(9, "Calculating score", "Weighted: OCR 35% + Color 30% + SSIM 20% + Barcode 15%")
            scores = self._calculate_scores(
                text_errors=text_errors,
                color_result=color_result,
                ssim_result=ssim_result,
                barcode_result=barcode_result,
                ocr_timeout=ocr_timeout,
            )

            # ── Step 10: Report Generation ─────────────────────────────────────
            progress(10, "Generating report", "PDF + Excel")
            annotated_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: create_annotated_image(
                    master_img,
                    aligned_img,
                    text_errors,
                    color_result.get("zone_results", []),
                    ssim_result.get("defects", []),
                    barcode_result,
                    job_id,
                ),
            )

            report_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_pdf_report(
                    job_id=job_id,
                    config=config,
                    scores=scores,
                    text_errors=text_errors,
                    color_result=color_result,
                    ssim_result=ssim_result,
                    barcode_result=barcode_result,
                    annotated_path=annotated_path,
                    ocr_timeout=ocr_timeout,
                ),
            )

            excel_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_excel_report(
                    job_id=job_id,
                    config=config,
                    scores=scores,
                    text_errors=text_errors,
                    color_result=color_result,
                    ssim_result=ssim_result,
                    barcode_result=barcode_result,
                ),
            )

            processing_ms = int((time.time() - start_time) * 1000)

            result = {
                "job_id": job_id,
                "overall_score": round(scores["overall"], 2),
                "pass_fail": scores["overall"] >= 75.0,
                "ocr_score": scores["ocr"],
                "color_score": scores["color"],
                "ssim_score_weighted": scores["ssim"],
                "barcode_score": scores["barcode"],
                "alignment_confidence": round(alignment_confidence, 3),
                "ssim_score": ssim_result.get("ssim_score", 0),
                "text_errors": text_errors,
                "color_results": color_result.get("zone_results", []),
                "barcode_results": barcode_result,
                "defects": ssim_result.get("defects", []),
                "annotated_image_path": str(annotated_path) if annotated_path else None,
                "report_pdf_path": str(report_path) if report_path else None,
                "excel_path": str(excel_path) if excel_path else None,
                "processing_time_ms": processing_ms,
                "ocr_timeout": ocr_timeout,
            }

            log.info(
                f"[{job_id}] Complete: score={result['overall_score']:.1f} "
                f"pass={result['pass_fail']} time={processing_ms}ms"
            )

            # Fire webhook if configured
            if settings.webhook_enabled and settings.webhook_url:
                asyncio.create_task(
                    post_webhook(result, settings.webhook_url, settings.webhook_secret)
                )

            return result

        except InspectionError:
            raise
        except Exception as e:
            log.exception(f"[{job_id}] Unexpected inspection error: {e}")
            raise InspectionError(f"Inspection failed: {str(e)}")

        finally:
            # CRITICAL: release memory for batch processing
            for arr in [master_img, scan_img, aligned_img]:
                if arr is not None:
                    del arr
            gc.collect()

    async def _prepare_file(self, file_path: str, role: str) -> Path:
        """Rasterize PDF to PNG or preprocess image"""
        p = Path(file_path)
        if not p.exists():
            raise InspectionError(f"File not found: {file_path}")

        if p.suffix.lower() == ".pdf":
            # Rasterize PDF to PNG
            png_path = await asyncio.get_event_loop().run_in_executor(
                None, lambda: rasterize_pdf(p, settings.pdf_raster_dpi)
            )
            return png_path
        else:
            # Preprocess image
            preprocessed = await asyncio.get_event_loop().run_in_executor(
                None, lambda: preprocess_image(p)
            )
            return preprocessed

    def _calculate_scores(
        self,
        text_errors: list,
        color_result: dict,
        ssim_result: dict,
        barcode_result: list,
        ocr_timeout: bool = False,
    ) -> dict:
        """
        Calculate weighted score:
        OCR 35% + Color 30% + SSIM 20% + Barcode 15%
        """
        # OCR Score: deduct per error type
        if ocr_timeout:
            ocr_score = 70.0  # Partial credit — cannot fully assess
        else:
            error_penalty = 0.0
            for err in text_errors:
                if err.get("type") == "REPLACE":
                    error_penalty += 15.0
                elif err.get("type") == "DELETE":
                    error_penalty += 10.0
                elif err.get("type") == "INSERT":
                    error_penalty += 8.0
            ocr_score = max(0.0, 100.0 - error_penalty)

        # Color Score: based on mean ΔE across all zones
        zone_results = color_result.get("zone_results", [])
        if zone_results:
            pct_fail = sum(1 for z in zone_results if not z.get("pass", True)) / len(zone_results)
            color_score = max(0.0, 100.0 - (pct_fail * 100.0))
        else:
            color_score = 100.0

        # SSIM Score: SSIM score directly → percentage
        ssim_raw = ssim_result.get("ssim_score", 1.0)
        ssim_score = round(ssim_raw * 100.0, 1)

        # Barcode Score
        if barcode_result:
            passed = sum(1 for b in barcode_result if b.get("pass", False))
            barcode_score = (passed / len(barcode_result)) * 100.0
        else:
            barcode_score = 100.0  # No barcodes to check = full score

        # Weighted overall
        overall = (
            ocr_score * 0.35 +
            color_score * 0.30 +
            ssim_score * 0.20 +
            barcode_score * 0.15
        )

        return {
            "overall": round(overall, 2),
            "ocr": round(ocr_score, 2),
            "color": round(color_score, 2),
            "ssim": round(ssim_score, 2),
            "barcode": round(barcode_score, 2),
        }


# Singleton engine instance
_engine_instance: Optional[InspectionEngine] = None


def get_engine() -> InspectionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InspectionEngine()
    return _engine_instance
