"""Greenpack Pro v3.0 — Prepress Router (No Auth - Production Ready)"""
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db, AsyncSessionLocal
from app.models.base import InspectionJob, InspectionResult, AuditLog, JobStatus
from app.services.pantone_service import (
    identify_pantone_colors_in_image,
    create_pantone_report_image,
    load_pantone_library,
    import_custom_library,
)
from app.services.prepress_inspection import get_prepress_engine, PrepressError

router = APIRouter()
log = logging.getLogger(__name__)
settings = get_settings()


# ── Pantone Color Identification ─────────────────────────────────────────────

@router.post("/identify-colors")
async def identify_colors(
    file: UploadFile = File(...),
    k: int = Form(default=8),
    ignore_white: bool = Form(default=True),
    top_n_per_color: int = Form(default=5),
):
    """Identify PANTONE color codes in a scanned image or PDF - NO AUTH"""
    
    if k < 2 or k > 15:
        raise HTTPException(status_code=422, detail="k must be 2-15")

    suffix = Path(file.filename or "scan").suffix.lower() or ".jpg"
    if suffix not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"]:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {suffix}")

    import tempfile
    import cv2
    import numpy as np
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            from app.services.preprocess import rasterize_pdf
            img_path = rasterize_pdf(Path(tmp_path), dpi=300)
        else:
            img_path = Path(tmp_path)

        img = cv2.imread(str(img_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: identify_pantone_colors_in_image(
                img, k=k, top_n_per_color=top_n_per_color, ignore_white=ignore_white,
            ),
        )

        # Generate report image
        report_img = create_pantone_report_image(img, result)
        report_dir = Path(settings.reports_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_filename = f"pantone_report_{uuid.uuid4().hex[:8]}.jpg"
        report_path = report_dir / report_filename
        cv2.imwrite(str(report_path), report_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return {
            "user_email": "user@example.com",
            "extracted_colors": result["extracted_colors"],
            "total_colors_found": result["total_colors_found"],
            "library_size": result["library_size"],
            "library_version": result.get("library_version", "1.0"),
            "method": result["method"],
            "report_image_url": f"/api/v1/prepress/pantone-report/{report_filename}",
        }

    except Exception as e:
        log.exception(f"Pantone identification error: {e}")
        raise HTTPException(status_code=500, detail=f"Identification failed: {e}")
    finally:
        try:
            Path(tmp_path).unlink()
        except:
            pass


@router.get("/pantone-report/{filename}")
async def get_pantone_report(filename: str):
    """Download annotated Pantone report image"""
    report_path = Path(settings.reports_dir) / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="image/jpeg", filename=filename)


@router.get("/pantone-library")
async def list_pantone_library(
    system: Optional[str] = None,
    finish: Optional[str] = None,
    limit: int = 100,
):
    """List the bundled PANTONE color library"""
    lib = load_pantone_library()
    colors = lib.get("colors", [])

    if system:
        colors = [c for c in colors if c.get("system", "").upper() == system.upper()]
    if finish:
        colors = [c for c in colors if c.get("finish", "").upper() == finish.upper()]

    return {
        "version": lib.get("version"),
        "total_colors": len(lib.get("colors", [])),
        "filtered_count": len(colors),
        "colors": colors[:limit],
    }


# ── Trial Comparison (No Auth) ─────────────────────────────────────────────

@router.post("/trial-comparison")
async def create_trial_comparison(
    final_design: UploadFile = File(...),
    trial_proofs: List[UploadFile] = File(...),
    job_ref: str = Form(default=""),
    client_name: str = Form(default=""),
    product_name: str = Form(default=""),
    color_threshold: float = Form(default=2.0),
    ssim_threshold: float = Form(default=0.75),
    min_accuracy_for_go: float = Form(default=90.0),
    min_font_size_pt: float = Form(default=6.0),
    check_expiry_dates: bool = Form(default=True),
    check_icon_sizes: bool = Form(default=True),
    spell_check: bool = Form(default=True),
    identify_pantones: bool = Form(default=True),
    waste_unit_cost_usd: float = Form(default=5.0),
    waste_run_size_m2: float = Form(default=1000.0),
):
    """Compare trial demo prints against the final design - NO AUTH"""
    
    if not trial_proofs:
        raise HTTPException(status_code=422, detail="Upload at least 1 trial proof")
    if len(trial_proofs) > 10:
        raise HTTPException(status_code=422, detail="Max 10 trial proofs per job")

    job_id = str(uuid.uuid4())
    files_dir = Path(settings.local_storage_root) / "prepress" / job_id
    files_dir.mkdir(parents=True, exist_ok=True)

    # Save final design
    final_suffix = Path(final_design.filename).suffix or ".pdf"
    final_path = files_dir / f"final{final_suffix}"
    with open(final_path, "wb") as f:
        f.write(await final_design.read())

    # Save trial proofs
    trial_paths = []
    for i, tp in enumerate(trial_proofs):
        ts = Path(tp.filename).suffix or ".jpg"
        tpath = files_dir / f"trial_{i + 1}{ts}"
        with open(tpath, "wb") as f:
            f.write(await tp.read())
        trial_paths.append(str(tpath))

    if not job_ref:
        job_ref = f"PREPRESS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    config = {
        "job_ref": job_ref,
        "client_name": client_name,
        "product_name": product_name,
        "inspector_name": "System",
        "color_threshold": color_threshold,
        "ssim_threshold": ssim_threshold,
        "min_accuracy_for_go": min_accuracy_for_go,
        "min_font_size_pt": min_font_size_pt,
        "check_expiry_dates": check_expiry_dates,
        "check_icon_sizes": check_icon_sizes,
        "spell_check": spell_check,
        "identify_pantones": identify_pantones,
        "waste_estimate": {
            "unit_cost_per_m2": waste_unit_cost_usd,
            "expected_run_m2": waste_run_size_m2,
        },
    }

    asyncio.create_task(
        _run_prepress_job(job_id, str(final_path), trial_paths, config, job_ref, client_name, product_name)
    )

    return {
        "job_id": job_id,
        "job_ref": job_ref,
        "status": "queued",
        "trial_count": len(trial_paths),
        "mode": "prepress_trial_comparison",
    }


async def _run_prepress_job(job_id: str, final_path: str, trial_paths: List[str], 
                             config: dict, job_ref: str, client_name: str, product_name: str):
    """Background worker for prepress comparison"""
    try:
        engine = get_prepress_engine()
        result = await engine.compare_trial_to_final(
            job_id=job_id,
            final_design_path=final_path,
            trial_proof_paths=trial_paths,
            config=config,
        )

        # Generate reports
        from app.services.prepress_report import generate_prepress_pdf, generate_prepress_excel
        pdf_path = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate_prepress_pdf(job_id, config, result)
        )
        excel_path = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate_prepress_excel(job_id, config, result)
        )

        log.info(f"Prepress job {job_id} complete: decision={result['decision']}")

    except PrepressError as e:
        log.error(f"Prepress job {job_id} error: {e}")
    except Exception as e:
        log.exception(f"Prepress job {job_id} unexpected error: {e}")


@router.get("/{job_id}")
async def get_prepress_result(job_id: str):
    """Get prepress comparison result"""
    result_file = Path(settings.local_storage_root) / "prepress" / job_id / "result.json"
    
    # Try to read result from file
    if result_file.exists():
        import json
        with open(result_file, "r") as f:
            return json.load(f)
    
    # Check if processing
    job_dir = Path(settings.local_storage_root) / "prepress" / job_id
    if job_dir.exists():
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Job is still processing. Please check back in a few moments."
        }
    
    raise HTTPException(status_code=404, detail="Job not found")