"""
Greenpack Pro v3.0 — Prepress Router

Endpoints:
  POST /api/v1/prepress/identify-colors    — extract Pantone codes from scan
  POST /api/v1/prepress/trial-comparison   — full trial vs final inspection
  GET  /api/v1/prepress/{job_id}           — get prepress job result
  GET  /api/v1/prepress/pantone-library    — list bundled Pantone library
  POST /api/v1/prepress/import-pantone-csv — import custom color library
"""
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db, AsyncSessionLocal
from app.models.base import InspectionJob, InspectionResult, AuditLog, JobStatus
from app.services.auth_service import get_current_user
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
    file: UploadFile = File(..., description="Scanned image or PDF of past sticker work"),
    k: int = Form(default=8, description="Number of dominant colors to extract (4-15)"),
    ignore_white: bool = Form(default=True, description="Skip near-white background"),
    top_n_per_color: int = Form(default=5, description="Show top N PANTONE matches per color"),
    current_user=Depends(get_current_user),
):
    """
    Identify PANTONE color codes in a scanned image or PDF.

    Workflow:
      1. User scans/uploads a sticker (e.g., past production work)
      2. Service extracts dominant colors using K-means in Lab space
      3. Each color is matched against the bundled PANTONE library
      4. Returns ranked list of PMS codes with ΔE distance + confidence
    """
    # Validate
    if k < 2 or k > 15:
        raise HTTPException(status_code=422, detail="k must be 2-15")

    suffix = Path(file.filename or "scan").suffix.lower() or ".jpg"
    if suffix not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"]:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {suffix}")

    # Save temp
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"pantone_{uuid.uuid4().hex[:8]}{suffix}"

    file_data = await file.read()
    if len(file_data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large (max 50MB)")

    with open(temp_path, "wb") as f:
        f.write(file_data)

    try:
        # Convert PDF to image if needed
        if suffix == ".pdf":
            from app.services.preprocess import rasterize_pdf
            img_path = await asyncio.get_event_loop().run_in_executor(
                None, lambda: rasterize_pdf(temp_path, dpi=300)
            )
        else:
            img_path = temp_path

        img = cv2.imread(str(img_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image")

        # Run identification
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

        # Audit
        log.info(f"User {current_user.email} identified {result['total_colors_found']} colors")

        return {
            "user_email": current_user.email,
            "extracted_colors": result["extracted_colors"],
            "total_colors_found": result["total_colors_found"],
            "library_size": result["library_size"],
            "library_version": result.get("library_version", "1.0"),
            "method": result["method"],
            "report_image_path": str(report_path),
            "report_image_url": f"/api/v1/prepress/pantone-report/{report_filename}",
        }

    except Exception as e:
        log.exception(f"Pantone identification error: {e}")
        raise HTTPException(status_code=500, detail=f"Identification failed: {e}")
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass


@router.get("/pantone-report/{filename}")
async def get_pantone_report(filename: str, current_user=Depends(get_current_user)):
    """Download annotated Pantone report image"""
    report_path = Path(settings.reports_dir) / filename
    if not report_path.exists() or ".." in filename or "/" in filename:
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="image/jpeg", filename=filename)


@router.get("/pantone-library")
async def list_pantone_library(
    system: Optional[str] = None,
    finish: Optional[str] = None,
    limit: int = 100,
    current_user=Depends(get_current_user),
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


@router.post("/import-pantone-csv")
async def import_custom_pantone_library(
    file: UploadFile = File(..., description="CSV with code,L,a,b columns from spectrophotometer"),
    current_user=Depends(get_current_user),
):
    """
    Import custom color library from CSV.
    Required columns: code, L, a, b
    Optional columns: system, finish, hex, r, g, b
    """
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Admin/manager only")

    suffix = Path(file.filename or "lib.csv").suffix.lower()
    if suffix != ".csv":
        raise HTTPException(status_code=422, detail="Must be a CSV file")

    temp_path = Path(settings.temp_dir) / f"custom_lib_{uuid.uuid4().hex[:8]}.csv"
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    data = await file.read()
    with open(temp_path, "wb") as f:
        f.write(data)

    try:
        custom_lib = import_custom_library(str(temp_path))
        log.info(f"User {current_user.email} imported {custom_lib['color_count']} colors")
        return {
            "imported_count": custom_lib["color_count"],
            "version": custom_lib["version"],
            "preview": custom_lib["colors"][:5],
        }
    finally:
        try: temp_path.unlink()
        except Exception: pass


# ── Trial Comparison ─────────────────────────────────────────────────────────

@router.post("/trial-comparison")
async def create_trial_comparison(
    final_design: UploadFile = File(..., description="Final approved design (PDF or image)"),
    trial_proofs: List[UploadFile] = File(..., description="One or more scanned trial prints"),
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
    waste_unit_cost_usd: float = Form(default=5.0, description="USD per m² for waste calc"),
    waste_run_size_m2: float = Form(default=1000.0, description="Expected run size m²"),
    db: AsyncSession = Depends(get_db),
):
    """
    Compare trial demo prints against the final design BEFORE production.
    Returns GO/HOLD/NO_GO recommendation + accuracy report + waste savings estimate.
    """
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

    # Create DB job
    job = InspectionJob(
        id=job_id,
        company_id=1,
        created_by=1,
        job_ref=job_ref,
        client_name=client_name,
        product_name=product_name,
        master_file_path=str(final_path),
        scan_file_path=trial_paths[0],
        input_source="prepress",
        status=JobStatus.queued,
    )
    db.add(job)
    db.add(AuditLog(
        company_id=current_user.company_id,
        user_id=current_user.id,
        action="PREPRESS_CREATED",
        resource_type="job",
        resource_id=job_id,
        details={"trial_count": len(trial_paths)},
    ))
    await db.commit()

    config = {
        "job_ref": job_ref,
        "client_name": client_name,
        "product_name": product_name,
        "inspector_name": current_user.full_name,
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
        _run_prepress_job(job_id, str(final_path), trial_paths, config)
    )

    return {
        "job_id": job_id,
        "job_ref": job_ref,
        "status": "queued",
        "trial_count": len(trial_paths),
        "mode": "prepress_trial_comparison",
    }


async def _run_prepress_job(job_id: str, final_path: str,
                             trial_paths: List[str], config: dict):
    """Background worker for prepress comparison"""
    async with AsyncSessionLocal() as db:
        r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
        job = r.scalar_one_or_none()
        if job:
            job.status = JobStatus.processing
            await db.commit()

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

        async with AsyncSessionLocal() as db:
            r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = r.scalar_one_or_none()
            if job:
                job.status = JobStatus.completed
                job.overall_score = result["accuracy_score"]
                job.pass_fail = (result["decision"] == "GO")
                job.processing_time_ms = result["processing_time_ms"]
                job.completed_at = datetime.utcnow()

                db_result = InspectionResult(
                    job_id=job_id,
                    ocr_errors=[],
                    color_results=result.get("trial_reports", []),
                    barcode_results=[],
                    defects=[],
                    overall_score=result["accuracy_score"],
                    pass_fail=(result["decision"] == "GO"),
                    report_pdf_path=str(pdf_path) if pdf_path else None,
                    excel_path=str(excel_path) if excel_path else None,
                )
                db.add(db_result)
                await db.commit()

        log.info(f"[{job_id}] Prepress complete: decision={result['decision']}")

    except PrepressError as e:
        async with AsyncSessionLocal() as db:
            r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = r.scalar_one_or_none()
            if job:
                job.status = JobStatus.failed
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await db.commit()
    except Exception as e:
        log.exception(f"[{job_id}] Prepress unexpected error")
        async with AsyncSessionLocal() as db:
            r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = r.scalar_one_or_none()
            if job:
                job.status = JobStatus.failed
                job.error_message = f"Unexpected: {e}"
                job.completed_at = datetime.utcnow()
                await db.commit()


@router.get("/{job_id}")
async def get_prepress_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get prepress comparison result"""
    job_r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
    job = job_r.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    res_r = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    res = res_r.scalar_one_or_none()

    return {
        "job_id": job_id,
        "job_ref": job.job_ref,
        "status": job.status,
        "client_name": job.client_name,
        "product_name": job.product_name,
        "decision": "GO" if job.pass_fail else "NO_GO",
        "accuracy_score": job.overall_score,
        "trial_reports": (res.color_results if res else []),
        "processing_time_ms": job.processing_time_ms,
        "error_message": job.error_message,
    }
