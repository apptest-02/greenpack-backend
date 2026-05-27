"""Greenpack Pro v2.0 — Multi-Up Jobs Router

Endpoints for multi-up label inspection:
  POST /api/v1/jobs/multi-up       — Create multi-up job
  GET  /api/v1/jobs/{id}/multi-up  — Get full multi-up result
  GET  /api/v1/jobs/{id}/labels/{label_id}  — Get single label detail
"""
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db, AsyncSessionLocal
from app.models.base import InspectionJob, InspectionResult, AuditLog, JobStatus
from app.services.auth_service import get_current_user
from app.services.multi_up_inspection import get_multi_engine, MultiUpInspectionError

router = APIRouter()
log = logging.getLogger(__name__)
settings = get_settings()


@router.post("/multi-up")
async def create_multi_up_job(
    master_file: UploadFile = File(..., description="Master label PDF or image (single label)"),
    scan_file: UploadFile = File(..., description="Scanned multi-up sheet"),
    job_ref: str = Form(default=""),
    client_name: str = Form(default=""),
    product_name: str = Form(default=""),
    expected_count: Optional[int] = Form(default=None, description="Expected number of labels (1-15)"),
    is_transparent: bool = Form(default=False, description="Are labels transparent/clear?"),
    color_threshold: float = Form(default=2.0),
    ssim_threshold: float = Form(default=0.75),
    check_braille: bool = Form(default=False),
    check_font_size: bool = Form(default=False),
    min_font_size_pt: float = Form(default=6.0),
    spell_check: bool = Form(default=False),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a multi-up inspection job. Detects 1-15 labels in a scanned cut-out."""
    job_id = str(uuid.uuid4())

    # Validate expected count
    if expected_count is not None:
        if expected_count < 1 or expected_count > 50:
            raise HTTPException(status_code=422, detail="expected_count must be 1-50")

    # Save files
    files_dir = Path(settings.local_storage_root) / "jobs" / job_id
    files_dir.mkdir(parents=True, exist_ok=True)

    master_suffix = Path(master_file.filename).suffix or ".pdf"
    scan_suffix = Path(scan_file.filename).suffix or ".jpg"
    master_path = files_dir / f"master{master_suffix}"
    scan_path = files_dir / f"scan{scan_suffix}"

    master_data = await master_file.read()
    scan_data = await scan_file.read()

    if len(master_data) > 100 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Master file too large (max 100MB)")
    if len(scan_data) > 150 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Scan file too large (max 150MB)")

    with open(master_path, "wb") as f:
        f.write(master_data)
    with open(scan_path, "wb") as f:
        f.write(scan_data)

    if not job_ref:
        job_ref = f"MULTI-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    job = InspectionJob(
        id=job_id,
        company_id=current_user.company_id,
        created_by=current_user.id,
        job_ref=job_ref,
        client_name=client_name,
        product_name=product_name,
        master_file_path=str(master_path),
        scan_file_path=str(scan_path),
        input_source="multi_up",
        status=JobStatus.queued,
    )
    db.add(job)
    db.add(AuditLog(
        company_id=current_user.company_id,
        user_id=current_user.id,
        action="MULTI_UP_CREATED",
        resource_type="job",
        resource_id=job_id,
        details={"expected_count": expected_count, "transparent": is_transparent},
    ))
    await db.commit()

    log.info(f"Multi-up job {job_id} ({job_ref}) created, expected={expected_count}")

    config = {
        "job_id": job_id,
        "job_ref": job_ref,
        "client_name": client_name,
        "product_name": product_name,
        "inspector_name": current_user.full_name,
        "expected_count": expected_count,
        "is_transparent": is_transparent,
        "color_threshold": color_threshold,
        "ssim_threshold": ssim_threshold,
        "check_braille": check_braille,
        "check_font_size": check_font_size,
        "min_font_size_pt": min_font_size_pt,
        "spell_check": spell_check,
        "barcode_rules": [],
    }

    asyncio.create_task(
        _run_multi_up_job(job_id, str(master_path), str(scan_path), config)
    )

    return {
        "job_id": job_id,
        "job_ref": job_ref,
        "status": "queued",
        "expected_count": expected_count,
        "mode": "multi_up",
    }


async def _run_multi_up_job(job_id: str, master_path: str, scan_path: str, config: dict):
    """Background worker for multi-up inspection"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            job.status = JobStatus.processing
            await db.commit()

    try:
        engine = get_multi_engine()
        res = await engine.inspect_sheet(
            job_id=job_id,
            master_path=master_path,
            scan_path=scan_path,
            config=config,
        )

        # Generate sheet PDF report
        from app.services.multi_up_report import generate_multi_up_pdf, generate_multi_up_excel
        pdf_path = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate_multi_up_pdf(job_id, config, res)
        )
        excel_path = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate_multi_up_excel(job_id, config, res)
        )

        async with AsyncSessionLocal() as db:
            r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = r.scalar_one_or_none()
            if job:
                job.status = JobStatus.completed
                job.overall_score = res["overall_score"]
                job.pass_fail = res["sheet_pass"]
                job.processing_time_ms = res["processing_time_ms"]
                job.completed_at = datetime.utcnow()

                db_result = InspectionResult(
                    job_id=job_id,
                    ocr_errors=[],
                    color_results=[],
                    barcode_results=[],
                    defects=res.get("all_defects", []),
                    overall_score=res["overall_score"],
                    pass_fail=res["sheet_pass"],
                    annotated_image_path=res.get("sheet_annotated_path"),
                    report_pdf_path=str(pdf_path) if pdf_path else None,
                    excel_path=str(excel_path) if excel_path else None,
                )
                # Store per-label results in defects field (reuse for now, will add proper columns in migration)
                db_result.color_results = res.get("per_label_results", [])  # reuse field
                db.add(db_result)
                await db.commit()

        log.info(f"Multi-up job {job_id} complete: {res['labels_passed']}/{res['labels_found']} passed")

    except MultiUpInspectionError as e:
        log.error(f"Multi-up job {job_id} inspection error: {e}")
        async with AsyncSessionLocal() as db:
            r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = r.scalar_one_or_none()
            if job:
                job.status = JobStatus.failed
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await db.commit()

    except Exception as e:
        log.exception(f"Multi-up job {job_id} unexpected error")
        async with AsyncSessionLocal() as db:
            r = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = r.scalar_one_or_none()
            if job:
                job.status = JobStatus.failed
                job.error_message = f"Unexpected error: {str(e)}"
                job.completed_at = datetime.utcnow()
                await db.commit()


@router.get("/{job_id}/multi-up")
async def get_multi_up_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get full multi-up inspection result with per-label details"""
    result = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    r = result.scalar_one_or_none()
    if not r:
        raise HTTPException(status_code=404, detail="Result not found (job may still be processing)")

    job_result = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    per_label = r.color_results or []

    return {
        "job_id": job_id,
        "job_ref": job.job_ref,
        "client_name": job.client_name,
        "product_name": job.product_name,
        "status": job.status,
        "mode": "multi_up",
        "overall_score": r.overall_score,
        "sheet_pass": r.pass_fail,
        "labels_found": len(per_label),
        "labels_passed": sum(1 for lb in per_label if lb.get("pass_fail")),
        "labels_failed": sum(1 for lb in per_label if not lb.get("pass_fail")),
        "per_label_results": per_label,
        "all_defects": r.defects or [],
        "sheet_annotated_path": r.annotated_image_path,
        "report_pdf_path": r.report_pdf_path,
        "excel_path": r.excel_path,
        "processing_time_ms": job.processing_time_ms,
    }


@router.get("/{job_id}/labels/{label_id}")
async def get_label_detail(
    job_id: str,
    label_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get detailed inspection for a single label in a multi-up job.
    label_id format: 'row-col' e.g. '1-3'"""
    result = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    r = result.scalar_one_or_none()
    if not r:
        raise HTTPException(status_code=404, detail="Result not found")

    per_label = r.color_results or []
    label = next((lb for lb in per_label if lb.get("label_id") == label_id), None)
    if not label:
        raise HTTPException(status_code=404, detail=f"Label {label_id} not found")

    return label


@router.get("/{job_id}/sheet-image")
async def download_sheet_image(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Download the annotated sheet image"""
    result = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    r = result.scalar_one_or_none()
    if not r or not r.annotated_image_path:
        raise HTTPException(status_code=404, detail="Sheet image not found")
    p = Path(r.annotated_image_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sheet image file missing on disk")
    return FileResponse(p, media_type="image/jpeg",
                       filename=f"greenpack_sheet_{job_id[:8]}.jpg")
