"""Greenpack Pro — Jobs Router"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.config import get_settings
from app.database import get_db
from app.models.base import InspectionJob, InspectionResult, AuditLog, JobStatus
from app.services.auth_service import get_current_user
from app.services.inspection_engine import get_engine, InspectionError
from app.services.report_service import print_report_windows

router = APIRouter()
log = logging.getLogger(__name__)
settings = get_settings()

# In-memory progress store (use Redis in Mode B)
_job_progress: dict = {}


@router.get("")
async def list_jobs(
    status: Optional[str] = None,
    client: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = select(InspectionJob).where(
        InspectionJob.company_id == current_user.company_id
    ).order_by(desc(InspectionJob.created_at)).limit(limit).offset(offset)

    if status:
        q = q.where(InspectionJob.status == status)
    if client:
        q = q.where(InspectionJob.client_name.ilike(f"%{client}%"))

    # Client role: only see own client's data
    if current_user.role == "client":
        q = q.where(InspectionJob.client_name == current_user.full_name)

    result = await db.execute(q)
    jobs = result.scalars().all()

    return [{
        "id": j.id, "job_ref": j.job_ref, "client_name": j.client_name,
        "product_name": j.product_name, "status": j.status,
        "overall_score": j.overall_score, "pass_fail": j.pass_fail,
        "input_source": j.input_source, "created_at": j.created_at.isoformat() if j.created_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
    } for j in jobs]


@router.post("")
async def create_job(
    master_file: UploadFile = File(...),
    scan_file: UploadFile = File(...),
    job_ref: str = Form(default=""),
    client_name: str = Form(default=""),
    product_name: str = Form(default=""),
    color_threshold: float = Form(default=2.0),
    ssim_threshold: float = Form(default=0.75),
    template_id: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    job_id = str(uuid.uuid4())

    # Save uploaded files
    files_dir = Path(settings.local_storage_root) / "jobs" / job_id
    files_dir.mkdir(parents=True, exist_ok=True)

    master_suffix = Path(master_file.filename).suffix or ".pdf"
    scan_suffix = Path(scan_file.filename).suffix or ".jpg"
    master_path = files_dir / f"master{master_suffix}"
    scan_path = files_dir / f"scan{scan_suffix}"

    with open(master_path, "wb") as f:
        f.write(await master_file.read())
    with open(scan_path, "wb") as f:
        f.write(await scan_file.read())

    # Validate file sizes
    if master_path.stat().st_size > 100 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Master file too large (max 100MB)")
    if scan_path.stat().st_size > 100 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Scan file too large (max 100MB)")

    # Create job record
    if not job_ref:
        from datetime import datetime
        job_ref = f"JOB-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    job = InspectionJob(
        id=job_id,
        company_id=current_user.company_id,
        created_by=current_user.id,
        template_id=template_id,
        job_ref=job_ref,
        client_name=client_name,
        product_name=product_name,
        master_file_path=str(master_path),
        scan_file_path=str(scan_path),
        input_source="upload",
        status=JobStatus.queued,
    )
    db.add(job)
    db.add(AuditLog(
        company_id=current_user.company_id, user_id=current_user.id,
        action="JOB_CREATED", resource_type="job", resource_id=job_id,
    ))
    await db.commit()

    log.info(f"Job created: {job_id} ({job_ref})")

    # Start inspection in background
    config = {
        "job_id": job_id, "job_ref": job_ref,
        "client_name": client_name, "product_name": product_name,
        "inspector_name": current_user.full_name,
        "color_threshold": color_threshold,
        "ssim_threshold": ssim_threshold,
        "barcode_rules": [],
    }
    asyncio.create_task(_run_job(job_id, str(master_path), str(scan_path), config))

    return {"job_id": job_id, "job_ref": job_ref, "status": "queued"}


async def _run_job(job_id: str, master_path: str, scan_path: str, config: dict):
    """Background task: run inspection and update DB"""
    from app.database import AsyncSessionLocal
    from datetime import datetime

    async with AsyncSessionLocal() as db:
        # Update status → processing
        result = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return
        job.status = JobStatus.processing
        await db.commit()

    try:
        engine = get_engine()
        inspection_result = await engine.run_inspection(
            job_id=job_id,
            master_path=master_path,
            scan_path=scan_path,
            config=config,
        )

        async with AsyncSessionLocal() as db:
            result_obj = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = result_obj.scalar_one_or_none()
            if job:
                job.status = JobStatus.completed
                job.overall_score = inspection_result["overall_score"]
                job.pass_fail = inspection_result["pass_fail"]
                job.processing_time_ms = inspection_result["processing_time_ms"]
                job.completed_at = datetime.utcnow()

                db_result = InspectionResult(
                    job_id=job_id,
                    ocr_errors=inspection_result["text_errors"],
                    color_results=inspection_result["color_results"],
                    barcode_results=inspection_result["barcode_results"],
                    defects=inspection_result["defects"],
                    ssim_score=inspection_result["ssim_score"],
                    alignment_confidence=inspection_result["alignment_confidence"],
                    ocr_score=inspection_result["ocr_score"],
                    color_score=inspection_result["color_score"],
                    ssim_score_weighted=inspection_result["ssim_score_weighted"],
                    barcode_score=inspection_result["barcode_score"],
                    overall_score=inspection_result["overall_score"],
                    pass_fail=inspection_result["pass_fail"],
                    annotated_image_path=inspection_result.get("annotated_image_path"),
                    report_pdf_path=inspection_result.get("report_pdf_path"),
                    excel_path=inspection_result.get("excel_path"),
                )
                db.add(db_result)
                await db.commit()

        log.info(f"Job {job_id} completed: score={inspection_result['overall_score']:.1f}")

    except InspectionError as e:
        log.error(f"Job {job_id} inspection error: {e}")
        async with AsyncSessionLocal() as db:
            result_obj = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = result_obj.scalar_one_or_none()
            if job:
                job.status = JobStatus.failed
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await db.commit()

    except Exception as e:
        log.exception(f"Job {job_id} unexpected error: {e}")
        async with AsyncSessionLocal() as db:
            result_obj = await db.execute(select(InspectionJob).where(InspectionJob.id == job_id))
            job = result_obj.scalar_one_or_none()
            if job:
                job.status = JobStatus.failed
                job.error_message = f"Unexpected error: {str(e)}"
                job.completed_at = datetime.utcnow()
                await db.commit()


@router.get("/{job_id}")
async def get_job(job_id: str, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    result = await db.execute(select(InspectionJob).where(
        InspectionJob.id == job_id, InspectionJob.company_id == current_user.company_id
    ))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id, "job_ref": job.job_ref, "client_name": job.client_name,
        "product_name": job.product_name, "status": job.status,
        "overall_score": job.overall_score, "pass_fail": job.pass_fail,
        "processing_time_ms": job.processing_time_ms,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.get("/{job_id}/result")
async def get_job_result(job_id: str, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    result = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    r = result.scalar_one_or_none()
    if not r:
        raise HTTPException(status_code=404, detail="Result not found — job may still be processing")
    return {
        "job_id": r.job_id,
        "overall_score": r.overall_score, "pass_fail": r.pass_fail,
        "ocr_score": r.ocr_score, "color_score": r.color_score,
        "ssim_score": r.ssim_score, "ssim_score_weighted": r.ssim_score_weighted,
        "barcode_score": r.barcode_score,
        "alignment_confidence": r.alignment_confidence,
        "ocr_errors": r.ocr_errors or [],
        "color_results": r.color_results or [],
        "barcode_results": r.barcode_results or [],
        "defects": r.defects or [],
        "report_pdf_path": r.report_pdf_path,
        "excel_path": r.excel_path,
        "annotated_image_path": r.annotated_image_path,
    }


@router.get("/{job_id}/report")
async def download_report(job_id: str, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    result = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    r = result.scalar_one_or_none()
    if not r or not r.report_pdf_path or not Path(r.report_pdf_path).exists():
        raise HTTPException(status_code=404, detail="Report PDF not found")
    return FileResponse(r.report_pdf_path, media_type="application/pdf",
                       filename=f"greenpack_report_{job_id[:8]}.pdf")


@router.get("/{job_id}/export/excel")
async def export_excel(job_id: str, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    result = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    r = result.scalar_one_or_none()
    if not r or not r.excel_path or not Path(r.excel_path).exists():
        raise HTTPException(status_code=404, detail="Excel export not found")
    return FileResponse(r.excel_path,
                       media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       filename=f"greenpack_results_{job_id[:8]}.xlsx")


@router.post("/{job_id}/print")
async def print_job_report(job_id: str, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    result = await db.execute(select(InspectionResult).where(InspectionResult.job_id == job_id))
    r = result.scalar_one_or_none()
    if not r or not r.report_pdf_path:
        raise HTTPException(status_code=404, detail="Report not found")
    success = print_report_windows(r.report_pdf_path)
    return {"printed": success}
