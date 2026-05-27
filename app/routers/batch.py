"""Greenpack Pro — Batch Router"""
import asyncio
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from app.database import get_db
from app.models.base import BatchQueue, InspectionJob, BatchStatus, JobStatus
from app.services.auth_service import get_current_user, require_manager

router = APIRouter()
log = logging.getLogger(__name__)

class BatchItem(BaseModel):
    master_path: str
    scan_path: str
    job_ref: str = ""
    client_name: str = ""
    product_name: str = ""
    color_threshold: float = 2.0
    ssim_threshold: float = 0.75

class CreateBatchRequest(BaseModel):
    name: str
    items: list[BatchItem]
    notify_email: Optional[str] = None

@router.post("")
async def create_batch(req: CreateBatchRequest, db: AsyncSession = Depends(get_db),
                       current_user=Depends(require_manager())):
    batch_id = str(uuid.uuid4())

    # Create batch record
    batch = BatchQueue(
        id=batch_id, company_id=current_user.company_id,
        created_by=current_user.id, name=req.name,
        status=BatchStatus.pending, total_jobs=len(req.items),
        notify_email=req.notify_email,
    )
    db.add(batch)

    # Create individual jobs
    job_ids = []
    for item in req.items:
        job_id = str(uuid.uuid4())
        job = InspectionJob(
            id=job_id, company_id=current_user.company_id,
            created_by=current_user.id, batch_id=batch_id,
            job_ref=item.job_ref or f"BATCH-{batch_id[:8]}-{len(job_ids)+1:03d}",
            client_name=item.client_name, product_name=item.product_name,
            master_file_path=item.master_path, scan_file_path=item.scan_path,
            input_source="batch", status=JobStatus.queued,
        )
        db.add(job)
        job_ids.append(job_id)

    await db.commit()

    # Start batch processing in background
    asyncio.create_task(_process_batch(batch_id, job_ids, req.items, current_user.id))

    return {"batch_id": batch_id, "total_jobs": len(req.items), "status": "pending"}

@router.get("/{batch_id}")
async def get_batch(batch_id: str, db: AsyncSession = Depends(get_db),
                    current_user=Depends(get_current_user)):
    result = await db.execute(select(BatchQueue).where(BatchQueue.id == batch_id))
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get job counts
    jobs_result = await db.execute(select(InspectionJob).where(InspectionJob.batch_id == batch_id))
    jobs = jobs_result.scalars().all()
    completed = sum(1 for j in jobs if j.status == JobStatus.completed)
    failed = sum(1 for j in jobs if j.status == JobStatus.failed)

    return {
        "id": batch.id, "name": batch.name, "status": batch.status,
        "total_jobs": batch.total_jobs, "completed_jobs": completed,
        "failed_jobs": failed, "pending_jobs": batch.total_jobs - completed - failed,
        "pct_complete": round((completed / batch.total_jobs * 100) if batch.total_jobs else 0, 1),
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "started_at": batch.started_at.isoformat() if batch.started_at else None,
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
    }

@router.get("/{batch_id}/jobs")
async def get_batch_jobs(batch_id: str, db: AsyncSession = Depends(get_db),
                         current_user=Depends(get_current_user)):
    result = await db.execute(select(InspectionJob).where(InspectionJob.batch_id == batch_id))
    jobs = result.scalars().all()
    return [{
        "id": j.id, "job_ref": j.job_ref, "status": j.status,
        "overall_score": j.overall_score, "pass_fail": j.pass_fail,
        "product_name": j.product_name, "error_message": j.error_message,
    } for j in jobs]

async def _process_batch(batch_id: str, job_ids: list, items: list, user_id: str):
    """Sequential batch processing (Mode A: one at a time to manage RAM)"""
    from app.database import AsyncSessionLocal
    from app.services.inspection_engine import get_engine, InspectionError
    from datetime import datetime

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(BatchQueue).where(BatchQueue.id == batch_id))
        batch = result.scalar_one_or_none()
        if batch:
            batch.status = BatchStatus.running
            batch.started_at = datetime.utcnow()
            await db.commit()

    engine = get_engine()
    failed_count = 0

    for job_id, item in zip(job_ids, items):
        config = {
            "job_id": job_id, "job_ref": item.job_ref,
            "client_name": item.client_name, "product_name": item.product_name,
            "color_threshold": item.color_threshold, "ssim_threshold": item.ssim_threshold,
            "barcode_rules": [],
        }
        try:
            from app.routers.jobs import _run_job
            await _run_job(job_id, item.master_path, item.scan_path, config)
        except Exception as e:
            log.error(f"Batch job {job_id} failed: {e}")
            failed_count += 1

        # Small delay between jobs to allow memory cleanup
        await asyncio.sleep(0.5)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(BatchQueue).where(BatchQueue.id == batch_id))
        batch = result.scalar_one_or_none()
        if batch:
            batch.status = BatchStatus.partial_fail if failed_count > 0 else BatchStatus.completed
            batch.completed_at = datetime.utcnow()
            batch.failed_jobs = failed_count
            await db.commit()

    log.info(f"Batch {batch_id} complete: {len(job_ids) - failed_count} passed, {failed_count} failed")
