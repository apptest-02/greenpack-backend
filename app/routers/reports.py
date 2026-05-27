"""Greenpack Pro — Reports Router"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.base import InspectionJob, InspectionResult, JobStatus
from app.services.auth_service import get_current_user

router = APIRouter()

@router.get("")
async def list_reports(
    limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = (select(InspectionJob, InspectionResult)
         .join(InspectionResult, InspectionResult.job_id == InspectionJob.id, isouter=True)
         .where(InspectionJob.company_id == current_user.company_id,
                InspectionJob.status == JobStatus.completed)
         .order_by(desc(InspectionJob.created_at))
         .limit(limit).offset(offset))

    result = await db.execute(q)
    rows = result.all()

    return [{
        "job_id": job.id, "job_ref": job.job_ref,
        "client_name": job.client_name, "product_name": job.product_name,
        "overall_score": job.overall_score, "pass_fail": job.pass_fail,
        "has_pdf": bool(res and res.report_pdf_path),
        "has_excel": bool(res and res.excel_path),
        "created_at": job.created_at.isoformat() if job.created_at else None,
    } for job, res in rows]
