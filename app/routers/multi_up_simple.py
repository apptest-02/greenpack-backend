"""Greenpack Pro v2.0 — Multi-Up Jobs Router (Database-Free Version)"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services.multi_up_inspection import get_multi_engine

router = APIRouter()
log = logging.getLogger(__name__)
settings = get_settings()

# Simple file-based storage for results
RESULTS_DIR = Path("/tmp/multi_up_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/multi-up")
async def create_multi_up_job(
    master_file: UploadFile = File(...),
    scan_file: UploadFile = File(...),
    job_ref: str = Form(default=""),
    client_name: str = Form(default=""),
    product_name: str = Form(default=""),
    expected_count: Optional[int] = Form(default=None),
    is_transparent: bool = Form(default=False),
    color_threshold: float = Form(default=2.0),
    ssim_threshold: float = Form(default=0.75),
    check_braille: bool = Form(default=False),
    check_font_size: bool = Form(default=False),
    min_font_size_pt: float = Form(default=6.0),
    spell_check: bool = Form(default=False),
):
    """Create a multi-up inspection job - NO DATABASE VERSION"""
    job_id = str(uuid.uuid4())
    if not job_ref:
        job_ref = f"MULTI-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Save files temporarily
    files_dir = Path(settings.local_storage_root) / "jobs" / job_id
    files_dir.mkdir(parents=True, exist_ok=True)
    
    master_suffix = Path(master_file.filename).suffix or ".pdf"
    scan_suffix = Path(scan_file.filename).suffix or ".jpg"
    master_path = files_dir / f"master{master_suffix}"
    scan_path = files_dir / f"scan{scan_suffix}"
    
    master_data = await master_file.read()
    scan_data = await scan_file.read()
    
    with open(master_path, "wb") as f:
        f.write(master_data)
    with open(scan_path, "wb") as f:
        f.write(scan_data)
    
    config = {
        "job_id": job_id,
        "job_ref": job_ref,
        "client_name": client_name,
        "product_name": product_name,
        "inspector_name": "Test User",
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


@router.get("/{job_id}/multi-up")
async def get_multi_up_result(job_id: str):
    """Get multi-up inspection result"""
    result_file = RESULTS_DIR / f"{job_id}.json"
    if not result_file.exists():
        return {
            "status": "processing", 
            "job_id": job_id, 
            "message": "Result not ready yet. Please wait 10-30 seconds."
        }
    
    with open(result_file, "r") as f:
        result = json.load(f)
    return result


async def _run_multi_up_job(job_id: str, master_path: str, scan_path: str, config: dict):
    """Background worker for multi-up inspection"""
    result_file = RESULTS_DIR / f"{job_id}.json"
    
    try:
        engine = get_multi_engine()
        res = await engine.inspect_sheet(
            job_id=job_id,
            master_path=master_path,
            scan_path=scan_path,
            config=config,
        )
        log.info(f"Multi-up job {job_id} complete: {res['labels_passed']}/{res['labels_found']} passed")
        
        # Save result to file
        with open(result_file, "w") as f:
            json.dump(res, f, default=str, indent=2)
            
    except Exception as e:
        log.exception(f"Multi-up job {job_id} failed: {e}")
        with open(result_file, "w") as f:
            json.dump({"status": "failed", "job_id": job_id, "error": str(e)}, f)