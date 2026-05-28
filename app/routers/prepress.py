"""Greenpack Pro v3.0 — Prepress Router (Simplified for Testing)"""
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings

router = APIRouter()
log = logging.getLogger(__name__)
settings = get_settings()


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
    """Compare trial proofs against final design - Simplified version"""
    
    job_id = str(uuid.uuid4())
    if not job_ref:
        job_ref = f"PREPRESS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Return mock response for testing
    return {
        "job_id": job_id,
        "job_ref": job_ref,
        "status": "queued",
        "trial_count": len(trial_proofs),
        "mode": "prepress_trial_comparison",
    }


@router.get("/{job_id}")
async def get_prepress_result(job_id: str):
    """Get prepress result - Mock response"""
    return {
        "job_id": job_id,
        "job_ref": f"PREPRESS-{job_id[:8]}",
        "status": "completed",
        "client_name": "Test Client",
        "product_name": "Test Product",
        "decision": "GO",
        "accuracy_score": 92.5,
        "trial_reports": [
            {
                "trial_idx": 1,
                "accuracy_score": 92.5,
                "passed": True,
                "scores": {"text": 95, "color": 90, "ssim": 88, "icon_size": 100, "expiry": 100},
                "error_summary": {"critical": [], "warning": [], "critical_count": 0, "warning_count": 1}
            }
        ],
        "processing_time_ms": 1234,
        "error_message": None,
    }


@router.post("/identify-colors")
async def identify_colors(
    file: UploadFile = File(...),
    k: int = Form(default=8),
    ignore_white: bool = Form(default=True),
    top_n_per_color: int = Form(default=5),
):
    """Identify Pantone colors - Simplified version"""
    
    return {
        "user_email": "test@example.com",
        "total_colors_found": 3,
        "extracted_colors": [
            {
                "hex": "#E03C31",
                "rgb": [224, 60, 49],
                "area_pct": 35,
                "best_match_code": "PANTONE 185 C",
                "best_match_delta_e": "1.2",
                "match_confidence": "high",
            }
        ],
        "library_size": 698,
        "method": "K-means clustering",
        "report_image_url": None,
    }


@router.get("/pantone-library")
async def list_pantone_library():
    """List Pantone library"""
    return {
        "version": "1.0",
        "total_colors": 698,
        "filtered_count": 698,
        "colors": [],
    }