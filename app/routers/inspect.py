"""
Greenpack Pro — Single Label Inspection Router
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services.ocr_service import get_ocr_reader, run_dual_ocr, diff_text_regions
from app.services.alignment import align_images
from app.services.color_service import analyze_color_zones
from app.services.ssim_service import detect_defects
from app.services.barcode_service import verify_barcodes
from app.services.preprocess import rasterize_pdf, preprocess_image

router = APIRouter()
log = logging.getLogger(__name__)
settings = get_settings()


@router.post("/inspect")
async def inspect_single_label(
    master: UploadFile = File(..., description="Master label image or PDF"),
    scan: UploadFile = File(..., description="Scanned printed label"),
):
    """
    Inspect a single printed label against the master design.
    Returns OCR, color, barcode, and defect analysis.
    """
    import cv2
    import numpy as np
    
    # Save uploaded files
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    master_suffix = Path(master.filename).suffix or ".jpg"
    scan_suffix = Path(scan.filename).suffix or ".jpg"
    
    master_path = temp_dir / f"master_{uuid.uuid4().hex[:8]}{master_suffix}"
    scan_path = temp_dir / f"scan_{uuid.uuid4().hex[:8]}{scan_suffix}"
    
    with open(master_path, "wb") as f:
        f.write(await master.read())
    with open(scan_path, "wb") as f:
        f.write(await scan.read())
    
    try:
        # Convert PDFs to images if needed
        if master_path.suffix.lower() == ".pdf":
            master_img_path = rasterize_pdf(master_path, dpi=300)
        else:
            master_img_path = preprocess_image(master_path)
        
        if scan_path.suffix.lower() == ".pdf":
            scan_img_path = rasterize_pdf(scan_path, dpi=300)
        else:
            scan_img_path = preprocess_image(scan_path)
        
        # Load images
        master_img = cv2.imread(str(master_img_path))
        scan_img = cv2.imread(str(scan_img_path))
        
        if master_img is None or scan_img is None:
            raise HTTPException(status_code=400, detail="Could not read images")
        
        # Align images
        aligned_scan, align_confidence = align_images(master_img, scan_img)
        
        # OCR analysis
        ocr_errors = []
        try:
            # Save temp files for OCR
            temp_master = temp_dir / f"ocr_master_{uuid.uuid4().hex[:8]}.png"
            temp_scan = temp_dir / f"ocr_scan_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(temp_master), master_img)
            cv2.imwrite(str(temp_scan), aligned_scan)
            
            ocr_result = run_dual_ocr(str(temp_master), str(temp_scan))
            ocr_errors = diff_text_regions(
                ocr_result.get("master_regions", []),
                ocr_result.get("scan_regions", []),
            )
            
            # Cleanup
            try:
                temp_master.unlink()
                temp_scan.unlink()
            except:
                pass
        except Exception as e:
            log.warning(f"OCR failed: {e}")
        
        # Color analysis
        color_result = analyze_color_zones(master_img, aligned_scan, threshold=2.0)
        
        # SSIM defect detection
        ssim_result = detect_defects(master_img, aligned_scan, threshold=0.75)
        
        # Barcode verification
        barcode_results = []
        try:
            barcode_results = verify_barcodes(str(scan_img_path), [])
        except Exception as e:
            log.warning(f"Barcode detection failed: {e}")
        
        # Calculate scores
        ocr_penalty = sum(15 if e.get("severity") == "high" else 10 if e.get("severity") == "medium" else 5 for e in ocr_errors)
        ocr_score = max(0, 100 - min(100, ocr_penalty))
        
        color_pct_fail = sum(1 for z in color_result.get("zone_results", []) if not z.get("pass", True))
        color_zones = len(color_result.get("zone_results", [])) or 1
        color_score = max(0, 100 - (color_pct_fail / color_zones) * 100)
        
        ssim_score = ssim_result.get("ssim_score", 0.85) * 100
        
        barcode_passed = sum(1 for b in barcode_results if b.get("pass"))
        barcode_total = len(barcode_results) or 1
        barcode_score = (barcode_passed / barcode_total) * 100 if barcode_total > 0 else 100
        
        # Overall score (weighted average)
        overall_score = (
            ocr_score * 0.35 + 
            color_score * 0.30 + 
            ssim_score * 0.20 + 
            barcode_score * 0.15
        )
        
        # Cleanup temp files
        try:
            master_path.unlink()
            scan_path.unlink()
            if master_img_path != master_path:
                master_img_path.unlink()
            if scan_img_path != scan_path:
                scan_img_path.unlink()
        except:
            pass
        
        return {
            "overall_score": round(overall_score, 1),
            "ocr_score": round(ocr_score, 1),
            "color_score": round(color_score, 1),
            "ssim_score": round(ssim_score, 1),
            "barcode_score": round(barcode_score, 1),
            "alignment_confidence": round(align_confidence, 3),
            "ocr_errors": ocr_errors[:20],  # Limit to 20 errors
            "defects": ssim_result.get("defects", [])[:20],
            "barcode_results": barcode_results,
            "color_results": color_result.get("zone_results", [])
        }
        
    except Exception as e:
        log.exception(f"Inspection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")
    finally:
        # Cleanup
        try:
            master_path.unlink()
            scan_path.unlink()
        except:
            pass


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "inspection"}