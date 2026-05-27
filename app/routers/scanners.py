"""Greenpack Pro — Scanners Router"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.auth_service import get_current_user
from app.services.scanner_service import (
    list_scanners, capture_from_scanner, get_camera_preview_frame
)

router = APIRouter()

class CaptureRequest(BaseModel):
    device_id: str
    resolution: int = 300
    output_dir: str = None

@router.get("")
async def get_scanners(current_user=Depends(get_current_user)):
    """List all connected TWAIN/WIA scanners and webcams"""
    scanners = list_scanners()
    return {"scanners": scanners, "count": len(scanners)}

@router.post("/capture")
async def capture_scan(req: CaptureRequest, current_user=Depends(get_current_user)):
    """Trigger scanner capture — returns path to saved image"""
    try:
        image_path = capture_from_scanner(req.device_id, req.resolution, req.output_dir)
        return {"image_path": image_path, "status": "captured"}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@router.get("/{device_id}/preview")
async def get_preview(device_id: str, current_user=Depends(get_current_user)):
    """Get single live preview frame as base64 JPEG"""
    frame_b64 = get_camera_preview_frame(device_id)
    if frame_b64 is None:
        raise HTTPException(status_code=503, detail="Preview unavailable for this device")
    return {"frame": frame_b64, "format": "jpeg"}
