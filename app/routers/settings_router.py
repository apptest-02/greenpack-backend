"""Greenpack Pro — Settings Router"""
from fastapi import APIRouter, Depends
from app.services.auth_service import get_current_user, require_admin
from app.services.backup_service import (
    create_backup, list_backups, check_disk_space, cleanup_old_files
)

router = APIRouter()

@router.get("")
async def get_settings_info(current_user=Depends(get_current_user)):
    from app.config import get_settings
    s = get_settings()
    return {
        "mode": s.greenpack_mode, "version": s.greenpack_version,
        "default_color_threshold": s.default_color_tolerance_de,
        "default_ssim_threshold": s.default_ssim_threshold,
        "default_scan_dpi": s.default_scan_resolution_dpi,
        "email_notifications": s.email_notifications,
        "toast_notifications": s.toast_notifications,
        "webhook_enabled": s.webhook_enabled,
        "backup_enabled": s.auto_backup_enabled,
        "report_retention_days": s.report_retention_days,
        "disk": check_disk_space(),
    }

@router.get("/backups")
async def get_backups(current_user=Depends(require_admin())):
    return {"backups": list_backups()}

@router.post("/backups")
async def trigger_backup(current_user=Depends(require_admin())):
    path = create_backup()
    return {"backup_path": str(path), "status": "created"}

@router.post("/cleanup")
async def run_cleanup(current_user=Depends(require_admin())):
    result = await cleanup_old_files()
    return result

@router.get("/disk")
async def disk_info(current_user=Depends(get_current_user)):
    return check_disk_space()
