"""
Greenpack Pro — Backup Service
Automated SQLite database backup with rotation.
Also handles data retention and disk space management.
"""
import logging
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


def create_backup() -> Path:
    """
    Create a hot backup of the SQLite database.
    Safe to run while the app is serving requests (SQLite online backup API).
    Returns path to backup file.
    """
    backup_dir = Path(settings.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"greenpack_{ts}.db"

    db_path = _get_db_path()
    if not db_path.exists():
        log.warning(f"Database not found at {db_path}, skipping backup")
        return backup_path

    # SQLite online backup (safe during writes)
    try:
        src = sqlite3.connect(str(db_path))
        dst = sqlite3.connect(str(backup_path))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        log.info(f"Database backup created: {backup_path} ({backup_path.stat().st_size / 1024:.1f}KB)")
    except Exception as e:
        log.error(f"Backup failed: {e}")
        raise

    # Rotate old backups
    rotate_backups()

    return backup_path


def rotate_backups():
    """Delete oldest backups when count exceeds limit"""
    backup_dir = Path(settings.backup_dir)
    backups = sorted(backup_dir.glob("greenpack_*.db"))
    while len(backups) > settings.backup_keep_days:
        oldest = backups.pop(0)
        oldest.unlink()
        log.info(f"Rotated old backup: {oldest}")


def restore_backup(backup_path: str) -> bool:
    """Restore from a backup file"""
    backup = Path(backup_path)
    if not backup.exists():
        log.error(f"Backup file not found: {backup_path}")
        return False

    db_path = _get_db_path()

    # Create safety backup before restore
    safety = db_path.parent / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    if db_path.exists():
        shutil.copy2(db_path, safety)
        log.info(f"Safety backup created: {safety}")

    # Restore
    shutil.copy2(backup, db_path)
    log.info(f"Database restored from {backup_path}")
    return True


def verify_backup(backup_path: str) -> bool:
    """Verify backup file integrity"""
    try:
        conn = sqlite3.connect(backup_path)
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        return result[0] == "ok"
    except Exception as e:
        log.error(f"Backup verification failed: {e}")
        return False


def list_backups() -> list[dict]:
    """List all available backups with metadata"""
    backup_dir = Path(settings.backup_dir)
    if not backup_dir.exists():
        return []

    backups = []
    for f in sorted(backup_dir.glob("greenpack_*.db"), reverse=True):
        stat = f.stat()
        verified = verify_backup(str(f))
        backups.append({
            "filename": f.name,
            "path": str(f),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "verified": verified,
        })
    return backups


def check_disk_space() -> dict:
    """Check available disk space"""
    try:
        total, used, free = shutil.disk_usage(Path.cwd().anchor)
        free_gb = free / (1024 ** 3)
        return {
            "free_gb": round(free_gb, 1),
            "total_gb": round(total / (1024 ** 3), 1),
            "used_gb": round(used / (1024 ** 3), 1),
            "status": (
                "critical" if free_gb < settings.disk_space_critical_gb
                else "warning" if free_gb < settings.disk_space_warning_gb
                else "ok"
            ),
            "message": f"{free_gb:.1f} GB free",
        }
    except Exception as e:
        log.error(f"Disk space check failed: {e}")
        return {"free_gb": 0, "status": "unknown", "message": str(e)}


async def cleanup_old_files():
    """Delete old reports and scans per retention policy"""
    now = datetime.now()
    deleted_count = 0
    freed_bytes = 0

    # Delete old reports
    reports_dir = Path(settings.reports_dir)
    if reports_dir.exists():
        cutoff = now - timedelta(days=settings.report_retention_days)
        for f in reports_dir.glob("*.pdf"):
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                size = f.stat().st_size
                f.unlink()
                deleted_count += 1
                freed_bytes += size
        for f in reports_dir.glob("*.xlsx"):
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                size = f.stat().st_size
                f.unlink()
                deleted_count += 1
                freed_bytes += size

    # Clean temp files (always clean files > 24 hours)
    temp_dir = Path(settings.temp_dir)
    if temp_dir.exists():
        cutoff = now - timedelta(hours=24)
        for f in temp_dir.iterdir():
            if f.is_file() and datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                size = f.stat().st_size
                f.unlink()
                deleted_count += 1
                freed_bytes += size

    log.info(
        f"Cleanup: deleted {deleted_count} files, "
        f"freed {freed_bytes / (1024*1024):.1f} MB"
    )
    return {"deleted_files": deleted_count, "freed_mb": round(freed_bytes / (1024*1024), 1)}


def _get_db_path() -> Path:
    """Extract database file path from connection URL"""
    db_url = settings.db_url
    if "sqlite" in db_url:
        # sqlite+aiosqlite:///./data/greenpack.db → ./data/greenpack.db
        path_part = db_url.split("///")[-1]
        return Path(path_part)
    return Path("./data/greenpack.db")
