"""
Greenpack Pro — FastAPI Application Entry Point
Starts the complete backend API server.
"""
import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path
from sqlalchemy import Integer, func
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app.routers import inspect
from app.config import get_settings
from app.database import init_db, check_db_integrity

log = logging.getLogger(__name__)
settings = get_settings()

# ── Configure Logging ──────────────────────────────────────────────────────────
def setup_logging():
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]

    try:
        from logging.handlers import RotatingFileHandler
        handlers.append(
            RotatingFileHandler(
                settings.log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
        )
    except Exception:
        pass

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=handlers,
    )


setup_logging()


# ── Startup/Shutdown ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup and shutdown"""
    log.info("=" * 60)
    log.info(f"Greenpack Pro v{settings.greenpack_version} starting")
    log.info(f"Mode: {settings.greenpack_mode}")
    log.info(f"Database: {settings.db_url[:50]}...")

    # Ensure directories exist
    settings.ensure_directories()

    # Check disk space
    from app.services.backup_service import check_disk_space
    disk = check_disk_space()
    if disk["status"] == "critical":
        log.warning(f"CRITICAL: Low disk space — {disk['free_gb']}GB free")
    elif disk["status"] == "warning":
        log.warning(f"Warning: Low disk space — {disk['free_gb']}GB free")

    # Initialize database
    await init_db()

    # Check SQLite integrity
    if "sqlite" in settings.db_url:
        ok = await check_db_integrity()
        if not ok:
            log.error("Database integrity check FAILED — consider restoring from backup")

    # Ensure default company + admin user exist
    await ensure_default_admin()

    # Pre-warm OCR model in background (first load takes ~10s)
    if settings.easyocr_download_enabled or Path(settings.easyocr_model_dir).exists():
        asyncio.create_task(_prewarm_ocr())

    log.info(f"API ready at http://{settings.api_host}:{settings.api_port}")
    log.info("=" * 60)

    yield

    # Shutdown
    log.info("Greenpack Pro shutting down")


async def _prewarm_ocr():
    """Pre-load EasyOCR model in background after startup"""
    await asyncio.sleep(2)  # Let app fully start first
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: __import__(
            "app.services.ocr_service", fromlist=["get_ocr_reader"]
        ).get_ocr_reader())
        log.info("OCR model pre-warmed successfully")
    except Exception as e:
        log.warning(f"OCR pre-warm failed (will load on first use): {e}")


async def ensure_default_admin():
    """Create default company and admin user if database is empty"""
    from app.database import AsyncSessionLocal
    from app.models.base import Company, User
    from app.services.auth_service import hash_password
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        # Check if any users exist
        result = await db.execute(select(User).limit(1))
        if result.scalar_one_or_none():
            return  # Already initialized

        log.info("First run: creating default company and admin user")

        # Create default company
        company = Company(
            name="My Factory",
            plan="solo",
            settings={},
        )
        db.add(company)
        await db.flush()

        # Create admin user
        admin = User(
            company_id=company.id,
            email="admin@greenpackpro.local",
            password_hash=hash_password("Admin123!"),
            full_name="System Administrator",
            role="admin",
        )
        db.add(admin)
        await db.commit()

        log.info(
            "Default admin created: admin@greenpackpro.local / Admin123! "
            "(CHANGE THIS PASSWORD IMMEDIATELY)"
        )


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Greenpack Pro API",
    description="Label Print Inspection & Verification System",
    version=settings.greenpack_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS - Allow frontend domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://test.copypaste.vendagas.com",
        "https://test.copypaste.vendagas.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:18080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Include Routers ────────────────────────────────────────────────────────────
from app.routers import auth, users, jobs, templates, scanners, batch, reports, settings_router, multi_up_simple as multi_up, prepress

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
app.include_router(multi_up.router, prefix="/api/v1/jobs", tags=["Multi-Up"])
app.include_router(prepress.router, prefix="/api/v1/prepress", tags=["Prepress"])
app.include_router(templates.router, prefix="/api/v1/templates", tags=["Templates"])
app.include_router(scanners.router, prefix="/api/v1/scanners", tags=["Scanners"])
app.include_router(batch.router, prefix="/api/v1/batch", tags=["Batch"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])
app.include_router(settings_router.router, prefix="/api/v1/settings", tags=["Settings"])
app.include_router(inspect.router, prefix="/api/v1", tags=["Inspection"])

# ── Health & Status ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    from app.services.backup_service import check_disk_space
    disk = check_disk_space()
    return {
        "status": "ok",
        "version": settings.greenpack_version,
        "mode": settings.greenpack_mode,
        "disk": disk,
    }


@app.get("/api/v1/dashboard/stats")
async def dashboard_stats():
    """Dashboard statistics — pass rates, job counts"""
    from app.database import AsyncSessionLocal
    from app.models.base import InspectionJob
    from sqlalchemy import select, func, case
    from datetime import datetime

    async with AsyncSessionLocal() as db:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Correct SQLAlchemy CASE syntax
        result = await db.execute(
            select(
                func.count(InspectionJob.id).label("total"),
                func.sum(
                    case((InspectionJob.pass_fail == True, 1), else_=0)
                ).label("passed"),
                func.avg(InspectionJob.overall_score).label("avg_score"),
            ).where(InspectionJob.created_at >= today)
        )
        row = result.one()

        total = row.total or 0
        passed = int(row.passed or 0)
        avg_score = round(float(row.avg_score or 0), 1)

        return {
            "today_total": total,
            "today_pass": passed,
            "today_fail": total - passed,
            "pass_rate": round((passed / total * 100) if total > 0 else 0, 1),
            "avg_score": avg_score,
        }


# ── Serve static files for reports ────────────────────────────────────────────
reports_dir = Path(settings.reports_dir)
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")


# ── Entry Point for Railway ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Railway uses PORT environment variable
    port = int(os.environ.get("PORT", settings.api_port))
    host = "0.0.0.0"  # Listen on all interfaces
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
