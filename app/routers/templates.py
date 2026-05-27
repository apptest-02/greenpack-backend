"""Greenpack Pro — Template Router"""
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.base import LabelTemplate
from app.services.auth_service import get_current_user, require_manager
from app.config import get_settings
import json

router = APIRouter()
settings = get_settings()


@router.get("")
async def list_templates(
    client: str = None, search: str = None,
    db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)
):
    q = select(LabelTemplate).where(
        LabelTemplate.company_id == current_user.company_id,
        LabelTemplate.active == True
    )
    if client:
        q = q.where(LabelTemplate.client_name.ilike(f"%{client}%"))
    if search:
        q = q.where(LabelTemplate.product_name.ilike(f"%{search}%"))
    result = await db.execute(q)
    templates = result.scalars().all()
    return [{
        "id": t.id, "client_name": t.client_name, "product_name": t.product_name,
        "version": t.version, "thumbnail_path": t.thumbnail_path,
        "color_threshold": t.color_threshold, "ssim_threshold": t.ssim_threshold,
        "barcode_rules": t.barcode_rules or [],
        "created_at": t.created_at.isoformat() if t.created_at else None,
    } for t in templates]


@router.post("")
async def create_template(
    file: UploadFile = File(...),
    client_name: str = Form(...),
    product_name: str = Form(...),
    version: str = Form(default="1.0"),
    color_threshold: float = Form(default=2.0),
    ssim_threshold: float = Form(default=0.75),
    barcode_rules: str = Form(default="[]"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_manager()),
):
    template_id = str(uuid.uuid4())
    templates_dir = Path(settings.templates_dir) / template_id
    templates_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".pdf"
    file_path = templates_dir / f"master{suffix}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Generate thumbnail
    thumb_path = _generate_thumbnail(str(file_path), str(templates_dir))

    try:
        rules = json.loads(barcode_rules)
    except Exception:
        rules = []

    template = LabelTemplate(
        id=template_id, company_id=current_user.company_id,
        client_name=client_name, product_name=product_name, version=version,
        file_path=str(file_path), thumbnail_path=thumb_path,
        color_threshold=color_threshold, ssim_threshold=ssim_threshold,
        barcode_rules=rules, created_by=current_user.id,
    )
    db.add(template)
    await db.commit()
    return {"id": template_id, "client_name": client_name, "product_name": product_name}


@router.get("/{template_id}")
async def get_template(template_id: str, db: AsyncSession = Depends(get_db),
                       current_user=Depends(get_current_user)):
    result = await db.execute(select(LabelTemplate).where(LabelTemplate.id == template_id))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"id": t.id, "client_name": t.client_name, "product_name": t.product_name,
            "version": t.version, "file_path": t.file_path, "thumbnail_path": t.thumbnail_path,
            "color_threshold": t.color_threshold, "ssim_threshold": t.ssim_threshold,
            "barcode_rules": t.barcode_rules or []}


@router.delete("/{template_id}")
async def deactivate_template(template_id: str, db: AsyncSession = Depends(get_db),
                              current_user=Depends(require_manager())):
    result = await db.execute(select(LabelTemplate).where(LabelTemplate.id == template_id))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
    t.active = False
    await db.commit()
    return {"deleted": True}


def _generate_thumbnail(file_path: str, out_dir: str) -> str | None:
    try:
        from pdf2image import convert_from_path
        import cv2
        p = Path(file_path)
        if p.suffix.lower() == ".pdf":
            images = convert_from_path(file_path, dpi=72, first_page=1, last_page=1)
            if images:
                thumb_path = Path(out_dir) / "thumbnail.jpg"
                img = images[0].resize((200, 140))
                img.save(str(thumb_path), "JPEG", quality=85)
                return str(thumb_path)
        else:
            import cv2
            img = cv2.imread(file_path)
            if img is not None:
                thumb = cv2.resize(img, (200, 140))
                thumb_path = Path(out_dir) / "thumbnail.jpg"
                cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return str(thumb_path)
    except Exception:
        pass
    return None
