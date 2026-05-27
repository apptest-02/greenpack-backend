"""Greenpack Pro — Users Router"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from app.database import get_db
from app.models.base import User
from app.services.auth_service import get_current_user, require_admin, hash_password
import uuid

router = APIRouter()

class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "inspector"

@router.get("")
async def list_users(db: AsyncSession = Depends(get_db), current_user=Depends(require_admin())):
    result = await db.execute(select(User).where(User.company_id == current_user.company_id))
    users = result.scalars().all()
    return [{"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role,
             "active": u.active, "last_login": u.last_login.isoformat() if u.last_login else None}
            for u in users]

@router.post("")
async def create_user(req: CreateUserRequest, db: AsyncSession = Depends(get_db),
                      current_user=Depends(require_admin())):
    # Check email unique
    result = await db.execute(select(User).where(User.email == req.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(id=str(uuid.uuid4()), company_id=current_user.company_id,
                email=req.email, password_hash=hash_password(req.password),
                full_name=req.full_name, role=req.role)
    db.add(user)
    await db.commit()
    return {"id": user.id, "email": user.email, "role": user.role}

@router.patch("/{user_id}")
async def update_user(user_id: str, data: dict, db: AsyncSession = Depends(get_db),
                      current_user=Depends(require_admin())):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if "role" in data:
        user.role = data["role"]
    if "active" in data:
        user.active = data["active"]
    if "password" in data:
        user.password_hash = hash_password(data["password"])
    await db.commit()
    return {"id": user.id, "role": user.role, "active": user.active}
