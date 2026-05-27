"""Greenpack Pro — Auth Router (NO AUTH)"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.database import get_db
from app.models.base import User, AuditLog, Company
from app.services.auth_service import create_access_token, create_refresh_token, hash_password
from datetime import datetime
import uuid

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str  # Not used, but kept for compatibility

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str
    full_name: str

@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Get or create user - NO PASSWORD CHECK
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()
    
    if not user:
        # Get or create company
        company_result = await db.execute(select(Company).limit(1))
        company = company_result.scalar_one_or_none()
        if not company:
            company = Company(id=str(uuid.uuid4()), name="Default Company")
            db.add(company)
            await db.commit()
            await db.refresh(company)
        
        # Create user with ANY email
        user = User(
            id=str(uuid.uuid4()),
            email=req.email,
            password_hash=hash_password("any"),
            full_name=req.email.split('@')[0] if '@' in req.email else req.email,
            role="admin",
            active=True,
            company_id=company.id
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    
    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()
    
    return TokenResponse(
        access_token=create_access_token(user.id, user.email, user.role),
        refresh_token=create_refresh_token(user.id),
        user_id=user.id,
        email=user.email,
        role=user.role,
        full_name=user.full_name or "",
    )

@router.get("/me")
async def me(current_user: User = Depends(lambda: None)):  # No auth needed
    return {"id": "test", "email": "test@test.com", "full_name": "Test User", "role": "admin"}