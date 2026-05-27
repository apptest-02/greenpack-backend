"""
Greenpack Pro — Database Models
All SQLAlchemy models. Same schema works for SQLite (Mode A) and PostgreSQL (Mode B).
"""
from sqlalchemy import (
    Column, String, Float, Boolean, Integer, DateTime, Text, JSON,
    ForeignKey, Enum as SAEnum, BigInteger, func
)
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
import uuid
import enum


def gen_uuid():
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    inspector = "inspector"
    manager = "manager"
    admin = "admin"
    client = "client"


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class InputSource(str, enum.Enum):
    upload = "upload"
    scanner = "scanner"
    batch = "batch"


class BatchStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    partial_fail = "partial_fail"
    cancelled = "cancelled"


# ── Models ────────────────────────────────────────────────────────────────────

class Company(Base):
    __tablename__ = "companies"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(200), nullable=False)
    logo_path = Column(Text)
    plan = Column(String(50), default="solo")
    settings = Column(JSON, default=dict)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="company")
    label_templates = relationship("LabelTemplate", back_populates="company")
    inspection_jobs = relationship("InspectionJob", back_populates="company")
    client_branding = relationship("ClientBranding", back_populates="company")


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    company_id = Column(String(36), ForeignKey("companies.id"), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    full_name = Column(String(200))
    role = Column(String(50), default=UserRole.inspector)
    last_login = Column(DateTime)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="users")
    created_jobs = relationship("InspectionJob", back_populates="created_by_user")


class LabelTemplate(Base):
    __tablename__ = "label_templates"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    company_id = Column(String(36), ForeignKey("companies.id"), nullable=False)
    client_name = Column(String(200), nullable=False)
    product_name = Column(String(300), nullable=False)
    version = Column(String(50), default="1.0")
    file_path = Column(Text, nullable=False)
    thumbnail_path = Column(Text)
    color_threshold = Column(Float, default=2.0)
    ssim_threshold = Column(Float, default=0.75)
    barcode_rules = Column(JSON, default=list)
    ocr_rules = Column(JSON, default=dict)
    active = Column(Boolean, default=True)
    created_by = Column(String(36), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="label_templates")
    jobs = relationship("InspectionJob", back_populates="template")


class InspectionJob(Base):
    __tablename__ = "inspection_jobs"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    company_id = Column(String(36), ForeignKey("companies.id"), nullable=False)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    template_id = Column(String(36), ForeignKey("label_templates.id"))
    batch_id = Column(String(36), ForeignKey("batch_queues.id"))
    job_ref = Column(String(100), index=True)
    client_name = Column(String(200))
    product_name = Column(String(300))
    master_file_path = Column(Text, nullable=False)
    scan_file_path = Column(Text, nullable=False)
    input_source = Column(String(50), default=InputSource.upload)
    status = Column(String(50), default=JobStatus.queued, index=True)
    overall_score = Column(Float)
    pass_fail = Column(Boolean)
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)

    company = relationship("Company", back_populates="inspection_jobs")
    created_by_user = relationship("User", back_populates="created_jobs")
    template = relationship("LabelTemplate", back_populates="jobs")
    result = relationship("InspectionResult", back_populates="job", uselist=False)
    scan_session = relationship("ScanSession", back_populates="job", uselist=False)


class InspectionResult(Base):
    __tablename__ = "inspection_results"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    job_id = Column(String(36), ForeignKey("inspection_jobs.id"), unique=True, nullable=False)
    ocr_errors = Column(JSON, default=list)
    color_results = Column(JSON, default=list)
    barcode_results = Column(JSON, default=list)
    defects = Column(JSON, default=list)
    ssim_score = Column(Float)
    alignment_confidence = Column(Float)
    ocr_score = Column(Float)
    color_score = Column(Float)
    ssim_score_weighted = Column(Float)
    barcode_score = Column(Float)
    overall_score = Column(Float)
    pass_fail = Column(Boolean)
    annotated_image_path = Column(Text)
    report_pdf_path = Column(Text)
    excel_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("InspectionJob", back_populates="result")


class ScanSession(Base):
    __tablename__ = "scan_sessions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    job_id = Column(String(36), ForeignKey("inspection_jobs.id"), unique=True)
    scanner_name = Column(String(200))
    scanner_type = Column(String(50))
    resolution_dpi = Column(Integer)
    pixel_type = Column(String(20))
    scan_timestamp = Column(DateTime, default=datetime.utcnow)
    file_path = Column(Text)

    job = relationship("InspectionJob", back_populates="scan_session")


class BatchQueue(Base):
    __tablename__ = "batch_queues"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    company_id = Column(String(36), ForeignKey("companies.id"), nullable=False)
    created_by = Column(String(36), ForeignKey("users.id"))
    name = Column(String(200), nullable=False)
    status = Column(String(50), default=BatchStatus.pending)
    total_jobs = Column(Integer, default=0)
    completed_jobs = Column(Integer, default=0)
    failed_jobs = Column(Integer, default=0)
    notify_email = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    jobs = relationship("InspectionJob", back_populates=None)


class ClientBranding(Base):
    __tablename__ = "client_branding"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    company_id = Column(String(36), ForeignKey("companies.id"), nullable=False)
    client_name = Column(String(200), nullable=False)
    logo_path = Column(Text)
    brand_color = Column(String(7), default="#0D1B2A")
    report_title = Column(String(300))
    footer_text = Column(Text)
    watermark_text = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="client_branding")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)  # Changed from BigInteger
    company_id = Column(String(36), ForeignKey("companies.id"))
    user_id = Column(String(36), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(Text)
    ip_address = Column(String(45))
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
