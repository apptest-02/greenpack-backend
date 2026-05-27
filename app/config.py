"""
Greenpack Pro — Application Configuration
Loads settings from .env file using pydantic-settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    # App
    greenpack_mode: str = Field(default="standalone")
    greenpack_version: str = Field(default="1.0.0")
    app_name: str = Field(default="Greenpack Pro")
    app_secret_key: str = Field(default="dev-secret-change-in-production-64chars")

    # Database
    db_url: str = Field(default="sqlite+aiosqlite:///./data/greenpack.db")

    # Server
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=18080)
    allowed_origins: str = Field(default="http://localhost:5173,http://localhost:18080")

    # JWT
    jwt_secret_key: str = Field(default="jwt-secret-change-in-production-64chars")
    jwt_access_token_expire_minutes: int = Field(default=15)
    jwt_refresh_token_expire_days: int = Field(default=7)
    bcrypt_rounds: int = Field(default=12)

    # File Storage
    storage_type: str = Field(default="local")
    local_storage_root: str = Field(default="./files")
    reports_dir: str = Field(default="./reports")
    templates_dir: str = Field(default="./templates")
    temp_dir: str = Field(default="./temp")
    backup_dir: str = Field(default="./backups")

    # OCR
    tesseract_path: str = Field(default="tesseract")
    easyocr_model_dir: str = Field(default="./models")
    easyocr_download_enabled: bool = Field(default=True)
    ocr_timeout_seconds: int = Field(default=120)
    ocr_min_confidence: float = Field(default=0.50)

    # Inspection defaults
    default_color_tolerance_de: float = Field(default=2.0)
    default_ssim_threshold: float = Field(default=0.75)
    default_scan_resolution_dpi: int = Field(default=300)
    pdf_raster_dpi: int = Field(default=300)

    # Scanner
    dwt_service_url: str = Field(default="http://127.0.0.1:18622")
    dwt_license_key: str = Field(default="")

    # Backup & Retention
    backup_keep_days: int = Field(default=30)
    auto_backup_enabled: bool = Field(default=True)
    report_retention_days: int = Field(default=90)
    scan_image_retention_days: int = Field(default=30)
    disk_space_warning_gb: float = Field(default=5.0)
    disk_space_critical_gb: float = Field(default=2.0)

    # Notifications
    toast_notifications: bool = Field(default=True)
    email_notifications: bool = Field(default=False)
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587)
    smtp_username: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from: str = Field(default="greenpackpro@yourfactory.com")
    notify_on_fail_only: bool = Field(default=True)

    # Webhooks
    webhook_enabled: bool = Field(default=False)
    webhook_url: str = Field(default="")
    webhook_secret: str = Field(default="")
    webhook_retry_attempts: int = Field(default=3)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./logs/engine.log")
    windows_event_log: bool = Field(default=False)

    # License
    license_salt: str = Field(default="GreenpackPro-AuraTech-2025-DEFAULT")

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_standalone(self) -> bool:
        return self.greenpack_mode == "standalone"

    def ensure_directories(self):
        """Create all required directories if they don't exist"""
        dirs = [
            self.local_storage_root,
            self.reports_dir,
            self.templates_dir,
            self.temp_dir,
            self.backup_dir,
            self.easyocr_model_dir,
            Path(self.log_file).parent,
            "./data",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
