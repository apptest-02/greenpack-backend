# -*- mode: python ; coding: utf-8 -*-
# Greenpack Pro — PyInstaller Build Spec
# Build: pyinstaller greenpack_engine.spec --clean
# Output: dist/greenpack_engine/greenpack_engine.exe

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

block_cipher = None

# ── Hidden imports that PyInstaller misses ─────────────────────────────────────
hidden_imports = [
    # FastAPI / Uvicorn / ASGI
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.middleware',
    'uvicorn.middleware.proxy_headers',
    'fastapi',
    'fastapi.middleware',
    'fastapi.middleware.cors',
    'fastapi.staticfiles',
    'starlette',
    'starlette.routing',
    'starlette.middleware',
    'anyio',
    'anyio._backends._asyncio',

    # Pydantic
    'pydantic',
    'pydantic.deprecated',
    'pydantic_settings',
    'pydantic_core',
    'email_validator',

    # Database
    'sqlalchemy',
    'sqlalchemy.dialects',
    'sqlalchemy.dialects.sqlite',
    'sqlalchemy.dialects.sqlite.aiosqlite',
    'sqlalchemy.ext.asyncio',
    'aiosqlite',
    'alembic',
    'alembic.runtime',
    'alembic.runtime.migration',
    'alembic.operations',

    # Auth
    'jose',
    'jose.jwt',
    'jose.jws',
    'jose.jwk',
    'passlib',
    'passlib.context',
    'passlib.handlers',
    'passlib.handlers.bcrypt',
    'bcrypt',

    # Image processing
    'cv2',
    'PIL',
    'PIL._imaging',
    'PIL.Image',
    'PIL.ImageFilter',
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    'skimage',
    'skimage.metrics',
    'skimage.metrics._structural_similarity',
    'skimage.color',
    'skimage.color.colorconv',
    'scipy',
    'scipy.ndimage',
    'scipy.special',
    'scipy.optimize',

    # PDF
    'pdf2image',
    'pypdfium2',

    # OCR
    'easyocr',
    'easyocr.detection',
    'easyocr.recognition',
    'easyocr.utils',
    'pytesseract',

    # Barcode
    'pyzbar',
    'pyzbar.pyzbar',
    'pyzbar.wrapper',
    'pyzbar.locations',

    # Reports
    'reportlab',
    'reportlab.platypus',
    'reportlab.lib',
    'reportlab.pdfgen',
    'openpyxl',
    'openpyxl.styles',
    'openpyxl.utils',

    # HTTP
    'httpx',
    'httpx._transports',
    'multipart',
    'python_multipart',

    # Windows-specific
    'win32api',
    'win32con',
    'win32print',
    'win32timezone',
    'win10toast',
    'pywintypes',

    # Scanner
    'dynamsoftservice',

    # Utilities
    'aiofiles',
    'dotenv',
    'h11',
    'certifi',
    'charset_normalizer',

    # App modules
    'app',
    'app.main',
    'app.config',
    'app.database',
    'app.models',
    'app.models.base',
    'app.routers',
    'app.routers.auth',
    'app.routers.jobs',
    'app.routers.users',
    'app.routers.templates',
    'app.routers.scanners',
    'app.routers.batch',
    'app.routers.reports',
    'app.routers.settings_router',
    'app.services',
    'app.services.inspection_engine',
    'app.services.alignment',
    'app.services.ocr_service',
    'app.services.color_service',
    'app.services.ssim_service',
    'app.services.barcode_service',
    'app.services.preprocess',
    'app.services.report_service',
    'app.services.annotator',
    'app.services.scanner_service',
    'app.services.backup_service',
    'app.services.auth_service',
    'app.services.webhook_service',
]

# ── Data files to bundle ───────────────────────────────────────────────────────
datas = [
    # App configuration
    ('.env', '.'),
    ('alembic.ini', '.'),
    ('alembic', 'alembic'),

    # ReportLab fonts and data
    *collect_data_files('reportlab', includes=['**/*']),

    # EasyOCR data files
    *collect_data_files('easyocr', includes=['**/*']),

    # Certifi CA bundle
    *collect_data_files('certifi'),

    # EasyOCR pre-downloaded models (must exist before building)
    ('models', 'models'),
]

# ── Binary files ───────────────────────────────────────────────────────────────
binaries = []

a = Analysis(
    ['app/main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'jupyter',
        'IPython',
        'notebook',
        'sphinx',
        'pytest',
        'black',
        'mypy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='greenpack_engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # Disabled: UPX can trigger AV false positives
    console=False,      # No console window — runs as silent Windows service
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/greenpack.ico',
    version='version_info.txt',
    uac_admin=True,     # Request admin rights (needed for service registration)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='greenpack_engine',
)
