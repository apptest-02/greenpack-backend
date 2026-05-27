# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['full_app.py'],
    pathex=[],
    binaries=[],
    datas=[('app', 'app'), ('..\\frontend\\dist', 'frontend'), ('venv\\Lib\\site-packages\\pypdfium2', 'pypdfium2')],
    hiddenimports=['aiosqlite', 'bcrypt', 'passlib.handlers.bcrypt', 'easyocr', 'openpyxl', 'reportlab', 'pypdfium2', 'cv2', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GreenpackPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
