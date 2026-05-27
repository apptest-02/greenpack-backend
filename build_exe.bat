@echo off
echo Building Greenpack Pro Desktop App...

cd /d "C:\Users\HARDEV-PC-7\Downloads\greenpack-pro\backend"

REM Activate virtual env
call venv\Scripts\activate.bat

REM Build with PyInstaller - NO console, ONE file
pyinstaller --onefile ^
            --windowed ^
            --name "GreenpackPro" ^
            --add-data "app;app" ^
            --add-data "templates;templates" ^
            --hidden-import=uvicorn ^
            --hidden-import=uvicorn.loops ^
            --hidden-import=uvicorn.loops.auto ^
            --hidden-import=uvicorn.protocols ^
            --hidden-import=uvicorn.protocols.http ^
            --hidden-import=uvicorn.protocols.http.auto ^
            --hidden-import=uvicorn.protocols.websockets ^
            --hidden-import=uvicorn.protocols.websockets.auto ^
            --hidden-import=anyio ^
            --hidden-import=jinja2 ^
            --collect-all=webview ^
            desktop_app.py

echo Build complete! Check the 'dist' folder.
pause