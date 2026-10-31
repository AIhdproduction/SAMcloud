@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:8080"

rem Do not start a second NiceGUI instance when SAMcloud is already running.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$listener = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue; if ($listener) { exit 0 } else { exit 1 }"
if %errorlevel% equ 0 (
    echo SAMcloud is already running on port 8080.
    echo Opening %APP_URL% in your browser...
    start "" "%APP_URL%"
    pause
    exit /b 0
)

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo Starting SAMcloud...
echo Open %APP_URL% in your browser if it does not open automatically.
"%~dp0venv\Scripts\python.exe" "%~dp0main.py"
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%" == "0" (
    echo.
    echo SAMcloud stopped with exit code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
