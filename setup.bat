@echo off
setlocal
cd /d "%~dp0"

echo === SAMcloud setup ===
echo.

where colmap >nul 2>nul
if %errorlevel% neq 0 (
    if not exist "colmap\COLMAP.bat" (
        echo COLMAP was not found.
        echo.
        echo Download the current Windows CUDA release from:
        echo   https://github.com/colmap/colmap/releases
        echo Extract the ZIP so this file exists:
        echo   %~dp0colmap\COLMAP.bat
        echo Or install COLMAP elsewhere and add it to PATH.
        echo.
        pause
        exit /b 1
    )
)
echo COLMAP found.

where py >nul 2>nul
if %errorlevel% neq 0 (
    echo Python Launcher was not found. Install Python 3.13 and try again.
    pause
    exit /b 1
)

set "PYTHON_CMD=py -3.13"
%PYTHON_CMD% -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,14)" >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.13 was not found. Install Python 3.13 and try again.
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
) else (
    venv\Scripts\python.exe -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,14)" >nul 2>nul
    if %errorlevel% neq 0 (
        echo The existing venv uses an incompatible Python version.
        echo Remove the venv folder and run setup.bat again.
        pause
        exit /b 1
    )
)

if not exist "models" mkdir models

echo Installing dependencies...
set "PIP_DEFAULT_TIMEOUT=300"
set "PIP_RETRIES=10"
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo SAM3 weights are access-gated and are not downloaded automatically.
echo Request access at https://huggingface.co/facebook/sam3, download sam3.pt,
echo and place it in models\sam3.pt before running classification.
echo.
echo Setup complete. Run start.bat to open SAMcloud.
pause
