@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo Starting SAMcloud...
echo Open http://127.0.0.1:8080 in your browser if it does not open automatically.
python main.py
