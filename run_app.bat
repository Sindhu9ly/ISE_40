@echo off
REM ==============================
REM Skin Disease Detection Launcher
REM ==============================

REM Change directory to this script’s folder
cd /d "%~dp0"

echo ========================================
echo   Starting Skin Disease Detection App...
echo ========================================

REM Check for virtual environment folder
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo [INFO] Installing required packages (may take a few minutes)...
pip install --upgrade pip >nul
pip install -r requirements.txt

REM Launch Streamlit app
echo [INFO] Launching Streamlit...
python -m streamlit run app.py

REM Pause window when Streamlit closes
pause
