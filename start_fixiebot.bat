@echo off
echo.
echo ========================================
echo    FixieBot - AI Ticket Fix Predictor
echo ========================================
echo.
echo Starting FixieBot...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip install -r requirements.txt >nul 2>&1

REM Start the application
echo Starting FixieBot server...
echo.
echo The application will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python run.py

pause 