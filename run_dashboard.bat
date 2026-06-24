@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ================================
echo KSL Sign Translator Dashboard
echo ================================
echo.

py -3.11 --version > nul 2>&1
if errorlevel 1 (
    echo Python 3.11 was not found.
    echo Please install Python 3.11 and try again.
    pause
    exit /b 1
)

echo.
echo Installing required packages with Python 3.11...
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt

echo.
echo Starting dashboard...
echo Browser URL: http://localhost:8501
echo To stop, press Ctrl + C in this window.
echo.

py -3.11 -m streamlit run dashboard.py

pause
