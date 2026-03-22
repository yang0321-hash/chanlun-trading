@echo off
chcp 65001 >nul
echo ========================================
echo ChanLun Trading System - Install
echo ========================================
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo.
echo Installing packages...
python -m pip install akshare pandas numpy plotly loguru python-dotenv scipy matplotlib

if errorlevel 1 (
    echo ERROR: Installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
pause
