@echo off
chcp 65001 >nul
echo ========================================
echo ChanLun Trading System - Run
echo ========================================
echo.

cd /d "%~dp0"

echo Running example...
echo.

python examples\basic_usage.py

if errorlevel 1 (
    echo.
    echo ERROR: Script failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Done! Check the output above.
echo ========================================
pause
