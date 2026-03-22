@echo off
chcp 65001 >nul
echo ========================================
echo Python Auto Installer
echo ========================================
echo.
echo Attempting to install Python using winget...
echo.

winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements

if errorlevel 1 (
    echo.
    echo winget failed or not available.
    echo.
    echo Please install manually:
    echo 1. Visit: https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe
    echo 2. Download and run the installer
    echo 3. CHECK "Add Python to PATH" during installation
    echo.
    start https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe
    pause
    exit /b 1
)

echo.
echo ========================================
echo Python installed successfully!
echo ========================================
echo.
echo Please CLOSE and reopen your PowerShell/Command Prompt
echo then run: install.bat
echo.
pause
