@echo off
chcp 65001 >nul
cd /d "%~dp0"
python scan_tdx.py
if errorlevel 1 (
    echo 发生错误，请查看上方信息
)
pause
