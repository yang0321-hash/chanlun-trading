@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo ========================================
echo TDX Data Auto Update
echo ========================================
echo Start: %date% %time%
python quick_update_tdx.py --limit 500
echo.
echo Update Complete!
echo End: %date% %time%
ping -n 4 127.0.0.1 >nul
