@echo off
REM 缠论每日自动化工作流
REM 由 Windows Task Scheduler 在每交易日 14:30 调用

cd /d "D:\新建文件夹\claude"

REM 使用当前 Python 环境
python -u daily_workflow.py --auto --notify >> signals\daily_workflow.log 2>&1

echo [%date% %time%] daily_workflow exit code=%errorlevel% >> signals\daily_workflow.log
