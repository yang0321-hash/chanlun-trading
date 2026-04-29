@echo off
cd /d "%~dp0"
python trading_agents\orchestrator.py --agent pre_market --force
