@echo off
cd /d "%~dp0"
python trading_agents\orchestrator.py --agent post_market --force
