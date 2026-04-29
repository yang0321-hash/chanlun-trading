#!/usr/bin/env python3
"""四层Agent交易系统 — 启动入口

使用方法:
  python run_trading_system.py --agent pre_market    # 盘前Agent
  python run_trading_system.py --agent intraday       # 盘中Agent
  python run_trading_system.py --agent daily_scan     # 每日扫描
  python run_trading_system.py --agent post_market    # 复盘Agent
  python run_trading_system.py --agent all            # 全部执行
  python run_trading_system.py --status               # 查看状态
"""
import sys
import os

sys.path.insert(0, '.')

from trading_agents.orchestrator import main

if __name__ == '__main__':
    main()
