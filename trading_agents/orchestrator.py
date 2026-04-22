#!/usr/bin/env python3
"""主Agent — 总调度

协调四层Agent执行:
  pre_market   07:00  盘前Agent
  intraday     09:30  盘中Agent
  daily_scan   14:30  每日扫描 (daily_workflow)
  post_market  21:30  复盘Agent
"""
import sys
import os
import json
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
          'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from dotenv import load_dotenv
load_dotenv()


def run_agent(agent_name: str, force: bool = False):
    """运行指定Agent"""
    if agent_name == 'pre_market':
        from trading_agents.pre_market import PreMarketAgent, check_today_done
        if not force and check_today_done('pre_market'):
            print('今日盘前分析已完成，跳过')
            return
        agent = PreMarketAgent()
        agent.run()

    elif agent_name == 'intraday':
        from trading_agents.intraday import IntradayAgent, check_today_done
        if not force and check_today_done('intraday'):
            print('今日盘中监控已完成，跳过')
            return
        agent = IntradayAgent()
        agent.run()

    elif agent_name == 'daily_scan':
        import subprocess
        cmd = [sys.executable, 'daily_workflow.py', '--notify']
        print(f'执行: {" ".join(cmd)}')
        subprocess.run(cmd, check=False)

    elif agent_name == 'post_market':
        from trading_agents.post_market import PostMarketAgent, check_today_done
        if not force and check_today_done('post_market'):
            print('今日复盘已完成，跳过')
            return
        agent = PostMarketAgent()
        agent.run()

    else:
        print(f'未知Agent: {agent_name}')
        print('可选: pre_market, intraday, daily_scan, post_market, all')


def run_all():
    """执行完整交易日流程"""
    print(f'=== 四层Agent完整交易日 {datetime.now().strftime("%Y-%m-%d")} ===')
    print()

    agents = ['pre_market', 'intraday', 'daily_scan', 'post_market']
    for name in agents:
        print(f'\n{"="*60}')
        print(f'  执行: {name}')
        print(f'{"="*60}')
        try:
            run_agent(name, force=True)
        except Exception as e:
            print(f'  [ERROR] {name} 执行失败: {e}')
            import traceback
            traceback.print_exc()
        print()

    print('全部完成')


def show_status():
    """显示当前状态"""
    log_path = 'signals/agent_log.json'
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
        today = datetime.now().strftime('%Y-%m-%d')
        entry = log.get(today, {})
        print(f'今日({today}) Agent状态:')
        for agent, info in entry.items():
            print(f'  {agent}: {info.get("status", "?")} ({info.get("time", "?")})')
    else:
        print('无agent_log.json')


# ==================== 入口 ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='四层Agent交易系统')
    parser.add_argument('--agent', '-a',
                       choices=['pre_market', 'intraday', 'daily_scan', 'post_market', 'all'],
                       default='all',
                       help='指定Agent (默认all)')
    parser.add_argument('--force', '-f', action='store_true', help='强制运行')
    parser.add_argument('--status', '-s', action='store_true', help='显示状态')
    parser.add_argument('--once', action='store_true', help='盘中Agent单次扫描')
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.agent == 'all':
        run_all()
    elif args.agent == 'intraday' and args.once:
        from trading_agents.intraday import IntradayAgent
        agent = IntradayAgent(once=True)
        agent.run()
    else:
        run_agent(args.agent, force=args.force)


if __name__ == '__main__':
    main()
