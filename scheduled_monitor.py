#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论30分钟2买监控 - 定时启动版

从指定时间开始持续监控
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入监控模块
from chanlun_30min_monitor import ChanLun30MinMonitor

def wait_until_start_time(start_time: datetime):
    """等待直到指定时间"""
    now = datetime.now()

    if now >= start_time:
        print(f"[INFO] 当前时间 {now.strftime('%Y-%m-%d %H:%M:%S')} 已超过启动时间 {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[INFO] 立即启动监控...")
        return

    wait_seconds = (start_time - now).total_seconds()
    wait_minutes = int(wait_seconds / 60)
    wait_hours = wait_minutes // 60
    wait_minutes = wait_minutes % 60

    print(f"\n{'='*70}")
    print(f"定时监控 - 等待启动")
    print(f"{'='*70}")
    print(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"等待时间: {wait_hours}小时{wait_minutes}分钟")
    print(f"{'='*70}\n")

    # 每小时提示一次
    while wait_seconds > 0:
        if wait_seconds > 3600:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待中... 距离启动还有 {int(wait_seconds/3600)} 小时")
            time.sleep(min(3600, wait_seconds))
            wait_seconds -= 3600
        elif wait_seconds > 60:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待中... 距离启动还有 {int(wait_seconds/60)} 分钟")
            time.sleep(min(60, wait_seconds))
            wait_seconds -= 60
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 即将启动监控...")
            time.sleep(wait_seconds)
            break

def main():
    # 启动时间: 2026年3月30日 9:00
    start_time = datetime(2026, 3, 30, 9, 0, 0)

    print(f"\n{'='*70}")
    print(f"缠论30分钟2买监控 - 定时任务")
    print(f"{'='*70}")
    print(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"监控频率: 每30分钟扫描一次")
    print(f"监控股票: 47只")
    print(f"通知方式: 飞书")
    print(f"{'='*70}")

    # 等待到启动时间
    wait_until_start_time(start_time)

    # 启动监控
    print(f"\n{'='*70}")
    print(f"开始监控！")
    print(f"{'='*70}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    monitor = ChanLun30MinMonitor(enable_notification=True)

    try:
        while True:
            signals = monitor.scan_watchlist()

            if signals:
                result = monitor.format_signals(signals)
                print(result)
                monitor.send_notification(signals)

            print(f"\n下次扫描时间: {(datetime.now() + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'-'*70}\n")

            time.sleep(30 * 60)  # 30分钟

    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"监控已停止")
        print(f"停止时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
