#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通达信数据快速更新脚本 v2
支持从 AKShare 获取最新数据并更新本地 .day 文件
"""

import os
import sys
import struct
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("Installing dependencies...")
    os.system("pip install akshare pandas -q")
    import akshare as ak
    import pandas as pd


# 配置
TDX_PATH = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"
VIPDOC_PATH = os.path.join(TDX_PATH, "vipdoc")


def read_day_file(filepath):
    """读取.day文件"""
    data = []
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(32)
                if len(chunk) < 32:
                    break
                values = struct.unpack('IIIIIfII', chunk)
                date_int = values[0]
                date_str = str(date_int)
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                data.append({
                    'date': date_str,
                    'open': values[1] / 100,
                    'high': values[2] / 100,
                    'low': values[3] / 100,
                    'close': values[4] / 100,
                    'amount': values[5],
                    'volume': values[6]
                })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()


def write_day_file(df, filepath):
    """写入.day文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = df.copy()
    df['date_int'] = df['date'].astype(str).str.replace('-', '').astype(int)
    df['open_int'] = (df['open'] * 100).astype(int)
    df['high_int'] = (df['high'] * 100).astype(int)
    df['low_int'] = (df['low'] * 100).astype(int)
    df['close_int'] = (df['close'] * 100).astype(int)

    with open(filepath, 'wb') as f:
        for _, row in df.iterrows():
            # 处理成交额溢出
            amount = float(row['amount'])
            if amount > 4294967295:
                amount = amount / 10000

            chunk = struct.pack(
                'IIIIIfII',
                int(row['date_int']) & 0xFFFFFFFF,
                int(row['open_int']) & 0xFFFFFFFF,
                int(row['high_int']) & 0xFFFFFFFF,
                int(row['low_int']) & 0xFFFFFFFF,
                int(row['close_int']) & 0xFFFFFFFF,
                amount,
                int(row['volume']) & 0xFFFFFFFF,
                0
            )
            f.write(chunk)


def get_ak_data(code, start_date=None, retry=2):
    """从AKShare获取数据，带重试"""
    for attempt in range(retry):
        try:
            symbol = code[2:]

            # 处理指数
            if code.endswith('000001') or code.endswith('000300') or code.endswith('399001') or code.endswith('399006'):
                if code.startswith('sh'):
                    df = ak.stock_zh_index_daily(symbol=code)
                else:
                    df = ak.stock_zh_index_daily(symbol=code)
            else:
                # 处理股票 - 使用更稳定的API
                try:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
                except:
                    # 尝试另一种方式
                    df = ak.stock_zh_a_daily(symbol=f"sh{symbol}" if code.startswith('sh') else f"sz{symbol}")

            if df is None or len(df) == 0:
                return None

            # 统一列名
            column_map = {
                'date': 'date',
                '日期': 'date',
                'open': 'open',
                '开盘': 'open',
                'high': 'high',
                '最高': 'high',
                'low': 'low',
                '最低': 'low',
                'close': 'close',
                '收盘': 'close',
                'volume': 'volume',
                '成交量': 'volume',
                'amount': 'amount',
                '成交额': 'amount'
            }
            df = df.rename(columns=column_map)

            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0

            # 统一日期格式
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            return df[required_cols]

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)
                continue
            return None


def update_stock(code, vipdoc_path, verbose=False):
    """更新单只股票"""
    market = code[:2]
    filepath = os.path.join(vipdoc_path, market, 'lday', f"{code}.day")

    if not os.path.exists(filepath):
        return False, "NO_FILE"

    # 读取本地数据
    local_df = read_day_file(filepath)
    if len(local_df) == 0:
        return False, "READ_ERROR"

    # 检查是否需要更新
    local_last = local_df['date'].max()
    today = datetime.now().strftime('%Y-%m-%d')

    # 计算工作日差异
    last_date = datetime.strptime(local_last, '%Y-%m-%d')
    days_diff = (datetime.now() - last_date).days

    # 如果是最新数据（今天或周五），跳过
    if local_last >= today:
        return True, "CURRENT"

    # 如果超过7天没更新，尝试更新
    if days_diff > 7:
        pass  # 需要更新
    # 周末检查：如果是周五数据，算是最新的
    elif last_date.weekday() == 4 and days_diff <= 3:
        return True, "CURRENT_FRIDAY"

    # 获取新数据
    update_start = (last_date + timedelta(days=1)).strftime('%Y%m%d')
    new_df = get_ak_data(code, update_start)

    if new_df is None or len(new_df) == 0:
        return False, "NO_NEW_DATA"

    # 过滤已有的数据
    new_df = new_df[new_df['date'] > local_last]

    if len(new_df) == 0:
        return True, "NO_NEW"

    # 合并
    combined_df = pd.concat([local_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
    combined_df = combined_df.sort_values('date').reset_index(drop=True)

    # 备份并写入
    try:
        shutil.copy2(filepath, filepath + '.bak')
        write_day_file(combined_df, filepath)
        if verbose:
            print(f"  {code}: +{len(new_df)} days, {local_last} -> {combined_df['date'].max()}")
        return True, f"UPDATED_{len(new_df)}"
    except Exception as e:
        return False, f"WRITE_ERROR"


def quick_update(codes=None, limit=100, verbose=False):
    """快速更新"""
    vipdoc_path = VIPDOC_PATH

    if codes is None:
        # 获取本地股票列表
        codes = []
        for market in ['sh', 'sz', 'bj']:
            lday_path = os.path.join(vipdoc_path, market, 'lday')
            if os.path.exists(lday_path):
                files = os.listdir(lday_path)
                codes.extend([f.replace('.day', '') for f in files if f.endswith('.day')])

        # 按优先级排序
        priority = ['sh000001', 'sh000300', 'sz399001', 'sz399006']
        codes = sorted(codes, key=lambda x: (priority.index(x) if x in priority else 999, x))

        if limit:
            codes = codes[:limit]

    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Checking {len(codes)} stocks for updates...\n")

    success = 0
    need_update = 0
    failed = 0
    current = 0

    for i, code in enumerate(codes):
        if i % 50 == 0:
            print(f"[{i}/{len(codes)}] Updated: {success}, Current: {current}, Failed: {failed}")

        ok, msg = update_stock(code, vipdoc_path, verbose)

        if msg == "CURRENT" or msg == "CURRENT_FRIDAY" or msg == "NO_NEW":
            current += 1
        elif msg.startswith("UPDATED"):
            success += 1
            need_update += 1
        else:
            failed += 1

        # 避免请求过快
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)

    print(f"\nComplete!")
    print(f"  Updated: {success} stocks")
    print(f"  Current: {current} stocks")
    print(f"  Failed: {failed} stocks")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TDX Data Quick Update')
    parser.add_argument('--limit', type=int, default=100, help='Update limit')
    parser.add_argument('--code', help='Specific stock code')
    parser.add_argument('--index', action='store_true', help='Update major indices only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.code:
        print(f"Updating {args.code}...")
        ok, msg = update_stock(args.code, VIPDOC_PATH, verbose=True)
        print(f"Result: {msg}")

    elif args.index:
        codes = ['sh000001', 'sh000300', 'sz399001', 'sz399006']
        print("Updating major indices...")
        quick_update(codes=codes, verbose=args.verbose)

    else:
        quick_update(limit=args.limit, verbose=args.verbose)
