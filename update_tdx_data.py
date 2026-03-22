#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通达信数据自动更新脚本
支持从 AKShare 获取最新数据并更新本地 .day 文件
"""

import os
import sys
import struct
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import argparse

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("请安装依赖: pip install akshare pandas")
    sys.exit(1)


# TDX 数据路径配置
TDX_PATH = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"
VIPDOC_PATH = os.path.join(TDX_PATH, "vipdoc")


class TDXDataUpdater:
    """通达信数据更新器"""

    def __init__(self, vipdoc_path: str = VIPDOC_PATH):
        self.vipdoc_path = vipdoc_path
        self.markets = {
            'sh': {'prefix': 'sh', 'name': '上海'},
            'sz': {'prefix': 'sz', 'name': '深圳'},
            'bj': {'prefix': 'bj', 'name': '北京'}
        }

    def read_day_file(self, filepath: str) -> pd.DataFrame:
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
        except Exception as e:
            return pd.DataFrame()

    def write_day_file(self, df: pd.DataFrame, filepath: str):
        """写入.day文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 转换日期格式 YYYYMMDD
        df = df.copy()
        df['date_int'] = df['date'].astype(str).str.replace('-', '').astype(int)

        # 价格转换为整数 (×100)
        df['open_int'] = (df['open'] * 100).astype(int)
        df['high_int'] = (df['high'] * 100).astype(int)
        df['low_int'] = (df['low'] * 100).astype(int)
        df['close_int'] = (df['close'] * 100).astype(int)

        with open(filepath, 'wb') as f:
            for _, row in df.iterrows():
                chunk = struct.pack(
                    'IIIIIfII',
                    int(row['date_int']),
                    int(row['open_int']),
                    int(row['high_int']),
                    int(row['low_int']),
                    int(row['close_int']),
                    float(row['amount']),
                    int(row['volume']),
                    0  # reserved
                )
                f.write(chunk)

    def get_local_codes(self, market: str) -> list:
        """获取本地已有的股票代码"""
        lday_path = os.path.join(self.vipdoc_path, market, 'lday')
        if not os.path.exists(lday_path):
            return []

        files = os.listdir(lday_path)
        codes = [f.replace('.day', '') for f in files if f.endswith('.day')]
        return codes

    def get_akshare_data(self, code: str, start_date: str = None) -> pd.DataFrame:
        """从AKShare获取股票数据"""
        try:
            # 转换代码格式
            if code.startswith('sh'):
                symbol = code[2:]
                if symbol.startswith('688'):
                    # 科创板
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
                elif symbol.startswith('6'):
                    # 上海主板
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
                else:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
            elif code.startswith('sz'):
                symbol = code[2:]
                if symbol.startswith('300'):
                    # 创业板
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
                elif symbol.startswith('002'):
                    # 中小板
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
                else:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
            elif code.startswith('bj'):
                # 北交所
                symbol = code[2:]
                df = ak.bj_stock_hist(symbol=symbol, period="daily", start_date=start_date, adjust="")
            else:
                return pd.DataFrame()

            if df is None or len(df) == 0:
                return pd.DataFrame()

            # 重命名列
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            })

            # 确保有必要的列
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0

            return df[required_cols]

        except Exception as e:
            return pd.DataFrame()

    def update_stock(self, code: str, force: bool = False) -> bool:
        """更新单个股票数据"""
        # 检查本地文件
        market = code[:2]
        filepath = os.path.join(self.vipdoc_path, market, 'lday', f"{code}.day")

        if not os.path.exists(filepath):
            return False

        # 读取本地数据
        local_df = self.read_day_file(filepath)
        if len(local_df) == 0:
            return False

        # 获取本地最新日期
        local_last_date = local_df['date'].max()
        today = datetime.now().strftime('%Y%m%d')

        if not force and local_last_date >= today:
            return True  # 已是最新

        # 计算需要更新的起始日期
        update_start = (datetime.strptime(local_last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y%m%d')

        # 从AKShare获取新数据
        new_df = self.get_akshare_data(code, update_start)
        if len(new_df) == 0:
            return False

        # 合并数据
        combined_df = pd.concat([local_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        combined_df = combined_df.sort_values('date').reset_index(drop=True)

        # 备份原文件
        backup_path = filepath + '.bak'
        shutil.copy2(filepath, backup_path)

        # 写入新数据
        try:
            self.write_day_file(combined_df, filepath)
            return True
        except:
            # 恢复备份
            shutil.copy2(backup_path, filepath)
            return False

    def update_index(self, code: str = 'sh000001') -> bool:
        """更新大盘指数"""
        filepath = os.path.join(self.vipdoc_path, 'sh', 'lday', f"{code}.day")

        if not os.path.exists(filepath):
            return False

        # 读取本地数据
        local_df = self.read_day_file(filepath)
        if len(local_df) == 0:
            return False

        local_last_date = local_df['date'].max()
        update_start = (datetime.strptime(local_last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y%m%d')

        try:
            # 获取指数数据
            if code == 'sh000001':
                new_df = ak.stock_zh_index_daily(symbol=f"sh000001")
            else:
                new_df = ak.stock_zh_index_daily(symbol=code)

            if new_df is None or len(new_df) == 0:
                return False

            new_df = new_df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'amount': 'amount'
            })

            # 过滤已有数据
            new_df = new_df[new_df['date'] > local_last_date]

            if len(new_df) == 0:
                return True

            # 合并并写入
            combined_df = pd.concat([local_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')

            self.write_day_file(combined_df, filepath)
            return True

        except Exception as e:
            return False

    def update_all(self, markets: list = None, limit: int = None):
        """更新所有股票数据"""
        if markets is None:
            markets = ['sh', 'sz', 'bj']

        total_updated = 0
        total_failed = 0

        for market in markets:
            codes = self.get_local_codes(market)
            if limit:
                codes = codes[:limit]

            print(f"\n{self.markets[market]['name']}市场: {len(codes)} 只股票")

            for i, code in enumerate(codes):
                if i % 50 == 0:
                    print(f"  进度: {i}/{len(codes)}, 已更新: {total_updated}")

                if self.update_stock(code):
                    total_updated += 1
                else:
                    total_failed += 1

        print(f"\n更新完成: 成功 {total_updated}, 失败 {total_failed}")

    def quick_update(self, days: int = 1):
        """快速更新最近几天的数据"""
        # 获取需要更新的股票（数据不完整的）
        outdated = []
        today = datetime.now()

        for market in ['sh', 'sz', 'bj']:
            codes = self.get_local_codes(market)
            for code in codes:
                filepath = os.path.join(self.vipdoc_path, market, 'lday', f"{code}.day")
                df = self.read_day_file(filepath)

                if len(df) > 0:
                    last_date = datetime.strptime(df['date'].max(), '%Y-%m-%d')
                    if (today - last_date).days >= days:
                        outdated.append(code)

        print(f"发现 {len(outdated)} 只股票需要更新")

        for i, code in enumerate(outdated):
            if i % 100 == 0:
                print(f"  进度: {i}/{len(outdated)}")
            self.update_stock(code)


def main():
    parser = argparse.ArgumentParser(description='通达信数据自动更新工具')
    parser.add_argument('--path', default=VIPDOC_PATH, help='通达信vipdoc路径')
    parser.add_argument('--market', choices=['sh', 'sz', 'bj', 'all'], default='all', help='更新市场')
    parser.add_argument('--code', help='更新指定股票代码')
    parser.add_argument('--index', action='store_true', help='更新大盘指数')
    parser.add_argument('--limit', type=int, help='限制更新数量')
    parser.add_argument('--quick', action='store_true', help='快速更新不完整数据')

    args = parser.parse_args()

    updater = TDXDataUpdater(args.path)

    if args.code:
        print(f"更新股票: {args.code}")
        success = updater.update_stock(args.code)
        print(f"结果: {'成功' if success else '失败'}")

    elif args.index:
        print("更新大盘指数...")
        updater.update_index()
        print("完成")

    elif args.quick:
        print("快速更新模式...")
        updater.quick_update()

    else:
        markets = ['sh', 'sz', 'bj'] if args.market == 'all' else [args.market]
        updater.update_all(markets=markets, limit=args.limit)


if __name__ == '__main__':
    main()
