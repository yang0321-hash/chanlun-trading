"""
缠论选股器 - 快速测试版
只扫描上海和深圳交易所的个股（排除指数）
"""

import os
import sys

# 确保输出编码正确
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from scanner.chanlun_screener import ChanLunScreener, print_scan_result, ScanResult
import pandas as pd
import json
import glob

def is_stock(symbol):
    """判断是否为个股（非指数）"""
    # 排除指数
    if symbol.startswith('sh000') or symbol.startswith('sz399'):
        return False
    # 上海个股：600xxx, 601xxx, 603xxx, 605xxx, 688xxx(科创板)
    if symbol.startswith('sh6'):
        return True
    # 深圳个股：000xxx, 001xxx, 002xxx, 300xxx
    if symbol.startswith('sz') and symbol[2:5] in ('000', '001', '002', '300'):
        return True
    return False

def quick_scan(limit=None):
    data_dir = "test_output"

    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在 - {data_dir}")
        return

    # 获取所有个股文件
    all_files = glob.glob(os.path.join(data_dir, "*.json"))
    files = [f for f in all_files if is_stock(os.path.basename(f).replace('.day.json', '').replace('.json', ''))]

    if limit:
        files = files[:limit]

    print("=" * 50)
    print(f"缠论选股器 - 沪深个股扫描")
    print(f"扫描 {len(files)} 只个股 (已排除指数和北交所)")
    print("=" * 50)

    screener = ChanLunScreener(use_macd=True, min_klines=60, exclude_bj=True)

    result = ScanResult()
    result.total_scanned = len(files)

    for i, filepath in enumerate(files):
        try:
            # 从文件名提取股票代码
            symbol = os.path.basename(filepath).replace('.day.json', '').replace('.json', '')

            # 读取JSON数据
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])

            signal = screener.scan_from_dataframe(df, symbol)

            if signal:
                if signal.signal_type == '1buy':
                    result.signals_1buy.append(signal)
                elif signal.signal_type == '2buy':
                    result.signals_2buy.append(signal)
                elif signal.signal_type == '3buy':
                    result.signals_3buy.append(signal)

        except Exception as e:
            pass

    print_scan_result(result)

if __name__ == '__main__':
    # 扫描前200只个股
    quick_scan(limit=200)
