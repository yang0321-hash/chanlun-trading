"""
缠论买卖点回测脚本 - 优化版

回测1买、2买、3买信号的成功率
"""

import os
import sys
from typing import List, Dict
import pandas as pd
import json
import numpy as np

# 确保输出编码正确
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector


def is_stock(symbol):
    """判断是否为个股（非指数）"""
    if symbol.startswith('sh000') or symbol.startswith('sz399'):
        return False
    if symbol.startswith('sh6'):
        return True
    if symbol.startswith('sz') and symbol[2:5] in ('000', '001', '002', '300'):
        return True
    return False


def detect_buy_point(df, i, pivots, strokes):
    """
    在位置i检测买点（简化版，避免重复计算）

    Returns:
        str or None: '1buy', '3buy' or None
    """
    if i < 50 or i >= len(df) - 10:
        return None

    if not pivots or not strokes or len(strokes) < 3:
        return None

    current_price = df['close'].iloc[i]
    last_pivot = pivots[-1]

    # 1买: 价格跌破中枢下沿 + 正在反弹
    if current_price < last_pivot.low:
        if len(strokes) >= 2 and strokes[-1].is_up:
            return '1buy'

    # 3买: 突破中枢后确认
    if current_price >= last_pivot.high * 0.97:
        # 检查是否有向上笔突破
        recent_strokes = strokes[-5:] if len(strokes) >= 5 else strokes
        has_breakout = any(s.high > last_pivot.high for s in recent_strokes)
        if has_breakout and strokes[-1].is_up:
            return '3buy'

    return None


def backtest_symbol(filepath):
    """
    回测单个股票

    Returns:
        Dict: {'1buy': [returns...], '3buy': [returns...]}
    """
    results = {'1buy': {'5d': [], '10d': [], '20d': []},
                '3buy': {'5d': [], '10d': [], '20d': []}}

    try:
        symbol = os.path.basename(filepath).replace('.day.json', '').replace('.json', '')

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])

        if len(df) < 100:
            return results

        # 转换为KLine
        kline = KLine.from_dataframe(df, strict_mode=True)

        # 预先计算缠论要素
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        if len(fractals) < 3:
            return results

        strokes = StrokeGenerator(kline, fractals).get_strokes()
        if len(strokes) < 3:
            return results

        pivots = PivotDetector(kline, strokes).get_pivots()
        if not pivots:
            return results

        # 简化回测：从第50个K线开始，每隔5天检查一次
        for i in range(50, len(df) - 20, 5):
            # 找到当前时刻的中枢和笔
            current_pivots = [p for p in pivots if p.end_index <= i]
            current_strokes = [s for s in strokes if s.end_index <= i]

            if not current_pivots or len(current_strokes) < 3:
                continue

            # 检测买点
            buy_type = detect_buy_point(df, i, current_pivots, current_strokes)

            if buy_type:
                entry_price = df['close'].iloc[i]

                # 计算未来收益
                for days, key in [(5, '5d'), (10, '10d'), (20, '20d')]:
                    exit_idx = min(i + days, len(df) - 1)
                    exit_price = df['close'].iloc[exit_idx]
                    return_pct = (exit_price - entry_price) / entry_price * 100
                    results[buy_type][key].append(return_pct)

    except Exception as e:
        pass

    return results


def run_backtest(limit=50):
    """运行回测"""
    data_dir = "test_output"

    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在 - {data_dir}")
        return

    import glob
    all_files = glob.glob(os.path.join(data_dir, "*.json"))
    files = [f for f in all_files if is_stock(os.path.basename(f).replace('.day.json', '').replace('.json', ''))]

    if limit:
        files = files[:limit]

    print("=" * 60)
    print(f"缠论买卖点回测 - 简化版")
    print(f"回测 {len(files)} 只个股")
    print("=" * 60)

    # 收集所有结果
    all_returns = {
        '1buy': {'5d': [], '10d': [], '20d': []},
        '3buy': {'5d': [], '10d': [], '20d': []}
    }

    # 处理每只股票
    for i, filepath in enumerate(files):
        if i % 10 == 0:
            print(f"进度: {i}/{len(files)}")

        result = backtest_symbol(filepath)

        for signal_type in ['1buy', '3buy']:
            for days in ['5d', '10d', '20d']:
                all_returns[signal_type][days].extend(result[signal_type][days])

    # 统计结果
    print("\n" + "=" * 60)
    print("回测结果统计")
    print("=" * 60)

    for signal_type in ['1buy', '3buy']:
        print(f"\n【{signal_type.upper()}回测结果】")

        has_data = False
        for days in ['5d', '10d', '20d']:
            if all_returns[signal_type][days]:
                has_data = True
                break

        if not has_data:
            print(f"  无信号数据")
            continue

        print("-" * 60)

        for days in ['5d', '10d', '20d']:
            returns = all_returns[signal_type][days]
            if not returns:
                continue

            returns_array = np.array(returns)

            # 统计
            win_count = np.sum(returns_array > 0)
            total_count = len(returns_array)
            win_rate = win_count / total_count * 100 if total_count > 0 else 0

            avg_return = np.mean(returns_array)
            median_return = np.median(returns_array)
            max_profit = np.max(returns_array)
            max_loss = np.min(returns_array)

            # 盈利比
            profit_returns = returns_array[returns_array > 0]
            loss_returns = returns_array[returns_array < 0]
            avg_profit = np.mean(profit_returns) if len(profit_returns) > 0 else 0
            avg_loss = np.mean(loss_returns) if len(loss_returns) > 0 else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

            days_label = days.replace('d', '天')

            print(f"\n  持有{days_label}: (样本数: {total_count})")
            print(f"    胜率: {win_rate:.1f}%")
            print(f"    平均收益: {avg_return:+.2f}%")
            print(f"    中位数收益: {median_return:+.2f}%")
            print(f"    最大盈利: {max_profit:+.2f}%")
            print(f"    最大亏损: {max_loss:+.2f}%")
            print(f"    盈亏比: {profit_loss_ratio:.2f}")

            # 收益分布
            very_loss = np.sum(returns_array < -5) / total_count * 100
            small_loss = np.sum((returns_array >= -5) & (returns_array < 0)) / total_count * 100
            small_profit = np.sum((returns_array >= 0) & (returns_array < 5)) / total_count * 100
            very_profit = np.sum(returns_array >= 5) / total_count * 100

            print(f"    分布: <-5%:{very_loss:.0f}% -5~0%:{small_loss:.0f}% 0~5%:{small_profit:.0f}% >5%:{very_profit:.0f}%")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    run_backtest(limit=50)
