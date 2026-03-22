"""
回测优化版1买信号
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
from indicator.macd import MACD


def is_stock(symbol):
    """判断是否为个股（非指数）"""
    if symbol.startswith('sh000') or symbol.startswith('sz399'):
        return False
    if symbol.startswith('sh6'):
        return True
    if symbol.startswith('sz') and symbol[2:5] in ('000', '001', '002', '300'):
        return True
    return False


def check_1buy_score(df, i, pivots, strokes, use_macd=True):
    """
    计算位置i的1买信号评分

    Returns:
        int or None: 评分(0-100)，None表示无信号
    """
    if i < 50 or i >= len(df) - 20:
        return None

    if not pivots or len(strokes) < 3:
        return None

    current_price = df['close'].iloc[i]
    last_pivot = pivots[-1]
    last_stroke = strokes[-1]

    # 基础条件：价格在中枢下方 + 反弹
    if current_price >= last_pivot.low:
        return None
    if not last_stroke.is_up:
        return None

    score = 0

    # 1. 成交量确认 (0-20)
    if i >= 25:
        recent_vol = df['volume'].iloc[i-5:i+1].mean()
        base_vol = df['volume'].iloc[i-25:i-5].mean()
        if base_vol > 0:
            vol_ratio = recent_vol / base_vol
            if vol_ratio >= 1.5:
                score += 20
            elif vol_ratio >= 1.2:
                score += 15
            elif vol_ratio >= 0.8:
                score += 10
            else:
                score += 5

    # 2. 趋势过滤 (0-20)
    if i >= 60:
        ma60 = df['close'].iloc[i-60:i+1].mean()
        ma60_ratio = current_price / ma60
        if ma60_ratio >= 0.95:
            score += 20
        elif ma60_ratio >= 0.85:
            score += 15
        elif ma60_ratio >= 0.75:
            score += 8
        else:
            score += 3

        # 创新低惩罚
        recent_low = df['low'].iloc[max(0, i-20):i+1].min()
        if current_price < recent_low * 1.02:
            score -= 5
    else:
        score += 10  # 默认中性

    # 3. 中枢质量 (0-15)
    pivot_range = last_pivot.range_value / last_pivot.low
    stroke_count = last_pivot.strokes_count

    if 0.05 <= pivot_range <= 0.20:
        range_score = 10
    elif pivot_range < 0.05:
        range_score = 5
    else:
        range_score = 8

    if stroke_count >= 5:
        stroke_score = 5
    elif stroke_count >= 3:
        stroke_score = 3
    else:
        stroke_score = 1

    score += range_score + stroke_score

    # 4. 回调幅度 (0-15)
    drop_ratio = (last_pivot.low - current_price) / last_pivot.low
    if drop_ratio <= 0.05:
        score += 15
    elif drop_ratio <= 0.15:
        score += 12
    elif drop_ratio <= 0.30:
        score += 6
    else:
        score += 2

    # 5. MACD背驰 (0-20)
    if use_macd and i >= 20:
        try:
            macd = MACD(df['close'].iloc[:i+1])
            if len(macd) >= 20:
                has_div, _ = macd.check_divergence(len(macd) - 20, len(macd) - 1, 'down')
                if has_div:
                    score += 20
                else:
                    score += 5
        except:
            pass

    # 6. K线形态 (0-10) - 简化版，默认给分
    score += 5

    return score


def backtest_optimized_1buy(limit=50, score_threshold=60):
    """
    回测优化版1买信号

    Args:
        limit: 扫描股票数量
        score_threshold: 最低评分要求
    """
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
    print(f"优化版1买回测 (评分阈值: {score_threshold}分)")
    print(f"回测 {len(files)} 只个股")
    print("=" * 60)

    # 收集结果
    all_returns = {'5d': [], '10d': [], '20d': []}
    all_scores = []
    signal_count = 0

    for i, filepath in enumerate(files):
        if i % 10 == 0:
            print(f"进度: {i}/{len(files)}")

        try:
            symbol = os.path.basename(filepath).replace('.day.json', '').replace('.json', '')

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])

            if len(df) < 100:
                continue

            kline = KLine.from_dataframe(df, strict_mode=True)

            # 预计算缠论要素
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            strokes = StrokeGenerator(kline, fractals).get_strokes()
            pivots = PivotDetector(kline, strokes).get_pivots()

            # 每隔5天检查一次
            for idx in range(50, len(df) - 20, 5):
                # 获取历史数据（截至当前位置）
                hist_pivots = [p for p in pivots if p.end_index <= idx]
                hist_strokes = [s for s in strokes if s.end_index <= idx]

                if not hist_pivots or len(hist_strokes) < 3:
                    continue

                # 计算评分
                score = check_1buy_score(df, idx, hist_pivots, hist_strokes, use_macd=True)

                if score is not None and score >= score_threshold:
                    signal_count += 1
                    all_scores.append(score)

                    entry_price = df['close'].iloc[idx]

                    # 计算未来收益
                    for days, key in [(5, '5d'), (10, '10d'), (20, '20d')]:
                        exit_idx = min(idx + days, len(df) - 1)
                        exit_price = df['close'].iloc[exit_idx]
                        return_pct = (exit_price - entry_price) / entry_price * 100
                        all_returns[key].append(return_pct)

        except Exception as e:
            pass

    # 统计结果
    print("\n" + "=" * 60)
    print(f"回测结果 (评分阈值 >= {score_threshold}分)")
    print(f"总信号数: {signal_count}")
    print("=" * 60)

    for days in ['5d', '10d', '20d']:
        returns = all_returns[days]
        if not returns:
            continue

        returns_array = np.array(returns)

        win_count = np.sum(returns_array > 0)
        total_count = len(returns_array)
        win_rate = win_count / total_count * 100 if total_count > 0 else 0

        avg_return = np.mean(returns_array)
        median_return = np.median(returns_array)
        max_profit = np.max(returns_array)
        max_loss = np.min(returns_array)

        profit_returns = returns_array[returns_array > 0]
        loss_returns = returns_array[returns_array < 0]
        avg_profit = np.mean(profit_returns) if len(profit_returns) > 0 else 0
        avg_loss = np.mean(loss_returns) if len(loss_returns) > 0 else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

        print(f"\n持有{days.replace('d', '天')}: (样本: {total_count})")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  平均收益: {avg_return:+.2f}%")
        print(f"  中位数: {median_return:+.2f}%")
        print(f"  盈亏比: {profit_loss_ratio:.2f}")
        print(f"  最大盈利: {max_profit:+.2f}%")
        print(f"  最大亏损: {max_loss:+.2f}%")

    # 对比原始1买
    print("\n" + "=" * 60)
    print("对比：原始1买 vs 优化版1买")
    print("=" * 60)
    print("\n原始1买 (之前回测):")
    print("  5天: 胜率53.7%, 平均+0.85%")
    print("  10天: 胜率57.7%, 平均+1.66%")
    print("  20天: 胜率61.9%, 平均+2.98%")


if __name__ == '__main__':
    # 测试不同阈值
    for threshold in [50, 60, 70]:
        print(f"\n\n{'#'*60}")
        print(f"测试阈值: {threshold}分")
        print(f"{'#'*60}")
        backtest_optimized_1buy(limit=50, score_threshold=threshold)
