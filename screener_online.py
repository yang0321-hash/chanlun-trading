"""
缠论1买实战选股器 - 在线数据版

使用AKShare获取最新数据
"""

import os
import sys
from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta

# 确保输出编码正确
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD


def get_online_data(symbols):
    """使用AKShare获取在线数据"""
    try:
        import akshare as ak
    except ImportError:
        print("错误: 需要安装akshare")
        print("请运行: pip install akshare")
        return {}

    data = {}
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')

    for symbol in symbols:
        try:
            # 标准化代码
            code = symbol.replace('sh', '').replace('sz', '')

            df = ak.stock_zh_a_hist(
                symbol=code,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'
            )

            if df is not None and not df.empty:
                # 列名映射
                df = df.rename(columns={
                    '日期': 'datetime',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })
                df['datetime'] = pd.to_datetime(df['datetime'])
                data[symbol] = df
                print(f"  {symbol}: {len(df)} 条K线, 最新价 ¥{df['close'].iloc[-1]:.2f}")

            # 请求间隔
            import time
            time.sleep(0.5)

        except Exception as e:
            print(f"  {symbol}: 获取失败 - {e}")

    return data


def calculate_1buy_score(df, kline):
    """计算1买信号评分"""
    if len(df) < 60:
        return None

    fractals = FractalDetector(kline, confirm_required=False).get_fractals()
    if len(fractals) < 3:
        return None

    strokes = StrokeGenerator(kline, fractals).get_strokes()
    if len(strokes) < 3:
        return None

    pivots = PivotDetector(kline, strokes).get_pivots()
    if not pivots:
        return None

    current_price = df['close'].iloc[-1]
    last_pivot = pivots[-1]
    last_stroke = strokes[-1]

    # 基础条件
    if current_price >= last_pivot.low:
        return None
    if not last_stroke.is_up:
        return None

    score = 0
    details = {}

    # 成交量
    recent_vol = df['volume'].iloc[-5:].mean()
    base_vol = df['volume'].iloc[-25:-5].mean()
    if base_vol > 0:
        vol_ratio = recent_vol / base_vol
        if vol_ratio >= 1.2:
            score += 15
            details['vol'] = f'放量{vol_ratio:.1f}x'
        else:
            score += 8
            details['vol'] = f'平量{vol_ratio:.1f}x'

    # 趋势
    ma60 = df['close'].iloc[-60:].mean()
    ma60_ratio = current_price / ma60
    if ma60_ratio >= 0.90:
        score += 20
        details['trend'] = '良好'
    elif ma60_ratio >= 0.80:
        score += 12
        details['trend'] = '偏弱'
    else:
        score += 5
        details['trend'] = '下跌'

    # 中枢
    pivot_range = last_pivot.range_value / last_pivot.low
    if 0.05 <= pivot_range <= 0.20:
        score += 12
        details['pivot'] = f'优质{pivot_range:.1%}'
    else:
        score += 6
        details['pivot'] = f'一般{pivot_range:.1%}'

    # 回调幅度
    drop_ratio = (last_pivot.low - current_price) / last_pivot.low
    if drop_ratio <= 0.10:
        score += 15
        details['drop'] = f'浅跌破{drop_ratio:.1%}'
    elif drop_ratio <= 0.20:
        score += 10
        details['drop'] = f'中度跌破{drop_ratio:.1%}'
    else:
        score += 5
        details['drop'] = f'深度跌破{drop_ratio:.1%}'

    # MACD
    try:
        macd = MACD(df['close'])
        if len(macd) > 20:
            has_div, _ = macd.check_divergence(len(macd) - 20, len(macd) - 1, 'down')
            if has_div:
                score += 20
                details['macd'] = '背驰'
            else:
                score += 5
                details['macd'] = '无背驰'
    except:
        score += 10

    # K线形态
    score += 8
    details['pattern'] = '底分型'

    stop_loss = last_pivot.low * 0.97
    target1 = last_pivot.high
    target2 = last_pivot.high + (last_pivot.high - last_pivot.low)

    risk = current_price - stop_loss
    reward1 = target1 - current_price
    reward_ratio = reward1 / risk if risk > 0 else 0

    return {
        'score': score,
        'confidence': min(score / 100, 0.95),
        'price': current_price,
        'stop_loss': stop_loss,
        'target1': target1,
        'target2': target2,
        'reward_ratio': reward_ratio,
        'pivot_high': last_pivot.high,
        'pivot_low': last_pivot.low,
        'details': details
    }


def scan_online(symbols):
    """在线扫描指定股票"""
    print("=" * 70)
    print("缠论1买实战选股器 - 在线版")
    print(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"扫描股票: {', '.join(symbols)}")
    print("=" * 70)
    print("\n获取在线数据...")

    data = get_online_data(symbols)

    if not data:
        print("未获取到任何数据")
        return []

    print(f"\n成功获取 {len(data)} 只股票数据\n")
    print("=" * 70)

    signals = []
    SCORE_THRESHOLD = 55

    for symbol, df in data.items():
        if len(df) < 60:
            continue

        try:
            kline = KLine.from_dataframe(df, strict_mode=True)
            result = calculate_1buy_score(df, kline)

            if result and result['score'] >= SCORE_THRESHOLD:
                signals.append({
                    'symbol': symbol,
                    **result
                })
        except Exception as e:
            print(f"{symbol}: 分析失败 - {e}")

    # 按评分排序
    signals.sort(key=lambda x: x['score'], reverse=True)

    # 打印结果
    print(f"\n发现 {len(signals)} 个1买信号 (评分 >= {SCORE_THRESHOLD}分)\n")

    if signals:
        print(f"{'代码':<10} {'价格':>8} {'评分':>4} {'胜率':>6} {'盈亏比':>6} {'止损':>8} {'目标1':>8} {'目标2':>8}")
        print("-" * 70)

        for sig in signals:
            print(f"{sig['symbol']:<10} "
                  f"¥{sig['price']:>6.2f} "
                  f"{sig['score']:>3} "
                  f"{sig['confidence']:>5.0%} "
                  f"{sig['reward_ratio']:>5.1f} "
                  f"¥{sig['stop_loss']:>6.2f} "
                  f"¥{sig['target1']:>6.2f} "
                  f"¥{sig['target2']:>6.2f}")

            detail_str = " | ".join([f"{k}:{v}" for k, v in sig['details'].items()])
            print(f"  └─ {detail_str}")

    print("\n" + "=" * 70)

    return signals


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='缠论1买在线选股器')
    parser.add_argument('--symbols', type=str, default='sh600519,sz000001,sz000002,sz300015,sh600163',
                       help='股票代码列表，逗号分隔')

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]
    scan_online(symbols)
