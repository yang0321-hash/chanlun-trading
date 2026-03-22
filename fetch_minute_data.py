#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分钟K线数据获取和展示

支持 1min, 5min, 15min, 30min, 60min
"""

import pandas as pd
from datetime import datetime, timedelta
from data.akshare_source import AKShareSource


def fetch_minute_data(symbol: str, period: str = '5min', days: int = 5):
    """
    获取分钟K线数据

    Args:
        symbol: 股票代码 (如 '600519')
        period: 周期 ('1min', '5min', '15min', '30min', '60min')
        days: 获取最近几天数据

    Returns:
        DataFrame
    """
    source = AKShareSource()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"获取 {symbol} {period} 数据...")
    print(f"时间范围: {start_date.date()} ~ {end_date.date()}")
    print("-" * 50)

    df = source.get_kline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period=period,
        adjust=''
    )

    if df.empty:
        print("未获取到数据")
        return None

    print(f"✓ 获取 {len(df)} 条数据")
    print(f"时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print()

    # 统计信息
    print("数据统计:")
    print(f"  平均成交量: {df['volume'].mean():.0f}")
    print(f"  平均成交额: {df['amount'].mean():.0f}")
    print(f"  价格区间: {df['low'].min():.2f} ~ {df['high'].max():.2f}")
    print()

    # 按日期统计
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    print("按交易日统计:")
    for date, group in df.groupby('date'):
        print(f"  {date}: {len(group)} 根K线")

    return df


def display_sample(df: pd.DataFrame, n: int = 20):
    """显示样本数据"""
    print("\n最近数据:")
    print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail(n).to_string())


def apply_chanlun(df: pd.DataFrame):
    """应用缠论处理"""
    from core.kline import KLine
    from core.fractal import FractalDetector

    print("\n" + "=" * 50)
    print("缠论分析")
    print("=" * 50)

    # 创建K线对象
    kline = KLine.from_dataframe(df, strict_mode=True)
    print(f"原始K线: {len(kline.raw_data)} 根")
    print(f"处理后: {len(kline.processed_data)} 根")
    print(f"合并: {len(kline.raw_data) - len(kline.processed_data)} 根")

    # 识别分型
    detector = FractalDetector(kline, confirm_required=False)
    print(f"\n识别分型: {len(detector)} 个")
    print(f"  顶分型: {len(detector.get_top_fractals())} 个")
    print(f"  底分型: {len(detector.get_bottom_fractals())} 个")

    # 显示最近分型
    if detector.fractals:
        print("\n最近分型:")
        for f in detector.fractals[-5:]:
            type_str = "顶" if f.is_top else "底"
            print(f"  {f.datetime} {type_str}分型: high={f.high:.2f}, low={f.low:.2f}")

    return kline, detector


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='获取分钟K线数据')
    parser.add_argument('--symbol', default='600519', help='股票代码')
    parser.add_argument('--period', default='5min', choices=['1min', '5min', '15min', '30min', '60min'],
                       help='K线周期')
    parser.add_argument('--days', type=int, default=5, help='获取天数')
    parser.add_argument('--chanlun', action='store_true', help='应用缠论分析')

    args = parser.parse_args()

    df = fetch_minute_data(args.symbol, args.period, args.days)

    if df is not None:
        display_sample(df)

        if args.chanlun:
            apply_chanlun(df)
