#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用通达信日线数据演示缠论分析

注：通达信本地数据只有日线，分钟数据需要在线获取
"""

import pandas as pd
from datetime import datetime, timedelta
from data.tdx_source import TDXDataSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator


def demo_chanlun_analysis(symbol: str = 'sh600519', days: int = 60):
    """
    使用通达信数据进行缠论分析

    Args:
        symbol: 股票代码
        days: 分析天数
    """
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc"
    source = TDXDataSource(tdx_path)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"获取 {symbol} 日线数据...")
    print(f"时间范围: {start_date.date()} ~ {end_date.date()}")
    print("-" * 60)

    df = source.get_kline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period='daily'
    )

    if df.empty:
        print("未获取到数据")
        return

    print(f"[OK] 获取 {len(df)} 条数据")

    # TDX数据使用datetime作为索引
    if 'datetime' in df.columns:
        print(f"时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    else:
        print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
        # 重置索引，将datetime变为列
        df = df.reset_index()

    print()

    # 创建K线对象
    kline = KLine.from_dataframe(df, strict_mode=True)
    print(f"原始K线: {len(kline.raw_data)} 根")
    print(f"合并处理后: {len(kline.processed_data)} 根")
    print(f"合并数量: {len(kline.raw_data) - len(kline.processed_data)} 根")
    print()

    # 识别分型
    detector = FractalDetector(kline, confirm_required=False)
    print(f"识别分型: {len(detector)} 个")
    print(f"  顶分型: {len(detector.get_top_fractals())} 个")
    print(f"  底分型: {len(detector.get_bottom_fractals())} 个")
    print()

    # 生成笔
    stroke_gen = StrokeGenerator(kline)
    strokes = stroke_gen.get_strokes()
    print(f"生成笔: {len(strokes)} 根")
    if strokes:
        print("  最近5笔:")
        for s in strokes[-5:]:
            direction = "↑" if s.is_up else "↓"
            print(f"    {s.start_datetime} ~ {s.end_datetime} {direction} "
                  f"{s.start_value:.2f} -> {s.end_value:.2f}")
    print()

    # 生成线段
    seg_gen = SegmentGenerator(kline, strokes)
    segments = seg_gen.get_segments()
    print(f"生成线段: {len(segments)} 根")
    if segments:
        print("  最近3段:")
        for s in segments[-3:]:
            direction = "↑" if s.direction == 'up' else "↓"
            print(f"    {s.start_time} ~ {s.end_time} {direction} "
                  f"{s.start_price:.2f} -> {s.end_price:.2f}")
    print()

    # 显示最近数据
    print("最近K线数据:")
    display_df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail(10)
    print(display_df.to_string(index=False))

    return kline, detector, strokes, segments


def intraday_time_analysis():
    """分析日内交易时间分布"""
    print("\n" + "=" * 60)
    print("日内交易时间分析")
    print("=" * 60)

    # A股交易时段
    sessions = [
        {"name": "早盘开盘", "time": "09:30", "desc": "波动剧烈，观望为主"},
        {"name": "早盘", "time": "09:30-10:00", "desc": "消化隔夜信息"},
        {"name": "早盘稳定", "time": "10:00-11:30", "desc": "趋势形成期"},
        {"name": "午间休市", "time": "11:30-13:00", "desc": "休息"},
        {"name": "午盘开盘", "time": "13:00-13:30", "desc": "早盘延续"},
        {"name": "午盘", "time": "13:30-14:30", "desc": "主力活跃期"},
        {"name": "尾盘", "time": "14:30-15:00", "desc": "尾盘拉升/砸盘"},
    ]

    for s in sessions:
        print(f"  {s['name']:12} {s['time']:20} {s['desc']}")

    print("\n缠论日内建议:")
    print("  - 5min K线: 每天约 48 根 (9:30-11:30, 13:00-15:00)")
    print("  - 分型确认: 3根K线确认 (15分钟)")
    print("  - 笔的确认: 需要相反分型确认 (约30-60分钟)")
    print("  - 最佳开仓: 10:00后 或 13:30后 (趋势稳定后)")
    print("  - 必须平仓: 14:50前 (不持仓过夜)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='通达信缠论分析')
    parser.add_argument('--symbol', default='sh600519', help='股票代码')
    parser.add_argument('--days', type=int, default=60, help='分析天数')

    args = parser.parse_args()

    demo_chanlun_analysis(args.symbol, args.days)
    intraday_time_analysis()
