#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内交易演示 - 缠论多周期策略

功能：
1. 获取5分钟K线数据
2. 应用多周期缠论策略
3. 运行回测
4. 生成可视化图表
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta, time

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from data.tdx_source import TDXDataSource
from core.kline import KLine, KLineData
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from backtest.intraday_engine import IntradayEngine, IntradayConfig
from strategies.multitimeframe_chanlun import create_strategy_fn
from plot.chanlun_chart import create_intraday_chart, ChanLunChart


def get_tdx_data(symbol: str, days: int = 5):
    """
    从通达信获取数据

    Args:
        symbol: 股票代码 (如 'sh600519')
        days: 获取天数
    """
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc"
    source = TDXDataSource(tdx_path)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"获取 {symbol} 数据...")
    print(f"时间范围: {start_date.date()} ~ {end_date.date()}")

    try:
        # 获取日线数据
        daily_df = source.get_kline(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period='daily'
        )

        if not daily_df.empty:
            daily_df = daily_df.reset_index()
            print(f"日线数据: {len(daily_df)} 条")
    except Exception as e:
        print(f"日线数据获取失败: {e}")
        daily_df = None

    return daily_df


def simulate_5min_data(symbol: str, days: int = 3):
    """
    模拟生成5分钟K线数据（用于演示）

    注：通达信本地数据通常只有日线
    实际使用时需要在线获取分钟数据
    """
    import numpy as np

    print("\n模拟生成5分钟K线数据...")

    # 使用日线数据生成5分钟数据
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc"
    source = TDXDataSource(tdx_path)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    daily_df = source.get_kline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period='daily'
    )

    if daily_df.empty:
        print("无法获取日线数据")
        return None, None

    daily_df = daily_df.reset_index()
    print(f"使用 {len(daily_df)} 天日线数据生成5分钟数据")

    # 为每一天生成5分钟K线
    all_5min = []

    for _, day in daily_df.iterrows():
        day_date = day['datetime'].date() if isinstance(day['datetime'], pd.Timestamp) else day['datetime']

        # A股交易时段：9:30-11:30, 13:00-15:00
        # 共4小时 = 240分钟 = 48根5分钟K线

        morning_start = datetime.combine(day_date, time(9, 30))
        morning_end = datetime.combine(day_date, time(11, 30))
        afternoon_start = datetime.combine(day_date, time(13, 0))
        afternoon_end = datetime.combine(day_date, time(15, 0))

        # 生成早盘数据
        morning_bars = _generate_intraday_bars(
            morning_start, morning_end,
            day['open'], day['high'], day['low'], day['close'],
            volume=day['volume'] * 0.5,
            period_minutes=5
        )

        # 生成午盘数据
        afternoon_bars = _generate_intraday_bars(
            afternoon_start, afternoon_end,
            morning_bars[-1]['close'], day['high'], day['low'], day['close'],
            volume=day['volume'] * 0.5,
            period_minutes=5
        )

        all_5min.extend(morning_bars + afternoon_bars)

    df_5min = pd.DataFrame(all_5min)
    print(f"生成5分钟K线: {len(df_5min)} 根")

    return df_5min, daily_df


def _generate_intraday_bars(
    start_time: datetime,
    end_time: datetime,
    open_price: float,
    day_high: float,
    day_low: float,
    close_price: float,
    volume: float,
    period_minutes: int = 5
):
    """生成单个交易时段的5分钟K线"""
    import random

    bars = []
    current_time = start_time

    total_minutes = int((end_time - start_time).total_seconds() / 60)
    num_bars = total_minutes // period_minutes

    prev_close = open_price

    for i in range(num_bars):
        bar_time = current_time + timedelta(minutes=period_minutes * i)

        # 模拟价格波动
        if i == 0:
            bar_open = prev_close
        else:
            bar_open = bars[-1]['close']

        # 随机波动（基于当日振幅）
        price_range = day_high - day_low
        volatility = price_range / num_bars * random.uniform(0.5, 1.5)

        # 趋势倾向（50%概率延续，50%概率反转）
        trend = 1 if random.random() > 0.5 else -1

        bar_high = bar_open + volatility * abs(random.random()) * trend
        bar_low = bar_open - volatility * abs(random.random()) * trend

        # 确保在当日范围内
        bar_high = min(bar_high, day_high)
        bar_low = max(bar_low, day_low)

        # 收盘价
        bar_close = bar_open + (bar_high - bar_low) * random.uniform(0.3, 0.7) * trend

        # 最后一根K线趋向收盘价
        if i == num_bars - 1:
            bar_close = close_price

        bar_high = max(bar_open, bar_close, bar_high)
        bar_low = min(bar_open, bar_close, bar_low)

        # 成交量
        bar_volume = volume / num_bars * random.uniform(0.5, 1.5)

        bars.append({
            'datetime': bar_time,
            'open': bar_open,
            'high': bar_high,
            'low': bar_low,
            'close': bar_close,
            'volume': int(bar_volume),
            'amount': bar_volume * bar_close
        })

    return bars


def run_backtest_demo(symbol: str = 'sh600519', days: int = 5):
    """运行回测演示"""
    print("=" * 60)
    print(f"日内交易回测演示 - {symbol}")
    print("=" * 60)

    # 获取数据
    df_5min, daily_df = simulate_5min_data(symbol, days)

    if df_5min is None:
        print("数据获取失败")
        return

    print("\n" + "-" * 60)
    print("数据预览:")
    print(df_5min[['datetime', 'open', 'high', 'low', 'close', 'volume']].head(10))
    print("...")
    print(df_5min[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail(5))

    # 创建策略 - 使用更简单的测试策略
    print("\n" + "-" * 60)
    print("创建策略...")

    from backtest.intraday_engine import Signal, SignalType

    # 简单测试策略：第一天买入，有持仓就卖出
    buy_done = [False]

    def simple_strategy(bar, context, daily_context):
        """简单测试策略"""
        # 第一根K线买入
        if not buy_done[0] and context['position'] == 0:
            buy_done[0] = True
            return Signal(
                type=SignalType.BUY,
                datetime=context['datetime'],
                price=bar['close'],
                reason="测试买入"
            )

        # 有持仓后，下一根K线就卖出
        if context['position'] > 0:
            return Signal(
                type=SignalType.SELL,
                datetime=context['datetime'],
                price=bar['close'],
                reason="测试卖出"
            )

        return None

    strategy = simple_strategy

    # 配置回测 - 增加初始资金以支持高价股
    config = IntradayConfig(
        initial_capital=500000,  # 50万初始资金
        commission_rate=0.0003,
        slippage=0.001,
        min_unit=100,
        max_position_pct=0.95,
        force_close_time=time(14, 50),
        t1_rule=False  # 日内交易不需要T+1
    )

    # 运行回测
    print("运行回测...")
    engine = IntradayEngine(config)

    results = engine.run(
        df=df_5min,
        strategy=strategy,
        daily_df=daily_df
    )

    # 打印结果
    engine.print_results(results)

    # 显示交易明细
    if results['trades']:
        print("\n交易明细:")
        print("-" * 60)
        for i, trade in enumerate(results['trades'], 1):
            duration = f"{trade.duration_minutes}分钟" if trade.duration_minutes else "未平仓"
            pnl_str = f"+{trade.pnl:.2f}" if trade.pnl > 0 else f"{trade.pnl:.2f}"
            exit_price_str = f"{trade.exit_price:.2f}" if trade.exit_price else "---"
            exit_time_str = trade.exit_time.strftime('%H:%M') if trade.exit_time else '---'
            print(f"{i}. {trade.entry_time.strftime('%m-%d %H:%M')} -> "
                  f"{exit_time_str} "
                  f"| {trade.entry_price:.2f} -> {exit_price_str} "
                  f"| {duration} | {pnl_str} ({trade.exit_reason})")

    # 生成图表
    print("\n生成图表...")

    # 从回测引擎获取信号
    signals = engine.signals

    fig = create_intraday_chart(
        df=df_5min,
        signals=signals,
        title=f"{symbol} 日内交易图 - 5分钟K线"
    )

    # 保存图表
    output_file = f"intraday_{symbol.replace('sh', '').replace('sz', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(output_file)
    print(f"图表已保存: {output_file}")
    print(f"请用浏览器打开文件查看图表")

    # 不自动显示图表，避免闪退
    # fig.show()

    return results, fig, df_5min


def analyze_fractals_5min(symbol: str = 'sh600519', days: int = 3):
    """分析5分钟K线的分型"""
    print("=" * 60)
    print(f"5分钟K线分型分析 - {symbol}")
    print("=" * 60)

    # 获取数据
    df_5min, _ = simulate_5min_data(symbol, days)

    if df_5min is None:
        return

    # 创建K线对象
    kline = KLine.from_dataframe(df_5min, strict_mode=True)

    print(f"原始K线: {len(kline.raw_data)} 根")
    print(f"合并后: {len(kline.processed_data)} 根")

    # 识别分型
    detector = FractalDetector(kline, confirm_required=False)

    print(f"\n识别分型: {len(detector)} 个")
    print(f"  顶分型: {len(detector.get_top_fractals())} 个")
    print(f"  底分型: {len(detector.get_bottom_fractals())} 个")

    # 生成笔
    stroke_gen = StrokeGenerator(kline)
    strokes = stroke_gen.get_strokes()
    print(f"\n生成笔: {len(strokes)} 根")

    # 创建图表
    chart = ChanLunChart(kline, title=f"{symbol} 5分钟缠论分析")
    chart.add_fractals(detector.fractals)
    chart.add_strokes(strokes)

    fig = chart.create_chart(height=800, width=1200)

    # 保存图表
    output_file = f"fractal_5min_{symbol.replace('sh', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(output_file)
    print(f"图表已保存: {output_file}")

    # 不自动显示图表，避免闪退
    # chart.show()

    return kline, detector, strokes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='日内交易回测演示')
    parser.add_argument('--symbol', default='sh600519', help='股票代码')
    parser.add_argument('--days', type=int, default=5, help='天数')
    parser.add_argument('--analyze', action='store_true', help='仅分析分型，不回测')

    args = parser.parse_args()

    if args.analyze:
        analyze_fractals_5min(args.symbol, args.days)
    else:
        run_backtest_demo(args.symbol, args.days)
