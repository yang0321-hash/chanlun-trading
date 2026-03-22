#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用通达信本地数据回测 002600 交易策略
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime

from core.kline import KLine, merge_inclusions
from core.fractal import Fractal, find_fractals, FractalType
from core.stroke import Stroke, generate_strokes
from core.segment import Segment, generate_segments
from core.pivot import Pivot, find_pivots
from indicator.macd import MACD
from strategies.weekly_daily_strategy import WeeklyDailyStrategy


def load_tdx_data(json_file: str) -> pd.DataFrame:
    """加载通达信JSON数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """将日线数据重采样为周线"""
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum'
    }).dropna()

    return weekly


def run_backtest_002600():
    """运行002600回测"""
    print("=" * 60)
    print("002600 驰宏锌锗 回测报告 (通达信数据)")
    print("=" * 60)

    # 加载数据
    data_file = "test_output/sz002600.day.json"
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        return

    daily_df = load_tdx_data(data_file)
    print(f"\n数据概览:")
    print(f"  日期范围: {daily_df.index[0].date()} ~ {daily_df.index[-1].date()}")
    print(f"  K线数量: {len(daily_df)} 条")
    print(f"  最新价格: {daily_df['close'].iloc[-1]:.2f}")
    print(f"  期间涨跌: {(daily_df['close'].iloc[-1] / daily_df['close'].iloc[0] - 1) * 100:.2f}%")

    # 转换为周线
    weekly_df = resample_to_weekly(daily_df)
    print(f"  周线数量: {len(weekly_df)} 条")

    # 创建K线对象
    daily_klines = [KLine(
        date=row.name,
        open=row['open'],
        high=row['high'],
        low=row['low'],
        close=row['close'],
        volume=row['volume']
    ) for _, row in daily_df.iterrows()]

    weekly_klines = [KLine(
        date=row.name,
        open=row['open'],
        high=row['high'],
        low=row['low'],
        close=row['close'],
        volume=row['volume']
    ) for _, row in weekly_df.iterrows()]

    # 处理包含关系
    daily_klines = merge_inclusions(daily_klines)
    weekly_klines = merge_inclusions(weekly_klines)

    # 计算MACD
    macd_indicator = MACD()
    daily_macd = macd_indicator.calculate(daily_df)
    weekly_macd = macd_indicator.calculate(weekly_df)

    # 识别分型
    daily_fractals = find_fractals(daily_klines)
    weekly_fractals = find_fractals(weekly_klines)

    print(f"\n缠论结构识别:")
    print(f"  日线分型: {len(daily_fractals)} 个")
    print(f"  周线分型: {len(weekly_fractals)} 个")

    # 生成笔
    daily_strokes = generate_strokes(daily_fractals, daily_klines)
    weekly_strokes = generate_strokes(weekly_fractals, weekly_klines)

    print(f"  日线笔: {len(daily_strokes)} 个")
    print(f"  周线笔: {len(weekly_strokes)} 个")

    # 生成线段
    daily_segments = generate_segments(daily_strokes, daily_klines)
    weekly_segments = generate_segments(weekly_strokes, weekly_klines)

    print(f"  日线线段: {len(daily_segments)} 个")
    print(f"  周线线段: {len(weekly_segments)} 个")

    # 识别中枢
    daily_pivots = find_pivots(daily_strokes, daily_klines)
    weekly_pivots = find_pivots(weekly_strokes, weekly_klines)

    print(f"  日线中枢: {len(daily_pivots)} 个")
    print(f"  周线中枢: {len(weekly_pivots)} 个")

    # 运行策略回测
    print("\n" + "=" * 60)
    print("策略回测 (周线2买 + 日线MACD/2卖)")
    print("=" * 60)

    strategy = WeeklyDailyStrategy()

    trades = []  # (entry_date, entry_price, exit_date, exit_price, profit_pct)

    position = 0  # 0: 空仓, 0.5: 半仓, 1: 满仓
    entry_price = 0
    entry_date = None

    for i in range(len(daily_klines)):
        if i < 50:  # 需要足够的历史数据
            continue

        current_kline = daily_klines[i]
        current_date = current_kline.date

        # 获取当前MACD值
        if i < len(daily_macd):
            curr_macd = daily_macd.iloc[i]
        else:
            continue

        # 简化的策略逻辑 (演示用)
        # 实际需要匹配周线2买信号

        # 买入信号: 周线级别2买 (简化判断)
        if position == 0:
            # 检查是否形成周线底分型回抽
            recent_weekly_fractals = [f for f in weekly_fractals
                                     if f.kline_index < len(weekly_klines) - 1]
            if recent_weekly_fractals:
                last_fractal = recent_weekly_fractals[-1]
                if last_fractal.fractal_type == FractalType.BOTTOM:
                    # 日线MACD金叉
                    if i > 0 and i < len(daily_macd):
                        prev_macd = daily_macd.iloc[i-1]
                        if (curr_macd['dif'] > curr_macd['dea'] and
                            prev_macd['dif'] <= prev_macd['dea']):
                            position = 1
                            entry_price = current_kline.close
                            entry_date = current_date
                            print(f"买入: {current_date.date()} @ {entry_price:.2f}")

        # 卖出信号
        elif position > 0:
            exit_signal = False
            exit_reason = ""

            # 止损: 跌破买入价5%
            if current_kline.close < entry_price * 0.95:
                exit_signal = True
                exit_reason = "止损"

            # 日线MACD顶背离 (简化)
            elif i > 10 and i < len(daily_macd):
                recent_highs = [daily_klines[j].close for j in range(i-10, i)]
                if max(recent_highs) == current_kline.close:
                    macd_recent = [daily_macd.iloc[j]['macd'] for j in range(i-10, i) if j < len(daily_macd)]
                    if len(macd_recent) > 0:
                        if curr_macd['macd'] < max(macd_recent) * 0.9:
                            exit_signal = True
                            exit_reason = "MACD顶背离"

            # 2卖信号 (简化)
            elif i > 0:
                prev_macd = daily_macd.iloc[i-1]
                if (curr_macd['dif'] < curr_macd['dea'] and
                    prev_macd['dif'] >= prev_macd['dea']):
                    exit_signal = True
                    exit_reason = "MACD死叉"

            if exit_signal:
                profit_pct = (current_kline.close / entry_price - 1) * 100
                trades.append((entry_date, entry_price, current_date, current_kline.close, profit_pct, exit_reason))
                print(f"卖出: {current_date.date()} @ {current_kline.close:.2f} ({exit_reason}) 收益: {profit_pct:.2f}%")
                position = 0
                entry_price = 0
                entry_date = None

    # 统计结果
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    if trades:
        winning_trades = [t for t in trades if t[4] > 0]
        losing_trades = [t for t in trades if t[4] <= 0]

        total_profit = sum(t[4] for t in trades)
        avg_profit = total_profit / len(trades)

        print(f"  总交易次数: {len(trades)}")
        print(f"  盈利次数: {len(winning_trades)}")
        print(f"  亏损次数: {len(losing_trades)}")
        print(f"  胜率: {len(winning_trades)/len(trades)*100:.2f}%")
        print(f"  总收益: {total_profit:.2f}%")
        print(f"  平均收益: {avg_profit:.2f}%")

        if winning_trades:
            avg_win = sum(t[4] for t in winning_trades) / len(winning_trades)
            max_win = max(t[4] for t in winning_trades)
            print(f"  平均盈利: {avg_win:.2f}%")
            print(f"  最大盈利: {max_win:.2f}%")

        if losing_trades:
            avg_loss = sum(t[4] for t in losing_trades) / len(losing_trades)
            max_loss = min(t[4] for t in losing_trades)
            print(f"  平均亏损: {avg_loss:.2f}%")
            print(f"  最大亏损: {max_loss:.2f}%")

        # 盈亏比
        if winning_trades and losing_trades:
            profit_loss_ratio = abs(sum(t[4] for t in winning_trades) / sum(t[4] for t in losing_trades))
            print(f"  盈亏比: {profit_loss_ratio:.2f}")
    else:
        print("  未产生交易信号")

    # 与买入持有比较
    buy_hold_return = (daily_df['close'].iloc[-1] / daily_df['close'].iloc[0] - 1) * 100
    print(f"\n  买入持有收益: {buy_hold_return:.2f}%")

    print("\n回测完成!")


if __name__ == "__main__":
    run_backtest_002600()
