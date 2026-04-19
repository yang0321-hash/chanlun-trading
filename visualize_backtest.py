#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测买卖点可视化脚本
绘制K线图并标注买卖点
"""

import json
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_tdx_data(json_file: str) -> pd.DataFrame:
    """加载通达信JSON数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df


def run_backtest_with_signals(symbol: str, data: pd.DataFrame,
                              weekly_strokes: int = 4,
                              daily_strokes: int = 4,
                              stop_loss: float = 0.05,
                              exit_ratio: float = 0.7):
    """运行回测并记录买卖信号"""

    # 创建策略
    strategy = WeeklyDailyChanLunStrategy(
        name='周日线缠论策略',
        weekly_min_strokes=weekly_strokes,
        daily_min_strokes=daily_strokes,
        stop_loss_pct=stop_loss,
        exit_ratio=exit_ratio
    )

    # 创建回测引擎
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100
    )

    engine = BacktestEngine(config)
    engine.add_data(symbol, data)
    engine.set_strategy(strategy)

    # 运行回测
    results = engine.run()

    # 从回测引擎获取信号（已自动记录）
    signals = engine.get_signals()

    # 转换信号格式以便绘图
    signal_list = []
    for sig in signals:
        signal_list.append({
            'date': sig.datetime,
            'price': sig.price,
            'type': sig.signal_type.value.upper(),  # 'BUY' or 'SELL'
            'quantity': sig.quantity,
            'reason': sig.reason
        })

    return results, signal_list


def plot_trading_signals(df: pd.DataFrame, signals: list,
                          symbol: str = 'sz000001',
                          save_path: str = None):
    """
    绘制K线图并标注买卖点

    Args:
        df: K线数据
        signals: 买卖信号列表
        symbol: 股票代码
        save_path: 保存路径
    """

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                     height_ratios=[3, 1],
                                     sharex=True)
    fig.subplots_adjust(hspace=0.05)

    df = df.reset_index(drop=False)

    # 转换日期格式
    dates = df['date'].values

    # ==================== 绘制K线 ====================
    for i, row in df.iterrows():
        color = 'red' if row['close'] >= row['open'] else 'green'
        # 影线
        ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        # 实体
        ax1.plot([i, i], [row['open'], row['close']], color=color, linewidth=2)

    ax1.set_xlim(-1, len(df))
    ax1.set_ylabel('价格', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{symbol} 回测买卖点图', fontsize=14, fontweight='bold')

    # ==================== 标注买卖信号 ====================
    buy_signals = [s for s in signals if s['type'] == 'BUY']
    sell_signals = [s for s in signals if s['type'] == 'SELL']

    for sig in buy_signals:
        # 找到对应的x位置
        sig_date = sig['date'] if isinstance(sig['date'], pd.Timestamp) else pd.to_datetime(sig['date'])
        # 确保date只比较日期部分
        sig_date_only = sig_date.date() if hasattr(sig_date, 'date') else sig_date
        idx = df[df['date'].dt.date == sig_date_only].index

        if len(idx) > 0:
            x = idx[0]
            y = sig['price']
            # 买入标记 - 红色向上箭头
            ax1.scatter(x, y, marker='^', s=200, color='red',
                      edgecolors='darkred', zorder=5, linewidths=1.5)
            # 添加标签
            ax1.text(x, y * 0.985, f'买\n{y:.2f}',
                    fontsize=8, color='red', ha='center', va='top',
                    fontweight='bold')

    for sig in sell_signals:
        sig_date = sig['date'] if isinstance(sig['date'], pd.Timestamp) else pd.to_datetime(sig['date'])
        sig_date_only = sig_date.date() if hasattr(sig_date, 'date') else sig_date
        idx = df[df['date'].dt.date == sig_date_only].index

        if len(idx) > 0:
            x = idx[0]
            y = sig['price']
            # 卖出标记 - 绿色向下箭头
            ax1.scatter(x, y, marker='v', s=200, color='green',
                      edgecolors='darkgreen', zorder=5, linewidths=1.5)
            ax1.text(x, y * 1.015, f'卖\n{y:.2f}',
                    fontsize=8, color='green', ha='center', va='bottom',
                    fontweight='bold')

    # ==================== 绘制持仓线 ====================
    # 模拟持仓变化
    position = 0
    for i, row in df.iterrows():
        # 检查当天是否有信号
        row_date = row['date']
        row_date_only = row_date.date() if hasattr(row_date, 'date') else row_date
        for sig in signals:
            sig_date = sig['date'] if isinstance(sig['date'], pd.Timestamp) else pd.to_datetime(sig['date'])
            sig_date_only = sig_date.date() if hasattr(sig_date, 'date') else sig_date
            if sig_date_only == row_date_only:
                if sig['type'] == 'BUY':
                    position += sig['quantity'] if sig['quantity'] else 100
                elif sig['type'] == 'SELL':
                    position -= sig['quantity'] if sig['quantity'] else 100

        # 绘制持仓线
        if position > 0:
            ax2.plot(i, position, 'o', color='orange', markersize=2)

    # ==================== 成交量 ====================
    colors = ['red' if df.loc[i, 'close'] >= df.loc[i, 'open'] else 'green'
              for i in range(len(df))]
    ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # X轴日期标签
    def format_date(x, pos):
        if x < 0 or x >= len(df):
            return ''
        return df.iloc[int(x)]['date'].strftime('%Y-%m')

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ==================== 图例 ====================
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color='red', label='阳线'),
        Patch(color='green', label='阴线'),
        Patch(facecolor='red', edgecolor='darkred', label='买入信号'),
        Patch(facecolor='green', edgecolor='darkgreen', label='卖出信号'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def create_backtest_chart(symbol: str = 'sz000001',
                          data_dir: str = 'test_output',
                          output_dir: str = 'backtest_charts'):
    """
    创建回测图表

    Args:
        symbol: 股票代码
        data_dir: 数据目录
        output_dir: 输出目录
    """
    print("=" * 70)
    print("回测买卖点可视化")
    print("=" * 70)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    data_file = f"{data_dir}/{symbol.lower()}.day.json"
    print(f"\n加载数据: {data_file}")
    data = load_tdx_data(data_file)
    print(f"数据范围: {data.index[0].date()} ~ {data.index[-1].date()}")
    print(f"K线数量: {len(data)}")

    # 使用最优参数运行回测
    print("\n运行回测...")
    print("参数: weekly_min_strokes=4, daily_min_strokes=4, stop_loss_pct=0.05, exit_ratio=0.7")

    results, signals = run_backtest_with_signals(
        symbol=symbol,
        data=data,
        weekly_strokes=4,
        daily_strokes=4,
        stop_loss=0.05,
        exit_ratio=0.7
    )

    print(f"\n回测结果:")
    print(f"  总收益率: {results.get('total_return', 0):.2%}")
    print(f"  年化收益: {results.get('annual_return', 0):.2%}")
    print(f"  夏普比率: {results.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {results.get('max_drawdown', 0):.2%}")
    print(f"  交易次数: {results.get('total_trades', 0)}")
    print(f"  胜率: {results.get('win_rate', 0):.2%}")

    print(f"\n信号数量: {len(signals)}")
    print(f"  买入信号: {len([s for s in signals if s['type'] == 'BUY'])}")
    print(f"  卖出信号: {len([s for s in signals if s['type'] == 'SELL'])}")

    # 绘制完整图表
    print("\n生成图表...")
    save_path = f"{output_dir}/{symbol}_backtest_signals.png"
    # 重置索引以便绘图函数访问date列
    data_for_plot = data.reset_index(drop=False)
    plot_trading_signals(data_for_plot, signals, symbol=symbol, save_path=save_path)

    # 绘制最近200根K线
    if len(data) > 200:
        print(f"\n生成最近200根K线图...")
        recent_data = data.tail(200).reset_index(drop=False)
        # 过滤最近的信号
        recent_start = recent_data['date'].min()
        recent_signals = [s for s in signals
                          if (s['date'] if isinstance(s['date'], pd.Timestamp) else pd.to_datetime(s['date'])) >= recent_start]

        save_path_recent = f"{output_dir}/{symbol}_recent_200_signals.png"
        plot_trading_signals(recent_data, recent_signals, symbol=symbol,
                              save_path=save_path_recent)
    else:
        recent_data = data.reset_index(drop=False)
        recent_start = recent_data['date'].min()
        recent_signals = [s for s in signals
                          if (s['date'] if isinstance(s['date'], pd.Timestamp) else pd.to_datetime(s['date'])) >= recent_start]

        save_path_recent = f"{output_dir}/{symbol}_recent_200_signals.png"
        plot_trading_signals(recent_data, recent_signals, symbol=symbol,
                              save_path=save_path_recent)

        save_path_recent = f"{output_dir}/{symbol}_recent_200_signals.png"
        plot_trading_signals(recent_data, recent_signals, symbol=symbol,
                              save_path=save_path_recent)

    print("\n" + "=" * 70)
    print("可视化完成!")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  1. {output_dir}/{symbol}_backtest_signals.png - 完整图表")
    print(f"  2. {output_dir}/{symbol}_recent_200_signals.png - 最近200根K线")

    return results, signals


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='回测买卖点可视化')
    parser.add_argument('--symbol', default='sz000001', help='股票代码')
    parser.add_argument('--data-dir', default='test_output', help='数据目录')
    parser.add_argument('--output-dir', default='backtest_charts', help='输出目录')

    args = parser.parse_args()

    create_backtest_chart(
        symbol=args.symbol,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
