#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
周线+日线策略参数优化

优化目标: 夏普比率
"""

import json
import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from loguru import logger
from itertools import product

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.optimization import GridSearchOptimizer
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy


def load_tdx_data(json_file: str) -> pd.DataFrame:
    """加载通达信JSON数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")

    return df


def run_backtest_with_params(params: dict, data: pd.DataFrame, symbol: str = "TEST") -> dict:
    """使用给定参数运行回测"""
    # 创建策略
    strategy = WeeklyDailyChanLunStrategy(
        name='周日线缠论策略',
        weekly_min_strokes=params.get('weekly_min_strokes', 3),
        daily_min_strokes=params.get('daily_min_strokes', 3),
        stop_loss_pct=params.get('stop_loss_pct', 0.08),
        exit_ratio=params.get('exit_ratio', 0.5)
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
    results['initial_capital'] = config.initial_capital

    return results


def print_optimization_summary(optimizer: GridSearchOptimizer):
    """打印优化摘要"""
    print("\n" + "=" * 70)
    print("优化结果摘要")
    print("=" * 70)

    print(f"\n最优参数:")
    for key, value in optimizer.best_params.items():
        print(f"  {key}: {value}")

    print(f"\n最优指标:")
    print(f"  夏普比率: {optimizer.best_score:.4f}")

    # 获取前10个结果
    top_n = optimizer.get_top_n(10)

    print(f"\n前10名参数组合:")
    print("-" * 70)
    print(f"{'排名':<4} {'夏普':<8} {'收益率':<10} {'回撤':<10} {'交易数':<6} {'参数'}")
    print("-" * 70)

    for i, result in enumerate(top_n, 1):
        params_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
        print(f"{i:<4} {result.sharpe_ratio:<8.2f} {result.total_return:<10.2%} "
              f"{result.max_drawdown:<10.2%} {result.trades:<6} {params_str}")


def param_sensitivity_analysis(data: pd.DataFrame, symbol: str = "TEST"):
    """参数敏感性分析"""
    print("\n" + "=" * 70)
    print("参数敏感性分析")
    print("=" * 70)

    # 基准参数
    base_params = {
        'weekly_min_strokes': 3,
        'daily_min_strokes': 3,
        'stop_loss_pct': 0.08,
        'exit_ratio': 0.5
    }

    # 分析每个参数
    param_ranges = {
        'weekly_min_strokes': [2, 3, 4, 5, 6],
        'daily_min_strokes': [2, 3, 4, 5, 6],
        'stop_loss_pct': [0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
        'exit_ratio': [0.3, 0.4, 0.5, 0.6, 0.7]
    }

    for param_name, values in param_ranges.items():
        print(f"\n{param_name} 敏感性:")
        print(f"{'值':<8} {'夏普':<8} {'收益率':<10} {'回撤':<10} {'交易数':<6}")
        print("-" * 50)

        sharpe_values = []

        for value in values:
            params = base_params.copy()
            params[param_name] = value

            result = run_backtest_with_params(params, data, symbol)
            sharpe = result.get('sharpe_ratio', 0)
            sharpe_values.append(sharpe)

            print(f"{value:<8} {sharpe:<8.2f} {result.get('total_return', 0):<10.2%} "
                  f"{result.get('max_drawdown', 0):<10.2%} {result.get('total_trades', 0):<6}")

        # 计算变异系数
        mean_sharpe = np.mean(sharpe_values)
        std_sharpe = np.std(sharpe_values)
        cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float('inf')

        print(f"  变异系数: {cv:.4f} ({'敏感' if cv > 0.3 else '稳定'})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='周线+日线策略参数优化')
    parser.add_argument('--symbol', default='sz002600', help='股票代码')
    parser.add_argument('--data-dir', default='test_output', help='数据目录')
    parser.add_argument('--sensitivity', action='store_true', help='运行敏感性分析')
    parser.add_argument('--quick', action='store_true', help='快速优化(小范围)')

    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")

    print("\n" + "=" * 70)
    print("周线+日线策略参数优化")
    print("=" * 70)
    print(f"标的: {args.symbol}")
    print(f"优化目标: 夏普比率")

    # 加载数据
    data_file = f"{args.data_dir}/{args.symbol.lower()}.day.json"

    if not os.path.exists(data_file):
        print(f"\n数据文件不存在: {data_file}")
        print("请先运行 backtest_weekly_daily.py 生成数据")
        return

    print(f"\n加载数据: {data_file}")
    data = load_tdx_data(data_file)
    print(f"数据范围: {data.index[0].date()} ~ {data.index[-1].date()}")
    print(f"K线数量: {len(data)}")

    # 敏感性分析
    if args.sensitivity:
        param_sensitivity_analysis(data, args.symbol)
        return

    # 定义参数网格
    if args.quick:
        param_grid = {
            'weekly_min_strokes': [2, 3, 4],
            'daily_min_strokes': [2, 3, 4],
            'stop_loss_pct': [0.06, 0.08, 0.10],
            'exit_ratio': [0.4, 0.5, 0.6],
        }
    else:
        param_grid = {
            'weekly_min_strokes': [2, 3, 4, 5],
            'daily_min_strokes': [2, 3, 4, 5],
            'stop_loss_pct': [0.05, 0.06, 0.08, 0.10, 0.12],
            'exit_ratio': [0.3, 0.4, 0.5, 0.6, 0.7],
        }

    # 计算组合数
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n参数网格:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print(f"\n总组合数: {total_combinations}")

    # 创建优化器
    optimizer = GridSearchOptimizer(
        objective='sharpe_ratio',
        maximize=True,
        n_jobs=1  # 单线程，避免多进程问题
    )

    # 定义回测函数
    def backtest_func(params):
        return run_backtest_with_params(params, data, args.symbol)

    # 执行优化
    print("\n开始优化...")
    print("-" * 70)

    best_result = optimizer.optimize(
        param_grid=param_grid,
        backtest_func=backtest_func,
        data=data,
        min_trades=5  # 最少交易次数
    )

    # 打印结果
    print_optimization_summary(optimizer)

    # 保存结果
    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{args.symbol}_optimization_results.json"
    optimizer.save_results(str(results_file))
    print(f"\n结果已保存到: {results_file}")

    # 使用最优参数回测
    if best_result:
        print("\n" + "=" * 70)
        print("最优参数详细回测")
        print("=" * 70)

        final_result = run_backtest_with_params(optimizer.best_params, data, args.symbol)

        print(f"\n初始资金: ¥{final_result.get('initial_capital', 100000):,.2f}")
        print(f"最终权益: ¥{final_result.get('final_equity', 0):,.2f}")
        print(f"总收益率: {final_result.get('total_return', 0):.2%}")
        print(f"年化收益: {final_result.get('annual_return', 0):.2%}")
        print(f"夏普比率: {final_result.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {final_result.get('max_drawdown', 0):.2%}")
        print(f"胜率: {final_result.get('win_rate', 0):.2%}")
        print(f"盈亏比: {final_result.get('profit_loss_ratio', 0):.2f}")
        print(f"交易次数: {final_result.get('total_trades', 0)}")


if __name__ == "__main__":
    main()
