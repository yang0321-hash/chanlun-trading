#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自适应策略回测 - 根据股票特性自动选择策略

对比:
1. 原版策略 (固定)
2. 优化策略 (固定)
3. 自适应策略 (智能选择)
"""

import json
import sys
import os
import subprocess
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy
from strategies.optimized_weekly_daily_strategy import OptimizedWeeklyDailyStrategy
from strategies.adaptive_strategy import AdaptiveChanLunStrategy


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


def parse_tdx_data(symbol: str) -> bool:
    """调用tdx-parser解析通达信数据"""
    project_root = Path(__file__).parent
    tdx_parser_dir = project_root / ".claude" / "skills" / "tdx-parser" / "scripts"
    parse_script = tdx_parser_dir / "parse_tdx.js"

    if not parse_script.exists():
        return False

    tdx_paths = [
        'D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc',
        'D:/新建通达信/vipdoc',
        'D:/通达信/vipdoc',
    ]

    tdx_path = None
    for p in tdx_paths:
        if Path(p).exists():
            tdx_path = p
            break

    if not tdx_path:
        return False

    try:
        market = "sh" if symbol.startswith(("sh", "6", "900")) else "sz"
        output_dir = project_root / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "node",
            str(parse_script),
            "--input", tdx_path,
            "--code", f"{market}{symbol.replace('sh', '').replace('sz', '')}",
            "--output", str(output_dir),
            "--format", "json",
            "--date-format", "iso"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
        return result.returncode == 0
    except:
        return False


def run_backtest(symbol: str, strategy, strategy_name: str) -> dict:
    """运行单个策略回测"""
    data_file = f"test_output/{symbol.lower()}.day.json"

    if not os.path.exists(data_file):
        if not parse_tdx_data(symbol):
            return None

    df = load_tdx_data(data_file)

    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100
    )

    engine = BacktestEngine(config)
    engine.add_data(symbol, df)
    engine.set_strategy(strategy)

    results = engine.run()
    results['initial_capital'] = config.initial_capital
    results['strategy_name'] = strategy_name

    # 计算买入持有收益
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    results['buy_hold_return'] = buy_hold_return

    return results


def print_comparison(symbol: str, results_dict: dict):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print(f"策略对比 - {symbol}")
    print("=" * 80)

    # 提取指标
    strategies = ['Original', 'Optimized', 'Adaptive']
    metrics = [
        ('Total Return', 'total_return', '%'),
        ('Annual Return', 'annual_return', '%'),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Profit/Loss', 'profit_loss_ratio', ''),
        ('Trades', 'total_trades', ''),
    ]

    print(f"\n{'Metric':<12} {strategies[0]:<15} {strategies[1]:<15} {strategies[2]:<15}")
    print("-" * 80)

    for name, key, unit in metrics:
        values = []
        for strategy in strategies:
            if strategy in results_dict and results_dict[strategy]:
                val = results_dict[strategy].get(key, 0)
                if key in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
                    val = val * 100
                if key in ['sharpe_ratio', 'profit_loss_ratio']:
                    values.append(f"{val:.2f}")
                elif key == 'total_trades':
                    values.append(f"{int(val)}")
                else:
                    values.append(f"{val:.2f}{unit}")
            else:
                values.append("-")

        print(f"{name:<12} {values[0]:<15} {values[1]:<15} {values[2]:<15}")

    # 对比买入持有
    if 'Original' in results_dict and results_dict['Original']:
        buy_hold = results_dict['Original'].get('buy_hold_return', 0)
        print("\n" + "-" * 80)
        print(f"Buy & Hold Return: {buy_hold:.2f}%")

        for strategy in strategies:
            if strategy in results_dict:
                ret = results_dict[strategy].get('total_return', 0) * 100
                excess = ret - buy_hold
                print(f"{strategy} Excess: {excess:+.2f}%")

    print("=" * 80)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='自适应策略回测')
    parser.add_argument('symbols', nargs='*', help='股票代码')

    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")

    print("\n" + "=" * 80)
    print("自适应缠论策略回测 - 智能选择策略")
    print("=" * 80)
    print("\n策略说明:")
    print("  - 震荡市 (R2<0.3) -> 优化策略 (趋势过滤减少无效交易)")
    print("  - 强趋势+高波动 (R2>0.7, 波动>30%) -> 原版策略 (捕捉机会)")
    print("  - 其他情况 -> 优化策略 (稳健交易)")

    # 默认股票
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = ['sz002600', 'sh600000']

    all_results = {symbol: {} for symbol in symbols}

    for symbol in symbols:
        print(f"\n正在回测 {symbol}...")

        # 运行三种策略
        strategies_config = [
            (WeeklyDailyChanLunStrategy(name='Original'), 'Original'),
            (OptimizedWeeklyDailyStrategy(name='Optimized'), 'Optimized'),
            (AdaptiveChanLunStrategy(name='Adaptive'), 'Adaptive'),
        ]

        for strategy, strategy_name in strategies_config:
            try:
                results = run_backtest(symbol, strategy, strategy_name)
                all_results[symbol][strategy_name] = results
            except Exception as e:
                print(f"  {strategy_name}失败: {e}")

        # 打印对比
        print_comparison(symbol, all_results[symbol])

    # 汇总统计
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for strategy_name in ['Original', 'Optimized', 'Adaptive']:
        returns = []
        count = 0
        for symbol in symbols:
            if strategy_name in all_results[symbol] and all_results[symbol][strategy_name]:
                ret = all_results[symbol][strategy_name].get('total_return', 0) * 100
                returns.append(ret)
                count += 1

        if returns:
            avg_return = sum(returns) / len(returns)
            print(f"  {strategy_name}: Avg Return {avg_return:.2f}% ({count}/{len(symbols)} stocks)")

    print("\nBacktest Complete!")


if __name__ == "__main__":
    main()
