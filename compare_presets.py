#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比周线+日线策略的不同预设参数
"""

import json
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from backtest.engine import BacktestEngine, BacktestConfig
from strategies.weekly_daily_presets import PRESETS, create_strategy, compare_presets


def load_tdx_data(json_file: str) -> pd.DataFrame:
    """加载通达信JSON数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df


def run_comparison(symbol: str = "sh600519", data_dir: str = "test_output"):
    """运行预设参数对比"""

    # 显示预设对比
    compare_presets()

    # 加载数据
    data_file = f"{data_dir}/{symbol.lower()}.day.json"

    if not os.path.exists(data_file):
        print(f"\n数据文件不存在: {data_file}")
        print("可用的数据文件:")
        test_dir = Path(data_dir)
        for f in sorted(test_dir.glob("*.json"))[:10]:
            print(f"  {f.stem}")
        return

    print(f"\n加载数据: {data_file}")
    data = load_tdx_data(data_file)
    print(f"数据范围: {data.index[0].date()} ~ {data.index[-1].date()}")
    print(f"K线数量: {len(data)}")

    # 对比各预设
    print("\n" + "=" * 80)
    print("回测结果对比")
    print("=" * 80)

    results = {}

    for preset_key in PRESETS.keys():
        strategy = create_strategy(preset_key)

        config = BacktestConfig(
            initial_capital=100000,
            commission=0.0003,
            slippage=0.0001,
            min_unit=100
        )

        engine = BacktestEngine(config)
        engine.add_data(symbol, data)
        engine.set_strategy(strategy)

        result = engine.run()
        results[preset_key] = result

    # 打印对比表格
    print(f"\n{'预设':<15} {'夏普':<8} {'收益率':<10} {'回撤':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 80)

    for preset_key, result in results.items():
        sharpe = result.get('sharpe_ratio', 0)
        returns = result.get('total_return', 0)
        dd = result.get('max_drawdown', 0)
        trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)

        print(f"{preset_key:<15} {sharpe:<8.2f} {returns:<10.2%} {dd:<10.2%} {trades:<6} {win_rate:<8.1%}")

    # 找出最优
    best_sharpe_key = max(results.keys(), key=lambda k: results[k].get('sharpe_ratio', 0))
    best_return_key = max(results.keys(), key=lambda k: results[k].get('total_return', 0))

    print("\n" + "-" * 80)
    print(f"最高夏普比率: {best_sharpe_key} ({results[best_sharpe_key].get('sharpe_ratio', 0):.2f})")
    print(f"最高收益率: {best_return_key} ({results[best_return_key].get('total_return', 0):.2%})")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='对比周线+日线策略预设参数')
    parser.add_argument('--symbol', default='sh600519', help='股票代码')
    parser.add_argument('--data-dir', default='test_output', help='数据目录')

    args = parser.parse_args()

    run_comparison(args.symbol, args.data_dir)
