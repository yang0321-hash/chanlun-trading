"""
测试最优配置在sz002600(领益智造)上的表现
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.chanlun_trading_system import ChanLunTradingSystem


def load_tdx_json(code: str, json_dir: str = '.claude/temp') -> pd.DataFrame:
    json_path = f"{json_dir}/{code}.day.json"
    if not os.path.exists(json_path):
        logger.error(f"未找到数据文件: {json_path}")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df['amount'] = df['volume'] * df['close']
    logger.info(f"加载 {code}: {len(df)} 条数据, 日期范围: {df.index[0]} ~ {df.index[-1]}")
    return df


def run_test(start_date: str = '2021-01-01', capital: float = 500000):
    df = load_tdx_json('sz002600')
    if df is None:
        return
    df = df[df.index >= start_date]

    logger.info("=" * 80)
    logger.info("sz002600 (领益智造) 最优配置回测")
    logger.info("=" * 80)

    results = {}
    config = BacktestConfig(
        initial_capital=capital,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    # A. 基准: 无任何优化
    logger.info("\n测试A: 基准配置 (2买, 无量能, 2卖清仓)")
    strategy_a = ChanLunTradingSystem(
        name='基准',
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=False,
        min_confidence=0.60,
        enable_volume_confirm=False,
        second_sell_partial_exit=False,
    )
    engine_a = BacktestEngine(config)
    engine_a.add_data('sz002600', df)
    engine_a.set_strategy(strategy_a)
    results['基准'] = engine_a.run()

    # B. 量能确认
    logger.info("\n测试B: +量能确认(1.2倍)")
    strategy_b = ChanLunTradingSystem(
        name='+量能确认',
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=False,
        min_confidence=0.60,
        enable_volume_confirm=True,
        min_volume_ratio=1.2,
        second_sell_partial_exit=False,
    )
    engine_b = BacktestEngine(config)
    engine_b.add_data('sz002600', df)
    engine_b.set_strategy(strategy_b)
    results['+量能确认'] = engine_b.run()

    # C. 量能确认 + 2卖减仓30%
    logger.info("\n测试C: +量能确认+2卖减仓30% (最优配置)")
    strategy_c = ChanLunTradingSystem(
        name='最优配置',
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=False,
        min_confidence=0.60,
        enable_volume_confirm=True,
        min_volume_ratio=1.2,
        second_sell_partial_exit=True,
        second_sell_exit_ratio=0.3,
    )
    engine_c = BacktestEngine(config)
    engine_c.add_data('sz002600', df)
    engine_c.set_strategy(strategy_c)
    results['最优配置'] = engine_c.run()

    # 打印结果
    print_results(results)

    return results


def print_results(results: dict):
    print("\n" + "=" * 100)
    print("sz002600 (领益智造) 配置对比报告".center(100))
    print("=" * 100 + "\n")

    print(f"{'配置':<20}{'收益率':<12}{'年化':<10}{'夏普':<8}{'回撤':<10}{'胜率':<10}{'盈亏比':<10}{'交易':<8}")
    print("-" * 100)

    for name, result in results.items():
        ret_str = f"[OK]{result['total_return']:>6.2%}" if result['total_return'] > 0 else f"[ ]{result['total_return']:>6.2%}"
        print(f"{name:<20}{ret_str:<12}{result['annual_return']:>8.2%}  "
              f"{result['sharpe_ratio']:>6.2f}  {result['max_drawdown']:>8.2%}  "
              f"{result['win_rate']:>8.2%}  {result['profit_loss_ratio']:>8.2f}  "
              f"{result['total_trades']:>6}")

    print("-" * 100)
    print()

    # 最佳配置
    best_name = max(results.keys(), key=lambda k: results[k]['total_return'])
    best_result = results[best_name]

    print("=" * 100)
    print(f"【最佳配置】: {best_name}")
    print("=" * 100)
    print(f"总收益率: {best_result['total_return']:.2%}")
    print(f"年化收益: {best_result['annual_return']:.2%}")
    print(f"最大回撤: {best_result['max_drawdown']:.2%}")
    print(f"胜率: {best_result['win_rate']:.2%}")
    print(f"盈亏比: {best_result['profit_loss_ratio']:.2f}")
    print(f"交易次数: {best_result['total_trades']}")
    print()


if __name__ == '__main__':
    run_test()
