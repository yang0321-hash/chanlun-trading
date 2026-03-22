"""
测试延长止盈参数配置

通过调整参数实现延长止盈，不改变策略逻辑。
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
    logger.info(f"加载 {code}: {len(df)} 条数据")
    return df


def run_test(start_date: str = '2021-08-01', capital: float = 500000):
    df = load_tdx_json('sh600519')
    if df is None:
        return
    df = df[df.index >= start_date]

    logger.info("=" * 80)
    logger.info("延长止盈参数测试（贵州茅台）")
    logger.info("=" * 80)

    results = {}
    config = BacktestConfig(
        initial_capital=capital,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    # 配置组：不同止盈参数
    configs = [
        {
            'name': 'A. 原始(8%/15%)',
            'trailing_stop': 0.08,
            'activate': 0.15,
        },
        {
            'name': 'B. 延长(10%/20%)',
            'trailing_stop': 0.10,
            'activate': 0.20,
        },
        {
            'name': 'C. 延长(12%/25%)',
            'trailing_stop': 0.12,
            'activate': 0.25,
        },
        {
            'name': 'D. 激进(15%/30%)',
            'trailing_stop': 0.15,
            'activate': 0.30,
        },
    ]

    for cfg in configs:
        logger.info(f"\n测试: {cfg['name']}")

        strategy = ChanLunTradingSystem(
            name=cfg['name'],
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=False,
            min_confidence=0.60,
            enable_volume_confirm=True,
            min_volume_ratio=1.2,
            trailing_stop_pct=cfg['trailing_stop'],
            trailing_activate_pct=cfg['activate'],
        )

        engine = BacktestEngine(config)
        engine.add_data('sh600519', df)
        engine.set_strategy(strategy)
        results[cfg['name']] = engine.run()

    # 打印对比
    print_results(results)

    return results


def print_results(results: dict):
    print("\n" + "=" * 100)
    print("延长止盈参数对比报告".center(100))
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

    # 找最佳
    best_name = max(results.keys(), key=lambda k: results[k]['total_return'])
    best_result = results[best_name]

    print("=" * 100)
    print(f"【最佳配置】: {best_name}")
    print("=" * 100)
    print(f"收益率: {best_result['total_return']:.2%}")
    print(f"盈亏比: {best_result['profit_loss_ratio']:.2f}")
    print(f"胜率: {best_result['win_rate']:.2%}")
    print()


if __name__ == '__main__':
    run_test()
