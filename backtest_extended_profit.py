"""
对比原始策略 vs 延长止盈策略
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.chanlun_trading_system import ChanLunTradingSystem
from strategies.chanlun_extended_profit import ExtendedProfitChanLun


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


def run_comparison(start_date: str = '2021-08-01', capital: float = 500000):
    df = load_tdx_json('sh600519')
    if df is None:
        return
    df = df[df.index >= start_date]

    logger.info("=" * 80)
    logger.info("延长止盈策略对比回测（贵州茅台）")
    logger.info("=" * 80)

    results = {}
    config = BacktestConfig(
        initial_capital=capital,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    # 1. 原始2买策略
    logger.info("\n测试: 原始2买策略")
    strategy1 = ChanLunTradingSystem(
        name='原始2买',
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=False,
        min_confidence=0.60,
        enable_volume_confirm=True,
        min_volume_ratio=1.2,
        trailing_stop_pct=0.08,      # 8%移动止损
        trailing_activate_pct=0.15,   # 15%启用
    )
    engine1 = BacktestEngine(config)
    engine1.add_data('sh600519', df)
    engine1.set_strategy(strategy1)
    results['原始策略'] = engine1.run()

    # 2. 延长止盈策略
    logger.info("\n测试: 延长止盈策略")
    strategy2 = ExtendedProfitChanLun(
        name='延长止盈',
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=False,
        min_confidence=0.60,
        enable_volume_confirm=True,
        min_volume_ratio=1.2,
        # 延长止盈参数
        target_profit_pct=0.25,       # 目标25%
        trailing_stop_pct=0.12,       # 12%移动止损
        trailing_activate_pct=0.20,   # 20%启用
        first_exit_pct=0.12,          # 12%第一次减仓
        first_exit_ratio=0.30,        # 减仓30%
        second_exit_pct=0.20,         # 20%第二次减仓
        second_exit_ratio=0.30,       # 减仓30%
    )
    engine2 = BacktestEngine(config)
    engine2.add_data('sh600519', df)
    engine2.set_strategy(strategy2)
    results['延长止盈'] = engine2.run()

    # 3. 更激进的延长止盈
    logger.info("\n测试: 激进延长止盈")
    strategy3 = ExtendedProfitChanLun(
        name='激进延长止盈',
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=False,
        min_confidence=0.60,
        enable_volume_confirm=True,
        min_volume_ratio=1.2,
        target_profit_pct=0.30,       # 目标30%
        trailing_stop_pct=0.15,       # 15%移动止损
        trailing_activate_pct=0.25,   # 25%启用
        first_exit_pct=0.15,          # 15%第一次减仓
        first_exit_ratio=0.25,        # 减仓25%
        second_exit_pct=0.25,         # 25%第二次减仓
        second_exit_ratio=0.25,       # 减仓25%
    )
    engine3 = BacktestEngine(config)
    engine3.add_data('sh600519', df)
    engine3.set_strategy(strategy3)
    results['激进延长'] = engine3.run()

    # 打印对比
    print_comparison(results)

    return results


def print_comparison(results: dict):
    print("\n" + "=" * 100)
    print("延长止盈策略对比报告".center(100))
    print("=" * 100 + "\n")

    print(f"{'策略':<15}{'收益率':<12}{'年化':<10}{'夏普':<8}{'回撤':<10}{'胜率':<10}{'盈亏比':<10}{'交易':<8}")
    print("-" * 100)

    for name, result in results.items():
        ret_str = f"[OK]{result['total_return']:>6.2%}" if result['total_return'] > 0 else f"[-]{result['total_return']:>6.2%}"
        print(f"{name:<15}{ret_str:<12}{result['annual_return']:>8.2%}  "
              f"{result['sharpe_ratio']:>6.2f}  {result['max_drawdown']:>8.2%}  "
              f"{result['win_rate']:>8.2%}  {result['profit_loss_ratio']:>8.2f}  "
              f"{result['total_trades']:>6}")

    print("-" * 100)
    print()

    # 详细分析最佳策略
    best_name = max(results.keys(), key=lambda k: results[k]['total_return'])
    best_result = results[best_name]

    print("=" * 100)
    print(f"【最佳策略】: {best_name}")
    print("=" * 100)
    print(f"总收益率: {best_result['total_return']:.2%}")
    print(f"年化收益: {best_result['annual_return']:.2%}")
    print(f"夏普比率: {best_result['sharpe_ratio']:.2f}")
    print(f"最大回撤: {best_result['max_drawdown']:.2%}")
    print(f"胜率: {best_result['win_rate']:.2%}")
    print(f"盈亏比: {best_result['profit_loss_ratio']:.2f}")
    print(f"交易次数: {best_result['total_trades']}")
    print()

    # 交易明细
    if best_result['trades']:
        print("交易明细:")
        print(f"  {'日期':<12}{'操作':<6}{'价格':<10}{'数量':<8}{'盈亏':<15}{'原因'}")
        print("  " + "-" * 80)

        entry_price = None
        entry_qty = None

        for t in best_result['trades']:
            is_buy = t.signal_type.value == 'buy'
            op = '买入' if is_buy else '卖出'

            profit_str = ''
            if is_buy:
                entry_price = t.price
                entry_qty = t.quantity
            elif entry_price:
                profit = (t.price - entry_price) * entry_qty
                profit_pct = (t.price - entry_price) / entry_price
                marker = ' ***' if profit > 0 else ''
                profit_str = f"{profit:+,.0f}({profit_pct:+.2%}){marker}"
                entry_price = None

            print(f"  {str(t.datetime)[:10]:<12}{op:<6}{t.price:<10.2f}{t.quantity:<8}{profit_str:<15}{t.reason}")

    print("=" * 100)


if __name__ == '__main__':
    run_comparison(start_date='2021-08-01', capital=500000)
