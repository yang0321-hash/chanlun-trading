"""
使用模拟数据测试缠论交易系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.chanlun_trading_system import ChanLunTradingSystem


def generate_mock_data(
    days: int = 500,
    start_price: float = 100,
    trend: str = 'up',
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    生成模拟K线数据

    Args:
        days: 天数
        start_price: 起始价格
        trend: 趋势方向 ('up', 'down', 'range')
        volatility: 波动率
    """
    np.random.seed(42)

    dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # 只保留工作日

    n = len(dates)
    prices = [start_price]

    # 生成价格序列
    for i in range(1, n):
        trend_bias = 0.0005 if trend == 'up' else (-0.0005 if trend == 'down' else 0)

        # 添加一些周期性波动（模拟缠论结构）
        cycle = np.sin(i / 20) * 0.01

        change = np.random.normal(trend_bias + cycle, volatility / np.sqrt(252))
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # 确保价格为正

    # 生成OHLC
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, volatility / 4)))
        low = close * (1 - abs(np.random.normal(0, volatility / 4)))
        open_price = low + (high - low) * np.random.random()

        # 确保OHLC关系正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = int(np.random.normal(1000000, 200000))

        data.append({
            'datetime': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': max(volume, 100000)
        })

    df = pd.DataFrame(data)
    df = df.set_index('datetime')
    return df


def test_with_mock_data():
    """使用模拟数据测试"""

    logger.info("="*60)
    logger.info("缠论交易系统 - 模拟数据测试")
    logger.info("="*60)

    # 生成不同趋势的模拟数据
    scenarios = [
        ('up_trend', 'up', 100, 0.025),
        ('range', 'range', 100, 0.03),
        ('volatile', 'up', 100, 0.04),
    ]

    all_results = {}

    for name, trend, start_price, volatility in scenarios:
        logger.info(f"\n测试场景: {name} (trend={trend}, volatility={volatility})")

        # 生成数据
        df = generate_mock_data(days=600, start_price=start_price, trend=trend, volatility=volatility)
        logger.info(f"生成数据: {len(df)} 条")

        # 创建策略
        strategy = ChanLunTradingSystem(
            name=f'缠论系统_{name}',
            max_risk_per_trade=0.02,
            max_drawdown_pct=0.20,
            enable_buy1=True,
            enable_buy2=True,
            enable_buy3=True,
            min_confidence=0.4,  # 降低阈值以看到更多信号
            trailing_activate_pct=0.05,
        )

        # 创建回测引擎
        config = BacktestConfig(
            initial_capital=100000,
            commission=0.0003,
            slippage=0.0001,
            min_unit=100,
            position_limit=0.95,
        )

        engine = BacktestEngine(config)
        engine.add_data(name, df)
        engine.set_strategy(strategy)

        # 运行回测
        results = engine.run()
        all_results[name] = (results, strategy)

        # 输出简要结果
        print(f"\n【{name}结果】")
        print(f"  总收益率: {results['total_return']:.2%}")
        print(f"  交易次数: {results['total_trades']}")
        print(f"  胜率: {results['win_rate']:.2%}")
        print(f"  最大回撤: {results['max_drawdown']:.2%}")
        print(f"  夏普比率: {results['sharpe_ratio']:.2f}")

    # 汇总比较
    print("\n" + "="*80)
    print("场景对比".center(80))
    print("="*80)
    print(f"{'场景':<15}{'收益率':<12}{'交易次数':<10}{'胜率':<10}{'最大回撤':<12}{'夏普比率':<10}")
    print("-"*80)

    for name, (results, _) in all_results.items():
        print(f"{name:<15}{results['total_return']:>10.2%}  "
              f"{results['total_trades']:>8}  {results['win_rate']:>8.2%}  "
              f"{results['max_drawdown']:>10.2%}  {results['sharpe_ratio']:>8.2f}")

    print("="*80)

    # 详细报告（最后一个场景）
    last_name, (results, strategy) = list(all_results.items())[-1]
    print_detailed_report(results, strategy, last_name)


def print_detailed_report(results: dict, strategy: ChanLunTradingSystem, scenario_name: str) -> None:
    """打印详细报告"""
    print(f"\n{'='*80}")
    print(f"{scenario_name} - 详细报告".center(80))
    print(f"{'='*80}\n")

    # 系统状态
    state = strategy.get_system_state()
    print("【系统状态】")
    print(f"  市场趋势: {state['market_trend']}")
    print(f"  趋势强度: {state['trend_strength']:.1%}")
    print(f"  峰值权益: CNY{state['peak_equity']:,.2f}")
    print()

    # 绩效指标
    print("【绩效指标】")
    print(f"  初始资金: CNY100,000.00")
    print(f"  最终资金: CNY{results['final_equity']:,.2f}")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  年化收益: {results['annual_return']:.2%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print(f"  盈亏比: {results['profit_loss_ratio']:.2f}")
    print(f"  交易次数: {results['total_trades']}")
    print()

    # 交易明细
    if results['trades']:
        print("【交易明细】")
        print(f"  {'日期':<12}{'类型':<8}{'代码':<12}{'价格':<10}{'数量':<10}{'原因':<35}")
        print("  " + "-"*90)

        for t in results['trades']:
            signal_type = '买入' if t.signal_type.value == 'buy' else '卖出'
            print(f"  {str(t.datetime)[:10]:<12}{signal_type:<8}{t.symbol:<12}"
                  f"{t.price:<10.2f}{t.quantity:<10}{t.reason:<33}")
        print()

    # 信号统计
    if results['signals']:
        buy_signals = [s for s in results['signals'] if s.is_buy()]
        sell_signals = [s for s in results['signals'] if s.is_sell()]

        print("【信号统计】")
        print(f"  买入信号: {len(buy_signals)} 次")
        print(f"  卖出信号: {len(sell_signals)} 次")

        if buy_signals:
            confs = [s.confidence for s in buy_signals]
            print(f"  信号置信度: 平均{np.mean(confs):.1%}, 最高{np.max(confs):.1%}")

            # 按类型统计
            buy1 = [s for s in buy_signals if '1买' in s.reason]
            buy2 = [s for s in buy_signals if '2买' in s.reason]
            buy3 = [s for s in buy_signals if '3买' in s.reason]
            print(f"  1买: {len(buy1)} 次, 2买: {len(buy2)} 次, 3买: {len(buy3)} 次")

            # 显示所有买入信号
            print("\n  【买入信号详情】")
            for s in buy_signals:
                print(f"    {s.datetime} {s.symbol} @{s.price:.2f} - {s.reason} (置信度:{s.confidence:.1%})")
        print()

    print("="*80)


if __name__ == '__main__':
    test_with_mock_data()
