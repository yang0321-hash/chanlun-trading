"""
使用通达信本地数据回测缠论交易系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json
import numpy as np
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.chanlun_trading_system import ChanLunTradingSystem


def load_tdx_json(code: str, json_dir: str = '.claude/temp') -> pd.DataFrame:
    """加载通达信JSON数据"""
    json_path = f"{json_dir}/{code}.day.json"

    if not os.path.exists(json_path):
        logger.error(f"未找到数据文件: {json_path}")
        logger.info(f"请先运行: node .claude/skills/tdx-parser/scripts/parse_tdx.js --code {code} --format json --output .claude/temp")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # JSON是直接数组格式
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')

    # 确保列名正确
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # 添加amount列
    df['amount'] = df['volume'] * df['close']

    logger.info(f"加载 {code}: {len(df)} 条数据 ({df.index[0]} ~ {df.index[-1]})")

    return df


def run_backtest_with_tdx(symbols: list, start_date: str = None):
    """使用通达信数据回测"""

    logger.info("="*60)
    logger.info("缠论交易系统 - 通达信数据回测")
    logger.info("="*60)

    # 加载数据
    data_map = {}
    for symbol in symbols:
        df = load_tdx_json(symbol)
        if df is not None:
            # 筛选日期
            if start_date:
                df = df[df.index >= start_date]
            if len(df) > 50:
                data_map[symbol] = df

    if not data_map:
        logger.error("没有可用数据")
        return

    logger.info(f"成功加载 {len(data_map)} 只股票")

    # 创建策略
    strategy = ChanLunTradingSystem(
        name='缠论稳定盈利系统',
        max_risk_per_trade=0.02,
        max_drawdown_pct=0.15,
        enable_buy1=True,     # 启用1买
        enable_buy2=True,
        enable_buy3=True,
        min_confidence=0.45,  # 降低阈值以产生更多交易信号用于测试
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

    # 添加数据
    for symbol, df in data_map.items():
        engine.add_data(symbol, df)

    # 设置策略
    engine.set_strategy(strategy)

    # 运行回测
    logger.info("\n开始回测...")
    results = engine.run()

    # 输出报告
    print_report(results, strategy, symbols)

    return results


def print_report(results: dict, strategy, symbols: list) -> None:
    """打印报告"""
    print("\n" + "="*80)
    print(f"缠论交易系统回测报告 - {', '.join(symbols)}".center(80))
    print("="*80 + "\n")

    # 系统状态
    state = strategy.get_system_state()
    print("【系统状态】")
    print(f"  市场趋势: {state['market_trend']}")
    print(f"  趋势强度: {state['trend_strength']:.1%}")
    print()

    # 绩效指标
    print("【绩效指标】")
    initial = results['final_equity'] / (1 + results['total_return'])
    print(f"  初始资金: CNY{initial:,.2f}")
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
        print(f"  {'日期':<12}{'类型':<8}{'代码':<12}{'价格':<10}{'数量':<8}{'盈亏':<12}{'原因':<30}")
        print("  " + "-"*100)

        entry_price = None
        entry_qty = None
        entry_date = None

        for t in results['trades']:
            signal_type = '买入' if t.signal_type.value == 'buy' else '卖出'

            # 计算盈亏
            profit_str = ''
            if signal_type == '买入':
                entry_price = t.price
                entry_qty = t.quantity
                entry_date = t.datetime
            elif entry_price is not None:
                profit = (t.price - entry_price) * entry_qty
                profit_pct = (t.price - entry_price) / entry_price
                profit_str = f"{profit:+,.0f} ({profit_pct:+.2%})"
                entry_price = None

            print(f"  {str(t.datetime)[:10]:<12}{signal_type:<8}{t.symbol:<12}"
                  f"{t.price:<10.2f}{t.quantity:<8}{profit_str:<12}{t.reason:<28}")
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
        print()

    # 权益曲线摘要
    if 'equity_curve' in results and not results['equity_curve'].empty:
        equity = results['equity_curve']
        print("【权益曲线摘要】")
        print(f"  起始: {equity.index[0]} -> CNY{equity.iloc[0]['equity']:,.2f}")
        print(f"  结束: {equity.index[-1]} -> CNY{equity.iloc[-1]['equity']:,.2f}")
        print(f"  峰值: CNY{equity['equity'].max():,.2f}")
        print(f"  谷值: CNY{equity['equity'].min():,.2f}")
        print()

    print("="*80)


if __name__ == '__main__':
    import sys

    # 支持命令行参数
    symbols = ['sh600519']  # 默认贵州茅台
    start = '2020-01-01'

    if len(sys.argv) > 1:
        symbols = sys.argv[1].split(',')

    if len(sys.argv) > 2:
        start = sys.argv[2]

    run_backtest_with_tdx(symbols, start)
