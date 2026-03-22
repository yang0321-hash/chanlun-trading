"""
使用本地通达信数据测试缠论交易系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.chanlun_trading_system import ChanLunTradingSystem


def load_tdx_data(code: str, tdx_path: str = None) -> pd.DataFrame:
    """
    加载通达信数据

    Args:
        code: 股票代码（如 sh600519）
        tdx_path: 通达信数据路径
    """
    import json

    if tdx_path is None:
        tdx_path = "D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc"

    # 先尝试从解析好的JSON文件加载
    json_path = f".claude/temp/{code}.day.json"

    if os.path.exists(json_path):
        logger.info(f"从JSON加载 {code}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime')
        return df[['open', 'high', 'low', 'close', 'volume']]

    # 如果JSON不存在，返回空
    logger.warning(f"未找到 {code} 的JSON数据")
    return None


def test_with_local_data():
    """使用本地数据测试"""

    # 测试股票代码
    symbols = ['sh600519', 'sh600000', 'sz000001']

    logger.info(f"测试缠论交易系统")
    logger.info(f"股票池: {symbols}")

    # 加载数据
    data_map = {}
    for symbol in symbols:
        df = load_tdx_data(symbol)
        if df is not None and len(df) > 100:
            # 筛选2022年后的数据
            df = df[df.index >= '2022-01-01']
            if len(df) > 50:
                data_map[symbol] = df
                logger.info(f"  {symbol}: {len(df)} 条数据")

    if not data_map:
        logger.error("没有可用数据，请先运行 tdx-parser 解析数据")
        return

    # 创建策略
    strategy = ChanLunTradingSystem(
        name='缠论稳定盈利系统',
        max_risk_per_trade=0.02,
        max_drawdown_pct=0.15,
        enable_buy1=False,    # 关闭1买，更稳健
        enable_buy2=True,
        enable_buy3=True,
        min_confidence=0.5,   # 降低阈值以便看到信号
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
    logger.info("\n" + "="*60)
    logger.info("开始回测...")
    logger.info("="*60 + "\n")

    results = engine.run()

    # 输出报告
    print_test_report(results, strategy)


def print_test_report(results: dict, strategy: ChanLunTradingSystem) -> None:
    """打印测试报告"""
    print("\n" + "="*80)
    print("缠论稳定盈利交易系统 - 回测报告".center(80))
    print("="*80 + "\n")

    # 系统状态
    state = strategy.get_system_state()
    print("【系统状态】")
    print(f"  市场趋势: {state['market_trend']}")
    print(f"  趋势强度: {state['trend_strength']:.1%}")
    print(f"  当前持仓: {state['positions']} 个")
    print()

    # 绩效指标
    print("【绩效指标】")
    print(f"  初始资金: ¥{results['final_equity'] / (1 + results['total_return']):,.2f}")
    print(f"  最终资金: ¥{results['final_equity']:,.2f}")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  年化收益: {results['annual_return']:.2%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print(f"  盈亏比: {results['profit_loss_ratio']:.2f}")
    print(f"  交易次数: {results['total_trades']}")
    print()

    # 权益曲线
    if 'equity_curve' in results and not results['equity_curve'].empty:
        print("【权益曲线】(最后10个交易日)")
        equity = results['equity_curve'].tail(10)
        for date, value in equity.itertuples():
            print(f"  {date}: ¥{value:,.2f}")
        print()

    # 交易明细
    if results['trades']:
        print("【交易明细】")
        print(f"  {'日期':<12}{'类型':<8}{'代码':<12}{'价格':<10}{'数量':<10}{'原因':<30}")
        print("  " + "-"*80)

        for t in results['trades']:
            signal_type = '买入' if t.signal_type.value == 'buy' else '卖出'
            print(f"  {str(t.datetime)[:10]:<12}{signal_type:<8}{t.symbol:<12}"
                  f"{t.price:<10.2f}{t.quantity:<10}{t.reason:<28}")
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

    print("="*80)


if __name__ == '__main__':
    test_with_local_data()
