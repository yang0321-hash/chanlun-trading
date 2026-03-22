"""
测试专注于2买的严格量能配置

配置要点：
- 只做2买（不抄底1买，不做追涨3买）
- 高置信度要求（0.70）
- 严格量能确认（1.5倍以上）
- 启用量能背离加分
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
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df['amount'] = df['volume'] * df['close']

    logger.info(f"加载 {code}: {len(df)} 条数据 ({df.index[0]} ~ {df.index[-1]})")
    return df


def run_buy2_focus_test(start_date: str = '2021-08-01', capital: float = 500000):
    """测试2买专注配置"""

    df = load_tdx_json('sh600519')
    if df is None:
        return

    df = df[df.index >= start_date]
    if len(df) < 50:
        logger.error("数据不足")
        return

    logger.info("=" * 80)
    logger.info("2买专注策略 - 严格量能配置回测（贵州茅台）")
    logger.info("=" * 80)
    logger.info(f"回测期间: {df.index[0]} ~ {df.index[-1]}")
    logger.info(f"初始资金: CNY{capital:,.0f}")
    logger.info("")

    results = {}
    configs = [
        {
            'name': 'A. 基础2买（无量能过滤）',
            'volume_confirm': False,
            'volume_ratio': 1.0,
            'divergence': False,
            'confidence': 0.55,
        },
        {
            'name': 'B. 2买+1.2倍量能',
            'volume_confirm': True,
            'volume_ratio': 1.2,
            'divergence': False,
            'confidence': 0.60,
        },
        {
            'name': 'C. 2买+1.5倍量能',
            'volume_confirm': True,
            'volume_ratio': 1.5,
            'divergence': False,
            'confidence': 0.65,
        },
        {
            'name': 'D. 2买+1.8倍量能',
            'volume_confirm': True,
            'volume_ratio': 1.8,
            'divergence': False,
            'confidence': 0.70,
        },
        {
            'name': 'E. 2买+1.5倍量能+背离',
            'volume_confirm': True,
            'volume_ratio': 1.5,
            'divergence': True,
            'confidence': 0.60,
        },
        {
            'name': 'F. 2买+1.8倍量能+背离',
            'volume_confirm': True,
            'volume_ratio': 1.8,
            'divergence': True,
            'confidence': 0.65,
        },
    ]

    base_config = BacktestConfig(
        initial_capital=capital,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试: {config['name']}")
        logger.info(f"{'='*60}")

        strategy = ChanLunTradingSystem(
            name=config['name'],
            # 核心配置：只做2买
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=False,
            min_confidence=config['confidence'],
            # 止损参数
            trailing_stop_pct=0.08,
            trailing_activate_pct=0.15,
            # 量能参数
            enable_volume_confirm=config['volume_confirm'],
            min_volume_ratio=config['volume_ratio'],
            enable_volume_divergence=config['divergence'],
        )

        engine = BacktestEngine(base_config)
        engine.add_data('sh600519', df)
        engine.set_strategy(strategy)

        result = engine.run()
        results[config['name']] = (result, strategy)

    # 打印对比报告
    print_comparison_report(results)

    return results


def print_comparison_report(results: dict) -> None:
    """打印对比报告"""
    print("\n" + "=" * 110)
    print("2买专注策略对比报告 - 贵州茅台(sh600519)".center(110))
    print("=" * 110 + "\n")

    # 表头
    print(f"{'配置':<28}{'收益率':<12}{'年化':<10}{'夏普':<8}{'回撤':<10}{'胜率':<10}{'盈亏比':<10}{'交易':<8}")
    print("-" * 110)

    sorted_results = []

    for name, (result, strategy) in results.items():
        sorted_results.append((
            name,
            result['total_return'],
            result['annual_return'],
            result['sharpe_ratio'],
            result['max_drawdown'],
            result['win_rate'],
            result['profit_loss_ratio'],
            result['total_trades']
        ))

    # 按收益率排序
    sorted_results.sort(key=lambda x: x[1], reverse=True)

    for name, ret, ann, sharpe, dd, wr, pl, trades in sorted_results:
        # 收益率颜色标记
        if ret > 0.05:
            ret_str = f"[OK]{ret:>7.2%}"
        elif ret > 0:
            ret_str = f"[+]{ret:>7.2%}"
        else:
            ret_str = f"[-]{ret:>7.2%}"

        # 胜率标记
        wr_str = f"{wr:.1%}{'*' if wr > 0.5 else ''}"

        print(f"{name:<28}{ret_str:<12}{ann:>7.2%}  {sharpe:>6.2f}  {dd:>8.2%}  {wr_str:<10}{pl:>8.2f}  {trades:>6}")

    print("-" * 110)
    print()

    # 详细分析最佳配置
    best_name = sorted_results[0][0]
    best_result, best_strategy = results[best_name]

    print("=" * 110)
    print(f"【最佳配置】: {best_name}")
    print("=" * 110)

    state = best_strategy.get_system_state()
    print(f"\n系统状态:")
    print(f"  市场趋势: {state['market_trend']}")
    print(f"  趋势强度: {state['trend_strength']:.1%}")
    print(f"  峰值权益: CNY{state['peak_equity']:,.2f}")

    print(f"\n绩效指标:")
    print(f"  总收益率: {best_result['total_return']:.2%}")
    print(f"  年化收益: {best_result['annual_return']:.2%}")
    print(f"  夏普比率: {best_result['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {best_result['max_drawdown']:.2%}")
    print(f"  胜率: {best_result['win_rate']:.2%}")
    print(f"  盈亏比: {best_result['profit_loss_ratio']:.2f}")
    print(f"  交易次数: {best_result['total_trades']}")

    # 信号分析
    if best_result['signals']:
        buy_signals = [s for s in best_result['signals'] if s.is_buy()]
        sell_signals = [s for s in best_result['signals'] if s.is_sell()]

        print(f"\n信号统计:")
        print(f"  买入信号: {len(buy_signals)} 次")
        print(f"  卖出信号: {len(sell_signals)} 次")

        if buy_signals:
            confs = [s.confidence for s in buy_signals]
            print(f"  平均置信度: {np.mean(confs):.1%}")
            print(f"  最高置信度: {np.max(confs):.1%}")
            print(f"  最低置信度: {np.min(confs):.1%}")

        # 买点类型统计
        buy2_count = sum(1 for s in buy_signals if '2买' in s.reason)
        other_count = len(buy_signals) - buy2_count
        print(f"  买点分布: 2买={buy2_count}, 其他={other_count}")

    # 交易明细（前10笔和后10笔）
    if best_result['trades']:
        print(f"\n交易明细 (共{len(best_result['trades'])}笔):")
        print(f"  {'日期':<12}{'操作':<6}{'价格':<10}{'数量':<8}{'盈亏':<12}{'原因':<40}")
        print("  " + "-" * 100)

        entry_price = None
        entry_qty = None
        entry_date = None

        for i, t in enumerate(best_result['trades']):
            is_buy = t.signal_type.value == 'buy'
            op = '买入' if is_buy else '卖出'

            profit_str = ''
            if is_buy:
                entry_price = t.price
                entry_qty = t.quantity
                entry_date = t.datetime
            elif entry_price:
                profit = (t.price - entry_price) * entry_qty
                profit_pct = (t.price - entry_price) / entry_price
                marker = ' ***' if profit > 0 else ''
                profit_str = f"{profit:+,.0f}({profit_pct:+.2%}){marker}"
                entry_price = None

            date_str = str(t.datetime)[:10]
            print(f"  {date_str:<12}{op:<6}{t.price:<10.2f}{t.quantity:<8}{profit_str:<12}{t.reason:<38}")

    print()
    print("=" * 110)


if __name__ == '__main__':
    run_buy2_focus_test(start_date='2021-08-01', capital=500000)
