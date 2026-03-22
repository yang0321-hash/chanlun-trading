"""
缠论交易系统 - 贵州茅台专项回测（量能增强版）

对比原始版本与量能增强版本的性能差异。
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


def compare_strategies(df: pd.DataFrame, start_date: str = '2021-08-01', capital: float = 500000):
    """对比不同策略配置"""

    df = df[df.index >= start_date]
    if len(df) < 50:
        logger.error("数据不足")
        return

    logger.info("=" * 80)
    logger.info("缠论交易系统 - 策略对比回测（贵州茅台）")
    logger.info("=" * 80)
    logger.info(f"回测期间: {df.index[0]} ~ {df.index[-1]}")
    logger.info(f"股价范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    logger.info(f"初始资金: CNY{capital:,.0f}")
    logger.info("")

    results = {}

    # 1. 基础策略（无量能分析）
    logger.info("[" + "-" * 30 + "]")
    logger.info("测试1: 基础策略（无量能确认）")
    logger.info("[" + "-" * 30 + "]")

    strategy1 = ChanLunTradingSystem(
        name='基础缠论系统',
        max_risk_per_trade=0.02,
        max_drawdown_pct=0.20,
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=True,
        min_confidence=0.55,
        trailing_stop_pct=0.08,
        trailing_activate_pct=0.15,
        enable_volume_confirm=False,  # 关闭量能确认
        enable_volume_divergence=False,
    )

    config = BacktestConfig(
        initial_capital=capital,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    engine1 = BacktestEngine(config)
    engine1.add_data('sh600519', df)
    engine1.set_strategy(strategy1)

    results['基础策略'] = engine1.run()

    # 2. 量能确认策略
    logger.info("")
    logger.info("[" + "-" * 30 + "]")
    logger.info("测试2: 量能确认策略")
    logger.info("[" + "-" * 30 + "]")

    strategy2 = ChanLunTradingSystem(
        name='量能确认缠论系统',
        max_risk_per_trade=0.02,
        max_drawdown_pct=0.20,
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=True,
        min_confidence=0.55,
        trailing_stop_pct=0.08,
        trailing_activate_pct=0.15,
        enable_volume_confirm=True,   # 启用量能确认
        min_volume_ratio=1.2,
        enable_volume_divergence=True,
    )

    engine2 = BacktestEngine(config)
    engine2.add_data('sh600519', df)
    engine2.set_strategy(strategy2)

    results['量能确认'] = engine2.run()

    # 3. 严格量能策略
    logger.info("")
    logger.info("[" + "-" * 30 + "]")
    logger.info("测试3: 严格量能策略（更高量比要求）")
    logger.info("[" + "-" * 30 + "]")

    strategy3 = ChanLunTradingSystem(
        name='严格量能缠论系统',
        max_risk_per_trade=0.02,
        max_drawdown_pct=0.20,
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=True,
        min_confidence=0.60,  # 更高置信度
        trailing_stop_pct=0.08,
        trailing_activate_pct=0.15,
        enable_volume_confirm=True,
        min_volume_ratio=1.5,  # 更高量比
        enable_volume_divergence=True,
    )

    engine3 = BacktestEngine(config)
    engine3.add_data('sh600519', df)
    engine3.set_strategy(strategy3)

    results['严格量能'] = engine3.run()

    # 打印对比报告
    print_comparison_report(results, {
        '基础策略': strategy1,
        '量能确认': strategy2,
        '严格量能': strategy3,
    })

    return results


def print_comparison_report(results: dict, strategies: dict) -> None:
    """打印对比报告"""
    print("\n" + "=" * 100)
    print("贵州茅台(sh600519) - 缠论策略对比报告".center(100))
    print("=" * 100 + "\n")

    # 表头
    print(f"{'指标':<20}{'基础策略':<20}{'量能确认':<20}{'严格量能':<20}{'最佳':<15}")
    print("-" * 100)

    # 收益率对比
    returns = {k: v['total_return'] for k, v in results.items()}
    best_return = max(returns, key=returns.get)
    print(f"{'总收益率':<20}", end='')
    for name, result in results.items():
        marker = " <--" if name == best_return else ""
        print(f"{result['total_return']:>17.2%}      {marker:<5}", end='')
    print()

    # 年化收益
    annual = {k: v['annual_return'] for k, v in results.items()}
    best_annual = max(annual, key=annual.get)
    print(f"{'年化收益':<20}", end='')
    for name, result in results.items():
        marker = " <--" if name == best_annual else ""
        print(f"{result['annual_return']:>17.2%}      {marker:<5}", end='')
    print()

    # 夏普比率
    sharpes = {k: v['sharpe_ratio'] for k, v in results.items()}
    best_sharpe = max(sharpes, key=sharpes.get)
    print(f"{'夏普比率':<20}", end='')
    for name, result in results.items():
        marker = " <--" if name == best_sharpe else ""
        print(f"{result['sharpe_ratio']:>17.2f}      {marker:<5}", end='')
    print()

    # 最大回撤
    drawdowns = {k: v['max_drawdown'] for k, v in results.items()}
    best_drawdown = min(drawdowns, key=drawdowns.get)
    print(f"{'最大回撤':<20}", end='')
    for name, result in results.items():
        marker = " <--" if name == best_drawdown else ""
        print(f"{result['max_drawdown']:>17.2%}      {marker:<5}", end='')
    print()

    # 胜率
    win_rates = {k: v['win_rate'] for k, v in results.items()}
    best_wr = max(win_rates, key=win_rates.get)
    print(f"{'胜率':<20}", end='')
    for name, result in results.items():
        marker = " <--" if name == best_wr else ""
        print(f"{result['win_rate']:>17.2%}      {marker:<5}", end='')
    print()

    # 盈亏比
    pl_ratios = {k: v['profit_loss_ratio'] for k, v in results.items()}
    best_pl = max(pl_ratios, key=pl_ratios.get)
    print(f"{'盈亏比':<20}", end='')
    for name, result in results.items():
        marker = " <--" if name == best_pl else ""
        print(f"{result['profit_loss_ratio']:>17.2f}      {marker:<5}", end='')
    print()

    # 交易次数
    trades = {k: v['total_trades'] for k, v in results.items()}
    print(f"{'交易次数':<20}", end='')
    for name, result in results.items():
        print(f"{result['total_trades']:>17}      ", end='')
    print()

    print("-" * 100)
    print()

    # 详细交易对比
    for name, (result, strategy) in zip(results.keys(), zip(results.values(), strategies.values())):
        print(f"【{name}详细分析】")

        state = strategy.get_system_state()
        print(f"  系统状态: 趋势={state['market_trend']}, 强度={state['trend_strength']:.1%}")
        print(f"  信号统计: ", end='')

        if result['signals']:
            buy_s = [s for s in result['signals'] if s.is_buy()]
            sell_s = [s for s in result['signals'] if s.is_sell()]
            print(f"买入{len(buy_s)}次, 卖出{len(sell_s)}次")

            if buy_s:
                confs = [s.confidence for s in buy_s]
                print(f"  平均置信度: {np.mean(confs):.1%} (最高{np.max(confs):.1%}, 最低{np.min(confs):.1%})")
        else:
            print("无交易信号")

        # 买卖点类型统计
        if result['signals']:
            buy_signals = [s for s in result['signals'] if s.is_buy()]
            if buy_signals:
                buy2_count = sum(1 for s in buy_signals if '2买' in s.reason)
                buy3_count = sum(1 for s in buy_signals if '3买' in s.reason)
                other_count = len(buy_signals) - buy2_count - buy3_count
                print(f"  买点分布: 2买={buy2_count}, 3买={buy3_count}, 其他={other_count}")

        print()

    # 最终评估
    print("=" * 100)
    print("【策略评估】")

    for name, result in results.items():
        score = 0
        reasons = []

        # 收益评分
        if result['total_return'] > 0.10:
            score += 2
            reasons.append("收益优秀")
        elif result['total_return'] > 0:
            score += 1
            reasons.append("收益正")
        else:
            reasons.append("收益负")

        # 风险评分
        if result['max_drawdown'] < 0.15:
            score += 2
            reasons.append("回撤小")
        elif result['max_drawdown'] < 0.25:
            score += 1
            reasons.append("回撤中")
        else:
            reasons.append("回撤大")

        # 稳定性评分
        if result['sharpe_ratio'] > 1.0:
            score += 2
            reasons.append("夏普高")
        elif result['sharpe_ratio'] > 0.5:
            score += 1
            reasons.append("夏普中")

        # 盈亏比评分
        if result['profit_loss_ratio'] > 1.5:
            score += 2
            reasons.append("盈亏比优")
        elif result['profit_loss_ratio'] > 1.0:
            score += 1
            reasons.append("盈亏比良")

        status = "[OK]" if score >= 5 else "[X] " if score <= 2 else "[--]"
        print(f"  {status} {name}: 综合得分 {score}/8 ({', '.join(reasons)})")

    print("=" * 100)


if __name__ == '__main__':
    # 检查数据文件是否存在
    data_path = '.claude/temp/sh600519.day.json'
    if os.path.exists(data_path):
        df = load_tdx_json('sh600519')
        if df is not None:
            compare_strategies(df, start_date='2021-08-01', capital=500000)
    else:
        logger.error(f"未找到数据文件: {data_path}")
        logger.info("请先运行: node .claude/skills/tdx-parser/scripts/parse_tdx.js --code sh600519 --format json --output .claude/temp")
