"""
缠论策略综合对比测试

测试不同指标组合的效果，找出最佳配置。
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
from strategies.chanlun_enhanced_system import EnhancedChanLunSystem


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


def run_comprehensive_comparison(start_date: str = '2021-08-01', capital: float = 500000):
    """运行综合对比测试"""

    # 加载数据
    df = load_tdx_json('sh600519')
    if df is None:
        return

    df = df[df.index >= start_date]
    if len(df) < 50:
        logger.error("数据不足")
        return

    logger.info("=" * 100)
    logger.info("缠论策略 - 综合指标对比测试（贵州茅台）")
    logger.info("=" * 100)
    logger.info(f"回测期间: {df.index[0]} ~ {df.index[-1]}")
    logger.info(f"初始资金: CNY{capital:,.0f}")
    logger.info("")

    results = {}
    configs = []

    # 配置1: 基础策略（无任何过滤）
    configs.append({
        'name': '1.基础策略',
        'strategy': ChanLunTradingSystem(
            name='基础缠论',
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=True,
            min_confidence=0.55,
            enable_volume_confirm=False,
        ),
        'color': 'gray'
    })

    # 配置2: 量能确认
    configs.append({
        'name': '2.量能确认',
        'strategy': ChanLunTradingSystem(
            name='量能确认',
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=True,
            min_confidence=0.55,
            enable_volume_confirm=True,
            min_volume_ratio=1.2,
        ),
        'color': 'blue'
    })

    # 配置3: 量能+RSI
    configs.append({
        'name': '3.量能+RSI',
        'strategy': EnhancedChanLunSystem(
            name='量能+RSI',
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=True,
            min_confidence=0.55,
            enable_volume_confirm=True,
            min_volume_ratio=1.2,
            enable_rsi_filter=True,
            rsi_oversold=40,
            rsi_max=70,
        ),
        'color': 'green'
    })

    # 配置4: 量能+RSI+波动率
    configs.append({
        'name': '4.量能+RSI+波动率',
        'strategy': EnhancedChanLunSystem(
            name='量能+RSI+波动率',
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=True,
            min_confidence=0.55,
            enable_volume_confirm=True,
            min_volume_ratio=1.2,
            enable_rsi_filter=True,
            rsi_oversold=40,
            rsi_max=70,
            enable_volatility_filter=True,
            max_atr_pct=0.05,
        ),
        'color': 'orange'
    })

    # 配置5: 终极版（所有过滤）
    configs.append({
        'name': '5.终极增强版',
        'strategy': EnhancedChanLunSystem(
            name='终极增强版',
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=True,
            min_confidence=0.60,
            enable_volume_confirm=True,
            min_volume_ratio=1.5,
            enable_rsi_filter=True,
            rsi_oversold=40,
            rsi_max=70,
            enable_volatility_filter=True,
            max_atr_pct=0.05,
            enable_regime_filter=True,
            ranging_position_scale=0.5,
        ),
        'color': 'red'
    })

    # 配置6: 保守版（高置信度）
    configs.append({
        'name': '6.保守高胜率版',
        'strategy': EnhancedChanLunSystem(
            name='保守版',
            enable_buy1=False,
            enable_buy2=True,
            enable_buy3=False,  # 只做2买
            min_confidence=0.70,
            enable_volume_confirm=True,
            min_volume_ratio=1.8,  # 更高量比
            enable_rsi_filter=True,
            rsi_oversold=35,  # 更深超卖
            rsi_max=65,
            enable_volatility_filter=True,
            max_atr_pct=0.04,
            enable_regime_filter=True,
            ranging_position_scale=0.3,
        ),
        'color': 'purple'
    })

    # 运行所有配置
    backtest_config = BacktestConfig(
        initial_capital=capital,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    for config in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试: {config['name']}")
        logger.info(f"{'='*50}")

        engine = BacktestEngine(backtest_config)
        engine.add_data('sh600519', df)
        engine.set_strategy(config['strategy'])

        result = engine.run()
        results[config['name']] = result

    # 打印综合报告
    print_comprehensive_report(results, configs)

    return results


def print_comprehensive_report(results: dict, configs: list) -> None:
    """打印综合报告"""
    print("\n" + "=" * 140)
    print("缠论策略综合对比报告 - 贵州茅台(sh600519)".center(140))
    print("=" * 140 + "\n")

    # 表头
    print(f"{'指标':<16}", end='')
    for config in configs:
        name = config['name'].split('.')[1] if '.' in config['name'] else config['name']
        print(f"{name:<20}", end='')
    print()
    print("-" * 140)

    # 收益率
    print(f"{'总收益率':<16}", end='')
    returns = {k: v['total_return'] for k, v in results.items()}
    best = max(returns, key=returns.get)
    for name in results.keys():
        marker = " <--" if name == best else ""
        print(f"{results[name]['total_return']:>18.2%}      {marker:<5}", end='')
    print()

    # 年化收益
    print(f"{'年化收益':<16}", end='')
    annual = {k: v['annual_return'] for k, v in results.items()}
    best = max(annual, key=annual.get)
    for name in results.keys():
        marker = " <--" if name == best else ""
        print(f"{results[name]['annual_return']:>18.2%}      {marker:<5}", end='')
    print()

    # 夏普比率
    print(f"{'夏普比率':<16}", end='')
    sharpes = {k: v['sharpe_ratio'] for k, v in results.items()}
    best = max(sharpes, key=sharpes.get)
    for name in results.keys():
        marker = " <--" if name == best else ""
        print(f"{results[name]['sharpe_ratio']:>18.2f}      {marker:<5}", end='')
    print()

    # 最大回撤
    print(f"{'最大回撤':<16}", end='')
    drawdowns = {k: v['max_drawdown'] for k, v in results.items()}
    best = min(drawdowns, key=drawdowns.get)
    for name in results.keys():
        marker = " <--" if name == best else ""
        print(f"{results[name]['max_drawdown']:>18.2%}      {marker:<5}", end='')
    print()

    # 胜率
    print(f"{'胜率':<16}", end='')
    win_rates = {k: v['win_rate'] for k, v in results.items()}
    best = max(win_rates, key=win_rates.get)
    for name in results.keys():
        marker = " <--" if name == best else ""
        print(f"{results[name]['win_rate']:>18.2%}      {marker:<5}", end='')
    print()

    # 盈亏比
    print(f"{'盈亏比':<16}", end='')
    pl_ratios = {k: v['profit_loss_ratio'] for k, v in results.items()}
    best = max(pl_ratios, key=pl_ratios.get)
    for name in results.keys():
        marker = " <--" if name == best else ""
        print(f"{results[name]['profit_loss_ratio']:>18.2f}      {marker:<5}", end='')
    print()

    # 交易次数
    print(f"{'交易次数':<16}", end='')
    for name in results.keys():
        print(f"{results[name]['total_trades']:>18}      ", end='')
    print()

    print("-" * 140)
    print()

    # 综合评分
    print("=" * 140)
    print("【综合评分排名】")
    print("=" * 140)

    scores = {}
    for name, result in results.items():
        score = 0
        details = []

        # 收益评分 (0-3分)
        if result['total_return'] > 0.15:
            score += 3
            details.append("收益++")
        elif result['total_return'] > 0.05:
            score += 2
            details.append("收益+")
        elif result['total_return'] > 0:
            score += 1
            details.append("收益正")

        # 风险评分 (0-2分)
        if result['max_drawdown'] < 0.15:
            score += 2
            details.append("回撤小")
        elif result['max_drawdown'] < 0.25:
            score += 1
            details.append("回撤中")

        # 效率评分 (0-2分)
        if result['sharpe_ratio'] > 1.0:
            score += 2
            details.append("夏普高")
        elif result['sharpe_ratio'] > 0:
            score += 1
            details.append("夏普正")

        # 稳定性评分 (0-2分)
        if result['win_rate'] > 0.55:
            score += 2
            details.append("胜率高")
        elif result['win_rate'] > 0.45:
            score += 1
            details.append("胜率中")

        # 盈亏比评分 (0-1分)
        if result['profit_loss_ratio'] > 1.5:
            score += 1
            details.append("盈亏比优")

        scores[name] = (score, details)

    # 按分数排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    print(f"{'排名':<6}{'策略':<20}{'得分':<6}{'评价'}")
    print("-" * 80)

    for rank, (name, (score, details)) in enumerate(sorted_scores, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<4}{name:<20}{score}/10  {', '.join(details)}")

    print()
    print("=" * 140)

    # 推荐
    best_name, (best_score, _) = sorted_scores[0]
    print(f"\n【推荐策略】: {best_name}")
    print(f"综合得分: {best_score}/10")

    best_result = results[best_name]
    print(f"预期收益: {best_result['total_return']:.2%}")
    print(f"胜率: {best_result['win_rate']:.2%}")
    print(f"最大回撤: {best_result['max_drawdown']:.2%}")
    print()


if __name__ == '__main__':
    run_comprehensive_comparison(start_date='2021-08-01', capital=500000)
