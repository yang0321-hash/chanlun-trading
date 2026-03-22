"""
缠论交易系统 - 贵州茅台专项回测
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


def run_moutai_backtest(start_date: str = '2021-08-01', capital: float = 500000):
    """回测贵州茅台"""

    logger.info("="*70)
    logger.info("缠论交易系统 - 贵州茅台(sh600519)回测")
    logger.info("="*70)

    # 加载数据
    df = load_tdx_json('sh600519')
    if df is None:
        return None

    # 筛选日期
    df = df[df.index >= start_date]
    if len(df) < 50:
        logger.error("数据不足")
        return None

    logger.info(f"回测期间: {df.index[0]} ~ {df.index[-1]}")
    logger.info(f"股价范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    logger.info(f"初始资金: ¥{capital:,.0f}")

    # 创建策略 - 使用优化参数适合茅台
    strategy = ChanLunTradingSystem(
        name='缠论系统-茅台',
        max_risk_per_trade=0.02,
        max_drawdown_pct=0.20,        # 茅台波动大，允许更大回撤
        enable_buy1=False,             # 关闭1买，更安全
        enable_buy2=True,
        enable_buy3=True,
        min_confidence=0.55,
        trailing_stop_pct=0.08,        # 8%移动止损
        trailing_activate_pct=0.15,    # 盈利15%后启用
    )

    # 创建回测引擎
    config = BacktestConfig(
        initial_capital=capital,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    engine = BacktestEngine(config)
    engine.add_data('sh600519', df)
    engine.set_strategy(strategy)

    # 运行回测
    logger.info("\n开始回测...")
    results = engine.run()

    # 打印报告
    print_moutai_report(results, strategy)

    return results


def print_moutai_report(results: dict, strategy) -> None:
    """打印茅台回测报告"""
    print("\n" + "="*80)
    print("贵州茅台(sh600519) - 缠论交易系统回测报告".center(80))
    print("="*80 + "\n")

    state = strategy.get_system_state()

    print("【市场环境】")
    print(f"  检测趋势: {state['market_trend']}")
    print(f"  趋势强度: {state['trend_strength']:.1%}")
    print()

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

    if results['trades']:
        print("【交易明细】")
        print(f"  {'日期':<12}{'操作':<6}{'价格':<10}{'数量':<8}{'盈亏':<12}{'原因':<35}")
        print("  " + "-"*90)

        entry_price = None
        entry_qty = None
        entry_date = None

        for t in results['trades']:
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
                profit_str = f"{profit:+,.0f} ({profit_pct:+.2%})"
                entry_price = None

            print(f"  {str(t.datetime)[:10]:<12}{op:<6}{t.price:<10.2f}{t.quantity:<8}{profit_str:<12}{t.reason:<33}")
        print()

    if results['signals']:
        buy_s = [s for s in results['signals'] if s.is_buy()]
        sell_s = [s for s in results['signals'] if s.is_sell()]
        print("【信号统计】")
        print(f"  买入信号: {len(buy_s)} 次")
        print(f"  卖出信号: {len(sell_s)} 次")
        if buy_s:
            confs = [s.confidence for s in buy_s]
            print(f"  平均置信度: {np.mean(confs):.1%}")
        print()

    if 'equity_curve' in results and not results['equity_curve'].empty:
        equity = results['equity_curve']
        print("【权益分析】")
        print(f"  起始: CNY{equity.iloc[0]['equity']:,.2f}")
        print(f"  结束: CNY{equity.iloc[-1]['equity']:,.2f}")
        print(f"  峰值: CNY{equity['equity'].max():,.2f}")
        print(f"  谷值: CNY{equity['equity'].min():,.2f}")
        print()

    # 评估
    print("【策略评估】")
    if results['total_return'] > 0:
        print("  [OK] 策略盈利")
    else:
        print("  [X] 策略亏损")

    if results['max_drawdown'] < 0.20:
        print("  [OK] 回撤控制良好")
    else:
        print("  [X] 回撤过大")

    if results['win_rate'] > 0.40:
        print("  [OK] 胜率可接受")
    else:
        print("  [X] 胜率偏低")

    print("="*80)


if __name__ == '__main__':
    # 茅台股价高，使用50万资金
    run_moutai_backtest(capital=500000)
