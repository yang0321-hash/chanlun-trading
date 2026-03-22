"""
缠论交易系统回测

运行缠论稳定盈利交易系统，生成详细报告
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import argparse

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.metrics import Metrics
from strategies.chanlun_trading_system import ChanLunTradingSystem
from data.akshare_source import AKShareSource
from datetime import datetime


def run_backtest(
    symbols: list,
    start_date: str = '2020-01-01',
    end_date: str = None,
    initial_capital: float = 100000,
    save_report: bool = True,
) -> dict:
    """
    运行回测

    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        save_report: 是否保存报告

    Returns:
        回测结果
    """
    logger.info(f"开始回测缠论交易系统")
    logger.info(f"股票池: {symbols}")
    logger.info(f"时间范围: {start_date} ~ {end_date or '至今'}")

    # 获取数据
    data_source = AKShareSource()

    data_map = {}
    for symbol in symbols:
        try:
            logger.info(f"获取 {symbol} 数据...")
            df = data_source.get_kline(
                symbol=symbol,
                start_date=datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date,
                end_date=datetime.strptime(end_date, '%Y-%m-%d') if end_date and isinstance(end_date, str) else (datetime.strptime(end_date, '%Y-%m-%d') if end_date else None),
                period='daily',
                adjust='qfq'
            )
            if df is not None and len(df) > 100:
                # 设置datetime为索引
                df = df.set_index('datetime')
                data_map[symbol] = df
                logger.info(f"  {symbol}: {len(df)} 条数据")
            else:
                logger.warning(f"  {symbol}: 数据不足，跳过")
        except Exception as e:
            logger.error(f"  {symbol}: 获取失败 - {e}")

    if not data_map:
        logger.error("没有获取到任何数据")
        return None

    logger.info(f"成功获取 {len(data_map)} 只股票的数据")

    # 创建策略
    strategy = ChanLunTradingSystem(
        name='缠论稳定盈利系统',
        max_risk_per_trade=0.02,     # 单笔最大风险2%
        max_drawdown_pct=0.15,       # 最大回撤15%
        enable_buy1=True,            # 交易1买
        enable_buy2=True,            # 交易2买
        enable_buy3=True,            # 交易3买
        min_confidence=0.6,          # 最低置信度60%
        stop_loss_atr_multiplier=2.0,# ATR止损倍数
        trailing_stop_pct=0.05,      # 移动止损5%
        trailing_activate_pct=0.10,  # 盈利10%后启用移动止损
    )

    # 创建回测引擎
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission=0.0003,   # 万三手续费
        slippage=0.0001,     # 万一滑点
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
    print_report(results, strategy)

    # 保存报告
    if save_report:
        save_backtest_report(results, symbols)

    return results


def print_report(results: dict, strategy: ChanLunTradingSystem) -> None:
    """打印回测报告"""
    print("\n" + "="*80)
    print("缠论稳定盈利交易系统 - 回测报告".center(80))
    print("="*80 + "\n")

    # 系统状态
    state = strategy.get_system_state()
    print("【系统状态】")
    print(f"  市场趋势: {state['market_trend']}")
    print(f"  趋势强度: {state['trend_strength']:.1%}")
    print(f"  当前持仓: {state['positions']} 个")
    print(f"  峰值权益: ¥{state['peak_equity']:,.2f}")
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

    # 交易明细
    if results['trades']:
        print("【最近交易】")
        trades = results['trades'][-10:] if len(results['trades']) > 10 else results['trades']

        print(f"  {'日期':<12}{'类型':<8}{'代码':<10}{'价格':<10}{'数量':<10}{'原因':<30}")
        print("  " + "-"*80)

        for t in reversed(trades):
            signal_type = '买入' if t.signal_type.value == 'buy' else '卖出'
            print(f"  {str(t.datetime)[:10]:<12}{signal_type:<8}{t.symbol:<10}"
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
        print()

    print("="*80)


def save_backtest_report(results: dict, symbols: list) -> None:
    """保存回测报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"backtest_report_chanlun_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("缠论稳定盈利交易系统 - 回测报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write(f"股票池: {', '.join(symbols)}\n\n")

        f.write("【绩效指标】\n")
        f.write(f"  初始资金: ¥{results['final_equity'] / (1 + results['total_return']):,.2f}\n")
        f.write(f"  最终资金: ¥{results['final_equity']:,.2f}\n")
        f.write(f"  总收益率: {results['total_return']:.2%}\n")
        f.write(f"  年化收益: {results['annual_return']:.2%}\n")
        f.write(f"  夏普比率: {results['sharpe_ratio']:.2f}\n")
        f.write(f"  最大回撤: {results['max_drawdown']:.2%}\n")
        f.write(f"  胜率: {results['win_rate']:.2%}\n")
        f.write(f"  盈亏比: {results['profit_loss_ratio']:.2f}\n")
        f.write(f"  交易次数: {results['total_trades']}\n\n")

        if results['trades']:
            f.write("【交易明细】\n")
            for t in results['trades']:
                signal_type = '买入' if t.signal_type.value == 'buy' else '卖出'
                f.write(f"  {t.datetime} {signal_type} {t.symbol} "
                       f"@{t.price} x{t.quantity} {t.reason}\n")

    logger.info(f"报告已保存至: {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='缠论交易系统回测')
    parser.add_argument('--symbols', nargs='+', default=['600519', '000001', '002600'],
                       help='股票代码列表')
    parser.add_argument('--start', default='2020-01-01', help='开始日期')
    parser.add_argument('--end', default=None, help='结束日期')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')

    args = parser.parse_args()

    run_backtest(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )


if __name__ == '__main__':
    main()
