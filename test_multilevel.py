"""多级别缠论策略测试"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)


def generate_data(days=300):
    """生成模拟数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    dates = [d for d in dates if d.weekday() < 5]

    # 生成带趋势和回调的价格走势
    price = 100.0
    prices = []
    trend = 0.0005

    for i in range(len(dates)):
        # 趋势 + 随机波动
        change = np.random.randn() * 0.02 + trend
        price = price * (1 + change)

        hl = price * 0.02 * np.random.rand()
        open_p = price + np.random.randn() * 0.01
        close_p = price + np.random.randn() * 0.01

        prices.append({
            'datetime': dates[i],
            'open': round(open_p, 2),
            'high': round(max(open_p, close_p) + hl, 2),
            'low': round(min(open_p, close_p) - hl, 2),
            'close': round(close_p, 2),
            'volume': np.random.randint(1000000, 50000000)
        })

    df = pd.DataFrame(prices)
    df.set_index('datetime', inplace=True)
    return df


def main():
    print("="*70)
    print("              多级别缠论策略回测")
    print("="*70)
    print("\n策略规则:")
    print("  1. 日线级别2买买入")
    print("  2. 跌破1买低点止损")
    print("  3. 30分钟MACD顶背离减仓50%")
    print("  4. 30分钟2卖卖出剩余")
    print("="*70)

    from strategies import MultiLevelChanLunStrategy
    from backtest import BacktestEngine, BacktestConfig

    # 生成数据
    print("\n[生成数据]")
    df = generate_data(300)
    symbol = "TEST001"
    print(f"  股票: {symbol}")
    print(f"  K线: {len(df)} 条")
    print(f"  区间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  价格: ¥{df['low'].min():.2f} ~ ¥{df['high'].max():.2f}")

    # 创建策略
    strategy = MultiLevelChanLunStrategy(
        daily_min_strokes=3,
        m30_min_strokes=5,
        stop_loss_pct=0.05,
        exit_ratio=0.5
    )

    # 配置回测
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100
    )

    # 运行回测
    print("\n[运行回测]")
    engine = BacktestEngine(config)
    engine.add_data(symbol, df)
    engine.set_strategy(strategy)

    results = engine.run()
    trades = engine.get_trades()
    signals = engine.get_signals()

    # 显示结果
    print("\n" + "="*70)
    print("                      回测结果")
    print("="*70)
    print(f"  初始资金:     ¥{config.initial_capital:,.2f}")
    print(f"  最终权益:     ¥{results['final_equity']:,.2f}")
    print(f"  总收益率:     {results['total_return']:.2%}")
    print(f"  年化收益:     {results['annual_return']:.2%}")
    print(f"  夏普比率:     {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤:     {results['max_drawdown']:.2%}")
    print(f"  胜率:         {results['win_rate']:.2%}")
    print(f"  盈亏比:       {results['profit_loss_ratio']:.2f}")
    print(f"  总交易:       {results['total_trades']} 次")
    print("="*70)

    # 交易明细
    if trades:
        print("\n[交易明细]")
        for i, trade in enumerate(trades, 1):
            side = "买入" if trade.signal_type.value == 'buy' else "卖出"
            print(f"  {i}. {trade.datetime.strftime('%Y-%m-%d')} "
                  f"{side} {trade.symbol} "
                  f"@ ¥{trade.price:.2f} x {trade.quantity} "
                  f"({trade.reason})")
    else:
        print("\n[交易明细] 无交易")

    # 信号统计
    print(f"\n[信号统计]")
    print(f"  总信号数: {len(signals)}")
    buy_signals = [s for s in signals if s.is_buy()]
    sell_signals = [s for s in signals if s.is_sell()]
    print(f"  买入信号: {len(buy_signals)}")
    print(f"  卖出信号: {len(sell_signals)}")

    # 保存结果
    with open('multilevel_strategy_report.txt', 'w', encoding='utf-8') as f:
        f.write("多级别缠论策略回测报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"初始资金: ¥{config.initial_capital:,.2f}\n")
        f.write(f"最终权益: ¥{results['final_equity']:,.2f}\n")
        f.write(f"总收益率: {results['total_return']:.2%}\n")
        f.write(f"年化收益: {results['annual_return']:.2%}\n")
        f.write(f"夏普比率: {results['sharpe_ratio']:.2f}\n")
        f.write(f"最大回撤: {results['max_drawdown']:.2%}\n")
        f.write(f"胜率: {results['win_rate']:.2%}\n")
        f.write(f"总交易: {results['total_trades']} 次\n")
        f.write("\n交易明细:\n")
        for i, trade in enumerate(trades, 1):
            side = "买入" if trade.signal_type.value == 'buy' else "卖出"
            f.write(f"{i}. {trade.datetime.strftime('%Y-%m-%d')} {side} @ ¥{trade.price:.2f} x {trade.quantity}\n")

    print("\n报告已保存到: multilevel_strategy_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出...")
