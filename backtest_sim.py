"""使用模拟002600走势数据回测"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


def generate_002600_like_data(days=1000):
    """
    生成模拟002600走势数据
    基于现实走势特征：波动率适中，有趋势和回调
    """
    print("生成002600模拟数据...")

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    dates = [d for d in dates if d.weekday() < 5]

    # 模拟股价走势：趋势 + 波动 + 偶尔的大幅波动
    price = 6.0  # 002600历史上在6-20元区间
    drift = 0.0004  # 年化约10%趋势
    volatility = 0.025  # 日波动率2.5%

    prices = []

    # 先生成基础走势
    in_trend = True
    trend_start = 0

    for i in range(len(dates)):
        # 随机游走 + 趋势
        change = np.random.randn() * volatility + drift

        # 偶尔改变趋势方向
        if i % 100 == 50:
            in_trend = not in_trend
            drift = -0.0002 if not in_trend else 0.0004

        price = price * (1 + change)

        # 生成OHLC
        hl_range = price * volatility * np.random.rand()
        open_p = price + np.random.randn() * volatility * 0.5
        close_p = price + np.random.randn() * volatility * 0.5
        high_p = max(open_p, close_p) + hl_range
        low_p = min(open_p, close_p) - hl_range

        # 偶尔制造大幅波动
        if i % 50 == 25:
            # 回调
            if np.random.rand() > 0.5:
                close_p = close_p * (1 - np.random.rand() * 0.08)

        prices.append({
            'datetime': dates[i],
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': np.random.randint(50000000, 200000000)
        })

    df = pd.DataFrame(prices)
    df.set_index('datetime', inplace=True)
    return df


def run_backtest(df):
    """运行回测"""
    from strategies import WeeklyDailyChanLunStrategy
    from backtest import BacktestEngine, BacktestConfig

    symbol = "002600(模拟)"

    print("\n" + "="*60)
    print("     运行回测")
    print("="*60)

    # 配置回测
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100
    )

    strategy = WeeklyDailyChanLunStrategy(
        weekly_min_strokes=3,
        daily_min_strokes=3,
        stop_loss_pct=0.08,
        exit_ratio=0.5
    )

    engine = BacktestEngine(config)
    engine.add_data(symbol, df)
    engine.set_strategy(strategy)

    print("正在回测...")
    results = engine.run()
    trades = engine.get_trades()

    # 显示结果
    print("\n" + "="*60)
    print("                      回测结果")
    print("="*60)
    print(f"  数据来源: 模拟数据(参考002600历史走势特征)")
    print(f"  策略: 周线2买 + 日线MACD/2卖出")
    print(f"  初始资金: ¥{config.initial_capital:,.2f}")
    print(f"  最终权益: ¥{results['final_equity']:,.2f}")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  年化收益: {results['annual_return']:.2%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print(f"  盈亏比: {results['profit_loss_ratio']:.2f}")
    print(f"  总交易: {results['total_trades']} 次")
    print("="*60)

    # 交易明细
    if trades:
        print("\n[交易明细]")
        for i, towade in enumerate(trades, 1):
            side = "买入" if trade.signal_type.value == 'buy' else "卖出"
            print(f"  {i}. {trade.datetime.strftime('%Y-%m-%d')} "
                  f"{side} @ ¥{trade.price:.2f} x {trade.quantity}")
    else:
        print("\n[交易明细] 无交易")
        print("  提示: 可调整策略参数增加交易频率")

    # 缠论分析
    print(f"\n[周线分析]")
    print(f"  分型: {len(strategy._weekly_fractals)} 个")
    print(f"  笔: {len(strategy._weekly_strokes)} 笔")
    print(f"  中枢: {len(strategy._weekly_pivots)} 个")
    if strategy._weekly_first_buy_price:
        print(f"  1买位置: ¥{strategy._weekly_first_buy_price:.2f}")
    if strategy._weekly_second_buy_price:
        print(f"  2买位置: ¥{strategy._weekly_second_buy_price:.2f}")

    print(f"\n[日线分析]")
    print(f"  分型: {len(strategy._daily_fractals)} 个")
    print(f"  笔: {len(strategy._daily_strokes)} 笔")
    print(f"  MACD顶背离: {'有' if strategy._check_daily_macd_divergence() else '无'}")

    # 数据统计
    print(f"\n[数据统计]")
    print(f"  总K线: {len(df)} 条")
    print(f"  价格: ¥{df['low'].min():.2f} ~ ¥{df['high'].max():.2f}")

    # 保存报告
    with open('002600_sim_backtest.txt', 'w', encoding='utf-8') as f:
        f.write(f"002600 缠论策略回测报告(模拟数据)\n")
        f.write("="*60 + "\n\n")
        f.write(f"数据: 模拟数据，参考002600历史走势特征\n")
        f.write(f"周期: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"价格区间: ¥{df['low'].min():.2f} ~ ¥{df['high'].max():.2f}\n\n")
        f.write(f"初始资金: ¥{config.initial_capital:,.2f}\n")
        f.write(f"最终权益: ¥{results['final_equity']:,.2f}\n")
        f.write(f"总收益率: {results['total_return']:.2%}\n")
        f.write(f"年化收益: {results['annual_return']:.2%}\n")
        f.write(f"夏普比率: {results['sharpe_ratio']:.2f}\n")
        f.write(f"最大回撤: {results['max_drawdown']:.2%}\n")
        f.write(f"总交易: {results['total_trades']} 次\n")

    print("\n报告已保存到: 002600_sim_backtest.txt")


def main():
    # 生成数据
    df = generate_002600_like_data(1000)

    # 运行回测
    run_backtest(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出...")
