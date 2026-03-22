"""多级别缠论策略回测 - 使用模拟AAPL数据"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


def generate_aapl_like_data(days=365):
    """
    生成模拟AAPL走势的数据
    基于AAPL的历史特征：波动率适中，长期向上趋势
    """
    print("生成AAPL模拟数据...")

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    dates = [d for d in dates if d.weekday() < 5]

    # AAPL约$180左右，年化波动率约25%
    price = 170.0
    daily_drift = 0.0003  # 年化约7.5%的趋势
    daily_vol = 0.015     # 日波动率约1.5%

    prices = []
    for i in range(len(dates)):
        # 几何布朗运动
        change = np.random.randn() * daily_vol + daily_drift
        price = price * (1 + change)

        # 生成OHLC
        hl_range = price * daily_vol * np.random.rand()
        open_p = price + np.random.randn() * daily_vol * 0.5
        close_p = price + np.random.randn() * daily_vol * 0.5
        high_p = max(open_p, close_p) + hl_range
        low_p = min(open_p, close_p) - hl_range

        # 添加一些典型的缠论形态
        if i % 50 == 30:  # 每50天左右制造一个回调
            close_p = close_p * (1 - np.random.rand() * 0.05)

        prices.append({
            'datetime': dates[i],
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': np.random.randint(50000000, 150000000)  # AAPL典型成交量
        })

    df = pd.DataFrame(prices)
    df.set_index('datetime', inplace=True)
    return df


def try_get_real_data():
    """尝试获取真实数据"""
    try:
        from data import YFinanceSource
        print("尝试获取AAPL真实数据...")

        source = YFinanceSource()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        df = source.get_kline('AAPL', start_date=start_date, end_date=end_date)
        if not df.empty:
            print(f"成功获取 {len(df)} 条真实数据")
            return df, True
    except Exception as e:
        print(f"获取真实数据失败: {e}")

    return None, False


def main():
    print("="*70)
    print("         AAPL 多级别缠论策略回测")
    print("="*70)
    print("\n策略规则:")
    print("  1. 日线级别2买买入")
    print("  2. 跌破1买低点止损")
    print("  3. 30分钟MACD顶背离减仓50%")
    print("  4. 30分钟2卖卖出剩余")
    print("="*70)

    from strategies import MultiLevelChanLunStrategy
    from backtest import BacktestEngine, BacktestConfig

    # 尝试获取真实数据，失败则使用模拟数据
    df, is_real = try_get_real_data()

    if df is None:
        df = generate_aapl_like_data(365)
        is_real = False

    symbol = 'AAPL'
    data_type = "真实" if is_real else "模拟"

    # 确保索引是datetime类型
    if not isinstance(df.index[0], pd.Timestamp):
        df.index = pd.to_datetime(df.index, errors='coerce')

    print(f"\n[数据信息]")
    print(f"  数据类型: {data_type}数据")
    print(f"  K线: {len(df)} 条")
    print(f"  时间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  价格: ${df['low'].min():.2f} ~ ${df['high'].max():.2f}")
    print(f"  最新: ${df['close'].iloc[-1]:.2f}")

    # 创建策略
    strategy = MultiLevelChanLunStrategy(
        daily_min_strokes=3,
        m30_min_strokes=5,
        stop_loss_pct=0.05,
        exit_ratio=0.5
    )

    # 配置回测
    config = BacktestConfig(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0001,
        min_unit=1
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
    print(f"  初始资金:     ${config.initial_capital:,.2f}")
    print(f"  最终权益:     ${results['final_equity']:,.2f}")
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
                  f"@ ${trade.price:.2f} x {trade.quantity} "
                  f"({trade.reason})")
    else:
        print("\n[交易明细] 无交易")
        print("  提示: 策略可能需要更长时间或更宽松的参数")

    # 缠论分析
    print(f"\n[缠论要素分析]")
    print(f"  分型数: {len(strategy._daily_fractals)}")
    print(f"  笔数: {len(strategy._daily_strokes)}")
    print(f"  中枢数: {len(strategy._daily_pivots)}")
    if strategy._first_buy_price:
        print(f"  1买位置: ${strategy._first_buy_price:.2f}")
    if strategy._second_buy_price:
        print(f"  2买位置: ${strategy._second_buy_price:.2f}")

    # 保存结果
    filename = f'aapl_backtest_{"real" if is_real else "sim"}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"AAPL 多级别缠论策略回测报告 ({data_type}数据)\n")
        f.write("="*70 + "\n\n")
        f.write(f"数据周期: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"K线数量: {len(df)}\n")
        f.write(f"价格区间: ${df['low'].min():.2f} ~ ${df['high'].max():.2f}\n\n")
        f.write(f"初始资金: ${config.initial_capital:,.2f}\n")
        f.write(f"最终权益: ${results['final_equity']:,.2f}\n")
        f.write(f"总收益率: {results['total_return']:.2%}\n")
        f.write(f"年化收益: {results['annual_return']:.2%}\n")
        f.write(f"夏普比率: {results['sharpe_ratio']:.2f}\n")
        f.write(f"最大回撤: {results['max_drawdown']:.2%}\n")
        f.write(f"胜率: {results['win_rate']:.2%}\n")
        f.write(f"总交易: {results['total_trades']} 次\n\n")
        if trades:
            f.write("交易明细:\n")
            for i, trade in enumerate(trades, 1):
                side = "BUY" if trade.signal_type.value == 'buy' else "SELL"
                f.write(f"{i}. {trade.datetime.strftime('%Y-%m-%d')} {side} @ ${trade.price:.2f} x {trade.quantity} - {trade.reason}\n")
        else:
            f.write("无交易\n")

    print(f"\n报告已保存到: {filename}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出...")
