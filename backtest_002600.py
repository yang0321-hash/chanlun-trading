"""回测 002600 - 通达信数据"""
import sys
sys.path.insert(0, '.')

import os
from datetime import datetime, timedelta


def main():
    print("="*70)
    print("         回测 002600 缠论策略")
    print("="*70)

    # 通达信路径
    tdx_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc"

    if not os.path.exists(tdx_path):
        print(f"\n错误: 通达信路径不存在")
        print(f"  {tdx_path}")
        return

    from data import TDXDataSource
    from strategies import MultiLevelChanLunStrategy
    from backtest import BacktestEngine, BacktestConfig

    # 创建数据源
    source = TDXDataSource(tdx_path)

    # 获取002600数据
    symbol = "002600"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2年数据

    print(f"\n[获取数据] {symbol}")
    df = source.get_kline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period='daily'
    )

    if df.empty:
        print(f"  未获取到 {symbol} 的数据")
        print("\n  请检查:")
        print("  1. 通达信路径是否正确")
        print("  2. 002600 数据文件是否存在")
        print("  3. 数据是否在 SzTdzxDay 目录下")
        return

    print(f"  K线: {len(df)} 条")
    print(f"  时间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  价格: ¥{df['low'].min():.2f} ~ ¥{df['high'].max():.2f}")
    print(f"  最新: ¥{df['close'].iloc[-1]:.2f}")

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
    print(f"  股票代码:     002600")
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
            pnl = (trade.price - trade.commission / trade.quantity) if side == "买入" else 0
            print(f"  {i}. {trade.datetime.strftime('%Y-%m-%d')} "
                  f"{side} @ ¥{trade.price:.2f} x {trade.quantity} "
                  f"({trade.reason})")
    else:
        print("\n[交易明细] 无交易")

    # 缠论分析
    print(f"\n[缠论要素分析]")
    print(f"  分型: {len(strategy._daily_fractals)} 个")
    print(f"  笔: {len(strategy._daily_strokes)} 笔")
    print(f"  中枢: {len(strategy._daily_pivots)} 个")

    if strategy._first_buy_price:
        print(f"  1买位置: ¥{strategy._first_buy_price:.2f}")
    if strategy._second_buy_price:
        print(f"  2买位置: ¥{strategy._second_buy_price:.2f}")

    # 趋势分析
    print(f"\n[趋势分析]")
    recent_strokes = strategy._daily_strokes[-10:] if len(strategy._daily_strokes) >= 10 else strategy._daily_strokes
    if recent_strokes:
        up_count = len([s for s in recent_strokes if s.is_up])
        down_count = len(recent_strokes) - up_count
        print(f"  最近{len(recent_strokes)}笔: 向上{up_count}笔, 向下{down_count}笔")

    # 保存报告
    with open('002600_backtest_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"002600 缠论策略回测报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"数据周期: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"K线数量: {len(df)}\n")
        f.write(f"价格区间: ¥{df['low'].min():.2f} ~ ¥{df['high'].max():.2f}\n\n")
        f.write(f"初始资金: ¥{config.initial_capital:,.2f}\n")
        f.write(f"最终权益: ¥{results['final_equity']:,.2f}\n")
        f.write(f"总收益率: {results['total_return']:.2%}\n")
        f.write(f"年化收益: {results['annual_return']:.2%}\n")
        f.write(f"夏普比率: {results['sharpe_ratio']:.2f}\n")
        f.write(f"最大回撤: {results['max_drawdown']:.2%}\n")
        f.write(f"胜率: {results['win_rate']:.2%}\n")
        f.write(f"总交易: {results['total_trades']} 次\n\n")
        f.write(f"缠论要素: 分型{len(strategy._daily_fractals)}个, "
                f"笔{len(strategy._daily_strokes)}笔, "
                f"中枢{len(strategy._daily_pivots)}个\n\n")
        if trades:
            f.write("交易明细:\n")
            for i, trade in enumerate(trades, 1):
                side = "买入" if trade.signal_type.value == 'buy' else "卖出"
                f.write(f"{i}. {trade.datetime.strftime('%Y-%m-%d')} {side} @ ¥{trade.price:.2f} x {trade.quantity} - {trade.reason}\n")

    print("\n报告已保存到: 002600_backtest_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出...")
