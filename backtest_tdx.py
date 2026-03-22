"""使用通达信数据回测缠论策略"""
import sys
sys.path.insert(0, '.')

import os
from datetime import datetime, timedelta


def main():
    print("="*70)
    print("         通达信数据 缠论策略回测")
    print("="*70)

    # 通达信路径
    tdx_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc"

    if not os.path.exists(tdx_path):
        print(f"\n错误: 通达信路径不存在")
        print(f"  {tdx_path}")
        return

    print(f"\n[通达信数据源]")
    print(f"  路径: {tdx_path}")

    from data import TDXDataSource
    from strategies import MultiLevelChanLunStrategy
    from backtest import BacktestEngine, BacktestConfig

    # 创建数据源
    try:
        source = TDXDataSource(tdx_path)
        print("  连接成功")
    except Exception as e:
        print(f"  连接失败: {e}")
        return

    # 获取股票列表
    print("\n[扫描股票]")
    stock_list = source.get_stock_list()
    print(f"  发现 {len(stock_list)} 只股票")

    if len(stock_list) == 0:
        print("  未发现股票数据，请检查通达信路径")
        return

    # 显示前10只股票
    print("\n  前10只股票:")
    for i, stock in stock_list.head(10).iterrows():
        print(f"    {stock['symbol']}")

    # 选择要回测的股票
    # 默认选择第一只股票
    test_symbol = stock_list.iloc[0]['code']
    print(f"\n[选择股票] {test_symbol}")

    # 获取数据
    print("\n[获取K线数据]")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1年数据

    df = source.get_kline(
        symbol=test_symbol,
        start_date=start_date,
        end_date=end_date,
        period='daily'
    )

    if df.empty:
        print("  未获取到数据")
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
    engine.add_data(test_symbol, df)
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

    # 缠论分析
    print(f"\n[缠论要素]")
    print(f"  分型: {len(strategy._daily_fractals)} 个")
    print(f"  笔: {len(strategy._daily_strokes)} 笔")
    print(f"  中枢: {len(strategy._daily_pivots)} 个")

    # 保存报告
    with open('tdx_backtest_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"通达信数据 缠论策略回测报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"股票代码: {test_symbol}\n")
        f.write(f"数据周期: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"K线数量: {len(df)}\n\n")
        f.write(f"初始资金: ¥{config.initial_capital:,.2f}\n")
        f.write(f"最终权益: ¥{results['final_equity']:,.2f}\n")
        f.write(f"总收益率: {results['total_return']:.2%}\n")
        f.write(f"年化收益: {results['annual_return']:.2%}\n")
        f.write(f"夏普比率: {results['sharpe_ratio']:.2f}\n")
        f.write(f"最大回撤: {results['max_drawdown']:.2%}\n")
        f.write(f"胜率: {results['win_rate']:.2%}\n")
        f.write(f"总交易: {results['total_trades']} 次\n\n")
        if trades:
            f.write("交易明细:\n")
            for i, trade in enumerate(trades, 1):
                side = "买入" if trade.signal_type.value == 'buy' else "卖出"
                f.write(f"{i}. {trade.datetime.strftime('%Y-%m-%d')} {side} @ ¥{trade.price:.2f} x {trade.quantity}\n")

    print("\n报告已保存到: tdx_backtest_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出...")
