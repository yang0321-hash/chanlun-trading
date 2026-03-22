"""AKShare 获取 002600 数据并回测"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from datetime import datetime, timedelta


def fetch_002600_data():
    """使用AKShare获取002600数据"""
    print("="*60)
    print("     获取 002600 数据 (AKShare)")
    print("="*60)

    try:
        import akshare as ak
    except ImportError:
        print("\n请先安装AKShare:")
        print("  pip install akshare")
        input()
        return None

    print("\n正在连接AKShare...")
    print("获取 002600 近3年数据...")

    try:
        # 获取日线数据
        df = ak.stock_zh_a_hist(
            symbol="002600",
            period="daily",
            start_date="20220101",
            end_date=datetime.now().strftime('%Y%m%d'),
            adjust="qfq"
        )

        if df.empty:
            print("获取数据为空")
            return None

        print(f"\n获取成功!")
        print(f"  数据条数: {len(df)} 条")
        print(f"  时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  价格区间: ¥{df['low'].min():.2f} ~ ¥{df['high'].max():.2f}")
        print(f"  最新收盘: ¥{df['close'].iloc[-1]:.2f}")

        # 保存数据
        df.to_csv("002600_data.csv", index=True, encoding='utf-8-sig')
        print(f"\n数据已保存到: 002600_data.csv")

        return df

    except Exception as e:
        print(f"获取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_backtest(df):
    """运行回测"""
    from strategies import WeeklyDailyChanLunStrategy
    from backtest import BacktestEngine, BacktestConfig

    symbol = "002600"

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
    print(f"  策略: 周线2买 + 日线MACD/2卖出")
    print(f"  初始资金: ¥{config.initial_capital:,.2f}")
    print(f"  最终权益: ¥{results['final_equity']:,.2f}")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  年化收益: {results['annual_return']:.2%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print(f"  总交易: {results['total_trades']} 次")
    print("="*60)

    # 交易明细
    if trades:
        print("\n[交易明细]")
        for i, trade in enumerate(trades, 1):
            side = "买入" if trade.signal_type.value == 'buy' else "卖出"
            print(f"  {i}. {trade.datetime.strftime('%Y-%m-%d')} "
                  f"{side} @ ¥{trade.price:.2f} x {trade.quantity}")
    else:
        print("\n[交易明细] 无交易")

    # 缠论要素
    print(f"\n[周线分析]")
    print(f"  分型: {len(strategy._weekly_fractals)} 个")
    print(f"  笔: {len(strategy._weekly_strokes)} 笔")
    print(f"  中枢: {len(strategy._weekly_pivots)} 个")
    if strategy._weekly_first_buy_price:
        print(f"  1买: ¥{strategy._weekly_first_buy_price:.2f}")
    if strategy._weekly_second_buy_price:
        print(f"  2买: ¥{strategy._weekly_second_buy_price:.2f}")

    # 保存报告
    with open('002600_backtest.txt', 'w', encoding='utf-8') as f:
        f.write(f"002600 周日线缠论策略回测报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"数据来源: AKShare\n")
        f.write(f"回测周期: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"策略规则: 周线2买 + 日线MACD顶背离减仓50% + 日线2卖\n")
        f.write("="*60 + "\n\n")
        f.write(f"初始资金: ¥{config.initial_capital:,.2f}\n")
        f.write(f"最终权益: ¥{results['final_equity']:,.2f}\n")
        f.write(f"总收益率: {results['total_return']:.2%}\n")
        f.write(f"夏普比率: {results['sharpe_ratio']:.2f}\n")
        f.write(f"最大回撤: {results['max_drawdown']:.2%}\n")
        f.write(f"总交易: {results['total_trades']} 次\n")

    print("\n报告已保存到: 002600_backtest.txt")


def main():
    # 获取数据
    df = fetch_002600_data()

    if df is None or df.empty:
        print("\n无法获取数据，尝试读取本地CSV...")
        try:
            df = pd.read_csv('002600_data.csv', index_col=0, parse_dates=True)
            print(f"从本地CSV读取: {len(df)} 条数据")
        except:
            print("没有本地CSV数据")
            return

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
