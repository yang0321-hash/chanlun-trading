"""缠论交易系统 - yfinance数据源演示"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("缠论交易系统 - yfinance数据源")
    print("=" * 60)

    print("\n[1/3] 安装yfinance...")
    import subprocess
    try:
        import yfinance as yf
        print("  yfinance已安装")
    except ImportError:
        print("  正在安装yfinance...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
            "yfinance"
        ])
        print("  安装完成")
        import yfinance as yf

    print("\n[2/3] 测试网络连接...")
    test_symbols = [
        ('AAPL', '苹果(美股)'),
        ('000001.SZ', '平安银行(A股)'),
        ('TSLA', '特斯拉(美股)'),
    ]

    working_symbol = None
    for symbol, name in test_symbols:
        try:
            print(f"  尝试 {name} ({symbol})...", end=" ")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            if not data.empty:
                print(f"成功! 最新价格: {data['Close'].iloc[-1]:.2f}")
                working_symbol = symbol
                break
            else:
                print("无数据")
        except Exception as e:
            print(f"失败: {str(e)[:50]}")

    if not working_symbol:
        print("\n  所有连接失败，可能原因:")
        print("  1. 需要科学上网(代理)")
        print("  2. 网络不稳定")
        print("\n  如果有代理，可以这样使用:")
        print("    from data import YFinanceSource")
        print("    source = YFinanceSource(proxy='http://127.0.0.1:7890')")
        input("\n按回车键退出...")
        return

    print(f"\n[3/3] 使用 {working_symbol} 进行回测...")

    try:
        from data import YFinanceSource
        from core import KLine, FractalDetector, StrokeGenerator, PivotDetector
        from backtest import BacktestEngine, BacktestConfig
        from strategies import ChanLunStrategy

        # 获取数据
        source = YFinanceSource()
        print(f"  正在获取数据...")
        df = source.get_kline(
            symbol=working_symbol,
            period='daily',
            adjust='qfq'
        )
        print(f"  获取数据成功，共 {len(df)} 条记录")
        print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")

        # 分析缠论
        kline = KLine.from_dataframe(df, strict_mode=True)

        detector = FractalDetector(kline, confirm_required=False)
        print(f"  顶分型: {len(detector.get_top_fractals())} 个")
        print(f"  底分型: {len(detector.get_bottom_fractals())} 个")

        generator = StrokeGenerator(kline, min_bars=5)
        strokes = generator.get_strokes()
        print(f"  笔: {len(strokes)} 笔")

        pivot_detector = PivotDetector(kline, strokes)
        pivots = pivot_detector.get_pivots()
        print(f"  中枢: {len(pivots)} 个")

        # 回测
        config = BacktestConfig(
            initial_capital=10000 if working_symbol != '000001.SZ' else 100000,
            commission=0.001,
            slippage=0.0001,
            min_unit=1
        )

        engine = BacktestEngine(config)
        engine.add_data(working_symbol, df)

        strategy = ChanLunStrategy(use_macd=False)
        engine.set_strategy(strategy)

        print("  正在回测...")
        results = engine.run()

        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)
        print(f"  初始资金: ¥{config.initial_capital:,.2f}")
        print(f"  最终权益: ¥{results['final_equity']:,.2f}")
        print(f"  总收益率: {results['total_return']:.2%}")
        print(f"  年化收益: {results['annual_return']:.2%}")
        print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {results['max_drawdown']:.2%}")
        print(f"  胜率: {results['win_rate']:.2%}")
        print(f"  总交易: {results['total_trades']} 次")
        print("=" * 60)

    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n演示完成！")

if __name__ == "__main__":
    main()
    input("\n按回车键退出...")
