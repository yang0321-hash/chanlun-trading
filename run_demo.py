"""缠论交易系统 - 简化演示"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("缠论交易系统 - 回测演示")
    print("=" * 60)

    # 测试导入
    print("\n[1/4] 导入模块...")
    try:
        import pandas as pd
        import numpy as np
        from data import AKShareSource
        from core import KLine, FractalDetector, StrokeGenerator, PivotDetector
        print("  模块导入成功")
    except Exception as e:
        print(f"  导入失败: {e}")
        return

    # 获取数据
    print("\n[2/4] 获取股票数据...")
    try:
        source = AKShareSource()
        symbol = "000001"
        print(f"  获取 {symbol} 数据...")
        df = source.get_kline(symbol=symbol, period='daily', adjust='qfq')
        print(f"  获取数据成功，共 {len(df)} 条记录")
        print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"  最新收盘价: {df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"  获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 分析缠论
    print("\n[3/4] 分析缠论要素...")
    try:
        kline = KLine.from_dataframe(df, strict_mode=True)
        print(f"  K线处理完成，原始{len(df)}根 -> 处理后{len(kline)}根")

        detector = FractalDetector(kline, confirm_required=False)
        top_fractals = detector.get_top_fractals()
        bottom_fractals = detector.get_bottom_fractals()
        print(f"  顶分型: {len(top_fractals)} 个")
        print(f"  底分型: {len(bottom_fractals)} 个")

        generator = StrokeGenerator(kline, min_bars=5)
        strokes = generator.get_strokes()
        print(f"  笔: {len(strokes)} 笔")

        pivot_detector = PivotDetector(kline, strokes)
        pivots = pivot_detector.get_pivots()
        print(f"  中枢: {len(pivots)} 个")

    except Exception as e:
        print(f"  分析失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 回测
    print("\n[4/4] 运行策略回测...")
    try:
        from backtest import BacktestEngine, BacktestConfig
        from strategies import ChanLunStrategy

        config = BacktestConfig(
            initial_capital=100000,
            commission=0.0003,
            slippage=0.0001,
            min_unit=100
        )

        engine = BacktestEngine(config)
        engine.add_data(symbol, df)

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
        print(f"  回测失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n演示完成！")

if __name__ == "__main__":
    main()
    input("\n按回车键退出...")
