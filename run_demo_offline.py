"""缠论交易系统 - 离线演示（使用模拟数据）"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_mock_data(symbol: str = "000001", days: int = 500) -> pd.DataFrame:
    """生成模拟K线数据"""
    np.random.seed(42)

    # 生成日期序列
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')

    # 只保留工作日
    dates = [d for d in dates if d.weekday() < 5]

    # 生成价格数据（随机游走）
    price = 10.0
    prices = []
    volumes = []

    for i in range(len(dates)):
        change = np.random.randn() * 0.03  # 3%波动
        price = price * (1 + change)
        open_price = price * (1 + np.random.randn() * 0.01)
        close_price = price * (1 + np.random.randn() * 0.01)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.01))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.01))

        prices.append({
            'datetime': dates[i],
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': int(np.random.randint(1000000, 50000000))
        })

    df = pd.DataFrame(prices)
    df.set_index('datetime', inplace=True)
    return df


def main():
    print("=" * 60)
    print("缠论交易系统 - 离线演示")
    print("=" * 60)

    # 测试导入
    print("\n[1/4] 导入模块...")
    try:
        from core import KLine, FractalDetector, StrokeGenerator, PivotDetector
        from backtest import BacktestEngine, BacktestConfig
        from strategies import ChanLunStrategy
        print("  模块导入成功")
    except Exception as e:
        print(f"  导入失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 生成模拟数据
    print("\n[2/4] 生成模拟数据...")
    try:
        symbol = "MOCK001"
        df = generate_mock_data(symbol, days=500)
        print(f"  生成模拟数据成功，共 {len(df)} 条记录")
        print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"  最新收盘价: {df['close'].iloc[-1]:.2f}")
        print(f"  价格区间: {df['low'].min():.2f} - {df['high'].max():.2f}")
    except Exception as e:
        print(f"  生成数据失败: {e}")
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

        # 显示最近几个分型
        if top_fractals:
            recent_tops = top_fractals[-3:]
            print(f"  最近顶分型: {[(f.datetime.strftime('%Y-%m-%d'), f.high) for f in recent_tops]}")
        if bottom_fractals:
            recent_bottoms = bottom_fractals[-3:]
            print(f"  最近底分型: {[(f.datetime.strftime('%Y-%m-%d'), f.low) for f in recent_bottoms]}")

        generator = StrokeGenerator(kline, min_bars=5)
        strokes = generator.get_strokes()
        print(f"  笔: {len(strokes)} 笔")
        print(f"    向上笔: {len([s for s in strokes if s.is_up])} 笔")
        print(f"    向下笔: {len([s for s in strokes if s.is_down])} 笔")

        pivot_detector = PivotDetector(kline, strokes)
        pivots = pivot_detector.get_pivots()
        print(f"  中枢: {len(pivots)} 个")

        for i, pivot in enumerate(pivots[-3:], 1):
            print(f"    中枢{i}: [{pivot.low:.2f}, {pivot.high:.2f}], 宽度{pivot.range_value:.2f}, {pivot.strokes_count}笔")

    except Exception as e:
        print(f"  分析失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 回测
    print("\n[4/4] 运行策略回测...")
    try:
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
    print("\n提示: 如需使用真实数据，请确保网络畅通后运行 run_demo.py")

if __name__ == "__main__":
    main()
    input("\n按回车键退出...")
