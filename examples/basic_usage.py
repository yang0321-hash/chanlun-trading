"""
缠论交易系统使用示例

演示如何使用缠论交易系统进行数据获取、分析和回测
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import AKShareSource
from core import (
    KLine, FractalDetector, StrokeGenerator,
    SegmentGenerator, PivotDetector, ChanLunPlotter
)
from indicator import MACD
from backtest import BacktestEngine, BacktestConfig
from strategies import ChanLunStrategy


def example_1_get_data():
    """示例1: 获取A股数据"""
    print("=" * 50)
    print("示例1: 获取A股数据")
    print("=" * 50)

    # 创建数据源
    source = AKShareSource()

    # 获取股票K线数据
    symbol = "000001"  # 平安银行
    df = source.get_kline(
        symbol=symbol,
        period='daily',
        adjust='qfq'  # 前复权
    )

    print(f"获取 {symbol} 数据，共 {len(df)} 条记录")
    print(df.tail())
    print()

    return df, symbol


def example_2_analyze_fractals(df, symbol):
    """示例2: 识别分型"""
    print("=" * 50)
    print("示例2: 识别分型")
    print("=" * 50)

    # 创建K线对象
    kline = KLine.from_dataframe(df, strict_mode=True)

    # 识别分型
    detector = FractalDetector(kline, confirm_required=False)
    top_fractals = detector.get_top_fractals()
    bottom_fractals = detector.get_bottom_fractals()

    print(f"识别到顶分型 {len(top_fractals)} 个")
    print(f"识别到底分型 {len(bottom_fractals)} 个")

    if top_fractals:
        print(f"\n最新顶分型: {top_fractals[-1].datetime}, 价格: {top_fractals[-1].high:.2f}")
    if bottom_fractals:
        print(f"最新底分型: {bottom_fractals[-1].datetime}, 价格: {bottom_fractals[-1].low:.2f}")
    print()

    return kline, detector.get_fractals()


def example_3_generate_strokes(kline, fractals):
    """示例3: 生成笔"""
    print("=" * 50)
    print("示例3: 生成笔")
    print("=" * 50)

    # 生成笔
    generator = StrokeGenerator(kline, fractals, min_bars=5)
    strokes = generator.get_strokes()

    print(f"生成笔 {len(strokes)} 笔")

    up_strokes = generator.get_up_strokes()
    down_strokes = generator.get_down_strokes()
    print(f"  向上笔: {len(up_strokes)} 笔")
    print(f"  向下笔: {len(down_strokes)} 笔")

    if strokes:
        last = strokes[-1]
        print(f"\n最后一笔: {last.type.value}, "
              f"从 {last.start_value:.2f} 到 {last.end_value:.2f}, "
              f"涨跌 {last.price_change_pct:.2f}%")
    print()

    return strokes


def example_4_identify_pivots(kline, strokes):
    """示例4: 识别中枢"""
    print("=" * 50)
    print("示例4: 识别中枢")
    print("=" * 50)

    # 识别中枢
    detector = PivotDetector(kline, strokes)
    pivots = detector.get_pivots()

    print(f"识别到中枢 {len(pivots)} 个")

    for i, pivot in enumerate(pivots[-3:], 1):  # 显示最后3个
        print(f"中枢{i}: 区间 [{pivot.low:.2f}, {pivot.high:.2f}], "
              f"宽度 {pivot.range_value:.2f}, "
              f"笔数 {pivot.strokes_count}")
    print()

    return pivots


def example_5_visualize(kline, fractals, strokes, pivots):
    """示例5: 可视化"""
    print("=" * 50)
    print("示例5: 生成可视化图表")
    print("=" * 50)

    try:
        plotter = ChanLunPlotter(kline)

        # 绘制完整分析图
        fig = plotter.plot_full_analysis(
            fractals=fractals,
            strokes=strokes,
            pivots=pivots,
            title='缠论完整分析'
        )

        # 保存图表
        output_path = './chanlun_analysis.html'
        plotter.save(fig, output_path)
        print(f"图表已保存到: {output_path}")

    except ImportError as e:
        print(f"可视化需要安装 plotly: pip install plotly")
    print()


def example_6_backtest(df, symbol):
    """示例6: 策略回测"""
    print("=" * 50)
    print("示例6: 策略回测")
    print("=" * 50)

    # 创建回测配置
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100
    )

    # 创建回测引擎
    engine = BacktestEngine(config)

    # 添加数据
    engine.add_data(symbol, df)

    # 设置策略
    strategy = ChanLunStrategy(use_macd=True)
    engine.set_strategy(strategy)

    # 运行回测
    try:
        results = engine.run()

        # 打印结果
        print(f"总收益率: {results['total_return']:.2%}")
        print(f"年化收益: {results['annual_return']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"胜率: {results['win_rate']:.2%}")
        print(f"总交易次数: {results['total_trades']}")

    except Exception as e:
        print(f"回测出错: {e}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("缠论交易系统 - 使用示例")
    print("=" * 50 + "\n")

    try:
        # 示例1: 获取数据
        df, symbol = example_1_get_data()

        # 示例2: 识别分型
        kline, fractals = example_2_analyze_fractals(df, symbol)

        # 示例3: 生成笔
        strokes = example_3_generate_strokes(kline, fractals)

        # 示例4: 识别中枢
        pivots = example_4_identify_pivots(kline, strokes)

        # 示例5: 可视化
        example_5_visualize(kline, fractals, strokes, pivots)

        # 示例6: 回测
        example_6_backtest(df, symbol)

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
