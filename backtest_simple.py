"""
简化版增强策略回测

使用宽松的过滤器参数进行测试
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from backtest.engine import BacktestEngine, BacktestConfig
from strategies.enhanced_chan_strategy import EnhancedChanLunStrategy
from data.tdx_source import TDXDataSource

# TDX数据路径
TDX_PATH = 'D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc'

# 测试股票列表
TEST_STOCKS = [
    'sh600000',  # 浦发银行
    'sh600519',  # 贵州茅台
    'sh600036',  # 招商银行
    'sz000001',  # 平安银行
    'sz000002',  # 万科A
]

def run_backtest(symbol: str, use_filters: bool = True):
    """运行单个股票回测"""
    print(f"\n{'='*60}")
    print(f"回测 {symbol}")
    print(f"{'='*60}")

    source = TDXDataSource(TDX_PATH)
    df = source.get_kline(symbol, period='daily')

    if df is None or len(df) < 500:
        print(f"数据不足: {len(df) if df is not None else 0}")
        return None

    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
    print(f"数据量: {len(df)} 条")

    # 创建策略 - 使用宽松参数
    if use_filters:
        strategy = EnhancedChanLunStrategy(
            name=f'增强策略({symbol})',
            position_method='risk_parity',
            risk_per_trade=0.025,        # 2.5%风险
            atr_multiplier=2.5,            # ATR止损2.5倍
            # 放宽过滤器
            use_trend_filter=False,       # 关闭趋势过滤
            use_volume_filter=False,      # 关闭成交量过滤
            use_rsi_filter=False,         # 关闭RSI过滤
            use_bollinger_filter=False,   # 关闭布林带过滤
            max_positions=5,
        )
        strategy_name = "增强策略(宽松过滤)"
    else:
        # 使用原版策略对比
        from strategies.chan_strategy import ChanLunStrategy
        strategy = ChanLunStrategy(name=f'原版策略({symbol})')
        strategy_name = "原版策略"

    # 创建回测引擎
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
    )

    engine = BacktestEngine(config)
    engine.add_data(symbol, df)
    engine.set_strategy(strategy)

    # 运行回测
    results = engine.run()

    # 打印结果
    print(f"\n{strategy_name} 回测结果:")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  年化收益: {results['annual_return']:.2%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print(f"  盈亏比: {results['profit_loss_ratio']:.2f}")
    print(f"  交易次数: {results['total_trades']}")

    # 打印交易明细
    if results['trades']:
        print(f"\n最近交易:")
        for trade in list(results['trades'])[-5:]:
            print(f"  {trade.datetime} {trade.signal_type.value} {trade.symbol} "
                  f"@ {trade.price:.2f} x {trade.quantity} ({trade.reason})")

    return {
        'symbol': symbol,
        'strategy': strategy_name,
        'total_return': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'win_rate': results['win_rate'],
        'total_trades': results['total_trades'],
    }


def main():
    """主函数"""
    print("="*60)
    print("缠论策略回测 - 宽松参数")
    print("="*60)

    all_results = []

    for symbol in TEST_STOCKS[:3]:  # 测试前3个
        try:
            # 原版策略
            result_orig = run_backtest(symbol, use_filters=False)

            # 增强策略
            result_enh = run_backtest(symbol, use_filters=True)

            if result_orig:
                all_results.append(result_orig)
            if result_enh:
                all_results.append(result_enh)

        except Exception as e:
            print(f"回测 {symbol} 失败: {e}")

    # 汇总对比
    print(f"\n{'='*60}")
    print("策略对比汇总")
    print(f"{'='*60}")
    print(f"{'股票':<10} {'策略':<15} {'收益率':<10} {'Sharpe':<10} {'最大回撤':<10} {'胜率':<10} {'交易数':<8}")
    print("-"*60)

    for r in all_results:
        print(f"{r['symbol']:<10} {r['strategy']:<15} "
              f"{r['total_return']:>9.2%} {r['sharpe_ratio']:>10.2f} "
              f"{r['max_drawdown']:>9.2%} {r['win_rate']:>9.2%} {r['total_trades']:>8}")


if __name__ == '__main__':
    main()
