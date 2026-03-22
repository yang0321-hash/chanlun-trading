"""
策略对比回测

对比原版缠论、周日线缠论、优化版周日线策略
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from backtest.engine import BacktestEngine, BacktestConfig
from data.tdx_source import TDXDataSource

# 导入各策略
from strategies.chan_strategy import ChanLunStrategy
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy
from strategies.optimized_weekly_daily_strategy import OptimizedWeeklyDailyStrategy

# TDX数据路径
TDX_PATH = 'D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc'

# 测试股票
TEST_STOCKS = [
    'sh600000',  # 浦发银行
    'sh600036',  # 招商银行
    'sh600519',  # 贵州茅台
    'sz000001',  # 平安银行
    'sz000002',  # 万科A
]

def run_backtest(symbol: str, strategy_class, strategy_name: str, **kwargs):
    """运行单个策略回测"""
    print(f"\n{'='*60}")
    print(f"[{symbol}] {strategy_name}")
    print(f"{'='*60}")

    source = TDXDataSource(TDX_PATH)
    df = source.get_kline(symbol, period='daily')

    if df is None or len(df) < 500:
        print(f"数据不足: {len(df) if df is not None else 0}")
        return None

    print(f"数据: {df.index.min()} ~ {df.index.max()} ({len(df)}条)")

    # 创建策略
    strategy = strategy_class(**kwargs)

    # 回测配置
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
    )

    engine = BacktestEngine(config)
    engine.add_data(symbol, df)
    engine.set_strategy(strategy)

    # 运行
    results = engine.run()

    # 打印结果
    print(f"  总收益: {results['total_return']:>8.2%}  "
          f"年化: {results['annual_return']:>6.2%}  "
          f"Sharpe: {results['sharpe_ratio']:>6.2f}  "
          f"回撤: {results['max_drawdown']:>7.2%}  "
          f"胜率: {results['win_rate']:>6.1%}  "
          f"交易: {results['total_trades']:>3}")

    return {
        'symbol': symbol,
        'strategy': strategy_name,
        'total_return': results['total_return'],
        'annual_return': results['annual_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'win_rate': results['win_rate'],
        'profit_loss_ratio': results['profit_loss_ratio'],
        'total_trades': results['total_trades'],
    }


def main():
    print("="*70)
    print("缠论策略对比回测")
    print("="*70)

    all_results = []

    for symbol in TEST_STOCKS:
        try:
            # 策略1: 原版缠论 (日线)
            r1 = run_backtest(
                symbol,
                ChanLunStrategy,
                '原版缠论(日线)',
                use_macd=True
            )

            # 策略2: 周日线缠论
            r2 = run_backtest(
                symbol,
                WeeklyDailyChanLunStrategy,
                '周日线缠论',
                weekly_min_strokes=3,
                daily_min_strokes=3,
                stop_loss_pct=0.08,
            )

            # 策略3: 优化版周日线 (宽松参数)
            r3 = run_backtest(
                symbol,
                OptimizedWeeklyDailyStrategy,
                '优化周日线',
                weekly_min_strokes=3,
                daily_min_strokes=3,
                ma_fast=20,
                ma_slow=60,
                atr_multiplier=2.0,
                stop_loss_pct=0.08,
                require_macd_cross=False,
            )

            for r in [r1, r2, r3]:
                if r:
                    all_results.append(r)

        except Exception as e:
            print(f"[{symbol}] 回测失败: {e}")

    # 汇总对比
    print(f"\n{'='*70}")
    print("策略对比汇总")
    print(f"{'='*70}")
    print(f"{'股票':<10} {'策略':<18} {'收益率':<10} {'年化':<10} {'Sharpe':<10} {'回撤':<10} {'胜率':<10} {'交易'}")
    print("-"*70)

    # 按股票和策略排序
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r['symbol']].append(r)

    for symbol in sorted(grouped.keys()):
        results = grouped[symbol]
        # 找出基准 (买入持有收益)
        buy_hold = results[0]['total_return']  # 假设第一个是基准

        for r in results:
            excess = r['total_return'] - buy_hold
            print(f"{symbol:<10} {r['strategy']:<18} "
                  f"{r['total_return']:>9.2%} {r['annual_return']:>9.2%} "
                  f"{r['sharpe_ratio']:>10.2f} {r['max_drawdown']:>9.2%} "
                  f"{r['win_rate']:>9.1%} {r['total_trades']:>4} "
                  f"({excess:+.2%})")

    # 策略排名
    print(f"\n{'='*70}")
    print("按收益率排名 (Top 10)")
    print(f"{'='*70}")

    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)

    print(f"{'排名':<6} {'股票':<10} {'策略':<20} {'收益率':<10} {'Sharpe':<10} {'回撤':<10} {'交易':<6}")
    print("-"*70)

    for i, r in enumerate(sorted_results[:10], 1):
        print(f"{i:<6} {r['symbol']:<10} {r['strategy']:<20} "
              f"{r['total_return']:>9.2%} {r['sharpe_ratio']:>10.2f} "
              f"{r['max_drawdown']:>9.2%} {r['total_trades']:>6}")

    # 统计各策略平均表现
    print(f"\n{'='*70}")
    print("各策略平均表现")
    print(f"{'='*70}")

    from collections import defaultdict
    strategy_stats = defaultdict(list)

    for r in all_results:
        strategy_stats[r['strategy']].append(r['total_return'])
        strategy_stats[r['strategy'] + '_sharpe'].append(r['sharpe_ratio'])
        strategy_stats[r['strategy'] + '_drawdown'].append(r['max_drawdown'])

    print(f"{'策略':<20} {'平均收益率':<15} {'平均Sharpe':<15} {'平均回撤':<15} {'样本数'}")
    print("-"*70)

    for strategy in ['原版缠论(日线)', '周日线缠论', '优化周日线']:
        if strategy in strategy_stats:
            returns = strategy_stats[strategy]
            sharpes = strategy_stats[strategy + '_sharpe']
            drawdowns = strategy_stats[strategy + '_drawdown']

            avg_ret = sum(returns) / len(returns)
            avg_sharpe = sum(sharpes) / len(sharpes)
            avg_dd = sum(drawdowns) / len(drawdowns)

            print(f"{strategy:<20} {avg_ret:>14.2%}  {avg_sharpe:>15.2f}  {avg_dd:>14.2%}  {len(returns)}")


if __name__ == '__main__':
    main()
