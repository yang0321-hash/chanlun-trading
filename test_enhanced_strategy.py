"""
增强策略回测与参数优化脚本

用法:
    python test_enhanced_strategy.py --symbol sh600519
    python test_enhanced_strategy.py --optimize
    python test_enhanced_strategy.py --walk-forward
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.optimization import (
    GridSearchOptimizer, WalkForwardOptimizer, create_default_param_grid
)
from backtest.position_sizing import RiskParitySizer, KellySizer
from backtest.portfolio import generate_portfolio_report
from strategies.enhanced_chan_strategy import EnhancedChanLunStrategy
from data.tdx_source import TDXDataSource


def run_enhanced_backtest(
    symbol: str,
    tdx_path: str = None,
    position_method: str = 'risk_parity',
    risk_per_trade: float = 0.02,
    atr_multiplier: float = 2.0,
) -> dict:
    """运行增强策略回测"""

    # 获取数据
    if tdx_path:
        source = TDXDataSource(tdx_path)
    else:
        from data.akshare_source import AkShareDataSource
        source = AkShareDataSource()

    logger.info(f"获取 {symbol} 数据...")
    df = source.get_kline(symbol, period='daily')

    if df is None or len(df) < 500:
        logger.error(f"数据不足: {len(df) if df is not None else 0}")
        return {}

    # 创建策略
    strategy = EnhancedChanLunStrategy(
        name=f'增强缠论({symbol})',
        position_method=position_method,
        risk_per_trade=risk_per_trade,
        atr_multiplier=atr_multiplier,
        use_volume_filter=True,
        use_rsi_filter=True,
        use_trend_filter=True,
        max_positions=5,
    )

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
    logger.info("开始回测...")
    results = engine.run()

    # 打印结果
    print("\n" + "="*50)
    print(f"增强策略回测结果 - {symbol}")
    print("="*50)
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益: {results['annual_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"盈亏比: {results['profit_loss_ratio']:.2f}")
    print(f"交易次数: {results['total_trades']}")

    # 打印过滤器统计
    if hasattr(strategy, 'filter_chain'):
        print("\n过滤器统计:")
        stats = strategy.filter_chain.get_stats()
        for name, stat in stats.items():
            print(f"  {name}: {stat['pass_count']}/{stat['total_count']} "
                  f"({stat['pass_rate']:.0%})")

    # 打印组合报告
    print("\n" + generate_portfolio_report(strategy.portfolio_manager))

    return results


def run_grid_search(symbol: str, tdx_path: str = None):
    """运行网格搜索优化"""

    # 获取数据
    if tdx_path:
        source = TDXDataSource(tdx_path)
    else:
        from data.akshare_source import AkShareDataSource
        source = AkShareDataSource()

    logger.info(f"获取 {symbol} 数据...")
    df = source.get_kline(symbol, period='daily')

    if df is None or len(df) < 500:
        logger.error(f"数据不足")
        return

    # 定义参数网格
    param_grid = {
        'risk_per_trade': [0.01, 0.015, 0.02, 0.025],
        'atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'stop_loss_pct': [0.05, 0.08, 0.10],
    }

    def backtest_func(params):
        strategy = EnhancedChanLunStrategy(
            name='优化中',
            position_method='risk_parity',
            risk_per_trade=params.get('risk_per_trade', 0.02),
            atr_multiplier=params.get('atr_multiplier', 2.0),
            stop_loss_pct=params.get('stop_loss_pct', 0.08),
        )

        config = BacktestConfig(
            initial_capital=100000,
            commission=0.0003,
            slippage=0.0001,
            min_unit=100,
        )

        engine = BacktestEngine(config)
        engine.add_data(symbol, df)
        engine.set_strategy(strategy)

        return engine.run()

    # 运行优化
    optimizer = GridSearchOptimizer(
        objective='sharpe_ratio',
        maximize=True,
        n_jobs=1,  # Windows下建议用1
    )

    logger.info("开始网格搜索...")
    best_result = optimizer.optimize(
        param_grid=param_grid,
        backtest_func=backtest_func,
        data=df,
        min_trades=5,
    )

    if best_result:
        print("\n" + "="*50)
        print("网格搜索最优参数:")
        print("="*50)
        print(f"参数: {best_result.params}")
        print(f"Sharpe: {best_result.sharpe_ratio:.4f}")
        print(f"总收益: {best_result.total_return:.2%}")
        print(f"最大回撤: {best_result.max_drawdown:.2%}")
        print(f"胜率: {best_result.win_rate:.2%}")

        # 打印前5名
        print("\n前5名参数组合:")
        for i, r in enumerate(optimizer.get_top_n(5)):
            print(f"{i+1}. {r.params} -> Sharpe={r.sharpe_ratio:.4f}, "
                  f"收益={r.total_return:.2%}")

        # 保存结果
        output_file = Path(__file__).parent / 'optimization_results.json'
        optimizer.save_results(str(output_file))
        logger.info(f"结果已保存到 {output_file}")


def run_walk_forward(symbol: str, tdx_path: str = None):
    """运行Walk-forward分析"""

    # 获取数据
    if tdx_path:
        source = TDXDataSource(tdx_path)
    else:
        from data.akshare_source import AkShareDataSource
        source = AkShareDataSource()

    logger.info(f"获取 {symbol} 数据...")
    df = source.get_kline(symbol, period='daily')

    if df is None or len(df) < 500:
        logger.error(f"数据不足")
        return

    # 定义参数网格
    param_grid = {
        'risk_per_trade': [0.015, 0.02, 0.025],
        'atr_multiplier': [2.0, 2.5],
    }

    def backtest_func(params, data):
        strategy = EnhancedChanLunStrategy(
            name='Walk-forward',
            position_method='risk_parity',
            risk_per_trade=params.get('risk_per_trade', 0.02),
            atr_multiplier=params.get('atr_multiplier', 2.0),
        )

        config = BacktestConfig(
            initial_capital=100000,
            commission=0.0003,
            slippage=0.0001,
            min_unit=100,
        )

        engine = BacktestEngine(config)
        engine.add_data(symbol, data)
        engine.set_strategy(strategy)

        return engine.run()

    # 运行Walk-forward
    optimizer = WalkForwardOptimizer(
        in_sample_period=252,   # 1年样本内
        out_sample_period=63,   # 3个月样本外
        step_period=21,         # 1个月步长
    )

    logger.info("开始Walk-forward分析...")
    summary = optimizer.optimize(
        param_grid=param_grid,
        backtest_func=backtest_func,
        data=df,
        objective='sharpe_ratio',
        maximize=True,
    )

    # 打印结果
    print("\n" + "="*50)
    print("Walk-forward分析结果:")
    print("="*50)
    print(f"窗口数: {summary['windows']}")
    print(f"样本内平均Sharpe: {summary['in_sample']['mean_sharpe']:.4f} "
          f"(±{summary['in_sample']['std_sharpe']:.4f})")
    print(f"样本外平均Sharpe: {summary['out_sample']['mean_sharpe']:.4f} "
          f"(±{summary['out_sample']['std_sharpe']:.4f})")
    print(f"样本内平均收益: {summary['in_sample']['mean_return']:.2%}")
    print(f"样本外平均收益: {summary['out_sample']['mean_return']:.2%}")
    print(f"稳定性比率: {summary['stability_ratio']:.2%}")

    # 保存结果
    output_file = Path(__file__).parent / 'walkforward_results.csv'
    optimizer.save_results(str(output_file))
    logger.info(f"结果已保存到 {output_file}")


def compare_position_methods(symbol: str, tdx_path: str = None):
    """比较不同仓位管理方法"""

    # 获取数据
    if tdx_path:
        source = TDXDataSource(tdx_path)
    else:
        from data.akshare_source import AkShareDataSource
        source = AkShareDataSource()

    logger.info(f"获取 {symbol} 数据...")
    df = source.get_kline(symbol, period='daily')

    if df is None or len(df) < 500:
        logger.error(f"数据不足")
        return

    methods = ['risk_parity', 'kelly', 'adaptive']
    results = {}

    for method in methods:
        logger.info(f"测试 {method} 仓位管理...")

        strategy = EnhancedChanLunStrategy(
            name=f'{method}策略',
            position_method=method,
        )

        config = BacktestConfig(
            initial_capital=100000,
            commission=0.0003,
            slippage=0.0001,
            min_unit=100,
        )

        engine = BacktestEngine(config)
        engine.add_data(symbol, df)
        engine.set_strategy(strategy)

        result = engine.run()
        results[method] = result

    # 打印对比结果
    print("\n" + "="*60)
    print(f"仓位管理方法对比 - {symbol}")
    print("="*60)
    print(f"{'方法':<15} {'收益率':<10} {'Sharpe':<10} {'最大回撤':<10} {'胜率':<10}")
    print("-"*60)

    for method, result in results.items():
        print(f"{method:<15} {result['total_return']:>9.2%} "
              f"{result['sharpe_ratio']:>10.2f} {result['max_drawdown']:>10.2%} "
              f"{result['win_rate']:>10.2%}")


def main():
    parser = argparse.ArgumentParser(description='增强策略回测与优化')
    parser.add_argument('--symbol', type=str, default='sh600519',
                       help='股票代码 (默认: sh600519)')
    parser.add_argument('--tdx-path', type=str,
                       default='D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc',
                       help='TDX数据路径')
    parser.add_argument('--position-method', type=str, default='risk_parity',
                       choices=['risk_parity', 'kelly', 'adaptive'],
                       help='仓位管理方法')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='每笔交易风险比例')
    parser.add_argument('--atr-mult', type=float, default=2.0,
                       help='ATR止损倍数')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--optimize', action='store_true',
                      help='运行网格搜索优化')
    group.add_argument('--walk-forward', action='store_true',
                      help='运行Walk-forward分析')
    group.add_argument('--compare', action='store_true',
                      help='比较仓位管理方法')

    args = parser.parse_args()

    if args.optimize:
        run_grid_search(args.symbol, args.tdx_path)
    elif args.walk_forward:
        run_walk_forward(args.symbol, args.tdx_path)
    elif args.compare:
        compare_position_methods(args.symbol, args.tdx_path)
    else:
        run_enhanced_backtest(
            args.symbol,
            args.tdx_path,
            args.position_method,
            args.risk,
            args.atr_mult,
        )


if __name__ == '__main__':
    main()
