"""缠论策略优化报告生成器"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_mock_data(symbol: str = "MOCK001", days: int = 500) -> pd.DataFrame:
    """生成模拟K线数据"""
    import pandas as pd
    import numpy as np
    from datetime import datetime

    np.random.seed(42)
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    dates = [d for d in dates if d.weekday() < 5]

    price = 10.0
    prices = []
    for i in range(len(dates)):
        change = np.random.randn() * 0.03
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


def run_backtest(strategy_class, df: pd.DataFrame, symbol: str) -> dict:
    """运行回测"""
    from backtest import BacktestEngine, BacktestConfig

    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100
    )

    engine = BacktestEngine(config)
    engine.add_data(symbol, df)
    strategy = strategy_class()
    engine.set_strategy(strategy)
    return engine.run()


def generate_report(original: dict, optimized: dict) -> str:
    """生成优化报告"""
    report = f"""
{'='*70}
                    缠论交易策略优化报告
{'='*70}

一、回测概况
{'-'*70}
  数据周期: 500个交易日
  初始资金: ¥100,000
  手续费率: 0.03%
  滑点: 0.01%

二、绩效对比
{'-'*70}
{'指标':<20} {'原版策略':<20} {'优化策略':<20} {'提升/下降'}
{'-'*70}
"""

    metrics = [
        ('最终权益', 'final_equity', '¥', ':,.2f'),
        ('总收益率', 'total_return', '', ':.2%'),
        ('年化收益率', 'annual_return', '', ':.2%'),
        ('夏普比率', 'sharpe_ratio', '', ':.2f'),
        ('最大回撤', 'max_drawdown', '', ':.2%'),
        ('胜率', 'win_rate', '', ':.2%'),
        ('盈亏比', 'profit_loss_ratio', '', ':.2f'),
        ('总交易次数', 'total_trades', '', 'd'),
        ('盈利交易', 'profitable_trades', '', 'd'),
    ]

    improvements = []
    for name, key, prefix, fmt in metrics:
        orig_val = original[key]
        opt_val = optimized[key]

        if fmt == ':.2%':
            orig_str = f"{orig_val:.2%}"
            opt_str = f"{opt_val:.2%}"
            diff = opt_val - orig_val
            diff_str = f"{diff:+.2%}"
            if key in ['total_return', 'annual_return', 'sharpe_ratio', 'win_rate']:
                improvements.append(diff)
        elif fmt == ':,.2f':
            orig_str = f"¥{orig_val:,.2f}"
            opt_str = f"¥{opt_val:,.2f}"
            diff = opt_val - orig_val
            diff_str = f"¥{diff:+,.2f}"
            improvements.append(diff / 100000)
        else:
            orig_str = f"{orig_val}"
            opt_str = f"{opt_val}"
            diff = opt_val - orig_val
            diff_str = f"{diff:+}"
            if key in ['sharpe_ratio', 'profit_loss_ratio']:
                improvements.append(diff / 10 if orig_val != 0 else diff)

        better = "  ✓" if diff > 0 else ("  ✗" if diff < 0 and key not in ['max_drawdown'] else "")
        if key == 'max_drawdown':
            better = "  ✓" if diff < 0 else ("  ✗" if diff > 0 else "")

        report += f"{name:<20} {orig_str:<20} {opt_str:<20} {diff_str}{better}\n"

    # 计算整体提升
    positive_improvements = sum(1 for x in improvements if x > 0)
    total_improvements = len(improvements)

    report += f"""
{'-'*70}

三、优化效果评估
{'-'*70}

  整体表现: {positive_improvements}/{total_improvements} 项指标改善

"""

    # 详细分析
    report += "  1. 收益能力:\n"
    return_diff = optimized['total_return'] - original['total_return']
    if return_diff > 0:
        report += f"     优化版收益率提升 {return_diff:.2%}，表现更优\n"
    else:
        report += f"     优化版收益率下降 {abs(return_diff):.2%}\n"

    report += "\n  2. 风险控制:\n"
    dd_diff = original['max_drawdown'] - optimized['max_drawdown']
    if dd_diff > 0:
        report += f"     最大回撤减少 {dd_diff:.2%}，风险控制能力提升\n"
    else:
        report += f"     最大回撤增加 {abs(dd_diff):.2%}\n"

    report += "\n  3. 交易质量:\n"
    if optimized['win_rate'] > original['win_rate']:
        report += f"     胜率提升 {(optimized['win_rate']-original['win_rate']):.2%}\n"
    if optimized['sharpe_ratio'] > original['sharpe_ratio']:
        report += f"     夏普比率提升 {optimized['sharpe_ratio']-original['sharpe_ratio']:.2f}\n"

    # 优化点说明
    report += f"""
{'-'*70}

四、优化内容说明
{'-'*70}

  【原版策略】
    - 基础缠论买卖点识别
    - 固定仓位入场
    - 简单止损(5%)
    - 无止盈机制
    - 每根K线重新计算

  【优化策略】
    - 缓存机制，避免重复计算，提升执行效率
    - 动态仓位管理，根据信号强度调整仓位
    - 多重止损止盈:
      * 固定止损: 5%
      * 固定止盈: 15%
      * 移动止损: 3%(盈利5%后启用)
    - 趋势过滤，只在有利趋势中交易
    - 多重确认信号(中枢+MACD+分型)
    - 完整的持仓记录管理

{'='*70}

五、结论
{'='*70}
"""

    # 总体评价
    score = 0
    if optimized['total_return'] > original['total_return']:
        score += 1
    if optimized['sharpe_ratio'] > original['sharpe_ratio']:
        score += 1
    if abs(optimized['max_drawdown']) < abs(original['max_drawdown']):
        score += 1
    if optimized['win_rate'] > original['win_rate']:
        score += 1

    if score >= 3:
        conclusion = "优化策略表现显著优于原版策略，建议采用。"
    elif score >= 2:
        conclusion = "优化策略有一定改善，可根据实际需求选择。"
    else:
        conclusion = "优化策略效果不明显，建议进一步调整参数。"

    report += f"  {conclusion}\n"
    report += f"{'='*70}\n"

    return report


def save_report(report: str, filename: str = "optimization_report.txt"):
    """保存报告到文件"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    return filepath


def main():
    print("="*60)
    print("缠论策略优化报告生成器")
    print("="*60)

    import pandas as pd
    from strategies import ChanLunStrategy, OptimizedChanLunStrategy

    # 生成数据
    print("\n[1/3] 生成测试数据...")
    symbol = "MOCK001"
    df = generate_mock_data(symbol, days=500)
    print(f"  生成 {len(df)} 条K线数据")

    # 回测原版
    print("\n[2/3] 回测原版策略...")
    original_results = run_backtest(ChanLunStrategy, df, symbol)
    print(f"  收益率: {original_results['total_return']:.2%}")

    # 回测优化版
    print("\n[3/3] 回测优化策略...")
    optimized_results = run_backtest(OptimizedChanLunStrategy, df, symbol)
    print(f"  收益率: {optimized_results['total_return']:.2%}")

    # 生成报告
    print("\n生成优化报告...")
    report = generate_report(original_results, optimized_results)

    # 打印报告
    print(report)

    # 保存报告
    report_file = save_report(report)
    print(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
    input("\n按回车键退出...")
