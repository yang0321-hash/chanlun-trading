"""缠论策略优化报告 - 详细版"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


def generate_realistic_data(days=500):
    """生成更真实的股价走势"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    dates = [d for d in dates if d.weekday() < 5]

    # 模拟更真实的价格走势（带趋势和波动）
    price = 100.0
    trend = 0.0003  # 每日趋势
    volatility = 0.02  # 波动率

    prices = []
    for i in range(len(dates)):
        # 随机游走 + 趋势
        change = np.random.randn() * volatility + trend
        price = price * (1 + change)

        # 生成OHLC
        high_low = price * volatility * np.random.rand()
        open_p = price + np.random.randn() * volatility * 0.5
        close_p = price + np.random.randn() * volatility * 0.5
        high_p = max(open_p, close_p) + high_low
        low_p = min(open_p, close_p) - high_low

        prices.append({
            'datetime': dates[i],
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': np.random.randint(1000000, 50000000)
        })

    df = pd.DataFrame(prices)
    df.set_index('datetime', inplace=True)
    return df


def run_backtest_with_details(strategy_class, df, symbol):
    """运行回测并返回详细信息"""
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

    results = engine.run()

    # 获取交易详情
    trades = engine.get_trades()
    signals = engine.get_signals()

    return {
        'results': results,
        'trades_count': len(trades),
        'signals_count': len(signals),
        'trades': trades,
        'signals': signals
    }


def print_report():
    """打印优化报告"""
    print("="*70)
    print("                    缠论交易策略优化报告")
    print("="*70)

    from strategies import ChanLunStrategy, OptimizedChanLunStrategy

    # 生成数据
    print("\n[数据准备]")
    symbol = "TEST001"
    df = generate_realistic_data(500)
    print(f"  股票代码: {symbol}")
    print(f"  数据周期: {len(df)} 个交易日")
    print(f"  时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  价格区间: ¥{df['low'].min():.2f} ~ ¥{df['high'].max():.2f}")

    # 回测
    print("\n[回测执行]")
    print("  运行原版策略...")
    orig_data = run_backtest_with_details(ChanLunStrategy, df, symbol)
    orig = orig_data['results']

    print("  运行优化策略...")
    opt_data = run_backtest_with_details(OptimizedChanLunStrategy, df, symbol)
    opt = opt_data['results']

    # 对比表
    print("\n" + "="*70)
    print("                          绩效对比")
    print("="*70)
    print(f"{'指标':<25} {'原版策略':<20} {'优化策略':<20} {'差异'}")
    print("-"*70)

    def fmt_val(val, is_pct=True, is_currency=False):
        if is_currency:
            return f"¥{val:,.2f}"
        elif is_pct:
            return f"{val:.2%}"
        return f"{val:.2f}"

    metrics = [
        ('初始资金', 'initial_capital', 100000, False, True),
        ('最终权益', 'final_equity', 0, False, True),
        ('总收益率', 'total_return', 0, True, False),
        ('年化收益率', 'annual_return', 0, True, False),
        ('夏普比率', 'sharpe_ratio', 0, False, False),
        ('最大回撤', 'max_drawdown', 0, True, False),
        ('胜率', 'win_rate', 0, True, False),
        ('盈亏比', 'profit_loss_ratio', 0, False, False),
        ('总交易次数', 'total_trades', 0, False, False),
        ('盈利交易', 'profitable_trades', 0, False, False),
    ]

    for name, key, default, is_pct, is_currency in metrics:
        orig_val = orig.get(key, default)
        opt_val = opt.get(key, default)

        if name == '初始资金':
            orig_str = fmt_val(orig_val, is_currency=True)
            opt_str = fmt_val(opt_val, is_currency=True)
            diff_str = "-"
        else:
            orig_str = fmt_val(orig_val, is_pct, is_currency)
            opt_str = fmt_val(opt_val, is_pct, is_currency)

            if is_pct:
                diff = opt_val - orig_val
                diff_str = f"{diff:+.2%}"
            elif is_currency:
                diff = opt_val - orig_val
                diff_str = f"¥{diff:+,.2f}"
            else:
                diff = opt_val - orig_val
                diff_str = f"{diff:+}"

        # 判断优劣
        better = ""
        if key not in ['initial_capital', 'max_drawdown']:
            if diff > 0:
                better = "  ✓"
            elif diff < 0:
                better = "  ✗"
        elif key == 'max_drawdiff':
            if diff < 0:
                better = "  ✓"
            elif diff > 0:
                better = "  ✗"

        print(f"{name:<25} {orig_str:<20} {opt_str:<20} {diff_str}{better}")

    # 交易统计
    print("\n" + "="*70)
    print("                          交易统计")
    print("="*70)
    print(f"{'项目':<25} {'原版策略':<20} {'优化策略':<20}")
    print("-"*70)
    print(f"{'信号数量':<25} {orig_data['signals_count']:<20} {opt_data['signals_count']:<20}")
    print(f"{'成交数量':<25} {orig_data['trades_count']:<20} {opt_data['trades_count']:<20}")

    # 分析
    print("\n" + "="*70)
    print("                          分析总结")
    print("="*70)

    # 原版策略分析
    print("\n[原版策略]")
    if orig_data['signals_count'] == 0:
        print("  ❌ 未产生任何交易信号")
        print("  原因: 买卖点判断条件过于严格")
        print("  建议: 放宽信号条件或调整参数")
    else:
        print(f"  产生 {orig_data['signals_count']} 个信号")
        print(f"  成交 {orig_data['trades_count']} 笔交易")

    # 优化版策略分析
    print("\n[优化策略]")
    if opt_data['trades_count'] == 0:
        print("  ❌ 未产生任何交易")
    else:
        print(f"  产生 {opt_data['signals_count']} 个信号")
        print(f"  成交 {opt_data['trades_count']} 笔交易")
        print(f"  胜率: {opt['win_rate']:.2%}")

        if opt['total_return'] > 0:
            print("  ✓ 获利策略")
        else:
            print("  ✗ 亏损策略 - 需要进一步优化")

    # 优化建议
    print("\n" + "="*70)
    print("                          优化建议")
    print("="*70)
    print("""
  1. 调整买卖点判断阈值
     - 降低第一类买卖点的确认要求
     - 增加第二、三类买卖点的权重

  2. 改进止损止盈策略
     - 根据ATR动态调整止损位
     - 分批止盈而非一次性退出

  3. 增加过滤条件
     - 只在特定趋势方向交易
     - 结合成交量确认信号

  4. 参数优化
     - min_stroke_bars: 当前5，可尝试3
     - stop_loss_pct: 当前5%，可尝试3%-8%
     - take_profit_pct: 当前15%，可尝试10%-20%
    """)

    print("="*70)

    # 保存报告
    save_report_to_file(orig, opt, orig_data, opt_data)

    return orig, opt


def save_report_to_file(orig, opt, orig_data, opt_data):
    """保存报告到文件"""
    content = f"""
缠论交易策略优化报告
{'='*70}

一、回测概况
{'-'*70}
数据周期: 500个交易日
初始资金: ¥100,000

二、绩效对比
{'-'*70}
指标              原版策略          优化策略          差异
{'-'*70}
最终权益          ¥{orig['final_equity']:,.2f}        ¥{opt['final_equity']:,.2f}        ¥{opt['final_equity']-orig['final_equity']:+,.2f}
总收益率          {orig['total_return']:.2%}           {opt['total_return']:.2%}           {opt['total_return']-orig['total_return']:+.2%}
年化收益率        {orig['annual_return']:.2%}           {opt['annual_return']:.2%}           {opt['annual_return']-orig['annual_return']:+.2%}
夏普比率          {orig['sharpe_ratio']:.2f}            {opt['sharpe_ratio']:.2f}            {opt['sharpe_ratio']-orig['sharpe_ratio']:+.2f}
最大回撤          {orig['max_drawdown']:.2%}           {opt['max_drawdown']:.2%}           {opt['max_drawdown']-orig['max_drawdown']:+.2%}
胜率              {orig['win_rate']:.2%}           {opt['win_rate']:.2%}           {opt['win_rate']-orig['win_rate']:+.2%}
盈亏比            {orig['profit_loss_ratio']:.2f}            {opt['profit_loss_ratio']:.2f}            {opt['profit_loss_ratio']-orig['profit_loss_ratio']:+.2f}
总交易次数        {orig['total_trades']}               {opt['total_trades']}               {opt['total_trades']-orig['total_trades']:+d}

三、交易统计
{'-'*70}
                  原版策略          优化策略
信号数量          {orig_data['signals_count']}               {opt_data['signals_count']}
成交数量          {orig_data['trades_count']}               {opt_data['trades_count']}

四、优化内容
{'-'*70}
1. 缓存机制 - 避免重复计算缠论要素
2. 止损止盈 - 5%止损、15%止盈、3%移动止损
3. 仓位管理 - 根据信号强度动态调整
4. 趋势过滤 - 只在有利趋势中交易
5. 多重确认 - 中枢+MACD+分型综合判断

{'='*70}
"""

    with open('optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n报告已保存到: optimization_report.txt")


if __name__ == "__main__":
    try:
        print_report()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出...")
