"""三策略对比测试"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)


def generate_data(days=500):
    """生成测试数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    dates = [d for d in dates if d.weekday() < 5]

    price = 100.0
    prices = []
    for i in range(len(dates)):
        change = np.random.randn() * 0.02 + 0.0003
        price = price * (1 + change)
        hl = price * 0.02 * np.random.rand()
        open_p = price + np.random.randn() * 0.01
        close_p = price + np.random.randn() * 0.01
        prices.append({
            'datetime': dates[i],
            'open': round(open_p, 2),
            'high': round(max(open_p, close_p) + hl, 2),
            'low': round(min(open_p, close_p) - hl, 2),
            'close': round(close_p, 2),
            'volume': np.random.randint(1000000, 50000000)
        })

    df = pd.DataFrame(prices)
    df.set_index('datetime', inplace=True)
    return df


def run_strategy(strategy_class, df, name):
    """运行单个策略"""
    from backtest import BacktestEngine, BacktestConfig

    config = BacktestConfig(initial_capital=100000, commission=0.0003, slippage=0.0001)
    engine = BacktestEngine(config)
    engine.add_data("TEST", df)
    engine.set_strategy(strategy_class())

    print(f"\n运行 {name}...")
    results = engine.run()
    trades = len(engine.get_trades())

    return results, trades


def main():
    print("="*70)
    print("                  缠论策略三版本对比测试")
    print("="*70)

    from strategies import ChanLunStrategy, OptimizedChanLunStrategy, AdvancedChanLunStrategy

    # 生成数据
    print("\n[生成数据]")
    df = generate_data(500)
    print(f"  K线数量: {len(df)}")
    print(f"  时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    # 运行三个策略
    print("\n[运行回测]")
    orig, orig_trades = run_strategy(ChanLunStrategy, df, "原版策略")
    opt, opt_trades = run_strategy(OptimizedChanLunStrategy, df, "优化策略")
    adv, adv_trades = run_strategy(AdvancedChanLunStrategy, df, "高级策略")

    # 对比结果
    print("\n" + "="*70)
    print("                          绩效对比")
    print("="*70)
    print(f"{'指标':<20} {'原版':<15} {'优化版':<15} {'高级版':<15}")
    print("-"*70)

    metrics = [
        ('最终权益', 'final_equity', ':,.2f'),
        ('总收益率', 'total_return', ':.2%'),
        ('夏普比率', 'sharpe_ratio', ':.2f'),
        ('最大回撤', 'max_drawdown', ':.2%'),
        ('胜率', 'win_rate', ':.2%'),
        ('交易次数', 'total_trades', 'd'),
    ]

    for name, key, fmt in metrics:
        orig_val = orig[key]
        opt_val = opt[key]
        adv_val = adv[key]

        if fmt == ':.2%':
            orig_s = f"{orig_val:.2%}"
            opt_s = f"{opt_val:.2%}"
            adv_s = f"{adv_val:.2%}"
        elif fmt == ':,.2f':
            orig_s = f"¥{orig_val:,.2f}"
            opt_s = f"¥{opt_val:,.2f}"
            adv_s = f"¥{adv_val:,.2f}"
        else:  # 整数
            orig_s = f"{int(orig_val)}" if orig_val == int(orig_val) else f"{orig_val}"
            opt_s = f"{int(opt_val)}" if opt_val == int(opt_val) else f"{opt_val}"
            adv_s = f"{int(adv_val)}" if adv_val == int(adv_val) else f"{adv_val}"

        print(f"{name:<20} {orig_s:<15} {opt_s:<15} {adv_s:<15}")

    # 找出最佳
    print("\n" + "="*70)
    print("                          版本对比")
    print("="*70)

    best_return = max(orig['total_return'], opt['total_return'], adv['total_return'])
    best_sharpe = max(orig['sharpe_ratio'], opt['sharpe_ratio'], adv['sharpe_ratio'])
    best_dd = min(orig['max_drawdown'], opt['max_drawdown'], adv['max_drawdown'])

    print(f"\n最佳收益率: {'原版' if orig['total_return']==best_return else ('优化版' if opt['total_return']==best_return else '高级版')}")
    print(f"最佳夏普比率: {'原版' if orig['sharpe_ratio']==best_sharpe else ('优化版' if opt['sharpe_ratio']==best_sharpe else '高级版')}")
    print(f"最小回撤: {'原版' if orig['max_drawdown']==best_dd else ('优化版' if opt['max_drawdown']==best_dd else '高级版')}")

    # 结论
    print("\n" + "="*70)
    print("                          结论")
    print("="*70)

    scores = {
        '原版': (orig['total_return'], orig['sharpe_ratio'], -orig['max_drawdown']),
        '优化版': (opt['total_return'], opt['sharpe_ratio'], -opt['max_drawdown']),
        '高级版': (adv['total_return'], adv['sharpe_ratio'], -adv['max_drawdown']),
    }

    best = max(scores.items(), key=lambda x: (x[1][0], x[1][1]))

    print(f"\n推荐使用: {best[0]}")
    print(f"  - 收益率: {best[1][0]:.2%}")
    print(f"  - 夏普: {best[1][1]:.2f}")
    print(f"  - 回撤: {best[1][2]:.2%}")

    print("\n" + "="*70)

    # 保存结果
    with open('strategy_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("缠论策略对比报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'指标':<20} {'原版':<15} {'优化版':<15} {'高级版':<15}\n")
        f.write("-"*70 + "\n")
        for name, key, fmt in metrics:
            orig_val = orig[key]
            opt_val = opt[key]
            adv_val = adv[key]
            if fmt == ':.2%':
                f.write(f"{name:<20} {orig_val:.2%}{'':<12} {opt_val:.2%}{'':<12} {adv_val:.2%}\n")
            elif fmt == ':,.2f':
                f.write(f"{name:<20} ¥{orig_val:,.2f}{'':<7} ¥{opt_val:,.2f}{'':<7} ¥{adv_val:,.2f}\n")
            else:
                f.write(f"{name:<20} {int(orig_val)}{'':<12} {int(opt_val)}{'':<12} {int(adv_val)}\n")
        f.write(f"\n推荐: {best[0]}\n")

    print("结果已保存到: strategy_comparison.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出...")
