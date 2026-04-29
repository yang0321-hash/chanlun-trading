"""
Backtest comparison: sz002600 vs sz300936 with original vs unified strategies

Strategies compared:
1. ChanLunStrategy (original chan_strategy)
2. OptimalChanLunStrategy (unified/optimal)
3. IntegratedChanLunStrategy (integrated)

Stocks: sz002600, sz300936
Data: TDX JSON files, last 500 bars
"""

import warnings
warnings.filterwarnings('ignore')

import json
import sys
import pandas as pd
import numpy as np

from loguru import logger
logger.remove()

sys.path.insert(0, '.')


def load_data(filepath):
    """Load TDX JSON data and prepare DataFrame"""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data.get('data', data.get('records', [])))

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if 'volume' not in df.columns and 'amount' in df.columns:
        df['volume'] = df['amount']

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    df = df.sort_index()
    return df


from backtest.engine import BacktestEngine, BacktestConfig
from strategies.chan_strategy import ChanLunStrategy
from strategies.optimal_chanlun_strategy import OptimalChanLunStrategy
from strategies.integrated_chanlun_strategy import IntegratedChanLunStrategy


# === Stock data ===
stocks = {
    'sz002600': 'test_output/sz002600.day.json',
    'sz300936': 'test_output/sz300936.day.json',
}

# === Strategy configurations ===
strategy_configs = [
    ('Original-ChanLun', ChanLunStrategy, {}),
    ('Unified-Optimal', OptimalChanLunStrategy, {}),
    ('Integrated', IntegratedChanLunStrategy, {}),
]


def run_single_backtest(strategy_name, strategy_cls, strategy_params, symbol, df):
    """Run a single backtest and return results"""
    try:
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        engine.add_data(symbol, df)
        strategy = strategy_cls(**strategy_params)
        engine.set_strategy(strategy)
        results = engine.run()

        trades = results.get('trades', [])
        trade_details = []
        for t in trades:
            trade_details.append({
                'type': t.signal_type.value,
                'price': round(t.price, 2),
                'quantity': t.quantity,
                'reason': t.reason[:60] if t.reason else '',
            })

        return {
            'total_return': results.get('total_return', 0),
            'annual_return': results.get('annual_return', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'win_rate': results.get('win_rate', 0),
            'profit_loss_ratio': results.get('profit_loss_ratio', 0),
            'total_trades': results.get('total_trades', 0),
            'final_equity': results.get('final_equity', 0),
            'trades': trade_details,
        }
    except Exception as e:
        import traceback
        print(f"ERROR {strategy_name} {symbol}: {e}")
        traceback.print_exc()
        return {'error': str(e)}


# === Main execution ===
print("=" * 80)
print("BACKTEST COMPARISON: sz002600 vs sz300936")
print("Strategies: Original ChanLun | Unified Optimal | Integrated")
print("=" * 80)

# Load data first
stock_data = {}
for sym, fp in stocks.items():
    df = load_data(fp)
    total_bars = len(df)
    if len(df) > 500:
        df = df.iloc[-500:]
    stock_data[sym] = df
    print(f"\n[Data] {sym}: {total_bars} total bars, using last {len(df)} bars")
    print(f"       Range: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"       Price: {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
    buy_hold = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
    print(f"       Buy&Hold: {buy_hold:.2%}")

# Run all backtests
all_results = {}
for strategy_name, strategy_cls, strategy_params in strategy_configs:
    print(f"\n{'=' * 80}")
    print(f"Strategy: {strategy_name}")
    print(f"{'=' * 80}")
    strategy_results = {}

    for sym, df in stock_data.items():
        print(f"\n--- {sym} ---")
        result = run_single_backtest(strategy_name, strategy_cls, strategy_params, sym, df)
        strategy_results[sym] = result

        if 'error' in result:
            print(f"  ERROR: {result['error'][:80]}")
        else:
            print(f"  Total Return:   {result['total_return']:.2%}")
            print(f"  Annual Return:  {result['annual_return']:.2%}")
            print(f"  Sharpe Ratio:   {result['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:   {result['max_drawdown']:.2%}")
            print(f"  Win Rate:       {result['win_rate']:.2%}")
            print(f"  Profit/Loss:    {result['profit_loss_ratio']:.2f}")
            print(f"  Total Trades:   {result['total_trades']}")
            print(f"  Final Equity:   {result['final_equity']:,.2f}")

            trades = result.get('trades', [])
            if trades:
                print(f"\n  Trade Details ({len(trades)} trades):")
                for i, t in enumerate(trades):
                    print(f"    #{i+1} {t['type'].upper():4s} @ {t['price']:8.2f} x {t['quantity']:>6d} | {t['reason']}")

    all_results[strategy_name] = strategy_results

# === Summary comparison table ===
print("\n\n")
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

# Header
print(f"\n{'Strategy':<22} {'Symbol':<10} {'Return':>8} {'Annual':>8} {'Sharpe':>7} {'Drawdown':>9} {'WinRate':>8} {'P/L':>6} {'Trades':>7} {'Equity':>12}")
print("-" * 100)

for strategy_name, strategy_results in all_results.items():
    for sym, result in strategy_results.items():
        if 'error' not in result:
            print(
                f"{strategy_name:<22} {sym:<10} "
                f"{result['total_return']:>7.2%} "
                f"{result['annual_return']:>7.2%} "
                f"{result['sharpe_ratio']:>7.2f} "
                f"{result['max_drawdown']:>8.2%} "
                f"{result['win_rate']:>7.2%} "
                f"{result['profit_loss_ratio']:>6.2f} "
                f"{result['total_trades']:>7d} "
                f"{result['final_equity']:>11,.0f}"
            )
        else:
            print(f"{strategy_name:<22} {sym:<10} ERROR: {result['error'][:50]}")
    print()

# Cross-stock comparison
print("\n" + "=" * 80)
print("STOCK-LEVEL ANALYSIS")
print("=" * 80)
for sym in stock_data.keys():
    print(f"\n{sym}:")
    for strategy_name, strategy_results in all_results.items():
        result = strategy_results.get(sym, {})
        if 'error' not in result:
            print(
                f"  {strategy_name:<22} "
                f"Return={result['total_return']:>7.2%} "
                f"Sharpe={result['sharpe_ratio']:>6.2f} "
                f"DD={result['max_drawdown']:>7.2%} "
                f"Win={result['win_rate']:>6.2%} "
                f"Trades={result['total_trades']:>3d}"
            )
        else:
            print(f"  {strategy_name:<22} ERROR")

# Best strategy per stock
print("\n" + "=" * 80)
print("BEST STRATEGY PER STOCK (by total return)")
print("=" * 80)
for sym in stock_data.keys():
    best_strategy = None
    best_return = -999
    for strategy_name, strategy_results in all_results.items():
        result = strategy_results.get(sym, {})
        if 'error' not in result and result['total_return'] > best_return:
            best_return = result['total_return']
            best_strategy = strategy_name
    if best_strategy:
        print(f"  {sym}: {best_strategy} (return={best_return:.2%})")

print("\nDone.")
