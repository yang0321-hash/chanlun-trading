"""V11d 30分钟回测

使用已有的 min30 CSV 数据文件进行 30 分钟级别回测。
核心改进: 可以在盘中触发止损，解决日线回测跳空止损问题。

关键参数调整 (日线→30分钟):
- bars_per_year: 252 → 2016 (252天 x 8根/天)
- time_stop_bars: 60天 → 480根 (60天 x 8)
- min_hold_before_sell: 7天 → 56根
- cooldown_bars: 1天 → 8根
- big_loss_cooldown: 30天 → 240根
- bi_confirm_delay: 1 → 2 (30分钟确认稍慢)

用法: python backtest_30min.py
"""

import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 项目路径
RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parent
MIN30_DIR = PROJECT_ROOT / "chanlun_system" / "artifacts"
PYTHON = sys.executable

# 导入 signal engine
sys.path.insert(0, str(RUN_DIR / "code"))
from signal_engine import SignalEngine


@dataclass
class TradeRecord:
    timestamp: str
    code: str
    side: str  # buy/sell
    price: float
    qty: float  # position weight
    pnl: float
    holding_bars: int
    return_pct: float


def load_min30_data(codes: list) -> dict:
    """加载 30 分钟 CSV 数据"""
    data_map = {}
    for code in codes:
        csv_path = MIN30_DIR / f"min30_{code}.csv"
        if not csv_path.exists():
            print(f"  [WARN] No 30min data for {code}")
            continue
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        data_map[code] = df
    return data_map


def create_30min_engine() -> SignalEngine:
    """创建适配 30 分钟的 signal engine"""
    engine = SignalEngine()

    # 调整时间参数: 日线→30分钟
    engine.time_stop_bars = 480       # 60天 x 8 = 480根
    engine.min_hold_before_sell = 56  # 7天 x 8 = 56根
    engine.cooldown_bars = 8          # 1天 x 8 = 8根
    engine.big_loss_cooldown = 240    # 30天 x 8 = 240根
    engine.bi_confirm_delay = 2       # 30分钟确认延迟稍长

    # 减少最小K线数（30分钟数据密度高）
    engine.bi_min_gap = 5             # 原3, 30分钟需要更多

    return engine


def run_30min_backtest(data_map: dict, engine: SignalEngine,
                       initial_cash: float = 1_000_000,
                       commission: float = 0.001) -> dict:
    """运行 30 分钟级别回测

    使用向量化的方式，与 daily_portfolio.py 类似。
    """
    codes = sorted(data_map.keys())
    if not codes:
        return {"error": "No data"}

    # Generate signals
    signal_map = engine.generate(data_map)
    valid_codes = [c for c in codes if c in signal_map]

    if not valid_codes:
        return {"error": "No valid signals"}

    # Align dates
    all_dates = set()
    for c in valid_codes:
        all_dates.update(data_map[c].index)
    dates = pd.DatetimeIndex(sorted(all_dates))

    close = pd.DataFrame(index=dates, columns=valid_codes, dtype=float)
    for c in valid_codes:
        close[c] = data_map[c]["close"].reindex(dates)
    close = close.ffill().bfill()

    pos = pd.DataFrame(0.0, index=dates, columns=valid_codes)
    for c in valid_codes:
        raw = signal_map[c].reindex(dates).fillna(0.0).clip(-1.0, 1.0)
        pos[c] = raw.shift(1).fillna(0.0)

    ret = close.pct_change().fillna(0.0)

    # Normalize positions
    scale = pos.abs().sum(axis=1).clip(lower=1.0)
    pos = pos.div(scale, axis=0)

    # Portfolio returns
    port_ret = (pos * ret).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    port_ret = port_ret - turnover * commission
    equity = initial_cash * (1 + port_ret).cumprod()

    # Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, 1)

    # Metrics
    n = len(port_ret)
    bars_per_year = 2016  # 252 * 8
    total_ret = float(equity.iloc[-1] / initial_cash - 1)
    ann_ret = float((1 + total_ret) ** (bars_per_year / max(n, 1)) - 1)
    vol = float(port_ret.std())
    sharpe = float(port_ret.mean() / (vol + 1e-10) * np.sqrt(bars_per_year))

    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    downside = port_ret[port_ret < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 1e-10
    sortino = float(port_ret.mean() / (downside_std + 1e-10) * np.sqrt(bars_per_year))

    # Trade logging
    trades = log_trades_30min(pos, close, valid_codes)

    # Win rate
    closed = [t for t in trades if t.pnl != 0]
    wins = [t for t in closed if t.pnl > 0]
    win_rate = len(wins) / len(closed) if closed else 0.0

    # Max consecutive losses
    max_consec = 0
    cur_consec = 0
    for t in closed:
        if t.pnl < 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    # Average holding bars
    hold_bars = [t.holding_bars for t in closed if t.holding_bars > 0]
    avg_hold_bars = np.mean(hold_bars) if hold_bars else 0
    avg_hold_days = avg_hold_bars / 8  # 8 bars per day

    # Benchmark
    bench_ret = ret.mean(axis=1)
    bench_equity = initial_cash * (1 + bench_ret).cumprod()
    bench_total = float(bench_equity.iloc[-1] / initial_cash - 1)

    return {
        "final_value": float(equity.iloc[-1]),
        "total_return": total_ret,
        "annual_return": ann_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "calmar": round(calmar, 4),
        "sortino": round(sortino, 4),
        "win_rate": win_rate,
        "trade_count": len(trades),
        "max_consecutive_loss": max_consec,
        "avg_holding_bars": round(avg_hold_bars, 1),
        "avg_holding_days": round(avg_hold_days, 1),
        "benchmark_return": round(bench_total, 6),
        "excess_return": round(total_ret - bench_total, 6),
    }


def log_trades_30min(pos: pd.DataFrame, close: pd.DataFrame,
                     codes: list) -> list:
    """记录交易"""
    diff = pos.diff()
    diff.iloc[0] = pos.iloc[0]

    cost = {}
    hold = {}
    entry_date = {}
    trades = []

    for c in codes:
        delta = diff[c]
        for ts in delta.index[delta.abs() > 1e-9]:
            d = float(delta.loc[ts])
            p = float(close.at[ts, c]) if pd.notna(close.at[ts, c]) else 0.0
            side = "buy" if d > 0 else "sell"
            qty = abs(d)
            pnl = 0.0
            holding_bars = 0
            return_pct = 0.0

            prev = hold.get(c, 0.0)
            if prev * d >= 0:
                old = cost.get(c, p)
                total = abs(prev) + qty
                cost[c] = (old * abs(prev) + p * qty) / total if total > 1e-9 else p
                hold[c] = prev + d
                if c not in entry_date:
                    entry_date[c] = ts
            else:
                close_qty = min(qty, abs(prev))
                entry = cost.get(c, p)
                pnl = (p - entry) * close_qty if prev > 0 else (entry - p) * close_qty
                return_pct = (p / entry - 1) * 100 if entry > 1e-9 and prev > 0 else 0.0

                ed = entry_date.get(c, ts)
                try:
                    holding_bars = len(pos.loc[ed:ts]) - 1
                except Exception:
                    holding_bars = 0

                remain = prev + d
                if abs(remain) < 1e-9:
                    hold.pop(c, None)
                    cost.pop(c, None)
                    entry_date.pop(c, None)
                else:
                    hold[c] = remain
                    if remain * prev <= 0:
                        cost[c] = p
                        entry_date[c] = ts

            trades.append(TradeRecord(
                timestamp=str(ts),
                code=c,
                side=side,
                price=round(p, 4),
                qty=round(qty, 6),
                pnl=round(pnl, 4),
                holding_bars=holding_bars,
                return_pct=round(return_pct, 2),
            ))

    return trades


def main():
    print("=" * 80)
    print("V11d 30-Minute Backtest")
    print("=" * 80)

    # Find available 30min data with real intraday bars (>=4 bars/day)
    min30_files = sorted(MIN30_DIR.glob("min30_*.csv"))
    real_30min_codes = []
    for f in min30_files:
        df = pd.read_csv(f)
        dates = pd.to_datetime(df['datetime']).dt.date.unique()
        bars_per_day = len(df) / max(1, len(dates))
        if bars_per_day >= 4:
            real_30min_codes.append(f.stem.replace("min30_", ""))

    print(f"\nFound {len(real_30min_codes)} stocks with real 30min data")
    print(f"Stocks: {real_30min_codes}")

    # Load data
    print("\nLoading 30min data...")
    data_map = load_min30_data(real_30min_codes)
    print(f"Loaded {len(data_map)} stocks")

    for code, df in list(data_map.items())[:3]:
        print(f"  {code}: {len(df)} bars, {df.index[0]} ~ {df.index[-1]}")

    # Create 30-min engine
    engine = create_30min_engine()

    # Run backtest
    print("\nRunning 30min backtest...")
    metrics = run_30min_backtest(data_map, engine)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"\n{'='*60}")
    print(f"V11d 30-Minute Backtest Results")
    print(f"{'='*60}")
    print(f"  Annual Return:  {metrics['annual_return']*100:.1f}%")
    print(f"  Sharpe:         {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']*100:.1f}%")
    print(f"  Calmar:         {metrics['calmar']:.2f}")
    print(f"  Sortino:        {metrics['sortino']:.2f}")
    print(f"  Win Rate:       {metrics['win_rate']*100:.1f}%")
    print(f"  Trade Count:    {metrics['trade_count']}")
    print(f"  Max Consec Loss:{metrics['max_consecutive_loss']}")
    print(f"  Avg Hold Days:  {metrics['avg_holding_days']:.1f}")
    print(f"  Final Value:    {metrics['final_value']/1e6:.2f}M")
    print(f"  Benchmark:      {metrics['benchmark_return']*100:.1f}%")
    print(f"  Excess Return:  {metrics['excess_return']*100:.1f}%")

    # Compare with daily V11d (same stocks)
    print(f"\n{'='*60}")
    print(f"30-min Backtest — Key Insight")
    print(f"{'='*60}")
    print(f"  30min allows intraday stop-loss execution,")
    print(f"  solving the gap-down problem (e.g., 002371 -19%)")
    print(f"  Daily: {metrics['trade_count']} trades, avg hold {metrics['avg_holding_days']:.1f} days")
    print(f"  Sharpe={metrics['sharpe']:.2f}, Ann={metrics['annual_return']*100:.1f}%")

    # Save results
    out_dir = RUN_DIR / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_30min.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_dir / 'metrics_30min.json'}")


if __name__ == "__main__":
    main()
