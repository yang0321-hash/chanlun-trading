"""V11d 30分钟参数优化 v3 — loose trailing 精细扫描

基于 v2 发现: g3+macd48/104+loose 是最优基础配置。
现在精细扫描 loose trailing 参数，寻找最优组合。

用法: python optimize_30min_v3.py
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parent
MIN30_DIR = PROJECT_ROOT / "chanlun_system" / "artifacts"

from backtest_30min import load_min30_data, run_30min_backtest
from optimize_30min_v2 import SignalEngine30minV2, SweepResult


def main():
    print("=" * 80)
    print("V11d 30-Minute Optimization v3 — Fine-tune Loose Trailing")
    print("Base: bi_min_gap=3, MACD 48/104/36, loose trailing")
    print("=" * 80)

    # Load data
    print("\nLoading 30min data...")
    real_codes = []
    for f in sorted(MIN30_DIR.glob("min30_*.csv")):
        df = pd.read_csv(f)
        dates = pd.to_datetime(df['datetime']).dt.date.unique()
        if len(df) / max(1, len(dates)) >= 4:
            real_codes.append(f.stem.replace("min30_", ""))
    data_map = load_min30_data(real_codes)
    print(f"  {len(real_codes)} stocks")

    results = []

    # Phase 1: trailing_start × trailing_tight grid
    print("\n--- Phase 1: trailing_start × trailing_tight grid ---")
    starts = [0.02, 0.025, 0.03, 0.035, 0.04]
    tights = [0.025, 0.03, 0.035, 0.04, 0.05]

    for ts in starts:
        for tt in tights:
            engine = SignalEngine30minV2(
                bi_min_gap=3, ema_fast=48, ema_slow=104, ema_signal=36,
                trailing_start=ts, trailing_tight=tt,
                trailing_medium=0.08, trailing_wide=0.10,
            )
            m = run_30min_backtest(data_map, engine)
            r = SweepResult(
                label=f"s{ts}_t{tt}",
                sharpe=m.get("sharpe", 0), annual_return=m.get("annual_return", 0),
                max_drawdown=m.get("max_drawdown", 0), calmar=m.get("calmar", 0),
                win_rate=m.get("win_rate", 0), trade_count=m.get("trade_count", 0),
                final_value=m.get("final_value", 0),
                params={"start": ts, "tight": tt},
            )
            results.append(r)
            if r.sharpe > 0.8:
                marker = " *"
            elif r.sharpe > 0.6:
                marker = " +"
            else:
                marker = ""
            print(f"  s={ts:.3f} t={tt:.3f}: S={r.sharpe:.3f} A={r.annual_return*100:.1f}% "
                  f"DD={r.max_drawdown*100:.1f}% T={r.trade_count}{marker}")

    # Phase 2: medium × wide grid (using best start+tight)
    sorted_p1 = sorted(results, key=lambda r: r.sharpe, reverse=True)
    best_start = sorted_p1[0].params["start"]
    best_tight = sorted_p1[0].params["tight"]
    print(f"\n--- Phase 2: medium × wide (best start={best_start}, tight={best_tight}) ---")

    mediums = [0.05, 0.06, 0.07, 0.08, 0.10]
    wides = [0.07, 0.08, 0.10, 0.12, 0.15]

    for tm in mediums:
        for tw in wides:
            engine = SignalEngine30minV2(
                bi_min_gap=3, ema_fast=48, ema_slow=104, ema_signal=36,
                trailing_start=best_start, trailing_tight=best_tight,
                trailing_medium=tm, trailing_wide=tw,
            )
            m = run_30min_backtest(data_map, engine)
            r = SweepResult(
                label=f"m{tm}_w{tw}",
                sharpe=m.get("sharpe", 0), annual_return=m.get("annual_return", 0),
                max_drawdown=m.get("max_drawdown", 0), calmar=m.get("calmar", 0),
                win_rate=m.get("win_rate", 0), trade_count=m.get("trade_count", 0),
                final_value=m.get("final_value", 0),
                params={"medium": tm, "wide": tw},
            )
            results.append(r)
            if r.sharpe > 1.0:
                marker = " **"
            elif r.sharpe > 0.8:
                marker = " *"
            else:
                marker = ""
            print(f"  m={tm:.2f} w={tw:.2f}: S={r.sharpe:.3f} A={r.annual_return*100:.1f}% "
                  f"DD={r.max_drawdown*100:.1f}% C={r.calmar:.2f} T={r.trade_count}{marker}")

    # Phase 3: Weekly threshold
    print(f"\n--- Phase 3: weekly threshold sweep ---")
    best_overall = sorted(
        [r for r in results if "medium" in r.params],
        key=lambda r: r.sharpe, reverse=True
    )[0]

    for wt in [0.95, 0.96, 0.97, 0.98]:
        engine = SignalEngine30minV2(
            bi_min_gap=3, ema_fast=48, ema_slow=104, ema_signal=36,
            trailing_start=best_start, trailing_tight=best_tight,
            trailing_medium=best_overall.params["medium"],
            trailing_wide=best_overall.params["wide"],
            weekly_threshold=wt,
        )
        m = run_30min_backtest(data_map, engine)
        r = SweepResult(
            label=f"wt{wt}",
            sharpe=m.get("sharpe", 0), annual_return=m.get("annual_return", 0),
            max_drawdown=m.get("max_drawdown", 0), calmar=m.get("calmar", 0),
            win_rate=m.get("win_rate", 0), trade_count=m.get("trade_count", 0),
            final_value=m.get("final_value", 0),
            params={"weekly": wt},
        )
        results.append(r)
        print(f"  wt={wt:.2f}: S={r.sharpe:.3f} A={r.annual_return*100:.1f}% "
              f"DD={r.max_drawdown*100:.1f}% C={r.calmar:.2f} T={r.trade_count}")

    # Final ranking
    sorted_all = sorted(results, key=lambda r: r.sharpe, reverse=True)
    print(f"\n\n{'='*80}")
    print("TOP 20 BY SHARPE")
    print(f"{'='*80}")
    for i, r in enumerate(sorted_all[:20]):
        print(f"  #{i+1:2d} {r.label:20s} | S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | C={r.calmar:.2f} | WR={r.win_rate*100:.1f}% | T={r.trade_count}")

    print(f"\n--- Reference ---")
    print(f"  Daily V11d (8 stocks): Sharpe=1.43, Ann=6.7%")
    print(f"  30min v2 best:         Sharpe=0.88, Ann=8.8%")

    # Save
    if sorted_all:
        best = sorted_all[0]
        out = RUN_DIR / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "best_30min_v3_config.json", "w") as f:
            json.dump({
                "engine": "30min_v3",
                "config": {
                    "bi_min_gap": 3,
                    "ema_fast": 48, "ema_slow": 104, "ema_signal": 36,
                    "trailing_start": best_start,
                    "trailing_tight": best_tight,
                    "trailing_medium": best_overall.params.get("medium", 0.08),
                    "trailing_wide": best_overall.params.get("wide", 0.10),
                },
                "metrics": {
                    "sharpe": best.sharpe,
                    "annual_return": best.annual_return,
                    "max_drawdown": best.max_drawdown,
                    "calmar": best.calmar,
                    "win_rate": best.win_rate,
                    "trade_count": best.trade_count,
                    "final_value": best.final_value,
                }
            }, f, indent=2)


if __name__ == "__main__":
    main()
