"""V11d 30分钟参数优化 v2 — MACD 周期独立扫描

v1 问题: MACD 按 ×8 缩放到 96/208 太慢，导致信号极少。
v2 方案: MACD 周期独立扫描，不按日线比例缩放。

30分钟缠论的正确思路:
- 笔检测: bi_min_gap 保持小值(3~6)，因为30分钟级别波动更密集
- MACD: 不缩放，直接用 12/26 或更短周期
- MA/Volume MA: 缩放到匹配日线含义
- ATR: 缩放到匹配日线含义
- 百分比参数: 不变

用法: python optimize_30min_v2.py
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parent
MIN30_DIR = PROJECT_ROOT / "chanlun_system" / "artifacts"

from backtest_30min import load_min30_data, run_30min_backtest

# Import base engine and override key methods
from optimize_30min import SignalEngine30min


class SignalEngine30minV2(SignalEngine30min):
    """30分钟引擎 v2: MACD 周期可调，不强制 ×8"""

    def __init__(self, bi_min_gap=4, ema_fast=12, ema_slow=26, ema_signal=9,
                 ma_period=160, vol_ma_period=160, atr_period=112,
                 trailing_start=0.02, trailing_tight=0.025,
                 trailing_medium=0.06, trailing_wide=0.07,
                 weekly_threshold=0.97, time_stop_days=60, min_hold_days=7):
        # Override MACD params before super init
        self._override_ema_fast = ema_fast
        self._override_ema_slow = ema_slow
        self._override_ema_signal = ema_signal
        self._override_ma_period = ma_period
        self._override_vol_ma_period = vol_ma_period
        self._override_atr_period = atr_period

        super().__init__(
            bi_min_gap=bi_min_gap,
            trailing_start=trailing_start,
            trailing_tight=trailing_tight,
            trailing_medium=trailing_medium,
            trailing_wide=trailing_wide,
            weekly_threshold=weekly_threshold,
            time_stop_days=time_stop_days,
            min_hold_days=min_hold_days,
        )

        # Override the ×8 scaled MACD back to custom
        self.ema_fast = self._override_ema_fast
        self.ema_slow = self._override_ema_slow
        self.ema_signal = self._override_ema_signal
        self.ma_period = self._override_ma_period
        self.vol_ma_period = self._override_vol_ma_period
        self.atr_period = self._override_atr_period


@dataclass
class SweepResult:
    label: str
    sharpe: float
    annual_return: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trade_count: int
    final_value: float
    params: dict


def run_test(data_map, engine):
    metrics = run_30min_backtest(data_map, engine)
    return metrics


def main():
    print("=" * 80)
    print("V11d 30-Minute Parameter Optimization v2")
    print("Key change: MACD periods NOT ×8 scaled")
    print("=" * 80)

    # Load data
    print("\nLoading 30min data...")
    real_30min_codes = []
    for f in sorted(MIN30_DIR.glob("min30_*.csv")):
        df = pd.read_csv(f)
        dates = pd.to_datetime(df['datetime']).dt.date.unique()
        if len(df) / max(1, len(dates)) >= 4:
            real_30min_codes.append(f.stem.replace("min30_", ""))
    print(f"  {len(real_30min_codes)} stocks: {real_30min_codes}")
    data_map = load_min30_data(real_30min_codes)

    results = []

    # ===== Phase 1: bi_min_gap + MACD 组合 =====
    print("\n--- Phase 1: bi_min_gap × MACD period combo ---")
    macd_configs = [
        ("12/26", 12, 26, 9),         # 标准日线 MACD（不缩放）
        ("24/52", 24, 52, 18),         # ×2 缩放
        ("48/104", 48, 104, 36),       # ×4 缩放
        ("6/13", 6, 13, 5),           # 半缩放（更灵敏）
    ]

    for gap in [3, 4, 5, 6]:
        for mname, ef, es, esig in macd_configs:
            engine = SignalEngine30minV2(bi_min_gap=gap, ema_fast=ef, ema_slow=es, ema_signal=esig)
            m = run_test(data_map, engine)
            r = SweepResult(
                label=f"g{gap}_macd{mname}",
                sharpe=m.get("sharpe", 0),
                annual_return=m.get("annual_return", 0),
                max_drawdown=m.get("max_drawdown", 0),
                calmar=m.get("calmar", 0),
                win_rate=m.get("win_rate", 0),
                trade_count=m.get("trade_count", 0),
                final_value=m.get("final_value", 0),
                params={"gap": gap, "macd": mname},
            )
            results.append(r)
            print(f"  g{gap} macd{mname:8s}: S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
                  f"DD={r.max_drawdown*100:.1f}% | T={r.trade_count}")

    # ===== Phase 2: Top combos + trailing =====
    # Find top 3 by Sharpe
    sorted_p1 = sorted(results, key=lambda r: r.sharpe, reverse=True)
    top3 = sorted_p1[:3]
    print(f"\n--- Phase 2: Top 3 + trailing sweep ---")
    for t in top3:
        print(f"  Top: {t.label} (S={t.sharpe:.3f})")

    trailing_configs = [
        ("v11d", 0.015, 0.02, 0.05, 0.07),
        ("v11c", 0.02, 0.025, 0.06, 0.07),
        ("tight", 0.01, 0.015, 0.04, 0.06),
        ("loose", 0.03, 0.04, 0.08, 0.10),
    ]

    for t in top3:
        gap = t.params["gap"]
        mname = t.params["macd"]
        mc = next(c for c in macd_configs if c[0] == mname)

        for tname, ts, tt, tm, tw in trailing_configs:
            engine = SignalEngine30minV2(
                bi_min_gap=gap, ema_fast=mc[1], ema_slow=mc[2], ema_signal=mc[3],
                trailing_start=ts, trailing_tight=tt, trailing_medium=tm, trailing_wide=tw,
            )
            m = run_test(data_map, engine)
            r = SweepResult(
                label=f"g{gap}_m{mname}_{tname}",
                sharpe=m.get("sharpe", 0),
                annual_return=m.get("annual_return", 0),
                max_drawdown=m.get("max_drawdown", 0),
                calmar=m.get("calmar", 0),
                win_rate=m.get("win_rate", 0),
                trade_count=m.get("trade_count", 0),
                final_value=m.get("final_value", 0),
                params={"gap": gap, "macd": mname, "trailing": tname},
            )
            results.append(r)
            print(f"  g{gap} m{mname} {tname:6s}: S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
                  f"DD={r.max_drawdown*100:.1f}% | T={r.trade_count}")

    # ===== Final ranking =====
    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)

    print(f"\n\n{'='*80}")
    print("FINAL RANKING BY SHARPE")
    print(f"{'='*80}")

    for i, r in enumerate(sorted_results[:20]):
        print(f"  #{i+1:2d} {r.label:25s} | S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | C={r.calmar:.2f} | WR={r.win_rate*100:.1f}% | "
              f"T={r.trade_count}")

    # Compare with daily on same stocks
    print(f"\n--- Reference: Daily V11d on same 8 stocks ---")
    print(f"  Daily: Sharpe=1.43, Ann=6.7%, DD=-5.3%")

    # Save best
    if sorted_results:
        best = sorted_results[0]
        out = RUN_DIR / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "engine": "30min_v2",
            "params": best.params,
            "metrics": {
                "sharpe": best.sharpe,
                "annual_return": best.annual_return,
                "max_drawdown": best.max_drawdown,
                "calmar": best.calmar,
                "win_rate": best.win_rate,
                "trade_count": best.trade_count,
                "final_value": best.final_value,
            }
        }
        with open(out / "best_30min_v2_config.json", "w") as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
