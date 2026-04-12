"""V11d Fine-tune: weekly_not_down threshold + combo tests

精细扫描 weekly_not_down 阈值，并测试与 V11c 参数的组合。

用法: python optimize_weekly_fine.py
"""

import sys
import json
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass

RUN_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
ENGINE_PATH = RUN_DIR / "code" / "signal_engine.py"


@dataclass
class TestResult:
    label: str
    sharpe: float
    annual_return: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trade_count: int
    final_value: float
    changes: str


def run_backtest() -> dict:
    result = subprocess.run(
        [PYTHON, "-m", "backtest.runner", "."],
        cwd=str(RUN_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}")
        return None
    stdout = result.stdout.strip()
    start = stdout.find('{')
    end = stdout.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(stdout[start:end+1])
        except json.JSONDecodeError:
            return None
    return None


def apply_weekly_threshold(original: str, threshold: float) -> str:
    """替换 weekly_not_down 阈值"""
    code = re.sub(
        r"(ma5\.iloc\[w_idx\] < ma10\.iloc\[w_idx\] \* )0\.\d+",
        rf"\g<1>{threshold}",
        original
    )
    return code


def apply_combo(original: str, weekly_threshold: float, trailing_start: float,
                trailing_tight: float, trailing_medium: float) -> str:
    """应用组合参数"""
    code = original
    # Weekly threshold
    code = re.sub(
        r"(ma5\.iloc\[w_idx\] < ma10\.iloc\[w_idx\] \* )0\.\d+",
        rf"\g<1>{weekly_threshold}",
        code
    )
    # Trailing params
    code = re.sub(
        r"(self\.trailing_start\s*=\s*)\d+\.?\d*",
        rf"\g<1>{trailing_start}",
        code
    )
    code = re.sub(
        r"(self\.trailing_tight\s*=\s*)\d+\.?\d*",
        rf"\g<1>{trailing_tight}",
        code
    )
    code = re.sub(
        r"(self\.trailing_medium\s*=\s*)\d+\.?\d*",
        rf"\g<1>{trailing_medium}",
        code
    )
    return code


def main():
    print("=" * 80)
    print("V11d Fine-tune: weekly_not_down threshold")
    print("=" * 80)

    original = ENGINE_PATH.read_text(encoding="utf-8")
    results = []

    # Phase 1: Fine-tune weekly threshold
    print("\n--- Phase 1: Fine-tune weekly_not_down threshold ---")
    thresholds = [0.955, 0.960, 0.965, 0.970, 0.972, 0.975]

    for t in thresholds:
        label = f"weekly_{t}"
        print(f"  {label}...", end=" ", flush=True)

        modified = apply_weekly_threshold(original, t)
        ENGINE_PATH.write_text(modified, encoding="utf-8")

        data = run_backtest()
        if data is None:
            print("FAILED")
            continue

        r = TestResult(
            label=label,
            sharpe=data.get("sharpe", 0),
            annual_return=data.get("annual_return", 0),
            max_drawdown=data.get("max_drawdown", 0),
            calmar=data.get("calmar", 0),
            win_rate=data.get("win_rate", 0),
            trade_count=data.get("trade_count", 0),
            final_value=data.get("final_value", 0),
            changes=f"weekly_not_down threshold={t}",
        )
        results.append(r)
        print(f"S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | DD={r.max_drawdown*100:.1f}% | "
              f"C={r.calmar:.2f} | T={r.trade_count} | F={r.final_value/1e6:.1f}M")

    # Phase 2: Best weekly + V11c trailing params combos
    print("\n--- Phase 2: Best weekly + trailing param combos ---")

    # Use top 2 weekly thresholds from Phase 1
    sorted_by_sharpe = sorted(results, key=lambda r: r.sharpe, reverse=True)
    best_thresholds = [float(r.label.split('_')[1]) for r in sorted_by_sharpe[:2]]

    # Test with different trailing params
    trailing_configs = [
        ("v11c", 0.02, 0.025, 0.06),      # V11c current
        ("v10", 0.03, 0.03, 0.05),         # V10 original
        ("tighter", 0.015, 0.02, 0.05),    # Even tighter
        ("looser", 0.03, 0.035, 0.07),     # Looser
    ]

    for wt in best_thresholds:
        for tname, ts, tt, tm in trailing_configs:
            label = f"w{wt}_{tname}"
            print(f"  {label}...", end=" ", flush=True)

            modified = apply_combo(original, wt, ts, tt, tm)
            ENGINE_PATH.write_text(modified, encoding="utf-8")

            data = run_backtest()
            if data is None:
                print("FAILED")
                continue

            r = TestResult(
                label=label,
                sharpe=data.get("sharpe", 0),
                annual_return=data.get("annual_return", 0),
                max_drawdown=data.get("max_drawdown", 0),
                calmar=data.get("calmar", 0),
                win_rate=data.get("win_rate", 0),
                trade_count=data.get("trade_count", 0),
                final_value=data.get("final_value", 0),
                changes=f"weekly={wt}, start={ts}, tight={tt}, medium={tm}",
            )
            results.append(r)
            print(f"S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | DD={r.max_drawdown*100:.1f}% | "
                  f"C={r.calmar:.2f} | T={r.trade_count} | F={r.final_value/1e6:.1f}M")

    # Restore
    ENGINE_PATH.write_text(original, encoding="utf-8")
    print("\nRestored original signal_engine.py")

    # Final ranking
    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)

    print("\n\n" + "=" * 80)
    print("FINAL RANKING BY SHARPE")
    print("=" * 80)

    for i, r in enumerate(sorted_results[:15]):
        print(f"  #{i+1} {r.label:25s} | S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | C={r.calmar:.2f} | WR={r.win_rate*100:.1f}% | "
              f"T={r.trade_count} | F={r.final_value/1e6:.1f}M")
        print(f"       {r.changes}")


if __name__ == "__main__":
    main()
