"""V11d Walk-Forward Validation

将 2020-2026 分为训练期(2020-2023)和验证期(2024-2026)，
分别运行回测，确认 V11d 不是过拟合。

用法: python walk_forward_v11d.py
"""

import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

RUN_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
CONFIG_PATH = RUN_DIR / "config.json"


@dataclass
class WFResult:
    period: str
    sharpe: float
    annual_return: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trade_count: int
    final_value: float
    start: str
    end: str
    years: float


def run_backtest(config: dict) -> dict:
    """写入临时 config 并运行回测"""
    # 保存原始 config
    original = CONFIG_PATH.read_text(encoding="utf-8")

    try:
        CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [PYTHON, "-m", "backtest.runner", "."],
            cwd=str(RUN_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[:300]}")
            return None

        stdout = result.stdout.strip()
        start = stdout.find('{')
        end = stdout.rfind('}')
        if start >= 0 and end > start:
            return json.loads(stdout[start:end+1])
        return None
    finally:
        # 恢复原始 config
        CONFIG_PATH.write_text(original, encoding="utf-8")


def calc_years(start: str, end: str) -> float:
    """计算年数"""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return (e - s).days / 365.25


def main():
    print("=" * 80)
    print("V11d Walk-Forward Validation")
    print("=" * 80)

    # Load base config
    base_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    # Define periods
    periods = [
        ("Full (2020-2026)", "2020-01-01", "2026-04-10"),
        ("Train (2020-2023)", "2020-01-01", "2023-12-31"),
        ("Test (2024-2026)", "2024-01-01", "2026-04-10"),
        # Extra splits for stability check
        ("2020-2021", "2020-01-01", "2021-12-31"),
        ("2022-2023", "2022-01-01", "2023-12-31"),
        ("2024-2025", "2024-01-01", "2025-12-31"),
        ("2026 only", "2026-01-01", "2026-04-10"),
    ]

    results = []

    for label, start, end in periods:
        years = calc_years(start, end)
        print(f"\n--- {label} ({start} ~ {end}, {years:.1f}年) ---")
        print(f"  Running...", end=" ", flush=True)

        config = {**base_config, "start_date": start, "end_date": end}
        data = run_backtest(config)

        if data is None:
            print("FAILED")
            continue

        r = WFResult(
            period=label,
            sharpe=data.get("sharpe", 0),
            annual_return=data.get("annual_return", 0),
            max_drawdown=data.get("max_drawdown", 0),
            calmar=data.get("calmar", 0),
            win_rate=data.get("win_rate", 0),
            trade_count=data.get("trade_count", 0),
            final_value=data.get("final_value", 0),
            start=start,
            end=end,
            years=years,
        )
        results.append(r)
        print(f"Sharpe={r.sharpe:.3f} | Ann={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | Calmar={r.calmar:.2f} | "
              f"WR={r.win_rate*100:.1f}% | Trades={r.trade_count}")

    # Analysis
    print("\n\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 80)

    # Find train and test results
    train = next((r for r in results if "Train" in r.period), None)
    test = next((r for r in results if "Test" in r.period), None)

    if train and test:
        print(f"\n  Train (2020-2023): Sharpe={train.sharpe:.3f}, Ann={train.annual_return*100:.1f}%, "
              f"DD={train.max_drawdown*100:.1f}%, Trades={train.trade_count}")
        print(f"  Test  (2024-2026): Sharpe={test.sharpe:.3f}, Ann={test.annual_return*100:.1f}%, "
              f"DD={test.max_drawdown*100:.1f}%, Trades={test.trade_count}")

        sharpe_decay = (train.sharpe - test.sharpe) / train.sharpe * 100 if train.sharpe > 0 else 0
        print(f"\n  Sharpe decay: {sharpe_decay:+.1f}%")

        if sharpe_decay > 30:
            print("  WARNING: >30% decay suggests overfitting")
        elif sharpe_decay > 15:
            print("  CAUTION: 15-30% decay, mild overfitting concern")
        elif sharpe_decay > 0:
            print("  OK: <15% decay, acceptable")
        else:
            print("  EXCELLENT: Test Sharpe >= Train Sharpe!")

    # Year-by-year breakdown
    print(f"\n\n  {'Period':20s} | {'Sharpe':>8s} | {'Ann':>8s} | {'DD':>8s} | {'Calmar':>8s} | {'WR':>6s} | {'Trades':>8s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
    for r in results:
        print(f"  {r.period:20s} | {r.sharpe:8.3f} | {r.annual_return*100:7.1f}% | "
              f"{r.max_drawdown*100:7.1f}% | {r.calmar:8.2f} | {r.win_rate*100:5.1f}% | {r.trade_count:8d}")

    # Stability metrics
    if len(results) >= 4:
        sharpes = [r.sharpe for r in results[3:]]  # sub-periods only
        anns = [r.annual_return * 100 for r in results[3:]]
        avg_sharpe = sum(sharpes) / len(sharpes)
        min_sharpe = min(sharpes)
        print(f"\n  Sub-period stability:")
        print(f"    Avg Sharpe: {avg_sharpe:.3f}")
        print(f"    Min Sharpe: {min_sharpe:.3f}")
        print(f"    Sharpe range: {min(sharpes):.3f} ~ {max(sharpes):.3f}")
        print(f"    Avg Annual: {sum(anns)/len(anns):.1f}%")
        print(f"    Min Annual: {min(anns):.1f}%")


if __name__ == "__main__":
    main()
