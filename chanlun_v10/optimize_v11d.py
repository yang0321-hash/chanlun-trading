"""V11d 增量优化测试

逐个测试优化方向，每次只改一个参数，与 V11c baseline 对比。

优化方向:
1. 收紧 weekly_not_down 阈值 (0.95 → 0.97/0.98)
2. 入场增加 MA20 趋势确认 (price > MA20)
3. 收紧连续亏损暂停 (3笔→2笔)

用法: python optimize_v11d.py
"""

import sys
import json
import subprocess
import re
import copy
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
    """运行回测，返回指标字典"""
    result = subprocess.run(
        [PYTHON, "-m", "backtest.runner", "."],
        cwd=str(RUN_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        return None

    stdout = result.stdout.strip()
    start = stdout.find('{')
    end = stdout.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(stdout[start:end+1])
        except json.JSONDecodeError:
            print(f"  JSON parse error")
            return None
    return None


def backup_engine() -> str:
    """备份当前 signal_engine.py"""
    return ENGINE_PATH.read_text(encoding="utf-8")


def restore_engine(original: str):
    """恢复 signal_engine.py"""
    ENGINE_PATH.write_text(original, encoding="utf-8")


def apply_single_change(original: str, change_name: str) -> str:
    """在原始代码上应用单个改动"""

    code = original

    if change_name == "weekly_0.97":
        # 收紧 weekly_not_down: 0.95 → 0.97
        code = re.sub(
            r"(ma5\.iloc\[w_idx\] < ma10\.iloc\[w_idx\] \* )0\.95",
            r"\g<1>0.97",
            code
        )

    elif change_name == "weekly_0.98":
        # 收紧 weekly_not_down: 0.95 → 0.98
        code = re.sub(
            r"(ma5\.iloc\[w_idx\] < ma10\.iloc\[w_idx\] \* )0\.95",
            r"\g<1>0.98",
            code
        )

    elif change_name == "ma20_entry":
        # 入场增加 MA20 趋势确认: price > MA20 * 0.97
        # 在 "if not bi_buy.iloc[i]:" 之前插入 MA20 过滤
        old = "                if not bi_buy.iloc[i]:\n                    continue"
        new = """                if not bi_buy.iloc[i]:
                    continue

                # [v11d] MA20 趋势确认: 价格应接近 MA20
                if not pd.isna(ma20.iloc[i]) and price < ma20.iloc[i] * 0.97:
                    continue"""
        code = code.replace(old, new)

    elif change_name == "ma20_strict":
        # 更严格的 MA20 过滤: price > MA20
        old = "                if not bi_buy.iloc[i]:\n                    continue"
        new = """                if not bi_buy.iloc[i]:
                    continue

                # [v11d] 严格MA20过滤: 价格必须在MA20之上
                if not pd.isna(ma20.iloc[i]) and price < ma20.iloc[i]:
                    continue"""
        code = code.replace(old, new)

    elif change_name == "consec_loss_2":
        # 连续亏损暂停: 3笔→2笔
        code = re.sub(
            r"(if self\._consecutive_losses >= )3:",
            r"\g<1>2:",
            code
        )

    elif change_name == "consec_loss_pause_5":
        # 连续亏损暂停天数: 2天→5天
        code = re.sub(
            r"(self\._loss_pause_until = bar_idx \+ )2",
            r"\g<1>5",
            code
        )

    elif change_name == "min_hold_10":
        # 最短持仓: 7→10天
        code = re.sub(
            r"(self\.min_hold_before_sell\s*=\s*)\d+",
            r"\g<1>10",
            code
        )

    elif change_name == "sell_profit_8pct":
        # 顶背驰/2卖盈利门槛: 5%→8%
        code = re.sub(
            r"(and profit_pct > )0\.05\)",
            r"\g<1>0.08)",
            code
        )

    elif change_name == "big_loss_cooldown_60":
        # 大亏冷却: 30天→60天
        code = re.sub(
            r"(self\.big_loss_cooldown\s*=\s*)\d+",
            r"\g<1>60",
            code
        )

    return code


def main():
    print("=" * 80)
    print("V11d Incremental Optimization Tests")
    print("=" * 80)
    print(f"Engine: {ENGINE_PATH}")
    print(f"Python: {PYTHON}")
    print()

    # 定义所有测试
    tests = [
        ("baseline", "V11c baseline (no change)"),
        ("weekly_0.97", "Weekly filter: 0.95→0.97"),
        ("weekly_0.98", "Weekly filter: 0.95→0.98"),
        ("ma20_entry", "Entry: price > MA20*0.97"),
        ("ma20_strict", "Entry: price > MA20 (strict)"),
        ("consec_loss_2", "Consec loss pause: 3→2"),
        ("consec_loss_pause_5", "Consec loss pause days: 2→5"),
        ("min_hold_10", "Min hold before sell: 7→10"),
        ("sell_profit_8pct", "Sell profit threshold: 5%→8%"),
        ("big_loss_cooldown_60", "Big loss cooldown: 30→60"),
    ]

    original = backup_engine()
    results = []

    for change_name, description in tests:
        print(f"\n--- Testing: {description} ({change_name}) ---")
        print(f"  Applying change...", end=" ", flush=True)

        # Apply change
        modified = apply_single_change(original, change_name)
        ENGINE_PATH.write_text(modified, encoding="utf-8")
        print("done")

        # Run backtest
        print(f"  Running backtest...", end=" ", flush=True)
        data = run_backtest()

        if data is None:
            print("FAILED")
            continue

        r = TestResult(
            label=change_name,
            sharpe=data.get("sharpe", 0),
            annual_return=data.get("annual_return", 0),
            max_drawdown=data.get("max_drawdown", 0),
            calmar=data.get("calmar", 0),
            win_rate=data.get("win_rate", 0),
            trade_count=data.get("trade_count", 0),
            final_value=data.get("final_value", 0),
            changes=description,
        )
        results.append(r)
        print(f"Sharpe={r.sharpe:.3f} | Ann={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | Calmar={r.calmar:.2f} | "
              f"WR={r.win_rate*100:.1f}% | Trades={r.trade_count} | "
              f"Final={r.final_value/1e6:.2f}M")

    # Restore original
    restore_engine(original)
    print("\n\nRestored original signal_engine.py")

    # Sort by Sharpe
    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)

    # Find baseline
    baseline = next((r for r in results if r.label == "baseline"), None)

    print("\n\n" + "=" * 80)
    print("RESULTS RANKED BY SHARPE")
    print("=" * 80)

    if baseline:
        print(f"  Baseline: Sharpe={baseline.sharpe:.3f} | Ann={baseline.annual_return*100:.1f}% | "
              f"DD={baseline.max_drawdown*100:.1f}% | Calmar={baseline.calmar:.2f} | "
              f"Trades={baseline.trade_count}")

    print()
    for i, r in enumerate(sorted_results):
        delta = ""
        if baseline and r.label != "baseline":
            ds = r.sharpe - baseline.sharpe
            da = (r.annual_return - baseline.annual_return) * 100
            delta = f" (ΔS={ds:+.3f}, ΔA={da:+.1f}%)"
        marker = " ★" if baseline and r.sharpe > baseline.sharpe else ""
        print(f"  #{i+1} {r.label:25s} | Sharpe={r.sharpe:.3f} | "
              f"Ann={r.annual_return*100:.1f}% | DD={r.max_drawdown*100:.1f}% | "
              f"Calmar={r.calmar:.2f} | WR={r.win_rate*100:.1f}% | "
              f"Trades={r.trade_count} | Final={r.final_value/1e6:.2f}M{delta}{marker}")

    # Identify winners
    if baseline:
        winners = [r for r in results if r.label != "baseline" and r.sharpe > baseline.sharpe]
        if winners:
            print(f"\n\nIMPROVEMENTS FOUND: {len(winners)}")
            for r in sorted(winners, key=lambda r: r.sharpe, reverse=True):
                ds = r.sharpe - baseline.sharpe
                da = (r.annual_return - baseline.annual_return) * 100
                print(f"  {r.changes}: Sharpe +{ds:.3f}, Annual {da:+.1f}%")
        else:
            print("\n\nNo improvements over baseline.")


if __name__ == "__main__":
    main()
