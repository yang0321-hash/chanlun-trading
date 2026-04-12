"""V11 参数扫描脚本

扫描关键参数组合，按 Sharpe 排名输出结果。
用法: python param_sweep.py
"""

import sys
import json
import subprocess
import time
import itertools
from pathlib import Path
from dataclasses import dataclass, replace

# 项目根目录
RUN_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# 参数扫描空间
SWEEP_PARAMS = {
    "trailing_tight":  [0.03, 0.04, 0.05],
    "trailing_medium": [0.05, 0.06, 0.07],
    "trailing_wide":   [0.07, 0.08, 0.10],
    "trailing_start":  [0.02, 0.03, 0.04],
    "min_hold_before_sell": [7, 10, 14],
    "time_stop_bars":  [60, 80, 100],
}


@dataclass
class SweepResult:
    params: dict
    annual_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trade_count: int
    final_value: float


def load_signal_template() -> str:
    """加载当前 signal_engine.py 作为模板"""
    return (RUN_DIR / "code" / "signal_engine.py").read_text(encoding="utf-8")


def apply_params(template: str, params: dict) -> str:
    """在模板中替换参数值"""
    code = template
    param_map = {
        "trailing_tight": "self.trailing_tight",
        "trailing_medium": "self.trailing_medium",
        "trailing_wide": "self.trailing_wide",
        "trailing_start": "self.trailing_start",
        "min_hold_before_sell": "self.min_hold_before_sell",
        "time_stop_bars": "self.time_stop_bars",
    }
    for key, attr in param_map.items():
        if key in params:
            value = params[key]
            # 匹配 self.xxx = 0.xx 或 self.xxx = xx
            import re
            pattern = rf"({attr}\s*=\s*)(\d+\.?\d*)"
            replacement = rf"\g<1>{value}"
            code = re.sub(pattern, replacement, code)
    return code


def run_backtest(code_content: str) -> dict:
    """替换 signal_engine.py 并运行回测"""
    engine_path = RUN_DIR / "code" / "signal_engine.py"

    # 保存当前内容
    original = engine_path.read_text(encoding="utf-8")

    try:
        engine_path.write_text(code_content, encoding="utf-8")
        result = subprocess.run(
            [PYTHON, "-m", "backtest.runner", "."],
            cwd=str(RUN_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return None

        # 解析 JSON 输出
        stdout = result.stdout.strip()
        start = stdout.find('{')
        end = stdout.rfind('}')
        if start >= 0 and end > start:
            return json.loads(stdout[start:end+1])
        return None
    except Exception:
        return None
    finally:
        # 恢复原始内容
        engine_path.write_text(original, encoding="utf-8")


def main():
    print("=" * 80)
    print("V11 Parameter Sweep")
    print("=" * 80)

    template = load_signal_template()

    # 策略1: 单参数扫描（每个参数独立变化，其他保持 V11b 默认值）
    print("\n--- Phase 1: Single parameter sweep ---")

    defaults = {
        "trailing_tight": 0.04,
        "trailing_medium": 0.05,
        "trailing_wide": 0.07,
        "trailing_start": 0.03,
        "min_hold_before_sell": 7,
        "time_stop_bars": 60,
    }

    results = []

    for param_name, values in SWEEP_PARAMS.items():
        print(f"\n  Sweeping {param_name}: {values}")
        for value in values:
            params = {**defaults, param_name: value}
            label = f"{param_name}={value}"
            print(f"    Running {label}...", end=" ", flush=True)

            code = apply_params(template, params)
            data = run_backtest(code)

            if data is None:
                print("FAILED")
                continue

            r = SweepResult(
                params=params,
                annual_return=data.get("annual_return", 0),
                sharpe=data.get("sharpe", 0),
                max_drawdown=data.get("max_drawdown", 0),
                calmar=data.get("calmar", 0),
                win_rate=data.get("win_rate", 0),
                trade_count=data.get("trade_count", 0),
                final_value=data.get("final_value", 0),
            )
            results.append(r)
            print(f"Sharpe={r.sharpe:.3f} Ann={r.annual_return*100:.1f}% DD={r.max_drawdown*100:.1f}% Trades={r.trade_count}")

    # Phase 2: Top combos
    print("\n\n--- Phase 2: Best combo from top individual params ---")

    # 找每个参数的最佳值
    best_params = {}
    for param_name, values in SWEEP_PARAMS.items():
        param_results = [r for r in results if list(r.params.keys()) == list(defaults.keys())]
        # 找该参数在所有结果中 Sharpe 最高的值
        best_sharpe = -999
        best_val = defaults[param_name]
        for r in results:
            if abs(r.params[param_name] - defaults[param_name]) < 1e-9:
                continue  # 跳过默认值
            # 只看只改变了该参数的结果
            same_as_default = True
            for k, v in defaults.items():
                if k == param_name:
                    continue
                if abs(r.params[k] - v) > 1e-9:
                    same_as_default = False
                    break
            if same_as_default and r.sharpe > best_sharpe:
                best_sharpe = r.sharpe
                best_val = r.params[param_name]

        if best_sharpe > 0:
            best_params[param_name] = best_val
        else:
            best_params[param_name] = defaults[param_name]

    print(f"  Best combo: {best_params}")
    code = apply_params(template, best_params)
    data = run_backtest(code)
    if data:
        print(f"  Result: Sharpe={data['sharpe']:.3f} Ann={data['annual_return']*100:.1f}% "
              f"DD={data['max_drawdown']*100:.1f}% Calmar={data['calmar']:.2f} "
              f"Trades={data['trade_count']} Final={data['final_value']/1e6:.2f}M")
        results.append(SweepResult(
            params=best_params,
            annual_return=data["annual_return"],
            sharpe=data["sharpe"],
            max_drawdown=data["max_drawdown"],
            calmar=data["calmar"],
            win_rate=data["win_rate"],
            trade_count=data["trade_count"],
            final_value=data["final_value"],
        ))

    # 按 Sharpe 排名输出 Top 10
    print("\n\n" + "=" * 80)
    print("TOP 10 by Sharpe")
    print("=" * 80)
    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        changes = {k: v for k, v in r.params.items() if abs(v - defaults[k]) > 1e-9}
        print(f"#{i+1} Sharpe={r.sharpe:.3f} | Ann={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | Calmar={r.calmar:.2f} | "
              f"WR={r.win_rate*100:.1f}% | Trades={r.trade_count} | "
              f"Final={r.final_value/1e6:.2f}M | Changes={changes}")

    print("\n" + "=" * 80)
    print("TOP 5 by Annual Return")
    print("=" * 80)
    by_return = sorted(results, key=lambda r: r.annual_return, reverse=True)
    for i, r in enumerate(by_return[:5]):
        changes = {k: v for k, v in r.params.items() if abs(v - defaults[k]) > 1e-9}
        print(f"#{i+1} Ann={r.annual_return*100:.1f}% | Sharpe={r.sharpe:.3f} | "
              f"DD={r.max_drawdown*100:.1f}% | Calmar={r.calmar:.2f} | "
              f"WR={r.win_rate*100:.1f}% | Trades={r.trade_count} | "
              f"Final={r.final_value/1e6:.2f}M | Changes={changes}")


if __name__ == "__main__":
    main()
