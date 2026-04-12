#!/usr/bin/env python3
"""
缠论v4策略系统性参数优化

策略：
1. 先做单维度扫描，每次只改一个参数，找到每个参数的最优值
2. 再做组合验证
3. 所有回测使用纯日线模式（禁用30分钟确认）

基线：年化25.72%, Sharpe 2.81, 最大回撤-3.35%, 胜率74.5%, 366笔交易
"""

import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

# === 配置 ===
SIGNAL_ENGINE_PATH = Path("/workspace/chanlun_system/code/signal_engine.py")
RUN_DIR = Path("/workspace/chanlun_system")
TUSHARE_TOKEN = "445af3e7113dd4984a0ac217c32686ec6321161eac11a435529bc07d"
RESULTS_PATH = Path("/workspace/chanlun_system/optimization_results.md")

# 基线参数（v4默认值）
BASELINE_PARAMS = {
    "risk_per_trade": 0.03,
    "max_positions": 5,
    "max_drawdown_pct": 0.15,
    "cooldown_bars": 3,
    "time_stop_bars": 45,
    "min_hold_before_sell": 7,
    "max_stop_pct": 0.25,
    "min_position": 0.10,
    "max_position": 0.25,
    "base_position": 0.12,
    "trailing_start": 0.08,
    "trailing_distance": 0.04,
    "profit_add_threshold": 0.05,
    "profit_add_ratio": 0.50,
}

# 参数搜索空间
SEARCH_SPACE = {
    "base_position": [0.10, 0.12, 0.15, 0.18],
    "max_position": [0.20, 0.25, 0.30, 0.35],
    "trailing_start": [0.06, 0.08, 0.10],
    "trailing_distance": [0.03, 0.04, 0.05],
    "time_stop_bars": [30, 45, 60],
    "cooldown_bars": [1, 3, 5],
}

# 原始signal_engine.py内容（会读取一次）
ORIGINAL_CONTENT = None


def read_original():
    global ORIGINAL_CONTENT
    ORIGINAL_CONTENT = SIGNAL_ENGINE_PATH.read_text(encoding="utf-8")


def restore_original():
    """恢复原始signal_engine.py"""
    SIGNAL_ENGINE_PATH.write_text(ORIGINAL_CONTENT, encoding="utf-8")


def apply_params(params: dict):
    """将参数写入signal_engine.py，同时禁用30分钟确认"""
    content = ORIGINAL_CONTENT

    for key, value in params.items():
        if isinstance(value, float):
            pattern = rf"(self\.{key}\s*=\s*)[\d.]+"
            replacement = rf"\g<1>{value}"
            content = re.sub(pattern, replacement, content)
        elif isinstance(value, int):
            pattern = rf"(self\.{key}\s*=\s*)\d+"
            replacement = rf"\g<1>{value}"
            content = re.sub(pattern, replacement, content)

    # 禁用30分钟确认 - 替换 _compute_min30_confirmation 调用
    content = content.replace(
        "min30_confirmed = self._compute_min30_confirmation(code, df)",
        "min30_confirmed = None  # 禁用30分钟确认，纯日线模式"
    )

    SIGNAL_ENGINE_PATH.write_text(content, encoding="utf-8")


def run_backtest(timeout=120) -> dict:
    """执行回测，返回结果字典"""
    env = {"TUSHARE_TOKEN": TUSHARE_TOKEN}
    # 继承当前环境
    import os
    env.update({k: v for k, v in os.environ.items() if k not in env})

    for attempt in range(2):
        try:
            result = subprocess.run(
                ["python3", "-m", "backtest.runner", "."],
                cwd=str(RUN_DIR),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if result.returncode != 0:
                print(f"  回测失败 (attempt {attempt+1}): {result.stderr[:200]}")
                if attempt == 0:
                    time.sleep(2)
                    continue
                return None

            # 解析stdout JSON - 可能是多行格式化的JSON
            stdout = result.stdout.strip()
            
            # 尝试1: 直接解析整个stdout
            try:
                data = json.loads(stdout)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
            
            # 尝试2: 找到第一个 { 和最后一个 } 之间的内容
            start = stdout.find('{')
            end = stdout.rfind('}')
            if start >= 0 and end > start:
                try:
                    data = json.loads(stdout[start:end+1])
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
            
            # 尝试3: 逐行找JSON行
            lines = stdout.split('\n')
            json_buffer = ""
            in_json = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('{'):
                    in_json = True
                    json_buffer = stripped
                elif in_json:
                    json_buffer += stripped
                if in_json and stripped.endswith('}'):
                    try:
                        data = json.loads(json_buffer)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        pass
                    in_json = False
                    json_buffer = ""

            print(f"  无法解析回测输出: {stdout[:300]}")
            return None

        except subprocess.TimeoutExpired:
            print(f"  回测超时 (attempt {attempt+1})")
            if attempt == 0:
                time.sleep(2)
                continue
            return None
        except Exception as e:
            print(f"  回测异常 (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(2)
                continue
            return None

    return None


def extract_metrics(result: dict) -> dict:
    """从回测结果提取关键指标"""
    if result is None or 'error' in result:
        return {
            "annual_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "win_rate": None,
            "trade_count": None,
            "calmar": None,
            "sortino": None,
        }
    return {
        "annual_return": result.get("annual_return"),
        "sharpe": result.get("sharpe"),
        "max_drawdown": result.get("max_drawdown"),
        "win_rate": result.get("win_rate"),
        "trade_count": result.get("trade_count"),
        "calmar": result.get("calmar"),
        "sortino": result.get("sortino"),
        "total_return": result.get("total_return"),
    }


def score_metrics(metrics: dict) -> float:
    """综合评分函数：年化收益*0.3 + Sharpe*0.3 + (1+max_drawdown)*0.2 + win_rate*0.2"""
    if metrics["annual_return"] is None:
        return -999
    return (
        metrics["annual_return"] * 0.3
        + (metrics["sharpe"] or 0) * 0.3
        + (1 + (metrics["max_drawdown"] or -1)) * 0.2
        + (metrics["win_rate"] or 0) * 0.2
    )


def run_single_scan():
    """单维度扫描：每次只改一个参数"""
    print("=" * 60)
    print("Phase 1: 单维度扫描")
    print("=" * 60)

    # 先跑基线
    print("\n--- 基线测试 ---")
    apply_params(BASELINE_PARAMS)
    result = run_backtest()
    baseline_metrics = extract_metrics(result)
    baseline_score = score_metrics(baseline_metrics)
    print(f"  基线: 年化={baseline_metrics['annual_return']:.4f}, "
          f"Sharpe={baseline_metrics['sharpe']:.4f}, "
          f"最大回撤={baseline_metrics['max_drawdown']:.4f}, "
          f"胜率={baseline_metrics['win_rate']:.4f}, "
          f"交易数={baseline_metrics['trade_count']}, "
          f"得分={baseline_score:.4f}")

    all_results = []
    best_params_per_dim = {}

    for param_name, values in SEARCH_SPACE.items():
        print(f"\n--- 扫描参数: {param_name} ---")
        dim_results = []

        for value in values:
            test_params = dict(BASELINE_PARAMS)
            test_params[param_name] = value

            print(f"  {param_name}={value} ... ", end="", flush=True)
            apply_params(test_params)
            t0 = time.time()
            result = run_backtest()
            elapsed = time.time() - t0
            metrics = extract_metrics(result)
            s = score_metrics(metrics)

            if metrics["annual_return"] is not None:
                print(f"年化={metrics['annual_return']:.4f}, "
                      f"Sharpe={metrics['sharpe']:.4f}, "
                      f"最大回撤={metrics['max_drawdown']:.4f}, "
                      f"胜率={metrics['win_rate']:.4f}, "
                      f"交易={metrics['trade_count']}, "
                      f"得分={s:.4f} ({elapsed:.1f}s)")
            else:
                print(f"失败 ({elapsed:.1f}s)")

            dim_results.append({
                "params": test_params,
                "metrics": metrics,
                "score": s,
                "changed": {param_name: value},
            })
            all_results.append({
                "params": test_params,
                "metrics": metrics,
                "score": s,
                "phase": "single_scan",
                "changed": {param_name: value},
            })

        # 找这个维度最优的值
        valid_results = [r for r in dim_results if r["metrics"]["annual_return"] is not None]
        if valid_results:
            best = max(valid_results, key=lambda r: r["score"])
            best_params_per_dim[param_name] = best["changed"][param_name]
            print(f"  >> {param_name} 最优值: {best['changed'][param_name]} "
                  f"(得分={best['score']:.4f}, 基线={baseline_score:.4f})")

    return all_results, best_params_per_dim, baseline_metrics, baseline_score


def run_combined_test(best_params_per_dim, baseline_metrics, baseline_score):
    """组合验证：将各维度最优参数组合测试"""
    print("\n" + "=" * 60)
    print("Phase 2: 组合验证")
    print("=" * 60)

    # 组合1：所有维度最优值
    combined_params = dict(BASELINE_PARAMS)
    for param_name, best_value in best_params_per_dim.items():
        combined_params[param_name] = best_value

    print(f"\n组合参数: {best_params_per_dim}")
    print(f"完整参数: {combined_params}")

    apply_params(combined_params)
    result = run_backtest()
    metrics = extract_metrics(result)
    s = score_metrics(metrics)

    combined_result = {
        "params": combined_params,
        "metrics": metrics,
        "score": s,
        "phase": "combined",
        "changed": best_params_per_dim,
    }

    if metrics["annual_return"] is not None:
        print(f"\n组合结果: 年化={metrics['annual_return']:.4f}, "
              f"Sharpe={metrics['sharpe']:.4f}, "
              f"最大回撤={metrics['max_drawdown']:.4f}, "
              f"胜率={metrics['win_rate']:.4f}, "
              f"交易={metrics['trade_count']}, "
              f"得分={s:.4f}")
    else:
        print("\n组合回测失败!")

    # 组合2：尝试小幅调整（如果组合1不如预期，尝试用Sharpe作为唯一排序）
    # 先看组合1是否优于基线
    results = [combined_result]

    if s <= baseline_score:
        print("\n组合1不优于基线，尝试保守组合...")
        # 只用显著改善的参数
        conservative_params = dict(BASELINE_PARAMS)
        for param_name, best_value in best_params_per_dim.items():
            # 只改比基线好的参数
            if best_value != BASELINE_PARAMS.get(param_name):
                # 对比
                test_p = dict(BASELINE_PARAMS)
                test_p[param_name] = best_value
                apply_params(test_p)
                r = run_backtest()
                m = extract_metrics(r)
                if m["annual_return"] is not None and score_metrics(m) > baseline_score:
                    conservative_params[param_name] = best_value
                    print(f"  保留 {param_name}={best_value}")

        if conservative_params != dict(BASELINE_PARAMS):
            print(f"\n保守组合: {conservative_params}")
            apply_params(conservative_params)
            result = run_backtest()
            metrics = extract_metrics(result)
            s2 = score_metrics(metrics)
            results.append({
                "params": conservative_params,
                "metrics": metrics,
                "score": s2,
                "phase": "combined_conservative",
                "changed": {k: v for k, v in conservative_params.items()
                            if v != BASELINE_PARAMS.get(k)},
            })
            if metrics["annual_return"] is not None:
                print(f"保守组合结果: 年化={metrics['annual_return']:.4f}, "
                      f"Sharpe={metrics['sharpe']:.4f}, "
                      f"最大回撤={metrics['max_drawdown']:.4f}, "
                      f"胜率={metrics['win_rate']:.4f}, "
                      f"交易={metrics['trade_count']}, "
                      f"得分={s2:.4f}")

    # 组合3：聚焦高Sharpe - 尝试更大仓位+更紧止损
    print("\n尝试高Sharpe组合（大仓位+紧止损）...")
    high_sharpe_params = dict(BASELINE_PARAMS)
    high_sharpe_params["base_position"] = 0.15
    high_sharpe_params["max_position"] = 0.30
    high_sharpe_params["trailing_start"] = 0.06
    high_sharpe_params["trailing_distance"] = 0.03
    high_sharpe_params["time_stop_bars"] = 30
    high_sharpe_params["cooldown_bars"] = 1

    apply_params(high_sharpe_params)
    result = run_backtest()
    metrics = extract_metrics(result)
    s3 = score_metrics(metrics)
    results.append({
        "params": high_sharpe_params,
        "metrics": metrics,
        "score": s3,
        "phase": "high_sharpe",
        "changed": {k: v for k, v in high_sharpe_params.items() if v != BASELINE_PARAMS.get(k)},
    })
    if metrics["annual_return"] is not None:
        print(f"高Sharpe组合: 年化={metrics['annual_return']:.4f}, "
              f"Sharpe={metrics['sharpe']:.4f}, "
              f"最大回撤={metrics['max_drawdown']:.4f}, "
              f"胜率={metrics['win_rate']:.4f}, "
              f"交易={metrics['trade_count']}, "
              f"得分={s3:.4f}")

    # 组合4：稳健型 - 小仓位+宽止损+长持仓
    print("\n尝试稳健组合（小仓位+宽止损+长持仓）...")
    stable_params = dict(BASELINE_PARAMS)
    stable_params["base_position"] = 0.10
    stable_params["max_position"] = 0.20
    stable_params["trailing_start"] = 0.10
    stable_params["trailing_distance"] = 0.05
    stable_params["time_stop_bars"] = 60
    stable_params["cooldown_bars"] = 5

    apply_params(stable_params)
    result = run_backtest()
    metrics = extract_metrics(result)
    s4 = score_metrics(metrics)
    results.append({
        "params": stable_params,
        "metrics": metrics,
        "score": s4,
        "phase": "stable",
        "changed": {k: v for k, v in stable_params.items() if v != BASELINE_PARAMS.get(k)},
    })
    if metrics["annual_return"] is not None:
        print(f"稳健组合: 年化={metrics['annual_return']:.4f}, "
              f"Sharpe={metrics['sharpe']:.4f}, "
              f"最大回撤={metrics['max_drawdown']:.4f}, "
              f"胜率={metrics['win_rate']:.4f}, "
              f"交易={metrics['trade_count']}, "
              f"得分={s4:.4f}")

    return results


def generate_report(single_results, combined_results, best_params_per_dim,
                    baseline_metrics, baseline_score):
    """生成优化报告"""
    # 所有结果合并排序
    all_results = single_results + combined_results
    valid_results = [r for r in all_results if r["metrics"]["annual_return"] is not None]
    valid_results.sort(key=lambda r: r["score"], reverse=True)

    lines = []
    lines.append("# 缠论v4策略参数优化报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 基线指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 年化收益 | {baseline_metrics['annual_return']*100:.2f}% |")
    lines.append(f"| Sharpe | {baseline_metrics['sharpe']:.4f} |")
    lines.append(f"| 最大回撤 | {baseline_metrics['max_drawdown']*100:.2f}% |")
    lines.append(f"| 胜率 | {baseline_metrics['win_rate']*100:.1f}% |")
    lines.append(f"| 交易数 | {baseline_metrics['trade_count']} |")
    lines.append(f"| 综合得分 | {baseline_score:.4f} |")
    lines.append("")

    lines.append("## 单维度扫描结果\n")
    lines.append("### 各参数最优值\n")
    lines.append("| 参数 | 基线值 | 最优值 |")
    lines.append("|------|--------|--------|")
    for param_name, best_value in best_params_per_dim.items():
        lines.append(f"| {param_name} | {BASELINE_PARAMS[param_name]} | {best_value} |")
    lines.append("")

    lines.append("### 单维度扫描详情\n")
    for param_name in SEARCH_SPACE:
        lines.append(f"#### {param_name}\n")
        lines.append("| 值 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 交易数 | 得分 |")
        lines.append("|----|----------|--------|----------|------|--------|------|")
        for r in single_results:
            if r["changed"].get(param_name) is not None:
                m = r["metrics"]
                val = r["changed"][param_name]
                lines.append(f"| {val} | {m['annual_return']*100:.2f}% | {m['sharpe']:.4f} | "
                           f"{m['max_drawdown']*100:.2f}% | {m['win_rate']*100:.1f}% | "
                           f"{m['trade_count']} | {r['score']:.4f} |")
        lines.append("")

    lines.append("## 组合验证结果\n")
    lines.append("| 组合 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 交易数 | 得分 | 变更参数 |")
    lines.append("|------|----------|--------|----------|------|--------|------|----------|")
    for r in combined_results:
        m = r["metrics"]
        if m["annual_return"] is None:
            lines.append(f"| {r['phase']} | FAILED | - | - | - | - | - | - |")
            continue
        changed_str = ", ".join(f"{k}={v}" for k, v in r.get("changed", {}).items())
        lines.append(f"| {r['phase']} | {m['annual_return']*100:.2f}% | {m['sharpe']:.4f} | "
                   f"{m['max_drawdown']*100:.2f}% | {m['win_rate']*100:.1f}% | "
                   f"{m['trade_count']} | {r['score']:.4f} | {changed_str} |")
    lines.append("")

    lines.append("## 总排名（Top 10）\n")
    lines.append("| 排名 | 阶段 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 交易数 | 得分 | 变更参数 |")
    lines.append("|------|------|----------|--------|----------|------|--------|------|----------|")
    for i, r in enumerate(valid_results[:10], 1):
        m = r["metrics"]
        changed_str = ", ".join(f"{k}={v}" for k, v in r.get("changed", {}).items())
        lines.append(f"| {i} | {r['phase']} | {m['annual_return']*100:.2f}% | {m['sharpe']:.4f} | "
                   f"{m['max_drawdown']*100:.2f}% | {m['win_rate']*100:.1f}% | "
                   f"{m['trade_count']} | {r['score']:.4f} | {changed_str} |")
    lines.append("")

    # 最佳推荐
    if valid_results:
        best = valid_results[0]
        m = best["metrics"]
        lines.append("## 最佳参数推荐\n")
        lines.append(f"**推荐参数组合** (得分: {best['score']:.4f}, 基线: {baseline_score:.4f})\n")
        lines.append("```python")
        lines.append("# signal_engine.py __init__ 参数")
        for k, v in best["params"].items():
            if isinstance(v, float):
                lines.append(f"self.{k} = {v}")
            else:
                lines.append(f"self.{k} = {v}")
        lines.append("```\n")

        lines.append("### vs 基线对比\n")
        lines.append("| 指标 | 基线 | 优化后 | 变化 |")
        lines.append("|------|------|--------|------|")
        ar_base = baseline_metrics['annual_return'] * 100
        ar_opt = m['annual_return'] * 100
        lines.append(f"| 年化收益 | {ar_base:.2f}% | {ar_opt:.2f}% | {ar_opt-ar_base:+.2f}% |")
        sh_base = baseline_metrics['sharpe']
        sh_opt = m['sharpe']
        lines.append(f"| Sharpe | {sh_base:.4f} | {sh_opt:.4f} | {sh_opt-sh_base:+.4f} |")
        dd_base = baseline_metrics['max_drawdown'] * 100
        dd_opt = m['max_drawdown'] * 100
        lines.append(f"| 最大回撤 | {dd_base:.2f}% | {dd_opt:.2f}% | {dd_opt-dd_base:+.2f}% |")
        wr_base = baseline_metrics['win_rate'] * 100
        wr_opt = m['win_rate'] * 100
        lines.append(f"| 胜率 | {wr_base:.1f}% | {wr_opt:.1f}% | {wr_opt-wr_base:+.1f}% |")
        tc_base = baseline_metrics['trade_count']
        tc_opt = m['trade_count']
        lines.append(f"| 交易数 | {tc_base} | {tc_opt} | {tc_opt-tc_base:+d} |")
        lines.append("")

    report = "\n".join(lines)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"\n报告已写入: {RESULTS_PATH}")
    return report


def main():
    print("缠论v4策略参数优化")
    print("=" * 60)
    start_time = time.time()

    # 读取原始文件
    read_original()

    try:
        # Phase 1: 单维度扫描
        single_results, best_params_per_dim, baseline_metrics, baseline_score = run_single_scan()

        # Phase 2: 组合验证
        combined_results = run_combined_test(
            best_params_per_dim, baseline_metrics, baseline_score
        )

        # 生成报告
        report = generate_report(
            single_results, combined_results, best_params_per_dim,
            baseline_metrics, baseline_score
        )

    finally:
        # 确保恢复原始文件
        restore_original()
        print("\n已恢复原始signal_engine.py")

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    main()
