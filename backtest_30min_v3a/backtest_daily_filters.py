#!/usr/bin/env python3
"""
T+0 策略每日过滤器快速回测

用法:
  python backtest_daily_filters.py           # 默认测试所有过滤器
  python backtest_daily_filters.py --quick   # 快速模式(10只)
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "code"))
from signal_engine import SignalEngine

# 确保能 import tushare_daily
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "chanlun_system"))

# ============================================================================
# 配置
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "chanlun_system" / "artifacts"
START_DATE = "2024-12-01"  # 缩短到半年，加快速度
END_DATE = "2026-04-17"
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.0003
SLIPPAGE = 0.001

# 过滤器测试矩阵
FILTER_TESTS = {
    'baseline': {},  # 无过滤器
    'tr_0.5':  {'turnover_rate_min': 0.5},
    'tr_1.0':  {'turnover_rate_min': 1.0},
    'tr_2.0':  {'turnover_rate_min': 2.0},
    'mf_0':    {'net_mf_min': 0},
    'mf_-5000':{'net_mf_min': -5000},
    'mf_5000': {'net_mf_min': 5000},
    'pe_30':   {'pe_max': 30},
    'pe_50':   {'pe_max': 50},
    'combo_1': {'turnover_rate_min': 0.5, 'net_mf_min': -5000},
    'combo_2': {'turnover_rate_min': 1.0, 'net_mf_min': 0},
    'combo_3': {'turnover_rate_min': 0.5, 'pe_max': 50},
}


# ============================================================================
# 数据加载
# ============================================================================

def load_data(codes=None, limit=None):
    """从artifacts加载30分钟CSV"""
    data_map = {}
    csv_files = sorted(DATA_DIR.glob("min30_*.csv"))

    if codes:
        csv_files = [f for f in csv_files if any(c in f.name for c in codes)]
    if limit:
        csv_files = csv_files[:limit]

    for csv_path in csv_files:
        # 从文件名提取代码: min30_000001.SZ.csv -> 000001.SZ
        code = csv_path.stem.replace('min30_', '')

        df = pd.read_csv(csv_path)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        df = df.rename(columns={'vol': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        df = df.sort_index()

        mask = (df.index >= START_DATE) & (df.index <= END_DATE)
        df = df[mask]

        if len(df) < 120:
            continue

        data_map[code] = df

    return data_map


def load_tushare_daily(data_map: dict) -> dict:
    """预加载所有交易日的 tushare 每日数据"""
    from tushare_daily import load_daily_data

    # 提取所有交易日期
    all_dates = set()
    for code, df in data_map.items():
        dates = df.index.strftime('%Y%m%d').unique()
        all_dates.update(dates)

    codes = list(data_map.keys())
    print(f"  [tushare] 预加载 {len(all_dates)} 个交易日 x {len(codes)} 只...")

    daily_data = {}
    for i, date in enumerate(sorted(all_dates)):
        if i % 20 == 0:
            print(f"    进度: {i}/{len(all_dates)}...", end='\r')
        try:
            day = load_daily_data(codes, date)
            if day:
                daily_data[date] = day
        except Exception:
            pass

    print(f"    完成: {len(daily_data)} 天有数据")
    return daily_data


# ============================================================================
# 回测引擎 (复用 backtest_30min_v3a.py 的逻辑)
# ============================================================================

def run_backtest(data_map, engine):
    """运行回测"""
    signals_map = engine.generate(data_map)

    all_dates = set()
    for code, df in data_map.items():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []

    for dt in all_dates:
        day_prices = {}
        for code, df in data_map.items():
            if dt in df.index:
                day_prices[code] = float(df.loc[dt, 'close'])

        # 处理信号
        for code, sig in signals_map.items():
            if code not in day_prices:
                continue
            if dt not in sig.index:
                continue

            target_pos = float(sig.loc[dt])
            price = day_prices[code] * (1 + SLIPPAGE)

            if target_pos > 0:
                target_value = (cash + sum(
                    positions[c]['shares'] * day_prices.get(c, 0)
                    for c in positions
                )) * target_pos
                current_value = positions[code]['shares'] * price if code in positions else 0
                need_value = target_value - current_value

                if need_value > 0 and cash >= need_value:
                    shares = int(need_value / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * (1 + COMMISSION)
                        if cost <= cash:
                            cash -= cost
                            positions[code] = {
                                'shares': shares,
                                'entry_price': price,
                                'cost': cost
                            }
                            trades.append({
                                'date': dt, 'code': code, 'action': 'buy',
                                'price': price, 'shares': shares,
                            })

            elif target_pos == 0 and code in positions:
                pos = positions[code]
                sell_price = price * (1 - SLIPPAGE)
                revenue = pos['shares'] * sell_price * (1 - COMMISSION)
                cash += revenue
                pnl_pct = (sell_price - pos['entry_price']) / pos['entry_price'] * 100

                trades.append({
                    'date': dt, 'code': code, 'action': 'sell',
                    'price': sell_price, 'pnl_pct': round(pnl_pct, 2),
                })
                del positions[code]

    return trades


def calc_metrics(trades, data_map):
    """计算回测指标"""
    sells = [t for t in trades if t['action'] == 'sell']
    if not sells:
        return {'Sharpe': 0, '胜率': '0%', '交易': 0, '均盈亏': '0%'}

    wins = [t for t in sells if t.get('pnl_pct', 0) > 0]
    win_rate = len(wins) / len(sells) * 100
    avg_pnl = np.mean([t['pnl_pct'] for t in sells])

    # 简化 Sharpe — 用交易收益的均值/标准差
    pnls = [t['pnl_pct'] for t in sells]
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / 20) if np.std(pnls) > 0 else 0  # 假设20笔/年

    return {
        'Sharpe': round(float(sharpe), 2),
        '胜率': f'{win_rate:.1f}%',
        '交易': len(sells),
        '均盈亏': f'{avg_pnl:+.2f}%',
        '盈利笔': len(wins),
        '亏损笔': len(sells) - len(wins),
    }


# ============================================================================
# 主流程
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式(10只)')
    args = parser.parse_args()

    limit = 10 if args.quick else None

    print(f"\n{'='*70}")
    print(f"T+0 每日过滤器快速回测")
    print(f"区间: {START_DATE} ~ {END_DATE}")
    print(f"{'='*70}\n")

    # 1. 加载30分钟数据
    print("[1/3] 加载K线数据...")
    data_map = load_data(limit=limit)
    codes = list(data_map.keys())
    print(f"  共{len(codes)}只: {', '.join(codes[:5])}{'...' if len(codes)>5 else ''}\n")

    # 2. 加载tushare每日数据
    print("[2/3] 加载tushare每日数据...")
    t0 = time.time()
    daily_data = load_tushare_daily(data_map)
    print(f"  耗时{time.time()-t0:.1f}s\n")

    # 3. 逐个测试过滤器
    print(f"[3/3] 测试 {len(FILTER_TESTS)} 种过滤器配置...\n")
    print(f"{'过滤器':<15} {'Sharpe':>8} {'胜率':>8} {'交易笔数':>8} {'均盈亏':>10} {'盈利笔':>6} {'亏损笔':>6}")
    print('-' * 70)

    results = {}
    for name, filters in FILTER_TESTS.items():
        engine = SignalEngine(daily_filters=filters, daily_data=daily_data)
        trades = run_backtest(data_map, engine)
        metrics = calc_metrics(trades, data_map)
        results[name] = {**metrics, 'filters': filters}

        marker = ' ← 基线' if name == 'baseline' else ''
        print(f"{name:<15} {metrics['Sharpe']:>8.2f} {metrics['胜率']:>8} "
              f"{metrics['交易']:>8} {metrics['均盈亏']:>10} {metrics['盈利笔']:>6} {metrics['亏损笔']:>6}{marker}")

    # 4. 找出最优
    baseline_sharpe = results['baseline']['Sharpe']
    print(f"\n{'='*70}")
    print(f"基线 Sharpe: {baseline_sharpe}")
    print(f"\n提升排名:")
    improvements = []
    for name, r in results.items():
        if name == 'baseline':
            continue
        delta = r['Sharpe'] - baseline_sharpe
        improvements.append((name, delta, r))
    improvements.sort(key=lambda x: -x[1])

    for name, delta, r in improvements:
        tag = '✅ 值得全量测试' if delta > 0.1 else ('⚠️ 无明显提升' if delta > -0.1 else '❌ 负优化')
        print(f"  {name:<15} ΔSharpe={delta:+.2f}  胜率={r['胜率']}  均盈亏={r['均盈亏']}  {tag}")

    # 保存
    out_path = Path(__file__).parent / "filter_test_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


if __name__ == '__main__':
    main()
