#!/usr/bin/env python3
"""SL止损参数测试 - 修正版（乘法非floor）"""
import sys, os, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, '/workspace')

records = pickle.load(open('/workspace/sl_records.pkl', 'rb'))
print(f"加载: {len(records)} signals")
monthly_sl_vals = sorted(set(round(r['monthly_sl'],4) for r in records))
print(f"monthly_sl值: {monthly_sl_vals}")
ds = defaultdict(int)
for r in records: ds[r['daily_state']] += 1
print(f"daily_state分布: {dict(ds)}")

TP_TRIGGER = 0.06
TP_TRAIL = 0.06

def bt(sl_base, bm=1.0, bem=1.0, nm=1.0,
        sl_type_mult=False, sl_1buy=None, sl_2buy=None, sl_3buy=None):
    """
    sl_base: 基准SL（小数）
    bm/bem/nm: 大盘状态乘数
    sl_type_mult: True=monthly_sl直接乘, False=max(monthly_sl, sl_base*mult)
    sl_1buy/sl_2buy/sl_3buy: 各买点独立SL（None则用sl_base）
    """
    pnls = []
    exit_counts = defaultdict(int)
    mult_map = {'bull': bm, 'bear': bem, 'neutral': nm, 'unknown': 1.0}
    for rec in records:
        btype = rec['btype']
        monthly_sl_val = rec['monthly_sl']
        daily_state = rec['daily_state']
        low_rel = rec['low_rel']
        close_rel = rec['close_rel']
        high_rel = rec['high_rel']
        n = len(close_rel)

        # SL基准
        if sl_1buy is not None and btype == '1buy': sl_base_use = sl_1buy
        elif sl_2buy is not None and btype in ('2buy', '2plus3buy'): sl_base_use = sl_2buy
        elif sl_3buy is not None and btype == '3buy': sl_base_use = sl_3buy
        else: sl_base_use = sl_base

        mult = mult_map.get(daily_state, 1.0)
        if sl_type_mult:
            # 乘法：monthly_sl * sl_base_use * 大盘mult
            sl_ratio = monthly_sl_val * sl_base_use * mult
        else:
            # 混合：monthly_sl作为floor
            sl_ratio = max(monthly_sl_val, sl_base_use * mult)

        sl_rel = sl_ratio - 1

        if np.any(low_rel <= sl_rel):
            exit_counts['sl'] += 1; pnls.append(sl_rel - 0.0006); continue

        trigger_idx = None
        for i in range(n):
            if close_rel[i] >= TP_TRIGGER: trigger_idx = i; break
        if trigger_idx is None:
            exit_counts['to'] += 1; pnls.append(close_rel[-1] - 0.0006); continue

        price_hwm = high_rel[0]
        tp_exit = None
        for i in range(trigger_idx, n):
            if high_rel[i] > price_hwm: price_hwm = high_rel[i]
            dd = (price_hwm - close_rel[i]) / (1 + price_hwm)
            if dd >= TP_TRAIL: tp_exit = close_rel[i]; break
        if tp_exit is not None:
            exit_counts['tp'] += 1; pnls.append(tp_exit - 0.0006)
        else:
            exit_counts['to'] += 1; pnls.append(close_rel[-1] - 0.0006)

    if not pnls: return None
    pnls = np.array(pnls)
    total = len(pnls)
    wr = (pnls > 0).mean() * 100
    avg = pnls.mean() * 100
    max_dd = abs(pnls.min()) * 100
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = 0.04 / std_w if std_w > 1e-8 else 0
    sl_n = exit_counts['sl']; tp_n = exit_counts['tp']; to_n = exit_counts['to']
    return {
        'sharpe': sharpe, 'win_rate': wr, 'avg_pnl': avg, 'max_dd': max_dd, 'n': total,
        'sl_pct': sl_n/total*100, 'tp_pct': tp_n/total*100, 'to_pct': to_n/total*100,
        'sl_count': sl_n, 'tp_count': tp_n, 'to_count': to_n,
    }

print(f"\n{'策略':<50} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'SL%':>6} {'止盈%':>6} {'超时%':>6}")
print("="*105)

results = []

# A: 乘法模式（monthly_sl * sl_base * 乘数）
print("\n── A组: monthly_sl乘法模式 ──")
for sl in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
    r = bt(sl, sl_type_mult=True)
    if r:
        desc = f'A: 乘法SL={int((1-sl)*100)}%'
        results.append((desc, r))
        print(f"{desc:<50} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

# B: 混合模式（max(floor, sl_base * mult)）
print("\n── B组: 混合模式（max floor）──")
for sl in [0.03, 0.04, 0.05, 0.06, 0.08]:
    r = bt(sl, sl_type_mult=False)
    if r:
        desc = f'B: 混合SL={int(sl*100)}%'
        results.append((desc, r))
        print(f"{desc:<50} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

# C: 大盘乘数（乘法模式）
print("\n── C组: 大盘乘数（乘法模式，基准SL=0.95）──")
for bm in [0.8, 0.9, 1.0]:
    for bem in [1.0, 1.1, 1.2]:
        for nm in [0.95, 1.0]:
            r = bt(0.95, bm, bem, nm, sl_type_mult=True)
            if r:
                desc = f'C: 基准5%×[多{bm}空{bem}中{nm}]'
                results.append((desc, r))

if results:
    # 显示C组top10
    c_results = [(d, r) for d, r in results if d.startswith('C:')]
    c_results.sort(key=lambda x: -x[1]['sharpe'])
    for desc, r in c_results[:10]:
        print(f"{desc:<50} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

# D: 各买点独立SL（乘法模式）
print("\n── D组: 各买点独立SL（乘法）──")
combos = [
    (0.95, 0.95, 0.95, 'D: 全95%'),
    (0.93, 0.95, 0.95, 'D: 1买93% 2买95% 3买95%'),
    (0.93, 0.93, 0.93, 'D: 全93%'),
    (0.95, 0.93, 0.93, 'D: 1买95% 2买93% 3买93%'),
    (0.90, 0.95, 0.93, 'D: 1买90% 2买95% 3买93%'),
]
for s1, s2, s3, desc in combos:
    r = bt(0.95, sl_type_mult=True, sl_1buy=s1, sl_2buy=s2, sl_3buy=s3)
    if r:
        results.append((desc, r))
        print(f"{desc:<50} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

# 排序总览
print("\n── 总排序Top20（Sharpe）──")
results.sort(key=lambda x: -x[1]['sharpe'])
print(f"{'#':>4} {'策略':<50} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'SL%':>6} {'止盈%':>6} {'超时%':>6}")
print("-"*110)
for i, (desc, r) in enumerate(results[:20]):
    print(f"{i+1:>4} {desc:<50} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

print("\n── 多维度Top5 ──")
for met, lab in [('sharpe','Sharpe'),('win_rate','胜率'),('avg_pnl','均盈'),('max_dd','DD最小')]:
    top = sorted(results, key=lambda x: -x[1][met])[:3]
    print(f"\n{lab}:")
    for desc, r in top:
        print(f"  {desc:<50} → {met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}%")

# 综合
sh_vals = [r['sharpe'] for _, r in results]
wr_vals = [r['win_rate'] for _, r in results]
av_vals = [r['avg_pnl'] for _, r in results]
dd_vals = [r['max_dd'] for _, r in results]
sh_min, sh_max = min(sh_vals), max(sh_vals)
wr_min, wr_max = min(wr_vals), max(wr_vals)
av_min, av_max = min(av_vals), max(av_vals)
dd_min, dd_max = min(dd_vals), max(dd_vals)
for desc, r in results:
    ns = (r['sharpe']-sh_min)/(sh_max-sh_min+1e-8)
    nw = (r['win_rate']-wr_min)/(wr_max-wr_min+1e-8)
    na = (r['avg_pnl']-av_min)/(av_max-av_min+1e-8)
    nd = (dd_max-r['max_dd'])/(dd_max-dd_min+1e-8)
    r['composite'] = 0.4*ns + 0.25*nw + 0.2*na + 0.15*nd

best = max(results, key=lambda x: x[1].get('composite', 0))
print(f"\n── 综合推荐 ──")
print(f"推荐: {best[0]}")
print(f"  Sharpe={best[1]['sharpe']:.3f} WR={best[1]['win_rate']:.0f}% 均盈={best[1]['avg_pnl']:+.2f}% DD={best[1]['max_dd']:.1f}%")
print(f"  SL率={best[1]['sl_pct']:.1f}% 止盈率={best[1]['tp_pct']:.1f}% 超时率={best[1]['to_pct']:.1f}%")
