#!/usr/bin/env python3
"""SL止损参数测试 - 复用partial_tp缓存"""
import sys, os, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, '/workspace')

# 复用partial_tp_v2的缓存
cache = '/workspace/sl_records.pkl'
if not os.path.exists(cache):
    print("缓存不存在，先运行partial_tp_v2.py生成缓存")
    sys.exit(1)

with open(cache, 'rb') as f:
    records = pickle.load(f)
print(f"加载缓存: {len(records)} signals")

TP_TRIGGER = 0.06
TP_TRAIL = 0.06

def bt(sl_val, bm=1.0, bem=1.0, nm=1.0, separate_types=False):
    """sl_val: 统一SL，或 dict{btype: sl}"""
    pnls = []
    exit_counts = defaultdict(int)
    for rec in records:
        btype = rec['btype']
        monthly_sl_val = rec['monthly_sl']
        daily_state = rec['daily_state']
        low_rel = rec['low_rel']
        close_rel = rec['close_rel']
        high_rel = rec['high_rel']
        n = len(close_rel)

        if isinstance(sl_val, dict):
            sl_base = sl_val.get(btype, 0.05)
        else:
            sl_base = sl_val

        mult_map = {'bull': bm, 'bear': bem, 'neutral': nm, 'unknown': 1.0}
        mult = mult_map.get(daily_state, 1.0)
        sl_ratio_val = max(monthly_sl_val, sl_base * mult)
        sl_rel = sl_ratio_val - 1

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
    }

# 关键SL测试
print(f"\n{'策略':<45} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'SL%':>6} {'止盈%':>6} {'超时%':>6}")
print("-"*95)

strategies = []

# 统一SL
for sl in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    r = bt(sl)
    if r:
        strategies.append((f'统一SL={int(sl*100)}%', r))

# 大盘乘数（基准5%）
for bm in [0.7, 0.8, 1.0]:
    for bem in [1.0, 1.2, 1.5]:
        for nm in [0.9, 1.0]:
            r = bt(0.05, bm, bem, nm)
            if r:
                strategies.append((f'基准5% ×[多{bm}空{bem}中{nm}]', r))

# 各买点独立SL
combos = [
    (0.06, 0.03, 0.04, '1买6% 2买3% 3买4%'),
    (0.05, 0.03, 0.04, '1买5% 2买3% 3买4%'),
    (0.05, 0.02, 0.03, '1买5% 2买2% 3买3%'),
    (0.06, 0.02, 0.03, '1买6% 2买2% 3买3%'),
    (0.04, 0.03, 0.04, '1买4% 2买3% 3买4%'),
]
for s1, s2, s3, desc in combos:
    r = bt({'1buy': s1, '2buy': s2, '3buy': s3, '2plus3buy': s2, 'other': s1})
    if r:
        strategies.append((desc, r))

strategies.sort(key=lambda x: -x[1]['sharpe'])
for desc, r in strategies[:30]:
    print(f"{desc:<45} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} "
          f"{r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

print("\n── 多维度Top5 ──")
for met, lab in [('sharpe','Sharpe'),('win_rate','胜率'),('avg_pnl','均盈'),('max_dd','DD最小')]:
    top = sorted(strategies, key=lambda x: -x[1][met])[:3]
    print(f"\n{lab}:")
    for desc, r in top:
        print(f"  {desc:<45} → {met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}%")

# 综合
sh_vals = [r['sharpe'] for _, r in strategies]
wr_vals = [r['win_rate'] for _, r in strategies]
av_vals = [r['avg_pnl'] for _, r in strategies]
dd_vals = [r['max_dd'] for _, r in strategies]
sh_min, sh_max = min(sh_vals), max(sh_vals)
wr_min, wr_max = min(wr_vals), max(wr_vals)
av_min, av_max = min(av_vals), max(av_vals)
dd_min, dd_max = min(dd_vals), max(dd_vals)
for desc, r in strategies:
    ns = (r['sharpe']-sh_min)/(sh_max-sh_min+1e-8)
    nw = (r['win_rate']-wr_min)/(wr_max-wr_min+1e-8)
    na = (r['avg_pnl']-av_min)/(av_max-av_min+1e-8)
    nd = (dd_max-r['max_dd'])/(dd_max-dd_min+1e-8)
    r1 = dict(r); r1['composite'] = 0.4*ns + 0.25*nw + 0.2*na + 0.15*nd

best = max(strategies, key=lambda x: x[1].get('composite', 0))
print(f"\n── 综合推荐 ──")
print(f"推荐: {best[0]}")
print(f"  Sharpe={best[1]['sharpe']:.3f} WR={best[1]['win_rate']:.0f}% 均盈={best[1]['avg_pnl']:+.2f}% DD={best[1]['max_dd']:.1f}%")
print(f"  SL率={best[1]['sl_pct']:.1f}% 止盈率={best[1]['tp_pct']:.1f}% 超时率={best[1]['to_pct']:.1f}%")
