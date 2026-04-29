#!/usr/bin/env python3
"""
周线Grade A vs D 回测对比 - 对齐时间范围版
修正：排除时间偏差 + 置换检验（p-value）
"""
import sys, os, pickle, struct, numpy as np, pandas as pd, random
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

# ── fib过滤 → strongest ───────────────────────────────────────────────────────
sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_strong = sig_df[fib.isin(['strong','strongest'])].copy().reset_index(drop=True)
print(f"strongest+strong: {len(sig_strong)} signals")

# ── 沪指日线 ─────────────────────────────────────────────────────────────────
base = '/workspace/tdx_data/sh/lday/sh000001.day'
rows = []
with open(base, 'rb') as f:
    data = f.read()
for i in range(len(data)//32):
    vals = struct.unpack('<8I', data[i*32:(i+1)*32])
    rows.append({'date': vals[0], 'close': vals[3]/100.0, 'volume': float(vals[6])})
idx_df = pd.DataFrame(rows)
idx_df['date'] = pd.to_datetime(idx_df['date'], format='%Y%m%d')
idx_df.set_index('date', inplace=True)
idx_df.sort_index(inplace=True)

# ── 月度Grade（用于止损基准，与v23_tp_compare一致）─────────────────────────────
def get_weekly_grade(daily_df, date_ts, lookback=104):
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(lookback)
        if len(df_s) < 20: return ('B', 0.8)
        wk = df_s.groupby(df_s.index.to_period('W')).agg(
            close=('close','last'), volume=('volume','sum'))
        nw = len(wk)
        if nw < 20: return ('B', 0.8)
        cw = wk['close'].astype(float).values
        ma5 = np.convolve(cw, np.ones(5)/5, mode='valid')
        ma10 = np.convolve(cw, np.ones(10)/10, mode='valid')
        c = cw[-1]; m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if nw >= 6 else 0
        vn = float(wk['volume'].iloc[-5:].mean())
        vp = float(wk['volume'].iloc[-10:-5].mean()) if nw >= 10 else vn
        vc = (vn-vp)/vp*100 if vp > 0 else 0
        sc = sum([m5>m10, c>m5, m5>pm5, l5w>0, vc>5])
        return ('A', 1.0) if sc >= 4 else ('B', 0.8) if sc >= 2 else ('C', 0.5)
    except: return ('B', 0.8)

def get_sl(row, month):
    btype = row['type']
    price = row['price']
    sl_base = 0.94 if btype in ('2buy', '2plus3buy') else 0.93
    g, cap = monthly_grade.get(month, ('B', 0.8))
    if g == 'C': sl_base = max(sl_base, 0.93)
    elif g == 'B': sl_base = max(sl_base, 0.94)
    return price * sl_base

# 月度grade（用于止损基准）
sig_strong['month'] = pd.to_datetime(sig_strong['date']).dt.to_period('M')
months = sorted(sig_strong['month'].unique())
monthly_grade = {}
for m in months:
    sample = min(pd.Timestamp(idx_df.index.max()), (m+1).to_timestamp() - pd.Timedelta(days=1))
    monthly_grade[m] = get_weekly_grade(idx_df, sample)
print(f"月度grade覆盖: {len(monthly_grade)} 个月")

# 周线score（用于分组）
def get_weekly_score(daily_df, date_ts, lookback=104):
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(lookback)
        if len(df_s) < 20: return 2
        wk = df_s.groupby(df_s.index.to_period('W')).agg(
            close=('close','last'), volume=('volume','sum'))
        nw = len(wk)
        if nw < 20: return 2
        cw = wk['close'].astype(float).values
        ma5 = np.convolve(cw, np.ones(5)/5, mode='valid')
        ma10 = np.convolve(cw, np.ones(10)/10, mode='valid')
        c = cw[-1]; m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if nw >= 6 else 0
        vn = float(wk['volume'].iloc[-5:].mean())
        vp = float(wk['volume'].iloc[-10:-5].mean()) if nw >= 10 else vn
        vc = (vn-vp)/vp*100 if vp > 0 else 0
        return sum([m5>m10, c>m5, m5>pm5, l5w>0, vc>5])
    except: return 2

# 预计算每周score（避免逐行apply）
unique_dates = pd.to_datetime(sig_strong['date']).unique()
date_to_score = {}
for j, d in enumerate(unique_dates):
    if j % 50 == 0: print(f"  score {j}/{len(unique_dates)}", flush=True)
    date_to_score[d] = get_weekly_score(idx_df, d)
sig_strong['week_score'] = pd.to_datetime(sig_strong['date']).map(date_to_score)

print("\nScore分布:")
for s in sorted(sig_strong['week_score'].unique()):
    n = (sig_strong['week_score']==s).sum()
    print(f"  score={s}: {n} ({n/len(sig_strong)*100:.1f}%)")

A_mask = sig_strong['week_score'] >= 4
D_mask = sig_strong['week_score'] <= 1
print(f"\nA级(score≥4): {A_mask.sum()} ({A_mask.sum()/len(sig_strong)*100:.1f}%)")
print(f"D级(score≤1): {D_mask.sum()} ({D_mask.sum()/len(sig_strong)*100:.1f}%)")

# ── 回测核心（与v23_tp_compare一致：30-bar窗口 + 月度grade止损）─────────────
def run_bt_grade(sub_df, desc=''):
    pnls, exit_reasons = [], defaultdict(int)
    for _, row in sub_df.iterrows():
        code = row['code']
        price = float(row['price'])
        sl_price = float(row['sl_price'])
        sl = sl_price  # 月度grade已在sl_price中体现
        date = row['date']
        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]
        n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)  # 30-bar窗口，与v23_tp_compare一致
        tp_seen = False
        exit_reason = 'timeout'; exit_price = price; ei = pos_bar
        for bi in range(pos_bar + 1, loop_end):
            low_bi = float(df_c['low'].iloc[bi])
            close_bi = float(df_c['close'].iloc[bi])
            if low_bi <= sl:
                exit_price = sl; exit_reason = 'stop_loss'; ei = bi; break
            profit_pct = (close_bi - price) / price
            if profit_pct >= 0.03:
                dd = (close_bi - price) / close_bi
                if dd >= 0.08 and not tp_seen:
                    exit_price = close_bi; exit_reason = 'take_profit_s'; ei = bi; tp_seen = True; break
            exit_reason = 'timeout'; ei = bi
        else:
            ei = loop_end - 1 if loop_end > pos_bar + 1 else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if ei < n else price
        pnl_pct = (exit_price - price) / price - 0.0003 * 2
        pnls.append(pnl_pct); exit_reasons[exit_reason] += 1
    if not pnls: return None
    pnls = np.array(pnls)
    wr = (pnls > 0).mean()
    avg = pnls.mean()
    max_dd = abs(pnls.min())
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = 0.04 / std_w if std_w > 1e-8 else 0
    total_ret = (1 + pnls).prod() - 1
    er = dict(exit_reasons)
    total = sum(er.values())
    sl_n = er.get('stop_loss', 0); tp_n = er.get('take_profit_s', 0); to_n = er.get('timeout', 0)
    return {
        'sharpe': sharpe, 'max_dd': max_dd * 100, 'win_rate': wr * 100, 'avg_pnl': avg * 100,
        'total_ret': total_ret * 100, 'n': len(pnls),
        'sl_pct': sl_n/total*100 if total else 0,
        'tp_pct': tp_n/total*100 if total else 0,
        'to_pct': to_n/total*100 if total else 0,
    }

# ── 对比：Grade A vs D ────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Group':<24} {'N':>6} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'止损%':>6} {'止盈%':>6}")
print("-"*75)

results = []
for label, mask in [
    ('全部(strongest)',         slice(None)),
    ('A级(score≥4)',           A_mask),
    ('B/C级(2~3)',             (~A_mask) & (~D_mask)),
    ('D级(score≤1)',           D_mask),
]:
    grp = sig_strong.loc[mask]
    r = run_bt_grade(grp, label)
    if r:
        print(f"{label:<24} {r['n']:>6} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} "
              f"{r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}%")
        results.append((label, r))

# ── 时间对齐检验：对每个Grade A信号的日期，找同一天或最近邻的Grade D信号 ───────
print("\n── 时间对齐检验（Grade A vs D 配对）──")
sig_A = sig_strong[A_mask].copy()
sig_D = sig_strong[D_mask].copy()

# 找每个Grade A信号对应日期的Grade D信号
paired_A_pnls = []
paired_D_pnls = []
unpaired_results = {'A': defaultdict(list), 'D': defaultdict(list)}

# 使用same-date pairing
sig_A_dates = set(pd.to_datetime(sig_A['date']))
sig_D_dates = set(pd.to_datetime(sig_D['date']))
common_dates = sig_A_dates & sig_D_dates
print(f"共同日期数: {len(common_dates)} (A有{len(sig_A_dates)}天, D有{len(sig_D_dates)}天)")

if len(common_dates) > 0:
    # 用共同日期的信号做配对对比
    sig_A_common = sig_A[pd.to_datetime(sig_A['date']).isin(common_dates)]
    sig_D_common = sig_D[pd.to_datetime(sig_D['date']).isin(common_dates)]
    print(f"A级共同日期信号: {len(sig_A_common)}, D级共同日期信号: {len(sig_D_common)}")

    r_A = run_bt_grade(sig_A_common, 'A级(共同日期)')
    r_D = run_bt_grade(sig_D_common, 'D级(共同日期)')
    if r_A and r_D:
        print(f"{'A':<24} {r_A['n']:>6} {r_A['sharpe']:>7.2f} {r_A['max_dd']:>6.1f} "
              f"{r_A['win_rate']:>5.0f} {r_A['avg_pnl']:>+7.2f} {r_A['sl_pct']:>5.1f}% {r_A['tp_pct']:>5.1f}%")
        print(f"{'D':<24} {r_D['n']:>6} {r_D['sharpe']:>7.2f} {r_D['max_dd']:>6.1f} "
              f"{r_D['win_rate']:>5.0f} {r_D['avg_pnl']:>+7.2f} {r_D['sl_pct']:>5.1f}% {r_D['tp_pct']:>5.1f}%")
        print(f"\n  WR差值: D - A = {r_D['win_rate']-r_A['win_rate']:+.0f}pp")
        print(f"  均盈差值: D - A = {r_D['avg_pnl']-r_A['avg_pnl']:+.2f}%")

# ── 置换检验：打乱week_score标签，看差异是否随机 ────────────────────────────
print("\n── 置换检验（permutation test, n=999）──")
def calc_wr_diff(df):
    pnls_A = []; pnls_D = []
    for _, row in df.iterrows():
        code, price, sl = row['code'], float(row['price']), float(row['sl_price'])
        date = row['date']
        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]; n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)
        tp_seen = False; pnl = 0
        for bi in range(pos_bar + 1, loop_end):
            low_bi = float(df_c['low'].iloc[bi]); close_bi = float(df_c['close'].iloc[bi])
            if low_bi <= sl:
                pnl = (sl - price) / price - 0.0003 * 2; break
            profit_pct = (close_bi - price) / price
            if profit_pct >= 0.03:
                dd = (close_bi - price) / close_bi
                if dd >= 0.08 and not tp_seen:
                    pnl = (close_bi - price) / price - 0.0003 * 2; tp_seen = True; break
        else:
            ei = loop_end - 1 if loop_end > pos_bar + 1 else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if ei < n else price
            pnl = (exit_price - price) / price - 0.0003 * 2
        if df.loc[_.name, 'week_score'] >= 4: pnls_A.append(pnl)
        else: pnls_D.append(pnl)
    if pnls_A and pnls_D:
        return np.mean(pnls_A), np.mean(pnls_D)
    return None, None

# 快速置换检验（用样本）
sample_A = sig_strong[A_mask].sample(n=min(2000, A_mask.sum()), random_state=42)
sample_D = sig_strong[D_mask].sample(n=min(2000, D_mask.sum()), random_state=42)
sample = pd.concat([sample_A, sample_D]).copy()
observed_diff = sample_D['avg_pnl'].mean() - sample_A['avg_pnl'].mean()

# 重置index避免 _.name 问题
sample = sample.reset_index(drop=True)
# 简化版：直接比较win_rate差
def quick_wr_diff(df):
    pnls_A = []; pnls_D = []
    for _, row in df.iterrows():
        code, price, sl = row['code'], float(row['price']), float(row['sl_price'])
        date = row['date']
        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]; n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)
        tp_seen = False; pnl = 0
        for bi in range(pos_bar + 1, loop_end):
            low_bi = float(df_c['low'].iloc[bi]); close_bi = float(df_c['close'].iloc[bi])
            if low_bi <= sl:
                pnl = (sl - price) / price - 0.0003 * 2; break
            profit_pct = (close_bi - price) / price
            if profit_pct >= 0.03:
                dd = (close_bi - price) / close_bi
                if dd >= 0.08 and not tp_seen:
                    pnl = (close_bi - price) / price - 0.0003 * 2; tp_seen = True; break
        else:
            ei = loop_end - 1 if loop_end > pos_bar + 1 else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if ei < n else price
            pnl = (exit_price - price) / price - 0.0003 * 2
        if row['week_score'] >= 4: pnls_A.append(pnl)
        else: pnls_D.append(pnl)
    if pnls_A and pnls_D:
        return (np.mean(pnls_A) - np.mean(pnls_D)) * 100  # in ppts
    return 0

observed_diff = quick_wr_diff(sample)
count = 0
n_perm = 999
print(f"  观察到的WR差值 (D-A): {observed_diff:+.2f}pp")
for i in range(n_perm):
    perm = sample.copy()
    perm['week_score'] = perm['week_score'].sample(frac=1, random_state=i).values
    diff = quick_wr_diff(perm)
    if diff <= observed_diff: count += 1
p_value = (count + 1) / (n_perm + 1)
print(f"  置换检验 p-value: {p_value:.4f} ({'显著' if p_value < 0.05 else '不显著'})")
