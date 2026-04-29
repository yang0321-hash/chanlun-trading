#!/usr/bin/env python3
"""
日线MA5/MA10 vs 周线Grade 对比回测
验证：哪个大盘判断指标对WR有更好的预测力
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

# fib过滤 → strongest
sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_strong = sig_df[fib.isin(['strong','strongest'])].copy().reset_index(drop=True)
print(f"strongest+strong: {len(sig_strong)} signals")

# 沪指日线
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

# ── 日线MA5/MA10状态 ─────────────────────────────────────────────────────────
idx_df['ma5'] = idx_df['close'].rolling(5).mean()
idx_df['ma10'] = idx_df['close'].rolling(10).mean()
idx_df['ma20'] = idx_df['close'].rolling(20).mean()

def get_daily_state(daily_df, date_ts):
    """日线大盘状态分类"""
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(10)
        if len(df_s) < 5: return 'unknown'
        last = df_s.iloc[-1]
        ma5 = last['ma5']; ma10 = last['ma10']; ma20 = last['ma20']
        close = last['close']
        if pd.isna(ma5) or pd.isna(ma10): return 'unknown'
        # 状态判断
        if ma5 > ma10 and close > ma5: return '上涨笔'
        elif ma5 < ma10 and close < ma5: return '下跌笔'
        else: return '震荡'
    except: return 'unknown'

# ── 周线score ────────────────────────────────────────────────────────────────
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

# 预计算
unique_dates = pd.to_datetime(sig_strong['date']).unique()
date_to_daily_state = {}
date_to_weekly_score = {}
for j, d in enumerate(unique_dates):
    if j % 50 == 0: print(f"  {j}/{len(unique_dates)}", flush=True)
    date_to_daily_state[d] = get_daily_state(idx_df, d)
    date_to_weekly_score[d] = get_weekly_score(idx_df, d)

sig_strong['daily_state'] = pd.to_datetime(sig_strong['date']).map(date_to_daily_state)
sig_strong['week_score'] = pd.to_datetime(sig_strong['date']).map(date_to_weekly_score)

# ── 月度grade（止损基准，与v23_tp_compare一致）────────────────────────────────
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

sig_strong['month'] = pd.to_datetime(sig_strong['date']).dt.to_period('M')
months = sorted(sig_strong['month'].unique())
monthly_grade = {}
for m in months:
    sample = min(pd.Timestamp(idx_df.index.max()), (m+1).to_timestamp() - pd.Timedelta(days=1))
    monthly_grade[m] = get_weekly_grade(idx_df, sample)

# 状态分布
print("\n日线状态分布:")
print(sig_strong['daily_state'].value_counts())
print("\n周线score分布:")
for s in sorted(sig_strong['week_score'].unique()):
    n = (sig_strong['week_score']==s).sum()
    print(f"  score={s}: {n} ({n/len(sig_strong)*100:.1f}%)")

# ── 回测核心 ─────────────────────────────────────────────────────────────────
def run_bt(sub_df):
    pnls, exit_reasons = [], defaultdict(int)
    for _, row in sub_df.iterrows():
        code = row['code']
        price = float(row['price'])
        # 月度grade止损
        month = row['month']
        btype = row['type']
        sl_base = 0.94 if btype in ('2buy', '2plus3buy') else 0.93
        g, cap = monthly_grade.get(month, ('B', 0.8))
        if g == 'C': sl_base = max(sl_base, 0.93)
        elif g == 'B': sl_base = max(sl_base, 0.94)
        sl = price * sl_base
        date = row['date']
        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]
        n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)
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
    return {
        'sharpe': sharpe, 'max_dd': max_dd * 100, 'win_rate': wr * 100,
        'avg_pnl': avg * 100, 'total_ret': total_ret * 100, 'n': len(pnls),
        'sl_pct': er.get('stop_loss',0)/total*100 if total else 0,
        'tp_pct': er.get('take_profit_s',0)/total*100 if total else 0,
        'to_pct': er.get('timeout',0)/total*100 if total else 0,
    }

# ── 日线状态对比 ────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("【日线MA5/MA10大盘状态】")
print(f"{'Group':<20} {'N':>6} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'止损%':>6} {'止盈%':>6}")
print("-"*70)

daily_results = {}
for state in ['上涨笔', '震荡', '下跌笔']:
    grp = sig_strong[sig_strong['daily_state'] == state]
    if len(grp) == 0: continue
    r = run_bt(grp)
    if r:
        daily_results[state] = r
        print(f"{state:<20} {r['n']:>6} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} "
              f"{r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}%")

# ── 周线score对比 ────────────────────────────────────────────────────────────
print("\n【周线score大盘状态】")
print(f"{'Group':<20} {'N':>6} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'止损%':>6} {'止盈%':>6}")
print("-"*70)

weekly_results = {}
for label, mask in [
    ('A级(score≥4)',       sig_strong['week_score']>=4),
    ('B/C级(2~3)',         (sig_strong['week_score']>=2)&(sig_strong['week_score']<=3)),
    ('D级(score≤1)',       sig_strong['week_score']<=1),
]:
    grp = sig_strong.loc[mask]
    if len(grp) == 0: continue
    r = run_bt(grp)
    if r:
        weekly_results[label] = r
        print(f"{label:<20} {r['n']:>6} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} "
              f"{r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}%")

# ── 交叉对比：日线状态 × 周线Grade ─────────────────────────────────────────
print("\n【交叉分析：日线状态 × 周线Grade】")
print(f"{'Group':<24} {'N':>6} {'WR%':>5} {'均盈%':>7}")
print("-"*45)

cross = sig_strong.groupby(['daily_state', pd.cut(sig_strong['week_score'], bins=[-1,1,3,5], labels=['D','B/C','A'])])
for name, grp in cross:
    if len(grp) < 50: continue
    r = run_bt(grp)
    if r:
        label = f"{name[0]}/{name[1]}"
        print(f"{label:<24} {r['n']:>6} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f}")

# ── 预测力评估：哪个指标更能区分WR？─────────────────────────────────────────
print("\n【预测力评估】")
# 计算日内state的WR方差 vs 周线score的WR方差
if daily_results:
    daily_wrs = [v['win_rate'] for v in daily_results.values()]
    items = [f"{k}={v['win_rate']:.0f}%" for k,v in daily_results.items()]
    print(f"日线状态WR分布: {items}")
    print(f"  WR极差: {max(daily_wrs)-min(daily_wrs):.0f}pp → {'强' if max(daily_wrs)-min(daily_wrs)>15 else '弱'}区分力")
if weekly_results:
    weekly_wrs = [v['win_rate'] for v in weekly_results.values()]
    items = [f"{k}={v['win_rate']:.0f}%" for k,v in weekly_results.items()]
    print(f"周线score WR分布: {items}")
    print(f"  WR极差: {max(weekly_wrs)-min(weekly_wrs):.0f}pp → {'强' if max(weekly_wrs)-min(weekly_wrs)>15 else '弱'}区分力")

print("\n结论:")
print("  日线MA5/MA10状态 vs 周线score，哪个对WR的区分度更大？")
if daily_results and weekly_results:
    d_range = max(daily_wrs)-min(daily_wrs)
    w_range = max(weekly_wrs)-min(weekly_wrs)
    if d_range > w_range:
        print(f"  → 日线MA5/MA10区分力更强({d_range:.0f}pp vs {w_range:.0f}pp)，推荐用作大盘环境判断")
    else:
        print(f"  → 周线score区分力更强({w_range:.0f}pp vs {d_range:.0f}pp)，维持原周线Grade方案")
