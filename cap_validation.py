#!/usr/bin/env python3
"""
仓位分级验证：新框架(日线状态×周线Grade) × 仓位 → Sharpe/DD
验证：不同大盘环境下，多少仓位最优
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
idx_df['ma5'] = idx_df['close'].rolling(5).mean()
idx_df['ma10'] = idx_df['close'].rolling(10).mean()

# 日线状态
def get_daily_state(daily_df, date_ts):
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(10)
        if len(df_s) < 5: return 'unknown'
        last = df_s.iloc[-1]
        ma5 = last['ma5']; ma10 = last['ma10']; close = last['close']
        if pd.isna(ma5) or pd.isna(ma10): return 'unknown'
        if ma5 > ma10 and close > ma5: return '上涨笔'
        elif ma5 < ma10 and close < ma5: return '下跌笔'
        else: return '震荡'
    except: return 'unknown'

# 周线score
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

# 月度grade（止损基准）
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

print("计算日线状态、周线score...")
unique_dates = pd.to_datetime(sig_strong['date']).unique()
date_to_state = {}
date_to_score = {}
for j, d in enumerate(unique_dates):
    if j % 50 == 0: print(f"  {j}/{len(unique_dates)}", flush=True)
    date_to_state[d] = get_daily_state(idx_df, d)
    date_to_score[d] = get_weekly_score(idx_df, d)

sig_strong['daily_state'] = pd.to_datetime(sig_strong['date']).map(date_to_state)
sig_strong['week_score'] = pd.to_datetime(sig_strong['date']).map(date_to_score)

# 合并日线和周线标签
def combined_label(row):
    ds = row['daily_state']
    ws = row['week_score']
    grade = 'A' if ws >= 4 else ('B/C' if ws >= 2 else 'D')
    return f"{ds}/{grade}"

sig_strong['combined'] = sig_strong.apply(combined_label, axis=1)

# 月度grade（止损基准）
sig_strong['month'] = pd.to_datetime(sig_strong['date']).dt.to_period('M')
months = sorted(sig_strong['month'].unique())
monthly_grade = {}
for m in months:
    sample = min(pd.Timestamp(idx_df.index.max()), (m+1).to_timestamp() - pd.Timedelta(days=1))
    monthly_grade[m] = get_weekly_grade(idx_df, sample)

# ── 回测核心（支持仓位比例）────────────────────────────────────────────────
def run_bt(sub_df, cap_ratio=1.0, desc=''):
    """cap_ratio: 仓位比例（0.2~1.0），用于计算实际收益和DD"""
    pnls, exit_reasons = [], defaultdict(int)
    for _, row in sub_df.iterrows():
        code = row['code']
        price = float(row['price'])
        month = row['month']
        btype = row['type']
        sl_base = 0.94 if btype in ('2buy', '2plus3buy') else 0.93
        g, _ = monthly_grade.get(month, ('B', 0.8))
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
        # 仓位调整：单笔收益按仓位比例缩放
        adj_pnl = pnl_pct * cap_ratio
        pnls.append(adj_pnl); exit_reasons[exit_reason] += 1
    if not pnls: return None
    pnls = np.array(pnls)
    wr = (pnls > 0).mean()
    avg = pnls.mean()
    max_dd = abs(pnls.min())
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = (avg * 252 / (std_w * np.sqrt(252))) if std_w > 1e-8 else 0
    total_ret = (1 + pnls).prod() - 1
    er = dict(exit_reasons)
    total = sum(er.values())
    return {
        'sharpe': sharpe, 'max_dd': max_dd * 100, 'win_rate': wr * 100,
        'avg_pnl': avg * 100, 'total_ret': total_ret * 100, 'n': len(pnls),
        'sl_pct': er.get('stop_loss',0)/total*100 if total else 0,
        'tp_pct': er.get('take_profit_s',0)/total*100 if total else 0,
    }

# ── 仓位验证：每个(大盘状态×周线Grade)组合下，测不同仓位 ──────────────────
print("\n" + "="*80)
print("【仓位分级验证：日线状态×周线Grade × 仓位比例】")
print("说明: Sharpe=年化(avg*252/daily_std), DD%=单笔最大亏损(已按仓位缩放)")
print()

# 简化止盈参数
TAKE_PROFIT_PCT = 0.03   # 浮盈≥3%启动
TRAILING_PCT = 0.08       # 回撤≥8%触发止盈

groups = sig_strong.groupby('combined')
caps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]

# 存储结果
results_table = []

for gname, grp in groups:
    if len(grp) < 100: continue
    best_sharpe = -999; best_cap = 0.2
    row_data = {'group': gname, 'n': len(grp)}
    for cap in caps:
        r = run_bt(grp, cap_ratio=cap)
        if r:
            row_data[f'sh_{int(cap*100)}'] = r['sharpe']
            row_data[f'dd_{int(cap*100)}'] = r['max_dd']
            if r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_cap = cap
    row_data['best_cap'] = best_cap
    row_data['best_sh'] = best_sharpe
    results_table.append(row_data)

# 打印表格
print(f"{'Group':<18} {'N':>5} {'Sharpe@20%':>10} {'Sharpe@40%':>10} {'Sharpe@60%':>10} {'Sharpe@100%':>11} {'最优仓位':>8} {'最优Sharpe':>10}")
print("-"*90)
for row in sorted(results_table, key=lambda x: -x.get('best_sh', 0)):
    g = row['group']
    n = row['n']
    sh20 = row.get('sh_20', 0)
    sh40 = row.get('sh_40', 0)
    sh60 = row.get('sh_60', 0)
    sh100 = row.get('sh_100', 0)
    bc = row.get('best_cap', 0)
    bs = row.get('best_sh', 0)
    print(f"{g:<18} {n:>5} {sh20:>10.2f} {sh40:>10.2f} {sh60:>10.2f} {sh100:>11.2f} {int(bc*100):>7}% {bs:>10.2f}")

# DD对比
print(f"\n{'Group':<18} {'N':>5} {'DD%@20%':>8} {'DD%@40%':>8} {'DD%@60%':>8} {'DD%@100%':>9} {'v2.0建议仓位':>12}")
print("-"*75)
for row in sorted(results_table, key=lambda x: -x.get('best_sh', 0)):
    g = row['group']
    n = row['n']
    dd20 = row.get('dd_20', 0)
    dd40 = row.get('dd_40', 0)
    dd60 = row.get('dd_60', 0)
    dd100 = row.get('dd_100', 0)
    bc = row.get('best_cap', 0)
    # v2.0建议仓位
    if '上涨笔' in g: sugg = '70-80%'
    elif '震荡' in g: sugg = '40-50%'
    else: sugg = '20-30%'
    print(f"{g:<18} {n:>5} {dd20:>7.1f}% {dd40:>7.1f}% {dd60:>7.1f}% {dd100:>8.1f}% {sugg:>12}")

# ── v2.0建议仓位 vs 回测最优仓位 ──────────────────────────────────────────
print("\n【v2.0建议仓位 vs 回测最优仓位】")
print(f"{'Group':<18} {'v2.0建议':>10} {'回测最优':>10} {'一致性':>8}")
print("-"*50)
for row in sorted(results_table, key=lambda x: -x.get('best_sh', 0)):
    g = row['group']
    bc = row.get('best_cap', 0)
    if '上涨笔' in g: sugg = '70-80%'; sugg_val = 0.75
    elif '震荡' in g: sugg = '40-50%'; sugg_val = 0.45
    else: sugg = '20-30%'; sugg_val = 0.25
    match = '✅ 一致' if abs(bc - sugg_val) <= 0.15 else '⚠️ 偏差'
    print(f"{g:<18} {sugg:>10} {int(bc*100):>8}% {match:>8}")
