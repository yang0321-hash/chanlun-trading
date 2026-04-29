#!/usr/bin/env python3
"""
止损参数SL网格搜索 - 预计算+精简并行版
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

NCORE = max(1, cpu_count() - 1)
print(f"使用 {NCORE} 核")

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
print(f"总信号: {len(sig_df)}")

base = '/workspace/tdx_data/sh/lday/sh000001.day'
rows = []
with open(base, 'rb') as f:
    data = f.read()
for i in range(len(data)//32):
    vals = struct.unpack('<8I', data[i*32:(i+1)*32])
    rows.append({'date': vals[0], 'close': vals[3]/100.0})
idx_df = pd.DataFrame(rows)
idx_df['date'] = pd.to_datetime(idx_df['date'], format='%Y%m%d')
idx_df.set_index('date', inplace=True)
idx_df.sort_index(inplace=True)

def get_monthly_sl(date_ts):
    try:
        df_s = idx_df[idx_df.index <= date_ts].tail(104)
        if len(df_s) < 20: return 0.94
        wk = df_s.groupby(df_s.index.to_period('W')).agg(close=('close','last'))
        if len(wk) < 20: return 0.94
        cw = wk['close'].astype(float).values
        ma5 = np.convolve(cw, np.ones(5)/5, mode='valid')
        ma10 = np.convolve(cw, np.ones(10)/10, mode='valid')
        m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if len(cw) >= 6 else 0
        sc = sum([m5>m10, cw[-1]>m5, m5>pm5, l5w>0])
        return 0.94 if sc >= 4 else (0.80 if sc >= 2 else 0.93)
    except: return 0.94

def get_daily_state(date_ts):
    try:
        df_s = idx_df[idx_df.index <= date_ts].tail(20)
        if len(df_s) < 10: return 'unknown'
        close = float(df_s['close'].iloc[-1])
        ma5 = float(df_s['close'].iloc[-5:].mean())
        ma10 = float(df_s['close'].iloc[-10:].mean())
        if ma5 > ma10 and close > ma5: return 'bull'
        elif ma5 < ma10 and close < ma5: return 'bear'
        else: return 'neutral'
    except: return 'unknown'

sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_strong = sig_df[fib.isin(['strong','strongest'])].copy().reset_index(drop=True)
print(f"strongest信号: {len(sig_strong)}")

sig_strong['daily_state'] = pd.to_datetime(sig_strong['date']).apply(get_daily_state)
sig_strong['month'] = pd.to_datetime(sig_strong['date']).dt.to_period('M')
sig_strong['monthly_sl'] = sig_strong['month'].apply(
    lambda m: get_monthly_sl((m+1).to_timestamp() - pd.Timedelta(days=1)))

TP_TRIGGER = 0.06
TP_TRAIL = 0.06

# 预计算全部records（只做一次）
print("预计算收益序列...")
records = []
bad = 0
for _, row in sig_strong.iterrows():
    code = row['code']
    price = float(row['price'])
    btype = row['type']
    date = row['date']
    monthly_sl_val = row['monthly_sl']
    daily_state = row['daily_state']
    if code not in data_map: bad += 1; continue
    df_c = data_map[code]
    bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
    if bi_list[0] < 0: bad += 1; continue
    pos_bar = bi_list[0]
    n = len(df_c)
    loop_end = min(pos_bar + 30, n - 1)
    if loop_end <= pos_bar + 1: bad += 1; continue
    try:
        lows = df_c['low'].iloc[pos_bar+1:loop_end+1].astype(float).values
        closes = df_c['close'].iloc[pos_bar+1:loop_end+1].astype(float).values
        highs = df_c['high'].iloc[pos_bar+1:loop_end+1].astype(float).values
        records.append({
            'btype': btype, 'price': price,
            'low_rel': lows/price - 1,
            'close_rel': closes/price - 1,
            'high_rel': highs/price - 1,
            'monthly_sl': monthly_sl_val,
            'daily_state': daily_state,
        })
    except: bad += 1
print(f"有效信号: {len(records)} (skip {bad})")

# 序列化records到磁盘，避免重复预计算
records_cache = '/workspace/sl_records_cache.pkl'
with open(records_cache, 'wb') as f:
    pickle.dump(records, f)
print(f"缓存已存: {records_cache}")

# ── 并行参数搜索（只传records，不传data_map）──
def bt_params(args):
    import pickle
    from collections import defaultdict
    records = pickle.load(open('/workspace/sl_records_cache.pkl', 'rb'))
    sl_1buy, sl_2buy, sl_3buy, bm, bem, nm = args
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
        if btype in ('2buy', '2plus3buy'): sl_base = sl_2buy
        elif btype == '3buy': sl_base = sl_3buy
        else: sl_base = sl_1buy
        mult = mult_map.get(daily_state, 1.0)
        sl_ratio = max(monthly_sl_val, sl_base * mult)
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
        'sl_1': sl_1buy, 'sl_2': sl_2buy, 'sl_3': sl_3buy,
        'mult_bull': bm, 'mult_bear': bem, 'mult_neutral': nm,
    }

# 参数组合
params = []
for sl in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    params.append((sl, sl, sl, 1.0, 1.0, 1.0))
for s1 in [0.04, 0.05, 0.06, 0.07]:
    for s2 in [0.02, 0.03, 0.04, 0.05]:
        for s3 in [0.03, 0.04, 0.05]:
            params.append((s1, s2, s3, 1.0, 1.0, 1.0))
for base_sl in [0.04, 0.05]:
    for bm in [0.7, 0.8, 1.0]:
        for bem in [1.0, 1.2, 1.5]:
            for nm in [0.9, 1.0]:
                params.append((base_sl, base_sl, base_sl, bm, bem, nm))

print(f"总组合: {len(params)}")

# 顺序执行（MP反而因重复load records反而更慢）
print("顺序回测...")
results = []
for i, p in enumerate(params):
    r = bt_params(p)
    if r: results.append(r)
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(params)} ({i+1}/len(params)*100:.0f}%)", flush=True)
print(f"完成 {len(results)} 结果")

results.sort(key=lambda x: -x['sharpe'])
print(f"\n{'#':>4} {'描述':<42} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'SL%':>6} {'止盈%':>6} {'超时%':>6}")
print("-"*100)
for i, r in enumerate(results[:25]):
    if r['sl_1'] == r['sl_2'] == r['sl_3']:
        if r['mult_bull'] != 1.0 or r['mult_bear'] != 1.0:
            desc = f"基准{int(r['sl_1']*100)}%×[多{r['mult_bull']}空{r['mult_bear']}中{r['mult_neutral']}]"
        else:
            desc = f"统一{int(r['sl_1']*100)}%"
    else:
        desc = f"1={int(r['sl_1']*100)}% 2={int(r['sl_2']*100)}% 3={int(r['sl_3']*100)}%"
    print(f"{i+1:>4} {desc:<42} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} "
          f"{r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

print("\n── 多维度Top5 ──")
for met, lab in [('sharpe','Sharpe'),('win_rate','胜率'),('avg_pnl','均盈'),('max_dd','DD最小')]:
    top = sorted(results, key=lambda x: -x[met])[:3]
    print(f"\n{lab}:")
    for r in top:
        if r['sl_1'] == r['sl_2'] == r['sl_3']:
            if r['mult_bull'] != 1.0 or r['mult_bear'] != 1.0:
                desc = f"基准{int(r['sl_1']*100)}%×[多{r['mult_bull']}空{r['mult_bear']}中{r['mult_neutral']}]"
            else:
                desc = f"统一{int(r['sl_1']*100)}%"
        else:
            desc = f"1={int(r['sl_1']*100)}% 2={int(r['sl_2']*100)}% 3={int(r['sl_3']*100)}%"
        print(f"  {desc} → {met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}%")

sh_min = min(r['sharpe'] for r in results); sh_max = max(r['sharpe'] for r in results)
wr_min = min(r['win_rate'] for r in results); wr_max = max(r['win_rate'] for r in results)
av_min = min(r['avg_pnl'] for r in results); av_max = max(r['avg_pnl'] for r in results)
dd_min = min(r['max_dd'] for r in results); dd_max = max(r['max_dd'] for r in results)
for r in results:
    ns = (r['sharpe']-sh_min)/(sh_max-sh_min+1e-8)
    nw = (r['win_rate']-wr_min)/(wr_max-wr_min+1e-8)
    na = (r['avg_pnl']-av_min)/(av_max-av_min+1e-8)
    nd = (dd_max-r['max_dd'])/(dd_max-dd_min+1e-8)
    r['composite'] = 0.4*ns + 0.25*nw + 0.2*na + 0.15*nd
best = max(results, key=lambda x: x['composite'])
if best['sl_1'] == best['sl_2'] == best['sl_3']:
    if best['mult_bull'] != 1.0 or best['mult_bear'] != 1.0:
        desc = f"基准{int(best['sl_1']*100)}%×[多{best['mult_bull']}空{best['mult_bear']}中{best['mult_neutral']}]"
    else:
        desc = f"统一{int(best['sl_1']*100)}%"
else:
    desc = f"1={int(best['sl_1']*100)}% 2={int(best['sl_2']*100)}% 3={int(best['sl_3']*100)}%"
print(f"\n── 综合推荐 ──")
print(f"推荐: {desc}")
print(f"  Sharpe={best['sharpe']:.3f} WR={best['win_rate']:.0f}% 均盈={best['avg_pnl']:+.2f}% DD={best['max_dd']:.1f}%")
print(f"  SL率={best['sl_pct']:.1f}% 止盈率={best['tp_pct']:.1f}% 超时率={best['to_pct']:.1f}%")
