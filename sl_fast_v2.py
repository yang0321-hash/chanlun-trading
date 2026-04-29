#!/usr/bin/env python3
"""
止损参数SL网格搜索 - 加速版
预计算所有信号的30日序列，直接查表计算PnL
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
print(f"总信号: {len(sig_df)}")

# 沪指
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
        c = cw[-1]; m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if len(cw) >= 6 else 0
        sc = sum([m5>m10, c>m5, m5>pm5, l5w>0])
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

sig_df['daily_state'] = pd.to_datetime(sig_df['date']).apply(get_daily_state)
sig_df['month'] = pd.to_datetime(sig_df['date']).dt.to_period('M')
sig_df['monthly_sl'] = sig_df['month'].apply(lambda m: get_monthly_sl((m+1).to_timestamp() - pd.Timedelta(days=1)))

TP_TRIGGER = 0.06
TP_TRAIL = 0.06

# 预计算：每个信号的30日序列（low/close相对entry）
print("预计算收益序列...")
records = []
skipped = 0
for _, row in sig_df.iterrows():
    code = row['code']
    price = float(row['price'])
    btype = row['type']
    date = row['date']
    if code not in data_map: skipped += 1; continue
    df_c = data_map[code]
    bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
    if bi_list[0] < 0: skipped += 1; continue
    pos_bar = bi_list[0]
    n = len(df_c)
    loop_end = min(pos_bar + 30, n - 1)
    if loop_end <= pos_bar + 1: skipped += 1; continue
    try:
        lows = df_c['low'].iloc[pos_bar+1:loop_end+1].astype(float).values
        closes = df_c['close'].iloc[pos_bar+1:loop_end+1].astype(float).values
        highs = df_c['high'].iloc[pos_bar+1:loop_end+1].astype(float).values
        # 月度SL已经算好
        monthly_sl_val = row['monthly_sl']
        daily_state = row['daily_state']
        records.append({
            'btype': btype, 'price': price,
            'low_rel': lows/price - 1,
            'close_rel': closes/price - 1,
            'high_rel': highs/price - 1,
            'monthly_sl': monthly_sl_val,
            'daily_state': daily_state,
        })
    except: skipped += 1
print(f"有效信号: {len(records)} (skip {skipped})")

def bt_sl(sl_1buy, sl_2buy, sl_3buy,
          sl_bull_mult=1.0, sl_bear_mult=1.0, sl_neutral_mult=1.0):
    """基于预计算records的向量化回测"""
    pnls = []
    exit_counts = defaultdict(int)
    mult_map = {'bull': sl_bull_mult, 'bear': sl_bear_mult, 'neutral': sl_neutral_mult, 'unknown': 1.0}

    for rec in records:
        btype = rec['btype']
        price = rec['price']
        monthly_sl_val = rec['monthly_sl']
        daily_state = rec['daily_state']
        low_rel = rec['low_rel']
        close_rel = rec['close_rel']
        high_rel = rec['high_rel']
        n = len(close_rel)

        # SL基准
        if btype in ('2buy', '2plus3buy'):
            sl_base = sl_2buy
        elif btype == '3buy':
            sl_base = sl_3buy
        else:
            sl_base = sl_1buy

        # 大盘乘数
        mult = mult_map.get(daily_state, 1.0)
        sl_ratio = max(monthly_sl_val, sl_base * mult)
        sl_rel = sl_ratio - 1

        # 止损检测
        if np.any(low_rel <= sl_rel):
            exit_counts['sl'] += 1
            pnls.append(sl_rel - 0.0006)
            continue

        # 触发点
        trigger_idx = None
        for i in range(n):
            if close_rel[i] >= TP_TRIGGER:
                trigger_idx = i; break
        if trigger_idx is None:
            exit_counts['to'] += 1
            pnls.append(close_rel[-1] - 0.0006)
            continue

        # 高水位+回撤止盈
        price_hwm = high_rel[0]
        tp_exit_rel = None
        for i in range(trigger_idx, n):
            if high_rel[i] > price_hwm: price_hwm = high_rel[i]
            dd = (price_hwm - close_rel[i]) / (1 + price_hwm)
            if dd >= TP_TRAIL:
                tp_exit_rel = close_rel[i]; break

        if tp_exit_rel is not None:
            exit_counts['tp'] += 1
            pnls.append(tp_exit_rel - 0.0006)
        else:
            exit_counts['to'] += 1
            pnls.append(close_rel[-1] - 0.0006)

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
        'sharpe': sharpe, 'win_rate': wr, 'avg_pnl': avg,
        'max_dd': max_dd, 'n': total,
        'sl_pct': sl_n/total*100, 'tp_pct': tp_n/total*100, 'to_pct': to_n/total*100,
        'sl_count': sl_n, 'tp_count': tp_n, 'to_count': to_n,
    }

# ── 网格搜索 ────────────────────────────────────────────────────────────────
print("\n开始网格搜索...")
results = []

# 1. 统一SL
print("  统一SL...", flush=True)
for sl in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    r = bt_sl(sl, sl, sl)
    if r:
        r['desc'] = f'统一{int(sl*100)}%'; r['sl_all'] = sl
        results.append(r)

# 2. 各买点独立SL
print("  各买点独立SL...", flush=True)
for s1 in [0.04, 0.05, 0.06]:
    for s2 in [0.02, 0.03, 0.04, 0.05]:
        for s3 in [0.03, 0.04, 0.05]:
            r = bt_sl(s1, s2, s3)
            if r:
                r['desc'] = f'1={int(s1*100)}% 2={int(s2*100)}% 3={int(s3*100)}%'
                r['sl_1'] = s1; r['sl_2'] = s2; r['sl_3'] = s3
                results.append(r)

# 3. 大盘乘数（基准SL=5%）
print("  大盘乘数...", flush=True)
for bm in [0.7, 0.8, 1.0]:
    for bem in [1.0, 1.2, 1.5]:
        for nm in [0.9, 1.0]:
            r = bt_sl(0.05, 0.05, 0.05, bm, bem, nm)
            if r:
                r['desc'] = f'基准5% ×[多{bem}空{bem}中{nm}]'
                r['sl_all'] = 0.05; r['mult_bull'] = bm
                r['mult_bear'] = bem; r['mult_neutral'] = nm
                results.append(r)

print(f"\n完成 {len(results)} 组合\n")

# 排序输出
results.sort(key=lambda x: -x['sharpe'])
print(f"{'#':>4} {'描述':<45} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'SL%':>6} {'止盈%':>6} {'超时%':>6}")
print("-"*95)
for i, r in enumerate(results[:25]):
    print(f"{i+1:>4} {r['desc']:<45} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} "
          f"{r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}%")

print("\n── 多维度Top5 ──")
for met, lab in [('sharpe','Sharpe'),('win_rate','胜率'),('avg_pnl','均盈'),('max_dd','DD最小')]:
    top = sorted(results, key=lambda x: -x[met])[:3]
    print(f"\n{lab}:")
    for r in top:
        print(f"  {r['desc']:<45} → {met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}% SL%={r['sl_pct']:.1f}%")

# 综合推荐
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
print(f"\n── 综合推荐 ──")
print(f"推荐: {best['desc']}")
print(f"  Sharpe={best['sharpe']:.3f} WR={best['win_rate']:.0f}% 均盈={best['avg_pnl']:+.2f}% DD={best['max_dd']:.1f}%")
print(f"  SL率={best['sl_pct']:.1f}% 止盈率={best['tp_pct']:.1f}% 超时率={best['to_pct']:.1f}%")
