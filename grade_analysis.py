#!/usr/bin/env python3
"""周线Grade A vs D 回测对比"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

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

def get_weekly_score_fast(daily_df, date_ts, lookback=104):
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(lookback)
        if len(df_s) < 20: return 2
        wk = df_s.groupby(df_s.index.to_period('W')).agg(close=('close','last'), volume=('volume','sum'))
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

print("计算周线score...")
unique_dates = pd.to_datetime(sig_df['date']).unique()
date_to_score = {}
for j, d in enumerate(unique_dates):
    if j % 20 == 0: print(f"  {j}/{len(unique_dates)}", flush=True)
    date_to_score[d] = get_weekly_score_fast(idx_df, d)
sig_df['week_score'] = pd.to_datetime(sig_df['date']).map(date_to_score)

score_dist = sig_df['week_score'].value_counts().sort_index()
print("\nWeek score distribution:")
for s in sorted(score_dist.index): print(f"  score={s}: {score_dist[s]}")
A = (sig_df['week_score']>=4).sum(); D = (sig_df['week_score']<=1).sum(); BC = len(sig_df)-A-D
print(f"\nA级(score≥4): {A} ({A/len(sig_df)*100:.1f}%)")
print(f"B/C级(2~3):  {BC} ({BC/len(sig_df)*100:.1f}%)")
print(f"D级(score≤1): {D} ({D/len(sig_df)*100:.1f}%)")

# 简化止盈回测核心
def run_simple_bt(sub_df):
    pnls, exit_reasons = [], defaultdict(int)
    for _, row in sub_df.iterrows():
        code, price, sl_price = row['code'], float(row['price']), float(row['sl_price'])
        date = row['date']
        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]
        n = len(df_c)
        sl = sl_price
        trailing_stop = price
        tp_seen = False
        loop_end = min(pos_bar + 30*8, n-1)
        exit_reason = 'timeout'; exit_price = price; ei = pos_bar
        for bi in range(pos_bar+1, loop_end):
            low_bi = float(df_c['low'].iloc[bi]); close_bi = float(df_c['close'].iloc[bi])
            if low_bi <= sl:
                exit_price = sl; exit_reason = 'stop_loss'; ei = bi; break
            profit_pct = (close_bi - price) / price
            if profit_pct >= 0.03:
                dd = (close_bi - price) / close_bi
                if dd >= 0.08 and not tp_seen:
                    exit_price = close_bi; exit_reason = 'take_profit_s'; ei = bi; tp_seen = True; break
            exit_reason = 'timeout'; ei = bi
        else:
            ei = loop_end-1 if loop_end > pos_bar+1 else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if pos_bar+ei < n else price
        pnl_pct = (exit_price - price) / price - 0.0003*2
        pnls.append(pnl_pct); exit_reasons[exit_reason] += 1
    if not pnls: return None
    pnls = np.array(pnls)
    wr = (pnls>0).mean()
    avg = pnls.mean()
    dd_arr = (pnls < 0).cumsum()
    max_dd = abs(pnls.min())
    r = 0.04 / pnls[pnls>0].std() if pnls[pnls>0].std() > 1e-8 else 0
    total_ret = (1+pnls).prod()-1
    return {'sharpe': r, 'max_dd': max_dd*100, 'win_rate': wr*100, 'avg_pnl': avg*100,
            'total_ret': total_ret*100, 'n': len(pnls), 'exit_reasons': dict(exit_reasons)}

print("\n" + "="*70)
print(f"{'Group':<22} {'N':>6} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'总收益%':>9}")
print("-"*75)

for label, mask in [
    ('全部(A/B/C/D)', slice(None)),
    ('A级(score≥4)', sig_df['week_score']>=4),
    ('B/C级(2~3)',   (sig_df['week_score']>=2)&(sig_df['week_score']<=3)),
    ('D级(score≤1)', sig_df['week_score']<=1),
]:
    grp = sig_df.loc[mask]
    print(f"\n运行: {label} ({len(grp)} signals)...", flush=True)
    r = run_simple_bt(grp)
    if r:
        er = r['exit_reasons']
        total = sum(er.values())
        sl = er.get('stop_loss',0); tp = er.get('take_profit_s',0); to = er.get('timeout',0)
        print(f"{label:<22} {r['n']:>6} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['total_ret']:>+9.1f}")
        print(f"  → 止损={sl}({sl/total*100:.1f}%) 止盈={tp}({tp/total*100:.1f}%) 超时={to}({to/total*100:.1f}%)")
