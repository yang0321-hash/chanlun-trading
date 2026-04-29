#!/usr/bin/env python3
"""
密集网格搜索 v2 - 加速版
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_strong = sig_df[fib.isin(['strong','strongest'])].copy().reset_index(drop=True)
print(f"signals: {len(sig_strong)}")

def get_weekly_grade(daily_df, date_ts, lookback=104):
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(lookback)
        if len(df_s) < 20: return 0.8
        wk = df_s.groupby(df_s.index.to_period('W')).agg(close=('close','last'))
        nw = len(wk)
        if nw < 20: return 0.8
        cw = wk['close'].astype(float).values
        ma5 = np.convolve(cw, np.ones(5)/5, mode='valid')
        ma10 = np.convolve(cw, np.ones(10)/10, mode='valid')
        c = cw[-1]; m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if nw >= 6 else 0
        sc = sum([m5>m10, c>m5, m5>pm5, l5w>0])
        return 0.94 if sc >= 4 else (0.80 if sc >= 2 else 0.93)
    except: return 0.8

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

sig_strong['month'] = pd.to_datetime(sig_strong['date']).dt.to_period('M')
months = sorted(sig_strong['month'].unique())
monthly_sl = {}
for m in months:
    sample = min(pd.Timestamp(idx_df.index.max()), (m+1).to_timestamp() - pd.Timedelta(days=1))
    monthly_sl[m] = get_weekly_grade(idx_df, sample)

# 预计算收益序列
print("预计算收益序列...")
signals_data = []
bad = 0
for _, row in sig_strong.iterrows():
    code = row['code']
    price = float(row['price'])
    month = row['month']
    btype = row['type']
    sl_base = 0.94 if btype in ('2buy', '2plus3buy') else 0.93
    sl_ratio_inner = max(sl_base, monthly_sl.get(month, 0.8))
    date = row['date']
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
        sl_rel = sl_ratio_inner - 1
        if np.any(lows/price - 1 <= sl_rel): bad += 1; continue
        signals_data.append((
            lows/price - 1,
            closes/price - 1,
            highs/price - 1,
            sl_rel
        ))
    except: bad += 1
print(f"有效信号: {len(signals_data)} (skip {bad})")

# 回测函数
def bt_fast(signals_data, trigger, trail, half_pct=0, half_trail=0):
    pnls = []
    exit_counts = defaultdict(int)
    for low_rel, close_rel, high_rel, sl_rel in signals_data:
        n = len(close_rel)
        # 止损检测
        sl_hit = False
        for i in range(n):
            if low_rel[i] <= sl_rel:
                sl_hit = True; break
        if sl_hit:
            exit_counts['sl'] += 1
            pnls.append(sl_rel - 0.0006); continue
        # 触发点
        trigger_idx = None
        for i in range(n):
            if close_rel[i] >= trigger:
                trigger_idx = i; break
        if trigger_idx is None:
            exit_counts['to'] += 1
            pnls.append(close_rel[-1] - 0.0006); continue
        # 追踪回撤
        hw = close_rel[0]
        half_exit_rel = None
        tp_exit_rel = None
        for i in range(trigger_idx, n):
            if close_rel[i] > hw: hw = close_rel[i]
            dd = (hw - close_rel[i]) / (1 + hw)
            if half_pct > 0 and half_trail > 0 and half_exit_rel is None and dd >= half_trail:
                half_exit_rel = close_rel[i]
            if dd >= trail:
                tp_exit_rel = close_rel[i]; break
        if tp_exit_rel is not None:
            exit_counts['tp'] += 1
            if half_exit_rel is not None:
                pnl = (half_exit_rel * half_pct + tp_exit_rel * (1 - half_pct)) - 0.0006
            else:
                pnl = tp_exit_rel - 0.0006
            pnls.append(pnl)
        elif half_exit_rel is not None:
            exit_counts['to_half'] += 1
            pnl = half_exit_rel * half_pct + close_rel[-1] * (1 - half_pct) - 0.0006
            pnls.append(pnl)
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
    sl_n = exit_counts['sl']; tp_n = exit_counts['tp']
    to_n = exit_counts['to'] + exit_counts['to_half']
    return {
        'sharpe': sharpe, 'win_rate': wr, 'avg_pnl': avg,
        'max_dd': max_dd, 'n': total,
        'sl_pct': sl_n/total*100, 'tp_pct': tp_n/total*100, 'to_pct': to_n/total*100,
    }

# 网格搜索
print("\n开始网格搜索...")
triggers = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
half_trails = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.0]
half_pcts = [0, 0.33, 0.50, 0.67]
trails = [0.06, 0.08, 0.10, 0.12, 0.15]

results = []
combos = []
for trigger in triggers:
    for half_trail in half_trails:
        for half_pct in half_pcts:
            if half_trail == 0 and half_pct > 0: continue
            if half_pct == 0: half_trail = 0
            for trail in trails:
                if half_trail >= trail: continue
                combos.append((trigger, trail, half_pct, half_trail))

print(f"总组合: {len(combos)}")
done = 0
for trigger, trail, half_pct, half_trail in combos:
    done += 1
    r = bt_fast(signals_data, trigger, trail, half_pct, half_trail)
    if r:
        ht = f"{int(half_trail*100)}%" if half_trail > 0 else "-"
        hp = f"{int(half_pct*100)}%" if half_pct > 0 else "-"
        desc = f"tr{int(trigger*100)}% ht={ht} hp={hp} tl{int(trail*100)}%"
        r['desc'] = desc
        r['trigger'] = trigger; r['half_trail'] = half_trail
        r['half_pct'] = half_pct; r['trail'] = trail
        results.append(r)
    if done % 30 == 0:
        print(f"  {done}/{len(combos)} ({done/len(combos)*100:.0f}%)", flush=True)

print(f"\n完成 {len(results)} 组合\n")

results.sort(key=lambda x: -x['sharpe'])
print(f"{'#':>4} {'触发':>6} {'半仓回撤':>8} {'比例':>5} {'清仓':>6} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'止损%':>6} {'止盈%':>6}")
print("-"*70)
for i, r in enumerate(results[:25]):
    ht = f"{int(r['half_trail']*100)}%" if r['half_trail'] > 0 else "-"
    hp = f"{int(r['half_pct']*100)}%" if r['half_pct'] > 0 else "-"
    print(f"{i+1:>4} {int(r['trigger']*100):>5}% {ht:>8} {hp:>5} {int(r['trail']*100):>5}% "
          f"{r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} "
          f"{r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}%")

print("\n── 多维度Top5 ──")
for met, lab in [('sharpe','Sharpe'),('win_rate','胜率'),('avg_pnl','均盈'),('max_dd','DD最小')]:
    top = sorted(results, key=lambda x: -x[met])[:3]
    print(f"\n{lab}:")
    for r in top:
        ht = f"{int(r['half_trail']*100)}%" if r['half_trail'] > 0 else "-"
        hp = f"{int(r['half_pct']*100)}%" if r['half_pct'] > 0 else "-"
        print(f"  tr={int(r['trigger']*100)}% ht={ht} hp={hp} tl={int(r['trail']*100)}% → "
              f"{met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}%")

print("\n── 综合推荐（权重: Sharpe40% WR25% 均盈20% DD15%）──")
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
ht = f"{int(best['half_trail']*100)}%" if best['half_trail'] > 0 else "无"
hp = f"{int(best['half_pct']*100)}%" if best['half_pct'] > 0 else "无"
print(f"推荐: 触发{int(best['trigger']*100)}% 卖半回撤{ht} 卖半比例{hp} 清仓{int(best['trail']*100)}%")
print(f"  Sharpe={best['sharpe']:.3f} WR={best['win_rate']:.0f}% 均盈={best['avg_pnl']:+.2f}% DD={best['max_dd']:.1f}% 综合={best['composite']:.3f}")
