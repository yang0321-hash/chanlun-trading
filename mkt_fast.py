#!/usr/bin/env python3
"""大盘择时优化 - 快速版"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, '/workspace')

print("加载数据...")
sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
sector_map = pickle.load(open('/workspace/sector_industry_map.pkl', 'rb'))
print(f"信号: {len(sig_df)}, 个股: {len(data_map)}")

# 沪指
base = '/workspace/tdx_data/sh/lday/sh000001.day'
rows = []
with open(base, 'rb') as f:
    data = f.read()
for i in range(len(data)//32):
    vals = struct.unpack('<8I', data[i*32:(i+1)*32])
    rows.append({'date': vals[0], 'close': vals[4]/100.0, 'volume': float(vals[6]),
                 'high': vals[2]/100.0, 'low': vals[3]/100.0})
idx_df = pd.DataFrame(rows)
idx_df['date'] = pd.to_datetime(idx_df['date'], format='%Y%m%d')
idx_df.set_index('date', inplace=True)
idx_df.sort_index(inplace=True)
print(f"沪指: {len(idx_df)} rows")

# 向量化计算沪指指标
close = idx_df['close'].astype(float)
volume = idx_df['volume'].astype(float)
ma5 = close.rolling(5, min_periods=1).mean()
ma10 = close.rolling(10, min_periods=1).mean()
ma20 = close.rolling(20, min_periods=1).mean()
ma5_dir = (ma5 > ma5.shift(1)).astype(int) - (ma5 < ma5.shift(1)).astype(int)
above_ma5 = (close > ma5).astype(int)
above_ma10 = (close > ma10).astype(int)
above_ma20 = (close > ma20).astype(int)

# 月度SL用周线数据批量算（向量化太复杂，用groupby）
wk = idx_df.groupby(idx_df.index.to_period('W')).agg(
    close=('close','last'))
wk.index = wk.index.to_timestamp('W')
cw = wk['close'].astype(float).values
# 预计算每个位置对应的月度SL
n_wk = len(cw)
ma5w = np.convolve(cw, np.ones(5)/5, mode='valid')
ma10w = np.convolve(cw, np.ones(10)/10, mode='valid')
# 每个周对应的月度SL（向前看4周评分）
monthly_sl_arr = np.full(n_wk, 0.93, dtype=float)
for i in range(4, n_wk):
    m5 = ma5w[min(i-4, len(ma5w)-1)]
    m10 = ma10w[min(i-4, len(ma10w)-1)]
    pm5 = ma5w[min(i-5, len(ma5w)-1)]
    l5w = (cw[i-1]-cw[i-6])/cw[i-6]*100 if i >= 6 else 0
    sc = sum([m5>m10, cw[i-1]>m5, m5>pm5, l5w>0])
    monthly_sl_arr[i] = 0.94 if sc>=4 else (0.80 if sc>=2 else 0.93)

# 为每个交易日分配月度SL（取该日所在周的月度SL）
wk_period = idx_df.index.to_period('W')
unique_wks = wk_period.unique()
wk_to_sl = {}
for i, wk_ts in enumerate(unique_wks):
    if i < len(monthly_sl_arr):
        wk_to_sl[wk_ts] = monthly_sl_arr[i]

# 大盘日线状态
daily_state_arr = np.full(len(idx_df), 'neutral', dtype=object)
for i in range(1, len(idx_df)):
    if ma5.iloc[i] > ma10.iloc[i] and above_ma5.iloc[i]: daily_state_arr[i] = 'bull'
    elif ma5.iloc[i] < ma10.iloc[i] and not above_ma5.iloc[i]: daily_state_arr[i] = 'bear'
    else: daily_state_arr[i] = 'neutral'

# 构建索引: date -> (monthly_sl, daily_state, mkt_metrics)
idx_dates = idx_df.index.tolist()
date_to_mkt = {}
for i, ts in enumerate(idx_dates):
    wk_p = ts.to_period('W')
    ms = wk_to_sl.get(wk_p, 0.93)
    ds = daily_state_arr[i]
    mkt = {
        'ma5': float(ma5.iloc[i]), 'ma10': float(ma10.iloc[i]),
        'above_ma5': int(above_ma5.iloc[i]), 'above_ma10': int(above_ma10.iloc[i]),
        'above_ma20': int(above_ma20.iloc[i]), 'ma5_dir': int(ma5_dir.iloc[i]),
    }
    date_to_mkt[ts.strftime('%Y-%m-%d')] = {'monthly_sl': ms, 'daily_state': ds, 'mkt': mkt}

print(f"大盘指标: {len(date_to_mkt)} 个交易日")

# 批量建records（向量化price/returns）
print("构建records...")
records = []
bad = 0
bar_arrays = {}  # code -> (low_rel, close_rel, high_rel, bar_indices)

# 用Python循环但避免iterrows
codes = sig_df['code'].values
dates = sig_df['date'].astype(str).str[:10].values
prices = sig_df['price'].values
btypes = sig_df['type'].values

n_total = len(sig_df)
print_every = n_total // 10

for idx in range(n_total):
    if idx > 0 and idx % print_every == 0:
        print(f"  进度 {idx}/{n_total} ({idx*100//n_total}%)", flush=True)
    
    code = codes[idx]
    date_str = dates[idx]
    price = float(prices[idx])
    btype = btypes[idx]
    
    if code not in data_map: bad += 1; continue
    df_c = data_map[code]
    
    # 找信号日期对应bar
    try:
        ts = pd.Timestamp(date_str)
        bi_list = df_c.index.get_indexer([ts], method='bfill')
        if bi_list[0] < 0: bad += 1; continue
        pos_bar = int(bi_list[0])
        n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)
        if loop_end <= pos_bar + 1: bad += 1; continue
        
        low_rel = df_c['low'].iloc[pos_bar+1:loop_end+1].astype(float).values / price - 1
        close_rel = df_c['close'].iloc[pos_bar+1:loop_end+1].astype(float).values / price - 1
        high_rel = df_c['high'].iloc[pos_bar+1:loop_end+1].astype(float).values / price - 1
        
        mkt_data = date_to_mkt.get(date_str, {'monthly_sl': 0.93, 'daily_state': 'neutral', 'mkt': {}})
        
        records.append({
            'code': code, 'btype': btype, 'price': price, 'date': date_str,
            'low_rel': low_rel, 'close_rel': close_rel, 'high_rel': high_rel,
            'monthly_sl': mkt_data['monthly_sl'],
            'daily_state': mkt_data['daily_state'],
            'mkt': mkt_data['mkt'],
            'sector': sector_map.get(code, '其他'),
        })
    except: bad += 1

print(f"有效records: {len(records)} (skip {bad})")

# 缓存
with open('/workspace/mkt_records.pkl', 'wb') as f:
    pickle.dump(records, f)
print("已缓存 /workspace/mkt_records.pkl")

# ── 回测函数 ────────────────────────────────────────────────────────────────
TP_TRIGGER = 0.06
TP_TRAIL = 0.06

def get_mkt_state(mkt, def_name):
    ma5_gt_ma10 = mkt.get('ma5', 0) > mkt.get('ma10', 0)
    price_gt_ma5 = mkt.get('above_ma5', 0) == 1
    ma5_dir = mkt.get('ma5_dir', 0)
    
    if def_name == 'def1':
        return 'bull' if (ma5_gt_ma10 and price_gt_ma5) else ('bear' if (not ma5_gt_ma10 and not price_gt_ma5) else 'neutral')
    elif def_name == 'def2':
        return 'bull' if (ma5_dir > 0 and ma5_gt_ma10) else ('bear' if (ma5_dir < 0 and not ma5_gt_ma10) else 'neutral')
    elif def_name == 'def3':
        return 'bull' if ma5_gt_ma10 else 'bear'
    elif def_name == 'def4':
        return 'bull' if mkt.get('above_ma20', 0) == 1 else 'bear'
    elif def_name == 'def5':
        if ma5_dir > 0 and ma5_gt_ma10 and price_gt_ma5: return 'bull'
        elif ma5_dir < 0 and not ma5_gt_ma10 and not price_gt_ma5: return 'bear'
        else: return 'neutral'
    return 'neutral'

def bt(mkt_def, bm, bem, nm):
    pnls = []; ec = defaultdict(int)
    msc = defaultdict(int)
    for rec in records:
        ms_val = rec['monthly_sl']; mkt_state = get_mkt_state(rec['mkt'], mkt_def)
        lr = rec['low_rel']; cr = rec['close_rel']; hr = rec['high_rel']; n = len(cr)
        msc[mkt_state] += 1
        mult = {'bull': bm, 'bear': bem, 'neutral': nm}.get(mkt_state, 1.0)
        sl_rel = ms_val * mult - 1
        if np.any(lr <= sl_rel): ec['sl']+=1; pnls.append(sl_rel-0.0006); continue
        ti = None
        for i in range(n):
            if cr[i] >= TP_TRIGGER: ti=i; break
        if ti is None: ec['to']+=1; pnls.append(cr[-1]-0.0006); continue
        hwm = hr[0]; exit_rel=None
        for i in range(ti, n):
            if hr[i] > hwm: hwm = hr[i]
            dd = (hwm - cr[i])/(1+hwm)
            if dd >= TP_TRAIL: exit_rel=cr[i]; break
        if exit_rel is not None: ec['tp']+=1; pnls.append(exit_rel-0.0006)
        else: ec['to']+=1; pnls.append(cr[-1]-0.0006)
    if not pnls: return None
    pnls = np.array(pnls)
    return dict(
        sharpe=0.04/pnls[pnls>0].std() if len(pnls[pnls>0])>0 else 0,
        win_rate=(pnls>0).mean()*100, avg_pnl=pnls.mean()*100,
        max_dd=abs(pnls.min())*100, n=len(pnls),
        sl_pct=ec['sl']/len(pnls)*100, tp_pct=ec['tp']/len(pnls)*100, to_pct=ec['to']/len(pnls)*100,
        mkt_dist=dict(msc),
    )

# ── A. 大盘择时定义对比 ─────────────────────────────────────────────────────
print("\n" + "="*85)
print("A. 大盘择时定义对比")
print("="*85)

configs_a = [
    ('def1', 0.8, 1.2, 0.95, 'A1: MA5>MA10+price>MA5 (原始)'),
    ('def2', 0.8, 1.2, 0.95, 'A2: MA5方向+MA5>MA10'),
    ('def3', 0.8, 1.2, 0.95, 'A3: 仅MA5>MA10'),
    ('def4', 0.8, 1.2, 0.95, 'A4: 仅价格>MA20'),
    ('def5', 0.8, 1.2, 0.95, 'A5: MA5方向+MA5>MA10+price>MA5'),
    ('def1', 1.0, 1.0, 1.0, 'A6: def1+无乘数(基准)'),
]
ra = []
for md, bm, bem, nm, desc in configs_a:
    r = bt(md, bm, bem, nm)
    if r: ra.append((desc, r))
ra.sort(key=lambda x: -x[1]['sharpe'])
print(f"\n{'定义':<40} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'DD%':>6} {'N':>6}")
print("-"*80)
for desc, r in ra:
    print(f"{desc:<40} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['max_dd']:>6.1f} {r['n']:>6}")
    print(f"  大盘分布: {r['mkt_dist']}")

# ── B. 大盘乘数网格 ─────────────────────────────────────────────────────────
print("\n" + "="*85)
best_def = ra[0][0].split(':')[0] if ra else 'def1'
print(f"B. 大盘乘数网格（def={best_def}）")
print("="*85)

configs_b = []
for bm in [0.7, 0.8, 0.9, 1.0, 1.1]:
    for bem in [1.0, 1.1, 1.2, 1.3]:
        for nm in [0.8, 0.9, 1.0, 1.1]:
            configs_b.append((best_def, bm, bem, nm, f'牛×{bm} 熊×{bem} 中×{nm}'))

rb = []
for md, bm, bem, nm, desc in configs_b:
    r = bt(md, bm, bem, nm)
    if r: rb.append((desc, r))
rb.sort(key=lambda x: -x[1]['sharpe'])
print(f"\n{'配置':<28} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'DD%':>6} {'SL%':>6} {'止盈%':>6} {'N':>6}")
print("-"*95)
for desc, r in rb[:20]:
    print(f"{desc:<28} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} "
          f"{r['max_dd']:>6.1f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['n']:>6}")

# ── 综合推荐 ────────────────────────────────────────────────────────────────
print("\n" + "="*85)
print("综合结论")
print("="*85)

sh_v = [r['sharpe'] for _, r in rb]
wr_v = [r['win_rate'] for _, r in rb]
av_v = [r['avg_pnl'] for _, r in rb]
dd_v = [r['max_dd'] for _, r in rb]
sh_min,sh_max = min(sh_v),max(sh_v)
wr_min,wr_max = min(wr_v),max(wr_v)
av_min,av_max = min(av_v),max(av_v)
dd_min,dd_max = min(dd_v),max(dd_v)
for desc, r in rb:
    ns=(r['sharpe']-sh_min)/(sh_max-sh_min+1e-8)
    nw=(r['win_rate']-wr_min)/(wr_max-wr_min+1e-8)
    na=(r['avg_pnl']-av_min)/(av_max-av_min+1e-8)
    nd=(dd_max-r['max_dd'])/(dd_max-dd_min+1e-8)
    r['composite']=0.4*ns+0.25*nw+0.2*na+0.15*nd

best_all = max(rb, key=lambda x: x[1]['composite'])
print(f"\n最优: {best_all[0]}")
print(f"  Sharpe={best_all[1]['sharpe']:.3f} WR={best_all[1]['win_rate']:.0f}% 均盈={best_all[1]['avg_pnl']:+.2f}% DD={best_all[1]['max_dd']:.1f}%")
print(f"  SL={best_all[1]['sl_pct']:.1f}% TP={best_all[1]['tp_pct']:.1f}% TO={best_all[1]['to_pct']:.1f}%")

for met,lab in [('sharpe','Sharpe'),('win_rate','胜率'),('avg_pnl','均盈'),('max_dd','DD最小')]:
    top=sorted(rb, key=lambda x:-x[1][met])[:3]
    print(f"\n{lab}Top3:")
    for d,r in top:
        print(f"  {d}: {met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}%")
