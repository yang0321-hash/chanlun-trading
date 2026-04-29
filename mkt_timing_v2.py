#!/usr/bin/env python3
"""
大盘择时优化 - 重新计算records（含date/code/sector）
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, '/workspace')

# ── 加载原始数据 ────────────────────────────────────────────────────────────
print("加载数据...")
sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
sector_map = pickle.load(open('/workspace/sector_industry_map.pkl', 'rb'))
print(f"信号: {len(sig_df)}, 个股: {len(data_map)}, 行业映射: {len(sector_map)}")

# ── 加载沪指数据 ────────────────────────────────────────────────────────────
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
print(f"沪指: {len(idx_df)} rows {idx_df.index.min().date()} ~ {idx_df.index.max().date()}")

# ── 预计算沪指每日指标 ─────────────────────────────────────────────────────
def compute_idx_metrics(idx_df):
    """为idx_df每一行计算大盘指标"""
    close = idx_df['close'].astype(float)
    volume = idx_df['volume'].astype(float)
    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    ma5_prev = ma5.shift(1)
    ma10_prev = ma10.shift(1)
    vol5_avg = volume.rolling(5, min_periods=1).mean()
    trend = (close - ma60) / ma60
    above_ma5 = (close > ma5).astype(int)
    above_ma10 = (close > ma10).astype(int)
    above_ma20 = (close > ma20).astype(int)
    ma5_dir = (ma5 > ma5_prev).astype(int) - (ma5 < ma5_prev).astype(int)
    vol_ratio = volume / vol5_avg
    return pd.DataFrame({
        'close': close, 'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'trend': trend, 'above_ma5': above_ma5, 'above_ma10': above_ma10,
        'above_ma20': above_ma20, 'ma5_dir': ma5_dir, 'vol_ratio': vol_ratio,
    })

print("计算沪指指标...")
idx_metrics = compute_idx_metrics(idx_df)
# 转为dict: date_str -> metrics
idx_metrics_dict = {}
for ts, row in idx_metrics.iterrows():
    d = ts.strftime('%Y-%m-%d')
    idx_metrics_dict[d] = {
        'close': float(row['close']), 'ma5': float(row['ma5']), 'ma10': float(row['ma10']),
        'ma20': float(row['ma20']), 'ma60': float(row['ma60']), 'trend': float(row['trend']),
        'above_ma5': int(row['above_ma5']), 'above_ma10': int(row['above_ma10']),
        'above_ma20': int(row['above_ma20']), 'ma5_dir': int(row['ma5_dir']),
        'vol_ratio': float(row['vol_ratio']),
    }
print(f"沪指指标: {len(idx_metrics_dict)} 个交易日")

# ── 重新计算records（含date/code/sector/monthly_sl/daily_state） ──────────
print("计算月度SL和日线状态...")
# 用沪指周线数据计算月度SL（周线MA5方向）
wk = idx_df.groupby(idx_df.index.to_period('W')).agg(
    close=('close','last'), high=('high','max'), low=('low','min'))
wk.index = wk.index.to_timestamp('W')
cw = wk['close'].astype(float).values
ma5_w = np.convolve(cw, np.ones(5)/5, mode='valid')
ma10_w = np.convolve(cw, np.ones(10)/10, mode='valid')
# 周线score
def get_monthly_sl(ts):
    try:
        ti = idx_df.index.get_indexer([ts], method='bfill')[0]
        if ti < 0: return 0.94
        df_s = idx_df.iloc[max(0,ti-103):ti+1]
        if len(df_s) < 20: return 0.94
        wk2 = df_s.groupby(df_s.index.to_period('W')).agg(c=('close','last'))
        if len(wk2) < 20: return 0.94
        cw2 = wk2['c'].astype(float).values
        m5 = np.convolve(cw2, np.ones(5)/5, mode='valid')[-1]
        m10 = np.convolve(cw2, np.ones(10)/10, mode='valid')[-1]
        pm5 = np.convolve(cw2, np.ones(5)/5, mode='valid')[-2] if len(cw2) > 5 else m5
        l5w = (cw2[-1]-cw2[-6])/cw2[-6]*100 if len(cw2) >= 6 else 0
        sc = sum([m5>m10, float(df_s['close'].iloc[-1])>m5, m5>pm5, l5w>0])
        return 0.94 if sc>=4 else (0.80 if sc>=2 else 0.93)
    except: return 0.94

def get_daily_state(ts):
    try:
        ti = idx_df.index.get_indexer([ts], method='bfill')[0]
        if ti < 0: return 'neutral'
        df_s = idx_df.iloc[max(0,ti-19):ti+1]
        if len(df_s) < 10: return 'neutral'
        close = float(df_s['close'].iloc[-1])
        ma5 = float(df_s['close'].iloc[-5:].mean())
        ma10 = float(df_s['close'].iloc[-10:].mean())
        if ma5 > ma10 and close > ma5: return 'bull'
        elif ma5 < ma10 and close < ma5: return 'bear'
        else: return 'neutral'
    except: return 'neutral'

print("预计算信号records...")
records = []
bad = 0
tp_trigger = 0.06
tp_trail = 0.06

for _, row in sig_df.iterrows():
    code = row['code']
    price = float(row['price'])
    btype = row['type']
    date_str = str(row['date'])[:10]
    if code not in data_map: bad += 1; continue
    df_c = data_map[code]
    bi_list = df_c.index.get_indexer([pd.Timestamp(date_str)], method='bfill')
    if bi_list[0] < 0: bad += 1; continue
    pos_bar = bi_list[0]
    n = len(df_c)
    loop_end = min(pos_bar + 30, n - 1)
    if loop_end <= pos_bar + 1: bad += 1; continue
    try:
        lows = df_c['low'].iloc[pos_bar+1:loop_end+1].astype(float).values
        closes = df_c['close'].iloc[pos_bar+1:loop_end+1].astype(float).values
        highs = df_c['high'].iloc[pos_bar+1:loop_end+1].astype(float).values
        ts = pd.Timestamp(date_str)
        monthly_sl = get_monthly_sl(ts)
        daily_state = get_daily_state(ts)
        mkt = idx_metrics_dict.get(date_str, {})
        sector = sector_map.get(code, '其他')
        records.append({
            'code': code, 'btype': btype, 'price': price, 'date': date_str,
            'sector': sector,
            'low_rel': lows/price - 1,
            'close_rel': closes/price - 1,
            'high_rel': highs/price - 1,
            'monthly_sl': monthly_sl,
            'daily_state': daily_state,
            'mkt': mkt,
        })
    except Exception as e:
        bad += 1

print(f"有效records: {len(records)} (skip {bad})")

# 缓存
cache = '/workspace/mkt_records.pkl'
with open(cache, 'wb') as f:
    pickle.dump(records, f)
print(f"已缓存: {cache}")

# ── 大盘状态定义 ─────────────────────────────────────────────────────────────
def get_mkt_state(mkt, def_name):
    if not mkt: return 'neutral'
    ma5_gt_ma10 = mkt.get('ma5', 0) > mkt.get('ma10', 0)
    price_gt_ma5 = mkt.get('above_ma5', 0) == 1
    price_gt_ma20 = mkt.get('above_ma20', 0) == 1
    ma5_dir = mkt.get('ma5_dir', 0)
    trend = mkt.get('trend', 0)
    
    if def_name == 'def1':  # 原始: MA5>MA10 + price>MA5
        return 'bull' if (ma5_gt_ma10 and price_gt_ma5) else ('bear' if (not ma5_gt_ma10 and not price_gt_ma5) else 'neutral')
    elif def_name == 'def2':  # MA5方向 + MA5>MA10
        return 'bull' if (ma5_dir > 0 and ma5_gt_ma10) else ('bear' if (ma5_dir < 0 and not ma5_gt_ma10) else 'neutral')
    elif def_name == 'def3':  # MA5>MA10 简化
        return 'bull' if ma5_gt_ma10 else 'bear'
    elif def_name == 'def4':  # 价格>MA20
        return 'bull' if price_gt_ma20 else 'bear'
    elif def_name == 'def5':  # MA5方向 + MA5>MA10 + price>MA5
        if ma5_dir > 0 and ma5_gt_ma10 and price_gt_ma5: return 'bull'
        elif ma5_dir < 0 and not ma5_gt_ma10 and not price_gt_ma5: return 'bear'
        else: return 'neutral'
    else: return 'neutral'

# ── 回测函数 ────────────────────────────────────────────────────────────────
def bt(mkt_def, bm, bem, nm, cap_enabled=False, sector_filter=False, sector_data=None):
    pnls = []
    exit_counts = defaultdict(int)
    monthly_sl_counts = defaultdict(int)
    mkt_state_counts = defaultdict(int)
    
    for rec in records:
        monthly_sl_val = rec['monthly_sl']
        mkt = rec['mkt']
        mkt_state = get_mkt_state(mkt, mkt_def)
        sector = rec['sector']
        low_rel = rec['low_rel']
        close_rel = rec['close_rel']
        high_rel = rec['high_rel']
        n = len(close_rel)
        
        # 板块过滤
        if sector_filter and sector_data:
            s_avg_sl = sector_data.get(sector, 0.93)
            if s_avg_sl < 0.90: continue
        
        # 仓位上限
        if cap_enabled:
            cap = {'bull': 0.8, 'neutral': 0.5, 'bear': 0.2}.get(mkt_state, 0.5)
        else:
            cap = 1.0
        
        monthly_sl_counts[monthly_sl_val] += 1
        mkt_state_counts[mkt_state] += 1
        
        mult = {'bull': bm, 'bear': bem, 'neutral': nm}.get(mkt_state, 1.0)
        sl_rel = monthly_sl_val * mult - 1
        
        if np.any(low_rel <= sl_rel):
            exit_counts['sl'] += 1; pnls.append((sl_rel - 0.0006) * cap); continue
        
        ti = None
        for i in range(n):
            if close_rel[i] >= tp_trigger: ti = i; break
        if ti is None:
            exit_counts['to'] += 1; pnls.append((close_rel[-1] - 0.0006) * cap); continue
        
        hwm = high_rel[0]; tp_exit = None
        for i in range(ti, n):
            if high_rel[i] > hwm: hwm = high_rel[i]
            dd = (hwm - close_rel[i]) / (1 + hwm)
            if dd >= tp_trail: tp_exit = close_rel[i]; break
        if tp_exit is not None:
            exit_counts['tp'] += 1; pnls.append((tp_exit - 0.0006) * cap)
        else:
            exit_counts['to'] += 1; pnls.append((close_rel[-1] - 0.0006) * cap)
    
    if not pnls: return None
    pnls = np.array(pnls)
    total = len(pnls)
    wr = (pnls > 0).mean() * 100
    avg = pnls.mean() * 100
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = 0.04 / std_w if std_w > 1e-8 else 0
    max_dd = abs(pnls.min()) * 100
    sl_n = exit_counts['sl']; tp_n = exit_counts['tp']; to_n = exit_counts['to']
    return {
        'sharpe': sharpe, 'win_rate': wr, 'avg_pnl': avg, 'max_dd': max_dd, 'n': total,
        'sl_pct': sl_n/total*100, 'tp_pct': tp_n/total*100, 'to_pct': to_n/total*100,
        'mkt_dist': dict(mkt_state_counts),
    }

# ── 预计算各板块月度SL均值 ─────────────────────────────────────────────────
print("计算板块平均月度SL...")
sector_sl = defaultdict(list)
for rec in records:
    sector_sl[rec['sector']].append(rec['monthly_sl'])
sector_avg_sl = {s: np.mean(v) for s, v in sector_sl.items()}
sector_data = {s: v for s, v in sector_avg_sl.items()}
print(f"板块数: {len(sector_data)}")

# ── 大盘择时定义对比 ────────────────────────────────────────────────────────
print("\n" + "="*90)
print("A. 大盘择时定义对比（止损: monthly_sl × 乘数）")
print("="*90)

configs_a = [
    ('def1', 0.8, 1.2, 0.95, 'A1: MA5>MA10+price>MA5'),
    ('def2', 0.8, 1.2, 0.95, 'A2: MA5方向+MA5>MA10'),
    ('def3', 0.8, 1.2, 0.95, 'A3: 仅MA5>MA10'),
    ('def4', 0.8, 1.2, 0.95, 'A4: 仅价格>MA20'),
    ('def5', 0.8, 1.2, 0.95, 'A5: MA5方向+MA5>MA10+price>MA5'),
    ('def1', 1.0, 1.0, 1.0, 'A6: def1+无乘数(基准)'),
]
results_a = []
for mkt_def, bm, bem, nm, desc in configs_a:
    r = bt(mkt_def, bm, bem, nm)
    if r: results_a.append((desc, r))

results_a.sort(key=lambda x: -x[1]['sharpe'])
print(f"\n{'定义':<35} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'DD%':>6} {'SL%':>6} {'止盈%':>6} {'N':>6}")
print("-"*95)
for desc, r in results_a:
    print(f"{desc:<35} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} "
          f"{r['max_dd']:>6.1f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['n']:>6}")
    print(f"  大盘分布: {r['mkt_dist']}")

# ── 大盘乘数网格 ─────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("B. 大盘乘数网格（使用最优def）")
print("="*90)

best_def = results_a[0][0].split(':')[0] if results_a else 'def1'
print(f"使用: {best_def}")

configs_b = []
for bm in [0.7, 0.8, 0.9, 1.0, 1.1]:
    for bem in [1.0, 1.1, 1.2, 1.3]:
        for nm in [0.8, 0.9, 1.0, 1.1]:
            configs_b.append((best_def, bm, bem, nm, f'牛×{bm} 熊×{bem} 中×{nm}'))

results_b = []
for mkt_def, bm, bem, nm, desc in configs_b:
    r = bt(mkt_def, bm, bem, nm)
    if r: results_b.append((desc, r))

results_b.sort(key=lambda x: -x[1]['sharpe'])
print(f"\n{'配置':<28} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'DD%':>6} {'SL%':>6} {'止盈%':>6} {'N':>6}")
print("-"*95)
for desc, r in results_b[:20]:
    print(f"{desc:<28} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} "
          f"{r['max_dd']:>6.1f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['n']:>6}")

# ── 仓位上限效果 ────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("C. 仓位上限效果（使用最优def和乘数）")
print("="*90)

if results_b:
    best = results_b[0]
    print(f"基准: {best[0]} Sharpe={best[1]['sharpe']:.3f}")
    bm_best = float(best[0].split('熊')[0].split('×')[1])
    bem_best = float(best[0].split('熊×')[1].split(' 中')[0])
    nm_best = float(best[0].split('中×')[1])
    
    for cap_en, cap_desc in [(False,'C1: 无仓位上限'), (True,'C2: 牛市80% 震荡50% 熊市20%')]:
        r = bt(best_def, bm_best, bem_best, nm_best, cap_enabled=cap_en)
        if r:
            print(f"{cap_desc:<35} Sharpe={r['sharpe']:.3f} WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% DD={r['max_dd']:.1f}%")

# ── 板块共振（用月度SL作为板块强度代理）──────────────────────────────────
print("\n" + "="*90)
print("D. 板块共振过滤（月度SL < 0.90 过滤）")
print("="*90)

r_no_filt = bt(best_def, 0.8, 1.2, 0.95, sector_filter=False)
r_filt = bt(best_def, 0.8, 1.2, 0.95, sector_filter=True, sector_data=sector_data)
for desc, r in [('D1: 不过滤', r_no_filt), ('D2: 过滤弱板块(SL<0.90)', r_filt)]:
    if r:
        print(f"{desc:<35} Sharpe={r['sharpe']:.3f} WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% DD={r['max_dd']:.1f}% N={r['n']}")

# ── 综合推荐 ────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("综合结论")
print("="*90)

all_results = [(d, r) for d, r in results_b]
sh_v = [r['sharpe'] for _, r in all_results]
wr_v = [r['win_rate'] for _, r in all_results]
av_v = [r['avg_pnl'] for _, r in all_results]
dd_v = [r['max_dd'] for _, r in all_results]
sh_min,sh_max = min(sh_v),max(sh_v)
wr_min,wr_max = min(wr_v),max(wr_v)
av_min,av_max = min(av_v),max(av_v)
dd_min,dd_max = min(dd_v),max(dd_v)
for desc, r in all_results:
    ns=(r['sharpe']-sh_min)/(sh_max-sh_min+1e-8)
    nw=(r['win_rate']-wr_min)/(wr_max-wr_min+1e-8)
    na=(r['avg_pnl']-av_min)/(av_max-av_min+1e-8)
    nd=(dd_max-r['max_dd'])/(dd_max-dd_min+1e-8)
    r['composite']=0.4*ns+0.25*nw+0.2*na+0.15*nd

best_all = max(all_results, key=lambda x: x[1]['composite'])
print(f"\n最优配置: {best_all[0]}")
print(f"  Sharpe={best_all[1]['sharpe']:.3f} WR={best_all[1]['win_rate']:.0f}% 均盈={best_all[1]['avg_pnl']:+.2f}% DD={best_all[1]['max_dd']:.1f}%")
print(f"  SL={best_all[1]['sl_pct']:.1f}% TP={best_all[1]['tp_pct']:.1f}% TO={best_all[1]['to_pct']:.1f}%")
print(f"\n最优大盘定义: {results_a[0][0] if results_a else 'N/A'}")
