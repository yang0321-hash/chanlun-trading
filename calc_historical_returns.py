#!/usr/bin/env python3
"""
calc_historical_returns.py - 只加载70只有信号股票
"""
import sys, os
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd

print("=" * 60)
print("历史T+5收益 (70只有信号股票, 交易日计算)")
print("=" * 60)

def read_tdx_day(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    if n == 0: return None
    arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
    dates = arr[:, 0].astype(str)
    return pd.DataFrame({
        'datetime': pd.to_datetime(dates, format='%Y%m%d'),
        'close': arr[:, 4]/100.0,
    }).set_index('datetime').sort_index()

# ── 加载信号 ─────────────────────────────────────────────────
sig = pd.read_pickle('/workspace/historical_signals_full.pkl')
print(f"信号: {len(sig)} 个, {sig['code'].nunique()} 只")

# ── 加载70只有信号股票的价格 ───────────────────────────────────
tdx_dir = '/workspace/tdx_data'
target_codes = set(sig['code'].unique())
print(f"目标股票: {len(target_codes)} 只")

price_dict = {}
trade_cal = []
loaded = 0

for m in ['sz', 'sh']:
    d = f'{tdx_dir}/{m}/lday'
    if not os.path.exists(d): continue
    for fname in os.listdir(d):
        if not fname.endswith('.day') or fname.endswith('.bak'): continue
        code_raw = fname[2:8].upper()
        mkt = 'SZ' if m == 'sz' else 'SH'
        code = f'{code_raw}.{mkt}'
        if code not in target_codes: continue
        
        path = os.path.join(d, fname)
        df = read_tdx_day(path)
        if df is None: continue
        for dt, row in df.iterrows():
            price_dict[(code, dt.strftime('%Y%m%d'))] = float(row['close'])
        if not trade_cal:
            trade_cal = list(df.index)
        loaded += 1

print(f"已加载: {loaded} 只股票, 价格字典: {len(price_dict)} 条")
print(f"交易日: {len(trade_cal)} 天 ({trade_cal[0].date()} ~ {trade_cal[-1].date()})")

# ── 交易日索引 ──────────────────────────────────────────────────
td_dates = [t.strftime('%Y%m%d') for t in trade_cal]

def get_tn_price(code, sig_ts, n):
    try:
        pos = trade_cal.index(sig_ts)
    except ValueError:
        return None
    tp = pos + n
    if tp >= len(trade_cal): return None
    return price_dict.get((code, td_dates[tp]))

# ── 计算T+5 ────────────────────────────────────────────────────
print("\n计算T+5收益...")
results = []
for _, row in sig.iterrows():
    code = row['code']
    sig_ts = pd.Timestamp(str(row['date'])[:10])
    
    p_t1 = get_tn_price(code, sig_ts, 1)
    p_t5 = get_tn_price(code, sig_ts, 5)
    
    if p_t1 and p_t5 and p_t1 > 0:
        ret = (p_t5 - p_t1) / p_t1 * 100
        results.append({**row.to_dict(), 'ret_5d': ret, 'p_t1': p_t1, 'p_t5': p_t5})

df = pd.DataFrame(results)
print(f"有效T+5: {len(df)} / {len(sig)} ({len(df)/len(sig)*100:.1f}%)")

if len(df) == 0:
    print("错误: 无有效数据"); sys.exit(1)

# ── 统计分析 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== T+5 收益 ===")
print("=" * 60)
r = df['ret_5d']
print(f"均值: {r.mean():+.2f}%")
print(f"中位数: {r.median():+.2f}%")
print(f"标准差: {r.std():.2f}%")
print(f">0%: {(r>0).mean()*100:.1f}%")
print(f">3%: {(r>3).mean()*100:.1f}%")
print(f">5%: {(r>5).mean()*100:.1f}%")
print(f"<0%: {(r<0).mean()*100:.1f}%")
print(f"<-5%: {(r<-5).mean()*100:.1f}%")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p}: {np.percentile(r, p):+.2f}%")

# ── 按月份 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 月度 T+5 ===")
print("=" * 60)
for ym in sorted(df['date'].str[:6].unique()):
    g = df[df['date'].str[:6] == ym]['ret_5d']
    n = len(g)
    if n >= 3:
        print(f"{ym}: n={n:3d}, 均={g.mean():+6.2f}%, 中={g.median():+6.2f}%, >0:{(g>0).mean()*100:5.0f}%, std={g.std():5.2f}%")

# ── 按类型 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 类型 T+5 ===")
print("=" * 60)
for t, g in df.groupby('type'):
    print(f"{t}: n={len(g):3d}, 均={g['ret_5d'].mean():+6.2f}%, 中={g['ret_5d'].median():+6.2f}%, >0:{(g['ret_5d']>0).mean()*100:5.0f}%")

# ── 质量分层 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== pos_size分层 ===")
print("=" * 60)
hq = df[df['pos_size'] > 0.15]
lq = df[df['pos_size'] <= 0.15]
print(f"高质量(pos>0.15): {len(hq)}个, 均={hq['ret_5d'].mean():+.2f}%, 中={hq['ret_5d'].median():+.2f}%")
print(f"低质量(pos≤0.15): {len(lq)}个, 均={lq['ret_5d'].mean():+.2f}%, 中={lq['ret_5d'].median():+.2f}%")
print(f"差异: {hq['ret_5d'].mean() - lq['ret_5d'].mean():+.2f}pp")

# ── 对比234样本 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 跨期对比 ===")
print("=" * 60)
recent = pd.read_pickle('/workspace/backtest_with_real_returns.pkl')['ret_5d'].dropna()
hist = df['ret_5d']
print(f"近期样本(234个): 均={recent.mean():+.2f}%, 中={recent.median():+.2f}%, >0:{(recent>0).mean()*100:.0f}%")
print(f"历史样本({len(hist)}个): 均={hist.mean():+.2f}%, 中={hist.median():+.2f}%, >0:{(hist>0).mean()*100:.0f}%")
print(f"差异: {hist.mean() - recent.mean():+.2f}pp")
print(f"\n结论: 历史样本 vs 近期样本 {'一致性高' if abs(hist.mean() - recent.mean()) < 1.0 else '存在差异'}")

df.to_pickle('/workspace/historical_returns_fixed.pkl')
print(f"\n已保存 historical_returns_fixed.pkl")
