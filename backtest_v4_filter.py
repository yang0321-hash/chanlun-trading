#!/usr/bin/env python3
"""V4增强信号回测脚本"""
import pickle, pandas as pd, numpy as np, sys, os
sys.path.insert(0, '/workspace')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)

sig_df = pd.read_pickle('/workspace/scanner_new_fw_signals_live.pkl')
sig_df['date_dt'] = pd.to_datetime(sig_df['date'])
recent = sig_df[sig_df['date_dt'] >= pd.Timestamp('today') - pd.Timedelta(days=30)].copy()
recent_codes = set(recent['code'].unique())
print(f'近30天信号: {len(recent)}个, 涉及: {len(recent_codes)}只')

data_map = {}
from pathlib import Path
tdx_root = Path('/workspace/tdx_data')
for market in ['sz', 'sh']:
    lday_dir = tdx_root / market / 'lday'
    if not lday_dir.exists(): continue
    for fpath in lday_dir.glob('*.day'):
        fname = fpath.name
        code = fname.replace('.day', '').upper() + ('.SZ' if fname.startswith('sz') else '.SH')
        if code not in recent_codes: continue
        try:
            data = fpath.read_bytes()
            n = len(data) // 32
            if n < 120: continue
            arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
            dates = pd.to_datetime(arr[:, 0].astype(str), format='%Y%m%d')
            first_price = float(arr[0, 1])
            if first_price > 10_000_000:
                prices = np.frombuffer(arr[:, 1:5].tobytes(), dtype=np.float32).reshape(n, 4)
            else:
                prices = arr[:, 1:5] / 100.0
            volumes = arr[:, 6].astype(np.int64)
            df = pd.DataFrame({
                'open': prices[:, 0], 'high': prices[:, 1], 'low': prices[:, 2],
                'close': prices[:, 3], 'volume': volumes
            }, index=dates).sort_index()
            data_map[code] = df
        except: pass
print(f'加载数据: {len(data_map)}只')

results = []
for _, row in recent.iterrows():
    code = row['code']
    if code not in data_map: continue
    df = data_map[code]
    pos = row['entry_idx']
    if pos + 1 >= len(df): continue
    entry = float(df['close'].iloc[pos + 1])
    sl = row['sl_price']
    if np.isnan(entry) or entry <= 0: continue
    pnl = None; exit_r = ''
    for d in range(pos + 1, min(pos + 21, len(df))):
        l = float(df['low'].iloc[d])
        c = float(df['close'].iloc[d])
        if l <= sl:
            pnl = (sl - entry) / entry * 100; exit_r = 'SL'; break
        pct = (c - entry) / entry * 100
        if pct >= 5: pnl = pct; exit_r = 'TP5'; break
        elif pct >= 3: pnl = pct; exit_r = 'TP3'; break
    if pnl is None:
        pnl = (float(df['close'].iloc[-1]) - entry) / entry * 100; exit_r = 'HOLD'
    results.append({'code': code, 'date': row['date'], 'entry': entry, 'pnl': pnl, 'exit': exit_r,
        'fib_strength': row.get('fib_strength', 'unknown'), 'v4_enhanced': row.get('v4_enhanced', False)})

tr = pd.DataFrame(results)
print(f'回测: {len(tr)}个')

def stat(df_sub, label):
    if len(df_sub)==0: return
    wr=(df_sub['pnl']>0).mean()*100; avg=df_sub['pnl'].mean()
    sl=(df_sub['exit']=='SL').mean()*100; tp3=(df_sub['exit']=='TP3').mean()*100
    tp5=(df_sub['exit']=='TP5').mean()*100; hold=(df_sub['exit']=='HOLD').mean()*100
    print(f'{label}: n={len(df_sub)}, 胜率{wr:.1f}%, 均盈{avg:.2f}%, SL{sl:.0f}% TP3{tp3:.0f}% TP5{tp5:.0f}% HOLD{hold:.0f}%')

print('\n=== 回测对比 ===')
stat(tr, '全部信号   ')
stat(tr[tr['v4_enhanced']==True], 'V4增强通过 ')
for fs in ['strongest','strong','medium','weak']:
    sub=tr[tr['fib_strength']==fs]
    stat(sub, f'  {fs}')
print('\nTop10:'); print(tr.nlargest(10,'pnl')[['code','date','pnl','exit','fib_strength']].to_string(index=False))