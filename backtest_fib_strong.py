#!/usr/bin/env python3
"""fib=strong 专项回测 + 对比"""
import pickle, pandas as pd, numpy as np, sys, os
sys.path.insert(0, '/workspace')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)

sig_df = pd.read_pickle('/workspace/scanner_new_fw_signals_live.pkl')
sig_df['date_dt'] = pd.to_datetime(sig_df['date'])

# ============ 加载数据 ============
recent_codes = set(sig_df['code'].unique())
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
print(f'加载: {len(data_map)}只')

# ============ 回测函数 ============
def backtest(signals_df, label, sl_pct=0.06, tp_pct=(0.03, 0.05), max_holding=20):
    results = []
    for _, row in signals_df.iterrows():
        code = row['code']
        if code not in data_map: continue
        df = data_map[code]
        pos = row['entry_idx']
        if pos + 1 >= len(df): continue
        entry = float(df['close'].iloc[pos + 1])  # T+1开盘价
        sl = entry * (1 - sl_pct)  # 固定6%止损（相对入场价）
        tp1, tp2 = entry * (1 + tp_pct[0]), entry * (1 + tp_pct[1])
        if np.isnan(entry) or entry <= 0: continue
        pnl = None; exit_r = ''; hold = 0
        for d in range(pos + 1, min(pos + 1 + max_holding, len(df))):
            h = float(df['high'].iloc[d]); l = float(df['low'].iloc[d]); c = float(df['close'].iloc[d])
            hold = d - pos
            if l <= sl: pnl = (sl - entry) / entry * 100; exit_r = 'SL'; break
            pct = (c - entry) / entry * 100
            if pct >= tp_pct[1]: pnl = pct; exit_r = 'TP5'; break
            elif pct >= tp_pct[0]: pnl = pct; exit_r = 'TP3'; break
        if pnl is None:
            c = float(df['close'].iloc[-1]); hold = len(df) - pos - 1
            pnl = (c - entry) / entry * 100; exit_r = 'HOLD'
        results.append({
            'code': code, 'date': row['date'], 'entry': entry,
            'pnl': pnl, 'exit': exit_r, 'hold': hold,
            'fib_ratio': row.get('fib_ratio'), 'fib_strength': row.get('fib_strength', 'unknown'),
            'v4_enhanced': row.get('v4_enhanced', False),
            'type': row.get('type', ''),
        })
    return pd.DataFrame(results)

def print_stats(df_sub, label):
    if len(df_sub) == 0: print(f'{label}: 无数据'); return
    wr = (df_sub['pnl'] > 0).mean() * 100
    avg = df_sub['pnl'].mean()
    std = df_sub['pnl'].std()
    sl = (df_sub['exit'] == 'SL').mean() * 100
    tp3 = (df_sub['exit'] == 'TP3').mean() * 100
    tp5 = (df_sub['exit'] == 'TP5').mean() * 100
    hold = (df_sub['exit'] == 'HOLD').mean() * 100
    avg_hold = df_sub['hold'].mean()
    max_dd = df_sub['pnl'].min()
    sharpe = (avg / std * np.sqrt(252 / max(1, avg_hold))) if std > 0 else 0
    print(f'{label}:')
    print(f'  n={len(df_sub)}, 胜率={wr:.1f}%, 均盈={avg:.2f}%, 标准差={std:.2f}%, Sharpe={sharpe:.2f}')
    print(f'  SL={sl:.0f}% TP3={tp3:.0f}% TP5={tp5:.0f}% HOLD={hold:.0f}%')
    print(f'  最大亏损={max_dd:.2f}%, 平均持仓={avg_hold:.1f}天')
    return df_sub

# ============ 全部近30天信号 ============
recent = sig_df[sig_df['date_dt'] >= pd.Timestamp('today') - pd.Timedelta(days=30)].copy()
print(f'\n近30天信号总数: {len(recent)}')

print('\n' + '='*60)
print('【对比1】全部 vs V4增强 vs fib=strong')
print('='*60)
all_tr = backtest(recent, '全部信号')
print_stats(all_tr, '全部信号   ')

v4_df = recent[recent['v4_enhanced'] == True]
v4_tr = backtest(v4_df, 'V4增强通过')
print_stats(v4_tr, 'V4增强通过 ')

strong_df = recent[recent['fib_strength'] == 'strong']
strong_tr = backtest(strong_df, 'fib=strong ')
print_stats(strong_tr, 'fib=strong  ')

# ============ 按月份 ============
print('\n' + '='*60)
print('【对比2】fib=strong 按月分组')
print('='*60)
recent['month'] = recent['date_dt'].dt.to_period('M')
for month in sorted(recent['month'].unique())[-6:]:
    sub = recent[recent['month'] == month]
    print(f'\n{month}: n={len(sub)}个信号')
    s_tr = backtest(sub, f'  {month}')
    print_stats(s_tr, f'  全部 ')

# ============ fib=strong 月度分布 ============
print('\n' + '='*60)
print('【fib=strong 详细分析】')
print('='*60)
if len(strong_tr) > 0:
    print(f'\n类型分布:')
    print(strong_tr['type'].value_counts().to_string())
    print(f'\n信号日期分布:')
    strong_tr['date_dt'] = pd.to_datetime(strong_tr['date'])
    print(strong_tr.groupby(strong_tr['date_dt'].dt.date).size().to_string())
    print(f'\nTop5盈利:')
    print(strong_tr.nlargest(5, 'pnl')[['code','date','pnl','exit','hold']].to_string(index=False))
    print(f'\nBottom5亏损:')
    print(strong_tr.nsmallest(5, 'pnl')[['code','date','pnl','exit','hold']].to_string(index=False))
    print(f'\nfib_ratio分布:')
    print(strong_tr['fib_ratio'].describe().to_string())

# ============ 不同SL参数扫描 ============
print('\n' + '='*60)
print('【参数扫描】fib=strong 不同SL/止盈')
print('='*60)
for sl_pct in [0.05, 0.06, 0.08]:
    for tp2_pct in [0.04, 0.05, 0.08]:
        tr = backtest(strong_df, f'SL={int(sl_pct*100)}% TP={int(tp2_pct*100)}%',
                      sl_pct=sl_pct, tp_pct=(0.03, tp2_pct))
        if len(tr) > 0:
            wr = (tr['pnl'] > 0).mean() * 100
            avg = tr['pnl'].mean()
            sl = (tr['exit'] == 'SL').mean() * 100
            tp5 = (tr['exit'] == 'TP5').mean() * 100
            std = tr['pnl'].std()
            sharpe = (avg / std * np.sqrt(252 / max(1, tr['hold'].mean()))) if std > 0 else 0
            print(f'  SL={int(sl_pct*100)}% TP={int(tp2_pct*100)}%: n={len(tr)}, 胜率={wr:.1f}%, 均盈={avg:.2f}%, SL={sl:.0f}% TP={tp5:.0f}%, Sharpe={sharpe:.2f}')

print('\n【fib=strongest 对比】')
strongest_df = recent[recent['fib_strength'] == 'strongest']
strongest_tr = backtest(strongest_df, 'fib=strongest')
print_stats(strongest_tr, 'fib=strongest')
