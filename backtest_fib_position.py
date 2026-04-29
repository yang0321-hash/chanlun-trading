#!/usr/bin/env python3
"""fib仓位乘数回测对比"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, '/workspace')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
from pathlib import Path

# 加载信号
sig_df = pd.read_pickle('/workspace/scanner_new_fw_signals_live.pkl')
sig_df['date_dt'] = pd.to_datetime(sig_df['date'])
recent = sig_df[sig_df['date_dt'] >= pd.Timestamp('today') - pd.Timedelta(days=30)].copy()
recent_codes = set(recent['code'].unique())
print(f'信号数: {len(recent)}')

# 加载TDX数据
data_map = {}
for market in ['sz', 'sh']:
    lday_dir = Path(f'/workspace/tdx_data/{market}/lday')
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
            fp = float(arr[0, 1])
            if fp > 10_000_000:
                prices = np.frombuffer(arr[:, 1:5].tobytes(), dtype=np.float32).reshape(n, 4)
            else:
                prices = arr[:, 1:5] / 100.0
            df = pd.DataFrame({
                'open': prices[:, 0], 'high': prices[:, 1], 'low': prices[:, 2],
                'close': prices[:, 3], 'volume': arr[:, 6].astype(np.int64)
            }, index=dates).sort_index()
            data_map[code] = df
        except: pass
print(f'加载: {len(data_map)}只')

def backtest(signals_df, label, pos_mult_fn=None, sl_pct=0.06, tp_pct=(0.03, 0.05), max_holding=20):
    """
    pos_mult_fn: func(signal_row) -> float, 仓位乘数
    返回: (results_df, summary_dict)
    """
    results = []
    for _, row in signals_df.iterrows():
        code = row['code']
        if code not in data_map: continue
        df = data_map[code]
        pos = row['entry_idx']
        if pos + 1 >= len(df): continue
        entry = float(df['close'].iloc[pos + 1])
        if np.isnan(entry) or entry <= 0: continue

        # 仓位乘数
        mult = pos_mult_fn(row) if pos_mult_fn else 1.0
        # SL: 固定入场价百分比，仓位影响资金利用率但不影响单笔pnl%
        sl = entry * (1 - sl_pct)
        tp1, tp2 = entry * (1 + tp_pct[0]), entry * (1 + tp_pct[1])

        pnl = None; exit_r = ''; hold = 0
        for d in range(pos + 1, min(pos + 1 + max_holding, len(df))):
            l = float(df['low'].iloc[d]); c = float(df['close'].iloc[d])
            hold = d - pos
            if l <= sl:
                pnl = (sl - entry) / entry * 100; exit_r = 'SL'; break
            pct = (c - entry) / entry * 100
            if pct >= tp_pct[1]: pnl = pct; exit_r = 'TP5'; break
            elif pct >= tp_pct[0]: pnl = pct; exit_r = 'TP3'; break
        if pnl is None:
            c = float(df['close'].iloc[-1]); hold = len(df) - pos - 1
            pnl = (c - entry) / entry * 100; exit_r = 'HOLD'

        # 收益 = 单笔pnl% * 仓位乘数
        results.append({
            'code': code, 'date': row['date'], 'entry': entry,
            'pnl': pnl * mult,  # 仓位调整后收益
            'raw_pnl': pnl,
            'mult': mult,
            'exit': exit_r, 'hold': hold,
            'fib_strength': row.get('fib_strength', 'unknown'),
            'v4_enhanced': row.get('v4_enhanced', False),
        })
    return pd.DataFrame(results)

def stats(df_sub, label):
    if len(df_sub) == 0:
        print(f'{label}: 无数据'); return None
    wr = (df_sub['pnl'] > 0).mean() * 100
    avg = df_sub['pnl'].mean()
    std = df_sub['pnl'].std()
    sl = (df_sub['exit'] == 'SL').mean() * 100
    tp3 = (df_sub['exit'] == 'TP3').mean() * 100
    tp5 = (df_sub['exit'] == 'TP5').mean() * 100
    hold_avg = df_sub['hold'].mean()
    sharpe = (avg / std * np.sqrt(252 / max(1, hold_avg))) if std > 0 else 0
    max_dd = df_sub['pnl'].min()
    print(f'{label}:')
    print(f'  n={len(df_sub)}, 胜率={wr:.1f}%, 均盈={avg:.3f}%, std={std:.3f}%, Sharpe={sharpe:.2f}')
    print(f'  SL={sl:.0f}% TP3={tp3:.0f}% TP5={tp5:.0f}% HOLD={(df_sub["exit"]=="HOLD").mean()*100:.0f}%')
    print(f'  最大亏损={max_dd:.2f}%, 平均持仓={hold_avg:.1f}天')
    return {'n': len(df_sub), 'wr': wr, 'avg': avg, 'std': std, 'sharpe': sharpe, 'max_dd': max_dd}

print('\n' + '='*65)
print('【对比1】旧策略(无仓位乘数) vs 新策略(有仓位乘数) - 全部信号')
print('='*65)

# 旧策略: 无仓位乘数
old_tr = backtest(recent, '旧策略', pos_mult_fn=None)
stats(old_tr, '旧策略(无乘数)')

# 新策略: fib仓位乘数
def new_mult(row):
    fib = row.get('fib_strength', 'unknown')
    if fib == 'strong': return 1.0
    elif fib == 'strongest': return 0.7
    elif fib == 'medium': return 0.85
    else: return 0.8

new_tr = backtest(recent, '新策略(fib乘数)', pos_mult_fn=new_mult)
stats(new_tr, '新策略(fib乘数)')

print('\n' + '='*65)
print('【对比2】仅fib=strong信号 - 旧vs新')
print('='*65)
strong_df = recent[recent['fib_strength'] == 'strong']
old_s = backtest(strong_df, '旧策略')
new_s = backtest(strong_df, '新策略', pos_mult_fn=new_mult)
stats(old_s, '旧策略(strong)')
stats(new_s, '新策略(strong)')

print('\n' + '='*65)
print('【对比3】仅fib=strongest信号 - 旧vs新')
print('='*65)
strongest_df = recent[recent['fib_strength'] == 'strongest']
old_ss = backtest(strongest_df, '旧策略')
new_ss = backtest(strongest_df, '新策略', pos_mult_fn=new_mult)
stats(old_ss, '旧策略(strongest)')
stats(new_ss, '新策略(strongest)')

print('\n' + '='*65)
print('【对比4】V4增强通过 - 旧vs新')
print('='*65)
v4_df = recent[recent['v4_enhanced'] == True]
old_v4 = backtest(v4_df, '旧策略')
new_v4 = backtest(v4_df, '新策略', pos_mult_fn=new_mult)
stats(old_v4, '旧策略(V4)')
stats(new_v4, '新策略(V4)')

print('\n' + '='*65)
print('【对比5】V4增强 + fib强信号 - 旧vs新')
print('='*65)
v4_strong_df = recent[(recent['v4_enhanced'] == True) & (recent['fib_strength'].isin(['strong', 'strongest']))]
old_vs = backtest(v4_strong_df, '旧策略')
new_vs = backtest(v4_strong_df, '新策略', pos_mult_fn=new_mult)
stats(old_vs, '旧策略(V4+fib强)')
stats(new_vs, '新策略(V4+fib强)')

print('\n' + '='*65)
print('【综合结论】')
print('='*65)
rows = [
    ('全部信号', old_tr, new_tr),
    ('fib=strong', old_s, new_s),
    ('fib=strongest', old_ss, new_ss),
    ('V4增强通过', old_v4, new_v4),
    ('V4+fib强', old_vs, new_vs),
]
print(f'{"组合":<18} {"旧Sharpe":>10} {"新Sharpe":>10} {"变化":>10} {"旧均盈":>10} {"新均盈":>10}')
print('-'*70)
for name, old, new in rows:
    s_old = old['pnl'].mean()/old['pnl'].std()*np.sqrt(252/max(1,old['hold'].mean())) if old['pnl'].std()>0 else 0
    s_new = new['pnl'].mean()/new['pnl'].std()*np.sqrt(252/max(1,new['hold'].mean())) if new['pnl'].std()>0 else 0
    chg = s_new - s_old
    arrow = '↑' if chg > 0 else '↓' if chg < 0 else '→'
    print(f'{name:<18} {s_old:>10.2f} {s_new:>10.2f} {arrow}{abs(chg):>9.2f} {old["pnl"].mean():>10.3f}% {new["pnl"].mean():>10.3f}%')
