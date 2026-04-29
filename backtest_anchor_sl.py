#!/usr/bin/env python3
"""锚定1买低点的SL回测 - 对比固定SL vs 锚定SL"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, '/workspace')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
from pathlib import Path

sig_df = pd.read_pickle('/workspace/scanner_new_fw_signals_live.pkl')
sig_df['date_dt'] = pd.to_datetime(sig_df['date'])
recent = sig_df[sig_df['date_dt'] >= pd.Timestamp('today') - pd.Timedelta(days=30)].copy()
recent_codes = set(recent['code'].unique())
print(f'信号: {len(recent)}个, 涉及: {len(recent_codes)}只')

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
            data_map[code] = pd.DataFrame({
                'open': prices[:, 0], 'high': prices[:, 1], 'low': prices[:, 2],
                'close': prices[:, 3], 'volume': arr[:, 6].astype(np.int64)
            }, index=dates).sort_index()
        except: pass
print(f'加载: {len(data_map)}只')

def backtest(signals_df, label, sl_config=None, tp_pct=(0.03, 0.05), max_holding=20):
    """
    sl_config: dict, key=sl策略名, value=func(row, entry, df, pos) -> sl_price
    """
    results_by_strategy = {name: [] for name in sl_config}
    for _, row in signals_df.iterrows():
        code = row['code']
        if code not in data_map: continue
        df = data_map[code]
        pos = row['entry_idx']
        if pos + 1 >= len(df): continue
        entry = float(df['close'].iloc[pos + 1])
        if np.isnan(entry) or entry <= 0: continue

        # 各策略SL
        sl_prices = {}
        for name, fn in sl_config.items():
            sl_prices[name] = fn(row, entry, df, pos)

        for name in sl_config:
            sl = sl_prices[name]
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
            results_by_strategy[name].append({
                'code': code, 'date': row['date'], 'entry': entry,
                'pnl': pnl, 'exit': exit_r, 'hold': hold,
                'fib_strength': row.get('fib_strength', 'unknown'),
                'sl_price': sl_prices[name],
                'v4_enhanced': row.get('v4_enhanced', False),
            })
    return {name: pd.DataFrame(rows) for name, rows in results_by_strategy.items()}

def stats(df, label):
    if len(df) == 0: print(f'{label}: 无数据'); return
    wr = (df['pnl'] > 0).mean() * 100
    avg = df['pnl'].mean()
    std = df['pnl'].std()
    sl = (df['exit'] == 'SL').mean() * 100
    tp3 = (df['exit'] == 'TP3').mean() * 100
    tp5 = (df['exit'] == 'TP5').mean() * 100
    hold_avg = df['hold'].mean()
    sharpe = (avg / std * np.sqrt(252 / max(1, hold_avg))) if std > 0 else 0
    max_dd = df['pnl'].min()
    print(f'{label}: n={len(df)}, 胜率={wr:.1f}%, 均盈={avg:.3f}%, std={std:.3f}%, Sharpe={sharpe:.2f}')
    print(f'  SL={sl:.0f}% TP3={tp3:.0f}% TP5={tp5:.0f}% HOLD={(df["exit"]=="HOLD").mean()*100:.0f}% | 最大DD={max_dd:.2f}%, 均持仓={hold_avg:.1f}天')

# ============ SL策略定义 ============
# 基准: 固定6%SL
def sl_fixed_6pct(row, entry, df, pos):
    return entry * 0.94

# 锚定1买低点 × 0.98（标准2买止损）
def sl_1buy_anchor(row, entry, df, pos):
    # scanner的sl_price就是1buy_low*0.98
    sl_scanner = row.get('sl_price', np.nan)
    if not np.isnan(sl_scanner) and sl_scanner > 0:
        return sl_scanner
    # fallback: entry下方6%
    return entry * 0.94

# 新策略: fib分档SL
def sl_fib_adaptive(row, entry, df, pos):
    sl_scanner = row.get('sl_price', np.nan)
    if np.isnan(sl_scanner) or sl_scanner <= 0:
        return entry * 0.94
    fib = row.get('fib_strength', 'unknown')
    if fib == 'strongest':
        # 回调极浅，SL贴近1买低点，不打止损等TP
        # SL = 1buy_low * 1.00（几乎不主动止损）
        buy1_low = sl_scanner / 0.98 if sl_scanner > 0 else entry * 0.94
        return buy1_low  # 锚定1买低点
    elif fib == 'strong':
        # 标准: 1buy_low * 0.98
        return sl_scanner
    elif fib == 'medium':
        return entry * 0.94  # 固定6%
    else:
        return entry * 0.94  # 固定6%

sl_config = {
    '固定6%SL': sl_fixed_6pct,
    '锚定1买(标准)': sl_1buy_anchor,
    'Fib分档SL': sl_fib_adaptive,
}

print('\n' + '='*70)
print('【全部信号】三种SL策略对比')
print('='*70)
results = backtest(recent, '全部', sl_config=sl_config)
for name, df in results.items():
    stats(df, f'  {name}')

print('\n' + '='*70)
print('【仅fib=strongest】三种SL策略对比')
print('='*70)
strongest_df = recent[recent['fib_strength'] == 'strongest']
results_ss = backtest(strongest_df, 'strongest', sl_config=sl_config)
for name, df in results_ss.items():
    stats(df, f'  {name}')

print('\n' + '='*70)
print('【仅fib=strong】三种SL策略对比')
print('='*70)
strong_df = recent[recent['fib_strength'] == 'strong']
results_s = backtest(strong_df, 'strong', sl_config=sl_config)
for name, df in results_s.items():
    stats(df, f'  {name}')

print('\n' + '='*70)
print('【仅V4增强+fib强】三种SL策略对比')
print('='*70)
v4_fib_df = recent[(recent['v4_enhanced'] == True) & (recent['fib_strength'].isin(['strong', 'strongest']))]
results_vf = backtest(v4_fib_df, 'V4+fib强', sl_config=sl_config)
for name, df in results_vf.items():
    stats(df, f'  {name}')

print('\n' + '='*70)
print('【综合对比表 - Sharpe】')
print('='*70)
groups = [
    ('全部信号', recent),
    ('fib=strongest', strongest_df),
    ('fib=strong', strong_df),
    ('V4+fib强', v4_fib_df),
]
print(f'{"组合":<18} {"固定6%SL":>12} {"锚定1买":>12} {"Fib分档SL":>12}')
print('-'*60)
for name, df_sub in groups:
    if len(df_sub) == 0: continue
    r = backtest(df_sub, name, sl_config=sl_config)
    sh = {}
    for n, d in r.items():
        sh[n] = (d['pnl'].mean() / d['pnl'].std() * np.sqrt(252 / max(1, d['hold'].mean()))) if d['pnl'].std() > 0 else 0
    print(f'{name:<18} {sh.get("固定6%SL", 0):>12.2f} {sh.get("锚定1买(标准)", 0):>12.2f} {sh.get("Fib分档SL", 0):>12.2f}')

print('\n' + '='*70)
print('【综合对比表 - 均盈%】')
print('='*70)
print(f'{"组合":<18} {"固定6%SL":>12} {"锚定1买":>12} {"Fib分档SL":>12}')
print('-'*60)
for name, df_sub in groups:
    if len(df_sub) == 0: continue
    r = backtest(df_sub, name, sl_config=sl_config)
    avgs = {n: d['pnl'].mean() for n, d in r.items()}
    print(f'{name:<18} {avgs.get("固定6%SL", 0):>12.3f}% {avgs.get("锚定1买(标准)", 0):>12.3f}% {avgs.get("Fib分档SL", 0):>12.3f}%')

print('\n' + '='*70)
print('【Fib分档SL - fib=strongest 止损明细】')
print('='*70)
ss_fib = results_ss['Fib分档SL']
print(f'SL触发率: {(ss_fib["exit"]=="SL").mean()*100:.0f}%')
print(f'HOLD率: {(ss_fib["exit"]=="HOLD").mean()*100:.0f}%')
print(f'SL均值: {ss_fib["sl_price"].mean():.2f}')
print(f'SL < entry*0.94的数量: {(ss_fib["sl_price"] < ss_fib["entry"]*0.94).sum()}/{len(ss_fib)}')
print(f'SL = entry*0.94的数量(锚1买但1buy_low低于entry94%): {(ss_fib["sl_price"] == ss_fib["entry"]*0.94).sum()}/{len(ss_fib)}')
