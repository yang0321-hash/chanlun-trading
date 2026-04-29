#!/usr/bin/env python3
"""
V4增强+fib=strong 强势市场验证
用已有信号数据，按fib_strength分组，看strong vs strongest在2026-03的表现差异
"""
import sys, os, pickle
sys.path.insert(0, '/workspace')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
from collections import defaultdict

print("加载信号...")
sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
print(f"信号总数: {len(sig_df)}")

sig_df['month'] = pd.to_datetime(sig_df['date']).dt.to_period('M')

# Fib分组统计
print("\nfib分布:")
print(sig_df['fib_strength'].value_counts())
print("\nfib×月分布:")
print(sig_df.groupby(['month','fib_strength']).size().unstack(fill_value=0))

def run_backtest_group(sig_df, data_map, months, fib_filter=None, label=''):
    mask = sig_df['month'].isin(months)
    df = sig_df[mask]
    if fib_filter:
        df = df[df['fib_strength'].isin(fib_filter)]
    if len(df) == 0:
        print(f"  {label}: 无数据")
        return None
    unique_dates = sorted(df['date'].unique())
    equity = 1_000_000.0
    equity_curve = [equity]
    active_slots = [None] * 5
    code_exit = {}
    cooldown = {}
    total_trades = 0
    winning = 0
    pnl_list = []

    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        for si in range(5):
            slot = active_slots[si]
            if not slot: continue
            ed = slot['exit_date']
            if isinstance(ed, str): ed = pd.Timestamp(ed)
            if ed <= d_ts:
                equity += (equity/5) * slot['pnl_pct']
                equity_curve.append(equity)
                total_trades += 1
                if slot['pnl_pct'] > 0: winning += 1
                pnl_list.append(slot['pnl_pct'])
                active_slots[si] = None
            code_exit[slot['code']] = d

        day_sigs = df[df['date'] == d]
        for _, row in day_sigs.iterrows():
            code, btype = row['code'], row['type']
            pos = int(row['pos'])
            price = row['price']
            sl_orig = row['sl_price']
            if code in cooldown and d_ts < cooldown[code]: continue
            if code in code_exit and code_exit[code] == d: continue
            si = next((i for i in range(5) if active_slots[i] is None), None)
            if si is None: break
            if code not in data_map: continue
            df_c = data_map[code]
            if pos >= len(df_c): continue

            # Fib分档SL
            fib = row.get('fib_strength', 'unknown')
            if fib == 'strongest':
                sl = max(sl_orig / 0.98, price * 0.90)
            elif fib == 'strong':
                sl = max(sl_orig, price * 0.94)
            else:
                sl = price * 0.94

            exit_reason = 'timeout'
            exit_price = float(df_c['close'].iloc[-1])
            actual_exit_idx = len(df_c) - 1
            max_exit = min(pos + 30*8, len(df_c)-1)
            tp_trail_start, tp_trail_pct = 0.03, 0.08

            for bi in range(pos+1, max_exit+1):
                if bi >= len(df_c): break
                low_bi = float(df_c['low'].iloc[bi])
                close_bi = float(df_c['close'].iloc[bi])
                if low_bi <= sl:
                    exit_price = sl
                    exit_reason = 'stop_loss'
                    actual_exit_idx = bi
                    break
                profit_pct = (close_bi - price) / price
                if profit_pct >= tp_trail_start:
                    dd = (close_bi - price) / close_bi
                    if dd >= tp_trail_pct:
                        exit_price = close_bi
                        exit_reason = 'trail_stop'
                        actual_exit_idx = bi
                        break

            pnl_pct = (exit_price - price) / price - 0.0003*2
            hold_days_actual = max(1, (actual_exit_idx - pos)//8)
            exit_date = d_ts + pd.Timedelta(days=hold_days_actual)
            active_slots[si] = {'code': code, 'btype': btype, 'pnl_pct': pnl_pct,
                                  'exit_date': exit_date, 'exit_reason': exit_reason}
            cooldown[code] = exit_date + pd.Timedelta(days=5)

    for si in range(5):
        slot = active_slots[si]
        if slot:
            d_end_ts = pd.Timestamp(unique_dates[-1])
            ed = slot['exit_date']
            if isinstance(ed, str): ed = pd.Timestamp(ed)
            if ed > d_end_ts:
                equity += (equity/5) * slot['pnl_pct']
                equity_curve.append(equity)
                total_trades += 1
                if slot['pnl_pct'] > 0: winning += 1
                pnl_list.append(slot['pnl_pct'])

    if total_trades == 0: return None
    eq = np.array(equity_curve)
    rets = np.diff(eq)/eq[:-1]
    sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    rmax = np.maximum.accumulate(eq)
    dd = (eq-rmax)/rmax
    max_dd = abs(dd.min())*100
    wr = winning/total_trades*100
    ap = np.mean(pnl_list)*100
    return {'sharpe':round(sharpe,2),'max_dd':round(max_dd,1),'win_rate':round(wr,1),
            'avg_pnl':round(ap,2),'total_trades':total_trades,'label':label}

print("\n" + "="*60)
print("V4增强fib=strong 回测对比（3月强势市场）")
print("="*60)

results = []
for fib_name, fib_list in [('strongest', ['strongest']), ('strong', ['strong']),
                             ('strong+strongest', ['strongest','strong'])]:
    for month_label, months in [('2026-03(强势)', [pd.Period('2026-03')]),
                                 ('2026-04(弱势)', [pd.Period('2026-04')])]:
        r = run_backtest_group(sig_df, data_map, months=months, fib_filter=fib_list,
                                label=f'{month_label} {fib_name}')
        if r:
            results.append(r)
            print(f"  {r['label']}: Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% "
                  f"WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% ({r['total_trades']}笔)")

print("\n--- 汇总（3月强势市场）---")
mar = [r for r in results if '03' in r['label']]
print(f"{'配置':<35} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'交易'}")
print('-'*65)
for r in sorted(mar, key=lambda x: x['sharpe'], reverse=True):
    print(f"{r['label']:<35} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['total_trades']:>5}")

with open('/workspace/v4_fib_strong_market_result.pkl','wb') as f:
    pickle.dump({'results':results}, f)
print("\n已保存: v4_fib_strong_market_result.pkl")
