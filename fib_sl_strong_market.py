#!/usr/bin/env python3
"""
Fib分档SL强势市场回测
用预生成信号（已有sl_price），按sl_price与1buy_low的差距推断fib_strength
- sl_price极接近1buy_low（如 sl/price > 0.97）→ strongest（锚定1buy低）
- sl_price接近1buy_low（sl/price 0.94-0.97）→ strong（1buy×0.98）
- sl_price固定6%（sl/price ≈ 0.94）→ medium/weak/unknown

对比: 2026-03(强势) vs 2026-04(弱势) 的Fib分档SL效果
"""
import sys, os, time, pickle
sys.path.insert(0, '/workspace')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
from collections import defaultdict

print("加载信号...")
sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
print(f"信号总数: {len(sig_df)}  日期范围: {sig_df['date'].min()} ~ {sig_df['date'].max()}")

# 推断fib_strength from sl_price
# sl_ratio = sl_price / price → strongest: >0.97, strong: 0.93-0.97, unknown: <0.93
def infer_fib(sig_df):
    sl_ratio = sig_df['sl_price'] / sig_df['price']
    fib = pd.Series('unknown', index=sig_df.index)
    fib[sl_ratio > 0.97] = 'strongest'
    fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
    fib[sl_ratio <= 0.92] = 'weak'
    return fib

sig_df['fib_inferred'] = infer_fib(sig_df)
print(f"\nfib分布:\n{sig_df['fib_inferred'].value_counts()}")

# 按月分组
sig_df['month'] = pd.to_datetime(sig_df['date']).dt.to_period('M')
print(f"\n月度信号数:\n{sig_df.groupby('month').size().to_string()}")

# ============================================================
# 回测函数: Fib分档SL vs 固定6%SL
# ============================================================
def run_backtest(sig_df, data_map, use_fib_sl=False, months=None, fib_filter=None):
    if months:
        mask = sig_df['month'].isin(months)
        df = sig_df[mask]
    else:
        df = sig_df
    if fib_filter:
        df = df[df['fib_inferred'].isin(fib_filter)]
    if len(df) == 0:
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
            if not slot:
                continue
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

            # Fib分档SL: strongest→锚定1buy低, strong→1buy×0.98, weak→固定6%
            if use_fib_sl:
                fib = row['fib_inferred']
                if fib == 'strongest':
                    # 锚定1buy低，让震荡穿过
                    sl = max(sl_orig / 0.98, price * 0.90)  # 最宽松
                elif fib == 'strong':
                    sl = max(sl_orig, price * 0.94)
                else:
                    sl = price * 0.94  # 固定6%
            else:
                sl = price * 0.94  # 统一固定6%SL

            exit_reason = 'timeout'
            exit_price = float(df_c['close'].iloc[-1])
            actual_exit_idx = len(df_c) - 1
            max_exit = min(pos + 30*8, len(df_c)-1)
            tp_trail_start = 0.03
            tp_trail_pct = 0.08

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

            active_slots[si] = {
                'code': code, 'btype': btype,
                'pnl_pct': pnl_pct, 'exit_date': exit_date,
                'exit_reason': exit_reason
            }
            cooldown[code] = exit_date + pd.Timedelta(days=5)

    for si in range(5):
        slot = active_slots[si]
        if slot:
            d_ts_end = unique_dates[-1]
            d_end_ts = pd.Timestamp(d_ts_end)
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
            'avg_pnl':round(ap,2),'total_trades':total_trades}

# ============================================================
# 6组对比实验
# ============================================================
print("\n" + "="*60)
print("6组对比: Fib分档SL vs 固定6%SL × 3月 vs 4月")
print("="*60)

results = []
for month_label, month_per in [('2026-03(强势)', [pd.Period('2026-03')]),
                                 ('2026-04(弱势)', [pd.Period('2026-04')])]:
    for use_fib, label in [(False,'固定6%SL'), (True,'Fib分档SL')]:
        r = run_backtest(sig_df, data_map, use_fib_sl=use_fib, months=month_per)
        if r:
            tag = f"{month_label} {label}"
            print(f"\n{tag}:")
            print(f"  Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% ({r['total_trades']}笔)")
            results.append({'label':tag, **r})

# strong+strongest子集
print("\n" + "="*60)
print("strong+strongest子集对比")
print("="*60)
for month_label, month_per in [('2026-03(强势)', [pd.Period('2026-03')]),
                                 ('2026-04(弱势)', [pd.Period('2026-04')])]:
    for use_fib, label in [(False,'固定6%SL'), (True,'Fib分档SL')]:
        r = run_backtest(sig_df, data_map, use_fib_sl=use_fib,
                         months=month_per, fib_filter=['strongest','strong'])
        if r:
            tag = f"{month_label} {label}(强信号)"
            print(f"\n{tag}:")
            print(f"  Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% ({r['total_trades']}笔)")
            results.append({'label':tag, **r})

print("\n--- 汇总 ---")
print(f"{'配置':<35} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'交易'}")
print('-'*65)
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    print(f"{r['label']:<35} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['total_trades']:>5}")

with open('/workspace/fib_sl_strong_market_result.pkl','wb') as f:
    pickle.dump({'results':results}, f)
print("\n已保存: fib_sl_strong_market_result.pkl")
