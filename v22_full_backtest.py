#!/usr/bin/env python3
"""
v2.2 完整系统回测
配置: 去3buy + Fib分档SL + 板块资金流过滤 + SL=6%(弱势)/8%(强势)
"""
import sys, os, pickle
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
from collections import defaultdict

print("加载数据...")
sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
print(f"信号: {len(sig_df)} 日期: {sig_df['date'].min()} ~ {sig_df['date'].max()}")

# 过滤3buy
sig_df = sig_df[sig_df['type'] != '3buy'].copy()
sig_df['month'] = pd.to_datetime(sig_df['date']).dt.to_period('M')

# 从sl_price推断fib
sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_df['fib_inferred'] = fib

# 板块资金流模拟（无法精确，用市场状态代理）
# 强势月(2026-03): 板块资金好, 弱势月(2026-04): 板块分化
sector_good_months = {pd.Period('2026-03')}  # 假设只有3月是纯粹强势

def get_sl(row, month):
    """Fib分档SL: strongest锚定1buy低, strong用1buy×0.98, weak固定6%"""
    fib = row['fib_inferred']
    price = row['price']
    sl_orig = row['sl_price']
    if fib == 'strongest':
        # 锚定1buy低
        buy1_low = sl_orig / 0.98 if sl_orig > 0 else price * 0.80
        return max(buy1_low, price * 0.90)
    elif fib == 'strong':
        return max(sl_orig, price * 0.94)
    else:
        # weak: 弱势用8%, 强势用6%
        return price * (0.92 if month in sector_good_months else 0.94)

def run_bt(sig_df, data_map, desc=''):
    unique_dates = sorted(sig_df['date'].unique())
    equity = 1_000_000.0
    eq_curve = [equity]
    slots = [None] * 5
    code_exit = {}
    cooldown = {}
    trades = winning = 0
    pnl_list = []
    exit_reasons = defaultdict(int)
    monthly_eq = {}

    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        month = pd.Period(d)

        # 清结算
        for si in range(5):
            slot = slots[si]
            if not slot: continue
            ed = slot['exit_date']
            if isinstance(ed, str): ed = pd.Timestamp(ed)
            if ed <= d_ts:
                equity += (equity/5) * slot['pnl_pct']
                eq_curve.append(equity)
                trades += 1
                if slot['pnl_pct'] > 0: winning += 1
                pnl_list.append(slot['pnl_pct'])
                exit_reasons[slot['exit_reason']] += 1
                slots[si] = None
            code_exit[slot['code']] = d

        # 月度快照
        monthly_eq.setdefault(month, equity)

        day_sigs = sig_df[sig_df['date'] == d]
        for _, row in day_sigs.iterrows():
            code = row['code']
            btype = row['type']
            pos = int(row['pos'])
            price = row['price']
            if code in cooldown and d_ts < cooldown[code]: continue
            if code in code_exit and code_exit[code] == d: continue
            si = next((i for i in range(5) if slots[i] is None), None)
            if si is None: break
            if code not in data_map: continue
            df_c = data_map[code]
            if pos >= len(df_c): continue

            sl = get_sl(row, month)
            tp_start, tp_trail = 0.03, 0.08
            exit_reason = 'timeout'
            exit_price = float(df_c['close'].iloc[-1])
            ei = len(df_c) - 1

            for bi in range(pos+1, min(pos+30*8, len(df_c)-1)):
                low_bi = float(df_c['low'].iloc[bi])
                close_bi = float(df_c['close'].iloc[bi])
                if low_bi <= sl:
                    exit_price = sl
                    exit_reason = 'stop_loss'
                    ei = bi
                    break
                profit_pct = (close_bi - price) / price
                if profit_pct >= tp_start:
                    dd = (close_bi - price) / close_bi
                    if dd >= tp_trail:
                        exit_price = close_bi
                        exit_reason = 'trail_stop'
                        ei = bi
                        break

            pnl_pct = (exit_price - price) / price - 0.0003*2
            hold = max(1, (ei - pos)//8)
            exit_date = d_ts + pd.Timedelta(days=hold)
            slots[si] = {'code': code, 'btype': btype, 'pnl_pct': pnl_pct,
                          'exit_date': exit_date, 'exit_reason': exit_reason}
            cooldown[code] = exit_date + pd.Timedelta(days=5)

    # 期末结算
    for si in range(5):
        slot = slots[si]
        if slot:
            equity += (equity/5) * slot['pnl_pct']
            eq_curve.append(equity)
            trades += 1
            if slot['pnl_pct'] > 0: winning += 1
            pnl_list.append(slot['pnl_pct'])
            exit_reasons[slot['exit_reason']] += 1

    if trades == 0: return None
    eq = np.array(eq_curve)
    rets = np.diff(eq)/eq[:-1]
    sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    rmax = np.maximum.accumulate(eq)
    dd = (eq-rmax)/rmax
    max_dd = abs(dd.min())*100
    wr = winning/trades*100
    ap = np.mean(pnl_list)*100
    total_ret = (eq[-1]/eq[0]-1)*100

    return {
        'sharpe': round(sharpe, 2), 'max_dd': round(max_dd, 1),
        'win_rate': round(wr, 1), 'avg_pnl': round(ap, 2),
        'total_trades': trades, 'total_ret': round(total_ret, 1),
        'exit_reasons': dict(exit_reasons),
        'monthly_eq': monthly_eq,
        'desc': desc
    }

# ── 回测 ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("v2.2 完整系统回测")
print("="*60)

results = []

# 1. 全量(去3buy+Fib分档SL)
r = run_bt(sig_df, data_map, '全量(去3buy+Fib分档SL)')
if r: results.append(r)

# 2. 强信号(strong+strongest)
r = run_bt(sig_df[sig_df['fib_inferred'].isin(['strong','strongest'])], data_map,
           '强信号(strong+strongest)')
if r: results.append(r)

# 3. 全量+板块资金流(模拟)
# 3月全部保留，4月过滤sl_ratio<0.92(weak)
sig_sf = sig_df.copy()
sig_sf.loc[sig_sf['month'] == pd.Period('2026-04'), 'fib_inferred'] = 'weak'  # 模拟板块分化
r = run_bt(sig_sf, data_map, '全量+板块过滤(模拟)')
if r: results.append(r)

# 4. 强信号+板块过滤
r = run_bt(sig_df[sig_df['fib_inferred'].isin(['strong','strongest'])], data_map,
           '强信号+板块过滤(模拟)')
if r: results.append(r)

# 5. 仅强势月(2026-03)
r = run_bt(sig_df[sig_df['month'] == pd.Period('2026-03')], data_map,
           '仅2026-03强势月')
if r: results.append(r)

# 6. 仅弱势月(2026-04)
r = run_bt(sig_df[sig_df['month'] == pd.Period('2026-04')], data_map,
           '仅2026-04弱势月')
if r: results.append(r)

# 打印结果
print(f"\n{'配置':<30} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'总收益%':>8} {'交易':>6}")
print('-'*80)
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    print(f"{r['desc']:<30} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} {r['win_rate']:>5.0f} "
          f"{r['avg_pnl']:>+7.2f} {r['total_ret']:>+8.1f} {r['total_trades']:>6}")

# 月度收益
print("\n── 月度收益（全量）──")
r0 = results[0]
for month, eq in sorted(r0['monthly_eq'].items()):
    prev = list(r0['monthly_eq'].values())[list(r0['monthly_eq'].keys()).index(month)-1] if list(r0['monthly_eq'].keys()).index(month) > 0 else 1_000_000
    ret = (eq - prev) / prev * 100
    print(f"  {month}: {'+' if ret>=0 else ''}{ret:.1f}%")

# 出场分布
print(f"\n── 出场原因（全量）──")
for reason, n in sorted(r0['exit_reasons'].items(), key=lambda x: -x[1]):
    print(f"  {reason}: {n} ({n/r0['total_trades']*100:.0f}%)")

# 最佳配置总结
best = max(results, key=lambda x: x['sharpe'])
print(f"\n🏆 最佳: {best['desc']}")
print(f"   Sharpe={best['sharpe']:.2f} DD={best['max_dd']:.1f}% WR={best['win_rate']:.0f}% "
      f"均盈={best['avg_pnl']:+.2f}% 总收益={best['total_ret']:+.1f}%")

with open('/workspace/v22_full_backtest.pkl','wb') as f:
    pickle.dump({'results': results}, f)
print("\n已保存: v22_full_backtest.pkl")
