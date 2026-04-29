#!/usr/bin/env python3
"""
v2.3 周线缠论仓位过滤回测
对比:
  A. 原v2.2: 无周线过滤
  B. v2.3: 周线C级减仓50%, B级减20%, A级不变
"""
import sys, os, pickle, struct
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


# ── 周线环境计算 ─────────────────────────────────────────────────────────────
def read_index_daily():
    """读取沪指日线（int×100格式，需要÷100还原真实价格）"""
    base = '/workspace/tdx_data/sh/lday/sh000001.day'
    rows = []
    with open(base, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    for i in range(n):
        b = data[i*32:(i+1)*32]
        vals = struct.unpack('<8I', b)
        date, open_i, close_i, high_i, low_i = vals[0], vals[2], vals[3], vals[4], vals[5]
        vol = vals[6]
        # int×100格式 → 除100还原
        rows.append({'date': date, 'open': open_i/100.0, 'close': close_i/100.0,
                     'high': high_i/100.0, 'low': low_i/100.0, 'volume': vol})
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    return df.sort_index()


def detect_bi(highs, lows):
    n = len(highs)
    def is_top(i):
        if i<=0 or i>=n-1: return False
        return float(highs.iloc[i])>float(highs.iloc[i-1]) and float(highs.iloc[i])>float(highs.iloc[i+1])
    def is_bottom(i):
        if i<=0 or i>=n-1: return False
        return float(lows.iloc[i])<float(lows.iloc[i-1]) and float(lows.iloc[i])<float(lows.iloc[i+1])
    tops = [i for i in range(1,n-1) if is_top(i)]
    bottoms = [i for i in range(1,n-1) if is_bottom(i)]
    all_pts = sorted([(i,'top') for i in tops] + [(i,'bottom') for i in bottoms])
    strokes = []
    i = 0
    while i < len(all_pts)-1:
        idx1,t1 = all_pts[i]
        idx2,t2 = all_pts[i+1]
        if t1=='top' and t2=='bottom':
            strokes.append((idx1,idx2,'down')); i+=2
        elif t1=='bottom' and t2=='top':
            strokes.append((idx1,idx2,'up')); i+=2
        else:
            i+=1
    merged = []
    for s in strokes:
        if not merged: merged.append(s)
        elif s[2]==merged[-1][2]: merged[-1]=(merged[-1][0],s[1],s[2])
        else: merged.append(s)
    return merged


def get_weekly_grade_for_date(daily_df, date_ts, lookback=104):
    """计算给定日期的周线环境等级"""
    try:
        df_sub = daily_df[daily_df.index <= date_ts].tail(lookback)
        n = len(df_sub)
        if n < 20: return ('B', 0.8)
        # 按周聚合
        periods = df_sub.index.to_period('W')
        week_df = df_sub.groupby(periods).agg(
            close=('close', 'last'),
            high=('high', 'max'),
            low=('low', 'min'),
            volume=('volume', 'sum')
        )
        nw = len(week_df)
        if nw < 20: return ('B', 0.8)
        close_w = week_df['close'].astype(float).values
        ma5_w = np.convolve(close_w, np.ones(5)/5, mode='valid')
        ma10_w = np.convolve(close_w, np.ones(10)/10, mode='valid')
        c = close_w[-1]
        m5 = ma5_w[-1] if len(ma5_w) else c
        m10 = ma10_w[-1] if len(ma10_w) else c
        pm5 = ma5_w[-2] if len(ma5_w) >= 2 else m5
        last5w = (close_w[-1] - close_w[-6]) / close_w[-6] * 100 if nw >= 6 else 0.0
        vol_now = float(week_df['volume'].iloc[-5:].mean())
        vol_prv = float(week_df['volume'].iloc[-10:-5].mean()) if nw >= 10 else vol_now
        vol_chg = (vol_now - vol_prv) / vol_prv * 100 if vol_prv > 0 else 0.0
        score = sum([m5 > m10, c > m5, m5 > pm5, last5w > 0, vol_chg > 5])
        if score >= 4: return ('A', 1.0)
        elif score >= 2: return ('B', 0.8)
        else: return ('C', 0.5)
    except Exception as e:
        return ('B', 0.8)


print("计算周线环境（预计算，按月）...")
idx_daily = read_index_daily()

# 预计算每月的周线grade
sig_df['month'] = pd.to_datetime(sig_df['date']).dt.to_period('M')
months = sorted(sig_df['month'].unique())
monthly_grade = {}
for m in months:
    # 取每月最后5个交易日的代表日期
    m_start = m.to_timestamp()
    m_end = (m+1).to_timestamp()
    sample_date = min(pd.Timestamp(idx_daily.index.max()), m_end - pd.Timedelta(days=1))
    grade, cap = get_weekly_grade_for_date(idx_daily, sample_date)
    monthly_grade[m] = (grade, cap)

print("月度周线环境:")
for m, (g, c) in sorted(monthly_grade.items()):
    print(f"  {m}: {g}级(仓位×{c})")

# 过滤3buy
sig_df = sig_df[sig_df['type'] != '3buy'].copy()

# 从sl_price推断fib
sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_df['fib_inferred'] = fib

sector_good_months = {pd.Period('2026-03')}


def get_sl(row, month):
    """Fib分档SL"""
    fib = row['fib_inferred']
    price = row['price']
    sl_orig = row['sl_price']
    if fib == 'strongest':
        buy1_low = sl_orig / 0.98 if sl_orig > 0 else price * 0.80
        return max(buy1_low, price * 0.90)
    elif fib == 'strong':
        return max(sl_orig, price * 0.94)
    else:
        return price * (0.92 if month in sector_good_months else 0.94)


def run_bt(sig_df, data_map, weekly_adj=False, desc=''):
    """
    weekly_adj=True: 启用周线仓位过滤（C级×0.5, B级×0.8, A级×1.0）
    """
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
    weekly_filter_rejects = 0
    grade_counts = defaultdict(int)

    debug_count = [0]
    first_settle = [None]  # (d, weight, pnl, equity_before, equity_after)

    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        month = pd.Period(d)
        week_grade, week_cap = monthly_grade.get(month, ('B', 0.8))

        # 清结算
        for si in range(5):
            slot = slots[si]
            if not slot: continue
            ed = slot['exit_date']
            if isinstance(ed, str): ed = pd.Timestamp(ed)
            if ed <= d_ts:
                slot_weight = slot.get('weight', 1.0) if weekly_adj else 1.0
                eq_before = equity
                equity += (equity/5) * slot['pnl_pct'] * slot_weight
                eq_curve.append(equity)
                if debug_count[0] < 3 and first_settle[0] is None:
                    print(f"  DEBUG settle: weekly_adj={weekly_adj} slot_wt={slot.get('weight','N/A')} used_wt={slot_weight}")
                    first_settle[0] = (d, slot_weight, slot['pnl_pct'], eq_before, equity)
                    debug_count[0] += 1
                trades += 1
                if slot['pnl_pct'] > 0: winning += 1
                pnl_list.append(slot['pnl_pct'] * slot_weight)
                exit_reasons[slot['exit_reason']] += 1
                slots[si] = None
            code_exit[slot['code']] = d

        monthly_eq.setdefault(month, equity)
        day_sigs = sig_df[sig_df['date'] == d]

        for _, row in day_sigs.iterrows():
            code = row['code']
            btype = row['type']
            pos = int(row['pos'])
            price = row['price']
            if code in cooldown and d_ts < cooldown[code]: continue
            if code in code_exit and code_exit[code] == d: continue

            # ── v2.3 周线仓位过滤 ──────────────────────────────────────
            if weekly_adj:
                grade_counts[week_grade] += 1
                if week_cap < 1.0:
                    weekly_filter_rejects += 1
            # ───────────────────────────────────────────────────────────

            si = next((i for i in range(5) if slots[i] is None), None)
            if si is None: break
            if code not in data_map: continue
            df_c = data_map[code]
            if pos >= len(df_c): continue

            sl = get_sl(row, month)
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
                if profit_pct >= 0.03:
                    dd = (close_bi - price) / close_bi
                    if dd >= 0.08:
                        exit_price = close_bi
                        exit_reason = 'trail_stop'
                        ei = bi
                        break

            pnl_pct = (exit_price - price) / price - 0.0003*2
            hold = max(1, (ei - pos)//8)
            exit_date = d_ts + pd.Timedelta(days=hold)
            # v2.3: 按周线grade调整实际仓位权重
            actual_weight = week_cap  # A=1.0, B=0.8, C=0.5
            slots[si] = {'code': code, 'btype': btype, 'pnl_pct': pnl_pct,
                          'exit_date': exit_date, 'exit_reason': exit_reason,
                          'weight': actual_weight}
            cooldown[code] = exit_date + pd.Timedelta(days=5)

    # 期末结算
    for si in range(5):
        slot = slots[si]
        if slot:
            slot_weight = slot.get('weight', 1.0) if weekly_adj else 1.0
            equity += (equity/5) * slot['pnl_pct'] * slot_weight
            eq_curve.append(equity)
            trades += 1
            if slot['pnl_pct'] > 0: winning += 1
            pnl_list.append(slot['pnl_pct'] * slot_weight)
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
        'desc': desc,
        'grade_counts': dict(grade_counts),
        'weekly_filter_rejects': weekly_filter_rejects,
    }


# ── 回测 ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("v2.3 周线仓位过滤回测对比")
print("="*70)

results = []

# 用强信号subset加速，但跑完整版
sig_strong = sig_df[sig_df['fib_inferred'].isin(['strong','strongest'])]

# A. v2.2 无周线过滤（强信号）
r = run_bt(sig_strong, data_map, weekly_adj=False, desc='v2.2 强信号(无周线)')
if r: results.append(r)

# B. v2.3 + 周线过滤（强信号）
r = run_bt(sig_strong, data_map, weekly_adj=True, desc='v2.3 强信号+周线过滤')
if r: results.append(r)

# C. 全量信号 无周线过滤
r = run_bt(sig_df, data_map, weekly_adj=False, desc='v2.2 全量(无周线)')
if r: results.append(r)

# D. 全量信号 + 周线过滤
r = run_bt(sig_df, data_map, weekly_adj=True, desc='v2.3 全量+周线过滤')
if r: results.append(r)

# 打印结果
print(f"\n{'配置':<28} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'总收益%':>8} {'交易':>6}")
print('-'*80)
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    print(f"{r['desc']:<28} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} {r['win_rate']:>5.0f} "
          f"{r['avg_pnl']:>+7.2f} {r['total_ret']:>+8.1f} {r['total_trades']:>6}")
    if r.get('grade_counts'):
        gc = r['grade_counts']
        print(f"  周线环境: A={gc.get('A',0)} B={gc.get('B',0)} C={gc.get('C',0)}  周线拒绝={r['weekly_filter_rejects']}")

# 月度收益对比（用实际标签区分）
v2_2_full = next((r for r in results if r['desc']=='v2.2 无周线过滤'), None)
v2_3_full = next((r for r in results if r['desc']=='v2.3 全量+周线过滤'), None)
if v2_2_full and v2_3_full:
    print("\n── 月度收益对比（全量版本）──")
    for month in sorted(v2_2_full['monthly_eq'].keys()):
        eq2 = v2_2_full['monthly_eq'].get(month, 1_000_000)
        eq3 = v2_3_full['monthly_eq'].get(month, 1_000_000)
        prev2 = list(v2_2_full['monthly_eq'].values())[list(v2_2_full['monthly_eq'].keys()).index(month)-1] if list(v2_2_full['monthly_eq'].keys()).index(month) > 0 else 1_000_000
        prev3 = list(v2_3_full['monthly_eq'].values())[list(v2_3_full['monthly_eq'].keys()).index(month)-1] if list(v2_3_full['monthly_eq'].keys()).index(month) > 0 else 1_000_000
        ret2 = (eq2-prev2)/prev2*100
        ret3 = (eq3-prev3)/prev3*100
        g, cap = monthly_grade.get(month, ('B', 0.8))
        flag = ' ⬅️' if abs(ret3-ret2) > 0.01 else ''
        print(f"  {month} [{g}级{cap}]: v2.2={ret2:+.1f}%  v2.3={ret3:+.1f}%  差异={ret3-ret2:+.2f}%{flag}")

# 出场原因
print(f"\n── 出场原因对比 ──")
for lbl, r in [('v2.2', v2_2), ('v2.3', v2_3)]:
    print(f"  [{lbl}]")
    for reason, n in sorted(r['exit_reasons'].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {n} ({n/r['total_trades']*100:.0f}%)")

with open('/workspace/v23_backtest.pkl', 'wb') as f:
    pickle.dump({'results': results, 'monthly_grade': monthly_grade}, f)
print("\n已保存: v23_backtest.pkl")
