#!/usr/bin/env python3
"""
止盈策略对比回测: 简化版 vs v2.0阶梯止盈
- 简化版: 浮盈≥3% + 回撤≥8% → 追踪止盈
- v2.0阶梯: 5%以下保本 / 5-10%成本+1% / 10-15%入场+3% / >15%前高-1%
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

# ── TDX 沪指读取 ─────────────────────────────────────────────────────────────
base = '/workspace/tdx_data/sh/lday/sh000001.day'
rows = []
with open(base, 'rb') as f:
    data = f.read()
for i in range(len(data)//32):
    vals = struct.unpack('<8I', data[i*32:(i+1)*32])
    rows.append({'date': vals[0], 'close': vals[3]/100.0, 'volume': float(vals[6])})
idx_df = pd.DataFrame(rows)
idx_df['date'] = pd.to_datetime(idx_df['date'], format='%Y%m%d')
idx_df.set_index('date', inplace=True)
idx_df.sort_index(inplace=True)

# ── 周线环境计算 ─────────────────────────────────────────────────────────────
def get_weekly_grade(daily_df, date_ts, lookback=104):
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(lookback)
        if len(df_s) < 20: return ('B', 0.8)
        wk = df_s.groupby(df_s.index.to_period('W')).agg(
            close=('close','last'), volume=('volume','sum'))
        nw = len(wk)
        if nw < 20: return ('B', 0.8)
        cw = wk['close'].astype(float).values
        ma5 = np.convolve(cw, np.ones(5)/5, mode='valid')
        ma10 = np.convolve(cw, np.ones(10)/10, mode='valid')
        c = cw[-1]
        m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if nw >= 6 else 0
        vn = float(wk['volume'].iloc[-5:].mean())
        vp = float(wk['volume'].iloc[-10:-5].mean()) if nw >= 10 else vn
        vc = (vn-vp)/vp*100 if vp > 0 else 0
        sc = sum([m5>m10, c>m5, m5>pm5, l5w>0, vc>5])
        return ('A', 1.0) if sc >= 4 else ('B', 0.8) if sc >= 2 else ('C', 0.5)
    except: return ('B', 0.8)

print("计算周线环境...")
sig_df['month'] = pd.to_datetime(sig_df['date']).dt.to_period('M')
months = sorted(sig_df['month'].unique())
monthly_grade = {}
for m in months:
    sample = min(pd.Timestamp(idx_df.index.max()), (m+1).to_timestamp() - pd.Timedelta(days=1))
    monthly_grade[m] = get_weekly_grade(idx_df, sample)

print("\n月度周线环境:")
for m, (g, c) in sorted(monthly_grade.items()):
    print(f"  {m}: {g}级(×{c})")

# ── SL 计算 ──────────────────────────────────────────────────────────────────
sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_df['fib_inferred'] = fib

def get_sl(row, month):
    btype = row['type']
    price = row['price']
    if btype == '2buy':
        sl_base = 0.94
    elif btype == '2plus3buy':
        sl_base = 0.94
    else:
        sl_base = 0.93
    g, cap = monthly_grade.get(month, ('B', 0.8))
    if g == 'C': sl_base = max(sl_base, 0.93)
    elif g == 'B': sl_base = max(sl_base, 0.94)
    return price * sl_base

# ── 回测核心（止盈模式参数化）────────────────────────────────────────────────
def run_bt(sig_df, data_map, weekly_adj=False, tp_mode='simplified', desc=''):
    """
    tp_mode:
      'simplified' : 浮盈≥3% + 回撤≥8% → 追踪止盈
      'v2'         : v2.0阶梯止盈（>15%前高-3%）
      'v2tight'    : v2.0阶梯止盈（>15%前高-1%）
      'chanlun_v2'  : 缠论主动卖点 + 分批止盈（+15%半仓+前高-3%）
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

    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        month = pd.Period(d)
        week_grade, week_cap = monthly_grade.get(month, ('B', 0.8))

        # ── 清结算 ────────────────────────────────────────────────────────────
        for si in range(5):
            slot = slots[si]
            if not slot: continue
            ed = slot['exit_date']
            if isinstance(ed, str): ed = pd.Timestamp(ed)
            if ed <= d_ts:
                slot_weight = slot.get('weight', 1.0) if weekly_adj else 1.0
                equity += (equity/5) * slot['pnl_pct'] * slot_weight
                eq_curve.append(equity)
                trades += 1
                if slot['pnl_pct'] > 0: winning += 1
                pnl_list.append(slot['pnl_pct'] * slot_weight)
                exit_reasons[slot['exit_reason']] += 1
                slots[si] = None
            code_exit[slot['code']] = d_ts

        # ── 月末equity记录 ────────────────────────────────────────────────────
        if d_ts.day >= 25 or d == unique_dates[-1]:
            monthly_eq[month] = equity

        # ── 入场 ──────────────────────────────────────────────────────────────
        btype = row['type'] if 'type' in locals() else 'unknown'
        for _, row in sig_df[sig_df['date'] == d].iterrows():
            code = row['code']
            price = row['price']
            pos = int(row['pos'])
            btype = row['type']
            if code in cooldown and d_ts <= cooldown[code]: continue
            if code in code_exit and code_exit.get(code) == d: continue

            si = next((i for i in range(5) if slots[i] is None), None)
            if si is None: break
            if code not in data_map: continue
            df_c = data_map[code]
            if pos >= len(df_c): continue

            sl = get_sl(row, month)

            # ── 出场逻辑（止盈模式）───────────────────────────────────────────
            exit_reason = 'timeout'
            exit_price = float(df_c['close'].iloc[-1])
            ei = len(df_c) - 1
            pnl_pct = 0.0
            hold = 1

            if tp_mode == 'simplified':
                loop_end = min(pos+30*8, len(df_c)-1)
                tp_seen = False
                for bi in range(pos+1, loop_end):
                    low_bi = float(df_c['low'].iloc[bi]); close_bi = float(df_c['close'].iloc[bi])
                    if low_bi <= sl: exit_price = sl; exit_reason = 'stop_loss'; ei = bi; break
                    profit_pct = (close_bi - price) / price
                    if profit_pct >= 0.03:
                        dd = (close_bi - price) / close_bi
                        if dd >= 0.08 and not tp_seen:
                            exit_price = close_bi; exit_reason = 'take_profit_s'; ei = bi; tp_seen = True; break
                    exit_reason = 'timeout'; ei = bi
                else:
                    ei = loop_end - 1 if loop_end > pos + 1 else pos
                    exit_price = float(df_c['close'].iloc[ei]); exit_reason = 'timeout'
                pnl_pct = (exit_price - price) / price - 0.0003*2
            elif tp_mode == 'v2':
                trailing_stop = sl; high_water = price; loop_end = min(pos+30*8, len(df_c)-1)
                tp_seen = False
                for bi in range(pos+1, loop_end):
                    low_bi = float(df_c['low'].iloc[bi]); high_bi = float(df_c['high'].iloc[bi]); close_bi = float(df_c['close'].iloc[bi])
                    if low_bi <= sl: exit_price = sl; exit_reason = 'stop_loss'; ei = bi; break
                    if high_bi > high_water: high_water = high_bi
                    profit_pct = (close_bi - price) / price
                    if profit_pct < 0.05: ts = price
                    elif profit_pct < 0.10: ts = price * 1.01
                    elif profit_pct < 0.15: ts = price * 1.03
                    else: ts = high_water * 0.97
                    if ts > trailing_stop: trailing_stop = ts
                    if low_bi <= trailing_stop: exit_price = trailing_stop; exit_reason = 'take_profit_v2'; ei = bi; tp_seen = True; break
                    exit_reason = 'timeout'; ei = bi
                else:
                    ei = loop_end - 1 if loop_end > pos + 1 else pos
                    exit_price = float(df_c['close'].iloc[ei]); exit_reason = 'timeout'
                pnl_pct = (exit_price - price) / price - 0.0003*2
            elif tp_mode == 'v2_low':
                # 低门槛版: >8%即启动追踪, -3%回撤
                trailing_stop = sl; high_water = price; loop_end = min(pos+30*8, len(df_c)-1)
                tp_seen = False
                for bi in range(pos+1, loop_end):
                    low_bi = float(df_c['low'].iloc[bi]); close_bi = float(df_c['close'].iloc[bi])
                    if low_bi <= sl: exit_price = sl; exit_reason = 'stop_loss'; ei = bi; break
                    if close_bi > high_water: high_water = close_bi
                    profit_pct = (close_bi - price) / price
                    if profit_pct < 0.05: ts = price
                    elif profit_pct < 0.08: ts = price * 1.01
                    else: ts = high_water * 0.97
                    if ts > trailing_stop: trailing_stop = ts
                    if low_bi <= trailing_stop: exit_price = trailing_stop; exit_reason = 'take_profit_v2'; ei = bi; tp_seen = True; break
                    exit_reason = 'timeout'; ei = bi
                else:
                    ei = loop_end - 1 if loop_end > pos + 1 else pos
                    exit_price = float(df_c['close'].iloc[ei]); exit_reason = 'timeout'
                pnl_pct = (exit_price - price) / price - 0.0003*2
            elif tp_mode == 'v2tight':
                trailing_stop = sl; high_water = price; loop_end = min(pos+30*8, len(df_c)-1)
                tp_seen = False
                for bi in range(pos+1, loop_end):
                    low_bi = float(df_c['low'].iloc[bi]); high_bi = float(df_c['high'].iloc[bi]); close_bi = float(df_c['close'].iloc[bi])
                    if low_bi <= sl: exit_price = sl; exit_reason = 'stop_loss'; ei = bi; break
                    if high_bi > high_water: high_water = high_bi
                    profit_pct = (close_bi - price) / price
                    if profit_pct < 0.05: ts = price
                    elif profit_pct < 0.10: ts = price * 1.01
                    elif profit_pct < 0.15: ts = price * 1.03
                    else: ts = high_water * 0.99
                    if ts > trailing_stop: trailing_stop = ts
                    if low_bi <= trailing_stop: exit_price = trailing_stop; exit_reason = 'take_profit_v2'; ei = bi; tp_seen = True; break
                    exit_reason = 'timeout'; ei = bi
                else:
                    ei = loop_end - 1 if loop_end > pos + 1 else pos
                    exit_price = float(df_c['close'].iloc[ei]); exit_reason = 'timeout'
                pnl_pct = (exit_price - price) / price - 0.0003*2
            elif tp_mode.startswith('chanlun_v2'):
                # chanlun_v2_NpM: N%=部分止盈门槛, M%=前高回撤跟踪
                # 例: chanlun_v2_15p03 = 15%触发, -3%跟踪
                #     chanlun_v2_20p05 = 20%触发, -5%跟踪
                parts = tp_mode.split('_')
                partial_thresh = 0.15
                trail_pct = 0.97
                if len(parts) >= 3:
                    p_str = parts[2]  # e.g. "20p05"
                    if 'p' in p_str:
                        thresh_str, trail_str = p_str.replace('p', 'p').split('p')
                        partial_thresh = float(thresh_str) / 100.0
                        trail_pct = 1.0 - float(trail_str) / 100.0

                # 计算MACD（DIF/DEA）
                prices_all = df_c['close'].astype(float).values
                highs_all  = df_c['high'].astype(float).values
                lows_all   = df_c['low'].astype(float).values
                lookback = min(pos + 60, len(prices_all))
                sub_prices = prices_all[pos:lookback]
                sub_highs  = highs_all[pos:lookback]
                sub_lows   = lows_all[pos:lookback]
                n = len(sub_prices)
                ema12 = sub_prices[0]
                ema26 = sub_prices[0]
                dif_list = []
                for j in range(n):
                    ema12 = ema12 * 11/13 + sub_prices[j] * 2/13
                    ema26 = ema26 * 25/27 + sub_prices[j] * 2/27
                    dif_list.append(ema12 - ema26)
                dea_list = []
                ema9 = dif_list[0]
                for j in range(n):
                    ema9 = ema9 * 8/10 + dif_list[j] * 2/10
                    dea_list.append(ema9)

                divergence_high_idx = None
                partial_exited = False
                trailing_stop = sl
                high_water = price
                half_closed = False
                half_exit_price = price
                half_exit_bi = None

                for bi in range(1, min(30*8, n - 1)):
                    low_bi  = sub_lows[bi]
                    high_bi = sub_highs[bi]
                    close_bi = sub_prices[bi]
                    dif_cur  = dif_list[bi]

                    if low_bi <= sl:
                        if half_closed:
                            # 半仓在之前已止盈，剩余半仓被止损
                            # 总pnl = 半仓pnl + 半仓亏损
                            pnl_from_half = (half_exit_price - price) / price * 0.5
                            pnl_from_remain = (sl - price) / price * 0.5 - 0.0003
                            pnl_pct = pnl_from_half + pnl_from_remain
                            exit_price = half_exit_price  # 报告用
                        else:
                            exit_price = sl
                        exit_reason = 'stop_loss'
                        ei = bi
                        break

                    # ── 分批止盈：threshold 半仓 ───────────────────────────────
                    profit_pct = (close_bi - price) / price
                    if not half_closed and profit_pct >= partial_thresh:
                        half_exit_price = close_bi
                        half_exit_bi = bi
                        half_closed = True
                        trailing_stop = high_bi * trail_pct

                    if half_closed:
                        if high_bi > high_water:
                            high_water = high_bi
                            trailing_stop = max(trailing_stop, high_water * trail_pct)
                        if low_bi <= trailing_stop:
                            exit_price = trailing_stop; exit_reason = 'chanlun_partial_tp'; ei = bi; break
                        continue
                    # ── 缠论1卖检测 ─────────────────────────────────────────
                    if divergence_high_idx is None and bi >= 5:
                        prev_high_idx = bi - 1
                        for kk in range(bi - 2, max(0, bi - 20), -1):
                            if sub_highs[kk] > sub_highs[prev_high_idx]:
                                prev_high_idx = kk
                        if high_bi > sub_highs[prev_high_idx] and dif_cur < dif_list[prev_high_idx]:
                            divergence_high_idx = bi
                else:
                    ei = min(30*8, n - 2) if n > 1 else 0
                    exit_price = float(df_c['close'].iloc[pos + ei]) if pos + ei < len(df_c) else price
                    exit_reason = 'timeout'
                if half_closed and exit_reason == 'chanlun_partial_tp':
                    pnl_pct = (0.5 * half_exit_price + 0.5 * exit_price) / price - 1 - 0.0003*2
                elif exit_reason == 'stop_loss' and half_closed:
                    pass  # already computed above
                else:
                    pnl_pct = (exit_price - price) / price - 0.0003*2
            exit_date = d_ts + pd.Timedelta(days=hold)
            actual_weight = week_cap
            slots[si] = {
                'code': code, 'btype': btype, 'pnl_pct': pnl_pct,
                'exit_date': exit_date, 'exit_reason': exit_reason,
                'weight': actual_weight
            }
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
    sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0
    rmax = np.maximum.accumulate(eq)
    dd = (eq - rmax)/rmax
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

# ── 用强信号subset加速 ───────────────────────────────────────────────────────
sig_strong = sig_df[sig_df['fib_inferred'].isin(['strong','strongest'])]

print("\n" + "="*70)
print("止盈策略对比回测")
print("="*70)

results = []
configs = [
    # 基准
    (sig_strong, False, 'simplified',      '简化版(3%触发+8%回撤)'),
    (sig_strong, False, 'v2',               'v2阶梯止盈(>15%前高-3%)'),
    (sig_strong, False, 'v2_low',           'v2_low阶梯止盈(>8%前高-3%)'),
    (sig_strong, False, 'v2tight',          'v2tight阶梯止盈(>15%前高-1%)'),
    # 缠论参数网格
    (sig_strong, False, 'chanlun_v2_15p03','缠论+15%门槛/-3%回撤(原参数)'),
    (sig_strong, False, 'chanlun_v2_15p05','缠论+15%门槛/-5%回撤'),
    (sig_strong, False, 'chanlun_v2_20p03','缠论+20%门槛/-3%回撤'),
    (sig_strong, False, 'chanlun_v2_20p05','缠论+20%门槛/-5%回撤'),
    (sig_strong, False, 'chanlun_v2_25p03','缠论+25%门槛/-3%回撤'),
    (sig_strong, False, 'chanlun_v2_25p05','缠论+25%门槛/-5%回撤'),
    # 全量+周线
    (sig_df,     True,  'simplified',       '简化版全量+周线'),
    (sig_df,     True,  'v2',                'v2阶梯止盈全量+周线'),
    (sig_df,     True,  'v2_low',           'v2_low阶梯止盈全量+周线'),
    (sig_df,     True,  'v2tight',           'v2tight阶梯止盈全量+周线'),
    (sig_df,     True,  'chanlun_v2_15p03', '缠论+15%门槛/-3%全量+周线'),
    (sig_df,     True,  'chanlun_v2_20p05', '缠论+20%门槛/-5%全量+周线'),
    (sig_df,     True,  'chanlun_v2_25p05', '缠论+25%门槛/-5%全量+周线'),
]

for sdf, wa, tp, label in configs:
    print(f"\n运行: {label} ...", flush=True)
    r = run_bt(sdf, data_map, weekly_adj=wa, tp_mode=tp, desc=label)
    if r:
        er = r['exit_reasons']
        total = sum(er.values())
        tp_keys = [k for k in er if 'take_profit' in k or 'chanlun' in k or 'sell' in k]
        print(f"  EXIT: total={total} SL={er.get('stop_loss',0)} timeout={er.get('timeout',0)} tp_keys={tp_keys}")
        for k in tp_keys:
            if er[k] > 0: print(f"    {k}: {er[k]}")
        results.append(r)

# ── 打印结果 ─────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print(f"{'配置':<36} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'总收益%':>10} {'交易':>6}")
print("-"*85)
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    print(f"{r['desc']:<36} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} {r['win_rate']:>5.0f} "
          f"{r['avg_pnl']:>+7.2f} {r['total_ret']:>+10.1f} {r['total_trades']:>6}")

# ── 出场原因分析 ─────────────────────────────────────────────────────────────
print("\n── 出场原因分析 ──")
for r in results:
    er = r['exit_reasons']
    total = sum(er.values())
    sl_n = er.get('stop_loss', 0)
    to_n = er.get('timeout', 0)
    tp_n = 0; tp_lbl = '止盈'
    if '简化' in r['desc']:
        tp_n = er.get('take_profit_s', 0); tp_lbl = '止盈'
    elif 'simplified' in r['desc']:
        tp_n = er.get('take_profit_s', 0); tp_lbl = '止盈'
    elif 'chanlun' in r['desc']:
        tp_n = er.get('chanlun_partial_tp', 0) + er.get('chanlun_2sell', 0) + er.get('chanlun_1sell', 0); tp_lbl = '分批/缠论'
    elif 'v2tight' in r['desc']:
        tp_n = er.get('take_profit_v2', 0); tp_lbl = '阶梯止盈'
    elif 'v2' in r['desc']:
        tp_n = er.get('take_profit_v2', 0); tp_lbl = '止盈'
    print(f"  {r['desc']:<36} 止损={sl_n:>4}({sl_n/total*100:>5.1f}%) "
          f"{tp_lbl}={tp_n:>4}({tp_n/total*100:>5.1f}%) 超时={to_n:>4}({to_n/total*100:>5.1f}%)")

# ── 月度equity对比（简化 vs v2） ─────────────────────────────────────────────
print("\n── 月度equity对比（全量+周线过滤）──")
v2_s = next((r for r in results if 'v2' in r['desc'] and '全量' in r['desc'] and '周线' in r['desc']), None)
smp_s = next((r for r in results if '简化' in r['desc'] and '全量' in r['desc'] and '周线' in r['desc']), None)
if v2_s and smp_s:
    all_months = sorted(set(list(v2_s['monthly_eq'].keys()) + list(smp_s['monthly_eq'].keys())))
    print(f"  {'月份':<10} {'简化版收益%':>12} {'v2阶梯收益%':>12} {'差异':>8} {'月线环境':>8}")
    print("  " + "-"*52)
    prev_v2 = 1_000_000.0
    prev_smp = 1_000_000.0
    for m in all_months:
        g, cap = monthly_grade.get(m, ('B', 0.8))
        eq_v2 = v2_s['monthly_eq'].get(m, prev_v2)
        eq_smp = smp_s['monthly_eq'].get(m, prev_smp)
        ret_v2 = (eq_v2 - prev_v2)/prev_v2*100
        ret_smp = (eq_smp - prev_smp)/prev_smp*100
        diff = ret_v2 - ret_smp
        better = "✓v2优" if diff > 0 else "✓简化优" if diff < 0 else "持平"
        flag = " ←" if abs(diff) > 2 else ""
        print(f"  {str(m):<10} {ret_smp:>+12.2f} {ret_v2:>+12.2f} {diff:>+8.2f}  {g}级{cap}{flag}")
        prev_v2 = eq_v2
        prev_smp = eq_smp
