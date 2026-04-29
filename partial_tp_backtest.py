#!/usr/bin/env python3
"""
分批止盈策略回测对比
对比：
  A: 简化版（≥3%启动，≥8%回撤清仓）
  B: 分批版（≥5%启动，回撤3%卖半，回撤8%清仓）
  C: 分批版（≥5%启动，回撤5%卖半，回撤8%清仓）
  D: 分批版（≥3%启动，回撤5%卖半，回撤8%清仓）
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

# fib过滤 → strongest
sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_strong = sig_df[fib.isin(['strong','strongest'])].copy().reset_index(drop=True)
print(f"signals: {len(sig_strong)}")

# 月度grade止损基准
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
        c = cw[-1]; m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if nw >= 6 else 0
        vn = float(wk['volume'].iloc[-5:].mean())
        vp = float(wk['volume'].iloc[-10:-5].mean()) if nw >= 10 else vn
        vc = (vn-vp)/vp*100 if vp > 0 else 0
        sc = sum([m5>m10, c>m5, m5>pm5, l5w>0, vc>5])
        return ('A', 1.0) if sc >= 4 else ('B', 0.8) if sc >= 2 else ('C', 0.5)
    except: return ('B', 0.8)

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

sig_strong['month'] = pd.to_datetime(sig_strong['date']).dt.to_period('M')
months = sorted(sig_strong['month'].unique())
monthly_grade = {}
for m in months:
    sample = min(pd.Timestamp(idx_df.index.max()), (m+1).to_timestamp() - pd.Timedelta(days=1))
    monthly_grade[m] = get_weekly_grade(idx_df, sample)
print(f"月度grade: {len(monthly_grade)} 个月")

# ── 4种止盈策略 ──────────────────────────────────────────────────────────────
def run_bt(sub_df, trigger_pct, trail_pct, half_at=None, desc=''):
    """
    trigger_pct: 浮盈达到此值启动监控
    trail_pct: 回撤达到此值清仓（或卖半）
    half_at: 若设置，则在此回撤值卖半（例3%），之后剩余仓位在trail_pct清仓
    """
    pnls, exit_reasons = [], defaultdict(int)
    half_sold = 0
    for _, row in sub_df.iterrows():
        code = row['code']
        price = float(row['price'])
        month = row['month']
        btype = row['type']
        sl_base = 0.94 if btype in ('2buy', '2plus3buy') else 0.93
        g, _ = monthly_grade.get(month, ('B', 0.8))
        if g == 'C': sl_base = max(sl_base, 0.93)
        elif g == 'B': sl_base = max(sl_base, 0.94)
        sl = price * sl_base
        date = row['date']
        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]
        n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)
        high_water = price
        tp_triggered = False
        half_exit_price = None
        exit_reason = 'timeout'; exit_price = price; ei = pos_bar

        for bi in range(pos_bar + 1, loop_end):
            low_bi = float(df_c['low'].iloc[bi])
            close_bi = float(df_c['close'].iloc[bi])
            high_bi = float(df_c['high'].iloc[bi])

            if low_bi <= sl:
                exit_price = sl; exit_reason = 'stop_loss'; ei = bi
                pnl = (exit_price - price) / price - 0.0003 * 2
                break

            if close_bi > high_water: high_water = close_bi
            profit_pct = (close_bi - price) / price
            drawdown = (high_water - close_bi) / high_water

            if not tp_triggered and profit_pct >= trigger_pct:
                tp_triggered = True

            if tp_triggered:
                # 分批止盈
                if half_at is not None and half_sold == 0 and drawdown >= half_at:
                    # 卖半仓
                    half_exit_price = close_bi
                    half_sold = 1
                    sl = min(sl, price)  # 成本价保本

                if drawdown >= trail_pct:
                    if half_sold == 1:
                        remaining_pnl = (close_bi - price) / price * 0.5
                        half_actual = (half_exit_price - price) / price * 0.5
                        pnl = half_actual + remaining_pnl - 0.0003 * 2
                    else:
                        pnl = (close_bi - price) / price - 0.0003 * 2
                    exit_price = close_bi
                    exit_reason = 'take_profit_half' if half_sold else 'take_profit'
                    ei = bi
                    break

            exit_reason = 'timeout'; ei = bi
        else:
            ei = loop_end - 1 if loop_end > pos_bar + 1 else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if ei < n else price
            if half_sold == 1 and half_exit_price is not None:
                pnl = ((half_exit_price - price) * 0.5 + (exit_price - price) * 0.5) / price - 0.0003 * 2
            else:
                pnl = (exit_price - price) / price - 0.0003 * 2

        pnls.append(pnl); exit_reasons[exit_reason] += 1; half_sold = 0

    if not pnls: return None
    pnls = np.array(pnls)
    wr = (pnls > 0).mean()
    avg = pnls.mean()
    max_dd = abs(pnls.min())
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = 0.04 / std_w if std_w > 1e-8 else 0
    total_ret = (1 + pnls).prod() - 1
    er = dict(exit_reasons)
    total = sum(er.values())
    tp_half = er.get('take_profit_half', 0)
    tp_full = er.get('take_profit', 0)
    return {
        'sharpe': sharpe, 'max_dd': max_dd * 100, 'win_rate': wr * 100,
        'avg_pnl': avg * 100, 'total_ret': total_ret * 100, 'n': len(pnls),
        'sl_pct': er.get('stop_loss', 0)/total*100 if total else 0,
        'tp_pct': (tp_full + tp_half)/total*100 if total else 0,
        'half_pct': tp_half/total*100 if total else 0,
        'timeout_pct': er.get('timeout', 0)/total*100 if total else 0,
        'exit_reasons': er,
    }

print("\n" + "="*80)
print(f"{'策略':<35} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'总收益%':>9} {'止损%':>6} {'止盈%':>6} {'半仓%':>6}")
print("-"*100)

results = []
strategies = [
    # (trigger%, trail%, half_at, desc)
    (0.03, 0.08, None,                    'A: 简化版(≥3%启动,≥8%回撤清仓)'),
    (0.05, 0.03, None,                    'B1: 5%启动,3%回撤清仓(无分批)'),
    (0.05, 0.08, None,                    'B2: 5%启动,8%回撤清仓(无分批)'),
    (0.05, 0.08, 0.03,                    'C: 5%启动,3%卖半,8%清仓'),
    (0.05, 0.08, 0.05,                    'D: 5%启动,5%卖半,8%清仓'),
    (0.03, 0.08, 0.05,                    'E: 3%启动,5%卖半,8%清仓'),
    (0.03, 0.08, 0.03,                    'F: 3%启动,3%卖半,8%清仓'),
    (0.05, 0.05, None,                    'G: 5%启动,5%回撤清仓(无分批)'),
]

for trigger, trail, half_at, desc in strategies:
    r = run_bt(sig_strong, trigger, trail, half_at)
    if r:
        er = r['exit_reasons']
        total = sum(er.values())
        print(f"{desc:<35} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} "
              f"{r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['total_ret']:>+9.1f} "
              f"{r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['half_pct']:>5.1f}%")
        results.append((desc, r))

# 找出最优Sharpe和最优总收益
best_sh = max(results, key=lambda x: x[1]['sharpe'])
best_tr = max(results, key=lambda x: x[1]['total_ret'])
best_wr = max(results, key=lambda x: x[1]['win_rate'])
best_avg = max(results, key=lambda x: x[1]['avg_pnl'])

print()
print(f"最优Sharpe:   {best_sh[0]} → {best_sh[1]['sharpe']:.2f}")
print(f"最高总收益:   {best_tr[0]} → {best_tr[1]['total_ret']:+.1f}%")
print(f"最高胜率:     {best_wr[0]} → {best_wr[1]['win_rate']:.0f}%")
print(f"最高均盈:     {best_avg[0]} → {best_avg[1]['avg_pnl']:+.2f}%")
