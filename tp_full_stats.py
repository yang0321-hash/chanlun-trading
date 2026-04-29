#!/usr/bin/env python3
"""
止盈策略完整统计：含SL亏损交易的全样本WR/均盈/Sharpe
对比：原始逐条回测 vs 网格搜索（验证一致性）
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

sl_ratio = sig_df['sl_price'] / sig_df['price']
fib = pd.Series('weak', index=sig_df.index)
fib[sl_ratio > 0.97] = 'strongest'
fib[(sl_ratio > 0.92) & (sl_ratio <= 0.97)] = 'strong'
sig_strong = sig_df[fib.isin(['strong','strongest'])].copy().reset_index(drop=True)
print(f"signals: {len(sig_strong)}")

def get_weekly_grade(daily_df, date_ts, lookback=104):
    try:
        df_s = daily_df[daily_df.index <= date_ts].tail(lookback)
        if len(df_s) < 20: return 0.8
        wk = df_s.groupby(df_s.index.to_period('W')).agg(close=('close','last'))
        nw = len(wk)
        if nw < 20: return 0.8
        cw = wk['close'].astype(float).values
        ma5 = np.convolve(cw, np.ones(5)/5, mode='valid')
        ma10 = np.convolve(cw, np.ones(10)/10, mode='valid')
        c = cw[-1]; m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if nw >= 6 else 0
        sc = sum([m5>m10, c>m5, m5>pm5, l5w>0])
        return 0.94 if sc >= 4 else (0.80 if sc >= 2 else 0.93)
    except: return 0.8

base = '/workspace/tdx_data/sh/lday/sh000001.day'
rows = []
with open(base, 'rb') as f:
    data = f.read()
for i in range(len(data)//32):
    vals = struct.unpack('<8I', data[i*32:(i+1)*32])
    rows.append({'date': vals[0], 'close': vals[3]/100.0})
idx_df = pd.DataFrame(rows)
idx_df['date'] = pd.to_datetime(idx_df['date'], format='%Y%m%d')
idx_df.set_index('date', inplace=True)
idx_df.sort_index(inplace=True)

sig_strong['month'] = pd.to_datetime(sig_strong['date']).dt.to_period('M')
months = sorted(sig_strong['month'].unique())
monthly_sl = {}
for m in months:
    sample = min(pd.Timestamp(idx_df.index.max()), (m+1).to_timestamp() - pd.Timedelta(days=1))
    monthly_sl[m] = get_weekly_grade(idx_df, sample)

def run_bt_full(sub_df, trigger, trail, half_pct=0, half_trail=0):
    """
    完整回测：含止损的全样本统计
    返回：(总体统计, 分触发/未触发的子样本统计)
    """
    pnls_all = []
    pnls_triggered = []   # 触发止盈监控的
    pnls_not_triggered = []  # 未触发止盈监控的（SL/超时）
    exit_counts = defaultdict(int)
    total = len(sub_df)

    for _, row in sub_df.iterrows():
        code = row['code']
        price = float(row['price'])
        month = row['month']
        btype = row['type']
        sl_base = 0.94 if btype in ('2buy', '2plus3buy') else 0.93
        sl_ratio_inner = max(sl_base, monthly_sl.get(month, 0.8))
        sl = price * sl_ratio_inner
        date = row['date']
        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]
        n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)
        tp_triggered = False
        half_exit_price = None
        exit_reason = 'timeout'; exit_price = price
        sl_hit = False

        price_hwm = price
        pnl = 0.0  # default
        for bi in range(pos_bar + 1, loop_end):
            low_bi = float(df_c['low'].iloc[bi])
            close_bi = float(df_c['close'].iloc[bi])
            high_bi = float(df_c['high'].iloc[bi])
            if high_bi > price_hwm: price_hwm = high_bi

            if low_bi <= sl:
                exit_price = sl; exit_reason = 'stop_loss'; sl_hit = True
                pnl = (sl - price) / price; break

            profit_pct = (close_bi - price) / price
            if profit_pct >= trigger and not tp_triggered:
                tp_triggered = True

            if tp_triggered:
                dd = (price_hwm - close_bi) / price_hwm

                if half_pct > 0 and half_trail > 0 and half_exit_price is None and dd >= half_trail:
                    half_exit_price = close_bi
                    sl = min(sl, price)

                if dd >= trail:
                    if half_exit_price is not None:
                        pnl = (half_exit_price - price) / price * half_pct + (close_bi - price) / price * (1 - half_pct)
                    else:
                        pnl = (close_bi - price) / price
                    exit_reason = 'take_profit'; exit_price = close_bi; break
        else:
            # 未触发止盈（超时）
            ei = min(loop_end, n-1) if loop_end > pos_bar else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if ei < n else price
            if half_exit_price is not None:
                pnl = (half_exit_price - price) / price * half_pct + (exit_price - price) / price * (1 - half_pct)
            else:
                pnl = (exit_price - price) / price

        pnl_net = pnl - 0.0006
        pnls_all.append(pnl_net)
        exit_counts[exit_reason] += 1

        if tp_triggered or sl_hit:
            pnls_triggered.append(pnl_net)
        else:
            pnls_not_triggered.append(pnl_net)

    pnls_all = np.array(pnls_all)
    pnls_t = np.array(pnls_triggered) if pnls_triggered else np.array([0])
    pnls_nt = np.array(pnls_not_triggered) if pnls_not_triggered else np.array([0])

    def stats(pnls):
        if len(pnls) == 0: return {'wr': 0, 'avg': 0, 'sharpe': 0, 'dd': 0, 'n': 0}
        wr = (pnls > 0).mean() * 100
        avg = pnls.mean() * 100
        dd = abs(pnls.min()) * 100
        std = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
        sharpe = 0.04 / std if std > 1e-8 else 0
        return {'wr': wr, 'avg': avg, 'sharpe': sharpe, 'dd': dd, 'n': len(pnls)}

    all_s = stats(pnls_all)
    t_s = stats(pnls_t)
    nt_s = stats(pnls_nt)
    total_count = sum(exit_counts.values())

    return {
        'all': all_s,
        'triggered': t_s,
        'not_triggered': nt_s,
        'exit_counts': dict(exit_counts),
        'trigger_rate': len(pnls_triggered) / total_count * 100 if total_count else 0,
    }

# ── 测试关键策略：完整统计 ─────────────────────────────────────────────────
strategies = [
    (0.03, 0.08, 0.00, 0.00, 'A: 简化版(3%启动,8%回撤)'),
    (0.05, 0.08, 0.00, 0.00, 'B: 5%启动,8%回撤'),
    (0.06, 0.06, 0.00, 0.00, 'C: 6%启动,6%回撤 [网格最优]'),
    (0.08, 0.06, 0.00, 0.00, 'D: 8%启动,6%回撤 [网格胜率最高]'),
    (0.03, 0.08, 0.00, 0.00, 'E: 3%启动,8%回撤 [原v2.0]'),
    (0.05, 0.08, 0.03, 0.03, 'F: 5%启动,3%卖半,8%清仓'),
    (0.03, 0.08, 0.03, 0.03, 'G: 3%启动,3%卖半,8%清仓'),
    (0.05, 0.08, 0.05, 0.05, 'H: 5%启动,5%卖半,8%清仓'),
]

print(f"\n{'策略':<35} {'N':>5} {'WR全':>5} {'均盈全':>7} {'Sharpe全':>9} {'触发率':>6} {'WR触':>5} {'均盈触':>7} {'WR未触':>5} {'均盈未':>7} {'SL次数':>6} {'止盈次':>6} {'超时常':>6}")
print("-"*130)

for trigger, trail, half_pct, half_trail, desc in strategies:
    r = run_bt_full(sig_strong, trigger, trail, half_pct, half_trail)
    a = r['all']; t = r['triggered']; nt = r['not_triggered']
    ec = r['exit_counts']
    sl_n = ec.get('stop_loss', 0)
    tp_n = ec.get('take_profit', 0)
    to_n = ec.get('timeout', 0)
    print(f"{desc:<35} {a['n']:>5} {a['wr']:>5.0f} {a['avg']:>+7.2f} {a['sharpe']:>9.3f} "
          f"{r['trigger_rate']:>5.1f}% {t['wr']:>5.0f} {t['avg']:>+7.2f} "
          f"{nt['wr']:>5.0f} {nt['avg']:>+7.2f} {sl_n:>6} {tp_n:>6} {to_n:>6}")
