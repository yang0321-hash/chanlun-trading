#!/usr/bin/env python3
"""
止损参数SL网格搜索
覆盖：SL基准 × 大盘状态 × 买点类型 的全组合
目标：全样本WR/均盈/Sharpe最优
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
sys.path.insert(0, '/workspace')

sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

# 不过滤fib，保留全量信号
print(f"总信号: {len(sig_df)}")

# 沪指数据
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

# 计算日线状态
def get_daily_state(date_ts):
    try:
        df_s = idx_df[idx_df.index <= date_ts].tail(20)
        if len(df_s) < 10: return 'unknown'
        close = float(df_s['close'].iloc[-1])
        ma5 = float(df_s['close'].iloc[-5:].mean())
        ma10 = float(df_s['close'].iloc[-10:].mean())
        if ma5 > ma10 and close > ma5: return 'bull'
        elif ma5 < ma10 and close < ma5: return 'bear'
        else: return 'neutral'
    except: return 'unknown'

# 月度SL基准（保持和止盈回测一致）
def get_monthly_sl(date_ts):
    try:
        df_s = idx_df[idx_df.index <= date_ts].tail(104)
        if len(df_s) < 20: return 0.94
        wk = df_s.groupby(df_s.index.to_period('W')).agg(close=('close','last'))
        if len(wk) < 20: return 0.94
        cw = wk['close'].astype(float).values
        ma5 = np.convolve(cw, np.ones(5)/5, mode='valid')
        ma10 = np.convolve(cw, np.ones(10)/10, mode='valid')
        c = cw[-1]; m5, m10 = ma5[-1], ma10[-1]
        pm5 = ma5[-2] if len(ma5) >= 2 else m5
        l5w = (cw[-1]-cw[-6])/cw[-6]*100 if len(cw) >= 6 else 0
        sc = sum([m5>m10, c>m5, m5>pm5, l5w>0])
        return 0.94 if sc >= 4 else (0.80 if sc >= 2 else 0.93)
    except: return 0.94

sig_df['daily_state'] = pd.to_datetime(sig_df['date']).apply(get_daily_state)
sig_df['month'] = pd.to_datetime(sig_df['date']).dt.to_period('M')

# 止盈参数固定为6%触发/6%回撤（已验证最优）
TP_TRIGGER = 0.06
TP_TRAIL = 0.06

def run_bt_sl(sl_1buy, sl_2buy, sl_3buy,
              sl_bull_mult=1.0, sl_bear_mult=1.0,
              sl_neutral_mult=1.0,
              desc=''):
    """
    sl_1buy/2buy/3buy: 止损比例（小数，如0.05=5%）
    sl_X_mult: 大盘状态 multipliers（乘数）
    返回全样本统计
    """
    pnls = []
    exit_counts = defaultdict(int)
    total = 0

    for _, row in sig_df.iterrows():
        code = row['code']
        price = float(row['price'])
        btype = row['type']
        date = row['date']
        daily_state = row['daily_state']
        month = row['month']
        monthly_sl = get_monthly_sl(pd.Timestamp(date))

        # 确定SL基准
        if btype in ('2buy', '2plus3buy'):
            sl_base = sl_2buy
        elif btype == '3buy':
            sl_base = sl_3buy
        else:  # 1buy
            sl_base = sl_1buy

        # 大盘乘数
        mult_map = {'bull': sl_bull_mult, 'bear': sl_bear_mult, 'neutral': sl_neutral_mult, 'unknown': 1.0}
        mult = mult_map.get(daily_state, 1.0)

        # 月度grade叠加
        combined_mult = max(monthly_sl, sl_base * mult)
        sl = price * combined_mult

        if code not in data_map: continue
        df_c = data_map[code]
        bi_list = df_c.index.get_indexer([pd.Timestamp(date)], method='bfill')
        if bi_list[0] < 0: continue
        pos_bar = bi_list[0]
        n = len(df_c)
        loop_end = min(pos_bar + 30, n - 1)
        if loop_end <= pos_bar + 1: continue

        price_hwm = price
        tp_triggered = False
        half_exit_price = None
        pnl = 0.0
        exit_reason = 'skip'
        total += 1

        for bi in range(pos_bar + 1, loop_end):
            low_bi = float(df_c['low'].iloc[bi])
            close_bi = float(df_c['close'].iloc[bi])
            high_bi = float(df_c['high'].iloc[bi])
            if high_bi > price_hwm: price_hwm = high_bi

            # 止损
            if low_bi <= sl:
                pnl = (sl - price) / price - 0.0006
                exit_reason = 'stop_loss'; break

            # 止盈监控
            profit_pct = (close_bi - price) / price
            if profit_pct >= TP_TRIGGER and not tp_triggered:
                tp_triggered = True

            if tp_triggered:
                dd = (price_hwm - close_bi) / price_hwm
                if dd >= TP_TRAIL:
                    pnl = (close_bi - price) / price - 0.0006
                    exit_reason = 'take_profit'; break
        else:
            # 超时
            ei = min(loop_end, n-1) if loop_end > pos_bar else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if ei < n else price
            pnl = (exit_price - price) / price - 0.0006
            exit_reason = 'timeout'

        pnls.append(pnl)
        exit_counts[exit_reason] += 1

    if not pnls: return None
    pnls = np.array(pnls)
    wr = (pnls > 0).mean() * 100
    avg = pnls.mean() * 100
    max_dd = abs(pnls.min()) * 100
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = 0.04 / std_w if std_w > 1e-8 else 0
    tot = sum(exit_counts.values())
    sl_n = exit_counts['stop_loss']
    tp_n = exit_counts['take_profit']
    to_n = exit_counts['timeout']
    return {
        'sharpe': sharpe, 'win_rate': wr, 'avg_pnl': avg,
        'max_dd': max_dd, 'n': tot,
        'sl_pct': sl_n/tot*100 if tot else 0,
        'tp_pct': tp_n/tot*100 if tot else 0,
        'to_pct': to_n/tot*100 if tot else 0,
        'sl_count': sl_n, 'tp_count': tp_n, 'to_count': to_n,
    }

# ── 网格搜索 ─────────────────────────────────────────────────────────────────
print("\n=== 网格搜索：止损参数 ===")
print("固定止盈参数: 触发6%, 回撤6%")
print()

results = []

# 方案1: 统一SL（所有买点相同）
print("搜索中...", flush=True)
count = 0
for sl_all in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    count += 1
    r = run_bt_sl(sl_all, sl_all, sl_all, 1.0, 1.0, 1.0, desc=f'统一SL={int(sl_all*100)}%')
    if r:
        r['desc'] = f'统一{int(sl_all*100)}%'
        r['sl_all'] = sl_all
        results.append(r)

# 方案2: 各买点独立SL
sl_1buy_opts = [0.04, 0.05, 0.06, 0.07]
sl_2buy_opts = [0.02, 0.03, 0.04, 0.05]
sl_3buy_opts = [0.03, 0.04, 0.05]

for s1 in sl_1buy_opts:
    for s2 in sl_2buy_opts:
        for s3 in sl_3buy_opts:
            count += 1
            r = run_bt_sl(s1, s2, s3, 1.0, 1.0, 1.0,
                          desc=f'1买{int(s1*100)}% 2买{int(s2*100)}% 3买{int(s3*100)}%')
            if r:
                r['desc'] = f'1={int(s1*100)}% 2={int(s2*100)}% 3={int(s3*100)}%'
                r['sl_1'] = s1; r['sl_2'] = s2; r['sl_3'] = s3
                results.append(r)

# 方案3: 大盘状态乘数（以2买为代表测试）
sl_base_opts = [0.04, 0.05]
bull_mult_opts = [0.7, 0.8, 1.0]  # 强势市SL更紧
bear_mult_opts = [1.0, 1.2, 1.5]   # 弱势市SL更松
neutral_mult_opts = [0.9, 1.0]

for base_sl in sl_base_opts:
    for bm in bull_mult_opts:
        for bem in bear_mult_opts:
            for nm in neutral_mult_opts:
                count += 1
                r = run_bt_sl(base_sl, base_sl, base_sl, bm, bem, nm,
                             desc=f'基准{int(base_sl*100)}% 多头×{bm} 空头×{bem} 中性×{nm}')
                if r:
                    r['desc'] = f'基准{int(base_sl*100)}% ×[多{bem}空{bem}中{nm}]'
                    r['sl_all'] = base_sl; r['mult_bull'] = bm
                    r['mult_bear'] = bem; r['mult_neutral'] = nm
                    results.append(r)

print(f"完成 {count} 种组合，{len(results)} 个有效结果\n")

# 排序输出
results.sort(key=lambda x: -x['sharpe'])
print(f"{'#':>4} {'描述':<45} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'SL%':>6} {'止盈%':>6} {'超时%':>6} {'SL次数':>6}")
print("-"*105)
for i, r in enumerate(results[:30]):
    print(f"{i+1:>4} {r['desc']:<45} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} "
          f"{r['avg_pnl']:>+7.2f} {r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% "
          f"{r['to_pct']:>5.1f}% {r['sl_count']:>6}")

print("\n── 多维度Top10 ──")
for met, lab in [('sharpe','Sharpe'),('win_rate','胜率'),('avg_pnl','均盈'),('max_dd','DD最小')]:
    top = sorted(results, key=lambda x: -x[met])[:5]
    print(f"\n{lab}:")
    for r in top:
        print(f"  {r['desc']:<45} → {met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}% SL%={r['sl_pct']:.1f}%")

# 综合推荐
print("\n── 综合推荐（权重: Sharpe40% WR25% 均盈20% DD15%）──")
sh_min = min(r['sharpe'] for r in results); sh_max = max(r['sharpe'] for r in results)
wr_min = min(r['win_rate'] for r in results); wr_max = max(r['win_rate'] for r in results)
av_min = min(r['avg_pnl'] for r in results); av_max = max(r['avg_pnl'] for r in results)
dd_min = min(r['max_dd'] for r in results); dd_max = max(r['max_dd'] for r in results)
for r in results:
    ns = (r['sharpe']-sh_min)/(sh_max-sh_min+1e-8)
    nw = (r['win_rate']-wr_min)/(wr_max-wr_min+1e-8)
    na = (r['avg_pnl']-av_min)/(av_max-av_min+1e-8)
    nd = (dd_max-r['max_dd'])/(dd_max-dd_min+1e-8)
    r['composite'] = 0.4*ns + 0.25*nw + 0.2*na + 0.15*nd
best = max(results, key=lambda x: x['composite'])
print(f"推荐: {best['desc']}")
print(f"  Sharpe={best['sharpe']:.3f} WR={best['win_rate']:.0f}% 均盈={best['avg_pnl']:+.2f}% DD={best['max_dd']:.1f}%")
print(f"  SL率={best['sl_pct']:.1f}% 止盈率={best['tp_pct']:.1f}% 超时率={best['to_pct']:.1f}%")
print(f"  综合得分={best['composite']:.3f}")
