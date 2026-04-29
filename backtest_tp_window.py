#!/usr/bin/env python3
"""
TP动态化 + 信号窗口 回测
TP对比:
  - 固定5%: TP=买入价×1.05
  - 动态3%+8%: 首层TP=3%，触发后移动止损=最高价×0.92

信号窗口对比: 15天 / 30天 / 60天 / 无限制
"""
import pickle, numpy as np, pandas as pd, time, sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_unified')

print("=" * 60)
print("TP动态化 + 信号窗口 回测")
print("=" * 60)

t0 = time.time()

# 加载数据
sig_df = pickle.load(open('/workspace/backtest_new_fw_signals.pkl', 'rb'))
data_map = pickle.load(open('/workspace/backtest_v15_all_a_data.pkl', 'rb'))
print(f"信号总数: {len(sig_df)}")
print(f"日期范围: {sig_df['date'].min()} ~ {sig_df['date'].max()}")
print(f"历史数据: {len(data_map)} 只")

# 只用2buy和2plus3buy
sig_df = sig_df[sig_df['type'].isin(['2buy', '2plus3buy'])].copy()
sig_df['date'] = pd.to_datetime(sig_df['date'])
print(f"2buy+2plus3buy: {len(sig_df)}")

# SL统一用8%
SL_PCT = 0.08

def run_backtest(sig_df_sub, tp_type='fixed', window_days=None):
    """
    tp_type: 'fixed_5' | 'dynamic_3_8'
    window_days: None=无限制, 15/30/60
    """
    # 按窗口过滤
    if window_days is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
        sig_df_sub = sig_df_sub[sig_df_sub['date'] >= cutoff].copy()

    trades = []
    for _, row in sig_df_sub.iterrows():
        code = row['code']
        entry_idx = row['entry_idx']
        entry_price = row['price']

        df = data_map.get(code)
        if df is None or len(df) <= entry_idx + 1:
            continue

        open_arr = df['open'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        n = len(df)

        if entry_idx + 1 >= n:
            continue
        buy_price = float(open_arr[entry_idx + 1])
        if buy_price <= 0:
            continue

        sl_price = buy_price * (1 - SL_PCT)
        exit_idx = None
        exit_price = None
        exit_reason = None
        hit_sl = False

        if tp_type == 'fixed_5':
            tp_price = buy_price * 1.05
            for d in range(entry_idx + 2, min(entry_idx + 30, n)):
                lo = float(open_arr[d])
                hi = float(high_arr[d])
                if lo <= sl_price:
                    hit_sl = True; exit_idx = d; exit_price = sl_price; exit_reason = 'SL'; break
                if hi >= tp_price:
                    exit_idx = d; exit_price = tp_price; exit_reason = 'TP5'; break

        elif tp_type == 'dynamic_3_8':
            # 阶段1: 等TP=3%
            tp1_price = buy_price * 1.03
            # 阶段2: 移动止损=最高价×0.92
            trail_triggered = False
            highest_after_tp1 = None

            for d in range(entry_idx + 2, min(entry_idx + 30, n)):
                lo = float(open_arr[d])
                hi = float(high_arr[d])
                lo_val = float(low_arr[d])

                if lo <= sl_price:
                    hit_sl = True; exit_idx = d; exit_price = sl_price; exit_reason = 'SL'; break

                if not trail_triggered:
                    if hi >= tp1_price:
                        trail_triggered = True
                        highest_after_tp1 = hi  # 记录触发TP1之后的高点
                    # else: 继续等
                else:
                    # 移动止损模式
                    if hi > highest_after_tp1:
                        highest_after_tp1 = hi
                    trail_sl = highest_after_tp1 * 0.92
                    if lo <= trail_sl:
                        exit_idx = d; exit_price = trail_sl; exit_reason = 'TSL'; break

            # 如果30天内既没触TP1也没SL，持仓不动算平
            if exit_idx is None:
                exit_idx = min(entry_idx + 30, n - 1)
                exit_price = float(open_arr[exit_idx])
                exit_reason = 'HOLD30'

        if exit_idx is None or exit_idx >= n:
            continue
        exit_price = float(open_arr[exit_idx])

        ret_pct = (exit_price - buy_price) / buy_price * 100
        hold_days = exit_idx - entry_idx - 1

        trades.append({
            'code': code,
            'type': row['type'],
            'entry_date': str(row['date'])[:10],
            'buy_price': buy_price,
            'exit_price': exit_price,
            'ret_pct': ret_pct,
            'hold_days': hold_days,
            'hit_sl': 1 if hit_sl else 0,
            'exit_reason': exit_reason
        })

    return trades

def stats(trades, label, show_yearly=False):
    if not trades:
        print(f"\n{label}: 无交易"); return None
    df = pd.DataFrame(trades)
    wins = df[df['ret_pct'] > 0]
    loss = df[df['ret_pct'] <= 0]
    total_ret = df['ret_pct'].sum()
    avg_ret = df['ret_pct'].mean()
    win_rate = len(wins) / len(df) * 100
    sharpe = df['ret_pct'].mean() / max(df['ret_pct'].std(), 0.01) * np.sqrt(252)
    sl_rate = df['hit_sl'].mean() * 100

    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")
    print(f"  总交易:    {len(df)} 笔")
    print(f"  胜率:      {win_rate:.1f}%  (盈{len(wins)}/亏{len(loss)})")
    print(f"  总收益:    {total_ret:.1f}%")
    print(f"  平均收益:  {avg_ret:.3f}%")
    print(f"  最大单笔:  {df['ret_pct'].max():.1f}%")
    print(f"  最大亏损:  {df['ret_pct'].min():.1f}%")
    print(f"  止损率:    {sl_rate:.1f}%")
    print(f"  Sharpe:    {sharpe:.2f}")
    print(f"  平均持仓:  {df['hold_days'].mean():.1f} 天")

    # 统计各exit_reason
    exit_counts = df['exit_reason'].value_counts()
    print(f"  退出原因: {exit_counts.to_dict()}")

    # 按类型
    for t in sorted(df['type'].unique()):
        sub = df[df['type'] == t]
        wr = len(sub[sub['ret_pct']>0])/len(sub)*100 if len(sub)>0 else 0
        print(f"    {t}: {len(sub)}笔, 胜率{wr:.0f}%, 均盈{sub['ret_pct'].mean():.2f}%")

    if show_yearly:
        df['year'] = df['entry_date'].str[:4]
        yearly = df.groupby('year')['ret_pct'].agg(['sum','count'])
        print(f"  年度明细:")
        for yr in sorted(yearly.index):
            r = yearly.loc[yr]
            print(f"    {yr}: {r['sum']:+.1f}% ({int(r['count'])}笔)")

    return df

# ═══════════════════════════════════════════════════════════════
# 第一部分: TP对比 (固定5% vs 动态3%+8%)
# 信号窗口: 无限制(全量)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("第一部分: TP对比 (信号窗口=无限制)")
print("="*60)

for tp in ['fixed_5', 'dynamic_3_8']:
    label = "TP=5%固定" if tp == 'fixed_5' else "TP=3%首层+8%移动止盈"
    trades = run_backtest(sig_df, tp_type=tp, window_days=None)
    df = stats(trades, label, show_yearly=True)

# ═══════════════════════════════════════════════════════════════
# 第二部分: 信号窗口对比 (用动态TP)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("第二部分: 信号窗口对比 (TP=动态3%+8%)")
print("="*60)

for wdays in [15, 30, 60, None]:
    label = f"窗口={wdays}天" if wdays else "窗口=无限制"
    trades = run_backtest(sig_df, tp_type='dynamic_3_8', window_days=wdays)
    df = stats(trades, label)

# ═══════════════════════════════════════════════════════════════
# 第三部分: 最优组合验证
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("第三部分: 最优组合汇总")
print("="*60)

configs = [
    ('fixed_5', None),
    ('fixed_5', 30),
    ('dynamic_3_8', None),
    ('dynamic_3_8', 30),
    ('dynamic_3_8', 15),
]

print(f"\n{'配置':>20} {'交易数':>7} {'胜率':>6} {'均盈':>8} {'总收益':>10} {'Sharpe':>7} {'止损率':>7}")
print("-" * 70)
for tp, wdays in configs:
    label = f"{'TP=5%固定' if tp=='fixed_5' else 'TP=动态3%+8%'} {'窗口='+str(wdays)+'天' if wdays else '窗口=全量'}"
    trades = run_backtest(sig_df, tp_type=tp, window_days=wdays)
    if not trades:
        print(f"  {label:>20}: 无数据"); continue
    df = pd.DataFrame(trades)
    wr = len(df[df['ret_pct']>0])/len(df)*100
    avg = df['ret_pct'].mean()
    tot = df['ret_pct'].sum()
    sh = df['ret_pct'].mean()/max(df['ret_pct'].std(),0.01)*np.sqrt(252)
    slr = df['hit_sl'].mean()*100
    print(f"  {label:>20} {len(df):>7} {wr:>6.1f}% {avg:>8.3f}% {tot:>10.1f}% {sh:>7.2f} {slr:>6.1f}%")

print(f"\n总耗时: {time.time()-t0:.0f}s")
