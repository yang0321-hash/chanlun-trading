#!/usr/bin/env python3
"""
SL参数对比回测 - 基于预生成信号 (2024-07 ~ 2026-04)
对比: SL=3%/4%/5%/6%/8%, TP=5%固定, 0.618过滤已含在信号中
"""
import pickle, numpy as np, pandas as pd, time, sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_unified')

print("=" * 60)
print("SL参数对比回测: 基于预生成信号")
print("固定: TP=5%, 0.618过滤(信号已含)")
print("=" * 60)

t0 = time.time()

# 加载预生成信号
sig_df = pickle.load(open('/workspace/backtest_new_fw_signals.pkl', 'rb'))
print(f"信号总数: {len(sig_df)}")
print(f"日期范围: {sig_df['date'].min()} ~ {sig_df['date'].max()}")
print(f"类型分布: {sig_df['type'].value_counts().to_dict()}")

# 只用2buy和2plus3buy
sig_df = sig_df[sig_df['type'].isin(['2buy', '2plus3buy'])].copy()
print(f"2buy+2plus3buy: {len(sig_df)}")

# 加载历史数据
data_map = pickle.load(open('/workspace/backtest_v15_all_a_data.pkl', 'rb'))
print(f"历史数据股票数: {len(data_map)}")

# 辅助函数
def run_backtest_for_sl(sig_df, data_map, sl_pct, tp_pct=0.05):
    """给定SL%和TP%，回测信号"""
    trades = []

    for _, row in sig_df.iterrows():
        code = row['code']
        entry_idx = row['entry_idx']
        entry_price = row['price']

        df = data_map.get(code)
        if df is None or len(df) <= entry_idx + 1:
            continue

        close_arr = df['close'].values
        open_arr = df['open'].values
        high_arr = df['high'].values
        n = len(df)

        # T+1入场
        if entry_idx + 1 >= n:
            continue
        buy_price = float(open_arr[entry_idx + 1])
        if buy_price <= 0:
            continue

        # SL = 入场价 × (1 - sl_pct)
        sl_price = buy_price * (1 - sl_pct)
        tp_price = buy_price * (1 + tp_pct)

        exit_idx = None
        exit_price = None
        sl_triggered = tp_triggered = False

        for d in range(entry_idx + 2, min(entry_idx + 30, n)):
            lo = float(open_arr[d])
            hi = float(high_arr[d])
            if lo <= sl_price:
                sl_triggered = True
                exit_idx = d
                exit_price = sl_price
                break
            if hi >= tp_price:
                tp_triggered = True
                exit_idx = d
                exit_price = tp_price
                break

        if exit_idx is None or exit_idx >= n:
            continue
        exit_price = float(open_arr[exit_idx])

        ret_pct = (exit_price - buy_price) / buy_price * 100
        hold_days = exit_idx - entry_idx - 1

        trades.append({
            'code': code,
            'type': row['type'],
            'entry_date': row['date'],
            'buy_price': buy_price,
            'exit_price': exit_price,
            'ret_pct': ret_pct,
            'hold_days': hold_days,
            'hit_sl': 1 if sl_triggered else 0,
            'exit_reason': 'SL' if sl_triggered else 'TP5'
        })

    return trades

def stats(trades, label):
    if not trades:
        print(f"\n{label}: 无交易")
        return None
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

    # 按类型
    print(f"  分类型:")
    for t in df['type'].unique():
        sub = df[df['type'] == t]
        print(f"    {t}: {len(sub)}笔, 胜率{len(sub[sub['ret_pct']>0])/len(sub)*100:.0f}%, 均盈{sub['ret_pct'].mean():.2f}%")

    # 年度
    df['year'] = df['entry_date'].str[:4]
    yearly = df.groupby('year')['ret_pct'].agg(['sum','count'])
    print(f"  年度明细:")
    for yr in sorted(yearly.index):
        r = yearly.loc[yr]
        print(f"    {yr}: {r['sum']:+.1f}% ({int(r['count'])}笔)")

    return df

# 对比测试
results = {}
for sl_pct in [0.03, 0.04, 0.05, 0.06, 0.08]:
    label = f"SL={sl_pct*100:.0f}%"
    print(f"\n>>> 回测 {label}...")
    trades = run_backtest_for_sl(sig_df, data_map, sl_pct)
    df = stats(trades, label)
    if df is not None:
        results[sl_pct] = df

# 汇总对比
if results:
    print(f"\n{'='*55}")
    print("SL参数对比汇总")
    print(f"{'='*55}")
    print(f"{'SL':>6} {'交易数':>7} {'胜率':>6} {'均盈':>8} {'总收益':>10} {'Sharpe':>7} {'止损率':>7}")
    print("-" * 55)
    for sl, df in sorted(results.items()):
        wr = len(df[df['ret_pct']>0])/len(df)*100
        avg = df['ret_pct'].mean()
        tot = df['ret_pct'].sum()
        sh = df['ret_pct'].mean()/max(df['ret_pct'].std(),0.01)*np.sqrt(252)
        slr = df['hit_sl'].mean()*100
        print(f"  {sl*100:>4.0f}% {len(df):>7} {wr:>6.1f}% {avg:>8.3f}% {tot:>10.1f}% {sh:>7.2f} {slr:>6.1f}%")

print(f"\n总耗时: {time.time()-t0:.0f}s")
