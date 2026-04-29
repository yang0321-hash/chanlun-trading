#!/usr/bin/env python3
"""
密集网格搜索：分批止盈最优参数
覆盖：触发门槛 × 首轮止盈方式 × 清仓门槛
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

def run_bt(sub_df, trigger, trail, half_pct=0, half_trail=0):
    """
    trigger: 浮盈达到此值启动监控（小数，如0.03）
    trail: 回撤达到此值清仓（剩余全部）
    half_pct: 卖半仓的比例（0=不卖半，0.5=卖一半，0.33=卖1/3，0.67=卖2/3）
    half_trail: 触发卖半仓的回撤阈值（0=不卖半）
    """
    pnls = []
    exit_stats = defaultdict(int)
    for _, row in sub_df.iterrows():
        code = row['code']
        price = float(row['price'])
        month = row['month']
        btype = row['type']
        sl_base = 0.94 if btype in ('2buy', '2plus3buy') else 0.93
        g = monthly_sl.get(month, 0.8)
        sl = price * max(sl_base, g)
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
        half_sold_pct = 0.0
        half_exit_price = 0.0
        pnl = 0.0
        exited = False

        for bi in range(pos_bar + 1, loop_end):
            low_bi = float(df_c['low'].iloc[bi])
            close_bi = float(df_c['close'].iloc[bi])
            if low_bi <= sl:
                pnl = (sl - price) / price - 0.0003 * 2
                exit_stats['stop_loss'] += 1
                exited = True; break
            if close_bi > high_water: high_water = close_bi
            profit_pct = (close_bi - price) / price
            if profit_pct >= trigger: tp_triggered = True
            if tp_triggered:
                drawdown = (high_water - close_bi) / high_water
                # 卖半仓
                if half_trail > 0 and half_sold_pct == 0 and drawdown >= half_trail:
                    half_sold_pct = half_pct
                    half_exit_price = close_bi
                    # 更新止损为成本价（锁定已卖部分）
                    sl = min(sl, price)
                # 清仓
                if drawdown >= trail:
                    if half_sold_pct > 0:
                        half_actual = (half_exit_price - price) / price * half_sold_pct
                        remaining_pct = 1 - half_sold_pct
                        remaining_pnl = (close_bi - price) / price * remaining_pct
                        pnl = half_actual + remaining_pnl - 0.0003 * 2
                    else:
                        pnl = (close_bi - price) / price - 0.0003 * 2
                    exit_stats['take_profit'] += 1
                    exited = True; break
        if not exited:
            ei = loop_end - 1 if loop_end > pos_bar + 1 else pos_bar
            exit_price = float(df_c['close'].iloc[ei]) if ei < n else price
            if half_sold_pct > 0:
                half_actual = (half_exit_price - price) / price * half_sold_pct
                remaining_pct = 1 - half_sold_pct
                pnl = half_actual + (exit_price - price) / price * remaining_pct - 0.0003 * 2
            else:
                pnl = (exit_price - price) / price - 0.0003 * 2
            exit_stats['timeout'] += 1
        pnls.append(pnl)
    if not pnls: return None
    pnls = np.array(pnls)
    wr = (pnls > 0).mean()
    avg = pnls.mean()
    max_dd = abs(pnls.min())
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = 0.04 / std_w if std_w > 1e-8 else 0
    total = len(pnls)
    return {
        'sharpe': sharpe, 'win_rate': wr * 100, 'avg_pnl': avg * 100,
        'max_dd': max_dd * 100, 'n': total,
        'sl_pct': exit_stats['stop_loss'] / total * 100,
        'tp_pct': exit_stats['take_profit'] / total * 100,
        'to_pct': exit_stats['timeout'] / total * 100,
    }

# ── 密集网格搜索 ──────────────────────────────────────────────────────────────
print("开始网格搜索...")
triggers = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]  # 启动门槛
half_trails = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.0]  # 卖半门槛（0=不用半仓）
half_pcts = [0, 0.33, 0.50, 0.67]  # 卖半比例
trails = [0.06, 0.08, 0.10, 0.12, 0.15]  # 清仓门槛

results = []
count = 0
for trigger in triggers:
    for half_trail in half_trails:
        for half_pct in half_pcts:
            # 如果half_trail=0，则half_pct也必须=0（无分批）
            if half_trail == 0 and half_pct > 0: continue
            # 如果half_pct=0，则half_trail无意义但设为0
            if half_pct == 0: half_trail = 0
            for trail in trails:
                if half_trail >= trail: continue  # 半仓门槛必须小于清仓门槛
                count += 1
                desc = f"tr={int(trigger*100)}% ht={int(half_trail*100) if half_trail>0 else 0}% hp={int(half_pct*100) if half_pct>0 else 0}% tl={int(trail*100)}%"
                r = run_bt(sig_strong, trigger, trail, half_pct, half_trail)
                if r:
                    r['desc'] = desc
                    r['trigger'] = trigger; r['half_trail'] = half_trail
                    r['half_pct'] = half_pct; r['trail'] = trail
                    results.append(r)

print(f"完成 {count} 种组合\n")

# 按Sharpe排序
results.sort(key=lambda x: -x['sharpe'])

print(f"{'排名':>4} {'触发':>6} {'半仓':>6} {'比例':>5} {'清仓':>6} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'止损%':>6} {'止盈%':>6}")
print("-"*75)
for i, r in enumerate(results[:30]):
    ht = f"{int(r['half_trail']*100)}%" if r['half_trail'] > 0 else "无"
    hp = f"{int(r['half_pct']*100)}%" if r['half_pct'] > 0 else "-"
    print(f"{i+1:>4} {int(r['trigger']*100):>5}% {ht:>6} {hp:>5} {int(r['trail']*100):>5}% "
          f"{r['sharpe']:>7.2f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} "
          f"{r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}%")

# Top5 多维度
print("\n── 多维度Top5 ──")
metrics = ['sharpe', 'win_rate', 'avg_pnl', 'max_dd']
labels = ['Sharpe最优', '胜率最高', '均盈最高', 'DD最小']
for met, lab in zip(metrics, labels):
    top = sorted(results, key=lambda x: -x[met])[:3]
    print(f"\n{lab}:")
    for r in top:
        ht = f"{int(r['half_trail']*100)}%" if r['half_trail'] > 0 else "无"
        hp = f"{int(r['half_pct']*100)}%" if r['half_pct'] > 0 else "-"
        print(f"  tr={int(r['trigger']*100)}% ht={ht} hp={hp} tl={int(r['trail']*100)}% → {met}={r[met]:.3f} WR={r['win_rate']:.0f}% avg={r['avg_pnl']:+.2f}%")

# 推荐综合最优
print("\n── 综合推荐 ──")
# 综合得分：标准化sharpe + win_rate/100 + avg_pnl/10
norm_sh = [(r['sharpe'] - min(x['sharpe'] for x in results)) / (max(x['sharpe'] for x in results) - min(x['sharpe'] for x in results) + 1e-8) for r in results]
norm_wr = [(r['win_rate']/100 - min(x['win_rate']/100 for x in results)) / (max(x['win_rate']/100 for x in results) - min(x['win_rate']/100 for x in results) + 1e-8) for r in results]
norm_av = [(r['avg_pnl']/10 - min(x['avg_pnl']/10 for x in results)) / (max(x['avg_pnl']/10 for x in results) - min(x['avg_pnl']/10 for x in results) + 1e-8) for r in results]
norm_dd = [-(r['max_dd']/100 - min(x['max_dd']/100 for x in results)) / (max(x['max_dd']/100 for x in results) - min(x['max_dd']/100 for x in results) + 1e-8) for r in results]
composite = [0.4*nsh + 0.25*nwr + 0.2*nav + 0.15*ndd for nsh, nwr, nav, ndd in zip(norm_sh, norm_wr, norm_av, norm_dd)]
best_idx = np.argmax(composite)
best = results[best_idx]
print(f"综合最优: tr={int(best['trigger']*100)}% ht={int(best['half_trail']*100) if best['half_trail']>0 else 0}% hp={int(best['half_pct']*100) if best['half_pct']>0 else 0}% tl={int(best['trail']*100)}%")
print(f"  Sharpe={best['sharpe']:.2f} WR={best['win_rate']:.0f}% 均盈={best['avg_pnl']:+.2f}% DD={best['max_dd']:.1f}%")
print(f"\n权重: Sharpe 40%, WR 25%, 均盈 20%, DD 15%")
