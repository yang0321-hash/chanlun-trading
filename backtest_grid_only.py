#!/usr/bin/env python3
"""
网格回测: 使用预计算的2+3合一/3买信号
"""
import sys, os, time
import numpy as np
import pandas as pd
from collections import defaultdict

for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)

# ============================================================
# [1] 加载预计算信号
# ============================================================
print("加载信号...")
sig_df = pd.read_pickle('/workspace/backtest_2plus3_signals.pkl')
sig_df = sig_df.sort_values('date').reset_index(drop=True)
print(f"信号总数: {len(sig_df)}  分布: {sig_df['type'].value_counts().to_dict()}")
unique_dates = sorted(sig_df['date'].unique())
print(f"交易日: {len(unique_dates)} 天")

# 加载data_map (只需要code->df用于未来获取close数据)
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')

# ============================================================
# [2] 快速回测函数
# ============================================================
def run_backtest(sig_df, data_map,
                 max_positions=5,
                 initial_capital=1_000_000.0,
                 pos_2plus3=0.35,
                 pos_3buy=0.25,
                 sl_pct=0.06,
                 tp_trail_start=0.03,
                 tp_trail_pct=0.05,
                 hold_days=30,
                 commission=0.0003,
                 slippage=0.0005):
    equity = initial_capital
    equity_curve = [initial_capital]
    active_slots = [None] * max_positions
    code_exit = {}
    cooldown = {}
    total_trades = 0
    winning = 0
    pnl_list = []
    exit_reasons = defaultdict(int)
    type_stats = defaultdict(lambda: {'n': 0, 'win': 0, 'pnl': 0.0})

    # 按date分组，避免每次全量过滤
    sigs_by_date = {d: grp for d, grp in sig_df.groupby('date')}

    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        day_sigs = sigs_by_date.get(d, pd.DataFrame()).to_dict('records')

        # 结算
        for si in range(max_positions):
            slot = active_slots[si]
            if not slot:
                continue
            ed = slot['exit_date']
            if isinstance(ed, str):
                ed = pd.Timestamp(ed)
            if ed <= d_ts:
                alloc = equity / max_positions
                pnl_pct = slot['pnl_pct'] - slippage * 2
                equity += alloc * pnl_pct
                equity_curve.append(equity)
                total_trades += 1
                if pnl_pct > 0:
                    winning += 1
                pnl_list.append(pnl_pct)
                exit_reasons[slot['exit_reason']] += 1
                ts = type_stats[slot['btype']]
                ts['n'] += 1
                ts['win'] += 1 if pnl_pct > 0 else 0
                ts['pnl'] += pnl_pct
                active_slots[si] = None
            code_exit[slot['code']] = d

        # 开仓
        for row in day_sigs:
            code = row['code']
            btype = row['type']
            pos = int(row['pos'])
            price = row['price']
            pos_pct = row['pos_pct']

            if code in cooldown and d_ts < cooldown[code]:
                continue
            if code in code_exit and code_exit[code] == d:
                continue
            si = next((i for i in range(max_positions) if active_slots[i] is None), None)
            if si is None:
                break
            if code not in data_map:
                continue
            df_c = data_map[code]
            if pos >= len(df_c):
                continue

            if btype == '2plus3buy':
                target_pos_pct = pos_2plus3
            else:
                target_pos_pct = pos_3buy

            sl_price = price * (1 - sl_pct)
            exit_reason = 'timeout'
            exit_price = float(df_c['close'].iloc[-1])
            actual_exit_idx = len(df_c) - 1

            max_exit = min(pos + hold_days * 8, len(df_c) - 1)
            for bi in range(pos + 1, max_exit + 1):
                if bi >= len(df_c):
                    break
                low_bi = float(df_c['low'].iloc[bi])
                close_bi = float(df_c['close'].iloc[bi])
                if low_bi <= sl_price:
                    exit_price = sl_price
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

            pnl_pct = (exit_price - price) / price - commission * 2
            hold_days_actual = max(1, (actual_exit_idx - pos) // 8)
            exit_date = d_ts + pd.Timedelta(days=hold_days_actual)

            active_slots[si] = {
                'code': code, 'btype': btype,
                'pnl_pct': pnl_pct,
                'exit_date': exit_date,
                'exit_reason': exit_reason,
                'pos_pct': target_pos_pct
            }
            cooldown[code] = exit_date + pd.Timedelta(days=5)

    for si in range(max_positions):
        slot = active_slots[si]
        if slot:
            equity += (equity / max_positions) * (slot['pnl_pct'] - slippage * 2)

    if total_trades == 0:
        return None
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = dd.max()
    total_ret = (equity - initial_capital) / initial_capital
    ann_factor = 250 / 20
    ann_ret = total_ret * ann_factor
    std_pnl = np.std(pnl_list) if len(pnl_list) > 1 else 0.01
    sharpe = ann_ret / std_pnl if std_pnl > 0 else 0
    win_rate = winning / total_trades * 100
    avg_pnl = np.mean(pnl_list) * 100

    type_detail = {}
    for bt in ['2plus3buy', '3buy']:
        ts = type_stats[bt]
        if ts['n'] > 0:
            type_detail[bt] = {
                'n': ts['n'],
                'win': ts['win'],
                'avg': ts['pnl'] / ts['n'] * 100
            }

    return {
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_trades': total_trades,
        'final_equity': equity,
        'total_ret': total_ret * 100,
        'type_detail': type_detail,
        'exit_reasons': dict(exit_reasons)
    }


# ============================================================
# [3] 参数网格搜索
# ============================================================
print("\n网格参数搜索...")
print("=" * 70)

sl_list         = [0.03, 0.04, 0.05, 0.06, 0.08]
tp_start_list   = [0.03, 0.05, 0.08, 0.10]
tp_trail_list   = [0.05, 0.08, 0.10, 0.15]
pos23_list      = [0.25, 0.35, 0.45, 0.55]
pos3_list       = [0.15, 0.25, 0.35]
hold_days_list  = [20, 30, 40]

total_combos = (len(sl_list) * len(tp_start_list) * len(tp_trail_list) *
                len(pos23_list) * len(pos3_list) * len(hold_days_list))
print(f"总组合数: {total_combos}")

results = []
combo_count = 0
t0 = time.time()

for sl in sl_list:
    for tp_start in tp_start_list:
        for tp_trail in tp_trail_list:
            if tp_trail <= tp_start:
                continue
            for pos23 in pos23_list:
                for pos3 in pos3_list:
                    for hold in hold_days_list:
                        combo_count += 1
                        r = run_backtest(
                            sig_df, data_map,
                            max_positions=5,
                            initial_capital=1_000_000.0,
                            pos_2plus3=pos23,
                            pos_3buy=pos3,
                            sl_pct=sl,
                            tp_trail_start=tp_start,
                            tp_trail_pct=tp_trail,
                            hold_days=hold
                        )
                        if r:
                            r['params'] = {
                                'sl': sl, 'tp_start': tp_start,
                                'tp_trail': tp_trail,
                                'pos23': pos23, 'pos3': pos3,
                                'hold_days': hold
                            }
                            results.append(r)

                        if combo_count % 100 == 0:
                            elapsed = time.time() - t0
                            rate = combo_count / elapsed * 60
                            remain = (total_combos - combo_count) / rate if rate > 0 else 0
                            print(f"  {combo_count}/{total_combos} ({elapsed:.0f}s elapsed, ~{remain:.0f}s remaining)", flush=True)

print(f"\n完成 {combo_count} 组, 有效 {len(results)} 组, 耗时 {time.time()-t0:.0f}s\n")

# TOP10 Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)
print("=" * 70)
print("TOP10 (按Sharpe排序)")
print("=" * 70)
for i, r in enumerate(results[:10]):
    p = r['params']
    print(f"\n#{i+1} Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% WR={r['win_rate']:.1f}% 均盈={r['avg_pnl']:+.2f}%")
    print(f"   SL={p['sl']} TP={p['tp_start']}/{p['tp_trail']} P2+3={p['pos23']} P3={p['pos3']} HOLD={p['hold_days']}天")
    print(f"   总交易: {r['total_trades']}笔")
    td = r['type_detail']
    for bt in ['2plus3buy', '3buy']:
        if bt in td:
            print(f"   {bt}: {td[bt]['n']}笔 胜率{td[bt]['win']/td[bt]['n']*100:.0f}% 均盈{td[bt]['avg']:+.2f}%")
    print(f"   出场: {r['exit_reasons']}")

# TOP5 2+3合一质量
print("\n" + "=" * 70)
print("TOP5 (按2+3合一胜率*均盈 排序)")
print("=" * 70)
def score_23(r):
    td = r['type_detail']
    if '2plus3buy' in td and td['2plus3buy']['n'] >= 5:
        return (td['2plus3buy']['win'] / td['2plus3buy']['n']) * td['2plus3buy']['avg']
    return 0

results_23 = sorted(results, key=score_23, reverse=True)
for i, r in enumerate(results_23[:5]):
    p = r['params']
    td = r['type_detail']
    print(f"\n#{i+1} 2+3评分={score_23(r):.4f}")
    print(f"   Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% 均盈={r['avg_pnl']:+.2f}%")
    print(f"   SL={p['sl']} TP={p['tp_start']}/{p['tp_trail']} P2+3={p['pos23']} P3={p['pos3']} HOLD={p['hold_days']}天")
    for bt in ['2plus3buy', '3buy']:
        if bt in td:
            print(f"   {bt}: {td[bt]['n']}笔 胜率{td[bt]['win']/td[bt]['n']*100:.0f}% 均盈{td[bt]['avg']:+.2f}%")
    print(f"   出场: {r['exit_reasons']}")

# 回撤<20% TOP5
print("\n" + "=" * 70)
print("TOP5 (回撤<20% 内 Sharpe最优)")
print("=" * 70)
filtered = [r for r in results if r['max_dd'] < 20]
filtered.sort(key=lambda x: x['sharpe'], reverse=True)
for i, r in enumerate(filtered[:5]):
    p = r['params']
    td = r['type_detail']
    print(f"\n#{i+1} Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% WR={r['win_rate']:.1f}% 均盈={r['avg_pnl']:+.2f}%")
    print(f"   SL={p['sl']} TP={p['tp_start']}/{p['tp_trail']} P2+3={p['pos23']} P3={p['pos3']} HOLD={p['hold_days']}天")
    for bt in ['2plus3buy', '3buy']:
        if bt in td:
            print(f"   {bt}: {td[bt]['n']}笔 胜率{td[bt]['win']/td[bt]['n']*100:.0f}% 均盈{td[bt]['avg']:+.2f}%")

print(f"\n[参考] CC15基线: Sharpe 1.96, 回撤10.2%, 胜率66%, 均盈5.2%")
