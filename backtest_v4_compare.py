#!/usr/bin/env python3
"""
新框架回测 v4: 修正TP参数，对比skill最优配置
对比实验：
  A. skill推荐: SL=8%, TP=3%+8%动态 (Sharpe=5.25 from skill)
  B. v3旧结果:  SL=6%, TP=5%+8%固定 (Sharpe=5.21 from v3旧)
  C. 组合验证:  SL=8%, TP=5%+8%固定
  D. 纯基准:    SL=6%, TP=3%+8%动态
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_unified')

print("=" * 60)
print("加载信号...")
sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
print(f"信号总数: {len(sig_df)}  分布: {sig_df['type'].value_counts().to_dict()}")
print(f"日期范围: {sig_df['date'].min()} ~ {sig_df['date'].max()}")
print(f"信号列: {list(sig_df.columns)}")

print("\n加载数据...")
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
print(f"  {len(data_map)} 只股票")

# ============================================================
# 回测函数
# ============================================================
def run_backtest(sig_df, data_map,
                 max_positions=5,
                 initial_capital=1_000_000.0,
                 pos_2buy=0.30,
                 pos_2plus3=0.30,
                 pos_3buy=0.30,
                 sl_pct=0.08,
                 tp_trail_start=0.03,
                 tp_trail_pct=0.08,
                 hold_days=30,
                 commission=0.0003,
                 slippage=0.0005,
                 filter_3buy=False,
                 label=''):

    unique_dates = sorted(sig_df['date'].unique())
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

    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        # 清结算
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

        day_sigs = sig_df[sig_df['date'] == d]
        for _, row in day_sigs.iterrows():
            code = row['code']
            btype = row['type']
            if filter_3buy and btype == '3buy':
                continue
            pos = int(row['pos'])
            price = row['price']
            sl_price_signal = row['sl_price']

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

            # 仓位 (cap 30%)
            if btype == '2plus3buy':
                target_pos_pct = min(pos_2plus3, 0.30)
            elif btype == '2buy':
                target_pos_pct = min(pos_2buy, 0.30)
            else:
                target_pos_pct = min(pos_3buy, 0.30)

            # 止损
            sl_price = max(price * (1 - sl_pct), float(sl_price_signal))

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
                # 动态止盈: 触發tp_trail_start后, 回调>tp_trail_pct出局
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
            d_ts_end = unique_dates[-1]
            d_end_ts = pd.Timestamp(d_ts_end)
            ed = slot['exit_date']
            if isinstance(ed, str):
                ed = pd.Timestamp(ed)
            if ed > d_end_ts:
                alloc = equity / max_positions
                pnl_pct = slot['pnl_pct']
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

    if total_trades == 0:
        return None

    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100
    win_rate = winning / total_trades * 100
    avg_pnl = np.mean(pnl_list) * 100

    # 按类型统计
    type_detail = {}
    for bt, ts in type_stats.items():
        n = ts['n']
        win = ts['win']
        pnl_sum = ts['pnl']
        type_detail[bt] = {
            'n': n, 'win': win,
            'win_rate': win / n * 100 if n > 0 else 0,
            'avg': pnl_sum / n * 100 if n > 0 else 0
        }

    return {
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd, 1),
        'win_rate': round(win_rate, 1),
        'avg_pnl': round(avg_pnl, 2),
        'total_trades': total_trades,
        'exit_reasons': dict(exit_reasons),
        'type_detail': type_detail,
    }

# ============================================================
# 对比实验
# ============================================================
t0 = time.time()
results = []

configs = [
    # A. skill推荐: SL=8%, TP=3%+8%动态
    dict(sl_pct=0.08, tp_trail_start=0.03, tp_trail_pct=0.08, label='A. skill推荐(SL=8% TP=3+8%动态)'),
    # B. v3旧结果: SL=6%, TP=5%+8%固定
    dict(sl_pct=0.06, tp_trail_start=0.05, tp_trail_pct=0.08, label='B. v3旧最优(SL=6% TP=5+8%固定)'),
    # C. 组合: SL=8%, TP=5%+8%固定
    dict(sl_pct=0.08, tp_trail_start=0.05, tp_trail_pct=0.08, label='C. SL=8% TP=5+8%固定'),
    # D. 纯基准: SL=6%, TP=3%+8%动态
    dict(sl_pct=0.06, tp_trail_start=0.03, tp_trail_pct=0.08, label='D. SL=6% TP=3+8%动态'),
    # E. 去3buy + skill最优
    dict(sl_pct=0.08, tp_trail_start=0.03, tp_trail_pct=0.08, filter_3buy=True, label='E. 去3buy + SL=8% TP=3+8%'),
    # F. 去3buy + SL=6%动态
    dict(sl_pct=0.06, tp_trail_start=0.03, tp_trail_pct=0.08, filter_3buy=True, label='F. 去3buy + SL=6% TP=3+8%'),
]

for cfg in configs:
    label = cfg.pop('label')
    filter_3buy = cfg.pop('filter_3buy', False)
    print(f"\n{'='*60}")
    print(f"[{label}]")
    r = run_backtest(sig_df, data_map,
                     max_positions=5, initial_capital=1_000_000.0,
                     pos_2buy=0.30, pos_2plus3=0.30, pos_3buy=0.30,
                     hold_days=30,
                     filter_3buy=filter_3buy,
                     **cfg)
    if r:
        print(f"  Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% "
              f"WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% "
              f"交易={r['total_trades']}笔")
        td = r['type_detail']
        for bt in ['2buy', '2plus3buy', '3buy']:
            if bt in td:
                print(f"    {bt}: {td[bt]['n']}笔 胜率{td[bt]['win_rate']:.0f}% 均盈{td[bt]['avg']:+.2f}%")
        print(f"    出场: {r['exit_reasons']}")
        results.append({'label': label, **r, 'cfg': cfg, 'filter_3buy': filter_3buy})

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"总耗时: {elapsed:.0f}s")

# 找最优
if results:
    best = max(results, key=lambda x: x['sharpe'])
    print(f"\n🏆 最优: {best['label']}")
    print(f"   Sharpe={best['sharpe']:.2f} DD={best['max_dd']:.1f}% "
          f"WR={best['win_rate']:.0f}% 均盈={best['avg_pnl']:+.2f}%")

    print("\n--- 汇总表 ---")
    print(f"{'配置':<40} {'Sharpe':>7} {'DD%':>6} {'WR%':>5} {'均盈%':>7} {'交易数':>6}")
    print('-' * 75)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        print(f"{r['label']:<40} {r['sharpe']:>7.2f} {r['max_dd']:>6.1f} "
              f"{r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['total_trades']:>6}")

# 保存
with open('/workspace/backtest_v4_compare_result.pkl', 'wb') as f:
    pickle.dump({'results': results}, f)
print("\n结果已保存: backtest_v4_compare_result.pkl")
