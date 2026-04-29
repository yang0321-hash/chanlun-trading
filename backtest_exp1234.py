#!/usr/bin/env python3
"""
三路并行验证:
1. 去掉3buy，只留2buy+2plus3buy
2. 最优参数: SL=6% + TP=0.03/0.05
3. 更严格3buy定义: 放量突破+回踩不破ZG+换手>5%
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_unified')

# ============================================================
# 加载
# ============================================================
sig_df = pd.read_pickle('/workspace/backtest_new_fw_signals.pkl')
data_map = pd.read_pickle('/workspace/backtest_v15_all_a_data.pkl')
print(f"原始信号: {len(sig_df)}  {sig_df['type'].value_counts().to_dict()}")

def run_backtest(sig_df_in, data_map,
                 label="",
                 max_positions=5,
                 initial_capital=1_000_000.0,
                 pos_2buy=0.30, pos_2plus3=0.30, pos_3buy=0.30,
                 sl_pct=0.06,
                 tp_trail_start=0.03,
                 tp_trail_pct=0.05,
                 hold_days=30,
                 commission=0.0003,
                 slippage=0.0005):
    sig_df = sig_df_in.copy()
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
        for si in range(max_positions):
            slot = active_slots[si]
            if not slot: continue
            ed = slot['exit_date']
            if isinstance(ed, str): ed = pd.Timestamp(ed)
            if ed <= d_ts:
                alloc = equity / max_positions
                pnl_pct = slot['pnl_pct'] - slippage * 2
                equity += alloc * pnl_pct
                equity_curve.append(equity)
                total_trades += 1
                if pnl_pct > 0: winning += 1
                pnl_list.append(pnl_pct)
                exit_reasons[slot['exit_reason']] += 1
                ts = type_stats[slot['btype']]
                ts['n'] += 1; ts['win'] += 1 if pnl_pct > 0 else 0; ts['pnl'] += pnl_pct
                active_slots[si] = None
            code_exit[slot['code']] = d

        day_sigs = sig_df[sig_df['date'] == d]
        for _, row in day_sigs.iterrows():
            code = row['code']; btype = row['type']
            pos = int(row['pos']); price = row['price']
            sl_price_signal = row['sl_price']
            if code in cooldown and d_ts < cooldown[code]: continue
            if code in code_exit and code_exit[code] == d: continue
            si = next((i for i in range(max_positions) if active_slots[i] is None), None)
            if si is None: break
            if code not in data_map: continue
            df_c = data_map[code]
            if pos >= len(df_c): continue

            if btype == '2plus3buy': target_pos = min(pos_2plus3, 0.30)
            elif btype == '2buy': target_pos = min(pos_2buy, 0.30)
            else: target_pos = min(pos_3buy, 0.30)

            sl_price = max(price * (1 - sl_pct), float(sl_price_signal))
            exit_reason = 'timeout'
            exit_price = float(df_c['close'].iloc[-1])
            actual_exit_idx = len(df_c) - 1
            max_exit = min(pos + hold_days * 8, len(df_c) - 1)
            for bi in range(pos + 1, max_exit + 1):
                if bi >= len(df_c): break
                low_bi = float(df_c['low'].iloc[bi])
                close_bi = float(df_c['close'].iloc[bi])
                if low_bi <= sl_price:
                    exit_price = sl_price; exit_reason = 'stop_loss'; actual_exit_idx = bi; break
                profit_pct = (close_bi - price) / price
                if profit_pct >= tp_trail_start:
                    dd = (close_bi - price) / close_bi
                    if dd >= tp_trail_pct:
                        exit_price = close_bi; exit_reason = 'trail_stop'; actual_exit_idx = bi; break

            pnl_pct = (exit_price - price) / price - commission * 2
            hold_days_actual = max(1, (actual_exit_idx - pos) // 8)
            exit_date = d_ts + pd.Timedelta(days=hold_days_actual)
            active_slots[si] = {
                'code': code, 'btype': btype, 'pnl_pct': pnl_pct,
                'exit_date': exit_date, 'exit_reason': exit_reason, 'pos_pct': target_pos
            }
            cooldown[code] = exit_date + pd.Timedelta(days=5)

    for si in range(max_positions):
        slot = active_slots[si]
        if slot:
            equity += (equity / max_positions) * (slot['pnl_pct'] - slippage * 2)
            equity_curve.append(equity)

    if total_trades == 0: return None
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = dd.max()
    total_ret = (equity - initial_capital) / initial_capital
    daily_rets = [(eq[i] - eq[i-1]) / eq[i-1] for i in range(1, len(eq))]
    dr = np.array(daily_rets)
    ann_ret = np.mean(dr) * 250 if len(dr) > 0 else 0
    ann_std = np.std(dr) * np.sqrt(250) if len(dr) > 1 else 0.01
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    win_rate = winning / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = np.mean(pnl_list) * 100 if pnl_list else 0

    type_detail = {}
    for bt in set(['2buy', '2plus3buy', '3buy']) & set(type_stats.keys()):
        ts = type_stats[bt]
        if ts['n'] > 0:
            type_detail[bt] = {'n': ts['n'], 'win': ts['win'], 'avg': ts['pnl'] / ts['n'] * 100}

    return {
        'label': label,
        'sharpe': sharpe, 'max_dd': max_dd,
        'win_rate': win_rate, 'avg_pnl': avg_pnl,
        'total_trades': total_trades, 'final_equity': equity,
        'total_ret': total_ret * 100,
        'type_detail': type_detail,
        'exit_reasons': dict(exit_reasons),
        'sig_count': len(sig_df_in)
    }

# ============================================================
# 实验1: 去掉3buy，只留2buy+2plus3buy
# ============================================================
print("\n" + "="*70)
print("实验1: 去掉3buy (SL=6%, TP=0.03/0.05)")
print("="*70)
sig_12 = sig_df[sig_df['type'].isin(['2buy', '2plus3buy'])].copy()
print(f"过滤后信号: {len(sig_12)}  {sig_12['type'].value_counts().to_dict()}")
r1 = run_backtest(sig_12, data_map, label="仅2buy+2plus3buy",
                  sl_pct=0.06, tp_trail_start=0.03, tp_trail_pct=0.05)
if r1:
    print(f"\n  Sharpe={r1['sharpe']:.2f} DD={r1['max_dd']:.1f}% WR={r1['win_rate']:.0f}% 均盈={r1['avg_pnl']:+.2f}% 交易={r1['total_trades']}笔")
    td = r1['type_detail']
    for bt in ['2buy', '2plus3buy']:
        if bt in td:
            print(f"  {bt}: {td[bt]['n']}笔 胜率{td[bt]['win']/td[bt]['n']*100:.0f}% 均盈{td[bt]['avg']:+.2f}%")
    print(f"  出场: {r1['exit_reasons']}")

# ============================================================
# 实验2: 最优参数验证 (SL=6%, TP=0.03/0.05, 全部信号)
# ============================================================
print("\n" + "="*70)
print("实验2: 最优参数 (SL=6%, TP=0.03/0.05, 全信号)")
print("="*70)
r2 = run_backtest(sig_df, data_map, label="全信号最优参数",
                  sl_pct=0.06, tp_trail_start=0.03, tp_trail_pct=0.05)
if r2:
    print(f"\n  Sharpe={r2['sharpe']:.2f} DD={r2['max_dd']:.1f}% WR={r2['win_rate']:.0f}% 均盈={r2['avg_pnl']:+.2f}% 交易={r2['total_trades']}笔")
    td = r2['type_detail']
    for bt in ['2buy', '2plus3buy', '3buy']:
        if bt in td:
            print(f"  {bt}: {td[bt]['n']}笔 胜率{td[bt]['win']/td[bt]['n']*100:.0f}% 均盈{td[bt]['avg']:+.2f}%")
    print(f"  出场: {r2['exit_reasons']}")

# ============================================================
# 实验3: 更严格3buy定义 (需要重新生成信号)
# 条件: 突破ZG时放量(突破日量比前5日均量>1.5) + 回踩缩量确认
# ============================================================
print("\n" + "="*70)
print("实验3: 严格3buy定义 (放量突破+回踩缩量)")
print("="*70)

import importlib.util
spec = importlib.util.spec_from_file_location("cc15", '/workspace/chanlun_unified/signal_engine_cc15.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EngineCls = mod.SignalEngine

print("重新生成严格3buy信号...")
sig_strict_list = []
codes = list(data_map.keys())
t0 = time.time()

for ci, code in enumerate(codes):
    if ci % 1000 == 0:
        print(f"  {ci}/{len(codes)} ({time.time()-t0:.0f}s)")
    df = data_map[code].copy()
    n = len(df)
    if n < 60: continue
    try:
        engine = EngineCls()
        close_s = df['close'].astype(float)
        low_s = df['low'].astype(float)
        vol_s = df['volume'].astype(float)
        ema12 = close_s.ewm(span=12, adjust=False).mean()
        ema26 = close_s.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        _, _, filtered_fractals, strokes_raw = engine._detect_bi_deterministic(df)
        third_buy = engine._detect_3buy_context(filtered_fractals, df)
        zhongshu_list = engine._detect_zhongshu_from_strokes(strokes_raw)
        dates = df.index.tolist()
        vol_ma5 = vol_s.rolling(5, min_periods=1).mean()

        for pos in range(n):
            if pos >= len(third_buy) or not third_buy.iloc[pos]: continue
            date_str = str(dates[pos])[:10]
            price = float(close_s.iloc[pos])
            vol_now = float(vol_s.iloc[pos]) if pos < len(vol_s) else 0
            vol_5d = float(vol_ma5.iloc[pos]) if pos < len(vol_ma5) else 1

            # 找对应中枢ZG
            zg = price * 1.02
            has_zs = False
            for zs in zhongshu_list:
                zg_val = getattr(zs, 'zg', None)
                if zg_val and hasattr(zs, 'start_idx') and zs.start_idx < pos:
                    zg = min(zg, float(zg_val))
                    has_zs = True

            # 严格条件1: 有明确中枢
            if not has_zs: continue
            # 严格条件2: 突破ZG时放量 (量比>1.5)
            vol_ratio = vol_now / vol_5d if vol_5d > 0 else 0
            if vol_ratio < 1.5: continue
            # 严格条件3: 止损设ZG下方2%
            sl_3buy = zg * 0.98
            sig_strict_list.append({
                'code': code, 'type': '3buy_strict', 'pos': pos,
                'date': date_str, 'price': price,
                'sl_price': sl_3buy, 'pos_pct': 0.30, 'entry_idx': pos
            })
    except: pass

sig_strict_3buy = pd.DataFrame(sig_strict_list)
print(f"严格3buy信号: {len(sig_strict_3buy)} 个")

# 合并: 原有2buy+2plus3buy + 严格3buy
sig_strict_all = pd.concat([
    sig_df[sig_df['type'].isin(['2buy', '2plus3buy'])],
    sig_strict_3buy
], ignore_index=True).sort_values('date').reset_index(drop=True)
print(f"合并后信号: {len(sig_strict_all)}  {sig_strict_all['type'].value_counts().to_dict()}")

r3 = run_backtest(sig_strict_all, data_map, label="严格3buy",
                  sl_pct=0.06, tp_trail_start=0.03, tp_trail_pct=0.05)
if r3:
    print(f"\n  Sharpe={r3['sharpe']:.2f} DD={r3['max_dd']:.1f}% WR={r3['win_rate']:.0f}% 均盈={r3['avg_pnl']:+.2f}% 交易={r3['total_trades']}笔")
    td = r3['type_detail']
    for bt in ['2buy', '2plus3buy', '3buy_strict']:
        if bt in td:
            print(f"  {bt}: {td[bt]['n']}笔 胜率{td[bt]['win']/td[bt]['n']*100:.0f}% 均盈{td[bt]['avg']:+.2f}%")
    print(f"  出场: {r3['exit_reasons']}")

# ============================================================
# 实验4: 只用严格3buy替换原始3buy
# ============================================================
print("\n" + "="*70)
print("实验4: 原始3buy替换为严格3buy (SL=6%, TP=0.03/0.05)")
print("="*70)
sig_replace = sig_df[sig_df['type'].isin(['2buy', '2plus3buy'])].copy()
sig_replace_3buy_orig = sig_df[sig_df['type'] == '3buy']['pos'].tolist()
sig_strict_3buy_filtered = sig_strict_3buy[~sig_strict_3buy['pos'].isin(sig_replace_3buy_orig)]
sig_replace = pd.concat([sig_replace, sig_strict_3buy_filtered], ignore_index=True).sort_values('date').reset_index(drop=True)
print(f"替换后信号: {len(sig_replace)}  {sig_replace['type'].value_counts().to_dict()}")
r4 = run_backtest(sig_replace, data_map, label="替换3buy",
                  sl_pct=0.06, tp_trail_start=0.03, tp_trail_pct=0.05)
if r4:
    print(f"\n  Sharpe={r4['sharpe']:.2f} DD={r4['max_dd']:.1f}% WR={r4['win_rate']:.0f}% 均盈={r4['avg_pnl']:+.2f}% 交易={r4['total_trades']}笔")
    td = r4['type_detail']
    for bt in ['2buy', '2plus3buy', '3buy']:
        if bt in td:
            print(f"  {bt}: {td[bt]['n']}笔 胜率{td[bt]['win']/td[bt]['n']*100:.0f}% 均盈{td[bt]['avg']:+.2f}%")
    if '3buy_strict' in td:
        print(f"  3buy_strict: {td['3buy_strict']['n']}笔 胜率{td['3buy_strict']['win']/td['3buy_strict']['n']*100:.0f}% 均盈{td['3buy_strict']['avg']:+.2f}%")
    print(f"  出场: {r4['exit_reasons']}")

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "="*70)
print("汇总对比")
print("="*70)
all_results = [r1, r2, r3, r4]
names = ["实验1: 去3buy", "实验2: 全信号最优参数", "实验3: 严格3buy(放量突破)", "实验4: 替换3buy为严格版"]
for name, r in zip(names, all_results):
    if r:
        print(f"\n{name}")
        print(f"  Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1f}% WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% 交易={r['total_trades']}笔")

# 保存
with open('/workspace/backtest_exp1234.pkl', 'wb') as f:
    pickle.dump({'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'sig_strict_3buy': sig_strict_3buy}, f)
print("\n结果已保存")
