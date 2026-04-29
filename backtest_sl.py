#!/usr/bin/env python3
"""
回测: SL参数对比 3% vs 6%
其他参数固定: TP=5%, 0.618过滤, 仓位30%
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_unified')

CACHE_FILE = '/workspace/scanner_v3_cache.pkl'

def load_all_data():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            if len(data) > 1000:
                print(f'  缓存加载: {len(data)} 只')
                return data
        except: pass
    return None

_global_data = None
def set_data(d): global _global_data; _global_data = d

def generate_signals(use_fib618=True, sl_pct=0.03):
    """生成信号，固定0.618过滤"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('cc15', '/workspace/chanlun_unified/signal_engine_cc15.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    EngineCls = mod.SignalEngine

    all_sigs = []
    codes = list(_global_data.keys())

    for code in codes:
        df = _global_data.get(code)
        if df is None or len(df) < 60: continue

        n = len(df)
        engine = EngineCls()
        close_s = df['close'].astype(float)
        low_s = df['low'].astype(float)
        ema12 = close_s.ewm(span=12, adjust=False).mean()
        ema26 = close_s.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2

        buy_bi, _, filtered_fractals, strokes_raw = engine._detect_bi_deterministic(df)
        buy_div, _, _, _ = engine._compute_area_divergence(strokes_raw, macd_hist, n)
        buy_2_set = engine._detect_2buy(strokes_raw, buy_div, n)

        if isinstance(buy_2_set, set):
            buy_2_positions = buy_2_set
        elif hasattr(buy_2_set, 'iloc'):
            buy_2_positions = {i for i in range(n) if buy_2_set.iloc[i]}
        else:
            buy_2_positions = set()

        # 0.618函数
        def check_fib618(strokes_raw, price):
            up_strokes = [s for s in strokes_raw if s['start_type'] == 'bottom' and s['end_type'] == 'top']
            if not up_strokes: return True, None
            recent_up = up_strokes[-1]
            high_point = max(recent_up['start_val'], recent_up['end_val'])
            down_before = None
            for s in reversed(strokes_raw):
                if s['start_type'] == 'top' and s['end_type'] == 'bottom':
                    if s['end_idx'] < recent_up['start_idx']:
                        down_before = s; break
            if down_before is None: return True, None
            low_before = min(down_before['start_val'], down_before['end_val'])
            fib618 = high_point - (high_point - low_before) * 0.618
            return (price >= fib618), fib618

        dates = df.index.tolist()
        cutoff = n - 30

        for pos in buy_2_positions:
            if pos < cutoff or pos >= n - 1: continue
            price = float(close_s.iloc[pos])
            if use_fib618:
                ok618, _ = check_fib618(strokes_raw, price)
                if not ok618: continue
            # SL = 买入价 × (1 - sl_pct)
            sl_price = price * (1 - sl_pct)
            all_sigs.append({
                'code': code, 'entry_idx': pos,
                'entry_price': price, 'sl_price': sl_price,
                'date': str(dates[pos])[:10]
            })
    return all_sigs

def run_backtest(sigs, tp_pct=0.05):
    """回测: T+1开盘买, SL/TP触发后T+1开盘卖"""
    trades = []
    for sig in sigs:
        code = sig['code']
        df = _global_data.get(code)
        if df is None: continue
        open_p = df['open'].values
        high_arr = df['high'].values
        n = len(df)
        entry_idx = sig['entry_idx']
        entry_price = sig['entry_price']
        sl_price = sig['sl_price']

        if entry_idx + 1 >= n: continue
        buy_price = float(open_p[entry_idx + 1])
        if buy_price <= 0: continue

        tp_price = buy_price * (1 + tp_pct)
        exit_idx = None; exit_price = None
        sl_triggered = tp_triggered = False

        for d in range(entry_idx + 2, min(entry_idx + 30, n)):
            lo = float(open_p[d])
            hi = float(high_arr[d])
            if lo <= sl_price:
                sl_triggered = True; exit_idx = d; exit_price = sl_price; break
            if hi >= tp_price:
                tp_triggered = True; exit_idx = d; exit_price = tp_price; break

        if exit_idx is None or exit_idx >= n: continue
        exit_price = float(open_p[exit_idx])

        ret_pct = (exit_price - buy_price) / buy_price * 100
        hold_days = exit_idx - entry_idx - 1

        trades.append({
            'code': code, 'entry_date': sig['date'],
            'buy_price': buy_price, 'exit_price': exit_price,
            'ret_pct': ret_pct, 'hold_days': hold_days,
            'hit_sl': 1 if sl_triggered else 0,
            'exit_reason': 'SL' if sl_triggered else 'TP5'
        })
    return trades

def stats(trades, label):
    if not trades:
        print(f"\n{label}: 无交易"); return
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

    # 年度
    df['year'] = df['entry_date'].str[:4]
    yearly = df.groupby('year')['ret_pct'].agg(['sum','count'])
    print(f"  年度明细:")
    for yr in sorted(yearly.index):
        r = yearly.loc[yr]
        print(f"    {yr}: {r['sum']:+.1f}% ({int(r['count'])}笔)")

    return df

def main():
    t0 = time.time()
    print("=" * 55)
    print("SL参数对比回测: 3% vs 6%")
    print("固定: TP=5%, 0.618过滤, 仓位30%")
    print("=" * 55)

    all_data = load_all_data()
    if all_data is None:
        print("缓存不存在，先运行scanner"); return
    set_data(all_data)
    print(f"股票数: {len(all_data)}")

    # 生成信号 (同一套信号，不同SL)
    print("\n生成信号(0.618过滤)...")
    sigs_base = generate_signals(use_fib618=True, sl_pct=0.03)
    print(f"  信号数: {len(sigs_base)}")

    for sl_pct in [0.03, 0.04, 0.05, 0.06, 0.08]:
        # 用同一套入场点，只改SL
        sigs = []
        for s in sigs_base:
            s2 = s.copy()
            s2['sl_price'] = s2['entry_price'] * (1 - sl_pct)
            sigs.append(s2)

        print(f"\n回测 SL={sl_pct*100:.0f}%...")
        trades = run_backtest(sigs)
        stats(trades, f"SL={sl_pct*100:.0f}%")

    print(f"\n总耗时: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
