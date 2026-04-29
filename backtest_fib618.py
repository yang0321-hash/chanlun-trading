#!/usr/bin/env python3
"""
回测: 0.618过滤对策略的影响
对比: 有0.618过滤 vs 无0.618过滤
"""
import os, sys, time, struct, pickle, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_unified')

# ── TDX读取 (复用scanner缓存，~8948只全A数据) ─────────────────
CACHE_FILE = '/workspace/scanner_v3_cache.pkl'

def load_all_data():
    """从scanner缓存加载全部数据"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            if len(data) > 1000:
                print(f'  缓存加载: {len(data)} 只股票')
                return data
        except Exception as e:
            print(f'  缓存读取失败: {e}')
    return None

# ── 信号生成 (scanner_v3_mp.py核心逻辑) ───────────────────────
_global_data = None

def set_data(data_map):
    global _global_data
    _global_data = data_map

def generate_signals(code, use_fib618=True):
    """从缓存数据生成信号，use_fib618=True=加0.618过滤"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('cc15', '/workspace/chanlun_unified/signal_engine_cc15.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    EngineCls = mod.SignalEngine

    if _global_data is None: return []
    df = _global_data.get(code)
    if df is None: return []
    if len(df) < 60: return []

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

    sigs = []
    dates = df.index.tolist()

    for pos in buy_2_positions:
        if pos < 5 or pos >= n - 1: continue
        price = float(close_s.iloc[pos])
        sl_price = float(low_s.iloc[pos]) * 0.97
        if use_fib618:
            ok618, _ = check_fib618(strokes_raw, price)
            if not ok618: continue
        market = 'sz' if code.startswith('SZ') else 'sh'
        sigs.append({'code': code, 'market': market, 'entry_idx': pos,
                     'entry_price': price, 'sl_price': sl_price, 'date': str(dates[pos])[:10]})
    return sigs

# ── 回测引擎 ─────────────────────────────────────────────────
_global_backtest_data = None

def set_bt_data(data_map):
    global _global_backtest_data
    _global_backtest_data = data_map

def run_backtest(sig):
    """固定仓位回测: T+1开收盘价，SL=3%止损/TP=5%止盈，T+1开盘价出"""
    if _global_backtest_data is None: return []
    code = sig['code']
    df = _global_backtest_data.get(code)
    if df is None: return []
    close = df['close'].values
    open_p = df['open'].values
    n = len(df)
    entry_idx = sig['entry_idx']
    entry_price = sig['entry_price']
    sl_price = sig['sl_price']

    if entry_idx + 1 >= n: return []
    buy_price = float(open_p[entry_idx + 1])
    if buy_price <= 0: return []

    tp_price = buy_price * 1.05
    exit_idx = None; exit_price = None; sl_triggered = tp_triggered = False

    for d in range(entry_idx + 2, min(entry_idx + 30, n)):
        lo = float(open_p[d])
        hi = float(df['high'].iloc[d])
        if lo <= sl_price:
            sl_triggered = True; exit_idx = d; exit_price = sl_price; break
        if hi >= tp_price:
            tp_triggered = True; exit_idx = d; exit_price = tp_price; break

    if exit_idx is None or exit_idx >= n: return []
    exit_price = float(open_p[exit_idx])

    ret_pct = (exit_price - buy_price) / buy_price * 100
    hold_days = exit_idx - entry_idx - 1

    return [{
        'code': code, 'entry_date': sig['date'],
        'buy_price': buy_price, 'exit_price': exit_price,
        'ret_pct': ret_pct, 'hold_days': hold_days,
        'hit_sl': 1 if sl_triggered else 0,
        'exit_reason': 'SL' if sl_triggered else 'TP5'
    }]

# ── 主程序 ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("回测: 0.618过滤影响")
    print("=" * 60)

    t0 = time.time()

    # 加载缓存数据
    print("加载scanner缓存...")
    all_data = load_all_data()
    if all_data is None:
        print("缓存不存在，先运行scanner_v3_mp.py生成缓存")
        return
    set_data(all_data)
    set_bt_data(all_data)

    codes = list(all_data.keys())
    print(f"  共 {len(codes)} 只股票")

    # 采样800只
    np.random.seed(42)
    sample = np.random.choice(len(codes), min(800, len(codes)), replace=False)
    sample_codes = [codes[i] for i in sample]
    print(f"采样: {len(sample_codes)} 只")

    # 生成信号 (无过滤 vs 有过滤)
    print("生成信号中...")
    sigs_no618 = []
    sigs_with618 = []

    for i, code in enumerate(sample_codes):
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(sample_codes)} ({time.time()-t0:.0f}s)")
        try:
            no6 = generate_signals(code, use_fib618=False)
            with6 = generate_signals(code, use_fib618=True)
            sigs_no618.extend(no6)
            sigs_with618.extend(with6)
        except: pass

    print(f"\n信号数量:")
    print(f"  无0.618过滤: {len(sigs_no618)} 个")
    print(f"  有0.618过滤: {len(sigs_with618)} 个")
    diff = len(sigs_no618) - len(sigs_with618)
    print(f"  过滤掉: {diff} 个 ({diff/max(len(sigs_no618),1)*100:.1f}%)")

    # 回测
    def run_bt_all(sigs):
        trades = []
        for sig in sigs:
            t = run_backtest(sig)
            trades.extend(t)
        return trades

    print(f"\n回测中...")
    trades_no618 = run_bt_all(sigs_no618)
    trades_with618 = run_bt_all(sigs_with618)

    def stats(trades, label):
        if not trades:
            print(f"\n{label}: 无交易"); return
        df = pd.DataFrame(trades)
        wins = df[df['ret_pct'] > 0]
        loss = df[df['ret_pct'] <= 0]
        total_ret = df['ret_pct'].sum()
        avg_ret = df['ret_pct'].mean()
        win_rate = len(wins) / len(df) * 100
        max_dd = df['ret_pct'].min()
        sharpe = df['ret_pct'].mean() / max(df['ret_pct'].std(), 0.01) * np.sqrt(252)

        print(f"\n{'='*50}")
        print(f"{label}")
        print(f"{'='*50}")
        print(f"  总交易:    {len(df)} 笔")
        print(f"  盈利:      {len(wins)} ({win_rate:.1f}%)")
        print(f"  亏损:      {len(loss)}")
        print(f"  总收益:    {total_ret:.1f}%")
        print(f"  平均收益:  {avg_ret:.2f}%")
        print(f"  最大单笔:  {df['ret_pct'].max():.1f}%")
        fmin = df['ret_pct'].min()
        print(f"  最大亏损:  {fmin:.1f}% (止损率: {df['hit_sl'].mean()*100:.1f}%)")
        print(f"  Sharpe:    {sharpe:.2f}")
        print(f"  平均持仓:  {df['hold_days'].mean():.1f} 天")

        # 按年度
        if 'entry_date' in df.columns:
            df['year'] = df['entry_date'].str[:4]
            yearly = df.groupby('year')['ret_pct'].agg(['sum','count'])
            print(f"  年度明细:")
            for yr, row in yearly.iterrows():
                print(f"    {yr}: {row['sum']:+.1f}% ({int(row['count'])}笔)")

    stats(trades_no618, "❌ 无0.618过滤")
    stats(trades_with618, "✅ 有0.618过滤")

    # 对比
    if trades_no618 and trades_with618:
        dn = pd.DataFrame(trades_no618)['ret_pct']
        dw = pd.DataFrame(trades_with618)['ret_pct']
        print(f"\n{'='*50}")
        print("对比: 有618 vs 无618")
        print(f"{'='*50}")
        print(f"  交易数:   {len(dw)} vs {len(dn)}  (减少 {len(dn)-len(dw)} 笔)")
        print(f"  胜率:     {len(dw[dw>0])/len(dw)*100:.1f}% vs {len(dn[dn>0])/len(dn)*100:.1f}%")
        print(f"  平均收益: {dw.mean():.3f}% vs {dn.mean():.3f}%")
        print(f"  总收益:   {dw.sum():.1f}% vs {dn.sum():.1f}%")
        print(f"  Sharpe:   {dw.mean()/max(dw.std(),0.01)*np.sqrt(252):.2f} vs {dn.mean()/max(dn.std(),0.01)*np.sqrt(252):.2f}")

    print(f"\n总耗时: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
