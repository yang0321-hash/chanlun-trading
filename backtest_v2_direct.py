"""
v2.0 过滤器回测 - 重写版
使用CC15引擎_detect_*方法直接提取缠论信号
对比: 基准(固定10%仓位) vs v2.0(大盘分层+仓位分级)

市场分层:
  up:    MA5>MA10 且 price>MA5
  trend: 其他情况
  down:  MA5<MA10 且 price<MA5

仓位分级(v2.0):
  up:    1买20%  2买40%  3买50%
  trend: 1买10%  2买30%  3买40%
  down:  1买10%  2买0%   3买0%
"""

import sys, os, struct, random, pickle, inspect
import numpy as np
import pandas as pd

TDX_SH = '/workspace/tdx_data/sh/lday'
TDX_SZ = '/workspace/tdx_data/sz/lday'
OUTPUT_DIR = '/workspace/backtest_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ CC15引擎 ============
sys.path.insert(0, '/workspace/chanlun_unified')
import importlib.util
spec = importlib.util.spec_from_file_location("cc15", '/workspace/chanlun_unified/signal_engine_cc15.py')
cc15_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc15_mod)
SignalEngine = cc15_mod.SignalEngine

# ============ 读取TDX日线 ============
def load_day_data(code):
    """读取TDX日线(32字节/条, 自动处理.bak上海截断情况)"""
    if code.startswith('sh'):
        day_path = f"{TDX_SH}/{code}.day"
        bak_path = f"{TDX_SH}/{code}.day.bak"
        if os.path.exists(day_path) and os.path.getsize(day_path) // 32 >= 100:
            path = day_path
        elif os.path.exists(bak_path):
            path = bak_path
        elif os.path.exists(day_path):
            path = day_path
        else:
            return None
    elif code.startswith('sz'):
        path = f"{TDX_SZ}/{code}.day"
    elif code.startswith('6'):
        day_path = f"{TDX_SH}/sh{code}.day"
        bak_path = f"{TDX_SH}/sh{code}.day.bak"
        if os.path.exists(day_path) and os.path.getsize(day_path) // 32 >= 100:
            path = day_path
        elif os.path.exists(bak_path):
            path = bak_path
        elif os.path.exists(day_path):
            path = day_path
        else:
            return None
    else:
        return None

    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read()
        n = len(data) // 32
        if n < 60:
            return None
        probe = struct.unpack_from('<I', data, 4)[0]
        use_float = not (50 < probe < 10000000)
        if use_float:
            arr = np.zeros((n, 8), dtype=object)
            for i in range(n):
                off = i * 32
                row = struct.unpack_from('<I4f f I I', data, off)
                arr[i] = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]]
            dates = arr[:, 0].astype(int).astype(str)
            df = pd.DataFrame({
                'datetime': pd.to_datetime(dates, format='%Y%m%d'),
                'open': arr[:, 1].astype(float),
                'high': arr[:, 2].astype(float),
                'low': arr[:, 3].astype(float),
                'close': arr[:, 4].astype(float),
                'volume': arr[:, 6].astype(int),
            })
        else:
            arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
            dates = arr[:, 0].astype(str)
            df = pd.DataFrame({
                'datetime': pd.to_datetime(dates, format='%Y%m%d'),
                'open': arr[:, 1] / 100.0,
                'high': arr[:, 2] / 100.0,
                'low': arr[:, 3] / 100.0,
                'close': arr[:, 4] / 100.0,
                'volume': arr[:, 6].astype(np.int64),
            })
        df = df[df['volume'] > 0].reset_index(drop=True)
        if len(df) < 60:
            return None
        return df
    except:
        return None

# ============ 大盘环境 ============
INDEX_CODE = 'sh000001'

def compute_market_env(df_index, on_date):
    """计算on_date时的大盘环境"""
    df = df_index[df_index['datetime'] <= on_date].tail(30)
    if len(df) < 20:
        return 'trend'
    close = df['close'].values
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    price = close[-1]
    ma5_c, ma10_c = ma5[-1], ma10[-1]
    if ma5_c > ma10_c and price > ma5_c:
        return 'up'
    elif ma5_c < ma10_c and price < ma5_c:
        return 'down'
    return 'trend'

# ============ 仓位/SL/TP表 ============
POS = {
    'up':    {'1buy': 0.20, '2buy': 0.40, '3buy': 0.50},
    'trend': {'1buy': 0.10, '2buy': 0.30, '3buy': 0.40},
    'down':  {'1buy': 0.10, '2buy': 0.00, '3buy': 0.00},
}
SL = {
    'up':    {'1buy': 0.06, '2buy': 0.03, '3buy': 0.03},
    'trend': {'1buy': 0.06, '2buy': 0.03, '3buy': 0.03},
    'down':  {'1buy': 0.06, '2buy': 0.00, '3buy': 0.00},
}
TP = {
    'up':    [0.03, 0.08, 0.15],
    'trend': [0.03, 0.06, 0.10],
    'down':  [0.03, 0.05, 0.08],
}

COMMISSION = 0.0003
SLIPPAGE = 0.001

# ============ 单股信号提取 ============
def get_buy_signals(code, lookback=None):
    """用CC15引擎提取买点信号"""
    df = load_day_data(code)
    if df is None:
        return []
    if lookback and len(df) > lookback:
        df = df.tail(lookback).reset_index(drop=True)

    close_s = df['close'].astype(float)
    high_s = df['high'].astype(float)
    low_s = df['low'].astype(float)
    vol_s = df['volume'].astype(float)
    n = len(df)

    # MACD
    ema12 = close_s.ewm(span=12, adjust=False).mean()
    ema26 = close_s.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2

    engine = SignalEngine()
    buy_bi, _, filtered_fractals, strokes_raw = engine._detect_bi_deterministic(df)
    buy_div, _, _, _ = engine._compute_area_divergence(strokes_raw, macd_hist, n)
    buy_2 = engine._detect_2buy(strokes_raw, buy_div, n)
    third_buy = engine._detect_3buy_context(filtered_fractals, df)
    pivot_list = engine._detect_pivots(strokes_raw)

    buy_bi_set = {i for i in range(n) if buy_bi.iloc[i]}
    one_buy_set = buy_bi_set & buy_div

    # 3买是bi_buy的子集
    third_buy_set = {i for i in range(n) if third_buy.iloc[i]}

    signals = []

    # 1买(排除3买)
    for idx in (one_buy_set - third_buy_set):
        if 0 <= idx < n:
            entry_price = float(low_s.iloc[idx])
            sig_date = df.iloc[idx]['datetime']
            signals.append({
                'type': '1buy', 'idx': idx, 'date': sig_date,
                'price': entry_price, 'stop_loss': entry_price * 0.94
            })

    # 3买
    for idx in third_buy_set:
        if 0 <= idx < n:
            entry_price = float(low_s.iloc[idx])
            sig_date = df.iloc[idx]['datetime']
            signals.append({
                'type': '3buy', 'idx': idx, 'date': sig_date,
                'price': entry_price, 'stop_loss': entry_price * 0.97
            })

    # 2买
    for ds in strokes_raw:
        if ds['start_type'] == 'top' and ds['end_type'] == 'bottom':
            sig_1buy = ds['end_idx'] + engine.bi_confirm_delay
            if sig_1buy not in buy_div:
                continue
            low_1buy = ds['end_val']
            bounce = next((us for us in strokes_raw
                          if us['start_type'] == 'bottom' and us['end_type'] == 'top'
                          and us['start_idx'] > ds['end_idx']), None)
            if not bounce:
                continue
            pullback = next((ds2 for ds2 in strokes_raw
                            if ds2['start_type'] == 'top' and ds2['end_type'] == 'bottom'
                            and ds2['start_idx'] > bounce['end_idx']), None)
            if not pullback:
                continue
            sig_2buy = pullback['end_idx'] + engine.bi_confirm_delay
            if sig_2buy >= n:
                continue
            entry_price = float(low_s.iloc[sig_2buy])
            sig_date = df.iloc[sig_2buy]['datetime']
            signals.append({
                'type': '2buy', 'idx': sig_2buy, 'date': sig_date,
                'price': entry_price, 'stop_loss': entry_price * 0.97
            })

    return signals

# ============ 交易模拟 ============
def simulate_trade(df, entry_idx, buy_type, market_env, direction=1):
    """模拟单笔交易"""
    entry_price = df.iloc[entry_idx]['close']
    sl_pct = SL.get(market_env, {}).get(buy_type, 0.06)
    pos = POS.get(market_env, {}).get(buy_type, 0.10)
    t1, t2, t3 = TP.get(market_env, [0.03, 0.08, 0.15])

    if pos <= 0 or sl_pct <= 0:
        return None

    stop_loss = entry_price * (1 - sl_pct)
    max_price = entry_price
    exit_price = None
    exit_reason = None

    for i in range(entry_idx + 1, min(entry_idx + 60, len(df))):
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        close = df.iloc[i]['close']
        max_price = max(max_price, high)

        if low <= stop_loss:
            exit_price = stop_loss
            exit_reason = 'SL'
            break

        pnl_pct = (max_price - entry_price) / entry_price
        if pnl_pct >= t3 and close < max_price * 0.99:
            exit_price = max_price * 0.99
            exit_reason = 'TP3'
            break
        elif pnl_pct >= t2 and close < max_price * 0.98:
            exit_price = max_price * 0.98
            exit_reason = 'TP2'
            break
        elif pnl_pct >= t1 and close < max_price * 0.97:
            exit_price = max_price * 0.97
            exit_reason = 'TP1'
            break

    if exit_price is None:
        exit_price = df.iloc[min(entry_idx + 60, len(df) - 1)]['close']
        exit_reason = 'HOLD60'

    net = ((exit_price - entry_price) / entry_price * direction) - COMMISSION - SLIPPAGE
    return {
        'exit_price': exit_price, 'exit_reason': exit_reason,
        'hold_days': min(60, len(df) - entry_idx - 1),
        'pos': pos, 'net_pnl_pct': net * 100,
        'real_pnl_pct': net * pos * 100,
        'market_env': market_env, 'buy_type': buy_type,
    }

# ============ 汇总统计 ============
def summarize(df_res, label):
    if len(df_res) == 0:
        print(f"\n{label}: 无数据")
        return
    wr = (df_res['net_pnl_pct'] > 0).mean() * 100
    avg = df_res['net_pnl_pct'].mean()
    std = df_res['net_pnl_pct'].std()
    avg_h = df_res['hold_days'].mean()
    sharpe = avg / std * np.sqrt(252 / avg_h) if std > 0 else 0
    max_dd = df_res['net_pnl_pct'].min()
    neg = df_res[df_res['net_pnl_pct'] < 0]['net_pnl_pct']
    pl = avg / abs(neg.mean()) if len(neg) > 0 and neg.mean() != 0 else 0
    print(f"\n{'='*50}")
    print(f"{label} ({len(df_res)} 笔)")
    print(f"{'='*50}")
    print(f"  胜率:    {wr:.1f}%")
    print(f"  均盈:    {avg:.2f}%")
    print(f"  Sharpe:  {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  均持仓:  {avg_h:.1f} 天")
    print(f"  盈亏比:  {pl:.2f}")
    for bt in ['1buy', '2buy', '3buy']:
        sub = df_res[df_res['buy_type'] == bt]
        if len(sub) > 0:
            wr_b = (sub['net_pnl_pct'] > 0).mean() * 100
            av_b = sub['net_pnl_pct'].mean()
            print(f"  {bt}: n={len(sub)}, WR={wr_b:.1f}%, 均盈={av_b:.2f}%")
    if 'v2' in label:
        for env in ['up', 'trend', 'down']:
            sub = df_res[df_res['market_env'] == env]
            if len(sub) > 0:
                wr_e = (sub['net_pnl_pct'] > 0).mean() * 100
                av_e = sub['net_pnl_pct'].mean()
                pos_e = sub['pos'].mean()
                print(f"  {env}: n={len(sub)}, WR={wr_e:.1f}%, 均盈={av_e:.2f}%, 均仓={pos_e:.0%}")

# ============ 主程序 ============
def main():
    print("=" * 60)
    print("v2.0 过滤器回测 (CC15引擎 + 大盘分层 + 仓位分级)")
    print("=" * 60)

    # 加载大盘
    print("\n[1/4] 加载大盘数据...")
    df_index = load_day_data(INDEX_CODE)
    if df_index is None:
        print("ERROR: 无法加载大盘数据")
        return
    print(f"  大盘: {len(df_index)} 条 {df_index['datetime'].min().date()} ~ {df_index['datetime'].max().date()}")

    # 收集股票列表
    print("\n[2/4] 扫描股票列表...")
    sh_files = [f.replace(".day","") for f in os.listdir(TDX_SH)
                if f.endswith('.day') and 'bak' not in f]
    sz_files = [f.replace(".day","") for f in os.listdir(TDX_SZ)
                if f.endswith('.day') and 'bak' not in f]
    all_codes = sh_files + sz_files
    valid = [c for c in all_codes
             if not any(x in c for x in ['ST', 'st', '退', 'BJ'])
             and not c.endswith('sh000001')
             and len(c) == 8]
    print(f"  有效股票: {len(valid)} 只 (sh={len(sh_files)}, sz={len(sz_files)})")

    # 采样
    random.seed(42)
    sample = random.sample(valid, min(200, len(valid)))
    print(f"  采样: {len(sample)} 只")

    # 回测
    print(f"\n[3/4] 执行回测...")
    all_results = []

    for i, code in enumerate(sample):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(sample)}")
        df = load_day_data(code)
        if df is None or len(df) < 60:
            continue

        signals = get_buy_signals(code)
        if not signals:
            continue

        for sig in signals:
            entry_idx = sig['idx']
            entry_date = sig['date']
            buy_type = sig['type']
            market_env = compute_market_env(df_index, entry_date)

            # 基准: 固定10%仓位
            res_b = simulate_trade(df, entry_idx, buy_type, 'trend', direction=1)
            if res_b:
                res_b['code'] = code
                res_b['strategy'] = 'baseline'
                res_b['entry_price'] = sig['price']
                all_results.append(res_b)

            # v2.0: 大盘分层+仓位分级
            res_v2 = simulate_trade(df, entry_idx, buy_type, market_env, direction=1)
            if res_v2:
                res_v2['code'] = code
                res_v2['strategy'] = 'v2'
                res_v2['entry_price'] = sig['price']
                all_results.append(res_v2)

    df_all = pd.DataFrame(all_results)
    print(f"  总交易笔数: {len(df_all)}")

    if len(df_all) == 0:
        print("  无交易信号!")
        return

    df_b = df_all[df_all['strategy'] == 'baseline']
    df_v = df_all[df_all['strategy'] == 'v2']

    summarize(df_b, "基准策略(固定10%仓位)")
    summarize(df_v, "v2.0策略(大盘分层+仓位分级)")

    # 对比
    if len(df_b) > 0 and len(df_v) > 0:
        print("\n" + "=" * 50)
        print("基准 vs v2.0 对比")
        print("=" * 50)
        wr_b = (df_b['net_pnl_pct'] > 0).mean() * 100
        wr_v = (df_v['net_pnl_pct'] > 0).mean() * 100
        avg_b = df_b['net_pnl_pct'].mean()
        avg_v = df_v['net_pnl_pct'].mean()
        sharpe_b = avg_b / df_b['net_pnl_pct'].std() * np.sqrt(252 / df_b['hold_days'].mean()) if df_b['net_pnl_pct'].std() > 0 else 0
        sharpe_v = avg_v / df_v['net_pnl_pct'].std() * np.sqrt(252 / df_v['hold_days'].mean()) if df_v['net_pnl_pct'].std() > 0 else 0
        print(f"  胜率:   基准={wr_b:.1f}%  v2={wr_v:.1f}%  差={wr_v-wr_b:+.1f}%")
        print(f"  均盈:   基准={avg_b:.2f}%  v2={avg_v:.2f}%  差={avg_v-avg_b:+.2f}%")
        print(f"  Sharpe: 基准={sharpe_b:.2f}  v2={sharpe_v:.2f}")
        print(f"  样本量: 基准={len(df_b)}  v2={len(df_v)}")

    # 保存
    out = f"{OUTPUT_DIR}/v2bt_cc15.pkl"
    df_all.to_pickle(out)
    print(f"\n已保存: {out}")

if __name__ == '__main__':
    main()
