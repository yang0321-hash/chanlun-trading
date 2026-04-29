"""
v2.0 过滤器滚动回测 (2024-07 ~ 2026-04)
=========================================
对比基准(无过滤) vs v2.0(大盘分层+仓位分级+板块过滤)

核心改进 vs backtest_v2_filters.py:
  1. 大盘环境: 逐信号按entry_date计算 (非固定末期值)
  2. 信号源:   backtest_new_fw_signals.pkl (50606条缠论信号)
  3. 板块:    用上证/深证指数近似(全市场代理)
  4. 回测:    滚动窗口 + 基准对比

过滤器逻辑 (v2.0):
  - 大盘分层: up/trend/down → 仓位上限
  - 2买板块否决: 板块跌幅>2% → 排除
  - 1买快进快出: 下跌笔中≤30%仓位
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============ 常量 ============
SCANNER_CACHE = '/workspace/scanner_new_fw_cache_120.pkl'
SIGNAL_CACHE = '/workspace/backtest_new_fw_signals.pkl'
INDEX_SH = '/workspace/sh000001.csv'    # 上证指数 CSV
INDEX_SZ = '/workspace/sz000001.csv'    # 深证成指 CSV
OUTPUT_DIR = '/workspace/backtest_results/v2_rolling'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMISSION = 0.0003        # 0.03%
SLIPPAGE = 0.001           # 0.1%

# v2.0 仓位分级
POSITION_TABLE = {
    'up':    {'1buy': 0.20, '2buy': 0.40, '3buy': 0.50},
    'trend': {'1buy': 0.10, '2buy': 0.30, '3buy': 0.40},
    'down':  {'1buy': 0.10, '2buy': 0.00, '3buy': 0.00},
}

# v2.0 止损 (强势/弱势)
SL_TABLE_ENV = {
    'up':    {'1buy': 0.06, '2buy': 0.03, '3buy': 0.03},
    'trend': {'1buy': 0.06, '2buy': 0.03, '3buy': 0.03},
    'down':  {'1buy': 0.05, '2buy': 0.03, '3buy': 0.03},  # 2buy: 0.00→0.03, 保留使用信号自带SL
}

# v2.0 动态止盈阶梯
TP_TABLE = {
    'up':    {'t1': 0.03, 't2': 0.08, 't3': 0.15},
    'trend': {'t1': 0.03, 't2': 0.06, 't3': 0.10},
    'down':  {'1buy': 0.03, '2buy': 0.05, '3buy': 0.08},  # 1buy only
}

# v2.0 大盘环境判断参数
LOOKBACK_MA = 20   # 判断是否在20日线上方


# ============ 数据加载 ============

def load_index_csv(path) -> pd.DataFrame:
    """加载指数 CSV (date=int YYYYMMDD)"""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except:
        return None


def signal_to_cache_key(code):
    """信号code (300461.SZ) → cache key (SZ300461.SZ)"""
    if '.' in code:
        num, exch = code.split('.')
        return f"{exch}{num}.{exch}"
    return code


def load_stock_from_cache(cache_dict, code) -> pd.DataFrame:
    """从 scanner cache dict 加载个股 DataFrame"""
    cache_key = signal_to_cache_key(code)
    if cache_key not in cache_dict:
        return None
    df = cache_dict[cache_key]
    if not isinstance(df, pd.DataFrame) or len(df) < 60:
        return None
    # cache DataFrame: date is in the index (DatetimeIndex), columns: open/high/low/close/volume
    df = df.copy()
    df = df.reset_index()  # 将 index (date) 变成 'date' 列
    df = df.rename(columns={'index': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


# ============ 大盘环境计算 ============

def compute_market_env_at_date(df_index: pd.DataFrame, target_date: pd.Timestamp) -> str:
    """
    在 target_date 当日计算大盘环境
    使用MA5/MA10/价格在MA5上方三重判断
    返回: 'up' | 'trend' | 'down'
    """
    if df_index is None or len(df_index) < 25:
        return 'trend'

    # 找到 target_date 对应的索引 (或最近的前一个交易日)
    idxs = df_index[df_index['date'] <= target_date].index
    if len(idxs) < 25:
        return 'trend'
    i = idxs[-1]

    close = df_index['close'].values
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values

    price = close[i]
    m5, m10, m20 = ma5[i], ma10[i], ma20[i]

    if m5 > m10 and price > m5:
        return 'up'
    elif m5 < m10 and price < m5:
        return 'down'
    else:
        return 'trend'


def compute_index_change_at_date(df_index: pd.DataFrame, target_date: pd.Timestamp, lookback=1) -> float:
    """计算大盘当日涨幅 (近似: 用收盘价变化)"""
    if df_index is None or len(df_index) < 2:
        return 0.0
    idxs = df_index[df_index['date'] <= target_date].index
    if len(idxs) < lookback + 1:
        return 0.0
    i = idxs[-1]
    j = i - lookback
    if j < 0:
        return 0.0
    change = (df_index['close'].values[i] - df_index['close'].values[j]) / df_index['close'].values[j]
    return change


# ============ 板块过滤 (简化代理) ============

def get_sector_proxy_change(code: str, df_sh: pd.DataFrame, df_sz: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """
    板块过滤代理:
    - 主板(6/9开头): 用上证指数
    - 中小(0/2/3开头): 用深证成指
    返回: 当日该"板块"相对前一日变化率
    """
    df = df_sh if (code.startswith('6') or code.startswith('9')) else df_sz
    if df is None:
        return 0.0

    # 找 target_date 对应的索引
    idxs = df[df['date'] <= target_date].index
    if len(idxs) < 2:
        return 0.0
    i = idxs[-1]
    if i <= 0:
        return 0.0
    chg = (df['close'].values[i] - df['close'].values[i-1]) / df['close'].values[i-1]
    return chg


# ============ 交易模拟 ============

def simulate_trade(entry_date, entry_price, buy_type, market_env,
                   sl_price_from_signal, df_stock: pd.DataFrame,
                   sector_change: float,
                   use_v2_filters: bool) -> dict:
    """
    模拟单笔交易

    use_v2_filters=True:  应用v2.0全部过滤
    use_v2_filters=False:  基准(无过滤,固定仓位)
    """
    # v2 过滤: 归一化买点类型 (2plus3buy → 3buy)
    norm_type = '3buy' if buy_type in ('3buy', '2plus3buy') else buy_type

    # --- v2.0 过滤 ---
    if use_v2_filters:
        # 仓位分级: 下跌笔只做1买轻仓快出, 2buy保留(胜率数据支持), 3buy全过滤
        if market_env == 'down':
            if norm_type in ('2buy',):
                # 下跌笔的2buy保留但降仓
                pos_pct = 0.20
            else:
                # 1buy轻仓 / 3buy禁止
                pos_pct = POSITION_TABLE.get(market_env, {}).get(norm_type, 0)
        else:
            pos_pct = POSITION_TABLE.get(market_env, {}).get(norm_type, 0)

        if pos_pct <= 0:
            return None

        # 板块否决: 2买/3买在板块跌幅>2%时过滤
        if norm_type in ('2buy', '3buy') and sector_change < -0.02:
            return None
    else:
        # 基准: 固定30%仓位, 无板块过滤
        pos_pct = 0.30

    # --- 止损设置 ---
    if use_v2_filters:
        sl_pct = SL_TABLE_ENV.get(market_env, {}).get(norm_type, 0.06)
    else:
        sl_pct = 0.06   # 基准固定6%

    if sl_pct <= 0:
        return None

    # 优先用信号自带SL(2buy用), 3buy统一用%计算
    # NOTE: backtest_new_fw_signals.pkl 的 3buy 信号自带SL≈entry_price(0.04%距离)，无法使用
    # 对 3buy 强制使用 3% 止损以获得合理比较
    if norm_type == '3buy':
        # 强制使用3%止损(信号自带SL=入场价，不可用于3buy)
        stop_loss = entry_price * (1 - 0.03)
    elif sl_price_from_signal and sl_price_from_signal > 0 and sl_price_from_signal < entry_price:
        # 对2buy使用信号自带SL
        sl_from_signal_pct = (entry_price - sl_price_from_signal) / entry_price
        # 仅当信号SL距离>2%时才使用(避免SL过近)
        if sl_from_signal_pct >= 0.02:
            stop_loss = sl_price_from_signal
        else:
            stop_loss = entry_price * (1 - sl_pct)
    else:
        stop_loss = entry_price * (1 - sl_pct)

    # --- 止盈参数 ---
    if use_v2_filters:
        tp_pcts = TP_TABLE.get(market_env, TP_TABLE['trend'])
        if norm_type == '1buy':
            t1 = tp_pcts.get('1buy', 0.03)
            t2 = tp_pcts.get('2buy', 0.05)
            t3 = tp_pcts.get('3buy', 0.08)
        else:
            t1 = tp_pcts.get('t1', 0.03)
            t2 = tp_pcts.get('t2', 0.06)
            t3 = tp_pcts.get('t3', 0.10)
    else:
        t1, t2, t3 = 0.03, 0.08, 0.15   # 基准统一

    # --- 找到入场日索引 ---
    stock_idxs = df_stock[df_stock['date'] >= entry_date].index
    if len(stock_idxs) < 2:
        return None
    start_i = stock_idxs[0]
    if start_i + 1 >= len(df_stock):
        return None

    # --- 逐日模拟 ---
    max_price = entry_price
    exit_price = None
    exit_reason = None
    hold_days = 0

    for i in range(start_i + 1, min(start_i + 60, len(df_stock))):
        high = df_stock.iloc[i]['high']
        low = df_stock.iloc[i]['low']
        close = df_stock.iloc[i]['close']
        max_price = max(max_price, high)

        # 止损
        if low <= stop_loss:
            exit_price = stop_loss
            exit_reason = 'SL'
            hold_days = i - start_i
            break

        # 动态止盈
        pnl_pct = (max_price - entry_price) / entry_price
        if pnl_pct >= t3 and close < max_price * 0.99:
            exit_price = max_price * 0.99
            exit_reason = 'TP3'
            hold_days = i - start_i
            break
        elif pnl_pct >= t2 and close < max_price * 0.98:
            exit_price = max_price * 0.98
            exit_reason = 'TP2'
            hold_days = i - start_i
            break
        elif pnl_pct >= t1 and close < max_price * 0.97:
            exit_price = max_price * 0.97
            exit_reason = 'TP1'
            hold_days = i - start_i
            break

    if exit_price is None:
        # 持有60天强制平仓
        exit_price = df_stock.iloc[min(start_i + 60, len(df_stock) - 1)]['close']
        exit_reason = 'HOLD60'
        hold_days = 60

    # 净收益率
    pnl_pct = (exit_price - entry_price) / entry_price
    net_pnl = pnl_pct - COMMISSION - SLIPPAGE
    real_pnl = net_pnl * pos_pct   # 按实际仓位

    return {
        'entry_date': entry_date,
        'code': df_stock.iloc[start_i]['date'] if False else None,
        'buy_type': buy_type,
        'market_env': market_env,
        'sector_change': sector_change,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_days': hold_days,
        'pos': pos_pct,
        'gross_pnl_pct': pnl_pct * 100,
        'net_pnl_pct': net_pnl * 100,
        'real_pnl_pct': real_pnl * 100,
        'filtered': use_v2_filters,
    }


# ============ 主回测循环 ============

def run_backtest():
    print("=" * 60)
    print("v2.0 过滤器滚动回测")
    print("=" * 60)

    # 1. 加载信号
    print("\n[1/6] 加载信号...")
    signals_df = pd.read_pickle(SIGNAL_CACHE)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    print(f"  信号总数: {len(signals_df)}")
    print(f"  日期范围: {signals_df['date'].min().date()} ~ {signals_df['date'].max().date()}")
    print(f"  买点分布:\n{signals_df['type'].value_counts()}")

    # 2. 加载 scanner cache (个股日线)
    print("\n[2/6] 加载个股数据 (scanner cache)...")
    with open(SCANNER_CACHE, 'rb') as f:
        stock_cache = pickle.load(f)
    print(f"  cache 股票数: {len(stock_cache)}")

    # 3. 加载大盘数据
    print("\n[3/6] 加载大盘数据...")
    df_sh = load_index_csv(INDEX_SH)
    df_sz = load_index_csv(INDEX_SZ)
    if df_sh is None:
        print("ERROR: 无法加载上证数据")
        return
    print(f"  上证: {len(df_sh)} 条, {df_sh['date'].min().date()} ~ {df_sh['date'].max().date()}")
    print(f"  深证: {len(df_sz) if df_sz is not None else 0} 条")

    # 4. 预计算每日的市场环境
    print("\n[4/6] 预计算大盘环境序列...")
    close_sh = df_sh['close'].values
    ma5_sh = pd.Series(close_sh).rolling(5).mean().values
    ma10_sh = pd.Series(close_sh).rolling(10).mean().values

    env_series = {}   # date -> env
    change_series = {}  # date -> sector change (sh or sz)

    signal_dates = signals_df['date'].unique()
    for sd in tqdm(signal_dates, desc="  计算市场环境"):
        # 找最近的大盘日期
        candidates = df_sh[df_sh['date'] <= sd]
        if len(candidates) < 25:
            env_series[sd] = 'trend'
            change_series[sd] = 0.0
            continue
        i = candidates.index[-1]
        price = close_sh[i]
        m5, m10 = ma5_sh[i], ma10_sh[i]
        if m5 > m10 and price > m5:
            env = 'up'
        elif m5 < m10 and price < m5:
            env = 'down'
        else:
            env = 'trend'
        env_series[sd] = env

        # 大盘当日变化 (用上证)
        if i > 0:
            chg = (close_sh[i] - close_sh[i-1]) / close_sh[i-1]
        else:
            chg = 0.0
        change_series[sd] = chg

    # 大盘环境分布
    env_counts = pd.Series(list(env_series.values())).value_counts()
    print(f"  大盘环境分布: {env_counts.to_dict()}")

    # 5. 模拟交易
    print("\n[5/6] 模拟交易...")
    results_benchmark = []   # 无过滤基准
    results_v2 = []          # v2.0 过滤

    grouped = signals_df.groupby('code')
    total_codes = len(grouped)

    for code, group in tqdm(grouped, desc="  处理股票", total=total_codes):
        # 从 cache 加载个股数据
        df_stock = load_stock_from_cache(stock_cache, code)
        if df_stock is None or len(df_stock) < 60:
            continue

        for _, sig in group.iterrows():
            entry_date = sig['date']
            entry_price = sig['price']
            buy_type = sig['type']   # '1buy', '2buy', '3buy', '2plus3buy'
            sl_price = sig.get('sl_price', None)

            if entry_price <= 0 or np.isnan(entry_price):
                continue

            # 大盘环境
            market_env = env_series.get(entry_date, 'trend')
            sector_change = change_series.get(entry_date, 0.0)

            # 模拟基准(无过滤)
            r_bench = simulate_trade(
                entry_date, entry_price, buy_type, market_env,
                sl_price, df_stock, sector_change,
                use_v2_filters=False
            )
            if r_bench:
                r_bench['code'] = code
                results_benchmark.append(r_bench)

            # 模拟v2.0(有过滤)
            r_v2 = simulate_trade(
                entry_date, entry_price, buy_type, market_env,
                sl_price, df_stock, sector_change,
                use_v2_filters=True
            )
            if r_v2:
                r_v2['code'] = code
                results_v2.append(r_v2)

    # 6. 统计结果
    print("\n[6/6] 统计结果...")

    df_bench = pd.DataFrame(results_benchmark) if results_benchmark else pd.DataFrame()
    df_v2 = pd.DataFrame(results_v2) if results_v2 else pd.DataFrame()

    def stats(df, label):
        if df is None or len(df) == 0:
            print(f"\n{label}: 无数据")
            return
        print(f"\n{'='*60}")
        print(f"{label} (n={len(df)})")
        print(f"{'='*60}")
        print(f"  总收益率(仓位加权): {df['real_pnl_pct'].sum():.2f}%")
        print(f"  胜率: {(df['net_pnl_pct']>0).mean()*100:.1f}%")
        print(f"  均盈: {df['net_pnl_pct'].mean():.3f}%")
        print(f"  均亏: {df[df['net_pnl_pct']<0]['net_pnl_pct'].mean():.3f}%")
        if df['net_pnl_pct'].std() > 0:
            sharpe = df['net_pnl_pct'].mean() / df['net_pnl_pct'].std() * np.sqrt(252 / df['hold_days'].mean())
            print(f"  Sharpe(简化): {sharpe:.2f}")
        print(f"  最大单笔亏损: {df['net_pnl_pct'].min():.2f}%")
        print(f"  平均持仓: {df['hold_days'].mean():.1f}天")
        print(f"  退出原因分布:\n{df['exit_reason'].value_counts().to_string()}")

        # 按市场环境分组
        print(f"\n  按大盘环境:")
        for env in ['up', 'trend', 'down']:
            sub = df[df['market_env'] == env]
            if len(sub) > 0:
                wr = (sub['net_pnl_pct'] > 0).mean() * 100
                ap = sub['net_pnl_pct'].mean()
                n = len(sub)
                print(f"    {env}: n={n}, WR={wr:.1f}%, 均盈={ap:.3f}%, 仓位={sub['pos'].mean()*100:.0f}%")

        # 按买点类型分组
        print(f"\n  按买点类型:")
        for bt in ['1buy', '2buy', '3buy']:
            sub = df[df['buy_type'] == bt]
            if len(sub) > 0:
                wr = (sub['net_pnl_pct'] > 0).mean() * 100
                ap = sub['net_pnl_pct'].mean()
                n = len(sub)
                print(f"    {bt}: n={n}, WR={wr:.1f}%, 均盈={ap:.3f}%")

    stats(df_bench, "【基准】无过滤, 固定30%仓位")
    stats(df_v2, "【v2.0】大盘分层+板块过滤+仓位分级")

    # ============ 对比汇总 ============
    print(f"\n{'='*60}")
    print("【对比汇总】基准 vs v2.0")
    print(f"{'='*60}")
    if len(df_bench) > 0 and len(df_v2) > 0:
        bench_total = df_bench['real_pnl_pct'].sum()
        v2_total = df_v2['real_pnl_pct'].sum()
        bench_wr = (df_bench['net_pnl_pct']>0).mean()*100
        v2_wr = (df_v2['net_pnl_pct']>0).mean()*100
        bench_avg = df_bench['net_pnl_pct'].mean()
        v2_avg = df_v2['net_pnl_pct'].mean()

        print(f"  信号数: 基准={len(df_bench)}, v2.0={len(df_v2)} (过滤掉={len(df_bench)-len(df_v2)})")
        print(f"  胜率:   基准={bench_wr:.1f}%, v2.0={v2_wr:.1f}%")
        print(f"  均盈:   基准={bench_avg:.3f}%, v2.0={v2_avg:.3f}%")
        print(f"  总收益: 基准={bench_total:.2f}%, v2.0={v2_total:.2f}%")
        print(f"  改善:   {'✓ v2.0更优' if v2_avg > bench_avg else '✗ 基准更优'} (均盈差={v2_avg-bench_avg:+.3f}%)")

    # 保存结果
    output = {
        'benchmark': df_bench.to_dict('records'),
        'v2_filtered': df_v2.to_dict('records'),
        'env_distribution': env_counts.to_dict(),
    }
    out_path = f"{OUTPUT_DIR}/v2_rolling_results.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    run_backtest()
