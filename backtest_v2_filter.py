#!/usr/bin/env python3
"""
v2.0三层过滤回测
- 对scanner_new_fw_signals_live.pkl中的历史信号做逐层过滤回测
- 验证每层过滤对Sharpe/胜率/均盈的提升效果
"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import struct

# ========== 数据加载 ==========

def load_tdx_fast(code, mkt):
    """快速加载单只股票TDX日线数据"""
    short = code.lower().replace('.sz','').replace('.sh','')
    fpath = Path(f'/workspace/tdx_data/{mkt}/lday/{short}.day')
    if not fpath.exists():
        return None
    data = fpath.read_bytes()
    n = len(data) // 32
    arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
    first_price_raw = float(arr[0, 1])
    if first_price_raw > 10_000_000:
        prices = np.frombuffer(arr[:, 1:5].tobytes(), dtype=np.float32).reshape(n, 4)
    else:
        prices = arr[:, 1:5] / 100.0
    volumes = arr[:, 6].astype(np.int64)
    dates = pd.to_datetime(arr[:, 0].astype(str), format='%Y%m%d')
    return pd.DataFrame({
        'open': prices[:, 0], 'high': prices[:, 1],
        'low': prices[:, 2], 'close': prices[:, 3],
        'volume': volumes
    }, index=dates).sort_index()

def load_index_data():
    """加载上证指数历史数据用于大盘环境判断"""
    return load_tdx_fast('sh000001', 'sh')

def compute_market_env(idx_df, entry_idx):
    """计算entry_idx时刻的大盘环境"""
    if entry_idx < 20:
        return 'unknown'
    closes = idx_df['close'].iloc[max(0, entry_idx-30):entry_idx+1].values
    if len(closes) < 20:
        return 'unknown'
    ma5 = closes[-5:].mean()
    ma10 = closes[-10:].mean()
    ma20 = closes[-20:].mean()
    latest = closes[-1]
    above_ma20 = latest > ma20
    ma5_above_ma10 = ma5 > ma10
    if above_ma20 and ma5_above_ma10:
        return 'uptrend'
    elif not above_ma20 and not ma5_above_ma10:
        return 'downtrend'
    else:
        return 'consolidation'

def bt_signal(df, entry_idx, sl_price, pos_pct, n=20):
    """单笔交易回测，返回PnL%"""
    if entry_idx + 1 >= len(df):
        return None
    entry = float(df['close'].iloc[entry_idx + 1])
    for d in range(entry_idx + 1, min(entry_idx + 1 + n, len(df))):
        lo = float(df['low'].iloc[d])
        cl = float(df['close'].iloc[d])
        pct = (cl - entry) / entry * 100
        if lo <= sl_price:
            return (sl_price - entry) / entry * 100
        if pct >= 5:
            return pct
        if pct >= 3:
            return pct
    # 未触发出局，尾盘收益
    cl = float(df['close'].iloc[-1])
    return (cl - entry) / entry * 100

def run_backtest():
    # 加载信号
    sig_df = pd.read_pickle('/workspace/scanner_new_fw_signals_live.pkl')
    print(f"信号总数: {len(sig_df)}")
    
    # 加载大盘指数数据
    idx_df = load_index_data()
    print(f"大盘数据: {len(idx_df)} bars ({idx_df.index[0].date()} ~ {idx_df.index[-1].date()})")
    
    results = []
    skipped = 0
    
    for _, sig in sig_df.iterrows():
        code = sig['code']
        entry_idx = int(sig['entry_idx'])
        sl_price = float(sig['sl_price'])
        pos_pct = float(sig['pos_pct'])
        fib = sig['fib_strength']
        mkt = 'sz' if code.startswith('SZ') else 'sh'
        
        df = load_tdx_fast(code, mkt)
        if df is None or entry_idx + 1 >= len(df):
            skipped += 1
            continue
        
        pnl = bt_signal(df, entry_idx, sl_price, pos_pct)
        if pnl is None:
            skipped += 1
            continue
        
        # 调整仓位乘数
        adj_pnl = pnl * (pos_pct / 0.3)
        
        # 计算大盘环境
        market_env = compute_market_env(idx_df, entry_idx)
        
        # 计算信号日涨跌幅（entry_idx那天的涨跌幅）
        chg = 0.0
        if entry_idx >= 1:
            prev_close = float(df['close'].iloc[entry_idx - 1])
            entry_close = float(df['close'].iloc[entry_idx])
            if prev_close > 0:
                chg = (entry_close - prev_close) / prev_close * 100
        
        results.append({
            'code': code,
            'date': sig['date'],
            'fib': fib,
            'market_env': market_env,
            'stock_chg': round(chg, 2),
            'sl_price': sl_price,
            'pos_pct': pos_pct,
            'pnl': pnl,
            'adj_pnl': adj_pnl,
            'type': sig['type'],
        })
    
    print(f"成功回测: {len(results)}/{len(sig_df)} (跳过{skipped})")
    rd = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("各层过滤效果对比")
    print("="*60)
    
    # Layer 0: 全部信号
    def stats(sub, label):
        if len(sub) == 0:
            print(f"{label}: 无数据")
            return
        wr = (sub['adj_pnl'] > 0).mean() * 100
        avg = sub['adj_pnl'].mean()
        std = sub['adj_pnl'].std()
        sh = avg / std * np.sqrt(252 / 15) if std > 0 else 0
        max_dd = (sub['adj_pnl'].cumsum().cummax() - sub['adj_pnl'].cumsum()).max()
        print(f"{label}: n={len(sub)} WR={wr:.1f}% 均盈={avg:.3f}% Sharpe={sh:.2f} 最大回撤={max_dd:.2f}%")
    
    all_sub = rd
    fib_sub = rd[rd['fib'].isin(['strong', 'strongest'])]
    mkt_sub = rd[(rd['fib'].isin(['strong', 'strongest'])) & (rd['market_env'] != 'unknown')]
    full_sub = rd[
        (rd['fib'].isin(['strong', 'strongest'])) &
        (rd['market_env'] != 'unknown') &
        (rd['stock_chg'] >= -2.0)
    ]
    
    print("\n--- 逐层叠加效果 ---")
    stats(all_sub, "L0 全部信号")
    stats(fib_sub, "L1 Fib(strong+strongest)")
    stats(mkt_sub, "L2 + 大盘环境过滤")
    stats(full_sub, "L3 + 个股跌幅过滤(>-2%)")
    
    print("\n--- 大盘环境分组 ---")
    for env in ['uptrend', 'consolidation', 'downtrend', 'unknown']:
        sub = rd[rd['market_env'] == env]
        if len(sub) == 0:
            continue
        wr = (sub['adj_pnl'] > 0).mean() * 100
        avg = sub['adj_pnl'].mean()
        std = sub['adj_pnl'].std()
        sh = avg / std * np.sqrt(252 / 15) if std > 0 else 0
        print(f"  {env}: n={len(sub)} WR={wr:.1f}% 均盈={avg:.3f}% Sharpe={sh:.2f}")
    
    print("\n--- Fib分组 ---")
    for fib in ['strongest', 'strong', 'medium', 'weak']:
        sub = rd[rd['fib'] == fib]
        if len(sub) == 0:
            continue
        wr = (sub['adj_pnl'] > 0).mean() * 100
        avg = sub['adj_pnl'].mean()
        std = sub['adj_pnl'].std()
        sh = avg / std * np.sqrt(252 / 15) if std > 0 else 0
        print(f"  {fib}: n={len(sub)} WR={wr:.1f}% 均盈={avg:.3f}% Sharpe={sh:.2f}")
    
    print("\n--- 个股跌幅分组 ---")
    bins = [-100, -2, 0, 2, 100]
    labels = ['<-2%', '-2%~0%', '0%~2%', '>2%']
    rd['chg_bin'] = pd.cut(rd['stock_chg'], bins=bins, labels=labels)
    for bin_lbl in labels:
        sub = rd[rd['chg_bin'] == bin_lbl]
        if len(sub) == 0:
            continue
        wr = (sub['adj_pnl'] > 0).mean() * 100
        avg = sub['adj_pnl'].mean()
        std = sub['adj_pnl'].std()
        sh = avg / std * np.sqrt(252 / 15) if std > 0 else 0
        print(f"  {bin_lbl}: n={len(sub)} WR={wr:.1f}% 均盈={avg:.3f}% Sharpe={sh:.2f}")
    
    print("\n--- 市场环境 x Fib 交叉 ---")
    for env in ['uptrend', 'consolidation', 'downtrend']:
        for fib in ['strongest', 'strong']:
            sub = rd[(rd['market_env'] == env) & (rd['fib'] == fib)]
            if len(sub) == 0:
                continue
            wr = (sub['adj_pnl'] > 0).mean() * 100
            avg = sub['adj_pnl'].mean()
            std = sub['adj_pnl'].std()
            sh = avg / std * np.sqrt(252 / 15) if std > 0 else 0
            print(f"  {env}+{fib}: n={len(sub)} WR={wr:.1f}% 均盈={avg:.3f}% Sharpe={sh:.2f}")
    
    # 结论
    print("\n" + "="*60)
    print("回测结论")
    print("="*60)
    l0 = all_sub
    l1 = fib_sub
    l2 = mkt_sub
    l3 = full_sub
    for name, sub in [("全部", l0), ("L1(Fib)", l1), ("L2(+大盘)", l2), ("L3(+跌幅)", l3)]:
        if len(sub) == 0:
            continue
        wr = (sub['adj_pnl'] > 0).mean() * 100
        avg = sub['adj_pnl'].mean()
        std = sub['adj_pnl'].std()
        sh = avg / std * np.sqrt(252 / 15) if std > 0 else 0
        print(f"{name}: n={len(sub)} WR={wr:.1f}% 均盈={avg:.3f}% Sharpe={sh:.2f}")

if __name__ == '__main__':
    run_backtest()
