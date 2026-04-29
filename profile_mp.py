#!/usr/bin/env python3
"""多进程参数优化测试"""
import sys, os, time, concurrent.futures
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_system')
os.chdir('/workspace')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import chanlun_system.scanner_new_framework as nf

# 加载
dm = nf.load_data()
codes = list(dm.keys())
stock_data = []
for code in codes:
    df = dm[code]
    n = len(df)
    if n < 60:
        stock_data.append((code, None, None))
    else:
        arr = df[['open','high','low','close','volume']].values.astype(np.float64)
        stock_data.append((code, arr, df.index))

print(f"{len(stock_data)} 只股票, 准备多进程测试...")
print(f"CPU: {cpu_count()} cores, n_workers={max(1, cpu_count()-1)}")

idx = nf._get_index_status()
menv = nf._get_market_env(idx)
ind_map = nf._get_industry_map()
chg_data = nf._get_sector_chg(codes, ind_map)

# 测试不同chunksize
for cs in [20, 50, 100, 200]:
    t = time.time()
    with Pool(processes=max(1, cpu_count()-1),
              initializer=nf._mp_init_worker,
              initargs=(idx, menv, ind_map, chg_data)) as pool:
        n_signals = 0
        for sigs in pool.imap_unordered(nf._mp_process_stock, stock_data, chunksize=cs):
            n_signals += len(sigs)
    elapsed = time.time() - t
    rate = len(stock_data) / elapsed if elapsed > 0 else 0
    print(f"chunksize={cs:3d}: {elapsed:.1f}s ({rate:.0f}只/秒) {n_signals}信号")

# 快速预筛测试：只用前2000只
print("\n--- 预筛测试 (前2000只) ---")
stock_data_2k = stock_data[:2000]
for cs in [50, 100, 200]:
    t = time.time()
    with Pool(processes=max(1, cpu_count()-1),
              initializer=nf._mp_init_worker,
              initargs=(idx, menv, ind_map, chg_data)) as pool:
        n_signals = 0
        for sigs in pool.imap_unordered(nf._mp_process_stock, stock_data_2k, chunksize=cs):
            n_signals += len(sigs)
    elapsed = time.time() - t
    rate = len(stock_data_2k) / elapsed if elapsed > 0 else 0
    print(f"2000只 chunksize={cs:3d}: {elapsed:.1f}s ({rate:.0f}只/秒) {n_signals}信号")

# TDX并行加载测试
print("\n--- TDX并行加载测试 ---")
def load_one(code_df):
    code, df = code_df
    n = len(df)
    if n < 60: return (code, None, None)
    arr = df[['open','high','low','close','volume']].values.astype(np.float64)
    return (code, arr, df.index)

t = time.time()
seq_count = len([sd for sd in stock_data if sd[1] is not None])
seq_time = time.time() - t
print(f"串行准备: {seq_time:.1f}s for {seq_count}只有效")

t = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    results = list(ex.map(load_one, [(c, dm[c]) for c in codes]))
par_time = time.time() - t
print(f"并行准备: {par_time:.1f}s for {sum(1 for r in results if r[1] is not None)}只有效")
print(f"加速比: {seq_time/par_time:.1f}x" if seq_time > 0 and par_time > 0 else "N/A")
