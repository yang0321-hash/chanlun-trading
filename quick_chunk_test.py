#!/usr/bin/env python3
"""快速chunksize对比：只测1000只股票"""
import sys, os, time
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_system')
os.chdir('/workspace')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import chanlun_system.scanner_new_framework as nf

dm = nf.load_data()
codes = list(dm.keys())[:2000]  # 只测2000只
idx = nf._get_index_status()
menv = nf._get_market_env(idx)
ind_map = nf._get_industry_map()
chg_data = nf._get_sector_chg(codes, ind_map)

stock_data = []
for code in codes:
    df = dm[code]
    n = len(df)
    if n < 60:
        stock_data.append((code, None, None))
    else:
        arr = df[['open','high','low','close','volume']].values.astype(np.float64)
        stock_data.append((code, arr, df.index))

n_w = max(1, cpu_count() - 1)
print(f"{len(stock_data)} 只, {n_w} workers, 测试chunksize...")

for cs in [10, 50, 100, 200]:
    t = time.time()
    n_sig = 0
    with Pool(processes=n_w,
              initializer=nf._mp_init_worker,
              initargs=(idx, menv, ind_map, chg_data)) as pool:
        for sigs in pool.imap_unordered(nf._mp_process_stock, stock_data, chunksize=cs):
            n_sig += len(sigs)
    elapsed = time.time() - t
    rate = len(stock_data) / elapsed
    print(f"chunksize={cs:3d}: {elapsed:.1f}s ({rate:.0f}只/秒) {n_sig}信号")
