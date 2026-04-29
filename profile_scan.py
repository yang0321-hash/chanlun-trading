#!/usr/bin/env python3
"""扫描器性能分析：定位瓶颈"""
import sys, os, time
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_system')
os.chdir('/workspace')

from chanlun_system.scanner_new_framework import (
    load_data, _get_index_status, _get_market_env,
    _get_industry_map, _get_sector_chg
)

print("=== 阶段1: 加载数据 ===")
t = time.time()
dm = load_data()
print(f"  load_data: {time.time()-t:.1f}s, {len(dm)} 只")

print("\n=== 阶段2: 大盘状态 ===")
t = time.time()
idx = _get_index_status()
env = _get_market_env(idx)
print(f"  _get_index_status: {time.time()-t:.1f}s")

print("\n=== 阶段3: 行业映射 ===")
t = time.time()
ind_map = _get_industry_map()
print(f"  _get_industry_map: {time.time()-t:.1f}s, {len(ind_map)} 只")

print("\n=== 阶段4: 腾讯行情 ===")
t = time.time()
codes = list(dm.keys())
chg_data = _get_sector_chg(codes, ind_map)
print(f"  _get_sector_chg: {time.time()-t:.1f}s, {len(chg_data.get('stock_chg', {}))} 只获取成功")

print("\n=== 阶段5: 信号生成(预估) ===")
print(f"  7948只股票, 多进程扫描")
print(f"  预估: {len(codes)}只 / (10-15只/秒) ≈ {len(codes)/12:.0f}s")
print(f"\n各阶段合计不含信号生成: {0:.0f}s (需运行generate_signals才有完整数据)")
