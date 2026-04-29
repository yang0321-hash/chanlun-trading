#!/usr/bin/env python3
"""对比 300750 日线笔划分: TDX标准答案 vs core模块"""
import sys, os
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

import pandas as pd
import numpy as np
import json, requests

# ============ 获取日线数据 ============
code = '300750'
session = requests.Session()
session.trust_env = False
url = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sz{code}&scale=240&ma=no&datalen=800'
resp = session.get(url, timeout=15)
resp.encoding = 'utf-8'
try:
    data = json.loads(resp.text)
except:
    data = eval(resp.text)
df = pd.DataFrame(data)
df.rename(columns={'day': 'datetime'}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)
df = df.sort_index()

# ============ TDX 标准答案 ============
tdx_strokes = [
    ('2025-04-07', 203.55, 'bottom'),
    ('2025-05-21', 278.98, 'top'),
    ('2025-06-23', 237.02, 'bottom'),
    ('2025-07-30', 291.19, 'top'),
    ('2025-08-12', 256.71, 'bottom'),  # 原始数据2056.71, 猜测为256.71
    ('2025-10-09', 424.36, 'top'),
]

# ============ 缠论分析 ============
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import KLine, FractalDetector, StrokeGenerator, SegmentGenerator, PivotDetector

kline = KLine.from_dataframe(df, strict_mode=True)
fractal_det = FractalDetector(kline, confirm_required=False)
fractals = fractal_det.get_fractals()
stroke_gen = StrokeGenerator(kline, fractals, min_bars=5)
strokes = stroke_gen.get_strokes()

# 获取处理后K线数据，建立 index → date 映射
processed = kline.processed_data
idx_to_date = {}
for i, kd in enumerate(processed):
    idx_to_date[i] = kd.datetime

# ============ 找2025年4月~10月的笔 ============
print('=' * 70)
print('TDX 标准答案 (笔端点):')
print('=' * 70)
for dt_str, val, tp in tdx_strokes:
    print(f'  {dt_str}  {val:>10.2f}  ({tp})')

print()
print('=' * 70)
print('Core模块检测结果 (2025-03 ~ 2026-04 期间的笔):')
print('=' * 70)

# 找时间范围内的笔
range_start = pd.Timestamp('2025-03-01')
range_end = pd.Timestamp('2026-04-30')

for i, s in enumerate(strokes):
    s_date = idx_to_date.get(s.start_index)
    e_date = idx_to_date.get(s.end_index)
    if s_date is None or e_date is None:
        continue
    if s_date < range_start and e_date < range_start:
        continue
    if s_date > range_end:
        break
    direction = '↑' if s.type.value == 'up' else '↓'
    print(f'  笔{i+1:>2} {direction}  {s_date.strftime("%Y-%m-%d")} {s.start_value:>8.2f}'
          f'  →  {e_date.strftime("%Y-%m-%d")} {s.end_value:>8.2f}'
          f'  H={s.high:.2f} L={s.low:.2f}')

# ============ 逐点对比 ============
print()
print('=' * 70)
print('逐点对比:')
print('=' * 70)

for dt_str, val, tp in tdx_strokes:
    target_dt = pd.Timestamp(dt_str)
    # 在core结果中找最近的笔端点
    best_match = None
    best_dist = 999
    for i, s in enumerate(strokes):
        for endpoint_idx, endpoint_val in [(s.start_index, s.start_value), (s.end_index, s.end_value)]:
            ep_dt = idx_to_date.get(endpoint_idx)
            if ep_dt is None:
                continue
            dist = abs((ep_dt - target_dt).days)
            if dist < best_dist:
                best_dist = dist
                best_match = (i, s, endpoint_idx, endpoint_val, ep_dt)

    if best_match:
        si, s, eidx, eval_, edt = best_match
        direction = '↑' if s.type.value == 'up' else '↓'
        price_diff = abs(eval_ - val)
        match_icon = 'OK' if best_dist <= 3 and price_diff < 5 else 'MISS'
        print(f'  TDX: {dt_str} {val:>8.2f} ({tp:>6})'
              f'  →  Core: {edt.strftime("%Y-%m-%d")} {eval_:>8.2f}'
              f'  Δ日期={best_dist}天  Δ价格={price_diff:.2f}  {match_icon}')

# ============ 在该日期附近查看原始K线数据 ============
print()
print('=' * 70)
print('TDX笔端点附近原始K线数据:')
print('=' * 70)

for dt_str, val, tp in tdx_strokes:
    target_dt = pd.Timestamp(dt_str)
    # 找前后3天的K线
    mask = (df.index >= target_dt - pd.Timedelta(days=5)) & (df.index <= target_dt + pd.Timedelta(days=5))
    nearby = df[mask]
    print(f'\n  {dt_str} (TDX值={val:.2f}, {tp}):')
    for dt, row in nearby.iterrows():
        marker = ' <<<' if dt.strftime('%Y-%m-%d') == dt_str else ''
        print(f'    {dt.strftime("%Y-%m-%d")}  O={row["open"]:.2f}  H={row["high"]:.2f}'
              f'  L={row["low"]:.2f}  C={row["close"]:.2f}{marker}')
