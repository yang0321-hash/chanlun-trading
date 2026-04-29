#!/usr/bin/env python3
"""300750 宁德时代 日线笔段中枢图 — 与缠论软件对比"""
import sys, os
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json, requests

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

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

print(f'日线数据: {len(df)} 根  {df.index[0].date()} ~ {df.index[-1].date()}')

# ============ 缠论核心分析 ============
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (KLine, FractalDetector, StrokeGenerator,
                  SegmentGenerator, PivotDetector, PivotLevel)

kline = KLine.from_dataframe(df, strict_mode=True)
print(f'处理后K线: {len(kline.processed_data)} 根')

# 分型
fractal_det = FractalDetector(kline, confirm_required=False)
fractals = fractal_det.get_fractals()
print(f'分型: {len(fractals)} 个')

# 笔
stroke_gen = StrokeGenerator(kline, fractals, min_bars=5)
strokes = stroke_gen.get_strokes()
print(f'笔: {len(strokes)} 根')
for i, s in enumerate(strokes):
    direction = '↑' if s.type.value == 'up' else '↓'
    print(f'  笔{i+1} {direction}: [{s.start_index}→{s.end_index}] '
          f'{s.start_value:.2f}→{s.end_value:.2f}  H={s.high:.2f} L={s.low:.2f}')

# 线段
seg_gen = SegmentGenerator(kline, strokes, min_strokes=3)
segments = seg_gen.get_segments()
print(f'线段: {len(segments)} 段')
for i, seg in enumerate(segments):
    d = seg.direction
    print(f'  线段{i+1} ({d}): [{seg.start_index}→{seg.end_index}] '
          f'{seg.start_value:.2f}→{seg.end_value:.2f}  H={seg.high:.2f} L={seg.low:.2f}')

# 中枢
pivot_det = PivotDetector(kline, strokes, level=PivotLevel.DAY)
pivots = pivot_det.get_pivots()
print(f'中枢: {len(pivots)} 个')
for i, pv in enumerate(pivots):
    print(f'  中枢{i+1}: [{pv.start_index}→{pv.end_index}] '
          f'ZG={pv.zg:.2f} ZD={pv.zd:.2f} ZZ={((pv.zg+pv.zd)/2):.2f} '
          f'GG={pv.gg:.2f} DD={pv.dd:.2f}')

# ============ 绘图 ============
# 核心: 原始K线用 x=[0..n-1] 画, 缠论结果用 processed_index → 日期 → 原始x坐标 映射
fig, ax = plt.subplots(1, 1, figsize=(24, 10))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#ffffff')

x = np.arange(len(df))
dates = df.index

# 建立: 原始日期 → 原始x坐标 映射
date_to_x = {dates[i]: i for i in range(len(dates))}

# 建立: processed_index → 原始x坐标 映射
# 用处理后K线的datetime找原始K线位置
processed = kline.processed_data
pidx_to_x = {}
for pi in range(len(processed)):
    dt = processed[pi].datetime
    # 精确匹配
    if dt in date_to_x:
        pidx_to_x[pi] = date_to_x[dt]
    else:
        # 找最近的日期
        ts = pd.Timestamp(dt)
        diffs = [(abs((d - ts).days), xi) for d, xi in date_to_x.items() if d >= ts - pd.Timedelta(days=5) and d <= ts + pd.Timedelta(days=5)]
        if diffs:
            pidx_to_x[pi] = min(diffs)[1]
        else:
            pidx_to_x[pi] = pi  # fallback

# K线
for i in range(len(df)):
    o, h, l, c = df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
    color = '#e74c3c' if c >= o else '#2ecc71'
    ax.plot([i, i], [l, h], color=color, linewidth=0.6, alpha=0.5)
    ax.plot([i, i], [o, c], color=color, linewidth=2.5, alpha=0.7, solid_capstyle='round')

# ---- 画中枢 (浅蓝色矩形) ----
for pv in pivots:
    si = pidx_to_x.get(pv.start_index, pv.start_index)
    ei = pidx_to_x.get(pv.end_index, pv.end_index)
    rect = Rectangle((si - 0.5, pv.zd),
                      ei - si + 1,
                      pv.zg - pv.zd,
                      linewidth=2, edgecolor='#3498db',
                      facecolor='#3498db', alpha=0.15)
    ax.add_patch(rect)
    # ZG ZD 虚线
    ax.plot([si - 0.5, ei + 0.5], [pv.zg, pv.zg],
            color='#3498db', linewidth=1, linestyle='--', alpha=0.7)
    ax.plot([si - 0.5, ei + 0.5], [pv.zd, pv.zd],
            color='#3498db', linewidth=1, linestyle='--', alpha=0.7)
    # 标注
    mid_x = (si + ei) / 2
    ax.text(mid_x, (pv.zg + pv.zd) / 2,
            f'ZG={pv.zg:.0f}\nZD={pv.zd:.0f}',
            ha='center', va='center', fontsize=8, color='#2980b9',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#3498db', alpha=0.8))

# ---- 画笔 (紫色线) ----
for s in strokes:
    sx = pidx_to_x.get(s.start_index, s.start_index)
    ex = pidx_to_x.get(s.end_index, s.end_index)
    ax.plot([sx, ex],
            [s.start_value, s.end_value],
            color='#8e44ad', linewidth=1.8, alpha=0.7, zorder=3)
    # 笔端点圆点
    ax.plot(sx, s.start_value, 'o', color='#8e44ad',
            markersize=4, alpha=0.6, zorder=4)
    ax.plot(ex, s.end_value, 'o', color='#8e44ad',
            markersize=4, alpha=0.6, zorder=4)

# ---- 线段暂时不画 ----
# for seg in segments:
#     ...

# ---- 分型标记 (小三角) ----
for f in fractals:
    fx = pidx_to_x.get(f.index, f.index)
    if f.type.value == 'top':
        ax.plot(fx, f.kline2.high, 'v', color='#e74c3c',
                markersize=5, alpha=0.5, zorder=4)
    else:
        ax.plot(fx, f.kline2.low, '^', color='#27ae60',
                markersize=5, alpha=0.5, zorder=4)

# 图例
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#8e44ad', linewidth=1.8, alpha=0.7, label='笔'),
    Patch(facecolor='#3498db', alpha=0.15, edgecolor='#3498db', linewidth=2, label='中枢'),
    Line2D([0], [0], marker='v', color='#e74c3c', linewidth=0, markersize=6, label='顶分型'),
    Line2D([0], [0], marker='^', color='#27ae60', linewidth=0, markersize=6, label='底分型'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

# 标题
ax.set_title(
    f'300750 宁德时代 | 日线笔段中枢 (core模块)\n'
    f'笔={len(strokes)}根  线段={len(segments)}段  中枢={len(pivots)}个',
    fontsize=13, pad=10
)
ax.set_ylabel('价格 (元)', fontsize=12)
ax.grid(True, alpha=0.2)

# X轴日期
step = max(1, len(df) // 20)
tick_pos = list(range(0, len(df), step))
tick_lbl = [dates[i].strftime('%Y-%m-%d') for i in tick_pos]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbl, rotation=45, fontsize=8)

plt.tight_layout()
out = 'backtest_charts/300750_daily_chanlun.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'\nChart saved: {out}')
