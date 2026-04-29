#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sz002230 缠论分析图 v3 - 匹配通达信缠论逻辑
min_bars=3 产生完整的笔/线段/中枢结构
"""
import sys, os
sys.path.insert(0, '.')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Rectangle
from loguru import logger
logger.remove()
logger.add(sys.stderr, level='WARNING')

from mootdx.quotes import Quotes
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from core.trend_track import TrendTrackDetector
from indicator.macd import MACD

client = Quotes.factory(market='sz')
api = client.client
data = api.get_security_bars(4, 0, '002230', 0, 800)
df = pd.DataFrame(data)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.rename(columns={'datetime': 'date'})
df = df[['date','open','close','high','low','vol','amount']].rename(columns={'vol': 'volume'})

kline = KLine.from_dataframe(df, strict_mode=True)
fractals = FractalDetector(kline, confirm_required=False).get_fractals()
strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
segments = SegmentGenerator(kline, strokes).get_segments()
pivots = PivotDetector(kline, strokes).get_pivots()
close_s = pd.Series([k.close for k in kline])
macd_obj = MACD(close_s)
td = TrendTrackDetector(strokes, pivots)
td.detect()
detector = BuySellPointDetector(fractals, strokes, segments, pivots, macd_obj, trend_tracks=td._tracks)
buys, sells = detector.detect_all()
klen = len(kline)

# Last 300 bars
si = max(0, klen - 300)
ei = klen
N = ei - si
ohlc = df.iloc[si:ei]

print(f'KLine={klen} Strokes={len(strokes)} Segments={len(segments)} Pivots={len(pivots)} Buys={len(buys)}')

def idx2x(idx):
    return idx - si

def in_view(s):
    return s.end_index >= si and s.start_index < ei

# === Create figure ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 18), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('sz002230 ChanLun (min_bars=3): Strokes + Segments + Pivots', fontsize=17, fontweight='bold')

# ===== Panel 1: Price Chart =====
ax1.set_title('K-Line + Strokes(Bi) + Segments(Duan) + Pivots(ZhongShu)', fontsize=13)
x = np.arange(N)

# Candlesticks
for i in range(N):
    o = float(ohlc.iloc[i]['open'])
    c = float(ohlc.iloc[i]['close'])
    h = float(ohlc.iloc[i]['high'])
    l = float(ohlc.iloc[i]['low'])
    col = 'red' if c >= o else '#00AA00'
    body = max(abs(c - o), 0.01)
    ax1.bar(i, body, bottom=min(o, c), color=col, alpha=0.5, width=0.8)
    ax1.bar(i, h - max(o, c), bottom=max(o, c), color=col, alpha=0.5, width=0.15)
    ax1.bar(i, min(o, c) - l, bottom=l, color=col, alpha=0.5, width=0.15)

# --- Draw all Strokes (笔) ---
for s in strokes:
    if not in_view(s):
        continue
    x1 = max(idx2x(s.start_index), 0)
    x2 = min(idx2x(s.end_index), N - 1)
    if x2 < x1:
        continue
    y1 = s.start_value
    y2 = s.end_value
    c = '#2196F3' if s.is_up else '#FF8C00'
    ax1.plot([x1, x2], [y1, y2], color=c, linewidth=1.0, alpha=0.5, zorder=5)

# --- Draw Segments (线段) ---
for seg in segments:
    if seg.end_index < si or seg.start_index >= ei:
        continue
    x1 = max(idx2x(seg.start_index), 0)
    x2 = min(idx2x(seg.end_index), N - 1)
    if x2 < x1:
        continue
    # Use segment's start/end values if available, else close price
    y1 = seg.start_value if hasattr(seg, 'start_value') else close_s.iloc[max(seg.start_index, si)]
    y2 = seg.end_value if hasattr(seg, 'end_value') else close_s.iloc[min(seg.end_index, ei - 1)]
    ax1.plot([x1, x2], [y1, y2], color='#9C27B0', linewidth=2.5, alpha=0.7, zorder=8,
             solid_capstyle='round')

# --- Draw Pivots (中枢) ---
pivot_in_view = []
for pi, p in enumerate(pivots):
    if p.end_index < si or p.start_index >= ei:
        continue
    ps = max(idx2x(p.start_index), 0)
    pe = min(idx2x(p.end_index), N - 1)
    if pe <= ps:
        continue

    # First 3 strokes overlap
    first3 = p.strokes[:3]
    overlap_h = min(s.high for s in first3)
    overlap_l = max(s.low for s in first3)
    if overlap_h <= overlap_l:
        continue

    pivot_in_view.append((pi, p, ps, pe, overlap_h, overlap_l))

    # Draw pivot zone as filled rectangle
    rect = Rectangle((ps, overlap_l), pe - ps, overlap_h - overlap_l,
                      facecolor='#87CEEB', alpha=0.3, edgecolor='#4169E1', linewidth=1.5, zorder=4)
    ax1.add_patch(rect)

    # Top/bottom lines extend a bit beyond
    ext = min(pe + 8, N - 1)
    ax1.hlines(overlap_h, ps, ext, colors='#4169E1', linewidth=1.2, linestyles='--', alpha=0.7)
    ax1.hlines(overlap_l, ps, ext, colors='#4169E1', linewidth=1.2, linestyles='--', alpha=0.7)

    # Label pivot with index and range
    ax1.text((ps + pe) / 2, overlap_h + 0.3,
             f'ZS{pi}[{overlap_l:.1f}-{overlap_h:.1f}]',
             fontsize=7, ha='center', color='#4169E1', fontweight='bold', alpha=0.8)

# --- Highlight the specific 3-buy pivot ---
# Find the LAST 3-buy signal, then its associated pivot
target_pivot = None
buy_pivot_idx = -1
last_3buy = None
for bp in buys:
    if bp.point_type == '3buy':
        last_3buy = bp
if last_3buy:
    # Find the pivot that this 3-buy refers to (pivot before the breakout)
    best_dist = 9999
    for pi, p in enumerate(pivots):
        dist = last_3buy.index - p.end_index
        if dist > 0 and dist < best_dist:
            best_dist = dist
            target_pivot = p
            buy_pivot_idx = pi

if target_pivot is None and pivots:
    target_pivot = pivots[-1]
    buy_pivot_idx = len(pivots) - 1

if target_pivot:
    p = target_pivot
    ps = max(idx2x(p.start_index), 0)
    pe = min(idx2x(p.end_index), N - 1)
    first3 = p.strokes[:3]
    oh = min(s.high for s in first3)
    ol = max(s.low for s in first3)

    # Highlight with thicker border
    if oh > ol:
        rect = Rectangle((ps, ol), pe - ps, oh - ol,
                          facecolor='gold', alpha=0.35, edgecolor='red', linewidth=2.5, zorder=6)
        ax1.add_patch(rect)
        ext = min(pe + 40, N - 1)
        ax1.hlines(oh, ps, ext, colors='red', linewidth=2, linestyles='-', zorder=9)
        ax1.hlines(ol, ps, ext, colors='green', linewidth=2, linestyles='-', zorder=9)

        # Label the 3 strokes that form the pivot
        for si2, s in enumerate(first3):
            x1 = max(idx2x(s.start_index), 0)
            x2 = min(idx2x(s.end_index), N - 1)
            if x2 >= x1:
                d = 'UP' if s.is_up else 'DN'
                ax1.plot([x1, x2], [s.start_value, s.end_value],
                         color='red', linewidth=2.5, zorder=11)
                mx = (x1 + x2) / 2
                my = s.end_value if s.is_up else s.start_value
                ax1.annotate(f'S{si2+1}({d})\n[{s.low:.1f}-{s.high:.1f}]',
                             xy=(mx, my), fontsize=8, fontweight='bold', color='red',
                             ha='center', va='bottom' if s.is_up else 'top',
                             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='red'))

        # Pivot info
        ax1.text(ext + 1, oh, f'H={oh:.2f}', fontsize=10, color='red', fontweight='bold')
        ax1.text(ext + 1, ol, f'L={ol:.2f}', fontsize=10, color='green', fontweight='bold')

    # Breakout stroke
    breakout = None
    pullback = None
    for s in strokes:
        if s.is_up and s.end_value > target_pivot.high and s.start_index >= target_pivot.end_index:
            breakout = s
            break
    if breakout:
        x1 = max(idx2x(breakout.start_index), 0)
        x2 = min(idx2x(breakout.end_index), N - 1)
        ax1.plot([x1, x2], [breakout.start_value, breakout.end_value],
                 color='magenta', linewidth=3, zorder=12)
        ax1.annotate('Breakout', xy=(x2, breakout.end_value), xytext=(10, 15),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
                     fontsize=10, fontweight='bold', color='magenta')

        # Pullback (3-Buy)
        for s in strokes:
            if s.is_down and s.start_index > breakout.end_index and s.end_value > target_pivot.high:
                pullback = s
                break
        if pullback:
            x1 = max(idx2x(pullback.start_index), 0)
            x2 = min(idx2x(pullback.end_index), N - 1)
            ax1.plot([x1, x2], [pullback.start_value, pullback.end_value],
                     color='blue', linewidth=3, zorder=12)
            ax1.annotate(f'3-BUY @ {pullback.end_value:.2f}',
                         xy=(x2, pullback.end_value), xytext=(15, -25),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='red', lw=2),
                         fontsize=12, fontweight='bold', color='red',
                         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

            # Divergence info
            entering_up = [s for s in target_pivot.strokes if s.is_up]
            entering_areas = [macd_obj.compute_area(s.start_index, s.end_index, 'up') for s in entering_up]
            max_entering = max(entering_areas) if entering_areas else 0
            leaving_area = macd_obj.compute_area(breakout.start_index, breakout.end_index, 'up')
            ratio = leaving_area / max_entering if max_entering > 0 else 0
            div_c = '#CC0000' if ratio < 1.0 else '#006600'
            box_c = '#FFE0E0' if ratio < 1.0 else '#E0FFE0'
            label = 'PIVOT DIV!' if ratio < 1.0 else 'No Div'
            ax1.text(0.98, 0.98,
                     f'Pivot Div Ratio={ratio:.2f}\nLeaving={leaving_area:.1f} Entering={max_entering:.1f}\n{label}',
                     fontsize=10, fontweight='bold', color=div_c,
                     transform=ax1.transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor=box_c, alpha=0.9, edgecolor=div_c, linewidth=2),
                     zorder=20)
            print(f'Target Pivot#{buy_pivot_idx}: H={oh:.2f} L={ol:.2f} Range={oh-ol:.2f}')
            print(f'Breakout: {breakout.start_value:.2f}->{breakout.end_value:.2f}')
            print(f'Pullback: {pullback.end_value:.2f}')
            print(f'Entering max: {max_entering:.1f}  Leaving: {leaving_area:.1f}  Ratio: {ratio:.2f}')

# Buy point markers
for bp in buys:
    if si <= bp.index < ei:
        marker = '^' if bp.point_type in ('1buy', '2buy', '3buy') else 'v'
        color = 'red' if 'buy' in bp.point_type else 'green'
        ax1.plot(bp.index - si, bp.price, marker, color=color, markersize=12, zorder=20)

ax1.axhline(close_s.iloc[-1], color='black', linewidth=0.5, alpha=0.5)
ax1.set_xlim(0, N - 1)
ax1.set_ylabel('Price')
ax1.grid(alpha=0.2)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2196F3', linewidth=1, alpha=0.6, label='Stroke UP'),
    Line2D([0], [0], color='#FF8C00', linewidth=1, alpha=0.6, label='Stroke DN'),
    Line2D([0], [0], color='#9C27B0', linewidth=2.5, alpha=0.7, label='Segment'),
    Line2D([0], [0], color='#4169E1', linewidth=1.2, linestyle='--', label='Pivot zone'),
    Line2D([0], [0], color='magenta', linewidth=3, label='Breakout'),
    Line2D([0], [0], color='blue', linewidth=3, label='3Buy Pullback'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# ===== Panel 2: MACD =====
ax2.set_title('MACD', fontsize=12)
mv = macd_obj.values
hist = [v.histogram for v in mv[si:ei]]
dif = [v.macd for v in mv[si:ei]]
dea = [v.signal for v in mv[si:ei]]
xm = np.arange(len(hist))
ax2.bar(xm, hist, color=['red' if h >= 0 else '#00AA00' for h in hist], alpha=0.7, width=0.8)
ax2.plot(xm, dif, color='blue', linewidth=1, label='DIF')
ax2.plot(xm, dea, color='orange', linewidth=1, label='DEA')
ax2.axhline(0, color='black', linewidth=0.5)

# Highlight pivot zones on MACD
for pi, p, ps, pe, oh, ol in pivot_in_view:
    ax2.axvspan(ps, pe, alpha=0.08, color='gold')

ax2.set_xlim(0, len(hist) - 1)
ax2.set_ylabel('MACD')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = 'backtest_charts/sz002230_3buy_analysis.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
plt.close()
