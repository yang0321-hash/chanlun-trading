#!/usr/bin/env python3
"""生成300750买卖点图表 + 缠论中枢"""
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

# ============ 30min 缠论组件 ============

def detect_fractals(df):
    n = len(df)
    if n < 5:
        return []
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    merged = [{'high': high[0], 'low': low[0], 'idx': 0}]
    direction = 0
    for i in range(1, n):
        prev = merged[-1]
        if len(merged) >= 2:
            prev2 = merged[-2]
            if prev['high'] > prev2['high'] and prev['low'] > prev2['low']:
                direction = 1
            elif prev['high'] < prev2['high'] and prev['low'] < prev2['low']:
                direction = -1
        pc = prev['high'] >= high[i] and prev['low'] <= low[i]
        cp = high[i] >= prev['high'] and low[i] <= prev['low']
        if pc or cp:
            if direction == 1:
                prev['high'] = max(prev['high'], high[i])
                prev['low'] = max(prev['low'], low[i])
            elif direction == -1:
                prev['high'] = min(prev['high'], high[i])
                prev['low'] = min(prev['low'], low[i])
            else:
                if cp:
                    prev['high'] = high[i]
                    prev['low'] = low[i]
        else:
            merged.append({'high': high[i], 'low': low[i], 'idx': i})

    fractals = []
    for j in range(1, len(merged) - 1):
        if merged[j]['high'] > merged[j-1]['high'] and merged[j]['high'] > merged[j+1]['high']:
            fractals.append({'type': 'top', 'idx': merged[j]['idx'], 'val': merged[j]['high'], 'midx': j})
        elif merged[j]['low'] < merged[j-1]['low'] and merged[j]['low'] < merged[j+1]['low']:
            fractals.append({'type': 'bottom', 'idx': merged[j]['idx'], 'val': merged[j]['low'], 'midx': j})

    if not fractals:
        return []
    filtered = [fractals[0]]
    for f in fractals[1:]:
        if f['type'] == filtered[-1]['type']:
            if f['type'] == 'top' and f['val'] > filtered[-1]['val']:
                filtered[-1] = f
            elif f['type'] == 'bottom' and f['val'] < filtered[-1]['val']:
                filtered[-1] = f
        else:
            if f['midx'] - filtered[-1]['midx'] >= 2:
                filtered.append(f)
    return filtered


def detect_strokes(fractals):
    strokes = []
    for j in range(len(fractals) - 1):
        f1, f2 = fractals[j], fractals[j + 1]
        strokes.append({
            'start_idx': f1['idx'], 'end_idx': f2['idx'],
            'start_type': f1['type'], 'end_type': f2['type'],
            'start_val': f1['val'], 'end_val': f2['val'],
            'high': max(f1['val'], f2['val']),
            'low': min(f1['val'], f2['val']),
        })
    return strokes


def detect_pivots(strokes):
    pivots = []
    j = 0
    while j <= len(strokes) - 3:
        s1, s2, s3 = strokes[j], strokes[j+1], strokes[j+2]
        zg = min(s1['high'], s2['high'], s3['high'])
        zd = max(s1['low'], s2['low'], s3['low'])
        if zg > zd:
            k = j + 3
            while k < len(strokes):
                sk = strokes[k]
                nzg = min(zg, sk['high'])
                nzd = max(zd, sk['low'])
                if nzg > nzd:
                    zg, zd = nzg, nzd
                    k += 1
                else:
                    break
            pivots.append({
                'ZG': zg, 'ZD': zd, 'ZZ': (zg + zd) / 2,
                'start_idx': strokes[j]['start_idx'],
                'end_idx': strokes[k-1]['end_idx'],
                'stroke_start': j, 'stroke_end': k - 1,
            })
            j = k
        else:
            j += 1
    return pivots


# ============ 获取数据 ============

code = '300750'
session = requests.Session()
session.trust_env = False
url = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sz{code}&scale=30&ma=no&datalen=2000'
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

# 缠论分析
fractals = detect_fractals(df)
strokes = detect_strokes(fractals)
pivots = detect_pivots(strokes)

# ============ 交易信息 ============
entry_date = pd.Timestamp('2025-07-22 14:00')
entry_price = 283.30
reduce_price = 430.09
reduce_mask = (df.index > entry_date) & (df['close'] >= 429) & (df['close'] <= 432)
reduce_date = df.index[reduce_mask][0] if reduce_mask.any() else df.index[-1]

# ============ 绘图 ============
mask = (df.index >= '2025-06-01') & (df.index <= '2026-04-16')
df_plot = df[mask].copy()

# 建立 plot_idx → 原始df idx 的映射
idx_map = {df_plot.index[i]: i for i in range(len(df_plot))}

fig, axes = plt.subplots(2, 1, figsize=(22, 12), gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#f8f9fa')

ax1 = axes[0]
ax1.set_facecolor('#ffffff')
x = np.arange(len(df_plot))
dates = df_plot.index

# K线
for i in range(len(df_plot)):
    o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
    color = '#e74c3c' if c >= o else '#2ecc71'
    ax1.plot([i, i], [l, h], color=color, linewidth=0.6, alpha=0.5)
    ax1.plot([i, i], [o, c], color=color, linewidth=2, alpha=0.7, solid_capstyle='round')

# 画中枢
for pv in pivots:
    start_dt = df.index[pv['start_idx']]
    end_dt = df.index[pv['end_idx']]
    # 只画在可见范围内的中枢
    if start_dt < df_plot.index[0] or end_dt > df_plot.index[-1]:
        # 裁剪到可见范围
        start_dt = max(start_dt, df_plot.index[0])
        end_dt = min(end_dt, df_plot.index[-1])
        if start_dt >= end_dt:
            continue

    # 找plot坐标
    start_plot = df_plot.index.get_indexer([start_dt], method='nearest')[0]
    end_plot = df_plot.index.get_indexer([end_dt], method='nearest')[0]

    # 判断中枢是否在入场后(用不同颜色)
    if start_dt >= entry_date:
        rect_color = '#3498db'  # 蓝色 - 入场后的中枢
        alpha = 0.25
        label = '入场后中枢'
    else:
        rect_color = '#95a5a6'  # 灰色 - 入场前的中枢
        alpha = 0.15
        label = '入场前中枢'

    rect = Rectangle((start_plot - 0.5, pv['ZD']),
                      end_plot - start_plot + 1,
                      pv['ZG'] - pv['ZD'],
                      linewidth=1.5, edgecolor=rect_color,
                      facecolor=rect_color, alpha=alpha)
    ax1.add_patch(rect)

    # ZG ZD 标注线
    ax1.plot([start_plot - 0.5, end_plot + 0.5], [pv['ZG'], pv['ZG']],
             color=rect_color, linewidth=0.8, linestyle='--', alpha=0.6)
    ax1.plot([start_plot - 0.5, end_plot + 0.5], [pv['ZD'], pv['ZD']],
             color=rect_color, linewidth=0.8, linestyle='--', alpha=0.6)

    # 中枢中心价标注
    mid_x = (start_plot + end_plot) / 2
    ax1.text(mid_x, pv['ZZ'], f'ZG:{pv["ZG"]:.0f}\nZD:{pv["ZD"]:.0f}',
             ha='center', va='center', fontsize=7, color=rect_color, alpha=0.8,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=rect_color, alpha=0.7))

# 画笔 (strokes)
for s in strokes:
    s_start = df.index[s['start_idx']]
    s_end = df.index[s['end_idx']]
    if s_start < df_plot.index[0] or s_end > df_plot.index[-1]:
        continue
    si = df_plot.index.get_indexer([s_start], method='nearest')[0]
    ei = df_plot.index.get_indexer([s_end], method='nearest')[0]
    ax1.plot([si, ei], [s['start_val'], s['end_val']],
             color='#8e44ad', linewidth=1.2, alpha=0.5, zorder=3)

# 画分型
for f in fractals:
    f_dt = df.index[f['idx']]
    if f_dt < df_plot.index[0] or f_dt > df_plot.index[-1]:
        continue
    fi = df_plot.index.get_indexer([f_dt], method='nearest')[0]
    if f['type'] == 'top':
        ax1.plot(fi, f['val'], 'rv', markersize=3, alpha=0.4, zorder=4)
    else:
        ax1.plot(fi, f['val'], 'r^', markersize=3, alpha=0.4, zorder=4)

# ========= 买卖点标注 =========

# 入场
entry_idx = df_plot.index.get_indexer([entry_date], method='nearest')[0]
ax1.plot(entry_idx, entry_price, marker='^', markersize=18, color='red', zorder=10,
         markeredgecolor='darkred', markeredgewidth=2)
ax1.annotate(f'买入 {entry_price}元', xy=(entry_idx, entry_price),
             xytext=(entry_idx + 30, entry_price - 40),
             fontsize=14, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe0e0', edgecolor='darkred', alpha=0.9))

# 减仓
reduce_idx = df_plot.index.get_indexer([reduce_date], method='nearest')[0]
ax1.plot(reduce_idx, reduce_price, marker='v', markersize=14, color='orange', zorder=10,
         markeredgecolor='darkorange', markeredgewidth=2)
ax1.annotate(f'减仓30% {reduce_price:.0f}元', xy=(reduce_idx, reduce_price),
             xytext=(reduce_idx - 60, reduce_price + 30),
             fontsize=13, fontweight='bold', color='darkorange',
             arrowprops=dict(arrowstyle='->', color='darkorange', lw=2),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', edgecolor='darkorange', alpha=0.9))

# 当前持仓
last_idx = len(df_plot) - 1
last_price = df_plot['close'].iloc[-1]
profit_pct = (last_price / entry_price - 1) * 100
ax1.plot(last_idx, last_price, marker='*', markersize=18, color='#27ae60', zorder=10,
         markeredgecolor='darkgreen', markeredgewidth=1.5)
ax1.annotate(f'持仓 {last_price:.0f}元 (+{profit_pct:.0f}%)',
             xy=(last_idx, last_price), xytext=(last_idx - 80, last_price + 30),
             fontsize=13, fontweight='bold', color='darkgreen',
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9', edgecolor='darkgreen', alpha=0.9))

# 买入价参考线
ax1.axhline(y=entry_price, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax1.text(5, entry_price + 5, f'买入价 {entry_price}元', color='red', alpha=0.5, fontsize=9)

ax1.set_ylabel('价格 (元)', fontsize=13)
ax1.grid(True, alpha=0.2)
ax1.set_title(
    f'300750 宁德时代 | 缠论30min中枢 + MTF V3策略'
    f'\n入场: {entry_date.strftime("%Y-%m-%d %H:%M")} @ {entry_price}元'
    f'  减仓: {reduce_date.strftime("%Y-%m-%d")} @ {reduce_price:.0f}元'
    f'  当前: {last_price:.0f}元  收益: +{profit_pct:.1f}%',
    fontsize=12, pad=10
)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', alpha=0.25, edgecolor='#3498db', label='入场后中枢'),
    Patch(facecolor='#95a5a6', alpha=0.15, edgecolor='#95a5a6', label='入场前中枢'),
    plt.Line2D([0], [0], color='#8e44ad', linewidth=1.5, alpha=0.6, label='笔'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

# 成交量子图
ax2 = axes[1]
ax2.set_facecolor('#ffffff')
vol_colors = ['#e74c3c' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else '#2ecc71'
              for i in range(len(df_plot))]
ax2.bar(x, df_plot['volume'] / 1e6, color=vol_colors, alpha=0.5, width=1)
ax2.set_ylabel('成交量 (百万)', fontsize=11)
ax2.grid(True, alpha=0.2)

# X轴
step = max(1, len(df_plot) // 18)
tick_pos = list(range(0, len(df_plot), step))
tick_lbl = [dates[i].strftime('%Y-%m-%d') for i in tick_pos]
for ax in axes:
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, fontsize=8)

plt.tight_layout()
out = 'backtest_charts/300750_mtf_v3.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Chart saved: {out}')
print(f'Pivots found: {len(pivots)}')
for i, pv in enumerate(pivots):
    s = df.index[pv['start_idx']].strftime('%Y-%m-%d')
    e = df.index[pv['end_idx']].strftime('%Y-%m-%d')
    print(f'  Pivot {i+1}: {s} ~ {e}  ZG={pv["ZG"]:.2f}  ZD={pv["ZD"]:.2f}')
print(f'Entry: {entry_date.strftime("%Y-%m-%d %H:%M")} @ {entry_price}')
print(f'Reduce: {reduce_date.strftime("%Y-%m-%d")} @ {reduce_price:.2f}')
print(f'Current: {last_price:.0f}, Return: +{profit_pct:.1f}%')
