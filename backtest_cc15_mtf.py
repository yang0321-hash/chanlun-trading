#!/usr/bin/env python3
"""CC15 多周期策略回测 V5 — 四级别联立 (2买版)

规则:
  入场: 月线定方向 + 周线看结构 + 日线2买 → 30min 2买确认入场
  止损(亏损端):
    - 日线2买最低点 / ATR动态止损 / -10%硬止损
    - 底背驰最低点(重新入场)
    - 亏损/微利(<5%)仓位: 480 bars时间止损
  减仓(盈利端, 结构确认):
    Step1: 30min形成中枢 + 跌破中枢ZG → 减30%
    Step2: 跌破ZG后反弹无法收回ZG → 再减20%
    剩余50%: 中枢上移追踪(最新中枢ZD做移动止损)
  清仓:
    - 跌破最新中枢ZD
    - 日线2卖信号
  盈利仓位(>5%)不受时间止损限制
"""
import sys, os
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD


# ============ 30分钟级别缠论组件 (Swing算法, 隔离自core模块) ============

def _merge_inclusion_30min(high, low):
    """30分钟K线包含关系处理, 返回合并后的K线列表"""
    n = len(high)
    if n == 0:
        return []
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
    return merged


def detect_fractals_30min(df):
    """30分钟分型检测 — 使用Core缠论引擎

    用于: 找入场底分型 (entry_fractal = bottom_after[1])
    """
    n = len(df)
    if n < 5:
        return []

    from core.kline import KLine
    from core.fractal import FractalDetector, FractalType

    klines = KLine.from_dataframe(df.reset_index())
    fd = FractalDetector(klines)

    fractals = []
    for f in fd.fractals:
        ftype = 'top' if f.type == FractalType.TOP else 'bottom'
        fractals.append({
            'type': ftype,
            'idx': f.index,
            'val': f.value,
            'midx': f.index,
        })
    return fractals


def detect_strokes_30min(df_or_fractals):
    """Core缠论笔检测 — 替代Swing算法

    Args:
        df_or_fractals: 传df(DataFrame)用Core缠论, 传fractals(list)用兼容逻辑
    Returns:
        list of stroke dicts (格式与旧接口兼容)
    """
    # 兼容: 如果传入的是fractals列表, 用旧逻辑
    if isinstance(df_or_fractals, list):
        fractals = df_or_fractals
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

    # Core缠论: 分型→笔
    df = df_or_fractals
    n = len(df)
    if n < 5:
        return []

    from core.kline import KLine
    from core.fractal import FractalDetector
    from core.stroke import StrokeGenerator

    klines = KLine.from_dataframe(df.reset_index())
    fd = FractalDetector(klines)
    sg = StrokeGenerator(klines, fd.fractals)

    strokes = []
    for s in sg.strokes:
        start_type = 'bottom' if s.is_up else 'top'
        end_type = 'top' if s.is_up else 'bottom'
        strokes.append({
            'start_idx': s.start_index,
            'end_idx': s.end_index,
            'start_type': start_type,
            'end_type': end_type,
            'start_val': s.start_value,
            'end_val': s.end_value,
            'high': s.high,
            'low': s.low,
        })
    return strokes


def detect_pivot_30min(strokes):
    """30分钟中枢检测: >=3笔重叠"""
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
                'ZG': zg, 'ZD': zd,
                'start_idx': strokes[j]['start_idx'],
                'end_idx': strokes[k-1]['end_idx'],
                'stroke_start': j, 'stroke_end': k - 1,
            })
            j = k
        else:
            j += 1
    return pivots


def detect_macd_divergence_30min(df, strokes):
    """30分钟MACD面积背驰检测（增强版：多笔回溯+衰减加权）"""
    close = df['close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    buy_div = set()  # 底背驰
    sell_div = set()  # 顶背驰

    down_strokes = [s for s in strokes if s['start_type'] == 'top' and s['end_type'] == 'bottom']
    up_strokes = [s for s in strokes if s['start_type'] == 'bottom' and s['end_type'] == 'top']

    # 底背驰: 多笔回溯（最多5笔），指数衰减加权
    # P0修复: c笔必须包含≥2个次级别中枢，即K线数≥5
    for k in range(1, len(down_strokes)):
        curr = down_strokes[k]
        # P0修复v2: c笔K线数≥3（1.5小时，平衡严格性与信号量）
        # c笔K线数过滤：至少3根K线（1.5小时，有效过滤噪音微笔）
        if curr['end_idx'] - curr['start_idx'] < 2:
            continue
        curr_area = abs(sum(hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
        is_divergence = False
        for lookback in range(1, min(6, k+1)):
            prev = down_strokes[k - lookback]
            prev_area = abs(sum(hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            # 价格更低 + 面积更小 = 底背驰
            if curr['end_val'] < prev['end_val'] and curr_area < prev_area:
                # 衰减加权：越近的笔权重越高
                weight = 0.7 ** (lookback - 1)
                if weight >= 0.3:  # 只接受前3笔的比较
                    is_divergence = True
                    break
        if is_divergence:
            buy_div.add(curr['end_idx'])

    # 顶背驰: 同样多笔回溯
    for k in range(1, len(up_strokes)):
        curr = up_strokes[k]
        if curr['end_idx'] - curr['start_idx'] < 2:
            continue
        curr_area = abs(sum(hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
        is_divergence = False
        for lookback in range(1, min(6, k+1)):
            prev = up_strokes[k - lookback]
            prev_area = abs(sum(hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            if curr['end_val'] > prev['end_val'] and curr_area < prev_area:
                weight = 0.7 ** (lookback - 1)
                if weight >= 0.3:
                    is_divergence = True
                    break
        if is_divergence:
            sell_div.add(curr['end_idx'])

    return buy_div, sell_div


# ============ 四级别联立: 月线→周线→日线→30min ============

def resample_to_weekly(df_daily):
    """日线→周线resample"""
    df = df_daily.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    weekly = df.resample('W').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return weekly


def resample_to_monthly(df_daily):
    """日线→月线resample"""
    df = df_daily.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    monthly = df.resample('ME').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return monthly


def check_monthly_direction(df_monthly, signal_date=None):
    """月线定方向: 价格在MA6上方=多头, 下方=空头, 否则盘整

    Args:
        df_monthly: 月线DataFrame
        signal_date: 信号日期(可选), 如果提供则检查该日期时的月线状态

    Returns: 'up' / 'down' / 'range'
    """
    if len(df_monthly) < 10:
        return 'range'
    close = df_monthly['close']
    ma6 = close.rolling(6).mean()
    ma3 = close.rolling(3).mean()

    if signal_date is not None:
        # 找到signal_date对应的月线位置
        ts = pd.Timestamp(signal_date)
        mask = df_monthly.index <= ts
        if mask.sum() < 10:
            return 'range'
        c = close[mask].iloc[-1]
        m6 = ma6[mask].iloc[-1]
        m3 = ma3[mask].iloc[-1]
    else:
        c = close.iloc[-1]
        m6 = ma6.iloc[-1]
        m3 = ma3.iloc[-1]

    if c > m6 and m3 > m6:
        return 'up'
    elif c < m6 and m3 < m6:
        return 'down'
    else:
        return 'range'


def check_weekly_structure(df_weekly):
    """周线看结构: 缠论笔方向 + 中枢位置

    Returns: dict with 'stroke_dir', 'has_buy_signal', 'pivot_below'
    """
    if len(df_weekly) < 30:
        return {'stroke_dir': 'unknown', 'has_buy_signal': False, 'pivot_below': False}

    # 用简单笔检测 (swing high/low)
    n = len(df_weekly)
    high = df_weekly['high'].values
    low = df_weekly['low'].values
    close = df_weekly['close'].values

    # Swing检测: lookback=3
    swing_points = []
    for i in range(3, n - 3):
        is_high = all(high[i] >= high[j] for j in range(i-3, i+4) if j != i)
        is_low = all(low[i] <= low[j] for j in range(i-3, i+4) if j != i)
        if is_high:
            swing_points.append(('high', i, high[i]))
        elif is_low:
            swing_points.append(('low', i, low[i]))

    # 交替合并
    if len(swing_points) < 2:
        return {'stroke_dir': 'unknown', 'has_buy_signal': False, 'pivot_below': False}

    merged = [swing_points[0]]
    for pt in swing_points[1:]:
        last = merged[-1]
        if pt[0] != last[0]:
            merged.append(pt)
        else:
            if pt[0] == 'high' and pt[2] > last[2]:
                merged[-1] = pt
            elif pt[0] == 'low' and pt[2] < last[2]:
                merged[-1] = pt

    # 最近笔方向
    if len(merged) >= 2:
        last_pt = merged[-1]
        if last_pt[0] == 'high':
            stroke_dir = 'down'  # 到顶了, 向下笔
        else:
            stroke_dir = 'up'  # 到底了, 向上笔
    else:
        stroke_dir = 'unknown'

    # 简单中枢检测: 最近3笔
    strokes = []
    for i in range(len(merged) - 1):
        p1, p2 = merged[i], merged[i+1]
        if p1[0] == 'low' and p2[0] == 'high':
            strokes.append({'high': p2[2], 'low': p1[2]})
        elif p1[0] == 'high' and p2[0] == 'low':
            strokes.append({'high': p1[2], 'low': p2[2]})

    has_buy_signal = False
    if len(strokes) >= 3:
        # 最近中枢
        s1, s2, s3 = strokes[-3], strokes[-2], strokes[-1]
        zg = min(s1['high'], s2['high'], s3['high'])
        zd = max(s1['low'], s2['low'], s3['low'])
        if zg > zd:
            c = close[-1]
            # 价格在中枢下方或刚突破 = 买点区域
            has_buy_signal = c <= zg * 1.02

    return {
        'stroke_dir': stroke_dir,
        'has_buy_signal': has_buy_signal,
        'last_swing': merged[-2:] if len(merged) >= 2 else merged,
        'n_swing': len(merged),
    }


def detect_30min_2buy(df_30, window_start=None):
    """在30min上运行核心缠论管线，检测2买信号 (BuySellPointDetector)

    Args:
        df_30: 30分钟DataFrame
        window_start: 可选, 只处理此日期之后的数据(加速)

    接受: 2buy, 2buy_strong, class2buy, 2b3bbuy

    Returns: list of {
        'entry_idx': int,  'entry_price': float,
        'stop_price': float,  '1buy_low': float,
        'date': datetime, 'confidence': float, 'point_type': str
    }
    """
    # 裁剪数据窗口以加速 (保留足够的前置数据用于缠论分析)
    if window_start is not None:
        ts = pd.Timestamp(window_start)
        # 保留信号前200根bar (约2个月) 用于上下文
        start_mask = df_30.index < ts
        if start_mask.sum() > 200:
            cutoff = start_mask.sum() - 200
            df_30 = df_30.iloc[cutoff:]
        elif len(df_30) > 800:
            df_30 = df_30.iloc[-800:]

    n = len(df_30)
    if n < 60:
        return []

    try:
        k30 = KLine.from_dataframe(df_30, strict_mode=False)
        closes_30 = [k.close for k in k30.processed_data]
        highs_30 = [k.high for k in k30.processed_data]
        lows_30 = [k.low for k in k30.processed_data]
        f30 = FractalDetector(k30, confirm_required=False).get_fractals()
        if len(f30) < 4:
            return []
        s30 = StrokeGenerator(k30, f30, min_bars=3).get_strokes()
        if len(s30) < 3:
            return []
        p30 = PivotDetector(k30, s30).get_pivots()
        if not p30:
            return []
        m30 = MACD(pd.Series(df_30['close'].values))
        d30 = BuySellPointDetector(f30, s30, [], p30, macd=m30,
                                    closes=closes_30, highs=highs_30, lows=lows_30)
        buys_30, _ = d30.detect_all()
    except Exception:
        return []

    # 去重: 同index保留最高confidence
    seen = {}
    for b in buys_30:
        if b.index not in seen or b.confidence > seen[b.index].confidence:
            seen[b.index] = b

    # 筛选2买类型
    buy2_types = ('2buy', '2buy_strong', 'class2buy', '2b3bbuy')
    results = []
    for b in seen.values():
        if b.point_type not in buy2_types:
            continue
        idx = b.index
        if idx < 5 or idx >= n:
            continue

        entry_price = df_30['close'].iloc[idx]
        stop = b.stop_loss if b.stop_loss > 0 else entry_price * 0.92
        if stop >= entry_price:
            stop = entry_price * 0.92

        results.append({
            'entry_idx': idx,
            'entry_price': entry_price,
            'stop_price': stop,
            '1buy_low': stop,
            'date': df_30.index[idx],
            'confidence': b.confidence,
            'point_type': b.point_type,
            'reason': b.reason,
        })

    results.sort(key=lambda x: x['entry_idx'])
    return results


# ============ 日线级别信号 (核心缠论引擎) ============


def find_daily_buy_signals(code, df, buys_cache=None):
    """找日线级别1买信号 — 使用核心缠论引擎

    V5: 回归1买策略 + 更严格的过滤

    Args:
        buys_cache: 如果提供, 使用缓存的(buys, sells)避免重复运行管线

    Returns: list of {
        'buy_idx': int, 'buy_price': float, 'buy_low': float,
        'buy_date': datetime, 'buy_type': str, 'confidence': float
    }
    """
    n = len(df)
    if n < 120:
        return []

    try:
        if buys_cache is not None:
            buys, _ = buys_cache
        else:
            buys, _ = run_core_chanlun(df)
    except Exception:
        return []

    # 去重: 同index保留最高confidence
    seen = {}
    for b in buys:
        if b.index not in seen or b.confidence > seen[b.index].confidence:
            seen[b.index] = b

    kl = KLine.from_dataframe(df, strict_mode=False)
    lows = [k.low for k in kl.processed_data]

    results = []
    for b in seen.values():
        if b.point_type not in ('2buy', '3buy', 'quasi3buy'):
            continue
        idx = b.index
        if idx < 50 or idx >= n:
            continue
        if b.confidence < 0.5:
            continue

        # 止损逻辑按买点类型区分 (缠论原文)
        if b.point_type == '2buy':
            # 2买止损: 一买低点 (回踩不破前低)
            lookback = max(0, idx - 60)
            local_lows = lows[lookback:idx + 1] if idx + 1 <= len(lows) else lows[lookback:]
            buy_low = min(local_lows) if local_lows else df['low'].iloc[idx]
        else:
            # 3买止损: 中枢ZG (回踩不应进入中枢)
            lookback = max(0, idx - 30)
            local_lows = lows[lookback:idx + 1] if idx + 1 <= len(lows) else lows[lookback:]
            buy_low = min(local_lows) if local_lows else df['low'].iloc[idx]

        results.append({
            'buy_idx': idx,
            'buy_price': b.price,
            'buy_low': buy_low,
            'buy_date': df.index[idx],
            'buy_type': b.point_type,
            'confidence': b.confidence,
            'stop_loss': b.stop_loss,
            'reason': b.reason,
        })

    results.sort(key=lambda x: x['buy_idx'])
    return results


def run_core_chanlun(df):
    """运行一次完整核心缠论管线, 返回(buys, sells)信号列表

    避免重复运行管线 (每只股票只跑一次)
    """
    n = len(df)
    if n < 120:
        return [], []

    kl = KLine.from_dataframe(df, strict_mode=False)
    closes = [k.close for k in kl.processed_data]
    lows = [k.low for k in kl.processed_data]
    highs = [k.high for k in kl.processed_data]
    fr = FractalDetector(kl, confirm_required=False).get_fractals()
    if len(fr) < 4:
        return [], []
    st = StrokeGenerator(kl, fr, min_bars=3).get_strokes()
    if len(st) < 3:
        return [], []
    pv = PivotDetector(kl, st).get_pivots()
    if not pv:
        return [], []
    mc = MACD(pd.Series(df['close'].values))
    det = BuySellPointDetector(fr, st, [], pv, macd=mc,
                                closes=closes, highs=highs, lows=lows)
    buys, sells = det.detect_all()
    return buys, sells


def find_daily_2sell(code, df, sells_cache=None):
    """找出日线2卖位置 — 使用核心缠论引擎

    Args:
        sells_cache: 如果提供, 使用缓存的(buys, sells)避免重复运行管线

    Returns: set of idx where 2-sell occurs
    """
    n = len(df)
    if n < 120:
        return set()

    try:
        if sells_cache is not None:
            _, sells = sells_cache
        else:
            _, sells = run_core_chanlun(df)
    except Exception:
        return set()

    sell_2sell = set()
    for s in sells:
        if s.point_type in ('2sell',) and 0 <= s.index < n:
            sell_2sell.add(s.index)

    return sell_2sell


def compute_daily_macd_death_cross(df):
    """计算日线MACD死叉日期集合 (DIF下穿DEA)

    Returns: set of dates where DIF crosses below DEA
    """
    close = df['close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()

    death_cross = set()
    for i in range(1, len(df)):
        if dif.iloc[i-1] >= dea.iloc[i-1] and dif.iloc[i] < dea.iloc[i]:
            death_cross.add(df.index[i])

    return death_cross


def detect_daily_buy_divergence(df):
    """检测日线MACD底背驰位置 (价格新低 + MACD面积缩小)

    Returns: list of {'idx': int, 'low': float, 'date': Timestamp}
             按时间排序, 每个代表一个底背驰(1买)位置
    """
    close = df['close']
    low = df['low']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    n = len(df)

    # 找MACD金叉和死叉位置, 划分MACD段
    crosses = []
    for i in range(1, n):
        if dif.iloc[i-1] < dea.iloc[i-1] and dif.iloc[i] >= dea.iloc[i]:
            crosses.append(('golden', i))
        elif dif.iloc[i-1] >= dea.iloc[i-1] and dif.iloc[i] < dea.iloc[i]:
            crosses.append(('death', i))

    # 提取每段MACD绿柱(负值段)
    neg_segments = []  # (start_idx, end_idx, area, lowest_price_idx)
    in_neg = False
    seg_start = 0
    for i in range(n):
        if dif.iloc[i] < dea.iloc[i] and not in_neg:
            in_neg = True
            seg_start = i
        elif dif.iloc[i] >= dea.iloc[i] and in_neg:
            in_neg = False
            area = abs(sum(hist.iloc[seg_start:i].values))
            # 找这段最低价的idx
            seg_low_idx = low.iloc[seg_start:i].idxmin()
            seg_low_idx_pos = df.index.get_loc(seg_low_idx) if seg_low_idx in df.index else seg_start
            neg_segments.append((seg_start, i, area, seg_low_idx_pos))
    if in_neg:
        area = abs(sum(hist.iloc[seg_start:n].values))
        seg_low_idx = low.iloc[seg_start:n].idxmin()
        seg_low_idx_pos = df.index.get_loc(seg_low_idx) if seg_low_idx in df.index else seg_start
        neg_segments.append((seg_start, n, area, seg_low_idx_pos))

    # 底背驰: 连续两段负值, 后段价格更低但面积更小
    div_points = []
    for k in range(1, len(neg_segments)):
        prev_start, prev_end, prev_area, prev_low_idx = neg_segments[k-1]
        curr_start, curr_end, curr_area, curr_low_idx = neg_segments[k]

        prev_low = low.iloc[prev_low_idx]
        curr_low = low.iloc[curr_low_idx]

        # P0修复v2: c段(当前段)必须包含≥3个交易日K线（1.5天）
        if curr_end - curr_start < 2:
            continue

        if curr_low < prev_low and curr_area < prev_area * 0.8:  # 面积缩小20%以上
            div_points.append({
                'idx': curr_low_idx,
                'low': float(curr_low),
                'date': df.index[curr_low_idx],
                'area_ratio': curr_area / prev_area if prev_area > 0 else 1.0,
            })

    return div_points


def _build_pivots_from_strokes(strokes):
    """从笔序列构建中枢列表(用于走势类型分类)

    Returns: list of {zg, zd, start_idx, end_idx, bi_count}
    """
    if len(strokes) < 3:
        return []

    pivots = []
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i+1], strokes[i+2]
        highs = [max(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
        lows = [min(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
        zg = min(highs)
        zd = max(lows)
        if zg > zd:
            end_i = i + 2
            for j in range(i + 3, len(strokes)):
                sj_high = max(strokes[j]['start_val'], strokes[j]['end_val'])
                sj_low = min(strokes[j]['start_val'], strokes[j]['end_val'])
                if min(sj_high, zg) > max(sj_low, zd):
                    end_i = j
                else:
                    break
            bi_count = end_i - i + 1
            start_idx = min(strokes[k]['start_idx'] for k in range(i, end_i + 1))
            end_idx = max(strokes[k]['end_idx'] for k in range(i, end_i + 1))
            pivots.append({
                'zg': zg, 'zd': zd,
                'start_idx': start_idx, 'end_idx': end_idx,
                'bi_count': bi_count,
            })
            i = end_i + 1
        else:
            i += 1
    return pivots





# ============ 多周期回测主逻辑 ============

def fetch_sina_30min(symbol, datalen=2000):
    """从新浪获取30分钟历史数据(最多2000根, 约1年)"""
    import requests, json
    session = requests.Session()
    session.trust_env = False

    # 转换代码格式: 000001 → sz000001, 600519 → sh600519
    code = symbol.replace('.SZ', '').replace('.SH', '').replace('.BJ', '')
    if code.startswith('6'):
        sina_code = f'sh{code}'
    else:
        sina_code = f'sz{code}'

    url = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={sina_code}&scale=30&ma=no&datalen={datalen}'
    try:
        resp = session.get(url, timeout=15)
        resp.encoding = 'utf-8'
        try:
            data = json.loads(resp.text)
        except:
            data = eval(resp.text)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df.rename(columns={'day': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df.sort_index()
    except:
        return pd.DataFrame()


def run_daily_backtest(codes, params=None, start_date=None, conf_1buy=0.75):
    """纯日线级别回测 — 覆盖完整数据周期(4+年)

    策略: 月线方向 + 周线结构 + 日线买点 → 日线级别出场
    不依赖30分钟数据, 样本量更大, 统计意义更强
    """
    p = params or {}
    max_positions = p.get('max_positions', 5)
    trail_pct = p.get('trail_pct', 0.12)  # 移动止损回撤%
    max_hold_days = p.get('max_hold_days', 60)  # 最大持仓天数

    hs = HybridSource()
    print(f'=== 纯日线回测 V5 (月线+周线+日线) ===')
    print(f'规则: 月线方向 + 周线结构 + 日线1买/3买 → 日线出场')
    print(f'出场: 结构止损 + 移动止损 + 日线2卖 + 时间止损')
    print()

    # 加载日线数据
    print('[1] 加载日线数据...')
    daily_map = {}
    for code in codes:
        df = hs.get_kline(code, period='daily')
        if start_date and isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index >= pd.Timestamp(start_date)]
        if len(df) >= 200:
            daily_map[code] = df
    print(f'   日线: {len(daily_map)} 只')

    # 识别买点信号
    print('[2] 识别日线买点信号...')
    all_signals = []
    daily_cache = {}
    for code in daily_map:
        try:
            cache = run_core_chanlun(daily_map[code])
            daily_cache[code] = cache
        except Exception:
            daily_cache[code] = ([], [])

        buys, _ = cache if code in daily_cache else ([], [])
        sells_cache = daily_cache.get(code, ([], []))

        # 去重
        seen = {}
        for b in buys:
            if b.index not in seen or b.confidence > seen[b.index].confidence:
                seen[b.index] = b

        kl = KLine.from_dataframe(daily_map[code], strict_mode=False)
        lows = [k.low for k in kl.processed_data]

        for b in seen.values():
            if b.point_type not in ('1buy', '3buy', 'quasi3buy'):
                continue
            if b.confidence < 0.6:
                continue
            # 1买需要更高置信度, 减少假底部
            if b.point_type == '1buy' and b.confidence < conf_1buy:
                continue
            idx = b.index
            if idx < 50 or idx >= len(daily_map[code]):
                continue
            lookback = max(0, idx - 20)
            local_lows = lows[lookback:idx + 1] if idx + 1 <= len(lows) else lows[lookback:]
            buy_low = min(local_lows) if local_lows else daily_map[code]['low'].iloc[idx]

            all_signals.append({
                'code': code,
                'buy_idx': idx,
                'buy_price': b.price,
                'buy_low': buy_low,
                'buy_date': daily_map[code].index[idx],
                'buy_type': b.point_type,
                'confidence': b.confidence,
                'stop_loss': b.stop_loss,
                'reason': b.reason,
            })

    # 去重
    dedup = {}
    for s in all_signals:
        key = (s['code'], s['buy_date'])
        if key not in dedup or s['confidence'] > dedup[key]['confidence']:
            dedup[key] = s
    all_signals = list(dedup.values())
    type_counts = {}
    for s in all_signals:
        t = s['buy_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f'   找到 {len(all_signals)} 个信号: {type_counts}')

    # 月线+周线+市场环境过滤
    print('[3] 月线+周线+市场环境过滤...')

    # 市场环境
    market_env = None
    try:
        from indicator.market_environment import MarketEnvironment
        market_env = MarketEnvironment()
        if not market_env.records:
            market_env = None
    except Exception:
        market_env = None

    filtered = []
    for sig in all_signals:
        code = sig['code']
        df = daily_map[code]
        buy_date = sig['buy_date']
        buy_type = sig['buy_type']

        # 月线方向
        df_monthly = resample_to_monthly(df)
        month_dir = check_monthly_direction(df_monthly, signal_date=buy_date)
        if month_dir == 'down':
            continue

        # 周线结构
        df_weekly = resample_to_weekly(df)
        weekly_struct = check_weekly_structure(df_weekly)
        if weekly_struct['stroke_dir'] == 'down' and not weekly_struct['has_buy_signal']:
            continue

        # 市场环境
        if market_env is not None:
            d = str(buy_date)[:10].replace('-', '')
            if len(d) == 8:
                state = market_env.get_state_for_date(d)
                if state == 'BEAR':
                    continue

        # 3买成交量过滤: 突破日成交量 > 20日均量 (突破需要量能配合)
        if buy_type in ('3buy', 'quasi3buy'):
            buy_idx = sig['buy_idx']
            if 'volume' in df.columns and buy_idx >= 20:
                vol_today = df['volume'].iloc[buy_idx]
                vol_ma20 = df['volume'].iloc[buy_idx-20:buy_idx].mean()
                if vol_ma20 > 0 and vol_today < vol_ma20:
                    continue  # 突破无量, 跳过

        # 1买成交量过滤: 底部需要量能信号 — 放量恐慌抛售或缩量卖盘枯竭
        if buy_type == '1buy':
            buy_idx = sig['buy_idx']
            if 'volume' in df.columns and buy_idx >= 20:
                vol_today = df['volume'].iloc[buy_idx]
                vol_ma20 = df['volume'].iloc[buy_idx-20:buy_idx].mean()
                if vol_ma20 > 0:
                    vol_ratio = vol_today / vol_ma20
                    # 要求: 放量(>=1.2x恐慌抛售有承接) 或 缩量(<=0.7x卖盘枯竭)
                    if not (vol_ratio >= 1.2 or vol_ratio <= 0.7):
                        continue  # 中性量能, 底部信号不可靠

        filtered.append(sig)

    print(f'   过滤后: {len(filtered)} 个信号')

    # 日线级别逐bar交易模拟
    print('[4] 日线交易模拟...')
    trades = []

    for sig in filtered:
        code = sig['code']
        df = daily_map[code]
        entry_idx = sig['buy_idx']
        entry_price = sig['buy_price']
        entry_date = sig['buy_date']
        buy_type = sig['buy_type']

        # 止损设置 — 按买点类型差异化
        engine_stop = sig.get('stop_loss', 0)
        if buy_type in ('3buy', 'quasi3buy'):
            # 3买: 宽trail(18%), 标准止损(-10%)
            candidates = [entry_price * 0.90]
            if 0 < engine_stop < entry_price:
                candidates.append(engine_stop)
            stop_price = min(candidates)
            stop_price = max(stop_price, entry_price * 0.90)
        elif buy_type == '2buy':
            pass  # unused
        else:
            # 1买: 标准trail(12%), 标准止损(-10%)
            stop_price = sig['buy_low']
            if stop_price <= 0 or stop_price >= entry_price:
                stop_price = entry_price * 0.90
            stop_price = max(stop_price, entry_price * 0.90)

        # 找日线2卖信号
        sells_cache = daily_cache.get(code, ([], []))
        _, sells = sells_cache
        sell_dates = set()
        for s in sells:
            if s.point_type in ('2sell',) and s.index > entry_idx:
                sell_dates.add(s.index)

        # 逐日模拟 — 遍历到数据末尾
        # 3买用更宽的移动止损(突破信号需要空间), 1买用标准止损
        sig_trail = 0.18 if buy_type in ('3buy', 'quasi3buy') else trail_pct
        highest = entry_price
        exit_price = None
        exit_date = None
        exit_reason = ''
        n = len(df)

        for i in range(entry_idx + 1, n):
            price = df['close'].iloc[i]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            date = df.index[i]
            days_held = i - entry_idx

            profit_pct = (price - entry_price) / entry_price

            # 0. 3买保本保护: 盈利>8%后, 止损移到入场价
            if buy_type in ('3buy', 'quasi3buy') and profit_pct > 0.08:
                stop_price = max(stop_price, entry_price)

            # 1. 止损
            if low <= stop_price:
                exit_price = stop_price
                exit_date = date
                exit_reason = '保本止损' if stop_price >= entry_price else '止损'
                break

            # 2. 日线2卖
            if i in sell_dates:
                exit_price = price
                exit_date = date
                exit_reason = '日线2卖'
                break

            # 3. 移动止损 (盈利>5%后, 回撤>信号类型对应trail_pct平仓)
            highest = max(highest, high)
            if profit_pct > 0.05:
                drawdown = (highest - low) / highest
                if drawdown > sig_trail:
                    exit_price = price
                    exit_date = date
                    exit_reason = f'移动止损(回撤>{sig_trail*100:.0f}%)'
                    break

            # 4. 时间止损: 持仓超过max_hold_days且未盈利
            if days_held >= max_hold_days and profit_pct < 0.05:
                exit_price = price
                exit_date = date
                exit_reason = '时间止损'
                break

            # 5. 安全止损: 持仓超过120天, 从高点回撤>20%
            if days_held > 120 and profit_pct > 0.05:
                drawdown = (highest - low) / highest
                if drawdown > 0.20:
                    exit_price = price
                    exit_date = date
                    exit_reason = '安全止损(回撤>20%)'
                    break

        if exit_price is None:
            # 理论上不应到这里, 但安全起见
            exit_price = df['close'].iloc[-1]
            exit_date = df.index[-1]
            exit_reason = '数据截止'

        ret = (exit_price - entry_price) / entry_price
        trades.append({
            'code': code,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'return': ret,
            'buy_type': buy_type,
            'confidence': sig['confidence'],
            'hold_days': (exit_date - entry_date).days if hasattr(exit_date, '__sub__') else 0,
        })

    # 去重: 同一股票同一入场日
    dedup = {}
    for t in trades:
        key = (t['code'], t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10])
        if key not in dedup or t['return'] > dedup[key]['return']:
            dedup[key] = t
    trades = list(dedup.values())

    # 组合模拟 — 风险感知仓位管理
    trades.sort(key=lambda x: x['entry_date'])
    open_pos = {}
    capital = 1_000_000
    peak_capital = 1_000_000
    equity = []
    closed = []
    max_dd_threshold = 0.25  # 回撤超25%停止开仓

    for t in trades:
        # 平到期仓位
        expired = [c for c, p in open_pos.items() if p['trade']['exit_date'] <= t['entry_date']]
        for c in expired:
            pos = open_pos.pop(c)
            alloc = pos['alloc']
            pnl = alloc * pos['trade']['return']
            capital += alloc + pnl
            pos['trade']['pnl'] = pnl
            closed.append(pos['trade'])
            equity.append((pos['trade']['exit_date'], capital))
            peak_capital = max(peak_capital, capital)

        if t['code'] in open_pos or len(open_pos) >= max_positions:
            continue

        # 回撤控制: 当前回撤超阈值时不开新仓
        current_dd = (capital - peak_capital) / peak_capital
        if current_dd < -max_dd_threshold:
            continue

        # 仓位: 基于置信度调整 (conf 0.6=60%, 0.8=80%, 1.0=100%)
        conf = t.get('confidence', 0.7)
        base_alloc = capital / max_positions
        alloc = base_alloc * min(conf / 0.8, 1.0)
        alloc = max(alloc, base_alloc * 0.5)  # 至少半仓

        capital -= alloc
        open_pos[t['code']] = {'trade': t, 'alloc': alloc}

    # 平剩余
    for c, pos in open_pos.items():
        alloc = pos['alloc']
        pnl = alloc * pos['trade']['return']
        capital += alloc + pnl
        pos['trade']['pnl'] = pnl
        closed.append(pos['trade'])
        equity.append((pos['trade']['exit_date'], capital))

    final = closed

    # 统计
    returns = [t['return'] for t in final]
    wins = [t for t in final if t['return'] > 0]
    losses = [t for t in final if t['return'] <= 0]
    win_rate = len(wins) / len(final) if final else 0
    total_return = (capital - 1_000_000) / 1_000_000
    avg_win = np.mean([t['return'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['return'] for t in losses])) if losses else 0.01
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

    if len(equity) >= 2:
        eq_arr = np.array([e[1] for e in equity])
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        max_dd = dd.min()
        daily_ret = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0
    else:
        max_dd = 0
        sharpe = 0

    # 退出原因
    reasons = {}
    for t in final:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    # 按买点类型统计
    type_stats = {}
    for t in final:
        bt = t.get('buy_type', '?')
        if bt not in type_stats:
            type_stats[bt] = {'n': 0, 'wins': 0, 'ret_sum': 0}
        type_stats[bt]['n'] += 1
        if t['return'] > 0:
            type_stats[bt]['wins'] += 1
        type_stats[bt]['ret_sum'] += t['return']

    print()
    print('=' * 60)
    print(f'纯日线回测 ({len(final)}笔, 最多{max_positions}只)')
    print('=' * 60)
    print(f'总收益:     {total_return*100:+.2f}%')
    print(f'最终权益:   {capital:,.0f}')
    print(f'最大回撤:   {max_dd*100:.2f}%')
    print(f'Sharpe:     {sharpe:.2f}')
    print(f'盈亏比:     {profit_factor:.2f}')
    print(f'胜率:       {win_rate*100:.1f}% ({len(wins)}/{len(final)})')
    print(f'平均收益:   {np.mean(returns)*100:.2f}%')
    print(f'盈利均值:   {avg_win*100:.2f}%')
    print(f'亏损均值:   {-avg_loss*100:.2f}%')
    if returns:
        print(f'最大单笔盈: {max(returns)*100:.2f}%')
        print(f'最大单笔亏: {min(returns)*100:.2f}%')
    print()
    print('买点类型统计:')
    for bt, st in sorted(type_stats.items()):
        wr = st['wins'] / st['n'] * 100 if st['n'] > 0 else 0
        avg_r = st['ret_sum'] / st['n'] * 100 if st['n'] > 0 else 0
        print(f'  {bt}: {st["n"]}笔, 胜率{wr:.0f}%, 平均{avg_r:+.1f}%')
    print()
    print('退出原因:')
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f'  {r}: {c}笔')
    print()
    print('=== 交易明细 ===')
    for t in final:
        pnl_str = f' PnL={t.get("pnl", 0):+,.0f}' if t.get('pnl') else ''
        print(f'  {t["code"]} {t["buy_type"]} {t["entry_date"].strftime("%Y-%m-%d")}'
              f' 入:{t["entry_price"]:.2f} 出:{t["exit_price"]:.2f} {t["return"]*100:+.2f}%'
              f' {t["exit_reason"]} ({t.get("hold_days", "?")}天){pnl_str}')

    return final, {
        'total_return': total_return, 'max_drawdown': max_dd,
        'sharpe': sharpe, 'profit_factor': profit_factor,
        'win_rate': win_rate, 'n_trades': len(final),
        'avg_return': np.mean(returns) if returns else 0,
        'avg_win': avg_win, 'avg_loss': avg_loss,
    }


def run_mtf_backtest(codes, start_date='2025-01-01', end_date='2026-04-14',
                     params=None, verbose=True):
    # Strategy parameters (defaults = current best)
    p = params or {}
    trail_phase0 = p.get('trail_phase0', 0.20)
    trail_phase1 = p.get('trail_phase1', 0.12)
    breakeven_trigger = p.get('breakeven_trigger', 0.20)
    time_stop_bars = p.get('time_stop_bars', 600)
    max_positions = p.get('max_positions', 5)

    hs = HybridSource()

    print(f'=== CC15 四级别联立回测 V5 (月线→周线→日线3买→30min) ===')
    print(f'规则: 月线方向 + 周线结构 + 日线3买(趋势跟随) → 30min 2买确认')
    print(f'引擎: 核心缠论引擎 (BuySellPointDetector)')
    print()

    # 加载日线数据
    print('[1] 加载日线数据...')
    daily_map = {}
    for code in codes:
        df = hs.get_kline(code, period='daily')
        if len(df) >= 200:
            daily_map[code] = df
    print(f'   日线: {len(daily_map)} 只')

    # 过滤低成交额股票 (排除大盘死股, 近20日均成交额<1亿)
    min_turnover = p.get('min_turnover', 1e8)
    filtered = {}
    for code, df in daily_map.items():
        recent = df.tail(20)
        if 'amount' in recent.columns:
            avg_amt = recent['amount'].mean()
        elif 'volume' in recent.columns:
            avg_close = recent['close'].mean()
            avg_amt = recent['volume'].mean() * avg_close * 100
        else:
            avg_amt = 0
        if avg_amt >= min_turnover:
            filtered[code] = df
    if len(filtered) < len(daily_map):
        print(f'   成交额过滤(>={min_turnover/1e8:.0f}亿): {len(daily_map)} → {len(filtered)} 只')
    daily_map = filtered

    # 过滤低价股 (低价股结构不清晰, 缠论识别准确率低)
    min_price = p.get('min_price', 10.0)
    price_filtered = {}
    for code, df in daily_map.items():
        last_close = df['close'].iloc[-1]
        if last_close >= min_price:
            price_filtered[code] = df
    if len(price_filtered) < len(daily_map):
        print(f'   低价股过滤(>={min_price:.0f}元): {len(daily_map)} → {len(price_filtered)} 只')
    daily_map = price_filtered

    # 找日线3买信号 (使用核心缠论引擎, 每只股票只跑一次管线)
    print('[2] 识别日线3买信号 (核心缠论引擎)...')
    all_buys = []
    daily_2sell_map = {}
    daily_cache = {}  # code -> (buys, sells)
    for code in daily_map:
        try:
            cache = run_core_chanlun(daily_map[code])
            daily_cache[code] = cache
        except Exception:
            daily_cache[code] = ([], [])
        signals = find_daily_buy_signals(code, daily_map[code], buys_cache=daily_cache[code])
        for s in signals:
            s['code'] = code
            all_buys.append(s)
        daily_2sell_map[code] = find_daily_2sell(code, daily_map[code], sells_cache=daily_cache[code])

    type_counts = {}
    for s in all_buys:
        t = s['buy_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f'   找到 {len(all_buys)} 个日线买点信号: {type_counts}')

    # 去重: 同一只股票同一买点日只保留一个
    seen = set()
    unique_buys = []
    for item in all_buys:
        key = (item['code'], str(item.get('buy_date', '')))
        if key not in seen:
            seen.add(key)
            unique_buys.append(item)
    all_buys = unique_buys
    print(f'   去重后: {len(all_buys)} 个')

    # 加载30分钟数据: 优先Sina(覆盖1年), TDX补充
    print('[3] 加载30分钟数据(Sina优先)...')
    min30_map = {}
    n_sina = 0
    n_tdx = 0
    for code in codes:
        # 优先用Sina (覆盖约1年, 2000 bars, 真实数据)
        df_sina = fetch_sina_30min(code)
        if df_sina is not None and len(df_sina) >= 200:
            min30_map[code] = df_sina
            n_sina += 1
            continue
        # Fallback: TDX (少数股票有更长的真实数据)
        df_30 = hs.get_kline(code, period='30min')
        if df_30 is not None and len(df_30) >= 100:
            valid_mask = df_30.index.year >= 2020
            df_30_clean = df_30[valid_mask]
            if len(df_30_clean) >= 200:
                min30_map[code] = df_30_clean
                n_tdx += 1
    print(f'   30分钟数据: {len(min30_map)} 只 (Sina={n_sina}, TDX={n_tdx})')

    print('[4] 匹配交易...')
    trades = []

    # 市场环境过滤: 使用TDX本地数据 (MA250+斜率+动量)
    print('   加载市场环境(上证指数TDX本地)...')
    market_env = None
    market_signal_weights = None
    try:
        from indicator.market_environment import MarketEnvironment
        market_env = MarketEnvironment()
        if market_env.records:
            ms = market_env.get_market_state()
            market_signal_weights = market_env.SIGNAL_WEIGHTS
            # 统计回测期间的市场状态
            states = {'BULL': 0, 'NEUTRAL': 0, 'BEAR': 0}
            for item in all_buys:
                d = str(item.get('buy_date', ''))[:10].replace('-', '')
                if d and len(d) == 8:
                    s = market_env.get_state_for_date(d)
                    states[s] = states.get(s, 0) + 1
            print(f'   当前: {ms.state} | 信号日分布: BULL={states["BULL"]} NEUTRAL={states["NEUTRAL"]} BEAR={states["BEAR"]}')
        else:
            print('   未找到上证指数数据(跳过环境过滤)')
            market_env = None
    except Exception as e:
        print(f'   市场环境加载失败(跳过过滤): {e}')

    # 预计算日线MACD底背驰(用于止损后重新入场)
    daily_buy_div_map = {}
    for code in daily_map:
        daily_buy_div_map[code] = detect_daily_buy_divergence(daily_map[code])

    for item in all_buys:
        code = item['code']
        if code not in min30_map:
            continue
        df_30 = min30_map[code]

        df_daily = daily_map[code]
        buy_date = item['buy_date']
        buy_type = item['buy_type']

        # ========= 第一层: 月线定方向 (用信号日期时的月线状态) =========
        df_monthly = resample_to_monthly(df_daily)
        month_dir = check_monthly_direction(df_monthly, signal_date=buy_date)
        if month_dir == 'down':
            continue  # 月线空头, 不做多
        item['monthly_dir'] = month_dir

        # ========= 第二层: 周线看结构 =========
        df_weekly = resample_to_weekly(df_daily)
        weekly_struct = check_weekly_structure(df_weekly)
        # 周线向下笔 + 无买点信号 = 不入场
        if weekly_struct['stroke_dir'] == 'down' and not weekly_struct['has_buy_signal']:
            continue
        item['weekly_dir'] = weekly_struct['stroke_dir']

        # ========= 市场环境过滤 =========
        if market_env is not None:
            buy_date_str = str(buy_date)[:10].replace('-', '')
            if len(buy_date_str) == 8:
                state = market_env.get_state_for_date(buy_date_str)
                weights = market_env.get_signal_weights_for_date(buy_date_str)
                if state == 'BEAR':
                    continue
                item['market_state'] = state
                item['market_weight'] = weights.get(buy_type, 1.0)

        # ========= 30min数据检查 =========
        min_30_date = df_30.index[0]
        max_30_date = df_30.index[-1]
        if pd.Timestamp(buy_date) < min_30_date or pd.Timestamp(buy_date) > max_30_date:
            continue

        # ========= 第四层: 30min入场确认 (底分型+阳线+量≥5日均) =========
        # 缠论原文: 30min底分型确认即入场, 不要求30min出2买
        from lib.entry_confirm_30min import confirm_entry_30min
        confirm_result = confirm_entry_30min(df_30, pd.Timestamp(buy_date))

        if confirm_result.confirmed:
            # 三条件全满足: 底分型+阳线+放量 → 入场
            entry_idx = confirm_result.confirm_bar_idx
            entry_price = confirm_result.entry_price
            entry_date = df_30.index[entry_idx]
        else:
            # 无确认: 跳过 (过滤低质量信号)
            continue

        # 止损逻辑: 按缠论原文区分买点类型
        buy_type = item['buy_type']
        daily_buy_low = item['buy_low']

        if buy_type == '2buy':
            # 2买止损 = 一买低点 (回踩不破前低, 给大空间)
            stop_price = daily_buy_low
            stop_price = max(stop_price, entry_price * 0.90)  # 硬止损-10%
        else:
            # 3买止损 = 中枢ZG 或 一买低点, 取较远的给空间
            # 日线ATR(14)计算
            buy_idx_daily = item['buy_idx']
            atr_val = entry_price * 0.03
            if buy_idx_daily >= 14 and buy_idx_daily < len(df_daily):
                tr_sum = 0
                for j in range(max(1, buy_idx_daily - 13), buy_idx_daily + 1):
                    h = float(df_daily['high'].iloc[j])
                    l = float(df_daily['low'].iloc[j])
                    pc = float(df_daily['close'].iloc[j - 1])
                    tr_sum += max(h - l, abs(h - pc), abs(l - pc))
                atr_val = tr_sum / 14
            atr_stop = entry_price - atr_val * 0.75
            candidates = [atr_stop]
            if 0 < daily_buy_low < entry_price:
                candidates.append(daily_buy_low)
            stop_price = min(candidates)
            stop_price = max(stop_price, entry_price * 0.88)  # 硬止损-12%

        # 最小止损距离过滤: 止损距入场<2%说明空间不够, 跳过
        stop_distance = (entry_price - stop_price) / entry_price
        if stop_distance < 0.02:
            continue

        # ========= 支持重新入场的交易循环 =========
        # sub_trades: 同一个2买信号可能产生多次交易(止损→重新入场)
        sub_trades = []
        current_entry_idx = entry_idx
        current_entry_price = entry_price
        current_entry_date = entry_date
        current_stop_price = stop_price
        current_buy_date = buy_date

        max_reentry = 1  # 最多重新入场1次(防止背了又背)
        reentry_count = 0

        while current_entry_idx is not None and reentry_count <= max_reentry:
            # 在30分钟上运行完整的缠论分析(从入场点开始)
            strokes_30 = detect_strokes_30min(df_30)  # swing算法
            pivots_30 = detect_pivot_30min(strokes_30)
            buy_div_30, sell_div_30 = detect_macd_divergence_30min(df_30, strokes_30)

            # 日线级别卖出信号
            sell_2sell_idx = daily_2sell_map.get(code, set())

            # 找入场后的中枢, 并跟踪最新中枢(中枢上移)
            entry_pivot = None
            latest_pivot = None
            for pv in pivots_30:
                if pv['start_idx'] > current_entry_idx:
                    if entry_pivot is None:
                        entry_pivot = pv
                    latest_pivot = pv

            # ========= 两段止盈: 第一笔顶分型卖50% + 5笔后顶背离清仓 =========
            reduce_phase = 0  # 0=全仓, 1=已减50%
            reduce1_price = None
            reduce1_date = None
            first_top_fractal_done = False
            stroke_count_after_entry = 0
            highest_price = current_entry_price

            # 找入场后的笔 (只看入场点之后的)
            entry_strokes = [s for s in strokes_30 if s['end_idx'] > current_entry_idx]

            # 逐bar模拟退出
            exit_price = None
            exit_date = None
            exit_reason = ''
            was_stopped_out = False

            for i in range(current_entry_idx + 1, len(df_30)):
                price = df_30['close'].iloc[i]
                date_30 = df_30.index[i]
                profit_pct = (price - current_entry_price) / current_entry_price

                # --- 止损 (亏损端) ---

                # 1. 结构止损
                if price <= current_stop_price:
                    exit_price = max(current_stop_price, df_30['low'].iloc[i])
                    exit_date = date_30
                    if reentry_count > 0:
                        exit_reason = '止损(底背驰低点)'
                    else:
                        exit_reason = '止损(结构止损)'
                    was_stopped_out = True
                    break

                # 2. -12%硬止损(安全网)
                if profit_pct < -0.12:
                    exit_price = max(current_entry_price * 0.88, df_30['low'].iloc[i])
                    exit_date = date_30
                    exit_reason = '硬止损(-12%)'
                    was_stopped_out = True
                    break

                # --- 日线终极卖出 ---
                df_daily_index_dates = set()
                for idx_2s in sell_2sell_idx:
                    if 0 <= idx_2s < len(df_daily):
                        d = df_daily.index[idx_2s]
                        df_daily_index_dates.add(d)
                for d_2s in df_daily_index_dates:
                    if d_2s >= pd.Timestamp(current_buy_date) and abs((date_30 - d_2s).total_seconds()) < 86400:
                        exit_price = price
                        exit_date = date_30
                        exit_reason = '日线2卖'
                        break
                if exit_price:
                    break

                # --- 盈利端: 两段止盈 ---

                # 统计入场后已完成的笔数
                completed_strokes = [s for s in entry_strokes if s['end_idx'] <= i]
                stroke_count_after_entry = len(completed_strokes)

                highest_price = max(highest_price, price)

                # 阶段1: 第一笔向上笔完成(顶分型) → 减仓50%
                if reduce_phase == 0 and stroke_count_after_entry >= 1:
                    first_stroke = completed_strokes[0]
                    # 第一笔是向上笔(底→顶)
                    if first_stroke['start_type'] == 'bottom' and first_stroke['end_type'] == 'top':
                        # 等第一笔结束后确认顶分型
                        if i >= first_stroke['end_idx']:
                            reduce_phase = 1
                            # 取顶分型附近的价格作为减仓价
                            reduce1_price = df_30['close'].iloc[first_stroke['end_idx']]
                            reduce1_date = df_30.index[first_stroke['end_idx']]

                # 阶段2: 走完5笔 + 顶背离(1卖) → 清仓剩余50%
                if reduce_phase == 1 and stroke_count_after_entry >= 5:
                    # 检查最新的向下笔是否有顶背离
                    # 顶背离: 第5笔(向下笔)低点创新低但MACD面积缩小
                    if i in sell_div_30:
                        exit_price = price
                        exit_date = date_30
                        exit_reason = '顶背离清仓(5笔1卖)'
                        break

                # 盈利保护: 盈利>8%后止损移到入场价+3% (保护利润, 避免被震出)
                if profit_pct > 0.08 and current_stop_price < current_entry_price * 1.03:
                    current_stop_price = current_entry_price * 1.03

                # 盈利后移动止损: 从最高点回撤15%
                if profit_pct > 0.05 and reduce_phase == 0:
                    drawdown = (highest_price - price) / highest_price
                    if drawdown > 0.15:
                        exit_price = price
                        exit_date = date_30
                        exit_reason = '移动止损(回撤15%)'
                        break
                elif profit_pct > 0.05 and reduce_phase == 1:
                    drawdown = (highest_price - price) / highest_price
                    if drawdown > 0.10:
                        exit_price = price
                        exit_date = date_30
                        exit_reason = '移动止损(减仓后回撤10%)'
                        break

                # 时间止损: 600根30min(约75个交易日)无盈利
                if profit_pct <= 0 and i - current_entry_idx >= time_stop_bars:
                    exit_price = price
                    exit_date = date_30
                    exit_reason = '时间止损(微利)'
                    break

            if exit_price is None:
                exit_price = df_30['close'].iloc[-1]
                exit_date = df_30.index[-1]
                exit_reason = '未退出(用最新价)'

            # 计算加权收益 (两段: 50%第一笔顶 + 50%清仓)
            if reduce_phase == 0:
                total_ret = (exit_price - current_entry_price) / current_entry_price
            else:
                ret1 = (reduce1_price - current_entry_price) / current_entry_price
                ret2 = (exit_price - current_entry_price) / current_entry_price
                total_ret = 0.5 * ret1 + 0.5 * ret2

            sub_trades.append({
                'code': code,
                'entry_date': current_entry_date,
                'entry_price': current_entry_price,
                'stop_price': current_stop_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'return': total_ret,
                'reduce_phase': reduce_phase,
                'reduce1_price': reduce1_price,
                'reduce1_date': reduce1_date,
                'reentry': reentry_count,
                'hold_bars': df_30.index.get_loc(exit_date) - current_entry_idx if exit_date in df_30.index else 0,
                'confidence': item.get('confidence', 0.5),
                'buy_type': buy_type,
            })

            # ========= 止损后重新入场逻辑 =========
            if not was_stopped_out or reentry_count >= max_reentry:
                break

            # 寻找止损后的日线MACD底背驰
            div_points = daily_buy_div_map.get(code, [])
            # 只看止损日之后的底背驰
            stopout_date = exit_date
            valid_divs = [d for d in div_points if d['date'] > stopout_date]

            if not valid_divs:
                break

            # 取第一个底背驰
            next_div = valid_divs[0]
            div_date = next_div['date']

            # 检查30min数据是否覆盖这个日期
            if pd.Timestamp(div_date) > max_30_date:
                break

            # 重新入场: 在30min找背驰日之后的2买信号
            reentry_signals = detect_30min_2buy(df_30, window_start=div_date)
            valid_reentry = [s for s in reentry_signals
                             if s['date'] >= pd.Timestamp(div_date)
                             and s['date'] <= pd.Timestamp(div_date) + pd.Timedelta(days=15)]
            if not valid_reentry:
                break

            reentry_sig = valid_reentry[0]
            new_entry_idx = reentry_sig['entry_idx']
            new_entry_price = reentry_sig['entry_price']
            new_entry_date = reentry_sig['date']
            new_stop_price = reentry_sig['stop_price']

            # 止损价不能高于入场价
            if new_stop_price >= new_entry_price:
                break

            reentry_count += 1
            current_entry_idx = new_entry_idx
            current_entry_price = new_entry_price
            current_entry_date = new_entry_date
            current_stop_price = new_stop_price
            current_buy_date = div_date

        # 把sub_trades加入总trades
        trades.extend(sub_trades)

    # 去重: 同一股票同一入场日只保留一笔 (取return最高的)
    trade_dedup = {}
    for t in trades:
        key = (t['code'], t['entry_date'].strftime('%Y-%m-%d'))
        if key not in trade_dedup or t['return'] > trade_dedup[key]['return']:
            trade_dedup[key] = t
    trades = list(trade_dedup.values())

    # 置信度过滤: 只保留confidence >= 0.6的交易
    conf_threshold = p.get('conf_threshold', 0.6)
    before_conf = len(trades)
    trades = [t for t in trades if t.get('confidence', 0) >= conf_threshold]
    print(f'   置信度过滤(>={conf_threshold}): {before_conf} → {len(trades)} 笔')

    # 加载行业映射 (用于同行业去重)
    sector_map = {}
    for sp in ['chanlun_system/full_sector_map.json', 'chanlun_system/thshy_sector_map.json']:
        if os.path.exists(sp):
            with open(sp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sector_map = data.get('stock_to_sector', {})
            break

    # 汇总
    print(f'   有效交易: {len(trades)} 笔 (最终)')
    print()

    if not trades:
        print('无有效交易')
        return [], {'total_return': 0, 'max_drawdown': 0, 'sharpe': 0,
                    'profit_factor': 0, 'win_rate': 0, 'n_trades': 0,
                    'avg_return': 0, 'avg_win': 0, 'avg_loss': 0, 'params': p}

    # ========= 组合级别模拟 =========
    trades.sort(key=lambda x: x['entry_date'])

    open_positions = {}   # code -> {trade, alloc, sector}
    capital = 1_000_000
    peak_capital = 1_000_000
    equity_curve = []     # [(date, equity)]
    closed_trades = []
    skipped = 0
    skipped_sector = 0
    skipped_dd = 0

    for trade in trades:
        code = trade['code']
        entry_dt = trade['entry_date']
        exit_dt = trade['exit_date']

        # 1. 先平掉到期/已退出的仓位
        expired = [c for c, p in open_positions.items()
                   if p['trade']['exit_date'] <= entry_dt]
        for c in expired:
            pos = open_positions.pop(c)
            alloc = pos['alloc']
            pnl = alloc * pos['trade']['return']
            capital += alloc + pnl
            pos['trade']['pnl'] = pnl
            closed_trades.append(pos['trade'])
            equity_curve.append((pos['trade']['exit_date'], capital))
            peak_capital = max(peak_capital, capital)

        # 2. 回撤控制: >15%减半仓, >25%停止开仓
        current_dd = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0
        if current_dd < -0.25:
            skipped_dd += 1
            continue
        dd_scale = 0.5 if current_dd < -0.15 else 1.0

        # 3. 检查同股是否已在仓
        if code in open_positions:
            skipped += 1
            continue

        # 4. 检查持仓上限
        if len(open_positions) >= max_positions:
            skipped += 1
            continue

        # 5. 同行业去重: 同行业最多1只
        trade_sector = sector_map.get(code, '')
        if trade_sector:
            same_sector = [c for c, p in open_positions.items()
                           if p.get('sector') == trade_sector]
            if same_sector:
                skipped_sector += 1
                continue

        # 6. 分配资金: 等权 × 回撤缩仓
        alloc = capital / max_positions * dd_scale
        capital -= alloc
        open_positions[code] = {'trade': trade, 'alloc': alloc, 'sector': trade_sector}

    # 平掉剩余持仓
    for c, pos in open_positions.items():
        alloc = pos['alloc']
        pnl = alloc * pos['trade']['return']
        capital += alloc + pnl
        pos['trade']['pnl'] = pnl
        closed_trades.append(pos['trade'])
        equity_curve.append((pos['trade']['exit_date'], capital))

    final_trades = closed_trades

    # ========= 指标计算 =========
    returns = [t['return'] for t in final_trades]
    wins = [t for t in final_trades if t['return'] > 0]
    losses = [t for t in final_trades if t['return'] <= 0]
    win_rate = len(wins) / len(final_trades) if final_trades else 0

    total_return = (capital - 1_000_000) / 1_000_000

    # 权益曲线计算Sharpe和最大回撤
    if len(equity_curve) >= 2:
        eq_arr = np.array([e[1] for e in equity_curve])
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak
        max_drawdown = drawdown.min()
        # Sharpe (日频, 年化)
        daily_ret = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0
    else:
        max_drawdown = 0
        sharpe = 0

    # 盈亏比
    avg_win = np.mean([r['return'] for r in wins]) if wins else 0
    avg_loss = abs(np.mean([r['return'] for r in losses])) if losses else 0.01
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

    # 退出原因分布
    reasons = {}
    for t in final_trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    # 减仓阶段分布
    phase_counts = {0: 0, 1: 0, 2: 0}
    for t in final_trades:
        phase_counts[t['reduce_phase']] = phase_counts.get(t['reduce_phase'], 0) + 1

    print('=' * 60)
    print(f'CC15 MTF V3 Portfolio ({len(final_trades)}笔, 最多{max_positions}只)')
    print('=' * 60)
    print(f'总收益:     {total_return*100:+.2f}%')
    print(f'最终权益:   {capital:,.0f}')
    print(f'最大回撤:   {max_drawdown*100:.2f}%')
    print(f'Sharpe:     {sharpe:.2f}')
    print(f'盈亏比:     {profit_factor:.2f}')
    print(f'胜率:       {win_rate*100:.1f}% ({len(wins)}/{len(final_trades)})')
    print(f'平均收益:   {np.mean(returns)*100:.2f}%')
    print(f'盈利均值:   {avg_win*100:.2f}%')
    print(f'亏损均值:   {-avg_loss*100:.2f}%')
    print(f'最大单笔盈: {max(returns)*100:.2f}%')
    print(f'最大单笔亏: {min(returns)*100:.2f}%')
    if skipped:
        print(f'跳过(仓位满/重复): {skipped}笔')
    if skipped_sector:
        print(f'跳过(同行业): {skipped_sector}笔')
    if skipped_dd:
        print(f'跳过(回撤控制): {skipped_dd}笔')
    print()
    print(f'减仓阶段分布:')
    print(f'  未减仓(满仓进出): {phase_counts.get(0, 0)}笔')
    print(f'  减仓30%(仅背驰):  {phase_counts.get(1, 0)}笔')
    print(f'  减仓50%(背驰+顶分型): {phase_counts.get(2, 0)}笔')
    print()
    print('退出原因分布:')
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f'  {r}: {c}笔')

    print()
    print('=== 交易明细 ===')
    for t in final_trades:
        phase_str = ''
        if t['reduce_phase'] == 1:
            phase_str = f' -> 减30%@{t["reduce1_price"]:.2f}'
        elif t['reduce_phase'] == 2:
            phase_str = f' -> 减30%@{t["reduce1_price"]:.2f} 减20%@{t["reduce2_price"]:.2f}'
        pnl_str = f' PnL={t.get("pnl", 0):+,.0f}' if t.get('pnl') else ''
        print(f'  {t["code"]} {t["entry_date"].strftime("%Y-%m-%d %H:%M")}'
              f' 入:{t["entry_price"]:.2f} 止损:{t["stop_price"]:.2f}'
              f' 出:{t["exit_price"]:.2f} {t["return"]*100:+.2f}%'
              f' {t["exit_reason"]}{phase_str}{pnl_str}')

    metrics = {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'n_trades': len(final_trades),
        'avg_return': np.mean(returns) if returns else 0,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'params': p,
    }
    return final_trades, metrics


# ==================== 兼容层: scan_enhanced_v5 依赖 ====================

def find_daily_1buy_2buy(
    engine,
    code: str,
    df: pd.DataFrame,
) -> list:
    """
    返回日线1买/2买配对信号（兼容层）

    1买信号 → find_daily_buy_signals (3buy/quasi3buy)
    2买信号 → detect_30min_2buy

    Args:
        engine: SignalEngine 实例（当前未使用，保留接口）
        code:   股票代码
        df:     日线 DataFrame

    Returns:
        [{'1buy_idx', '1buy_price', '1buy_low', '2buy_idx', '2buy_price', '2buy_low'}, ...]
    """
    pairs = []

    buys = find_daily_buy_signals(code, df)

    for buy in buys:
        idx_1b = buy['buy_idx']
        if idx_1b >= len(df) - 5:
            continue

        df_after = df.iloc[idx_1b + 1:]
        if len(df_after) < 30:
            continue

        try:
            result_2b = detect_30min_2buy(df_after, min_confidence=0.5)
        except Exception:
            continue

        if not result_2b:
            continue

        r2 = result_2b[0]
        abs_2b_idx = idx_1b + 1 + r2['index']

        pairs.append({
            '1buy_idx': idx_1b,
            '1buy_price': buy['buy_price'],
            '1buy_low': buy['buy_low'],
            '2buy_idx': abs_2b_idx,
            '2buy_price': r2['price'],
            '2buy_low': r2.get('stop', r2['price'] * 0.97),
        })

    return pairs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stocks', type=int, default=50, help='Number of stocks to test')
    parser.add_argument('--all', action='store_true', help='Test all available stocks')
    parser.add_argument('--daily', action='store_true', help='Pure daily-level backtest (no 30min)')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    args = parser.parse_args()

    # 股票池: 沪深300成分股
    pool = [
        '000001','000002','000063','000333','000338','000425','000538','000568',
        '000625','000651','000661','000725','000768','000776','000783','000786',
        '000858','000895','000938','000963','000977',
        '002001','002007','002008','002024','002027','002049','002120',
        '002415','002460','002475','002500','002555','002594','002600',
        '002601','002714','002736',
        '300003','300014','300015','300033','300059','300122','300124',
        '300308','300347','300408','300413','300433','300496','300502',
        '300750','300760',
        '600009','600016','600019','600025','600028','600029',
        '600030','600031','600036','600048','600050','600061','600085','600089',
        '600104','600109','600111','600115','600132','600150','600153','600160',
        '600176','600196','600208','600219','600276','600282','600298','600309',
        '600325','600332','600346','600352','600362','600369','600372','600383',
        '600390','600406','600415','600426','600433','600436','600438','600460',
        '600519','600535','600547','600570','600588','600589',
        '600595','600600','600606','600623','600637',
        '600655','600660','600669','600674','600690','600702','600703','600704',
        '600718','600732','600733','600739','600741',
        '600745','600748','600754','600755','600763','600779',
        '600782','600786','600787','600795',
        '600801','600803','600809',
        '600823','600837','600845','600848','600854','600859',
        '600862','600867','600872','600875','600879','600884','600885','600887',
        '600893','600895','600900','600901','600903','600908','600909','600918',
        '600919','600926','600933','600941','600958',
        '600989','600993','600995','600998','600999',
        '601006','601009','601012','601015',
        '601016','601018','601021','601038',
        '601056','601060',
        '601066','601069','601077','601079',
        '601088','601099',
        '601100','601101','601108','601111',
        '601117','601118','601126','601127',
        '601138','601155','601156',
        '601162','601166','601168','601169',
        '601177','601179','601182',
        '601185','601186','601198',
        '601199','601200','601203',
        '601205','601206','601208','601211',
        '601212','601216','601217',
        '601221','601225','601226','601228',
        '601229','601231','601233',
        '601236','601238','601239',
        '601258','601277','601279',
        '601288','601298','601311','601318','601319',
        '601326','601328','601336',
        '601348','601360','601368','601369','601377',
        '601390','601399','601600','601601','601606','601607',
        '601611','601615','601618',
        '601628','601633','601636','601637','601641',
        '601643','601648','601658','601660',
        '601664','601665','601666','601668','601669',
        '601677','601678','601685','601686','601689',
        '601696','601698','601699','601700',
        '601717','601718','601727','601728',
        '601731','601732','601733','601734',
        '601739','601746','601755','601766',
        '601777','601778','601788','601789','601790',
        '601799','601800','601808',
        '601816','601818','601820','601825',
        '601827','601828','601833','601838',
        '601839','601857','601858','601860',
        '601862','601865','601866','601869',
        '601872','601875','601877','601878',
        '601880','601881','601882','601886',
        '601888','601889','601890','601893',
        '601895','601896','601898','601899',
        '601900','601901','601906','601908',
        '601912','601916','601919','601921',
        '601928','601929','601933','601939',
        '601952','601956','601958','601963',
        '601965','601968','601969','601971',
        '601975','601976','601978','601989',
        '601991','601992','601995','601996',
        '601997','601998','601999',
    ]

    if args.all:
        codes = pool
    else:
        # Select N stocks from pool
        codes = pool[:args.stocks]

    if args.daily:
        trades, metrics = run_daily_backtest(codes, start_date=args.start)
    else:
        trades, metrics = run_mtf_backtest(codes)
    print()
    print('=== 回测结果 ===')
    for k, v in metrics.items():
        if k != 'params':
            print(f'  {k}: {v}')
