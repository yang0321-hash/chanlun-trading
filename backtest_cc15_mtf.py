#!/usr/bin/env python3
"""CC15 多周期策略回测 V3 — 截断亏损,让利润奔跑

规则:
  入场: 日线2买信号 → 等待30分钟第2个底分型买入
  止损(亏损端):
    - 日线1买最低点 / -15%硬止损(首次入场)
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
from datetime import datetime, timedelta
from data.hybrid_source import HybridSource


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
    """30分钟分型检测 — 兼容接口, 返回格式不变

    用于: 找入场底分型 (entry_fractal = bottom_after[1])
    """
    n = len(df)
    if n < 5:
        return []

    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    merged = _merge_inclusion_30min(high, low)

    # 分型检测
    fractals = []
    for j in range(1, len(merged) - 1):
        if (merged[j]['high'] > merged[j-1]['high'] and
                merged[j]['high'] > merged[j+1]['high']):
            fractals.append({
                'type': 'top', 'idx': merged[j]['idx'],
                'val': merged[j]['high'], 'midx': j
            })
        elif (merged[j]['low'] < merged[j-1]['low'] and
              merged[j]['low'] < merged[j+1]['low']):
            fractals.append({
                'type': 'bottom', 'idx': merged[j]['idx'],
                'val': merged[j]['low'], 'midx': j
            })

    # 顶底交替 + 间距
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


def detect_strokes_30min(df_or_fractals):
    """Swing算法构建笔 — 对齐TDX缠论

    Args:
        df_or_fractals: 可以传df(DataFrame)或fractals(list)
                        传df时用swing算法, 传fractals时用旧算法(兼容)
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

    # 新逻辑: Swing算法
    df = df_or_fractals
    n = len(df)
    if n < 5:
        return []

    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    # 包含处理
    merged = _merge_inclusion_30min(high, low)

    # Swing High/Low 检测
    lookback = 5  # 左右各5根确认, 对齐日线lookback
    nm = len(merged)
    swing_points = []  # (merged_idx, 'high'|'low', value, original_idx)

    for i in range(nm):
        is_hi = True
        is_lo = True
        lo = max(0, i - lookback)
        hi = min(nm - 1, i + lookback)
        for j in range(lo, hi + 1):
            if j == i:
                continue
            if merged[j]['high'] > merged[i]['high']:
                is_hi = False
            if merged[j]['low'] < merged[i]['low']:
                is_lo = False
            if not is_hi and not is_lo:
                break
        if is_hi:
            swing_points.append((i, 'high', merged[i]['high'], merged[i]['idx']))
        elif is_lo:
            swing_points.append((i, 'low', merged[i]['low'], merged[i]['idx']))

    if len(swing_points) < 2:
        return []

    # 合并相邻同类型 (保留更极端)
    sp = [swing_points[0]]
    for pt in swing_points[1:]:
        last = sp[-1]
        if pt[1] != last[1]:
            sp.append(pt)
        else:
            if pt[1] == 'high' and pt[2] > last[2]:
                sp[-1] = pt
            elif pt[1] == 'low' and pt[2] < last[2]:
                sp[-1] = pt

    # 最小间距过滤
    filtered = [sp[0]]
    for pt in sp[1:]:
        if pt[3] - filtered[-1][3] >= 5:
            filtered.append(pt)
        else:
            last = filtered[-1]
            if pt[1] == last[1]:
                if pt[1] == 'high' and pt[2] > last[2]:
                    filtered[-1] = pt
                elif pt[1] == 'low' and pt[2] < last[2]:
                    filtered[-1] = pt

    # 确保顶底交替
    final = [filtered[0]]
    for pt in filtered[1:]:
        if pt[1] != final[-1][1]:
            final.append(pt)
        else:
            if pt[1] == 'high' and pt[2] > final[-1][2]:
                final[-1] = pt
            elif pt[1] == 'low' and pt[2] < final[-1][2]:
                final[-1] = pt

    # 构建strokes (格式兼容旧接口)
    strokes = []
    for i in range(len(final) - 1):
        start = final[i]
        end = final[i + 1]
        if start[1] == 'low' and end[1] == 'high':
            start_type, end_type = 'bottom', 'top'
        elif start[1] == 'high' and end[1] == 'low':
            start_type, end_type = 'top', 'bottom'
        else:
            continue
        strokes.append({
            'start_idx': start[3], 'end_idx': end[3],
            'start_type': start_type, 'end_type': end_type,
            'start_val': start[2], 'end_val': end[2],
            'high': max(start[2], end[2]),
            'low': min(start[2], end[2]),
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
    for k in range(1, len(down_strokes)):
        curr = down_strokes[k]
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


# ============ 日线级别信号 ============

def run_daily_cc15(data_map):
    """运行CC15日线引擎, 返回信号和买卖点位置"""
    sys.path.insert(0, 'chanlun_unified')
    from signal_engine_cc15 import SignalEngine
    engine = SignalEngine()
    signals = engine.generate(data_map, live_mode=False)
    return engine, signals


def find_daily_1buy_2buy(engine, code, df):
    """找出每只股票的日线1买(底背驰)和2买位置

    Returns: list of {
        '1buy_idx': int, '1buy_price': float, '1buy_low': float,
        '2buy_idx': int, '2buy_price': float
    }
    """
    n = len(df)
    if n < 120:
        return []

    close = df['close']
    high = df['high']
    low = df['low']

    # 检测笔
    bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    # 底背驰(1买)
    buy_div_set, sell_div_set, _, _ = engine._compute_area_divergence(strokes, hist, n)

    # 2买
    buy_2buy_set = engine._detect_2buy(strokes, buy_div_set, n)

    # 构建1买信息: {idx: {'low': float, 'price': float}}
    buy1_info = {}
    for idx in buy_div_set:
        if 0 <= idx < n:
            # 找这个1买对应的下跌笔的最低点
            buy1_info[idx] = {
                'low': low.iloc[idx],
                'price': close.iloc[idx],
            }

    # 构建2买信息, 关联到前面的1买
    results = []
    # 找每个2买前面最近的1买
    sorted_1buy = sorted(buy1_info.keys())

    down_strokes = [s for s in strokes if s['start_type'] == 'top' and s['end_type'] == 'bottom']
    up_strokes = [s for s in strokes if s['start_type'] == 'bottom' and s['end_type'] == 'top']

    for idx_2buy in sorted(buy_2buy_set):
        if idx_2buy >= n:
            continue
        # 找前面最近的1买
        prev_1buys = [i for i in sorted_1buy if i < idx_2buy]
        if not prev_1buys:
            continue
        idx_1buy = prev_1buys[-1]

        results.append({
            '1buy_idx': idx_1buy,
            '1buy_price': buy1_info[idx_1buy]['price'],
            '1buy_low': buy1_info[idx_1buy]['low'],
            '2buy_idx': idx_2buy,
            '2buy_price': close.iloc[idx_2buy],
            '2buy_date': df.index[idx_2buy],
        })

    return results


def find_daily_2sell(engine, code, df):
    """找出日线2卖位置 (类比2买的镜像: 顶背驰后回调不创新高的卖点)

    Returns: set of idx where 2-sell occurs
    """
    n = len(df)
    if n < 120:
        return set()

    close = df['close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)

    # 顶背驰(1卖)
    _, sell_div_set, _, _ = engine._compute_area_divergence(strokes, hist, n)

    # 2卖: 1卖之后, 上涨笔高点不超过1卖前的高点
    up_strokes = [s for s in strokes if s['start_type'] == 'bottom' and s['end_type'] == 'top']
    sell_2sell = set()

    for idx_1sell in sell_div_set:
        if idx_1sell >= n:
            continue
        # 找包含1卖的上涨笔
        sell_up = None
        for s in up_strokes:
            if s['end_idx'] == idx_1sell or (s['start_idx'] <= idx_1sell <= s['end_idx']):
                sell_up = s
                break
        if sell_up is None:
            continue

        sell_high = sell_up['end_val']  # 1卖时上涨笔的高点

        # 找1卖之后所有上涨笔, 高点不超过sell_high → 2卖
        for s in up_strokes:
            if s['start_idx'] > idx_1sell and s['end_val'] < sell_high:
                sell_2sell.add(s['end_idx'])

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


def find_daily_1buy_2buy(engine, code, df):
    """找出每只股票的日线1买(底背驰)和2买位置

    Returns: list of {
        '1buy_idx': int, '1buy_price': float, '1buy_low': float,
        '2buy_idx': int, '2buy_price': float,
        'trend_type': str, 'trend_strength': float
    }
    """
    n = len(df)
    if n < 120:
        return []

    close = df['close']
    high = df['high']
    low = df['low']

    # 检测笔
    bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    # 底背驰(1买)
    buy_div_set, sell_div_set, _, _ = engine._compute_area_divergence(strokes, hist, n)

    # 2买
    buy_2buy_set = engine._detect_2buy(strokes, buy_div_set, n)

    # 构建1买信息: {idx: {'low': float, 'price': float}}
    buy1_info = {}
    for idx in buy_div_set:
        if 0 <= idx < n:
            buy1_info[idx] = {
                'low': low.iloc[idx],
                'price': close.iloc[idx],
            }

    # 走势类型分类
    trend_type_val = ''
    trend_strength_val = 0.0
    if len(strokes) >= 6:
        pivots_2buy = _build_pivots_from_strokes(strokes)
        if len(pivots_2buy) >= 2:
            from core.trend_type import classify_trend_type
            tr = classify_trend_type(pivots_2buy)
            trend_type_val = tr.current_type.value
            trend_strength_val = round(tr.trend_strength, 3)

    # 构建2买信息, 关联到前面的1买
    results = []
    sorted_1buy = sorted(buy1_info.keys())

    for idx_2buy in sorted(buy_2buy_set):
        if idx_2buy >= n:
            continue
        prev_1buys = [i for i in sorted_1buy if i < idx_2buy]
        if not prev_1buys:
            continue
        idx_1buy = prev_1buys[-1]

        results.append({
            '1buy_idx': idx_1buy,
            '1buy_price': buy1_info[idx_1buy]['price'],
            '1buy_low': buy1_info[idx_1buy]['low'],
            '2buy_idx': idx_2buy,
            '2buy_price': close.iloc[idx_2buy],
            '2buy_date': df.index[idx_2buy],
            'trend_type': trend_type_val,
            'trend_strength': trend_strength_val,
        })

    return results


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


def run_mtf_backtest(codes, start_date='2025-01-01', end_date='2026-04-14',
                     params=None, verbose=True):
    # Strategy parameters (defaults = current best)
    p = params or {}
    trail_phase0 = p.get('trail_phase0', 0.20)
    trail_phase1 = p.get('trail_phase1', 0.12)
    breakeven_trigger = p.get('breakeven_trigger', 0.20)
    time_stop_bars = p.get('time_stop_bars', 600)
    max_positions = p.get('max_positions', 10)

    hs = HybridSource()

    print(f'=== CC15 多周期策略回测 V3 (结构确认 + 让利润奔跑) ===')
    print(f'规则: 日线2买 → 30min第2底分型入场 → 中枢结构确认减仓 → 盈利不加时间止损')
    print()

    # 加载日线数据
    print('[1] 加载日线数据...')
    daily_map = {}
    for code in codes:
        df = hs.get_kline(code, period='daily')
        if len(df) >= 200:
            daily_map[code] = df
    print(f'   日线: {len(daily_map)} 只')

    # 运行CC15日线引擎
    print('[2] 运行CC15日线引擎...')
    engine, daily_signals = run_daily_cc15(daily_map)

    # 找日线1买/2买
    print('[3] 识别日线1买/2买 + 2卖...')
    all_2buys = []
    daily_2sell_map = {}   # code -> set of 2-sell idx
    for code in daily_map:
        pairs = find_daily_1buy_2buy(engine, code, daily_map[code])
        for p in pairs:
            p['code'] = code
            all_2buys.append(p)
        daily_2sell_map[code] = find_daily_2sell(engine, code, daily_map[code])
    print(f'   找到 {len(all_2buys)} 个日线2买信号')

    # 去重: 同一只股票同一2买日只保留一个
    seen = set()
    unique_2buys = []
    for item in all_2buys:
        key = (item['code'], str(item.get('2buy_date', '')))
        if key not in seen:
            seen.add(key)
            unique_2buys.append(item)
    all_2buys = unique_2buys
    print(f'   去重后: {len(all_2buys)} 个')

    # 加载30分钟数据(新浪源, ~1年)
    print('[4] 加载30分钟数据(新浪, ~2000 bars/只)...')
    min30_map = {}
    for code in codes:
        df_30 = fetch_sina_30min(code)
        if len(df_30) >= 100:
            min30_map[code] = df_30
            print(f'   {code}: {len(df_30)} bars ({df_30.index[0].strftime("%Y-%m-%d")} ~ {df_30.index[-1].strftime("%Y-%m-%d")})')
    print(f'   30分钟数据: {len(min30_map)} 只')

    print('[5] 匹配交易...')
    trades = []

    # 市场环境过滤: 仅在明确熊市(MA20<MA60)时过滤
    print('   加载市场环境(上证指数)...')
    market_bearish_dates = None
    try:
        import akshare as ak
        idx_df = ak.stock_zh_index_daily(symbol="sh000001")
        idx_df['date'] = pd.to_datetime(idx_df['date'])
        idx_df = idx_df.set_index('date').sort_index()
        idx_df['ma20'] = idx_df['close'].rolling(20).mean()
        idx_df['ma60'] = idx_df['close'].rolling(60).mean()
        # MA20 < MA60 = 明确熊市, 禁止入场
        market_bearish_dates = set(
            idx_df[idx_df['ma20'] < idx_df['ma60']].index.strftime('%Y-%m-%d').tolist()
        )
        # 统计熊市比例
        total_days = len(idx_df.iloc[-300:])
        bear_days = len([d for d in market_bearish_dates
                         if d >= idx_df.index[-300].strftime('%Y-%m-%d')])
        print(f'   近300天熊市日: {bear_days}/{total_days} ({bear_days*100//max(total_days,1)}%)')
    except Exception as e:
        print(f'   市场环境加载失败(跳过过滤): {e}')

    # 预计算日线MACD底背驰(用于止损后重新入场)
    daily_buy_div_map = {}
    for code in daily_map:
        daily_buy_div_map[code] = detect_daily_buy_divergence(daily_map[code])

    for item in all_2buys:
        code = item['code']
        if code not in min30_map:
            continue
        df_30 = min30_map[code]

        df_daily = daily_map[code]
        buy_date = item['2buy_date']

        # 市场环境过滤: 仅在明确熊市(MA20<MA60)时跳过
        if market_bearish_dates is not None:
            buy_date_str = pd.Timestamp(buy_date).strftime('%Y-%m-%d')
            if buy_date_str in market_bearish_dates:
                continue  # 明确熊市, 跳过此信号

        # 30分钟数据从什么时候开始
        min_30_date = df_30.index[0]
        max_30_date = df_30.index[-1]

        # 2买信号必须在30min数据范围内
        if pd.Timestamp(buy_date) < min_30_date or pd.Timestamp(buy_date) > max_30_date:
            continue

        # 止损价: 日线1买最低点
        stop_price = item['1buy_low']

        # 在30分钟级别找入场点: 2买信号日之后的第2个底分型
        fractals_30 = detect_fractals_30min(df_30)
        if not fractals_30:
            continue

        # 找2买日期之后的底分型
        bottom_after = [f for f in fractals_30
                        if f['type'] == 'bottom' and df_30.index[f['idx']] >= pd.Timestamp(buy_date)]

        if len(bottom_after) < 2:
            continue

        # 第2个底分型作为入场点
        entry_fractal = bottom_after[1]
        entry_idx = entry_fractal['idx']
        entry_price = entry_fractal['val']  # 底分型的val就是low

        # 量能过滤: 增强版 — 缩量转折+量价配合
        if 'volume' in df_30.columns:
            vol = df_30['volume'].values.astype(float)
            if entry_idx >= 5 and vol[entry_idx] > 0:
                avg_vol_5 = np.mean(vol[max(0, entry_idx-5):entry_idx])
                avg_vol_20 = np.mean(vol[max(0, entry_idx-20):entry_idx]) if entry_idx >= 20 else avg_vol_5
                entry_vol = vol[entry_idx]

                # 基本过滤: 成交量不能极低
                if avg_vol_5 > 0 and entry_vol < avg_vol_5 * 0.5:
                    continue

                # 缩量→放量转折加分（后续在评分中使用，这里只做极端过滤）
                # 极端缩量+价格不涨 = 无效信号
                if avg_vol_20 > 0 and entry_vol < avg_vol_20 * 0.3:
                    # 近5根K线价格也没有上涨
                    if entry_idx >= 5:
                        recent_return = (df_30['close'].iloc[entry_idx] - df_30['close'].iloc[entry_idx-5]) / df_30['close'].iloc[entry_idx-5]
                        if recent_return < 0:
                            continue  # 缩量下跌，跳过
        entry_date = df_30.index[entry_idx]

        # 止损检查: 入场价不能低于止损价太多
        if entry_price < stop_price:
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

            # ========= V3: 结构确认减仓 + 盈利不加时间止损 =========
            reduce_phase = 0
            reduce1_price = None
            reduce1_date = None
            reduce2_price = None
            reduce2_date = None
            broke_zg_bar = None  # 记录跌破ZG的bar位置
            highest_price = current_entry_price  # 移动止损追踪

            # 逐bar模拟退出
            exit_price = None
            exit_date = None
            exit_reason = ''
            was_stopped_out = False

            for i in range(current_entry_idx + 1, len(df_30)):
                price = df_30['close'].iloc[i]
                date_30 = df_30.index[i]
                profit_pct = (price - current_entry_price) / current_entry_price

                # --- 止损优先级最高(亏损端) ---

                # 1. 硬止损: 1买低点(首次) / 底背驰低点(重入场)
                if price <= current_stop_price:
                    exit_price = max(current_stop_price, df_30['low'].iloc[i])
                    exit_date = date_30
                    if reentry_count > 0:
                        exit_reason = '止损(底背驰低点)'
                    else:
                        exit_reason = '止损(1买低点)'
                    was_stopped_out = True
                    break

                # 2. -15%硬止损(仅首次入场)
                if reentry_count == 0 and profit_pct < -0.15:
                    exit_price = max(current_entry_price * 0.85, df_30['low'].iloc[i])
                    exit_date = date_30
                    exit_reason = '硬止损(-15%)'
                    break

                # --- 日线终极卖出 ---

                # 3. 日线2卖 → 全清
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

                # --- 30min结构确认卖出(分级) ---

                # 更新最新中枢(中枢上移追踪)
                for pv in pivots_30:
                    if pv['start_idx'] > current_entry_idx and pv['end_idx'] <= i:
                        if latest_pivot is None or pv['end_idx'] > latest_pivot['end_idx']:
                            latest_pivot = pv

                # 4. Phase 0→1: 形成中枢 + 跌破ZG → 减30%(结构性转弱)
                #    必须在ZD之前检查, 否则ZD全清会跳过ZG减仓
                if reduce_phase == 0 and latest_pivot and i > latest_pivot['end_idx']:
                    if price < latest_pivot['ZG']:
                        reduce_phase = 1
                        reduce1_price = price
                        reduce1_date = date_30
                        broke_zg_bar = i

                # 5. Phase 1→2: 跌破ZG后反弹无法收回 → 再减20%(确认趋势结束)
                if reduce_phase == 1 and broke_zg_bar is not None:
                    if i > broke_zg_bar + 1:
                        # 反弹到ZG以上 = 趋势恢复, 取消减仓
                        if latest_pivot and price > latest_pivot['ZG']:
                            reduce_phase = 0
                            reduce1_price = None
                            reduce1_date = None
                            broke_zg_bar = None
                        # 继续下跌或横盘超过8根bar没收回 → 确认减20%
                        elif i - broke_zg_bar >= 8:
                            reduce_phase = 2
                            reduce2_price = price
                            reduce2_date = date_30

                # 6. 跌破最新中枢ZD → 全清(止损线)
                #    在减仓检查之后执行, 确保ZG减仓有机会先触发
                if latest_pivot and i > latest_pivot['end_idx']:
                    if price < latest_pivot['ZD']:
                        exit_price = price
                        exit_date = date_30
                        exit_reason = '清仓(跌破中枢ZD)'
                        break

                # 7. 盈利保护: 盈利>breakeven_trigger后止损移到成本价
                if profit_pct > breakeven_trigger and current_stop_price < current_entry_price:
                    current_stop_price = current_entry_price

                # 8. 分级移动止损:
                #    Phase 0: 宽止损(20%回撤), 让ZG减仓有机会先触发
                #    Phase 1+: 紧止损(12%回撤), 结构已转弱, 加速保护
                highest_price = max(highest_price, price)
                if profit_pct > 0.05:
                    drawdown = (highest_price - price) / highest_price
                    trail_threshold = trail_phase0 if reduce_phase == 0 else trail_phase1
                    if drawdown > trail_threshold:
                        exit_price = price
                        exit_date = date_30
                        exit_reason = f'移动止损(P{reduce_phase}回撤)'
                        break
                else:
                    # 9. 亏损/微利仓位: 时间止损
                    if i - current_entry_idx >= time_stop_bars:
                        exit_price = price
                        exit_date = date_30
                        exit_reason = '时间止损(微利)'
                        break

            if exit_price is None:
                exit_price = df_30['close'].iloc[-1]
                exit_date = df_30.index[-1]
                exit_reason = '未退出(用最新价)'

            # 计算加权收益
            if reduce_phase == 0:
                total_ret = (exit_price - current_entry_price) / current_entry_price
            elif reduce_phase == 1:
                ret1 = (reduce1_price - current_entry_price) / current_entry_price
                ret2 = (exit_price - current_entry_price) / current_entry_price
                total_ret = 0.3 * ret1 + 0.7 * ret2
            elif reduce_phase == 2:
                ret1 = (reduce1_price - current_entry_price) / current_entry_price
                ret2 = (reduce2_price - current_entry_price) / current_entry_price
                ret3 = (exit_price - current_entry_price) / current_entry_price
                total_ret = 0.3 * ret1 + 0.2 * ret2 + 0.5 * ret3

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
                'reduce2_price': reduce2_price,
                'reduce2_date': reduce2_date,
                'reentry': reentry_count,
                'hold_bars': df_30.index.get_loc(exit_date) - current_entry_idx if exit_date in df_30.index else 0,
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

            # 在30min找底背驰日之后的第2个底分型 → 重新入场
            bottom_reentry = [f for f in fractals_30
                              if f['type'] == 'bottom' and df_30.index[f['idx']] >= pd.Timestamp(div_date)]

            if len(bottom_reentry) < 2:
                # 只有1个底分型, 用第一个也行
                if len(bottom_reentry) < 1:
                    break
                reentry_fractal = bottom_reentry[0]
            else:
                reentry_fractal = bottom_reentry[1]

            new_entry_idx = reentry_fractal['idx']
            new_entry_price = reentry_fractal['val']
            new_entry_date = df_30.index[new_entry_idx]

            # 新止损 = 底背驰最低点(跌破说明背驰是假的)
            new_stop_price = next_div['low']

            # 止损价不能高于入场价(理论上不应该, 保底处理)
            if new_stop_price >= new_entry_price:
                break  # 入场价已在背驰低点之下, 不合理, 放弃重入场

            reentry_count += 1
            current_entry_idx = new_entry_idx
            current_entry_price = new_entry_price
            current_entry_date = new_entry_date
            current_stop_price = new_stop_price
            current_buy_date = div_date

        # 把sub_trades加入总trades
        trades.extend(sub_trades)

    # 汇总
    print(f'   有效交易: {len(trades)} 笔')
    print()

    if not trades:
        print('无有效交易')
        return trades

    # ========= 组合级别模拟 =========
    trades.sort(key=lambda x: x['entry_date'])

    open_positions = {}   # code -> {trade, alloc}
    capital = 1_000_000
    equity_curve = []     # [(date, equity)]
    closed_trades = []
    skipped = 0

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

        # 2. 检查同股是否已在仓
        if code in open_positions:
            skipped += 1
            continue

        # 3. 检查持仓上限
        if len(open_positions) >= max_positions:
            skipped += 1
            continue

        # 4. 分配资金: 等权
        alloc = capital / max_positions
        capital -= alloc
        open_positions[code] = {'trade': trade, 'alloc': alloc}

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


if __name__ == '__main__':
    codes = ['000001', '000333', '000858', '002415', '002600',
             '600036', '600519', '601318', '300750', '300308',
             '601899', '300502', '002594', '600276', '000651',
             '603259', '601138', '300059', '600030', '601166']

    trades, metrics = run_mtf_backtest(codes)
