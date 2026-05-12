"""小级别转大级别检测 (V7.3策略手册)

定义: 子级别(30min)卖点 + 日线最后两笔幅度衰减>30% + 30min≥2中枢
定位: 提前预警趋势衰竭, 非自动卖出
调整: 卖点置信度+0.16 / 买点置信度-0.10

用法:
    from lib.small_to_large import detect_small_to_large
    result = detect_small_to_large(sell_points_30m, strokes_daily, pivots_30m)
    if result:
        confidence = apply_stl_adjust(confidence, '2buy', result)
"""
import numpy as np


def detect_small_to_large(sell_points_30m, strokes_daily, pivots_30m):
    """检测小转大信号

    Args:
        sell_points_30m: 30min卖点列表, 每项需有 'confidence' 字段
        strokes_daily: 日线笔列表, 每项需有 start_value/end_value 或 start_price/end_price
        pivots_30m: 30min中枢列表, 只看数量

    Returns:
        dict or None: 检测到时返回详情
    """
    # 条件1: 30min有高置信卖点
    if not sell_points_30m:
        return None
    recent_sell = sell_points_30m[-1]
    if recent_sell.get('confidence', 0) < 0.6:
        return None

    # 条件2: 日线最后两笔幅度衰减>30%
    if len(strokes_daily) < 3:
        return None

    last_2 = strokes_daily[-2]
    last_1 = strokes_daily[-1]

    amp_2 = _stroke_amplitude(last_2)
    amp_1 = _stroke_amplitude(last_1)

    if amp_2 == 0:
        return None

    decay_pct = (amp_2 - amp_1) / amp_2 * 100
    if decay_pct < 30:
        return None

    # 条件3: 30min≥2个中枢
    if len(pivots_30m) < 2:
        return None

    direction = '上涨衰竭' if _stroke_direction(last_1) == 'up' else '下跌衰竭'

    return {
        'detected': True,
        'direction': direction,
        'sell_30m_type': recent_sell.get('type', recent_sell.get('point_type', '')),
        'sell_30m_conf': recent_sell.get('confidence', 0),
        'decay_pct': round(decay_pct, 1),
        'pivot_count_30m': len(pivots_30m),
        'score_adjust': {'sell': +0.16, 'buy': -0.10},
        'msg': (f'小转大预警: 30min卖点 + 日线笔衰减{decay_pct:.0f}%'
                f' + {len(pivots_30m)}中枢 → {direction}'),
    }


def apply_stl_adjust(confidence, signal_type, stl_result):
    """应用小转大置信度调整

    Args:
        confidence: 原始置信度
        signal_type: '1buy'/'2buy'/'3buy'/'1sell'/'2sell'/...
        stl_result: detect_small_to_large() 返回值

    Returns:
        float: 调整后的置信度 [0, 1]
    """
    if not stl_result or not stl_result.get('detected'):
        return confidence

    adjust = stl_result['score_adjust']
    if 'sell' in signal_type.lower():
        adj = adjust['sell']
    else:
        adj = adjust['buy']

    return max(0.0, min(1.0, confidence + adj))


def _stroke_amplitude(stroke):
    """获取笔的幅度"""
    if hasattr(stroke, 'start_value'):
        return abs(stroke.end_value - stroke.start_value)
    if hasattr(stroke, 'start_price'):
        return abs(stroke.end_price - stroke.start_price)
    if isinstance(stroke, dict):
        sv = stroke.get('start_value', stroke.get('start_price', 0))
        ev = stroke.get('end_value', stroke.get('end_price', 0))
        return abs(ev - sv)
    return 0


def _stroke_direction(stroke):
    """获取笔的方向"""
    if hasattr(stroke, 'start_value'):
        return 'up' if stroke.end_value > stroke.start_value else 'down'
    if hasattr(stroke, 'start_price'):
        return 'up' if stroke.end_price > stroke.start_price else 'down'
    if isinstance(stroke, dict):
        sv = stroke.get('start_value', stroke.get('start_price', 0))
        ev = stroke.get('end_value', stroke.get('end_price', 0))
        return 'up' if ev > sv else 'down'
    return 'unknown'


def detect_daily_pen_decay(strokes_daily, min_decay_pct=30):
    """日线笔幅度衰减检测 (扫描器简化版, 不需要30min数据)

    只检测日线最后两笔幅度衰减, 用于扫描器预过滤。

    Args:
        strokes_daily: 日线笔列表
        min_decay_pct: 最小衰减百分比阈值 (默认30%)

    Returns:
        dict or None
    """
    if len(strokes_daily) < 3:
        return None

    last_2 = strokes_daily[-2]
    last_1 = strokes_daily[-1]

    amp_2 = _stroke_amplitude(last_2)
    amp_1 = _stroke_amplitude(last_1)

    if amp_2 == 0:
        return None

    decay_pct = (amp_2 - amp_1) / amp_2 * 100
    if decay_pct < min_decay_pct:
        return None

    direction = '上涨衰竭' if _stroke_direction(last_1) == 'up' else '下跌衰竭'

    return {
        'detected': True,
        'direction': direction,
        'decay_pct': round(decay_pct, 1),
        'score_adjust': {'sell': +0.16, 'buy': -0.10},
        'msg': f'笔衰减预警: 日线笔幅度衰减{decay_pct:.0f}% → {direction}',
    }
