"""
级别递归模块

缠论级别递归的核心思想：
低级别（如30min）的笔序列，经合并处理后，
可以视为高级别（如日线）的K线序列。
然后在这个虚拟K线序列上重新做分型/笔/线段/中枢，
得到更精确的高级别结构。

实际应用：
1. 30min笔 → 日线虚拟K线 → 日线笔（比直接用日线K线更精确）
2. 日线笔 → 周线虚拟K线 → 周线笔（比直接用周线K线更精确）

这样得到的"递归笔"比直接在高级别K线上画的笔更细腻，
能捕捉到直接用高级别K线遗漏的结构。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

from .kline import KLine, KLineData
from .fractal import Fractal, FractalDetector
from .stroke import Stroke, StrokeGenerator
from .segment import Segment, SegmentGenerator
from .pivot import Pivot, PivotDetector, PivotLevel


@dataclass
class LevelMapping:
    """
    级别映射结果

    Attributes:
        virtual_kline: 低级别笔映射成的高级虚拟K线
        recursive_fractals: 虚拟K线上的分型
        recursive_strokes: 虚拟K线上的笔（递归笔）
        recursive_segments: 虚拟K线上的线段
        recursive_pivots: 虚拟K线上的中枢
        source_level: 源级别名称
        target_level: 目标级别名称
    """
    virtual_kline: KLine
    recursive_fractals: List[Fractal]
    recursive_strokes: List[Stroke]
    recursive_segments: List[Segment]
    recursive_pivots: List[Pivot]
    source_level: str
    target_level: str

    @property
    def stroke_count(self) -> int:
        return len(self.recursive_strokes)

    @property
    def pivot_count(self) -> int:
        return len(self.recursive_pivots)


def strokes_to_virtual_kline(strokes: List[Stroke],
                              source_level: str = '30min',
                              target_level: str = 'daily') -> Optional[KLine]:
    """
    将低级别笔序列转换为高级别虚拟K线序列。

    每根低级别笔视为一根虚拟K线：
    - open = 笔的起始值
    - high = 笔的最高值
    - low = 笔的最低值
    - close = 笔的结束值
    - volume = 笔的成交量
    - datetime = 笔的结束时间

    Args:
        strokes: 低级别笔列表
        source_level: 源级别名称
        target_level: 目标级别名称

    Returns:
        KLine对象（包含虚拟K线数据），如果笔数不足则返回None
    """
    if len(strokes) < 5:
        return None

    virtual_bars = []
    for s in strokes:
        bar = KLineData(
            open=s.start_value,
            high=s.high,
            low=s.low,
            close=s.end_value,
            volume=s.volume if hasattr(s, 'volume') and s.volume else 0,
            datetime=s.end_datetime,
        )
        virtual_bars.append(bar)

    kline = KLine(virtual_bars, strict_mode=False)
    return kline


def recursive_level_analysis(
    strokes: List[Stroke],
    source_level: str = '30min',
    target_level: str = 'daily',
    target_pivot_level: PivotLevel = PivotLevel.DAY,
    min_bars: int = 3,
) -> Optional[LevelMapping]:
    """
    级别递归分析：将低级别笔映射为高级别结构。

    流程：
    1. 低级别笔 → 虚拟K线
    2. 虚拟K线 → 包含处理（KLine自动处理）
    3. 虚拟K线 → 分型 → 笔 → 线段 → 中枢

    Args:
        strokes: 低级别笔列表
        source_level: 源级别名称
        target_level: 目标级别名称
        target_pivot_level: 目标中枢级别
        min_bars: 最小笔K线数

    Returns:
        LevelMapping 或 None
    """
    virtual_kline = strokes_to_virtual_kline(strokes, source_level, target_level)
    if virtual_kline is None:
        return None

    try:
        # 分型检测
        fractals = FractalDetector(virtual_kline, confirm_required=False).get_fractals()
        if len(fractals) < 2:
            return None

        # 笔生成
        recursive_strokes = StrokeGenerator(
            virtual_kline, fractals, min_bars=min_bars
        ).get_strokes()
        if len(recursive_strokes) < 3:
            return None

        # 线段生成
        recursive_segments = SegmentGenerator(
            virtual_kline, recursive_strokes
        ).get_segments()

        # 中枢检测
        recursive_pivots = PivotDetector(
            virtual_kline, recursive_strokes, level=target_pivot_level
        ).get_pivots()

        return LevelMapping(
            virtual_kline=virtual_kline,
            recursive_fractals=fractals,
            recursive_strokes=recursive_strokes,
            recursive_segments=recursive_segments,
            recursive_pivots=recursive_pivots,
            source_level=source_level,
            target_level=target_level,
        )

    except Exception as e:
        logger.debug(f"级别递归分析失败 ({source_level}→{target_level}): {e}")
        return None


def compare_direct_vs_recursive(
    direct_strokes: List[Stroke],
    direct_pivots: List[Pivot],
    sub_level_strokes: List[Stroke],
) -> dict:
    """
    对比直接分析和递归分析的结果差异。

    用于验证级别递归是否提供额外的结构信息。

    Args:
        direct_strokes: 直接在高级别K线上生成的笔
        direct_pivots: 直接在高级别K线上生成的中枢
        sub_level_strokes: 低级别笔列表

    Returns:
        对比结果 dict
    """
    mapping = recursive_level_analysis(sub_level_strokes)
    if mapping is None:
        return {
            'recursive_available': False,
            'direct_strokes': len(direct_strokes),
            'direct_pivots': len(direct_pivots),
        }

    return {
        'recursive_available': True,
        'direct_strokes': len(direct_strokes),
        'direct_pivots': len(direct_pivots),
        'recursive_strokes': mapping.stroke_count,
        'recursive_pivots': mapping.pivot_count,
        'stroke_ratio': mapping.stroke_count / max(len(direct_strokes), 1),
        'pivot_ratio': mapping.pivot_count / max(len(direct_pivots), 1),
        'extra_structure': mapping.stroke_count > len(direct_strokes),
    }
