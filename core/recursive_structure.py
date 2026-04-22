"""
多级别递归构建器

将笔视为虚拟K线，递归调用 分型→笔→线段→中枢 流程，
自动构建多级别缠论结构。

核心思路来自通达信GSKZ指标的递归模式：
Level 0: 原始K线 → 分型 → 笔 → 线段 → 中枢
Level 1: 笔作为虚拟K线 → 分型 → 笔(线段级) → 线段(高级别)
Level 2: 线段级笔作为虚拟K线 → ...
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .kline import KLine, KLineData
from .fractal import Fractal, FractalDetector
from .stroke import Stroke, StrokeGenerator
from .segment import Segment, SegmentGenerator
from .pivot import Pivot, PivotDetector, PivotLevel


def stroke_to_virtual_kline(strokes: List[Stroke]) -> KLine:
    """
    将笔列表转换为虚拟K线对象。

    每一笔生成一根虚拟K线：
    - open = 笔的起始值
    - close = 笔的结束值
    - high = 笔的最高价
    - low = 笔的最低价
    - datetime = 笔的起始时间
    - volume = 笔内K线成交量之和

    Args:
        strokes: 笔列表

    缠论原文：高级别的虚拟K线仍需处理包含关系。
    核心逻辑（包含→分型→笔→线段→中枢）在每个级别都应完整执行。

    Returns:
        虚拟KLine对象（strict_mode=True，包含关系处理启用）
    """
    from datetime import datetime

    virtual_data = []
    for s in strokes:
        volume = sum(k.volume for k in s.bars) if s.bars else 0
        amount = sum(k.amount for k in s.bars) if s.bars else 0

        virtual_data.append(KLineData(
            datetime=s.start_datetime,
            open=s.start_value,
            high=s.high,
            low=s.low,
            close=s.end_value,
            volume=volume,
            amount=amount,
        ))

    return KLine(virtual_data, strict_mode=True)


@dataclass
class LevelResult:
    """
    单一级别的缠论分析结果

    Attributes:
        level: 级别编号（0=原始K线级, 1=笔级, 2=线段级...）
        kline: 该级别的K线数据
        fractals: 该级别的分型列表
        strokes: 该级别的笔列表
        segments: 该级别的线段列表
        pivots: 该级别的中枢列表
        index_map: 高级别索引 → 原始K线索引的映射
    """
    level: int
    kline: KLine
    fractals: List[Fractal]
    strokes: List[Stroke]
    segments: List[Segment]
    pivots: List[Pivot]
    index_map: Dict[int, int] = field(default_factory=dict)

    @property
    def level_name(self) -> str:
        """级别名称"""
        names = {0: 'K线级', 1: '笔级', 2: '线段级', 3: '中枢级'}
        return names.get(self.level, f'Level {self.level}')

    def to_dict(self) -> dict:
        return {
            'level': self.level,
            'level_name': self.level_name,
            'kline_count': len(self.kline),
            'fractal_count': len(self.fractals),
            'stroke_count': len(self.strokes),
            'segment_count': len(self.segments),
            'pivot_count': len(self.pivots),
        }


class RecursiveStructureBuilder:
    """
    多级别递归缠论结构构建器

    将缠论分析递归应用：
    1. 原始K线 → 分型 → 笔 → 线段 → 中枢 (Level 0)
    2. Level 0的笔 → 虚拟K线 → 分型 → 笔 → 线段 → 中枢 (Level 1)
    3. Level 1的笔 → 虚拟K线 → ... (Level 2)
    ...

    每一级别产出的"笔"等价于上一级别的"线段"，
    这与缠论中"笔的笔"概念一致。
    """

    def __init__(
        self,
        kline: KLine,
        max_levels: int = 3,
        min_bars_for_fractal: int = 3,
        min_bars_for_stroke: int = 3,
        pivot_level: PivotLevel = PivotLevel.DAY,
    ):
        """
        Args:
            kline: 原始K线数据
            max_levels: 最大递归级别数
            min_bars_for_fractal: 分型确认最小K线数（高级别用）
            min_bars_for_stroke: 笔生成最小K线数（高级别用）
            pivot_level: 中枢级别标识
        """
        self.kline = kline
        self.max_levels = max_levels
        self.min_bars_for_fractal = min_bars_for_fractal
        self.min_bars_for_stroke = min_bars_for_stroke
        self.pivot_level = pivot_level
        self._results: Dict[int, LevelResult] = {}

    def build(self) -> Dict[int, LevelResult]:
        """
        构建所有级别的缠论结构

        Returns:
            {级别编号: LevelResult} 字典
        """
        self._results = {}

        # Level 0: 原始K线级
        level0 = self._build_level(self.kline, level=0)
        level0.index_map = {i: i for i in range(len(self.kline))}
        self._results[0] = level0

        logger.info(
            f"Level 0 (K线级): "
            f"{len(level0.fractals)} 分型, "
            f"{len(level0.strokes)} 笔, "
            f"{len(level0.segments)} 线段, "
            f"{len(level0.pivots)} 中枢"
        )

        # Level 1+: 递归构建
        for lvl in range(1, self.max_levels + 1):
            prev = self._results[lvl - 1]
            if len(prev.strokes) < 5:
                logger.info(f"Level {lvl}: 笔数不足({len(prev.strokes)}), 停止递归")
                break

            virtual_kline = stroke_to_virtual_kline(prev.strokes)
            result = self._build_level(virtual_kline, level=lvl)

            # 建立索引映射：高级别索引 → 原始K线索引
            result.index_map = self._build_index_map(prev)

            self._results[lvl] = result

            logger.info(
                f"Level {lvl} ({result.level_name}): "
                f"{len(result.fractals)} 分型, "
                f"{len(result.strokes)} 笔, "
                f"{len(result.segments)} 线段, "
                f"{len(result.pivots)} 中枢"
            )

            # 如果高级别没有产出新的笔，停止递归
            if len(result.strokes) < 3:
                logger.info(f"Level {lvl}: 高级别笔数不足, 停止递归")
                break

        return self._results

    def _build_level(self, kline: KLine, level: int) -> LevelResult:
        """
        构建单个级别的缠论结构

        Args:
            kline: 该级别的K线数据
            level: 级别编号

        Returns:
            该级别的分析结果
        """
        # 分型检测
        fractal_detector = FractalDetector(kline, confirm_required=False)
        fractals = fractal_detector.get_fractals()

        # 笔生成（高级别用更小的min_bars）
        stroke_gen = StrokeGenerator(
            kline, fractals,
            min_bars=self.min_bars_for_stroke if level > 0 else 5
        )
        strokes = stroke_gen.get_strokes()

        # 线段生成
        segments = []
        if len(strokes) >= 3:
            seg_gen = SegmentGenerator(kline, strokes)
            segments = seg_gen.get_segments()

        # 中枢识别
        pivots = []
        if len(strokes) >= 3:
            pivot_detector = PivotDetector(kline, strokes, level=self.pivot_level)
            pivots = pivot_detector.get_pivots()

        return LevelResult(
            level=level,
            kline=kline,
            fractals=fractals,
            strokes=strokes,
            segments=segments,
            pivots=pivots,
        )

    def _build_index_map(self, prev_level: LevelResult) -> Dict[int, int]:
        """
        建立高级别索引到原始K线索引的映射

        Level N 的第 i 个元素（虚拟K线/笔）对应
        Level N-1 的第 i 个笔，该笔的 start_index 是原始K线索引。

        Args:
            prev_level: 上一级别的结果

        Returns:
            {高级别索引: 原始K线索引}
        """
        index_map = {}
        for i, stroke in enumerate(prev_level.strokes):
            # 递归追溯到 Level 0 的索引
            if prev_level.index_map:
                # 上一级已经有映射，继续追溯到 Level 0
                orig_idx = prev_level.index_map.get(stroke.start_index, stroke.start_index)
            else:
                orig_idx = stroke.start_index
            index_map[i] = orig_idx
        return index_map

    # ==================== 查询接口 ====================

    def get_level(self, level: int) -> Optional[LevelResult]:
        """获取指定级别的结果"""
        return self._results.get(level)

    def get_all_levels(self) -> Dict[int, LevelResult]:
        """获取所有级别结果"""
        return self._results.copy()

    def get_highest_pivots(self) -> List[Tuple[int, Pivot]]:
        """
        获取最高级别的中枢列表

        Returns:
            [(级别编号, 中枢对象)] 列表
        """
        if not self._results:
            return []

        max_level = max(self._results.keys())
        result = self._results[max_level]
        return [(max_level, p) for p in result.pivots]

    def map_to_original_index(self, level: int, index_in_level: int) -> int:
        """
        将高级别索引映射回原始K线索引

        Args:
            level: 级别编号
            index_in_level: 该级别内的索引

        Returns:
            原始K线索引
        """
        result = self._results.get(level)
        if result is None:
            return index_in_level

        mapped = result.index_map.get(index_in_level, index_in_level)

        # 递归映射直到 Level 0
        current_level = level
        while current_level > 0:
            current_level -= 1
            r = self._results.get(current_level)
            if r is None or not r.index_map:
                break
            mapped = r.index_map.get(mapped, mapped)

        return mapped

    def get_multi_level_pivots(self) -> Dict[int, List[Pivot]]:
        """
        获取所有级别的中枢

        Returns:
            {级别编号: 中枢列表}
        """
        return {
            level: result.pivots
            for level, result in self._results.items()
        }

    def find_pivot_at_original_index(self, original_index: int) -> List[Tuple[int, Pivot]]:
        """
        查找覆盖原始K线索引的所有级别中枢

        Args:
            original_index: 原始K线索引

        Returns:
            [(级别, 中枢)] 列表
        """
        found = []
        for level, result in self._results.items():
            for pivot in result.pivots:
                # 将中枢的start/end映射回原始索引
                start_orig = self.map_to_original_index(level, pivot.start_index)
                end_orig = self.map_to_original_index(level, pivot.end_index)
                if start_orig <= original_index <= end_orig:
                    found.append((level, pivot))
        return found

    def summary(self) -> str:
        """生成多级别结构摘要"""
        lines = ["=== 多级别缠论结构 ==="]
        for level, result in sorted(self._results.items()):
            lines.append(
                f"Level {level} ({result.level_name}): "
                f"K线={len(result.kline)}, "
                f"分型={len(result.fractals)}, "
                f"笔={len(result.strokes)}, "
                f"线段={len(result.segments)}, "
                f"中枢={len(result.pivots)}"
            )
        return "\n".join(lines)
