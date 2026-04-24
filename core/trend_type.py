"""
走势类型分类模块

缠论走势类型定义：
- 上涨走势：连续两个以上同级别中枢，且后一个中枢的ZG/ZD高于前一个（中枢上移）
- 下跌走势：连续两个以上同级别中枢，且后一个中枢的ZG/ZD低于前一个（中枢下移）
- 盘整走势：只有一个中枢，或多个中枢但有重叠（中枢不移动）

核心判断依据是中枢之间的ZG/ZD关系：
- 上涨：ZD_n > ZG_{n-1} 或 ZD_n > ZD_{n-1} 且 ZG_n > ZG_{n-1}
- 下跌：ZG_n < ZD_{n-1} 或 ZG_n < ZG_{n-1} 且 ZD_n < ZD_{n-1}
- 盘整：中枢之间有重叠
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from loguru import logger

from .pivot import Pivot


class TrendType(Enum):
    """走势类型"""
    UP = 'up'              # 上涨走势（中枢上移）
    DOWN = 'down'          # 下跌走势（中枢下移）
    CONSOLIDATION = 'consolidation'  # 盘整走势（中枢重叠或单一中枢）
    UNKNOWN = 'unknown'    # 数据不足


@dataclass
class TrendSegment:
    """
    走势段 — 一次完整的走势（从一个中枢到下一个中枢的过渡）

    Attributes:
        trend_type: 走势类型
        start_pivot: 起始中枢
        end_pivot: 结束中枢（None表示当前走势尚未完成）
        start_index: 走势起始位置
        end_index: 走势结束位置
        high: 走势最高价
        low: 走势最低价
        strength: 走势强度 (0-1)
    """
    trend_type: TrendType
    start_pivot: Pivot
    end_pivot: Optional[Pivot]
    start_index: int
    end_index: int
    high: float
    low: float
    strength: float = 0.0

    @property
    def is_up(self) -> bool:
        return self.trend_type == TrendType.UP

    @property
    def is_down(self) -> bool:
        return self.trend_type == TrendType.DOWN

    @property
    def is_consolidation(self) -> bool:
        return self.trend_type == TrendType.CONSOLIDATION


@dataclass
class TrendTypeResult:
    """
    走势类型分析结果

    Attributes:
        current_type: 当前走势类型
        trend_segments: 所有走势段
        pivot_count: 中枢总数
        last_pivot: 最近的中枢
        current_trend: 当前走势段
        trend_strength: 当前走势强度
        pivot_progression: 中枢演进列表 [(index, zg, zd, type)]
    """
    current_type: TrendType
    trend_segments: List[TrendSegment]
    pivot_count: int
    last_pivot: Optional[Pivot]
    current_trend: Optional[TrendSegment]
    trend_strength: float
    pivot_progression: List[Tuple[int, float, float, str]]

    @property
    def is_uptrend(self) -> bool:
        return self.current_type == TrendType.UP

    @property
    def is_downtrend(self) -> bool:
        return self.current_type == TrendType.DOWN

    @property
    def is_consolidation(self) -> bool:
        return self.current_type == TrendType.CONSOLIDATION

    @property
    def trend_label(self) -> str:
        labels = {
            TrendType.UP: '上涨',
            TrendType.DOWN: '下跌',
            TrendType.CONSOLIDATION: '盘整',
            TrendType.UNKNOWN: '未知',
        }
        return labels.get(self.current_type, '未知')

    def to_dict(self) -> dict:
        return {
            'trend_type': self.current_type.value,
            'trend_label': self.trend_label,
            'pivot_count': self.pivot_count,
            'trend_strength': round(self.trend_strength, 3),
            'pivot_progression': [
                {'idx': i, 'zg': round(zg, 2), 'zd': round(zd, 2), 'dir': d}
                for i, zg, zd, d in self.pivot_progression
            ],
        }


class TrendTypeClassifier:
    """
    走势类型分类器

    基于中枢序列的ZG/ZD关系判断走势类型。

    用法：
        pivots = PivotDetector(kline, strokes).get_pivots()
        classifier = TrendTypeClassifier(pivots)
        result = classifier.classify()
    """

    def __init__(self, pivots: List[Pivot]):
        self.pivots = pivots

    def classify(self) -> TrendTypeResult:
        """
        对中枢序列进行走势类型分类。

        Returns:
            TrendTypeResult
        """
        if len(self.pivots) < 2:
            return self._single_pivot_result()

        # 按时间排序
        sorted_pivots = sorted(self.pivots, key=lambda p: p.start_index)

        # 构建走势段
        trend_segments = self._build_trend_segments(sorted_pivots)

        # 判断当前走势类型
        current_type = self._determine_current_type(sorted_pivots, trend_segments)

        # 中枢演进
        progression = [
            (i, p.zg, p.zd, p.direction)
            for i, p in enumerate(sorted_pivots)
        ]

        # 走势强度
        strength = self._calc_trend_strength(sorted_pivots, current_type)

        # 当前走势段
        current_trend = trend_segments[-1] if trend_segments else None

        return TrendTypeResult(
            current_type=current_type,
            trend_segments=trend_segments,
            pivot_count=len(sorted_pivots),
            last_pivot=sorted_pivots[-1],
            current_trend=current_trend,
            trend_strength=strength,
            pivot_progression=progression,
        )

    def _single_pivot_result(self) -> TrendTypeResult:
        """单中枢结果 — 盘整"""
        pivot = self.pivots[0] if self.pivots else None
        return TrendTypeResult(
            current_type=TrendType.CONSOLIDATION,
            trend_segments=[],
            pivot_count=len(self.pivots),
            last_pivot=pivot,
            current_trend=None,
            trend_strength=0.0,
            pivot_progression=[
                (0, pivot.zg, pivot.zd, pivot.direction)
            ] if pivot else [],
        )

    def _build_trend_segments(self, pivots: List[Pivot]) -> List[TrendSegment]:
        """
        从相邻中枢对构建走势段。

        两个相邻中枢之间的关系决定了走势段类型：
        - 后ZD > 前ZG → 上涨（中枢完全不重叠，后高于前）
        - 后ZG < 前ZD → 下跌（中枢完全不重叠，后低于前）
        - 其他 → 盘整（中枢有重叠）
        """
        segments = []
        for i in range(len(pivots) - 1):
            p1, p2 = pivots[i], pivots[i + 1]
            rel = self._classify_pair(p1, p2)

            seg = TrendSegment(
                trend_type=rel,
                start_pivot=p1,
                end_pivot=p2,
                start_index=p1.start_index,
                end_index=p2.end_index,
                high=max(p1.gg, p2.gg),
                low=min(p1.dd, p2.dd),
                strength=self._calc_pair_strength(p1, p2, rel),
            )
            segments.append(seg)
        return segments

    def _classify_pair(self, p1: Pivot, p2: Pivot) -> TrendType:
        """
        分类两个中枢之间的关系。

        判断逻辑（缠论严格定义）：
        1. ZD2 > ZG1 → 上涨（后中枢完全在前中枢之上）
        2. ZG2 < ZD1 → 下跌（后中枢完全在前中枢之下）
        3. 其他 → 盘整（中枢有重叠）

        宽松模式：同时检查ZG/ZD是否都上移/下移
        """
        if p2.zd > p1.zg:
            return TrendType.UP
        if p2.zg < p1.zd:
            return TrendType.DOWN

        # 中枢有重叠，但检查ZG/ZD是否都在上移/下移（趋势中的回调重叠）
        if p2.zg > p1.zg and p2.zd > p1.zd:
            return TrendType.UP
        if p2.zg < p1.zg and p2.zd < p1.zd:
            return TrendType.DOWN

        return TrendType.CONSOLIDATION

    def _calc_pair_strength(self, p1: Pivot, p2: Pivot, trend: TrendType) -> float:
        """计算两个中枢之间的走势强度"""
        if trend == TrendType.UP:
            if p1.zd > 0:
                return (p2.zg - p1.zg) / p1.zd
        elif trend == TrendType.DOWN:
            if p1.zg > 0:
                return (p1.zd - p2.zd) / p1.zg
        return 0.0

    def _determine_current_type(self, pivots: List[Pivot],
                                 segments: List[TrendSegment]) -> TrendType:
        """
        判断当前整体走势类型。

        规则：
        1. 最近3个走势段同向 → 该方向
        2. 最近2个走势段同向 → 该方向（较弱）
        3. 最近走势段为盘整 → 盘整
        4. 走势段交替 → 盘整（无明确方向）
        """
        if not segments:
            return TrendType.CONSOLIDATION

        # 检查最近走势段
        recent = segments[-3:] if len(segments) >= 3 else segments

        up_count = sum(1 for s in recent if s.trend_type == TrendType.UP)
        down_count = sum(1 for s in recent if s.trend_type == TrendType.DOWN)
        cons_count = sum(1 for s in recent if s.trend_type == TrendType.CONSOLIDATION)

        # 最近一段的权重更高
        last = segments[-1]

        if cons_count == len(recent):
            return TrendType.CONSOLIDATION

        # 最近走势段决定大方向，但需要至少2段同向确认
        if last.trend_type in (TrendType.UP, TrendType.DOWN):
            same_dir = up_count if last.trend_type == TrendType.UP else down_count
            if same_dir >= 2:
                return last.trend_type
            # 只有一段同向但最近一段很强
            if same_dir == 1 and last.strength > 0.1:
                return last.trend_type

        # 最近一段是盘整
        if last.trend_type == TrendType.CONSOLIDATION:
            # 之前有明确趋势且盘整时间不长，保持原方向
            if len(segments) >= 2:
                prev = segments[-2]
                if prev.trend_type in (TrendType.UP, TrendType.DOWN):
                    # 短暂盘整不改变方向
                    cons_bars = last.end_index - last.start_index
                    if cons_bars < 30:
                        return prev.trend_type
            return TrendType.CONSOLIDATION

        return TrendType.CONSOLIDATION

    def _calc_trend_strength(self, pivots: List[Pivot],
                              trend: TrendType) -> float:
        """
        计算当前走势强度 (0-1)。

        因子：
        1. 中枢移动幅度：连续上移/下移的中枢越多越强
        2. 中枢间距离：距离越大越强
        3. 最近中枢的紧凑度：越紧凑后续突破越有力
        """
        if len(pivots) < 2 or trend == TrendType.CONSOLIDATION:
            return 0.0

        # 连续同向中枢数
        consecutive = 1
        for i in range(len(pivots) - 1, 0, -1):
            p1, p2 = pivots[i - 1], pivots[i]
            if trend == TrendType.UP and p2.zd > p1.zd:
                consecutive += 1
            elif trend == TrendType.DOWN and p2.zg < p1.zg:
                consecutive += 1
            else:
                break

        # 标准化: 1个=0.3, 2个=0.5, 3个=0.7, 4+=0.9
        consec_score = min(0.3 + consecutive * 0.2, 0.9)

        # 最近中枢紧凑度
        last_p = pivots[-1]
        if last_p.middle > 0:
            tightness = max(0.0, 1.0 - last_p.range_value / last_p.middle / 0.15) * 0.1
        else:
            tightness = 0.0

        return min(consec_score + tightness, 1.0)


def classify_trend_type(pivots: List[Pivot]) -> TrendTypeResult:
    """
    便捷函数：对中枢列表进行走势类型分类。
    支持Pivot对象和dict格式的中枢。

    Args:
        pivots: 中枢列表 (Pivot对象 或 dict)

    Returns:
        TrendTypeResult
    """
    if not pivots:
        return TrendTypeResult(
            current_type=TrendType.UNKNOWN,
            trend_segments=[],
            pivot_count=0,
            last_pivot=None,
            current_trend=None,
            trend_strength=0.0,
            pivot_progression=[],
        )

    # 兼容dict格式中枢
    if isinstance(pivots[0], dict):
        from .pivot import Pivot as PivotCls, PivotLevel
        converted = []
        for i, pv in enumerate(pivots):
            p = PivotCls(
                level=PivotLevel.DAY,
                start_index=pv.get('start_idx', pv.get('start_index', 0)),
                end_index=pv.get('end_idx', pv.get('end_index', 0)),
                high=pv.get('zg', pv.get('high', 0)),
                low=pv.get('zd', pv.get('low', 0)),
                zg=pv.get('zg', 0),
                zd=pv.get('zd', 0),
                gg=pv.get('gg', pv.get('zg', 0)),
                dd=pv.get('dd', pv.get('zd', 0)),
                direction=pv.get('direction', ''),
            )
            converted.append(p)
        pivots = converted

    classifier = TrendTypeClassifier(pivots)
    return classifier.classify()
    """
    便捷函数：对中枢序列进行走势类型分类。

    Args:
        pivots: 中枢列表

    Returns:
        TrendTypeResult
    """
    classifier = TrendTypeClassifier(pivots)
    return classifier.classify()
