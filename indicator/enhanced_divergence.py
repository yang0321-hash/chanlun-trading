"""
增强背离检测模块

改进MACD背离检测：
1. 多笔比较：回溯多笔，指数衰减加权
2. 动量上下文：判断DIF加速/减速趋势
3. 隐藏背离检测：趋势延续信号
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np

from .macd import MACD
from core.stroke import Stroke


class DivergenceType(Enum):
    """背离类型"""
    REGULAR = 'regular'         # 常规背离（反转信号）
    HIDDEN = 'hidden'           # 隐藏背离（趋势延续信号）
    MULTI_PERIOD = 'multi_period'  # 多笔综合背离


@dataclass
class DivergenceResult:
    """背离检测结果"""
    has_divergence: bool
    strength: float             # 背离强度 0-1
    divergence_type: DivergenceType
    momentum_context: str       # 'decelerating', 'accelerating', 'stable'
    compared_count: int = 1     # 比较了多少笔
    reason: str = ''


class EnhancedDivergenceDetector:
    """
    增强背离检测器

    相比MACD.check_divergence()的改进：
    - 多笔回溯（最多5笔），指数衰减加权
    - DIF动量上下文
    - 隐藏背离检测（趋势延续信号）
    """

    def __init__(self, macd: MACD, strokes: List[Stroke]):
        self.macd = macd
        self.strokes = strokes

    def detect_trend_divergence(
        self,
        direction: str,
        lookback_strokes: int = 5,
        decay_factor: float = 0.7
    ) -> DivergenceResult:
        """
        多笔背离检测

        Args:
            direction: 'up'检测顶背离, 'down'检测底背离
            lookback_strokes: 回溯笔数（最多）
            decay_factor: 衰减因子，越早的笔权重越低

        Returns:
            DivergenceResult
        """
        if direction == 'up':
            return self._detect_top_divergence(lookback_strokes, decay_factor)
        else:
            return self._detect_bottom_divergence(lookback_strokes, decay_factor)

    def _detect_top_divergence(
        self, lookback: int, decay: float
    ) -> DivergenceResult:
        """顶背离检测（向上笔序列）"""
        up_strokes = [s for s in self.strokes if s.is_up]
        if len(up_strokes) < 2:
            return self._no_divergence()

        # 取最近lookback笔
        recent = up_strokes[-lookback:]
        if len(recent) < 2:
            return self._no_divergence()

        last = recent[-1]
        prev = recent[-2]

        # === 常规背离检测 ===
        regular = self._check_regular_top(recent, decay)
        if regular.has_divergence:
            regular.momentum_context = self._get_momentum_context(recent, 'up')
            return regular

        # === 隐藏背离检测 ===
        hidden = self._check_hidden_top(recent, decay)
        if hidden.has_divergence:
            hidden.momentum_context = self._get_momentum_context(recent, 'up')
            return hidden

        # === 多笔趋势背离 ===
        if len(recent) >= 3:
            multi = self._check_multi_top(recent, decay)
            if multi.has_divergence:
                multi.momentum_context = self._get_momentum_context(recent, 'up')
                return multi

        return self._no_divergence()

    def _detect_bottom_divergence(
        self, lookback: int, decay: float
    ) -> DivergenceResult:
        """底背离检测（向下笔序列）"""
        down_strokes = [s for s in self.strokes if s.is_down]
        if len(down_strokes) < 2:
            return self._no_divergence()

        recent = down_strokes[-lookback:]
        if len(recent) < 2:
            return self._no_divergence()

        # === 常规背离检测 ===
        regular = self._check_regular_bottom(recent, decay)
        if regular.has_divergence:
            regular.momentum_context = self._get_momentum_context(recent, 'down')
            return regular

        # === 隐藏背离检测 ===
        hidden = self._check_hidden_bottom(recent, decay)
        if hidden.has_divergence:
            hidden.momentum_context = self._get_momentum_context(recent, 'down')
            return hidden

        # === 多笔趋势背离 ===
        if len(recent) >= 3:
            multi = self._check_multi_bottom(recent, decay)
            if multi.has_divergence:
                multi.momentum_context = self._get_momentum_context(recent, 'down')
                return multi

        return self._no_divergence()

    def _check_regular_top(
        self, strokes: List[Stroke], decay: float
    ) -> DivergenceResult:
        """
        常规顶背离：价格创新高，MACD(DIF/面积)不创新高

        多笔加权：不只比较相邻两笔，而是回溯多笔加权平均
        """
        last = strokes[-1]
        last_price_high = last.high
        last_dif = self._get_stroke_dif_peak(last, 'max')
        last_area = self._get_stroke_area(last, 'up')

        if last_dif is None or last_area is None:
            return self._no_divergence()

        # 多笔加权比较
        total_weight = 0.0
        weighted_dif_ratio = 0.0
        weighted_area_ratio = 0.0
        compared = 0

        for i, s in enumerate(strokes[:-1]):
            weight = decay ** (len(strokes) - 2 - i)
            prev_dif = self._get_stroke_dif_peak(s, 'max')
            prev_area = self._get_stroke_area(s, 'up')

            if prev_dif is None or prev_area is None:
                continue

            # 价格必须比当前笔低（新高）
            if s.high >= last_price_high:
                continue

            dif_ratio = prev_dif / last_dif if last_dif != 0 else 1.0
            area_ratio = prev_area / last_area if last_area != 0 else 1.0

            # DIF或面积更大 → 背离
            weighted_dif_ratio += (dif_ratio - 1.0) * weight
            weighted_area_ratio += (area_ratio - 1.0) * weight
            total_weight += weight
            compared += 1

        if total_weight == 0:
            return self._no_divergence()

        avg_dif_ratio = weighted_dif_ratio / total_weight
        avg_area_ratio = weighted_area_ratio / total_weight

        # DIF背离 或 面积背离（至少一个）
        dif_divergent = avg_dif_ratio > 0  # 前笔DIF更高
        area_divergent = avg_area_ratio > 0  # 前笔面积更大

        if dif_divergent or area_divergent:
            strength = 0.0
            if dif_divergent and area_divergent:
                strength = min((avg_dif_ratio + avg_area_ratio) / 2, 1.0)
            elif dif_divergent:
                strength = min(avg_dif_ratio * 0.7, 1.0)
            else:
                strength = min(avg_area_ratio * 0.7, 1.0)

            return DivergenceResult(
                has_divergence=True,
                strength=strength,
                divergence_type=DivergenceType.REGULAR,
                momentum_context='',
                compared_count=compared,
                reason=f'常规顶背离: DIF比率={avg_dif_ratio:.3f}, 面积比率={avg_area_ratio:.3f}, 比较{compared}笔'
            )

        return self._no_divergence()

    def _check_regular_bottom(
        self, strokes: List[Stroke], decay: float
    ) -> DivergenceResult:
        """常规底背离：价格创新低，MACD(DIF/面积)不创新低"""
        last = strokes[-1]
        last_price_low = last.low
        last_dif = self._get_stroke_dif_peak(last, 'min')
        last_area = self._get_stroke_area(last, 'down')

        if last_dif is None or last_area is None:
            return self._no_divergence()

        total_weight = 0.0
        weighted_dif_ratio = 0.0
        weighted_area_ratio = 0.0
        compared = 0

        for i, s in enumerate(strokes[:-1]):
            weight = decay ** (len(strokes) - 2 - i)
            prev_dif = self._get_stroke_dif_peak(s, 'min')
            prev_area = self._get_stroke_area(s, 'down')

            if prev_dif is None or prev_area is None:
                continue

            # 价格必须比当前笔高（新低）
            if s.low <= last_price_low:
                continue

            # 底背离：前笔DIF更低（更负）
            dif_ratio = (last_dif - prev_dif) / abs(prev_dif) if prev_dif != 0 else 0.0
            area_ratio = (last_area - prev_area) / prev_area if prev_area != 0 else 0.0

            weighted_dif_ratio += dif_ratio * weight
            weighted_area_ratio += area_ratio * weight
            total_weight += weight
            compared += 1

        if total_weight == 0:
            return self._no_divergence()

        avg_dif_ratio = weighted_dif_ratio / total_weight
        avg_area_ratio = weighted_area_ratio / total_weight

        # DIF背离：当前DIF比前笔高（没那么低）
        dif_divergent = avg_dif_ratio > 0
        area_divergent = avg_area_ratio > 0

        if dif_divergent or area_divergent:
            strength = 0.0
            if dif_divergent and area_divergent:
                strength = min((avg_dif_ratio + avg_area_ratio) / 2, 1.0)
            elif dif_divergent:
                strength = min(avg_dif_ratio * 0.7, 1.0)
            else:
                strength = min(avg_area_ratio * 0.7, 1.0)

            return DivergenceResult(
                has_divergence=True,
                strength=strength,
                divergence_type=DivergenceType.REGULAR,
                momentum_context='',
                compared_count=compared,
                reason=f'常规底背离: DIF比率={avg_dif_ratio:.3f}, 面积比率={avg_area_ratio:.3f}, 比较{compared}笔'
            )

        return self._no_divergence()

    def _check_hidden_top(
        self, strokes: List[Stroke], decay: float
    ) -> DivergenceResult:
        """
        隐藏顶背离：价格未创新高，但DIF创新高

        含义：虽然价格没创新高，但动能在增强 → 趋势可能延续
        这不是卖出信号，而是趋势强度的确认
        """
        last = strokes[-1]
        prev = strokes[-2]

        last_price_high = last.high
        prev_price_high = prev.high

        # 价格未创新高
        if last_price_high >= prev_price_high:
            return self._no_divergence()

        last_dif = self._get_stroke_dif_peak(last, 'max')
        prev_dif = self._get_stroke_dif_peak(prev, 'max')

        if last_dif is None or prev_dif is None:
            return self._no_divergence()

        # DIF创新高（但价格没有）→ 隐藏背离
        if last_dif > prev_dif and prev_dif != 0:
            strength = (last_dif - prev_dif) / abs(prev_dif)
            return DivergenceResult(
                has_divergence=True,
                strength=min(strength, 1.0),
                divergence_type=DivergenceType.HIDDEN,
                momentum_context='accelerating',
                compared_count=2,
                reason=f'隐藏顶背离: 价格未新高但DIF新高，趋势延续信号'
            )

        return self._no_divergence()

    def _check_hidden_bottom(
        self, strokes: List[Stroke], decay: float
    ) -> DivergenceResult:
        """
        隐藏底背离：价格未创新低，但DIF创新低

        含义：价格虽然没跌破前低，但做空动能在增强 → 下跌趋势可能延续
        """
        last = strokes[-1]
        prev = strokes[-2]

        last_price_low = last.low
        prev_price_low = prev.low

        # 价格未创新低
        if last_price_low <= prev_price_low:
            return self._no_divergence()

        last_dif = self._get_stroke_dif_peak(last, 'min')
        prev_dif = self._get_stroke_dif_peak(prev, 'min')

        if last_dif is None or prev_dif is None:
            return self._no_divergence()

        # DIF创新低（但价格没有）→ 隐藏背离
        if last_dif < prev_dif and prev_dif != 0:
            strength = (prev_dif - last_dif) / abs(prev_dif)
            return DivergenceResult(
                has_divergence=True,
                strength=min(strength, 1.0),
                divergence_type=DivergenceType.HIDDEN,
                momentum_context='decelerating',
                compared_count=2,
                reason=f'隐藏底背离: 价格未新低但DIF新低，趋势延续信号'
            )

        return self._no_divergence()

    def _check_multi_top(
        self, strokes: List[Stroke], decay: float
    ) -> DivergenceResult:
        """
        多笔趋势背离：连续多笔DIF递减 + 面积递减

        即使单笔比较不满足背离条件，连续3+笔的趋势性减弱也是背离信号
        """
        dif_peaks = []
        areas = []

        for s in strokes:
            dif = self._get_stroke_dif_peak(s, 'max')
            area = self._get_stroke_area(s, 'up')
            if dif is not None and area is not None:
                dif_peaks.append(dif)
                areas.append(area)

        if len(dif_peaks) < 3:
            return self._no_divergence()

        # 检查DIF是否连续递减
        dif_decreasing = all(dif_peaks[i] > dif_peaks[i + 1]
                             for i in range(len(dif_peaks) - 1))
        area_decreasing = all(areas[i] > areas[i + 1]
                              for i in range(len(areas) - 1))

        if dif_decreasing or area_decreasing:
            total_decline_dif = (dif_peaks[0] - dif_peaks[-1]) / abs(dif_peaks[0]) if dif_peaks[0] != 0 else 0
            total_decline_area = (areas[0] - areas[-1]) / areas[0] if areas[0] != 0 else 0

            strength = max(total_decline_dif, total_decline_area)
            return DivergenceResult(
                has_divergence=True,
                strength=min(strength, 1.0),
                divergence_type=DivergenceType.MULTI_PERIOD,
                momentum_context='decelerating',
                compared_count=len(dif_peaks),
                reason=f'多笔趋势背离: {len(dif_peaks)}笔DIF连续递减，趋势动能衰竭'
            )

        return self._no_divergence()

    def _check_multi_bottom(
        self, strokes: List[Stroke], decay: float
    ) -> DivergenceResult:
        """多笔趋势底背离：连续多笔DIF递增（不那么负）"""
        dif_peaks = []
        areas = []

        for s in strokes:
            dif = self._get_stroke_dif_peak(s, 'min')
            area = self._get_stroke_area(s, 'down')
            if dif is not None and area is not None:
                dif_peaks.append(dif)
                areas.append(area)

        if len(dif_peaks) < 3:
            return self._no_divergence()

        # 底背离：DIF连续递增（不那么负）
        dif_increasing = all(dif_peaks[i] < dif_peaks[i + 1]
                             for i in range(len(dif_peaks) - 1))
        area_decreasing = all(areas[i] > areas[i + 1]
                              for i in range(len(areas) - 1))

        if dif_increasing or area_decreasing:
            total_improve = (dif_peaks[-1] - dif_peaks[0]) / abs(dif_peaks[0]) if dif_peaks[0] != 0 else 0
            strength = min(total_improve, 1.0)
            return DivergenceResult(
                has_divergence=True,
                strength=strength,
                divergence_type=DivergenceType.MULTI_PERIOD,
                momentum_context='decelerating',
                compared_count=len(dif_peaks),
                reason=f'多笔趋势底背离: {len(dif_peaks)}笔DIF连续改善，下跌动能衰竭'
            )

        return self._no_divergence()

    def _get_momentum_context(
        self, strokes: List[Stroke], direction: str
    ) -> str:
        """
        判断DIF动量上下文

        连续3笔的DIF峰/谷趋势：
        - decelerating: DIF极值在递减 → 趋势减弱
        - accelerating: DIF极值在递增 → 趋势增强
        - stable: 变化不大
        """
        peaks = []
        for s in strokes[-3:]:
            if direction == 'up':
                dif = self._get_stroke_dif_peak(s, 'max')
            else:
                dif = self._get_stroke_dif_peak(s, 'min')
            if dif is not None:
                peaks.append(dif)

        if len(peaks) < 3:
            return 'stable'

        if direction == 'up':
            if peaks[-1] < peaks[-2] < peaks[-3]:
                return 'decelerating'
            elif peaks[-1] > peaks[-2] > peaks[-3]:
                return 'accelerating'
        else:
            if peaks[-1] > peaks[-2] > peaks[-3]:
                return 'decelerating'
            elif peaks[-1] < peaks[-2] < peaks[-3]:
                return 'accelerating'

        return 'stable'

    def _get_stroke_dif_peak(
        self, stroke: Stroke, mode: str
    ) -> Optional[float]:
        """获取一笔范围内的DIF极值"""
        if not self.macd.values:
            return None

        offset = self.macd._kline_offset
        start = max(0, min(stroke.start_index - offset, len(self.macd.values) - 1))
        end = max(0, min(stroke.end_index - offset, len(self.macd.values) - 1))
        if start >= end:
            return None

        difs = [self.macd.values[i].macd for i in range(start, end + 1)]
        if not difs:
            return None

        if mode == 'max':
            return max(difs)
        else:
            return min(difs)

    def _get_stroke_area(
        self, stroke: Stroke, direction: str
    ) -> Optional[float]:
        """获取一笔范围内的MACD柱面积"""
        if not self.macd.values:
            return None

        offset = self.macd._kline_offset
        start = max(0, min(stroke.start_index - offset, len(self.macd.values) - 1))
        end = max(0, min(stroke.end_index - offset, len(self.macd.values) - 1))
        if start >= end:
            return None

        area = 0.0
        for i in range(start, end + 1):
            h = self.macd.values[i].histogram
            if direction == 'up':
                area += max(0, h)
            else:
                area += max(0, -h)
        return area

    def _no_divergence(self) -> DivergenceResult:
        return DivergenceResult(
            has_divergence=False,
            strength=0.0,
            divergence_type=DivergenceType.REGULAR,
            momentum_context='stable',
            compared_count=0,
            reason='无背离'
        )
