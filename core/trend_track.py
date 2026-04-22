"""
趋势轨道与断点检测模块

基于中枢边界构建趋势轨道线（红轨/绿轨），
检测轨道断点，辅助买卖点置信度评估。

设计来源：通达信GSKZ指标中的红轨/绿轨逻辑：
- 绿轨 = 最近下跌中枢的上下沿延伸
- 红轨 = 最近上涨中枢的上下沿延伸
- 轨道断点 = 价格离开轨道一定距离后出现转折
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from loguru import logger

from .kline import KLine
from .stroke import Stroke
from .pivot import Pivot


class TrackDirection(Enum):
    """轨道方向"""
    UP = 'up'        # 红轨（上涨趋势）
    DOWN = 'down'    # 绿轨（下跌趋势）


class TrackBreakType(Enum):
    """轨道断点类型"""
    NONE = ''                  # 无断裂
    UPPER_BREAK = 'upper'      # 向上突破轨道
    LOWER_BREAK = 'lower'      # 向下跌破轨道


class TrendStatus(Enum):
    """趋势状态"""
    STRONG_UP = 'strong_up'      # 强上涨（红轨完好）
    WEAK_UP = 'weak_up'          # 弱上涨（红轨有断点迹象）
    STRONG_DOWN = 'strong_down'  # 强下跌（绿轨完好）
    WEAK_DOWN = 'weak_down'      # 弱下跌（绿轨有断点迹象）
    NEUTRAL = 'neutral'          # 震荡（无明显轨道）


@dataclass
class TrendTrack:
    """
    趋势轨道数据

    基于中枢边界构建的轨道线，用于判断趋势状态。

    Attributes:
        direction: 轨道方向（红轨=上涨, 绿轨=下跌）
        upper_line: 轨道上沿
        lower_line: 轨道下沿
        middle_line: 轨道中线
        start_index: 轨道起始K线索引
        end_index: 轨道结束K线索引（None表示仍在延伸）
        pivot: 关联的中枢
        broken: 轨道是否已断裂
        break_type: 断裂类型
        break_index: 断裂位置索引
    """
    direction: TrackDirection
    upper_line: float
    lower_line: float
    middle_line: float
    start_index: int
    end_index: Optional[int]
    pivot: Pivot
    broken: bool = False
    break_type: TrackBreakType = TrackBreakType.NONE
    break_index: Optional[int] = None

    @property
    def range_value(self) -> float:
        """轨道宽度"""
        return self.upper_line - self.lower_line

    @property
    def is_active(self) -> bool:
        """轨道是否仍然有效"""
        return not self.broken

    def contains_price(self, price: float) -> bool:
        """价格是否在轨道内"""
        return self.lower_line <= price <= self.upper_line

    def to_dict(self) -> dict:
        return {
            'direction': self.direction.value,
            'upper_line': self.upper_line,
            'lower_line': self.lower_line,
            'middle_line': self.middle_line,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'broken': self.broken,
            'break_type': self.break_type.value,
            'is_active': self.is_active,
        }


class TrendTrackDetector:
    """
    趋势轨道检测器

    从中枢结构构建趋势轨道，检测轨道断点。

    使用方式：
    1. 传入笔列表和中枢列表
    2. 调用 detect() 构建轨道
    3. 调用 get_current_track() 获取当前有效轨道
    4. 调用 get_trend_status() 获取当前趋势状态
    """

    def __init__(
        self,
        strokes: List[Stroke],
        pivots: List[Pivot],
        min_break_distance: int = 3,
    ):
        """
        Args:
            strokes: 笔列表
            pivots: 中枢列表
            min_break_distance: 轨道断裂所需的最小笔数距离
        """
        self.strokes = strokes
        self.pivots = pivots
        self.min_break_distance = min_break_distance
        self._tracks: List[TrendTrack] = []

    def detect(self) -> List[TrendTrack]:
        """
        检测所有趋势轨道

        Returns:
            轨道列表（按时间排序）
        """
        self._tracks = []
        if not self.pivots:
            return self._tracks

        for pivot in self.pivots:
            track = self._create_track(pivot)
            if track is not None:
                # 检测该轨道的断裂点
                self._check_breakpoint(track)
                self._tracks.append(track)

        return self._tracks

    def _create_track(self, pivot: Pivot) -> Optional[TrendTrack]:
        """
        从中枢创建轨道

        红轨（上涨中枢）：中枢上沿和下沿作为轨道
        绿轨（下跌中枢）：中枢上沿和下沿作为轨道

        Args:
            pivot: 中枢对象

        Returns:
            轨道对象
        """
        direction = TrackDirection.UP if pivot.direction == 'up' else TrackDirection.DOWN
        middle = (pivot.high + pivot.low) / 2

        return TrendTrack(
            direction=direction,
            upper_line=pivot.high,
            lower_line=pivot.low,
            middle_line=middle,
            start_index=pivot.end_index,
            end_index=None,
            pivot=pivot,
        )

    def _check_breakpoint(self, track: TrendTrack) -> None:
        """
        检测轨道断裂点

        断裂条件（来自TDX指标的断点逻辑）：
        - 绿轨（下跌）：出现低点（底分型）高于轨道上沿 → 趋势结束
        - 红轨（上涨）：出现高点（顶分型）低于轨道下沿 → 趋势结束

        同时检查高级断裂：
        - 绿轨：高点低于轨道下沿（加速下跌，但也是偏离）
        - 红轨：低点高于轨道上沿（加速上涨）

        Args:
            track: 待检测的轨道
        """
        start_idx = track.start_index
        distance = 0
        first_break_found = False

        for i, stroke in enumerate(self.strokes):
            if stroke.start_index <= start_idx:
                continue

            distance += 1

            if distance < self.min_break_distance:
                continue

            if track.direction == TrackDirection.DOWN:
                # 绿轨（下跌趋势）
                # 断裂：向上笔的低点高于轨道上沿（价格回到中枢上方）
                if stroke.is_up and stroke.low > track.upper_line:
                    if not first_break_found:
                        track.broken = True
                        track.break_type = TrackBreakType.UPPER_BREAK
                        track.break_index = stroke.end_index
                        track.end_index = stroke.end_index
                        first_break_found = True
                        break

                # 断裂：向下笔的高点低于轨道下沿（加速下跌脱离轨道）
                if stroke.is_down and stroke.high < track.lower_line:
                    # 这种情况不一定是断裂，可能是加速下跌
                    # 只记录但不标记为断裂
                    pass

            elif track.direction == TrackDirection.UP:
                # 红轨（上涨趋势）
                # 断裂：向下笔的高点低于轨道下沿（价格回到中枢下方）
                if stroke.is_down and stroke.high < track.lower_line:
                    if not first_break_found:
                        track.broken = True
                        track.break_type = TrackBreakType.LOWER_BREAK
                        track.break_index = stroke.end_index
                        track.end_index = stroke.end_index
                        first_break_found = True
                        break

                # 断裂：向上笔的低点高于轨道上沿（加速上涨脱离轨道）
                if stroke.is_up and stroke.low > track.upper_line:
                    pass

    # ==================== 查询接口 ====================

    def get_tracks(self) -> List[TrendTrack]:
        """获取所有轨道"""
        return self._tracks.copy()

    def get_active_tracks(self) -> List[TrendTrack]:
        """获取所有仍有效的轨道"""
        return [t for t in self._tracks if t.is_active]

    def get_current_track(self) -> Optional[TrendTrack]:
        """
        获取当前有效的轨道（最近的未断裂轨道）

        Returns:
            当前轨道，如果无有效轨道则返回None
        """
        active = self.get_active_tracks()
        if active:
            return active[-1]

        # 如果所有轨道都断裂了，返回最后一个轨道
        if self._tracks:
            return self._tracks[-1]
        return None

    def get_trend_status(self) -> TrendStatus:
        """
        获取当前趋势状态

        判断逻辑：
        1. 找到最近的两个轨道
        2. 如果最新轨道是红轨且未断裂 → STRONG_UP
        3. 如果最新轨道是绿轨且未断裂 → STRONG_DOWN
        4. 如果最新轨道已断裂 → WEAK_XXX
        5. 无轨道 → NEUTRAL

        Returns:
            当前趋势状态
        """
        if not self._tracks:
            return TrendStatus.NEUTRAL

        current = self.get_current_track()
        if current is None:
            return TrendStatus.NEUTRAL

        if current.direction == TrackDirection.UP:
            return TrendStatus.WEAK_UP if current.broken else TrendStatus.STRONG_UP
        else:
            return TrendStatus.WEAK_DOWN if current.broken else TrendStatus.STRONG_DOWN

    def get_track_confidence_modifier(self, signal_direction: str) -> float:
        """
        根据轨道状态返回买卖信号置信度修正值

        Args:
            signal_direction: 'buy' 或 'sell'

        Returns:
            置信度修正值 (-0.1 ~ +0.1)
        """
        status = self.get_trend_status()
        current_track = self.get_current_track()

        if status == TrendStatus.NEUTRAL:
            return 0.0

        modifier = 0.0

        if signal_direction == 'buy':
            # 买入信号
            if status == TrendStatus.STRONG_UP:
                modifier += 0.05  # 上涨趋势中的买点更可信
            elif status == TrendStatus.STRONG_DOWN:
                modifier += 0.08  # 下跌趋势完好时的1买（底背驰）最有价值
            elif status == TrendStatus.WEAK_DOWN:
                modifier += 0.10  # 下跌趋势断裂 = 可能反转 = 买点信号强
            elif status == TrendStatus.WEAK_UP:
                modifier -= 0.05  # 上涨趋势断裂 = 可能调整 = 买点需谨慎

        elif signal_direction == 'sell':
            # 卖出信号
            if status == TrendStatus.STRONG_DOWN:
                modifier += 0.05  # 下跌趋势中的卖点更可信
            elif status == TrendStatus.STRONG_UP:
                modifier += 0.08  # 上涨趋势完好时的1卖（顶背驰）最有价值
            elif status == TrendStatus.WEAK_UP:
                modifier += 0.10  # 上涨趋势断裂 = 可能反转 = 卖点信号强
            elif status == TrendStatus.WEAK_DOWN:
                modifier -= 0.05  # 下跌趋势断裂 = 可能反弹 = 卖点需谨慎

        # 轨道完好性加分
        if current_track and current_track.is_active:
            modifier += 0.02

        # 限制在 [-0.1, 0.1]
        return max(-0.1, min(0.1, modifier))

    def summary(self) -> str:
        """生成轨道摘要"""
        lines = ["=== 趋势轨道 ==="]
        status = self.get_trend_status()
        lines.append(f"当前趋势状态: {status.value}")

        for i, track in enumerate(self._tracks[-5:]):  # 最近5个轨道
            state = "有效" if track.is_active else f"断裂({track.break_type.value})"
            lines.append(
                f"轨道{i + 1} [{track.direction.value}]: "
                f"上沿={track.upper_line:.2f}, 下沿={track.lower_line:.2f} "
                f"状态={state}"
            )
        return "\n".join(lines)
