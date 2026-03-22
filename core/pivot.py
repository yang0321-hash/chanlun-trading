"""
中枢识别模块

识别缠论中的中枢结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from .kline import KLine
from .fractal import Fractal
from .stroke import Stroke, StrokeType
from .segment import Segment


class PivotLevel(Enum):
    """中枢级别"""
    MIN_1 = '1m'       # 1分钟级别
    MIN_5 = '5m'       # 5分钟级别
    MIN_30 = '30m'     # 30分钟级别
    DAY = 'day'        # 日线级别
    WEEK = 'week'      # 周线级别
    MONTH = 'month'    # 月线级别


@dataclass
class Pivot:
    """
    中枢数据

    中枢是价格在一定区间内的震荡区域

    Attributes:
        level: 中枢级别
        start_index: 起始K线索引
        end_index: 结束K线索引
        high: 中枢上沿（高点中的最低点）
        low: 中枢下沿（低点中的最高点）
        strokes: 组成中枢的笔列表
        direction: 中枢方向 ('up'表示上涨中枢, 'down'表示下跌中枢)
        confirmed: 中枢是否确认
        extended: 中枢是否延伸
        segments: 关联的线段
    """
    level: PivotLevel
    start_index: int
    end_index: int
    high: float
    low: float
    strokes: List[Stroke] = field(default_factory=list)
    direction: str = ''
    confirmed: bool = False
    extended: bool = False
    segments: List[Segment] = field(default_factory=list)

    @property
    def range_value(self) -> float:
        """中枢区间宽度"""
        return self.high - self.low

    @property
    def middle(self) -> float:
        """中枢中点"""
        return (self.high + self.low) / 2

    @property
    def strokes_count(self) -> int:
        """组成中枢的笔数量"""
        return len(self.strokes)

    @property
    def start_datetime(self) -> datetime:
        """起始时间"""
        if self.strokes:
            return self.strokes[0].start_datetime
        return datetime.now()

    @property
    def end_datetime(self) -> datetime:
        """结束时间"""
        if self.strokes:
            return self.strokes[-1].end_datetime
        return datetime.now()

    def contains(self, price: float) -> bool:
        """
        判断价格是否在中枢区间内

        Args:
            price: 价格

        Returns:
            是否在中枢内
        """
        return self.low <= price <= self.high

    def is_above(self, price: float) -> bool:
        """判断价格是否在中枢上方"""
        return price > self.high

    def is_below(self, price: float) -> bool:
        """判断价格是否在中枢下方"""
        return price < self.low

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'level': self.level.value,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'high': self.high,
            'low': self.low,
            'range': self.range_value,
            'middle': self.middle,
            'strokes_count': self.strokes_count,
            'direction': self.direction,
            'confirmed': self.confirmed,
            'extended': self.extended,
            'start_datetime': self.start_datetime,
            'end_datetime': self.end_datetime
        }


class PivotDetector:
    """
    中枢识别器

    识别K线序列中的中枢结构

    中枢定义（缠论）：
    1. 至少由3笔组成，且存在重叠区间
    2. 中枢区间：高点中的最低点 到 低点中的最高点
    3. 后续笔在中枢区间内震荡则为中枢延伸
    4. 突破中枢后回抽不破则为第三类买卖点
    """

    def __init__(
        self,
        kline: KLine,
        strokes: Optional[List[Stroke]] = None,
        level: PivotLevel = PivotLevel.DAY
    ):
        """
        初始化中枢识别器

        Args:
            kline: K线对象
            strokes: 笔列表，如果为None则自动生成
            level: 中枢级别
        """
        self.kline = kline
        self.level = level

        if strokes is None:
            from .stroke import StrokeGenerator
            stroke_gen = StrokeGenerator(kline)
            self.strokes = stroke_gen.get_strokes()
        else:
            self.strokes = strokes

        self.pivots: List[Pivot] = []
        self._detect()

    def _detect(self) -> None:
        """检测所有中枢"""
        if len(self.strokes) < 3:
            return

        i = 0
        while i <= len(self.strokes) - 3:
            pivot = self._try_create_pivot(i)

            if pivot is not None:
                self.pivots.append(pivot)
                # 跳过已使用的笔
                i += pivot.strokes_count - 1
            else:
                i += 1

        # 检查中枢延伸
        self._check_pivot_extension()

    def _try_create_pivot(self, start_idx: int) -> Optional[Pivot]:
        """
        尝试从指定位置创建中枢

        Args:
            start_idx: 起始笔索引

        Returns:
            中枢对象，如果不能创建则返回None
        """
        if start_idx + 2 >= len(self.strokes):
            return None

        # 获取3笔
        s1 = self.strokes[start_idx]
        s2 = self.strokes[start_idx + 1]
        s3 = self.strokes[start_idx + 2]

        # 检查方向是否交替
        if s1.type == s2.type or s2.type == s3.type:
            return None

        # 计算重叠区间
        pivot_strokes = [s1, s2, s3]

        # 计算中枢区间
        # 高点中的最低点
        pivot_high = min(s.high for s in pivot_strokes)
        # 低点中的最高点
        pivot_low = max(s.low for s in pivot_strokes)

        # 检查是否有有效的重叠区间
        if pivot_high <= pivot_low:
            return None

        # 确定中枢方向
        # 第一笔向上则是上涨中枢，向下则是下跌中枢
        direction = 'up' if s1.is_up else 'down'

        start_index = s1.start_index
        end_index = s3.end_index

        # 尝试扩展中枢
        idx = start_idx + 3
        while idx < len(self.strokes):
            next_stroke = self.strokes[idx]

            # 检查下一笔是否在中枢区间内
            if self._is_within_pivot(next_stroke, pivot_high, pivot_low):
                pivot_strokes.append(next_stroke)
                end_index = next_stroke.end_index
                idx += 1
            else:
                break

        return Pivot(
            level=self.level,
            start_index=start_index,
            end_index=end_index,
            high=pivot_high,
            low=pivot_low,
            strokes=pivot_strokes,
            direction=direction,
            confirmed=True
        )

    def _is_within_pivot(
        self,
        stroke: Stroke,
        pivot_high: float,
        pivot_low: float
    ) -> bool:
        """
        判断笔是否在中枢区间内

        Args:
            stroke: 笔对象
            pivot_high: 中枢高点
            pivot_low: 中枢低点

        Returns:
            是否在中枢内
        """
        # 笔的高点 <= 中枢高点 且 笔的低点 >= 中枢低点
        return stroke.high <= pivot_high and stroke.low >= pivot_low

    def _check_pivot_extension(self) -> None:
        """检查中枢延伸"""
        for i in range(len(self.pivots) - 1):
            pivot1 = self.pivots[i]
            pivot2 = self.pivots[i + 1]

            # 如果两个中枢有重叠，则中枢延伸
            if not (pivot1.high < pivot2.low or pivot2.high < pivot1.low):
                pivot1.extended = True
                pivot2.extended = True

    def get_pivots(self) -> List[Pivot]:
        """获取所有中枢"""
        return self.pivots.copy()

    def get_pivot_at(self, index: int) -> Optional[Pivot]:
        """
        获取指定K线位置所在的中枢

        Args:
            index: K线索引

        Returns:
            中枢对象
        """
        for pivot in self.pivots:
            if pivot.start_index <= index <= pivot.end_index:
                return pivot
        return None

    def get_latest_pivot(self) -> Optional[Pivot]:
        """获取最新的中枢"""
        if self.pivots:
            return self.pivots[-1]
        return None

    def check_pivot_breakout(self, pivot: Pivot, current_price: float) -> Tuple[bool, str]:
        """
        检查是否突破中枢

        Args:
            pivot: 中枢对象
            current_price: 当前价格

        Returns:
            (是否突破, 突破方向: 'up'/'down'/None)
        """
        if current_price > pivot.high:
            return (True, 'up')
        elif current_price < pivot.low:
            return (True, 'down')
        return (False, '')

    def __len__(self) -> int:
        return len(self.pivots)

    def __getitem__(self, index: int) -> Pivot:
        return self.pivots[index]


def detect_pivots(
    kline: KLine,
    strokes: Optional[List[Stroke]] = None,
    level: PivotLevel = PivotLevel.DAY
) -> List[Pivot]:
    """
    便捷函数：检测K线序列的所有中枢

    Args:
        kline: K线对象
        strokes: 笔列表，None则自动生成
        level: 中枢级别

    Returns:
        中枢列表
    """
    detector = PivotDetector(kline, strokes, level)
    return detector.get_pivots()


class MultiLevelPivot:
    """
    多级别中枢分析

    同时分析不同级别的中枢结构
    """

    def __init__(self, kline_dict: dict):
        """
        初始化多级别中枢分析

        Args:
            kline_dict: 字典，key为级别名称，value为KLine对象
                {
                    '1m': KLine(...),
                    '5m': KLine(...),
                    '30m': KLine(...),
                    'day': KLine(...)
                }
        """
        self.kline_dict = kline_dict
        self.detectors = {}
        self._analyze()

    def _analyze(self) -> None:
        """分析所有级别的中枢"""
        level_map = {
            '1m': PivotLevel.MIN_1,
            '5m': PivotLevel.MIN_5,
            '30m': PivotLevel.MIN_30,
            'day': PivotLevel.DAY,
            'week': PivotLevel.WEEK,
            'month': PivotLevel.MONTH
        }

        for level_name, kline in self.kline_dict.items():
            level = level_map.get(level_name, PivotLevel.DAY)
            self.detectors[level_name] = PivotDetector(kline, level=level)

    def get_pivots_by_level(self, level: str) -> List[Pivot]:
        """
        获取指定级别的中枢

        Args:
            level: 级别名称 ('1m', '5m', '30m', 'day', 'week', 'month')

        Returns:
            中枢列表
        """
        if level in self.detectors:
            return self.detectors[level].get_pivots()
        return []

    def get_aligned_pivots(self) -> dict:
        """
        获取各级别对齐的中枢

        Returns:
            字典，key为级别，value为中枢列表
        """
        return {level: detector.get_pivots()
                for level, detector in self.detectors.items()}
