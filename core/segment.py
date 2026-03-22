"""
线段识别模块

根据笔生成线段
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from .kline import KLine
from .fractal import Fractal
from .stroke import Stroke, StrokeType, StrokeGenerator


@dataclass
class Segment:
    """
    线段数据

    线段是由至少3笔组成的更高级别的走势结构

    Attributes:
        strokes: 组成线段的笔列表
        start_index: 起始K线索引
        end_index: 结束K线索引
        start_value: 线段起始值
        end_value: 线段结束值
        high: 线段最高价
        low: 线段最低价
        direction: 线段方向 ('up' 或 'down')
        broken: 是否被破坏
    """
    strokes: List[Stroke]
    start_index: int
    end_index: int
    start_value: float
    end_value: float
    high: float
    low: float
    direction: str
    broken: bool = False

    @property
    def is_up(self) -> bool:
        """是否向上线段"""
        return self.direction == 'up'

    @property
    def is_down(self) -> bool:
        """是否向下线段"""
        return self.direction == 'down'

    @property
    def length(self) -> int:
        """线段包含的笔数量"""
        return len(self.strokes)

    @property
    def bars_count(self) -> int:
        """线段包含的K线数量"""
        if not self.strokes:
            return 0
        return self.strokes[-1].end_index - self.strokes[0].start_index + 1

    @property
    def price_change(self) -> float:
        """价格变化"""
        return self.end_value - self.start_value

    @property
    def price_change_pct(self) -> float:
        """价格变化百分比"""
        if self.start_value == 0:
            return 0
        return (self.end_value - self.start_value) / self.start_value * 100

    @property
    def start_datetime(self) -> datetime:
        """起始时间"""
        return self.strokes[0].start_datetime

    @property
    def end_datetime(self) -> datetime:
        """结束时间"""
        return self.strokes[-1].end_datetime

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'direction': self.direction,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'start_value': self.start_value,
            'end_value': self.end_value,
            'high': self.high,
            'low': self.low,
            'length': self.length,
            'bars_count': self.bars_count,
            'price_change': self.price_change,
            'price_change_pct': self.price_change_pct,
            'start_datetime': self.start_datetime,
            'end_datetime': self.end_datetime,
            'broken': self.broken
        }


class SegmentGenerator:
    """
    线段生成器

    根据笔生成线段

    线段生成规则（缠论简化版）：
    1. 线段至少由3笔组成
    2. 第3笔破坏第1笔的端点时，线段可能结束
    3. 线段破坏后形成新线段
    """

    def __init__(
        self,
        kline: KLine,
        strokes: Optional[List[Stroke]] = None,
        min_strokes: int = 3
    ):
        """
        初始化线段生成器

        Args:
            kline: K线对象
            strokes: 笔列表，如果为None则自动生成
            min_strokes: 线段最小笔数
        """
        self.kline = kline
        self.min_strokes = min_strokes

        if strokes is None:
            stroke_gen = StrokeGenerator(kline)
            self.strokes = stroke_gen.get_strokes()
        else:
            self.strokes = strokes

        self.segments: List[Segment] = []
        self._generate()

    def _generate(self) -> None:
        """生成所有线段"""
        if len(self.strokes) < self.min_strokes:
            return

        i = 0
        while i <= len(self.strokes) - self.min_strokes:
            # 尝试从位置i开始生成线段
            segment = self._try_create_segment(i)

            if segment is not None:
                self.segments.append(segment)
                # 跳过已使用的笔
                i += segment.length - 1  # 保留最后一笔作为新线段的开始
            else:
                i += 1

        # 检查线段破坏
        self._check_segment_breaks()

    def _try_create_segment(self, start_idx: int) -> Optional[Segment]:
        """
        尝试从指定位置创建线段

        Args:
            start_idx: 起始笔索引

        Returns:
            线段对象，如果不能创建则返回None
        """
        # 至少需要3笔
        if start_idx + 2 >= len(self.strokes):
            return None

        s1 = self.strokes[start_idx]
        s2 = self.strokes[start_idx + 1]
        s3 = self.strokes[start_idx + 2]

        # 检查笔的方向是否交替
        if s1.type == s2.type or s2.type == s3.type:
            return None

        # 确定线段方向（第一笔的方向）
        direction = 'up' if s1.is_up else 'down'

        # 向上线段: s1(上) -> s2(下) -> s3(上)
        # 向下线段: s1(下) -> s2(上) -> s3(下)

        segment_strokes = [s1, s2, s3]

        # 检查s3是否破坏s1
        is_broken = self._check_stroke_break(s1, s3, direction)

        if not is_broken:
            # s3没有破坏s1，线段可能继续延伸
            # 检查后续笔
            idx = start_idx + 3
            while idx < len(self.strokes):
                next_stroke = self.strokes[idx]

                # 检查方向是否交替
                if next_stroke.type == self.strokes[idx - 1].type:
                    break

                segment_strokes.append(next_stroke)

                # 检查是否破坏
                if direction == 'up' and next_stroke.is_up:
                    if next_stroke.end_value < s1.start_value:
                        is_broken = True
                        break
                elif direction == 'down' and next_stroke.is_down:
                    if next_stroke.end_value > s1.start_value:
                        is_broken = True
                        break

                idx += 1

        # 计算线段属性
        start_idx_k = s1.start_index
        end_idx_k = segment_strokes[-1].end_index

        # 确定线段的起止值
        if direction == 'up':
            start_value = min(s.start_value for s in segment_strokes)
            end_value = max(s.end_value for s in segment_strokes)
        else:
            start_value = max(s.start_value for s in segment_strokes)
            end_value = min(s.end_value for s in segment_strokes)

        # 计算最高最低价
        high = max(s.high for s in segment_strokes)
        low = min(s.low for s in segment_strokes)

        return Segment(
            strokes=segment_strokes,
            start_index=start_idx_k,
            end_index=end_idx_k,
            start_value=start_value,
            end_value=end_value,
            high=high,
            low=low,
            direction=direction,
            broken=is_broken
        )

    def _check_stroke_break(
        self,
        s1: Stroke,
        s3: Stroke,
        direction: str
    ) -> bool:
        """
        检查s3是否破坏s1

        Args:
            s1: 第一笔
            s3: 第三笔
            direction: 线段方向

        Returns:
            是否破坏
        """
        if direction == 'up':
            # 向上线段：s3的终点跌破s1的起点
            return s3.end_value < s1.start_value
        else:
            # 向下线段：s3的终点突破s1的起点
            return s3.end_value > s1.start_value

    def _check_segment_breaks(self) -> None:
        """检查线段之间的破坏关系"""
        for i in range(len(self.segments) - 1):
            seg1 = self.segments[i]
            seg2 = self.segments[i + 1]

            # 如果seg1的direction和seg2的direction相同
            # 检查seg2是否破坏了seg1
            if seg1.direction == seg2.direction:
                if seg1.is_up:
                    # 向上线段：seg2的起点低于seg1的终点
                    if seg2.start_value < seg1.end_value:
                        seg1.broken = True
                else:
                    # 向下线段：seg2的起点高于seg1的终点
                    if seg2.start_value > seg1.end_value:
                        seg1.broken = True

    def get_segments(
        self,
        direction: Optional[str] = None
    ) -> List[Segment]:
        """
        获取线段列表

        Args:
            direction: 方向过滤 ('up'/'down'/None)

        Returns:
            线段列表
        """
        if direction is None:
            return self.segments.copy()

        return [s for s in self.segments if s.direction == direction]

    def get_up_segments(self) -> List[Segment]:
        """获取所有向上线段"""
        return self.get_segments('up')

    def get_down_segments(self) -> List[Segment]:
        """获取所有向下线段"""
        return self.get_segments('down')

    def get_last_segment(self) -> Optional[Segment]:
        """获取最后一条线段"""
        if self.segments:
            return self.segments[-1]
        return None

    def get_segment_at(self, index: int) -> Optional[Segment]:
        """
        获取指定K线位置所在的线段

        Args:
            index: K线索引

        Returns:
            线段对象
        """
        for segment in self.segments:
            if segment.start_index <= index <= segment.end_index:
                return segment
        return None

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Segment:
        return self.segments[index]


def generate_segments(
    kline: KLine,
    strokes: Optional[List[Stroke]] = None,
    min_strokes: int = 3
) -> List[Segment]:
    """
    便捷函数：生成K线序列的所有线段

    Args:
        kline: K线对象
        strokes: 笔列表，None则自动生成
        min_strokes: 最小笔数

    Returns:
        线段列表
    """
    generator = SegmentGenerator(kline, strokes, min_strokes)
    return generator.get_segments()
