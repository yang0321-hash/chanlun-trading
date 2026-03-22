"""
笔识别模块

根据分型生成笔
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .kline import KLine, KLineData
from .fractal import Fractal, FractalType, FractalDetector


class StrokeType(Enum):
    """笔的类型"""
    UP = 'up'      # 向上笔（从底分型到顶分型）
    DOWN = 'down'  # 向下笔（从顶分型到底分型）


@dataclass
class Stroke:
    """
    笔数据

    笔是由相邻的顶分型和底分型连接而成的线段

    Attributes:
        type: 笔的类型（向上/向下）
        start_index: 起始K线索引
        end_index: 结束K线索引
        start_fractal: 起始分型
        end_fractal: 结束分型
        start_value: 起始值（起点分型的关键值）
        end_value: 结束值（终点分型的关键值）
        high: 笔的最高价
        low: 笔的最低价
        length: 笔包含的K线数量
        bars: 笔包含的K线数据
    """
    type: StrokeType
    start_index: int
    end_index: int
    start_fractal: Fractal
    end_fractal: Fractal
    start_value: float
    end_value: float
    high: float
    low: float
    length: int
    bars: List[KLineData]

    @property
    def is_up(self) -> bool:
        """是否向上笔"""
        return self.type == StrokeType.UP

    @property
    def is_down(self) -> bool:
        """是否向下笔"""
        return self.type == StrokeType.DOWN

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
        return self.start_fractal.datetime

    @property
    def end_datetime(self) -> datetime:
        """结束时间"""
        return self.end_fractal.datetime

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'type': self.type.value,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'start_value': self.start_value,
            'end_value': self.end_value,
            'high': self.high,
            'low': self.low,
            'length': self.length,
            'price_change': self.price_change,
            'price_change_pct': self.price_change_pct,
            'start_datetime': self.start_datetime,
            'end_datetime': self.end_datetime
        }


class StrokeGenerator:
    """
    笔生成器

    根据分型生成笔

    笔的生成规则（缠论）：
    1. 笔连接相邻的一顶一底分型
    2. 顶底之间至少有一根K线
    3. 后一笔的起点必须是前一笔的终点
    4. 笔的方向交替出现
    """

    def __init__(
        self,
        kline: KLine,
        fractals: Optional[List[Fractal]] = None,
        min_bars: int = 5
    ):
        """
        初始化笔生成器

        Args:
            kline: K线对象
            fractals: 分型列表，如果为None则自动检测
            min_bars: 笔的最小K线数量
        """
        self.kline = kline
        self.min_bars = min_bars

        if fractals is None:
            detector = FractalDetector(kline, confirm_required=False)
            self.fractals = detector.get_fractals()
        else:
            self.fractals = fractals

        self.strokes: List[Stroke] = []
        self._generate()

    def _generate(self) -> None:
        """生成所有笔"""
        if len(self.fractals) < 2:
            return

        # 按位置排序分型
        sorted_fractals = sorted(self.fractals, key=lambda f: f.index)

        i = 0
        while i < len(sorted_fractals) - 1:
            f1 = sorted_fractals[i]
            f2 = sorted_fractals[i + 1]

            # 必须是一顶一底
            if f1.type == f2.type:
                # 同类型分型，跳过中间的分型
                # 保留更强的那个
                if self._compare_fractal_strength(f1, f2) > 0:
                    i += 1
                else:
                    sorted_fractals.pop(i)
                continue

            # 确定笔的方向
            if f1.is_bottom and f2.is_top:
                stroke_type = StrokeType.UP
            else:
                stroke_type = StrokeType.DOWN

            # 检查是否满足最小K线数
            bars_between = f2.index - f1.index + 1
            if bars_between < self.min_bars:
                # 不满足最小K线数，跳过
                i += 1
                continue

            # 检查是否被破坏
            # 顶底之间不能有更极端的价格
            if self._is_fractal_broken(f1, f2, sorted_fractals):
                i += 1
                continue

            # 创建笔
            stroke = self._create_stroke(f1, f2, stroke_type)
            self.strokes.append(stroke)

            # 移动到下一个分型
            i += 1

    def _compare_fractal_strength(self, f1: Fractal, f2: Fractal) -> int:
        """
        比较两个同类型分型的强度

        Args:
            f1: 第一个分型
            f2: 第二个分型

        Returns:
            1: f1更强, -1: f2更强, 0: 相同
        """
        if f1.is_top:
            # 顶分型：高的更强
            if f1.high > f2.high:
                return 1
            elif f1.high < f2.high:
                return -1
            return 0
        else:
            # 底分型：低的更强
            if f1.low < f2.low:
                return 1
            elif f1.low > f2.low:
                return -1
            return 0

    def _is_fractal_broken(
        self,
        f1: Fractal,
        f2: Fractal,
        all_fractals: List[Fractal]
    ) -> bool:
        """
        检查两个分型之间是否被其他分型破坏

        Args:
            f1: 起始分型
            f2: 结束分型
            all_fractals: 所有分型列表

        Returns:
            是否被破坏
        """
        # 检查f1和f2之间的K线
        # 如果f1是顶分型，f2是底分型（向下笔）
        # 中间不能有比f1更高或比f2更低的分型

        start_idx = f1.index
        end_idx = f2.index

        for f in all_fractals:
            if f.index <= start_idx or f.index >= end_idx:
                continue

            if f1.is_top and f2.is_bottom:
                # 向下笔：中间不能有更高的顶或更低的底
                if f.is_top and f.high > f1.high:
                    return True
                if f.is_bottom and f.low < f2.low:
                    return True
            else:
                # 向上笔：中间不能有更低的底或更高的顶
                if f.is_bottom and f.low < f1.low:
                    return True
                if f.is_top and f.high > f2.high:
                    return True

        return False

    def _create_stroke(
        self,
        start_fractal: Fractal,
        end_fractal: Fractal,
        stroke_type: StrokeType
    ) -> Stroke:
        """创建笔对象"""
        start_idx = start_fractal.index
        end_idx = end_fractal.index

        # 获取笔包含的K线
        bars = self.kline.data[start_idx:end_idx + 1]

        # 计算笔的最高和最低价
        high = max(k.high for k in bars)
        low = min(k.low for k in bars)

        return Stroke(
            type=stroke_type,
            start_index=start_idx,
            end_index=end_idx,
            start_fractal=start_fractal,
            end_fractal=end_fractal,
            start_value=start_fractal.value,
            end_value=end_fractal.value,
            high=high,
            low=low,
            length=len(bars),
            bars=bars
        )

    def get_strokes(
        self,
        stroke_type: Optional[StrokeType] = None
    ) -> List[Stroke]:
        """
        获取笔列表

        Args:
            stroke_type: 笔类型过滤，None表示全部

        Returns:
            笔列表
        """
        if stroke_type is None:
            return self.strokes.copy()

        return [s for s in self.strokes if s.type == stroke_type]

    def get_up_strokes(self) -> List[Stroke]:
        """获取所有向上笔"""
        return self.get_strokes(StrokeType.UP)

    def get_down_strokes(self) -> List[Stroke]:
        """获取所有向下笔"""
        return self.get_strokes(StrokeType.DOWN)

    def get_last_stroke(self) -> Optional[Stroke]:
        """获取最后一笔"""
        if self.strokes:
            return self.strokes[-1]
        return None

    def get_stroke_at(self, index: int) -> Optional[Stroke]:
        """
        获取指定K线位置所在的笔

        Args:
            index: K线索引

        Returns:
            笔对象，如果不在任何笔中则返回None
        """
        for stroke in self.strokes:
            if stroke.start_index <= index <= stroke.end_index:
                return stroke
        return None

    def get_stroke_before(self, index: int) -> Optional[Stroke]:
        """
        获取指定位置之前最近的笔

        Args:
            index: K线索引

        Returns:
            笔对象
        """
        for stroke in reversed(self.strokes):
            if stroke.end_index < index:
                return stroke
        return None

    def get_stroke_after(self, index: int) -> Optional[Stroke]:
        """
        获取指定位置之后最近的笔

        Args:
            index: K线索引

        Returns:
            笔对象
        """
        for stroke in self.strokes:
            if stroke.start_index > index:
                return stroke
        return None

    def __len__(self) -> int:
        return len(self.strokes)

    def __getitem__(self, index: int) -> Stroke:
        return self.strokes[index]


def generate_strokes(
    kline: KLine,
    fractals: Optional[List[Fractal]] = None,
    min_bars: int = 5
) -> List[Stroke]:
    """
    便捷函数：生成K线序列的所有笔

    Args:
        kline: K线对象
        fractals: 分型列表，None则自动检测
        min_bars: 最小K线数

    Returns:
        笔列表
    """
    generator = StrokeGenerator(kline, fractals, min_bars)
    return generator.get_strokes()
