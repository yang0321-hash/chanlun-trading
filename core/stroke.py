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
    def strength_score(self) -> float:
        """
        笔强度评分 (0-1)

        衡量笔的力度：振幅越大、长度越长 = 越有力量
        """
        if self.length == 0 or self.start_value == 0:
            return 0.0
        amplitude_pct = abs(self.price_change_pct)
        # 10%振幅 = 满分，线性归一化
        return min(amplitude_pct / 10.0, 1.0)

    @property
    def amplitude(self) -> float:
        """笔的振幅（最高-最低）"""
        return self.high - self.low

    @property
    def amplitude_pct(self) -> float:
        """笔的振幅百分比"""
        mid = (self.high + self.low) / 2
        if mid == 0:
            return 0.0
        return (self.high - self.low) / mid * 100

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
        """生成所有笔

        两阶段算法 (对齐TDX缠论笔划分):
        1. Swing检测: 在包含处理后的K线上找摆动高低点 (HHV/LLV确认)
        2. 最小间距过滤: 确保相邻端点之间有足够间距

        对应TDX指标 GSKZ30~GSKZ49 的核心逻辑:
        - 不急于确认分型, 等到真正的极值出现且不被超越
        - 用 HHV(最高价) / LLV(最低价) 做确认
        """
        if len(self.fractals) < 2:
            return

        kline_data = self.kline.data
        n = len(kline_data)
        if n < self.min_bars:
            return

        # ---- 阶段1: Swing High/Low 检测 ----
        # 找到所有局部极值点: 左右各lookback根K线内, 当前K线是最高/最低
        # lookback越大, 笔越少越长 (类似TDX的 SUMBARS N层展开)
        lookback = max(3, self.min_bars)

        swing_points = []  # (index, 'high'|'low')

        for i in range(n):
            is_high = True
            is_low = True

            lo = max(0, i - lookback)
            hi = min(n - 1, i + lookback)

            for j in range(lo, hi + 1):
                if j == i:
                    continue
                if kline_data[j].high > kline_data[i].high:
                    is_high = False
                if kline_data[j].low < kline_data[i].low:
                    is_low = False
                if not is_high and not is_low:
                    break

            if is_high:
                swing_points.append((i, 'high', kline_data[i].high))
            elif is_low:
                swing_points.append((i, 'low', kline_data[i].low))

        if len(swing_points) < 2:
            return

        # ---- 阶段2: 合并相邻同类型极值 ----
        # TDX逻辑: GSKZ44/GSKZ45 — COUNT=1 确保首次出现, 同类只保留最极端
        merged = [swing_points[0]]
        for pt in swing_points[1:]:
            last = merged[-1]
            if pt[1] != last[1]:
                # 不同类型 (高→低 或 低→高), 直接添加
                merged.append(pt)
            else:
                # 同类型: 保留更极端的
                if pt[1] == 'high' and pt[2] > last[2]:
                    merged[-1] = pt
                elif pt[1] == 'low' and pt[2] < last[2]:
                    merged[-1] = pt

        # ---- 阶段3: 最小间距过滤 ----
        # 确保相邻端点之间至少有 min_bars 根K线
        filtered = [merged[0]]
        for pt in merged[1:]:
            if pt[0] - filtered[-1][0] >= self.min_bars:
                filtered.append(pt)
            else:
                # 太近了, 保留更极端的
                last = filtered[-1]
                if pt[1] == 'high' and last[1] == 'high':
                    if pt[2] > last[2]:
                        filtered[-1] = pt
                elif pt[1] == 'low' and last[1] == 'low':
                    if pt[2] < last[2]:
                        filtered[-1] = pt
                # 不同类型但太近: 丢弃较弱的一个
                elif pt[1] != last[1]:
                    # 看哪个更极端 (距离前一个端点更远)
                    prev = filtered[-2] if len(filtered) >= 2 else None
                    if prev is not None:
                        # 保留与前一个端点类型不同的、且距离更远的
                        if (pt[0] - prev[0]) > (last[0] - prev[0]):
                            filtered[-1] = pt

        # ---- 阶段4: 确保顶底交替 ----
        # 最终清理: 如果还有连续同类型, 只保留最极端的
        final = [filtered[0]]
        for pt in filtered[1:]:
            last = final[-1]
            if pt[1] != last[1]:
                final.append(pt)
            else:
                if pt[1] == 'high' and pt[2] > last[2]:
                    final[-1] = pt
                elif pt[1] == 'low' and pt[2] < last[2]:
                    final[-1] = pt

        # ---- 阶段5: 构建笔 ----
        for i in range(len(final) - 1):
            start_pt = final[i]
            end_pt = final[i + 1]
            si, ei = start_pt[0], end_pt[0]

            if start_pt[1] == 'low' and end_pt[1] == 'high':
                stype = StrokeType.UP
            elif start_pt[1] == 'high' and end_pt[1] == 'low':
                stype = StrokeType.DOWN
            else:
                continue  # 不应该出现, 跳过

            lo_idx, hi_idx = min(si, ei), max(si, ei)
            bars = kline_data[lo_idx:hi_idx + 1]
            if not bars:
                continue

            stroke_high = max(k.high for k in bars)
            stroke_low = min(k.low for k in bars)

            # 构建Fractal
            start_f = Fractal(
                type=FractalType.BOTTOM if stype == StrokeType.UP else FractalType.TOP,
                index=si,
                kline1=kline_data[max(0, si - 1)],
                kline2=kline_data[si],
                kline3=kline_data[min(n - 1, si + 1)],
                datetime=kline_data[si].datetime,
                high=kline_data[si].high,
                low=kline_data[si].low,
            )
            end_f = Fractal(
                type=FractalType.TOP if stype == StrokeType.UP else FractalType.BOTTOM,
                index=ei,
                kline1=kline_data[max(0, ei - 1)],
                kline2=kline_data[ei],
                kline3=kline_data[min(n - 1, ei + 1)],
                datetime=kline_data[ei].datetime,
                high=kline_data[ei].high,
                low=kline_data[ei].low,
            )

            start_val = start_pt[2]
            end_val = end_pt[2]

            self.strokes.append(Stroke(
                type=stype,
                start_index=si,
                end_index=ei,
                start_fractal=start_f,
                end_fractal=end_f,
                start_value=start_val,
                end_value=end_val,
                high=stroke_high,
                low=stroke_low,
                length=len(bars),
                bars=bars,
            ))

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
