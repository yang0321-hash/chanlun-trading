"""
分型识别模块

实现缠论中的顶分型和底分型识别
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from .kline import KLine, KLineData


class FractalType(Enum):
    """分型类型"""
    TOP = 'top'        # 顶分型
    BOTTOM = 'bottom'  # 底分型


@dataclass
class Fractal:
    """
    分型数据

    Attributes:
        type: 分型类型（顶分型/底分型）
        index: 在K线序列中的索引位置
        kline1: 第一根K线
        kline2: 第二根K线（中间K线）
        kline3: 第三根K线
        datetime: 分型时间（取中间K线时间）
        high: 分型高点（顶分型取最高，底分型取中间）
        low: 分型低点（底分型取最低，顶分型取中间）
        confirmed: 是否已确认（后续K线是否破坏）
    """
    type: FractalType
    index: int
    kline1: KLineData
    kline2: KLineData
    kline3: KLineData
    datetime: datetime
    high: float
    low: float
    confirmed: bool = False

    @property
    def is_top(self) -> bool:
        """是否顶分型"""
        return self.type == FractalType.TOP

    @property
    def is_bottom(self) -> bool:
        """是否底分型"""
        return self.type == FractalType.BOTTOM

    @property
    def value(self) -> float:
        """
        获取分型关键值

        - 顶分型返回高点
        - 底分型返回低点
        """
        return self.high if self.is_top else self.low

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'type': self.type.value,
            'index': self.index,
            'datetime': self.datetime,
            'high': self.high,
            'low': self.low,
            'value': self.value,
            'confirmed': self.confirmed
        }


class FractalDetector:
    """
    分型识别器

    识别K线序列中的顶分型和底分型

    缠论分型定义：
    - 顶分型：第1根K线高点最高，第2根K线高点 < 第1根，第3根K线高点 < 第1根
    - 底分型：第1根K线低点最低，第2根K线低点 > 第1根，第3根K线低点 > 第1根
    """

    def __init__(self, kline: KLine, confirm_required: bool = True):
        """
        初始化分型识别器

        Args:
            kline: K线对象
            confirm_required: 是否要求分型确认（后续K线不破坏分型）
        """
        self.kline = kline
        self.confirm_required = confirm_required
        self.fractals: List[Fractal] = []
        self._detect()

    def _detect(self) -> None:
        """检测所有分型"""
        n = len(self.kline)

        # 至少需要3根K线才能形成分型
        if n < 3:
            return

        i = 0
        while i < n - 2:
            # 尝试从位置i开始识别分型
            fractal = self._detect_fractal_at(i)

            if fractal is not None:
                # 检查是否需要确认
                if self.confirm_required:
                    if self._check_fractal_confirmed(fractal):
                        self.fractals.append(fractal)
                        # 分型已确认，跳过已使用的K线
                        i += 2
                    else:
                        # 未确认，继续下一根K线
                        i += 1
                else:
                    self.fractals.append(fractal)
                    i += 2
            else:
                i += 1

        # 标记确认状态
        self._update_confirmation()

    def _detect_fractal_at(self, index: int) -> Optional[Fractal]:
        """
        在指定位置尝试识别分型

        Args:
            index: 起始位置索引

        Returns:
            分型对象，如果不是分型则返回None
        """
        if index + 2 >= len(self.kline):
            return None

        k1 = self.kline[index]
        k2 = self.kline[index + 1]
        k3 = self.kline[index + 2]

        # 检查顶分型
        if self._is_top_fractal(k1, k2, k3):
            return Fractal(
                type=FractalType.TOP,
                index=index + 1,  # 记录中间K线的位置
                kline1=k1,
                kline2=k2,
                kline3=k3,
                datetime=k2.datetime,
                high=max(k1.high, k2.high, k3.high),
                low=min(k1.low, k2.low, k3.low),
                confirmed=False
            )

        # 检查底分型
        if self._is_bottom_fractal(k1, k2, k3):
            return Fractal(
                type=FractalType.BOTTOM,
                index=index + 1,
                kline1=k1,
                kline2=k2,
                kline3=k3,
                datetime=k2.datetime,
                high=max(k1.high, k2.high, k3.high),
                low=min(k1.low, k2.low, k3.low),
                confirmed=False
            )

        return None

    def _is_top_fractal(self, k1: KLineData, k2: KLineData, k3: KLineData) -> bool:
        """
        判断是否为顶分型

        顶分型条件：
        - 中间K线的高点是三者中最高的
        - k1的高点 <= k2的高点（严格或宽松）

        Args:
            k1: 第一根K线
            k2: 第二根K线（中间）
            k3: 第三根K线

        Returns:
            是否顶分型
        """
        # 严格模式：k2的高点必须严格高于k1和k3
        # 宽松模式：允许相等（根据包含关系处理后的K线通常严格）

        # k2的高点最高
        if k2.high < k1.high or k2.high < k3.high:
            return False

        # k1和k3的高点都低于k2
        # 低点没有要求，但顶分型通常是上升后的转折
        return True

    def _is_bottom_fractal(self, k1: KLineData, k2: KLineData, k3: KLineData) -> bool:
        """
        判断是否为底分型

        底分型条件：
        - 中间K线的低点是三者中最低的

        Args:
            k1: 第一根K线
            k2: 第二根K线（中间）
            k3: 第三根K线

        Returns:
            是否底分型
        """
        # k2的低点最低
        if k2.low > k1.low or k2.low > k3.low:
            return False

        return True

    def _check_fractal_confirmed(self, fractal: Fractal) -> bool:
        """
        检查分型是否被确认

        确认条件：
        - 顶分型：后续K线收盘价跌破分型低点
        - 底分型：后续K线收盘价突破分型高点

        Args:
            fractal: 分型对象

        Returns:
            是否确认
        """
        # 检查分型之后的K线
        start_index = fractal.index + 2  # 分型结束位置

        if start_index >= len(self.kline):
            return False

        # 检查后续几根K线
        check_count = min(5, len(self.kline) - start_index)

        for i in range(start_index, start_index + check_count):
            k = self.kline[i]

            if fractal.is_top:
                # 顶分型确认：收盘价跌破分型低点
                if k.close < fractal.low:
                    return True
            else:
                # 底分型确认：收盘价突破分型高点
                if k.close > fractal.high:
                    return True

        return False

    def _update_confirmation(self) -> None:
        """更新所有分型的确认状态"""
        for fractal in self.fractals:
            if not fractal.confirmed:
                confirmed = self._check_fractal_confirmed(fractal)
                fractal.confirmed = confirmed

    def get_fractals(self, fractal_type: Optional[FractalType] = None) -> List[Fractal]:
        """
        获取分型列表

        Args:
            fractal_type: 分型类型过滤，None表示全部

        Returns:
            分型列表
        """
        if fractal_type is None:
            return self.fractals.copy()

        return [f for f in self.fractals if f.type == fractal_type]

    def get_top_fractals(self) -> List[Fractal]:
        """获取所有顶分型"""
        return self.get_fractals(FractalType.TOP)

    def get_bottom_fractals(self) -> List[Fractal]:
        """获取所有底分型"""
        return self.get_fractals(FractalType.BOTTOM)

    def get_fractal_at(self, index: int) -> Optional[Fractal]:
        """
        获取指定位置的分型

        Args:
            index: K线索引

        Returns:
            分型对象，如果该位置不是分型则返回None
        """
        for fractal in self.fractals:
            if fractal.index == index:
                return fractal
        return None

    def get_fractal_after(self, index: int) -> Optional[Fractal]:
        """
        获取指定位置之后的第一个分型

        Args:
            index: 起始位置

        Returns:
            分型对象，如果没有则返回None
        """
        for fractal in self.fractals:
            if fractal.index > index:
                return fractal
        return None

    def is_fractal_index(self, index: int) -> bool:
        """
        判断指定位置是否为分型

        Args:
            index: K线索引

        Returns:
            是否为分型
        """
        return self.get_fractal_at(index) is not None

    def __len__(self) -> int:
        return len(self.fractals)

    def __getitem__(self, index: int) -> Fractal:
        return self.fractals[index]


def detect_fractals(kline: KLine, confirm_required: bool = True) -> List[Fractal]:
    """
    便捷函数：检测K线序列中的所有分型

    Args:
        kline: K线对象
        confirm_required: 是否要求分型确认

    Returns:
        分型列表
    """
    detector = FractalDetector(kline, confirm_required)
    return detector.get_fractals()
