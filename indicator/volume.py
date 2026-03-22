"""
量能分析模块

提供成交量分析功能，用于缠论交易系统的量价配合验证。
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class VolumePattern:
    """成交量形态"""
    is_expanding: bool = False      # 放量
    is_contracting: bool = False    # 缩量
    is_heavy: bool = False          # 巨量
    is_light: bool = False          # 地量
    ratio_vs_avg: float = 1.0       # 与均量的比值
    trend: str = 'neutral'          # up, down, neutral


@dataclass
class VolumeDivergence:
    """量能背离"""
    has_divergence: bool = False
    direction: str = ''  # 'bottom' or 'top'
    strength: float = 0.0


class VolumeAnalyzer:
    """
    成交量分析器

    功能：
    1. 识别放量/缩量
    2. 检测量能背离
    3. 量价配合分析
    4. 计算量能趋势
    """

    # 默认参数
    DEFAULT_MA_PERIOD = 20
    HEAVY_VOLUME_THRESHOLD = 2.0    # 超过均量2倍为巨量
    LIGHT_VOLUME_THRESHOLD = 0.5    # 低于均量50%为地量
    EXPAND_THRESHOLD = 1.3          # 超过均量30%为放量
    CONTRACT_THRESHOLD = 0.7        # 低于均量30%为缩量

    def __init__(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        ma_period: int = DEFAULT_MA_PERIOD
    ):
        """
        初始化分析器

        Args:
            prices: 价格序列
            volumes: 成交量序列
            ma_period: 均量周期
        """
        self.prices = np.array(prices)
        self.volumes = np.array(volumes)
        self.ma_period = ma_period

        # 计算均量线
        self.volume_ma = self._calculate_volume_ma()

        # 计算量比
        self.volume_ratio = self._calculate_volume_ratio()

    def _calculate_volume_ma(self) -> np.ndarray:
        """计算均量线"""
        ma = np.full_like(self.volumes, np.nan, dtype=float)
        for i in range(self.ma_period - 1, len(self.volumes)):
            ma[i] = np.mean(self.volumes[i - self.ma_period + 1:i + 1])
        return ma

    def _calculate_volume_ratio(self) -> np.ndarray:
        """计算量比（当前量/均量）"""
        ratio = np.ones_like(self.volumes, dtype=float)
        valid = ~np.isnan(self.volume_ma)
        ratio[valid] = self.volumes[valid] / self.volume_ma[valid]
        return ratio

    def get_current_pattern(self, index: int = -1) -> VolumePattern:
        """
        获取当前成交量形态

        Args:
            index: K线索引，默认最新

        Returns:
            VolumePattern对象
        """
        if index < 0:
            index = len(self.volumes) + index

        if index < 0 or index >= len(self.volumes):
            return VolumePattern()

        current_ratio = self.volume_ratio[index]
        current_vol = self.volumes[index]

        # 判断放量/缩量
        is_expanding = current_ratio > self.EXPAND_THRESHOLD
        is_contracting = current_ratio < self.CONTRACT_THRESHOLD

        # 判断巨量/地量
        is_heavy = current_ratio > self.HEAVY_VOLUME_THRESHOLD
        is_light = current_ratio < self.LIGHT_VOLUME_THRESHOLD

        # 判断量能趋势
        trend = self._determine_volume_trend(index)

        return VolumePattern(
            is_expanding=is_expanding,
            is_contracting=is_contracting,
            is_heavy=is_heavy,
            is_light=is_light,
            ratio_vs_avg=current_ratio,
            trend=trend
        )

    def _determine_volume_trend(self, index: int, lookback: int = 5) -> str:
        """判断量能趋势"""
        if index < lookback:
            return 'neutral'

        recent_ratios = self.volume_ratio[index - lookback + 1:index + 1]

        # 计算趋势斜率
        if len(recent_ratios) < 3:
            return 'neutral'

        x = np.arange(len(recent_ratios))
        slope, _ = np.polyfit(x, recent_ratios, 1)

        if slope > 0.05:
            return 'up'
        elif slope < -0.05:
            return 'down'
        return 'neutral'

    def check_volume_confirmation(
        self,
        index: int = -1,
        min_ratio: float = 1.2
    ) -> Tuple[bool, str]:
        """
        检查是否有量能确认

        Args:
            index: K线索引
            min_ratio: 最小量比要求

        Returns:
            (是否确认, 原因说明)
        """
        pattern = self.get_current_pattern(index)

        if pattern.is_heavy:
            return True, f"巨量确认({pattern.ratio_vs_avg:.1f}倍)"
        elif pattern.is_expanding and pattern.ratio_vs_avg >= min_ratio:
            return True, f"放量确认({pattern.ratio_vs_avg:.1f}倍)"
        elif pattern.trend == 'up':
            return True, f"量能趋势向上"

        return False, f"量能不足({pattern.ratio_vs_avg:.1f}倍)"

    def check_divergence(
        self,
        index: int = -1,
        lookback: int = 20
    ) -> VolumeDivergence:
        """
        检测量价背离

        底背离：价格创新低，但量能递减
        顶背离：价格创新高，但量能递减

        Args:
            index: 当前K线索引
            lookback: 回溯K线数

        Returns:
            VolumeDivergence对象
        """
        if index < lookback + 3:
            return VolumeDivergence()

        start_idx = max(0, index - lookback)
        prices = self.prices[start_idx:index + 1]
        volumes = self.volumes[start_idx:index + 1]

        # 找价格低点
        price_lows = self._find_extrema(prices, mode='low', n=2)
        # 找价格高点
        price_highs = self._find_extrema(prices, mode='high', n=2)

        # 检查底背离
        if len(price_lows) >= 2:
            low1_idx, low1_val = price_lows[-2]
            low2_idx, low2_val = price_lows[-1]

            # 价格创新低
            if low2_val < low1_val:
                # 量能递减
                vol1 = volumes[low1_idx]
                vol2 = volumes[low2_idx]
                if vol2 < vol1 * 0.8:  # 量能减少20%以上
                    strength = (vol1 - vol2) / vol1
                    return VolumeDivergence(
                        has_divergence=True,
                        direction='bottom',
                        strength=strength
                    )

        # 检查顶背离
        if len(price_highs) >= 2:
            high1_idx, high1_val = price_highs[-2]
            high2_idx, high2_val = price_highs[-1]

            # 价格创新高
            if high2_val > high1_val:
                # 量能递减
                vol1 = volumes[high1_idx]
                vol2 = volumes[high2_idx]
                if vol2 < vol1 * 0.85:
                    strength = (vol1 - vol2) / vol1
                    return VolumeDivergence(
                        has_divergence=True,
                        direction='top',
                        strength=strength
                    )

        return VolumeDivergence()

    def _find_extrema(
        self,
        data: np.ndarray,
        mode: str = 'low',
        n: int = 2
    ) -> List[Tuple[int, float]]:
        """找到局部极值点"""
        extrema = []

        for i in range(1, len(data) - 1):
            if mode == 'low':
                if data[i] < data[i-1] and data[i] < data[i+1]:
                    extrema.append((i, data[i]))
            else:  # high
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    extrema.append((i, data[i]))

        # 按极值大小排序，返回前n个
        if mode == 'low':
            extrema.sort(key=lambda x: x[1])
        else:
            extrema.sort(key=lambda x: x[1], reverse=True)

        return extrema[:n]

    def check_price_volume_match(
        self,
        price_change_pct: float,
        index: int = -1
    ) -> Tuple[bool, str]:
        """
        检查量价配合

        理想配合：
        - 价涨量增
        - 价跌量缩
        - 价平量缩

        不良配合：
        - 价涨量缩（量价背离，上涨无力）
        - 价跌量增（恐慌性抛售）

        Args:
            price_change_pct: 价格变化百分比（正数为涨，负数为跌）
            index: K线索引

        Returns:
            (是否良好配合, 说明)
        """
        pattern = self.get_current_pattern(index)

        # 价格上涨
        if price_change_pct > 0:
            if pattern.is_expanding:
                return True, "价涨量增(健康)"
            elif pattern.is_contracting:
                return False, "价涨量缩(乏力)"
            else:
                return True, "价涨量平"

        # 价格下跌
        elif price_change_pct < 0:
            if pattern.is_contracting:
                return True, "价跌量缩(惜售)"
            elif pattern.is_expanding:
                return False, "价跌量增(恐慌)"
            else:
                return True, "价跌量平"

        # 价格持平
        else:
            if pattern.is_contracting or pattern.is_light:
                return True, "价平量缩(整理)"
            return True, "价平量平"

    def calculate_volume_trend_strength(self, lookback: int = 10) -> float:
        """
        计算量能趋势强度

        Returns:
            0-1之间的值，越大表示趋势越明显
        """
        if len(self.volumes) < lookback:
            return 0.0

        recent_volumes = self.volumes[-lookback:]
        recent_ratios = self.volume_ratio[-lookback:]

        # 计算线性回归斜率
        x = np.arange(len(recent_ratios))
        slope, _ = np.polyfit(x, recent_ratios, 1)

        # 计算R²
        y_mean = np.mean(recent_ratios)
        ss_tot = np.sum((recent_ratios - y_mean) ** 2)
        y_pred = slope * x + (recent_ratios[0] - slope * 0)
        ss_res = np.sum((recent_ratios - y_pred) ** 2)

        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        # 综合斜率和拟合度
        strength = min(abs(slope) * 2, 1.0) * r_squared
        return max(0, min(strength, 1.0))

    def is_accumulation_phase(self, index: int = -1, lookback: int = 20) -> bool:
        """
        判断是否处于吸筹阶段

        吸筹特征：
        - 价格在相对低位震荡
        - 量能温和放大
        - 出现多次地量后放量

        Returns:
            是否处于吸筹阶段
        """
        if index < lookback:
            return False

        # 检查是否出现地量
        recent_light = sum(
            1 for i in range(index - lookback, index)
            if self.volume_ratio[i] < self.LIGHT_VOLUME_THRESHOLD
        )

        # 检查是否有温和放量
        recent_expand = sum(
            1 for i in range(index - lookback, index)
            if self.EXPAND_THRESHOLD < self.volume_ratio[i] < self.HEAVY_VOLUME_THRESHOLD
        )

        # 价格波动率
        price_range = (
            max(self.prices[index - lookback:index]) -
            min(self.prices[index - lookback:index])
        ) / np.mean(self.prices[index - lookback:index])

        # 吸筹判断：多次地量+温和放量+价格波动小
        return (
            recent_light >= 2 and
            recent_expand >= lookback // 3 and
            price_range < 0.15
        )

    def is_distribution_phase(self, index: int = -1, lookback: int = 20) -> bool:
        """
        判断是否处于派发阶段

        派发特征：
        - 价格在高位震荡
        - 巨量频现但价格不再上涨
        - 量价背离

        Returns:
            是否处于派发阶段
        """
        if index < lookback:
            return False

        # 检查巨量次数
        recent_heavy = sum(
            1 for i in range(index - lookback, index)
            if self.volume_ratio[i] > self.HEAVY_VOLUME_THRESHOLD
        )

        # 检查量价背离
        divergence = self.check_divergence(index, lookback)

        # 价格是否在高位
        price_level = np.mean(self.prices[index - lookback:index])
        historical_max = np.max(self.prices[:index + 1])
        is_high_level = price_level > historical_max * 0.9

        return (
            recent_heavy >= 2 and
            divergence.has_divergence and
            divergence.direction == 'top' and
            is_high_level
        )


def analyze_volume_pattern(
    df: pd.DataFrame,
    index: int = -1
) -> dict:
    """
    便捷函数：分析量能形态

    Args:
        df: 包含OHLCV的DataFrame
        index: 分析的K线索引

    Returns:
        包含量能分析结果的字典
    """
    if 'volume' not in df.columns or 'close' not in df.columns:
        return {}

    prices = df['close'].values
    volumes = df['volume'].values

    analyzer = VolumeAnalyzer(prices, volumes)

    pattern = analyzer.get_current_pattern(index)
    divergence = analyzer.check_divergence(index)

    return {
        'pattern': {
            'expanding': pattern.is_expanding,
            'contracting': pattern.is_contracting,
            'heavy': pattern.is_heavy,
            'light': pattern.is_light,
            'ratio': pattern.ratio_vs_avg,
            'trend': pattern.trend
        },
        'divergence': {
            'has': divergence.has_divergence,
            'direction': divergence.direction,
            'strength': divergence.strength
        },
        'accumulation': analyzer.is_accumulation_phase(index),
        'distribution': analyzer.is_distribution_phase(index)
    }
