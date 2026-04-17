"""
MACD指标模块

用于辅助判断背驰
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from talib import abstract as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


@dataclass
class MACDValue:
    """单个MACD值"""
    datetime: pd.Timestamp
    macd: float  # DIF值
    signal: float  # DEA值
    histogram: float  # MACD柱线
    price: float  # 价格

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'datetime': self.datetime,
            'macd': self.macd,
            'signal': self.signal,
            'histogram': self.histogram,
            'price': self.price
        }


class MACD:
    """
    MACD指标计算类

    MACD (Moving Average Convergence Divergence)
    - DIF = EMA(12) - EMA(26)
    - DEA = EMA(DIF, 9)
    - MACD = (DIF - DEA) * 2
    """

    def __init__(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        """
        初始化MACD计算器

        Args:
            prices: 价格序列（通常是收盘价）
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线EMA周期
        """
        self.prices = prices
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.values: List[MACDValue] = []
        self._calculate()

    def _calculate(self) -> None:
        """计算MACD指标"""
        if TALIB_AVAILABLE:
            self._calculate_with_talib()
        else:
            self._calculate_manual()

    def _calculate_with_talib(self) -> None:
        """使用TA-Lib计算MACD"""
        # 准备数据
        prices = np.array(self.prices, dtype=float)

        # 使用TA-Lib计算
        macd, signal, hist = ta.MACD(
            prices,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )

        # 构建结果
        for i, price in enumerate(self.prices):
            if not pd.isna(macd[i]) and not pd.isna(signal[i]):
                self.values.append(MACDValue(
                    datetime=self.prices.index[i] if hasattr(self.prices, 'index') else i,
                    macd=float(macd[i]),
                    signal=float(signal[i]),
                    histogram=float(hist[i] if not pd.isna(hist[i]) else 0),
                    price=float(price)
                ))

    def _calculate_manual(self) -> None:
        """手动计算MACD"""
        # 计算EMA
        def ema(series, period):
            return series.ewm(span=period, adjust=False).mean()

        prices = self.prices
        ema_fast = ema(prices, self.fast_period)
        ema_slow = ema(prices, self.slow_period)

        # DIF
        dif = ema_fast - ema_slow

        # DEA
        dea = ema(dif, self.signal_period)

        # MACD柱线
        macd_hist = (dif - dea) * 2

        # 构建结果
        for i, price in enumerate(prices):
            if not pd.isna(dif.iloc[i]) and not pd.isna(dea.iloc[i]):
                self.values.append(MACDValue(
                    datetime=prices.index[i] if hasattr(prices, 'index') else i,
                    macd=float(dif.iloc[i]),
                    signal=float(dea.iloc[i]),
                    histogram=float(macd_hist.iloc[i]),
                    price=float(price)
                ))

    def get_value_at(self, index: int) -> Optional[MACDValue]:
        """
        获取指定位置的MACD值

        Args:
            index: 索引位置

        Returns:
            MACD值对象
        """
        if 0 <= index < len(self.values):
            return self.values[index]
        return None

    def get_latest(self) -> Optional[MACDValue]:
        """获取最新的MACD值"""
        if self.values:
            return self.values[-1]
        return None

    def get_dif_series(self) -> pd.Series:
        """获取DIF序列"""
        return pd.Series([v.macd for v in self.values])

    def get_dea_series(self) -> pd.Series:
        """获取DEA序列"""
        return pd.Series([v.signal for v in self.values])

    def get_histogram_series(self) -> pd.Series:
        """获取MACD柱线序列"""
        return pd.Series([v.histogram for v in self.values])

    def check_divergence(
        self,
        start_idx: int,
        end_idx: int,
        direction: str,
        prev_start: int = None,
        prev_end: int = None
    ) -> Tuple[bool, float]:
        """
        检查是否存在背驰（缠论双条件）

        缠论原文（第24课 MACD定律）：
        - 背驰条件：黄白线(DIF)背离 + 柱状面积减少，两者必须同时满足
        - 一买总是在0轴下方背驰形成
        - 一卖总是在0轴上方背驰形成

        检测逻辑：
        - 顶背驰：价格创新高，但DIF不创新高 AND 红柱面积减少
        - 底背驰：价格创新低，但DIF不创新低 AND 绿柱面积减少

        Args:
            start_idx: 当前趋势段起始索引
            end_idx: 当前趋势段结束索引
            direction: 方向 ('up'检查顶背驰, 'down'检查底背驰)
            prev_start: 上一趋势段起始索引（None则自动回溯等长区间）
            prev_end: 上一趋势段结束索引

        Returns:
            (是否背驰, 背驰强度)
        """
        if end_idx >= len(self.values) or start_idx < 0:
            return (False, 0)

        if direction == 'up':
            # 顶背驰：价格新高，DIF不新高 + 红柱面积减少
            price_high = max(self.values[i].price for i in range(start_idx, end_idx + 1))
            dif_high = max(self.values[i].macd for i in range(start_idx, end_idx + 1))

            # 确定前一波区间
            if prev_start is not None and prev_end is not None:
                p_start = max(0, prev_start)
                p_end = min(len(self.values) - 1, prev_end)
            else:
                wave_length = end_idx - start_idx + 1
                p_end = max(0, start_idx - 1)
                p_start = max(0, p_end - wave_length + 1)

            if p_start >= p_end or p_start < 0:
                return (False, 0)

            prev_price_high = max(self.values[i].price for i in range(p_start, p_end + 1))
            prev_dif_high = max(self.values[i].macd for i in range(p_start, p_end + 1))

            # 条件1：价格创新高
            if price_high <= prev_price_high:
                return (False, 0)

            # 条件2：DIF不创新高（黄白线背离）
            dif_divergence = dif_high < prev_dif_high

            # 条件3：红柱面积减少
            curr_area = self.compute_area(start_idx, end_idx, 'up')
            prev_area = self.compute_area(p_start, p_end, 'up')
            area_divergence = curr_area < prev_area if prev_area > 0 else False

            # 缠论要求：DIF背离 AND 面积减少（两者同时满足）
            if dif_divergence and area_divergence:
                dif_strength = (prev_dif_high - dif_high) / prev_dif_high if prev_dif_high != 0 else 0
                area_strength = (prev_area - curr_area) / prev_area if prev_area > 0 else 0
                strength = (dif_strength + area_strength) / 2
                return (True, strength)

            # 缠论严格要求双条件，仅DIF背离不算背驰
            # 返回False，不使用宽松模式

        else:
            # 底背驰：价格新低，DIF不新低 + 绿柱面积减少
            price_low = min(self.values[i].price for i in range(start_idx, end_idx + 1))
            dif_low = min(self.values[i].macd for i in range(start_idx, end_idx + 1))

            # 确定前一波区间
            if prev_start is not None and prev_end is not None:
                p_start = max(0, prev_start)
                p_end = min(len(self.values) - 1, prev_end)
            else:
                wave_length = end_idx - start_idx + 1
                p_end = max(0, start_idx - 1)
                p_start = max(0, p_end - wave_length + 1)

            if p_start >= p_end or p_start < 0:
                return (False, 0)

            prev_price_low = min(self.values[i].price for i in range(p_start, p_end + 1))
            prev_dif_low = min(self.values[i].macd for i in range(p_start, p_end + 1))

            # 条件1：价格创新低
            if price_low >= prev_price_low:
                return (False, 0)

            # 条件2：DIF不创新低（黄白线背离）
            dif_divergence = dif_low > prev_dif_low

            # 条件3：绿柱面积减少
            curr_area = self.compute_area(start_idx, end_idx, 'down')
            prev_area = self.compute_area(p_start, p_end, 'down')
            area_divergence = curr_area < prev_area if prev_area > 0 else False

            # 缠论要求：DIF背离 AND 面积减少（两者同时满足）
            if dif_divergence and area_divergence:
                dif_strength = (dif_low - prev_dif_low) / abs(prev_dif_low) if prev_dif_low != 0 else 0
                area_strength = (prev_area - curr_area) / prev_area if prev_area > 0 else 0
                strength = (dif_strength + area_strength) / 2
                return (True, strength)

            # 缠论严格要求双条件，仅DIF背离不算背驰
            # 返回False，不使用宽松模式

        return (False, 0)

    def compute_area(
        self,
        start_idx: int,
        end_idx: int,
        direction: str = 'auto'
    ) -> float:
        """
        计算MACD柱线面积

        用于比较不同笔的MACD力度，是中枢背驰检测的基础。

        Args:
            start_idx: 起始索引
            end_idx: 结束索引
            direction: 'up'只计算红柱, 'down'只计算绿柱, 'auto'全部绝对值

        Returns:
            MACD柱线面积
        """
        if start_idx < 0 or not self.values or start_idx > end_idx:
            return 0.0

        end_idx = min(end_idx, len(self.values) - 1)
        area = 0.0
        for i in range(start_idx, end_idx + 1):
            h = self.values[i].histogram
            if direction == 'up':
                area += max(0, h)
            elif direction == 'down':
                area += max(0, -h)
            else:
                area += abs(h)
        return area

    def check_golden_cross(self) -> bool:
        """
        检查金叉（DIF上穿DEA）

        Returns:
            是否金叉
        """
        if len(self.values) < 2:
            return False

        latest = self.values[-1]
        prev = self.values[-2]

        return (prev.macd <= prev.signal and
                latest.macd > latest.signal)

    def check_death_cross(self) -> bool:
        """
        检查死叉（DIF下穿DEA）

        Returns:
            是否死叉
        """
        if len(self.values) < 2:
            return False

        latest = self.values[-1]
        prev = self.values[-2]

        return (prev.macd >= prev.signal and
                latest.macd < latest.signal)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> MACDValue:
        return self.values[index]


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> MACD:
    """
    便捷函数：计算MACD指标

    Args:
        prices: 价格序列
        fast_period: 快速周期
        slow_period: 慢速周期
        signal_period: 信号周期

    Returns:
        MACD对象
    """
    return MACD(prices, fast_period, slow_period, signal_period)


def check_macd_divergence(
    macd: MACD,
    price_highs: List[Tuple[int, float]],
    price_lows: List[Tuple[int, float]],
    direction: str
) -> List[Tuple[int, bool, float]]:
    """
    检查指定价格点是否存在MACD背驰

    Args:
        macd: MACD对象
        price_highs: 价格高点列表 [(index, price), ...]
        price_lows: 价格低点列表 [(index, price), ...]
        direction: 'up' 检查顶背驰, 'down' 检查底背驰

    Returns:
        [(index, 是否背驰, 背驰强度), ...]
    """
    results = []

    if direction == 'up' and len(price_highs) >= 2:
        # 检查相邻两个高点
        for i in range(len(price_highs) - 1):
            idx1, price1 = price_highs[i]
            idx2, price2 = price_highs[i + 1]

            if price2 > price1:  # 价格创新高
                macd1 = macd.get_value_at(idx1)
                macd2 = macd.get_value_at(idx2)

                if macd1 and macd2:
                    if macd2.macd < macd1.macd:  # MACD没有创新高
                        strength = (macd1.macd - macd2.macd) / abs(macd1.macd) if macd1.macd != 0 else 0
                        results.append((idx2, True, strength))
                    else:
                        results.append((idx2, False, 0))

    elif direction == 'down' and len(price_lows) >= 2:
        # 检查相邻两个低点
        for i in range(len(price_lows) - 1):
            idx1, price1 = price_lows[i]
            idx2, price2 = price_lows[i + 1]

            if price2 < price1:  # 价格创新低
                macd1 = macd.get_value_at(idx1)
                macd2 = macd.get_value_at(idx2)

                if macd1 and macd2:
                    if macd2.macd > macd1.macd:  # MACD没有创新低
                        strength = (macd2.macd - macd1.macd) / abs(macd1.macd) if macd1.macd != 0 else 0
                        results.append((idx2, True, strength))
                    else:
                        results.append((idx2, False, 0))

    return results
