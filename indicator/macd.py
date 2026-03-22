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
        direction: str
    ) -> Tuple[bool, float]:
        """
        检查是否存在背驰

        背驰定义：
        - 顶背驰：价格创新高，但MACD没有创新高（红柱缩短或DIF下降）
        - 底背驰：价格创新低，但MACD没有创新低（绿柱缩短或DIF上升）

        Args:
            start_idx: 起始索引
            end_idx: 结束索引
            direction: 方向 ('up'检查顶背驰, 'down'检查底背驰)

        Returns:
            (是否背驰, 背驰强度)
        """
        if end_idx >= len(self.values) or start_idx < 0:
            return (False, 0)

        if direction == 'up':
            # 顶背驰：价格新高，MACD不新高
            price_high = max(self.values[i].price for i in range(start_idx, end_idx + 1))
            macd_high = max(self.values[i].macd for i in range(start_idx, end_idx + 1))

            # 比较前一波
            if start_idx > 10:
                prev_price_high = max(self.values[i].price for i in range(start_idx - 10, start_idx))
                prev_macd_high = max(self.values[i].macd for i in range(start_idx - 10, start_idx))

                if price_high > prev_price_high and macd_high < prev_macd_high:
                    # 背驰强度 = (前一波MACD - 当前MACD) / 前一波MACD
                    strength = (prev_macd_high - macd_high) / prev_macd_high if prev_macd_high != 0 else 0
                    return (True, strength)
        else:
            # 底背驰：价格新低，MACD不新低
            price_low = min(self.values[i].price for i in range(start_idx, end_idx + 1))
            macd_low = min(self.values[i].macd for i in range(start_idx, end_idx + 1))

            # 比较前一波
            if start_idx > 10:
                prev_price_low = min(self.values[i].price for i in range(start_idx - 10, start_idx))
                prev_macd_low = min(self.values[i].macd for i in range(start_idx - 10, start_idx))

                if price_low < prev_price_low and macd_low > prev_macd_low:
                    # 背驰强度 = (当前MACD - 前一波MACD) / 前一波MACD
                    strength = (macd_low - prev_macd_low) / abs(prev_macd_low) if prev_macd_low != 0 else 0
                    return (True, strength)

        return (False, 0)

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
