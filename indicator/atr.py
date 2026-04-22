"""
ATR指标模块 (Average True Range)

用于动态止损和仓位管理
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

try:
    from talib import abstract as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


@dataclass
class ATRValue:
    """单个ATR值"""
    datetime: pd.Timestamp
    atr: float    # ATR值
    tr: float     # True Range值
    price: float  # 收盘价


class ATR:
    """
    ATR指标计算类

    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = EMA(TR, period)
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ):
        """
        初始化ATR计算器

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: ATR周期 (默认14)
        """
        self.high = high
        self.low = low
        self.close = close
        self.period = period
        self.values: List[ATRValue] = []
        self._calculate()

    def _calculate(self) -> None:
        """计算ATR指标"""
        if TALIB_AVAILABLE:
            self._calculate_with_talib()
        else:
            self._calculate_manual()

    def _calculate_with_talib(self) -> None:
        """使用TA-Lib计算ATR"""
        high = np.array(self.high, dtype=float)
        low = np.array(self.low, dtype=float)
        close = np.array(self.close, dtype=float)

        atr_values = ta.ATR(high, low, close, timeperiod=self.period)

        for i, price in enumerate(self.close):
            if not pd.isna(atr_values[i]):
                self.values.append(ATRValue(
                    datetime=self.close.index[i] if hasattr(self.close, 'index') else i,
                    atr=float(atr_values[i]),
                    tr=float(high[i] - low[i]) if i == 0 else float(max(
                        high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1])
                    )),
                    price=float(price)
                ))

    def _calculate_manual(self) -> None:
        """手动计算ATR"""
        # True Range
        prev_close = self.close.shift(1)
        tr1 = self.high - self.low
        tr2 = (self.high - prev_close).abs()
        tr3 = (self.low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = EMA(TR)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        for i in range(len(self.close)):
            if not pd.isna(atr.iloc[i]) and not pd.isna(tr.iloc[i]):
                self.values.append(ATRValue(
                    datetime=self.close.index[i] if hasattr(self.close, 'index') else i,
                    atr=float(atr.iloc[i]),
                    tr=float(tr.iloc[i]),
                    price=float(self.close.iloc[i])
                ))

    def get_value_at(self, index: int) -> Optional[ATRValue]:
        """获取指定位置的ATR值"""
        if 0 <= index < len(self.values):
            return self.values[index]
        return None

    def get_latest(self) -> Optional[ATRValue]:
        """获取最新的ATR值"""
        if self.values:
            return self.values[-1]
        return None

    def get_atr_series(self) -> pd.Series:
        """获取ATR序列"""
        return pd.Series([v.atr for v in self.values])

    def get_tr_series(self) -> pd.Series:
        """获取True Range序列"""
        return pd.Series([v.tr for v in self.values])

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> ATRValue:
        return self.values[index]


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> ATR:
    """
    便捷函数：计算ATR指标

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ATR周期

    Returns:
        ATR对象
    """
    return ATR(high, low, close, period)
