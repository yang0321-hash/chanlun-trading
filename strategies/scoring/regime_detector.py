"""
市场状态检测器

使用ADX+ATR+MA判断当前市场状态：
- strong_trend: ADX > 30，方向明确
- mild_trend: 20 < ADX < 30
- sideways: ADX < 20，低波动
- volatile: ATR在80分位以上
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """市场状态"""
    STRONG_TREND = 'strong_trend'   # 强趋势
    MILD_TREND = 'mild_trend'       # 弱趋势
    SIDEWAYS = 'sideways'           # 震荡
    VOLATILE = 'volatile'           # 高波动


@dataclass
class RegimeInfo:
    """市场状态信息"""
    regime: MarketRegime
    adx: float                      # ADX值
    trend_direction: str            # 'up', 'down', 'neutral'
    volatility_pct: float           # ATR占价格的百分比
    volatility_rank: float          # ATR在历史中的分位 (0-1)


class MarketRegimeDetector:
    """
    市场状态检测器

    使用方法：
        detector = MarketRegimeDetector(df)
        info = detector.detect()
        print(info.regime, info.trend_direction)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        adx_period: int = 14,
        atr_period: int = 14,
        ma_period: int = 60,
    ):
        """
        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume)
            adx_period: ADX计算周期
            atr_period: ATR计算周期
            ma_period: 趋势MA周期
        """
        self.df = df
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.ma_period = ma_period

    def detect(self) -> RegimeInfo:
        """
        检测当前市场状态

        Returns:
            RegimeInfo
        """
        adx = self._calculate_adx()
        atr_pct, atr_rank = self._calculate_volatility()
        trend_dir = self._detect_trend_direction()

        # 判断状态
        if atr_rank > 0.8:
            regime = MarketRegime.VOLATILE
        elif adx > 30:
            regime = MarketRegime.STRONG_TREND
        elif adx > 20:
            regime = MarketRegime.MILD_TREND
        else:
            regime = MarketRegime.SIDEWAYS

        return RegimeInfo(
            regime=regime,
            adx=adx,
            trend_direction=trend_dir,
            volatility_pct=atr_pct,
            volatility_rank=atr_rank,
        )

    def _calculate_adx(self) -> float:
        """计算ADX（平均趋向指数）"""
        if len(self.df) < self.adx_period * 3:
            return 20.0  # 默认值

        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values

        # True Range
        tr = np.zeros(len(self.df))
        for i in range(1, len(self.df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # +DM / -DM
        plus_dm = np.zeros(len(self.df))
        minus_dm = np.zeros(len(self.df))
        for i in range(1, len(self.df)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # 平滑
        period = self.adx_period
        atr_smooth = self._wilder_smooth(tr, period)
        plus_di = self._wilder_smooth(plus_dm, period)
        minus_di = self._wilder_smooth(minus_dm, period)

        # +DI / -DI
        with np.errstate(divide='ignore', invalid='ignore'):
            pos_di = np.where(atr_smooth > 0, plus_di / atr_smooth * 100, 0)
            neg_di = np.where(atr_smooth > 0, minus_di / atr_smooth * 100, 0)

        # DX
        di_sum = pos_di + neg_di
        with np.errstate(divide='ignore', invalid='ignore'):
            dx = np.where(di_sum > 0, np.abs(pos_di - neg_di) / di_sum * 100, 0)

        # ADX = DX的平滑
        if len(dx) < period * 2:
            return 20.0

        adx = self._wilder_smooth(dx, period)
        return float(adx[-1]) if len(adx) > 0 else 20.0

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder平滑（类似EMA但alpha=1/period）"""
        result = np.zeros_like(data)
        if len(data) < period:
            return result

        result[period - 1] = np.mean(data[1:period])
        for i in range(period, len(data)):
            result[i] = result[i - 1] - result[i - 1] / period + data[i]
        return result

    def _calculate_volatility(self) -> Tuple[float, float]:
        """
        计算波动率

        Returns:
            (atr占价格百分比, atr在历史中的分位)
        """
        if len(self.df) < self.atr_period + 1:
            return (0.0, 0.5)

        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values

        # ATR
        tr = np.zeros(len(self.df))
        for i in range(1, len(self.df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # 滚动ATR
        atr_series = np.zeros(len(self.df))
        for i in range(self.atr_period, len(self.df)):
            atr_series[i] = np.mean(tr[i - self.atr_period + 1:i + 1])

        # ATR占价格的百分比
        current_close = close[-1]
        current_atr = atr_series[-1]
        atr_pct = current_atr / current_close * 100 if current_close > 0 else 0

        # ATR在历史中的分位
        valid_atr = atr_series[atr_series > 0]
        if len(valid_atr) > 20:
            atr_rank = float(np.mean(valid_atr <= current_atr))
        else:
            atr_rank = 0.5

        return (atr_pct, atr_rank)

    def _detect_trend_direction(self) -> str:
        """检测趋势方向（基于MA和价格关系）"""
        if len(self.df) < self.ma_period:
            return 'neutral'

        close = self.df['close'].values
        ma = np.mean(close[-self.ma_period:])
        current = close[-1]

        # 短期MA
        if len(close) >= 20:
            ma_short = np.mean(close[-20:])
        else:
            ma_short = current

        if ma_short > ma * 1.02 and current > ma_short:
            return 'up'
        elif ma_short < ma * 0.98 and current < ma_short:
            return 'down'
        else:
            return 'neutral'
