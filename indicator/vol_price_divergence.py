"""
量价背离检测 — 4象限分类

从价格趋势和成交量趋势的关系判断资金流向:
- 价涨量缩 → 资金出逃（看跌信号）
- 价跌量增 → 资金进场（看涨信号）
- 放量上涨 → 多头强势
- 缩量下跌 → 空头衰竭
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DivergenceType(Enum):
    BULLISH_ACCUMULATION = '价跌量增'    # 资金进场
    BEARISH_DISTRIBUTION = '价涨量缩'    # 资金出逃
    STRONG_BULL = '放量上涨'             # 多头强势
    WEAK_BEAR = '缩量下跌'              # 空头衰竭
    NEUTRAL = '中性'


@dataclass
class VolumePriceDivergence:
    divergence_type: DivergenceType
    price_change_pct: float   # 价格变化百分比
    volume_change_pct: float  # 量能变化百分比
    vol_ratio: float          # 短期量/长期量
    strength: float           # 信号强度 0-1


def detect_volume_price_divergence(
    closes: list,
    volumes: list,
    short_period: int = 5,
    long_period: int = 20,
    price_threshold: float = 3.0,
    vol_threshold: float = 20.0,
) -> Optional[VolumePriceDivergence]:
    """
    检测量价背离

    Args:
        closes: 收盘价序列
        volumes: 成交量序列
        short_period: 短期均量周期
        long_period: 长期均量周期
        price_threshold: 价格变化阈值(%)
        vol_threshold: 量能变化阈值(%)
    """
    n = len(closes)
    if n < long_period + 1:
        return None

    # 价格变化 (short_period日)
    price_now = closes[-1]
    price_prev = closes[-short_period] if n >= short_period else closes[0]
    price_chg = (price_now - price_prev) / price_prev * 100 if price_prev > 0 else 0

    # 量能趋势
    vol_short = sum(volumes[-short_period:]) / short_period
    vol_long = sum(volumes[-long_period:]) / long_period
    vol_chg = (vol_short - vol_long) / vol_long * 100 if vol_long > 0 else 0
    vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

    # 4象限分类
    price_up = price_chg > price_threshold
    price_down = price_chg < -price_threshold
    vol_up = vol_chg > vol_threshold
    vol_down = vol_chg < -vol_threshold

    if price_up and vol_down:
        div_type = DivergenceType.BEARISH_DISTRIBUTION
        strength = min(1.0, abs(price_chg) / 10 * abs(vol_chg) / 40)
    elif price_down and vol_up:
        div_type = DivergenceType.BULLISH_ACCUMULATION
        strength = min(1.0, abs(price_chg) / 10 * abs(vol_chg) / 40)
    elif price_up and vol_up:
        div_type = DivergenceType.STRONG_BULL
        strength = min(1.0, abs(price_chg) / 10 * vol_ratio / 2)
    elif price_down and vol_down:
        div_type = DivergenceType.WEAK_BEAR
        strength = min(1.0, abs(price_chg) / 10 * (2 - min(vol_ratio, 2)))
    else:
        div_type = DivergenceType.NEUTRAL
        strength = 0.0

    return VolumePriceDivergence(
        divergence_type=div_type,
        price_change_pct=round(price_chg, 2),
        volume_change_pct=round(vol_chg, 1),
        vol_ratio=round(vol_ratio, 2),
        strength=round(strength, 3),
    )
