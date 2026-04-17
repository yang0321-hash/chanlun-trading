"""
市场状态过滤器

在不利的市场状态下阻止买入信号。
"""

from typing import Tuple
from strategies.scoring.regime_detector import MarketRegime, MarketRegimeDetector
from .base_filter import SignalFilter
from core.buy_sell_points import BuySellPoint


class RegimeFilter(SignalFilter):
    """
    市场状态过滤器

    规则：
    - 强下跌趋势中阻止买入（除非是1买，可能有反转机会）
    - 高波动状态下提高买入门槛
    """

    def __init__(
        self,
        block_strong_down: bool = True,
        block_volatile: bool = False,
        volatile_min_confidence: float = 0.7,
    ):
        self.block_strong_down = block_strong_down
        self.block_volatile = block_volatile
        self.volatile_min_confidence = volatile_min_confidence

    def should_enter(
        self, signal: BuySellPoint, context: dict
    ) -> Tuple[bool, str]:
        regime_info = context.get('regime_info')
        if regime_info is None:
            return (True, '')  # 无状态信息，放行

        # 强下跌趋势中阻止买入（1买除外）
        if (self.block_strong_down
                and regime_info.regime == MarketRegime.STRONG_TREND
                and regime_info.trend_direction == 'down'
                and signal.point_type != '1buy'):
            return (
                False,
                f'强下跌趋势中禁止买入({signal.point_type})'
            )

        # 高波动状态提高门槛
        if (self.block_volatile
                and regime_info.regime == MarketRegime.VOLATILE
                and signal.confidence < self.volatile_min_confidence):
            return (
                False,
                f'高波动状态: 置信度{signal.confidence:.2f} < {self.volatile_min_confidence}'
            )

        return (True, '')
