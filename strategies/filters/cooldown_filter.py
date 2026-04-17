"""
冷却期过滤器

在卖出后的一段时间内阻止再次买入。
"""

from typing import Dict, Tuple
from .base_filter import SignalFilter
from core.buy_sell_points import BuySellPoint


class CooldownFilter(SignalFilter):
    """
    冷却期过滤器

    卖出后N根K线内不再次买入，避免频繁交易。
    """

    def __init__(self, cooldown_bars: int = 10):
        """
        Args:
            cooldown_bars: 冷却期K线数
        """
        self.cooldown_bars = cooldown_bars
        self._last_sell_index: Dict[str, int] = {}

    def record_sell(self, symbol: str, bar_index: int):
        """记录卖出事件"""
        self._last_sell_index[symbol] = bar_index

    def should_enter(
        self, signal: BuySellPoint, context: dict
    ) -> Tuple[bool, str]:
        symbol = context.get('symbol', '')
        current_idx = context.get('bar_index', 0)

        last_sell = self._last_sell_index.get(symbol, -999)
        bars_since_sell = current_idx - last_sell

        if bars_since_sell < self.cooldown_bars:
            return (
                False,
                f'冷却期: 距上次卖出仅{bars_since_sell}根K线 < {self.cooldown_bars}'
            )

        return (True, '')
