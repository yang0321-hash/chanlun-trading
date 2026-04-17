"""
过滤器基类
"""

from abc import ABC, abstractmethod
from typing import Tuple
from core.buy_sell_points import BuySellPoint


class SignalFilter(ABC):
    """信号过滤器基类"""

    @abstractmethod
    def should_enter(
        self, signal: BuySellPoint, context: dict
    ) -> Tuple[bool, str]:
        """
        判断信号是否应通过过滤器

        Args:
            signal: 买卖点信号
            context: 上下文信息（包含df, price, volume等）

        Returns:
            (是否通过, 如果不通过则给出原因)
        """
        ...
