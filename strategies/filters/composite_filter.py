"""
组合过滤器

将多个过滤器串联，所有过滤器都通过才放行。
"""

from typing import List, Tuple
from .base_filter import SignalFilter
from core.buy_sell_points import BuySellPoint


class CompositeFilter(SignalFilter):
    """
    组合过滤器

    串联多个过滤器，所有过滤器都通过才放行。
    按顺序执行，遇到第一个不通过即返回。
    """

    def __init__(self, filters: List[SignalFilter] = None):
        self.filters = filters or []

    def add(self, f: SignalFilter):
        """添加过滤器"""
        self.filters.append(f)

    def should_enter(
        self, signal: BuySellPoint, context: dict
    ) -> Tuple[bool, str]:
        for f in self.filters:
            passed, reason = f.should_enter(signal, context)
            if not passed:
                return (False, reason)
        return (True, '')
