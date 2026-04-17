"""
成交量过滤器

验证信号是否有足够的成交量配合。
"""

from typing import Tuple
import pandas as pd

from .base_filter import SignalFilter
from core.buy_sell_points import BuySellPoint


class VolumeFilter(SignalFilter):
    """
    成交量确认过滤器

    买入信号需要成交量配合：
    - 当前成交量 > 均量 × 阈值
    """

    def __init__(
        self,
        min_volume_ratio: float = 1.2,
        volume_ma_period: int = 20,
    ):
        """
        Args:
            min_volume_ratio: 最低量比阈值
            volume_ma_period: 成交量均线周期
        """
        self.min_volume_ratio = min_volume_ratio
        self.volume_ma_period = volume_ma_period

    def should_enter(
        self, signal: BuySellPoint, context: dict
    ) -> Tuple[bool, str]:
        df = context.get('df')
        if df is None or len(df) < self.volume_ma_period:
            return (True, '')  # 数据不足，放行

        # 使用最近3根K线的平均成交量（更稳健，避免单根K线偶然低量）
        tail_len = min(3, len(df))
        recent_vol = df['volume'].tail(tail_len).mean()
        vol_ma = df['volume'].tail(self.volume_ma_period).mean()

        if vol_ma <= 0:
            return (True, '')

        ratio = recent_vol / vol_ma
        if ratio < self.min_volume_ratio:
            return (
                False,
                f'量能不足: 近{tail_len}根量比{ratio:.2f} < 阈值{self.min_volume_ratio}'
            )

        return (True, '')
