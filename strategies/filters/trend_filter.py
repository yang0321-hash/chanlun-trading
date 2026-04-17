"""
趋势确认过滤器

在逆势买入时阻止信号，要求买入方向与均线趋势一致。
核心逻辑：
- 2买/3买：要求价格在MA20之上（顺势回踩，不是下跌中继）
- 1买：豁免（1买本身就是抄底，需要逆势）
- MACD方向确认：MACD柱状线应为正或即将转正
"""

from typing import Tuple
import pandas as pd
from .base_filter import SignalFilter
from core.buy_sell_points import BuySellPoint


class TrendAlignmentFilter(SignalFilter):
    """
    趋势对齐过滤器

    规则：
    - 2买/3买：价格必须在MA20上方（确认是上升趋势中的回调）
    - 1买：豁免（底背驰本身就是逆势信号）
    - 可选：MACD柱状线方向确认
    """

    def __init__(
        self,
        ma_period: int = 20,
        require_ma_above: bool = True,
        require_macd_turn: bool = False,
        strict_mode: bool = False,
    ):
        """
        Args:
            ma_period: 均线周期（默认20）
            require_ma_above: 是否要求价格在MA上方
            require_macd_turn: 是否要求MACD柱状线转正
            strict_mode: 严格模式要求价格在MA60上方
        """
        self.ma_period = ma_period
        self.require_ma_above = require_ma_above
        self.require_macd_turn = require_macd_turn
        self.strict_mode = strict_mode

    def should_enter(
        self, signal: BuySellPoint, context: dict
    ) -> Tuple[bool, str]:
        df = context.get('df')
        if df is None or len(df) < self.ma_period:
            return (True, '')  # 数据不足，放行

        # 1买豁免
        if signal.point_type == '1buy':
            return (True, '')

        current_price = context.get('price', df['close'].iloc[-1])

        # MA趋势检查
        if self.require_ma_above:
            ma = df['close'].rolling(self.ma_period).mean().iloc[-1]
            if pd.isna(ma):
                return (True, '')

            long_ma_period = 60 if self.strict_mode else self.ma_period
            if self.strict_mode and len(df) >= 60:
                ma60 = df['close'].rolling(60).mean().iloc[-1]
                if not pd.isna(ma60):
                    ma = min(ma, ma60)  # 严格模式：两者都检查

            if current_price < ma:
                return (
                    False,
                    f'趋势不利: 价格{current_price:.2f} < MA{self.ma_period}={ma:.2f} '
                    f'({signal.point_type}要求价格在MA上方)'
                )

        # MACD方向检查
        if self.require_macd_turn:
            macd_info = self._check_macd(df)
            if macd_info == 'negative_accel':
                return (
                    False,
                    f'MACD加速下跌中，不适合{signal.point_type}入场'
                )

        return (True, '')

    def _check_macd(self, df: pd.DataFrame) -> str:
        """检查MACD柱状线方向"""
        if len(df) < 35:
            return 'unknown'

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        dif = exp1 - exp2
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = dif - dea

        current_hist = hist.iloc[-1]
        prev_hist = hist.iloc[-2] if len(hist) >= 2 else 0

        # 柱状线加速下跌（连续负且越来越负）
        if current_hist < 0 and current_hist < prev_hist:
            return 'negative_accel'

        return 'ok'
