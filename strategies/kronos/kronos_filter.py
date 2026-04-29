"""
Kronos 信号确认过滤器

在缠论买卖点触发后，使用 Kronos AI 模型预测未来 N 根K线，
确认预测方向与信号方向一致后才放行。

降级策略：Kronos 不可用时 (无GPU/无依赖) 永不阻塞信号。
"""

from typing import Tuple

import pandas as pd
from loguru import logger

from strategies.filters.base_filter import SignalFilter
from core.buy_sell_points import BuySellPoint
from .kronos_config import KronosConfig
from .kronos_predictor import KronosPredictor


class KronosFilter(SignalFilter):
    """
    Kronos AI 预测确认过滤器

    买入确认：预测收盘价上涨超过 min_upside_pct 且预测最低价不低于止损位
    卖出确认：预测收盘价下跌
    豁免信号：1买 (抄底信号，Kronos 趋势预测可能不确认)
    """

    def __init__(
        self,
        predictor: KronosPredictor,
        pred_len: int = 5,
        min_upside_pct: float = 0.01,
        max_downside_pct: float = 0.02,
        exempt_types: Tuple[str, ...] = ('1buy',),
    ):
        self._predictor = predictor
        self._pred_len = pred_len
        self._min_upside_pct = min_upside_pct
        self._max_downside_pct = max_downside_pct
        self._exempt_types = exempt_types

    def should_enter(
        self, signal: BuySellPoint, context: dict
    ) -> Tuple[bool, str]:
        """
        判断信号是否通过 Kronos 确认

        Returns:
            (是否通过, 不通过时的原因)
        """
        # 1. Kronos 不可用 → 放行
        if not self._predictor.is_available():
            return (True, '')

        # 2. 豁免类型 → 放行
        if signal.point_type in self._exempt_types:
            return (True, '')

        # 3. 获取历史数据
        df = context.get('df')
        if df is None or len(df) < 10:
            return (True, '')  # 数据不足，不阻塞

        # 4. 获取时间戳
        if hasattr(df.index, 'to_series'):
            timestamps = df.index.to_series().reset_index(drop=True)
        else:
            timestamps = pd.Series(df.index)

        symbol = context.get('symbol', '')

        # 5. 调用预测
        pred_df = self._predictor.predict(
            df=df, timestamps=timestamps,
            pred_len=self._pred_len, symbol=symbol,
        )
        if pred_df is None:
            return (True, '')  # 预测失败，不阻塞

        current_price = signal.price
        stop_loss = signal.stop_loss

        # 6. 根据信号类型判断
        if signal.is_buy:
            return self._check_buy(pred_df, current_price, stop_loss)
        elif signal.is_sell:
            return self._check_sell(pred_df, current_price)
        else:
            return (True, '')

    def _check_buy(
        self, pred_df: pd.DataFrame, current_price: float, stop_loss: float
    ) -> Tuple[bool, str]:
        """买入信号确认"""
        pred_close_last = pred_df['close'].iloc[-1]
        pred_low_min = pred_df['low'].min()
        predicted_return = (pred_close_last - current_price) / current_price
        predicted_drawdown = (pred_low_min - current_price) / current_price

        # 检查预测涨幅
        if predicted_return < self._min_upside_pct:
            return (
                False,
                f'Kronos预测不确认: 预测涨幅{predicted_return:.2%} '
                f'< 最低要求{self._min_upside_pct:.2%}'
            )

        # 检查预测回撤 (仅当有止损位时)
        if stop_loss > 0 and pred_low_min < stop_loss:
            return (
                False,
                f'Kronos预测风险过高: 预测最低价{pred_low_min:.2f} '
                f'< 止损位{stop_loss:.2f}'
            )

        # 检查最大回撤
        if predicted_drawdown < -self._max_downside_pct:
            return (
                False,
                f'Kronos预测回撤过大: {predicted_drawdown:.2%} '
                f'< 最大允许{-self._max_downside_pct:.2%}'
            )

        return (True, '')

    def _check_sell(
        self, pred_df: pd.DataFrame, current_price: float
    ) -> Tuple[bool, str]:
        """卖出信号确认"""
        pred_close_last = pred_df['close'].iloc[-1]
        predicted_return = (pred_close_last - current_price) / current_price

        # 卖出信号需要预测下跌
        if predicted_return > self._min_upside_pct:
            return (
                False,
                f'Kronos预测不确认卖出: 预测涨幅{predicted_return:.2%}'
            )

        return (True, '')
