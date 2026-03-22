"""
多周期缠论日内策略

策略逻辑：
1. 日线判断趋势方向（日线2买确认上升趋势）
2. 5分钟级别寻找买卖点
3. 日内必须平仓
"""

from typing import Optional, Dict, List
from datetime import datetime, time
import pandas as pd

from core.kline import KLine
from core.fractal import FractalDetector, FractalType
from core.stroke import StrokeGenerator
from backtest.intraday_engine import Signal, SignalType


class MultiTimeframeChanLun:
    """
    多周期缠论日内策略

    核心思想：
    - 日线定大方向（只在上升趋势中做多）
    - 5分钟找精确入场点（底分型确认、MACD金叉）
    - 严格止损和日内平仓
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.01,      # 止损 1%
        take_profit_pct: float = 0.02,    # 止盈 2%
        min_bars_confirm: int = 3,        # 分型确认K线数
        force_close_time: time = time(14, 50)
    ):
        """
        初始化策略

        Args:
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            min_bars_confirm: 分型确认需要的K线数
            force_close_time: 强制平仓时间
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_bars_confirm = min_bars_confirm
        self.force_close_time = force_close_time

        # 策略状态
        self.daily_trend = "neutral"  # up, down, neutral
        self.last_buy_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

    def prepare_daily_data(self, daily_df: pd.DataFrame):
        """
        准备日线数据，判断趋势

        Args:
            daily_df: 日线数据
        """
        if daily_df is None or len(daily_df) < 20:
            self.daily_trend = "neutral"
            return

        # 创建K线对象
        kline = KLine.from_dataframe(daily_df, strict_mode=True)

        # 识别分型
        detector = FractalDetector(kline, confirm_required=True)

        # 生成笔
        stroke_gen = StrokeGenerator(kline)
        strokes = stroke_gen.get_strokes()

        if not strokes:
            self.daily_trend = "neutral"
            return

        # 判断趋势：最后一笔方向
        last_stroke = strokes[-1]
        self.daily_trend = "up" if last_stroke.is_up else "down"

        # 检查是否有日线2买（底分型不破前低）
        bottom_fractals = detector.get_bottom_fractals()
        if len(bottom_fractals) >= 2:
            # 简化：如果最后两个底分型依次抬高，视为上升趋势
            if bottom_fractals[-1].low > bottom_fractals[-2].low:
                self.daily_trend = "up"

    def generate_signal(
        self,
        bar: pd.Series,
        context: Dict,
        daily_context: Optional[Dict],
        kline_5min: KLine = None
    ) -> Optional[Signal]:
        """
        生成交易信号

        Args:
            bar: 当前K线
            context: 上下文信息
            daily_context: 日线上下文
            kline_5min: 5分钟K线对象

        Returns:
            交易信号或None
        """
        bar_time = context['datetime']
        bar_price = bar['close']

        # 检查止损止盈
        if self.last_buy_price and self.stop_loss_price:
            if bar['low'] <= self.stop_loss_price:
                return Signal(
                    type=SignalType.SELL,
                    datetime=bar_time,
                    price=self.stop_loss_price,
                    reason=f"止损 ({self.stop_loss_pct:.1%})"
                )

            if bar['high'] >= self.take_profit_price:
                return Signal(
                    type=SignalType.SELL,
                    datetime=bar_time,
                    price=self.take_profit_price,
                    reason=f"止盈 ({self.take_profit_pct:.1%})"
                )

        # 检查强制平仓时间
        if bar_time.time() >= self.force_close_time:
            if context['position'] > 0:
                return Signal(
                    type=SignalType.SELL,
                    datetime=bar_time,
                    price=bar_price,
                    reason="强制平仓"
                )

        # 检查日线趋势
        if daily_context:
            if daily_context.get('trend') == 'down':
                # 日线下降趋势，不开多仓
                return None

        # 只有多仓策略
        if context['position'] > 0:
            # 已有持仓，检查卖出信号
            return self._check_sell_signal(bar, context, kline_5min)
        else:
            # 无持仓，检查买入信号
            return self._check_buy_signal(bar, context, kline_5min)

    def _check_buy_signal(
        self,
        bar: pd.Series,
        context: Dict,
        kline_5min: KLine = None
    ) -> Optional[Signal]:
        """检查买入信号"""

        # 简单策略：5分钟底分型确认
        if kline_5min and len(kline_5min) >= 3:
            detector = FractalDetector(kline_5min, confirm_required=False)
            bottom_fractals = detector.get_bottom_fractals()

            if bottom_fractals:
                last_bottom = bottom_fractals[-1]

                # 检查是否是最近的分型（最近5根K线内）
                current_idx = len(kline_5min) - 1
                if current_idx - last_bottom.index <= 5:
                    # 确认分型：后续K线没有破坏
                    confirmed = self._check_fractal_confirmed(
                        kline_5min, last_bottom, is_bottom=True
                    )

                    if confirmed:
                        # 当前价格在分型低点之上
                        if bar['close'] > last_bottom.low:
                            self.last_buy_price = bar['close']
                            self.stop_loss_price = bar['close'] * (1 - self.stop_loss_pct)
                            self.take_profit_price = bar['close'] * (1 + self.take_profit_pct)

                            return Signal(
                                type=SignalType.BUY,
                                datetime=context['datetime'],
                                price=bar['close'],
                                reason=f"5min底分型确认 ({last_bottom.low:.2f})"
                            )

        return None

    def _check_sell_signal(
        self,
        bar: pd.Series,
        context: Dict,
        kline_5min: KLine = None
    ) -> Optional[Signal]:
        """检查卖出信号"""

        if not kline_5min or len(kline_5min) < 3:
            return None

        # 策略1: 顶分型确认
        detector = FractalDetector(kline_5min, confirm_required=False)
        top_fractals = detector.get_top_fractals()

        if top_fractals:
            last_top = top_fractals[-1]
            current_idx = len(kline_5min) - 1

            if current_idx - last_top.index <= 5:
                confirmed = self._check_fractal_confirmed(
                    kline_5min, last_top, is_bottom=False
                )

                if confirmed and bar['close'] < last_top.high:
                    self._reset_levels()
                    return Signal(
                        type=SignalType.SELL,
                        datetime=context['datetime'],
                        price=bar['close'],
                        reason=f"5min顶分型确认 ({last_top.high:.2f})"
                    )

        # 策略2: 盈利回撤（盈利后回撤一半平仓）
        if self.last_buy_price:
            pnl_pct = (bar['close'] - self.last_buy_price) / self.last_buy_price

            if pnl_pct > 0.005:  # 盈利超过0.5%
                # 回撤到盈利的一半时平仓
                if bar['close'] < self.last_buy_price * (1 + pnl_pct / 2):
                    self._reset_levels()
                    return Signal(
                        type=SignalType.SELL,
                        datetime=context['datetime'],
                        price=bar['close'],
                        reason="盈利回撤止盈"
                    )

        return None

    def _check_fractal_confirmed(
        self,
        kline: KLine,
        fractal,
        is_bottom: bool
    ) -> bool:
        """检查分型是否被确认"""
        start_idx = fractal.index + 2
        if start_idx >= len(kline):
            return False

        check_count = min(3, len(kline) - start_idx)

        for i in range(start_idx, start_idx + check_count):
            k = kline[i]

            if is_bottom:
                # 底分型确认：收盘价突破分型高点
                if k.close > fractal.high:
                    return True
            else:
                # 顶分型确认：收盘价跌破分型低点
                if k.close < fractal.low:
                    return True

        return False

    def _reset_levels(self):
        """重置买卖点"""
        self.last_buy_price = None
        self.stop_loss_price = None
        self.take_profit_price = None


def create_strategy_fn(min_bars: int = 20):
    """
    创建策略函数（用于回测引擎）

    Args:
        min_bars: 最小K线数
    """
    strategy = MultiTimeframeChanLun()
    kline_5min = KLine(data=[])

    def strategy_fn(bar, context, daily_context):
        nonlocal kline_5min

        # 添加当前K线
        from core.kline import KLineData
        new_bar = KLineData(
            datetime=context['datetime'],
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar.get('volume', 0),
            amount=bar.get('amount', 0)
        )

        # 更新K线对象
        if len(kline_5min.raw_data) == 0:
            kline_5min = KLine(data=[new_bar], strict_mode=True)
        else:
            kline_5min.raw_data.append(new_bar)
            kline_5min._process()

        # 至少需要一定数量的K线才能分析
        if len(kline_5min) < min_bars:
            return None

        return strategy.generate_signal(bar, context, daily_context, kline_5min)

    return strategy_fn
