"""
多级别缠论策略 - 周线+日线

交易规则：
1. 周线级别2买买入
2. 跌破1买低点止损
3. 日线级别MACD顶背离减仓50%
4. 日线级别2卖卖出剩余50%
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD


class WeeklyDailyChanLunStrategy(Strategy):
    """
    周线+日线 多级别缠论策略
    """

    def __init__(
        self,
        name: str = '周日线缠论策略',
        weekly_min_strokes: int = 3,
        daily_min_strokes: int = 3,
        stop_loss_pct: float = 0.08,        # 8%止损
        exit_ratio: float = 0.5,             # 减仓50%
    ):
        super().__init__(name)
        self.weekly_min_strokes = weekly_min_strokes
        self.daily_min_strokes = daily_min_strokes
        self.stop_loss_pct = stop_loss_pct
        self.exit_ratio = exit_ratio

        # 周线数据
        self._weekly_data: Optional[pd.DataFrame] = None
        self._weekly_fractals: List[Fractal] = []
        self._weekly_strokes: List[Stroke] = []
        self._weekly_pivots: List[Pivot] = []
        self._weekly_first_buy_price: Optional[float] = None
        self._weekly_second_buy_price: Optional[float] = None

        # 日线数据
        self._daily_data: Optional[pd.DataFrame] = None
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_macd: Optional[MACD] = None
        self._daily_second_sell_detected: bool = False

        # 缓存
        self._last_weekly_count: int = 0
        self._last_daily_count: int = 0

        # 持仓记录
        self._position_record: Dict[str, Dict] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金{capital:,.0f}")

    def reset(self) -> None:
        """重置策略"""
        super().reset()
        self._position_record = {}
        self._weekly_first_buy_price = None
        self._weekly_second_buy_price = None

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        daily_df = context['data'].get(symbol)
        if daily_df is None or len(daily_df) < 100:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 生成周线数据
        weekly_df = self._convert_to_weekly(daily_df)

        # 更新分析
        self._update_weekly_analysis(weekly_df)
        self._update_daily_analysis(daily_df)

        # 已有持仓：检查卖出信号
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar, daily_df)

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar)

    def _convert_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """将日线数据转换为周线数据"""
        # 按周重采样
        weekly = daily_df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }).dropna()

        return weekly

    def _update_weekly_analysis(self, df: pd.DataFrame) -> None:
        """更新周线分析"""
        if len(df) == self._last_weekly_count:
            return

        kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(kline, confirm_required=False)
        self._weekly_fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(kline, self._weekly_fractals, min_bars=self.weekly_min_strokes)
        self._weekly_strokes = stroke_gen.get_strokes()

        # 识别中枢
        pivot_detector = PivotDetector(kline, self._weekly_strokes)
        self._weekly_pivots = pivot_detector.get_pivots()

        # 找出买卖点
        self._find_weekly_buy_points()

        self._last_weekly_count = len(df)

    def _update_daily_analysis(self, df: pd.DataFrame) -> None:
        """更新日线分析"""
        if len(df) == self._last_daily_count:
            return

        kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(kline, confirm_required=False)
        self._daily_fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(kline, self._daily_fractals, min_bars=self.daily_min_strokes)
        self._daily_strokes = stroke_gen.get_strokes()

        # 计算MACD
        self._daily_macd = MACD(df['close'])

        # 检测日线2卖
        self._daily_second_sell_detected = self._detect_daily_second_sell()

        self._last_daily_count = len(df)

    def _find_weekly_buy_points(self) -> None:
        """找出周线买卖点"""
        if not self._weekly_strokes:
            return

        # 第一类买点：最后一个向下笔的低点
        down_strokes = [s for s in self._weekly_strokes if s.is_down]
        if down_strokes:
            self._weekly_first_buy_price = down_strokes[-1].low

        # 第二类买点：第一类买点后的向上笔
        if len(self._weekly_strokes) >= 2:
            last = self._weekly_strokes[-1]
            if last.is_up:
                # 向上笔的起点是潜在2买位置
                self._weekly_second_buy_price = last.start_value

    def _detect_daily_second_sell(self) -> bool:
        """检测日线第二类卖点"""
        if len(self._daily_strokes) < 3:
            return False

        # 检查是否有向上笔然后向下转折
        last = self._daily_strokes[-1]
        second_last = self._daily_strokes[-2]

        if last.is_down and second_last.is_up:
            # 反弹不破前高
            if last.end_value < second_last.start_value * 0.98:
                return True

        return False

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series
    ) -> Optional[Signal]:
        """检查买入信号 - 周线2买"""
        if not self._weekly_strokes:
            return None

        # 检查是否形成周线2买
        if self._weekly_second_buy_price is None:
            return None

        # 价格接近2买位置
        if price > self._weekly_second_buy_price * 0.97 and price < self._weekly_second_buy_price * 1.03:
            # 确认是向上笔
            if self._weekly_strokes[-1].is_up:
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    reason=f'周线2买 (1买:{self._weekly_first_buy_price:.2f}, 2买:{self._weekly_second_buy_price:.2f})',
                    confidence=0.8
                )

        return None

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查卖出信号"""
        record = self._position_record.get(symbol)
        current_qty = self.get_position(symbol)

        if not record:
            # 初始化持仓记录
            record = {
                'entry_price': price,
                'stop_loss': self._weekly_first_buy_price if self._weekly_first_buy_price else price * 0.98,
                'partial_exited': False,
            }
            self._position_record[symbol] = record

        # 1. 止损：跌破周线1买最低点（精确止损）
        if self._weekly_first_buy_price and price < self._weekly_first_buy_price:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'止损:跌破周线1买最低点({self._weekly_first_buy_price:.2f})',
                confidence=1.0
            )

        # 2. 日线MACD顶背离减仓50%
        if self._use_partial_exit and not record['partial_exited']:
            if self._check_daily_macd_divergence():
                record['partial_exited'] = True
                exit_qty = int(current_qty * self.exit_ratio)
                exit_qty = (exit_qty // 100) * 100  # 取整到100

                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason='日线MACD顶背离减仓50%',
                    confidence=0.8
                )

        # 3. 日线2卖卖出剩余
        if self._daily_second_sell_detected:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason='日线2卖卖出剩余',
                confidence=0.9
            )

        return None

    @property
    def _use_partial_exit(self) -> bool:
        """是否使用分批止盈"""
        return True

    def _check_daily_macd_divergence(self) -> bool:
        """检查日线MACD顶背驰"""
        if not self._daily_macd or len(self._daily_macd) < 20:
            return False

        # 检查最近是否有顶背驰
        has_divergence, _ = self._daily_macd.check_divergence(
            len(self._daily_macd) - 20,
            len(self._daily_macd) - 1,
            'up'
        )

        return has_divergence

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调"""
        symbol = signal.symbol

        if signal.is_buy():
            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

            # 初始化持仓记录
            self._position_record[symbol] = {
                'entry_price': executed_price,
                'stop_loss': self._weekly_first_buy_price * (1 - self.stop_loss_pct) if self._weekly_first_buy_price else executed_price * 0.92,
                'partial_exited': False,
            }

        elif signal.is_sell():
            qty_to_sell = signal.quantity if signal.quantity else self.get_position(symbol)
            self.position[symbol] = self.position.get(symbol, 0) - qty_to_sell
            self.cash += executed_price * qty_to_sell

            # 如果全部卖出，清除记录
            if self.get_position(symbol) == 0:
                if symbol in self._position_record:
                    del self._position_record[symbol]
