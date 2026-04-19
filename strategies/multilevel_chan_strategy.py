"""
多级别缠论策略 - 日线+30分钟

.. deprecated::
    本策略使用日线数据冒充30分钟数据（_get_m30_data直接返回日线），
    30分钟级别分析结果不可靠。请使用 IntegratedChanLunStrategy 或
    WeeklyDailyChanLunStrategy 替代。

交易规则：
1. 日线级别2买买入
2. 跌破1买低点止损
3. 30分钟级别MACD顶背离减仓50%
4. 30分钟级别2卖卖出剩余部分
"""

import warnings

warnings.warn(
    "MultiLevelChanLunStrategy 使用日线冒充30分钟数据，结果不可靠。"
    "请使用 IntegratedChanLunStrategy 或 WeeklyDailyChanLunStrategy 替代。",
    DeprecationWarning,
    stacklevel=2
)

from typing import List, Optional, Dict, Any, Tuple
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


class MultiLevelChanLunStrategy(Strategy):
    """
    多级别缠论策略

    使用日线和30分钟两个级别进行交易决策
    """

    def __init__(
        self,
        name: str = '多级别缠论策略',
        # 日线参数
        daily_min_strokes: int = 3,
        # 30分钟参数
        m30_min_strokes: int = 5,
        # 止损参数
        stop_loss_pct: float = 0.05,
        # 减仓参数
        use_partial_exit: bool = True,
        exit_ratio: float = 0.5,  # 减仓比例
    ):
        super().__init__(name)

        self.daily_min_strokes = daily_min_strokes
        self.m30_min_strokes = m30_min_strokes
        self.stop_loss_pct = stop_loss_pct
        self.use_partial_exit = use_partial_exit
        self.exit_ratio = exit_ratio

        # 缓存
        self._daily_last_count: int = 0
        self._m30_last_count: int = 0

        # 日线级别数据
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_pivots: List[Pivot] = []
        self._first_buy_price: Optional[float] = None
        self._second_buy_price: Optional[float] = None

        # 30分钟级别数据
        self._m30_fractals: List[Fractal] = []
        self._m30_strokes: List[Stroke] = []
        self._m30_macd: Optional[MACD] = None
        self._m30_second_sell_detected: bool = False

        # 持仓记录
        self._position_record: Dict[str, Dict] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金¥{capital:,}")

    def reset(self) -> None:
        """重置策略"""
        super().reset()
        self._position_record = {}
        self._first_buy_price = None
        self._second_buy_price = None

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        # 获取日线数据
        daily_df = context['data'].get(symbol)
        if daily_df is None or len(daily_df) < 100:
            return None

        # 获取30分钟数据（从context中获取或从日线数据生成）
        m30_df = self._get_m30_data(daily_df)
        if m30_df is None or len(m30_df) < 50:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 更新分析
        self._update_daily_analysis(daily_df)
        self._update_m30_analysis(m30_df)

        # 已有持仓：检查卖出信号
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar, m30_df)

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar)

    def _get_m30_data(self, daily_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """获取30分钟数据（简化版：从日线生成）"""
        # 在实际应用中，应该从数据源获取真实的30分钟数据
        # 这里简化处理，使用日线数据代替
        return daily_df

    def _update_daily_analysis(self, df: pd.DataFrame) -> None:
        """更新日线分析"""
        if len(df) == self._daily_last_count:
            return

        kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(kline, confirm_required=False)
        self._daily_fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(kline, self._daily_fractals, min_bars=self.daily_min_strokes)
        self._daily_strokes = stroke_gen.get_strokes()

        # 识别中枢
        pivot_detector = PivotDetector(kline, self._daily_strokes)
        self._daily_pivots = pivot_detector.get_pivots()

        # 找出第一类买点价格
        self._find_buy_points()

        self._daily_last_count = len(df)

    def _update_m30_analysis(self, df: pd.DataFrame) -> None:
        """更新30分钟分析"""
        if len(df) == self._m30_last_count:
            return

        kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(kline, confirm_required=False)
        self._m30_fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(kline, self._m30_fractals, min_bars=self.m30_min_strokes)
        self._m30_strokes = stroke_gen.get_strokes()

        # 计算MACD
        self._m30_macd = MACD(df['close'])

        # 检测30分钟2卖
        self._m30_second_sell_detected = self._detect_m30_second_sell()

        self._m30_last_count = len(df)

    def _find_buy_points(self) -> None:
        """找出第一类和第二类买点"""
        if not self._daily_strokes:
            return

        # 第一类买点：最后一个向下笔的低点（底背驰位置）
        down_strokes = [s for s in self._daily_strokes if s.is_down]
        if down_strokes:
            self._first_buy_price = down_strokes[-1].low

        # 第二类买点：第一类买点后的第一个向上笔的低点
        if len(self._daily_strokes) >= 2:
            last = self._daily_strokes[-1]
            if last.is_up:
                self._second_buy_price = last.start_value

    def _detect_m30_second_sell(self) -> bool:
        """检测30分钟级别第二类卖点"""
        if len(self._m30_strokes) < 3:
            return False

        # 检查是否有向上的笔然后向下转折
        last = self._m30_strokes[-1]
        second_last = self._m30_strokes[-2]

        if last.is_down and second_last.is_up:
            # 检查是否形成第二类卖点（反弹不破前高）
            if last.end_value < second_last.start_value * 0.98:
                return True

        return False

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series
    ) -> Optional[Signal]:
        """检查买入信号 - 日线2买"""
        if not self._daily_strokes:
            return None

        # 检查是否形成日线2买
        if self._second_buy_price is None:
            return None

        # 价格接近2买位置
        if price > self._second_buy_price * 0.98 and price < self._second_buy_price * 1.02:
            # 确认是向上笔
            if self._daily_strokes[-1].is_up:
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    reason=f'日线2买 (1买:{self._first_buy_price:.2f}, 2买:{self._second_buy_price:.2f})',
                    confidence=0.8
                )

        return None

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        m30_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查卖出信号"""
        record = self._position_record.get(symbol)
        current_qty = self.get_position(symbol)

        if not record:
            # 初始化持仓记录
            record = {
                'entry_price': price,
                'stop_loss': self._first_buy_price * (1 - self.stop_loss_pct) if self._first_buy_price else price * 0.95,
                'partial_exited': False,
            }
            self._position_record[symbol] = record

        # 1. 止损：跌破1买低点
        if self._first_buy_price and price <= self._first_buy_price * (1 - self.stop_loss_pct):
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'止损:跌破1买({self._first_buy_price:.2f})',
                confidence=1.0
            )

        # 2. 30分钟MACD顶背离减仓50%
        if self.use_partial_exit and not record['partial_exited']:
            if self._check_m30_macd_divergence():
                record['partial_exited'] = True
                exit_qty = int(current_qty * self.exit_ratio / 100) * 100

                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason='30分钟MACD顶背离减仓50%',
                    confidence=0.8
                )

        # 3. 30分钟2卖卖出剩余
        if self._m30_second_sell_detected:
            # 如果已经减仓，卖出剩余；否则全部卖出
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason='30分钟2卖卖出剩余',
                confidence=0.9
            )

        return None

    def _check_m30_macd_divergence(self) -> bool:
        """检查30分钟MACD顶背驰"""
        if not self._m30_macd or len(self._m30_macd) < 20:
            return False

        # 检查最近是否有顶背驰
        has_divergence, _ = self._m30_macd.check_divergence(
            len(self._m30_macd) - 20,
            len(self._m30_macd) - 1,
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
                'stop_loss': self._first_buy_price * (1 - self.stop_loss_pct) if self._first_buy_price else executed_price * 0.95,
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


def create_multi_level_strategy(
    daily_strokes: int = 3,
    m30_strokes: int = 5,
    stop_loss: float = 0.05,
    exit_ratio: float = 0.5
) -> MultiLevelChanLunStrategy:
    """创建多级别策略"""
    return MultiLevelChanLunStrategy(
        daily_min_strokes=daily_strokes,
        m30_min_strokes=m30_strokes,
        stop_loss_pct=stop_loss,
        exit_ratio=exit_ratio
    )
