"""
平衡版缠论策略

结合原始策略的入场条件 + 新策略的优化出场机制
"""

from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD

from strategies.enhanced_exit_conditions import (
    DynamicExitManager,
    PositionRecord,
    ExitSignal
)


class BalancedChanLunStrategy(Strategy):
    """
    平衡版缠论策略

    入场: 使用原始的周线2买逻辑（不过度过滤）
    出场: 使用优化的跟踪止损和分批止盈
    """

    def __init__(
        self,
        name: str = '平衡版缠论策略',
        weekly_min_strokes: int = 3,
        daily_min_strokes: int = 3,
        stop_loss_pct: float = 0.06,        # 6%止损
        exit_ratio: float = 0.5,
        # 出场优化参数
        use_trailing_stop: bool = True,
        trailing_activation: float = 0.03,   # 盈利3%启动跟踪
        trailing_offset: float = 0.05,       # 5%回撤
        profit_targets: List[tuple] = None,   # 分批止盈目标
    ):
        super().__init__(name)
        self.weekly_min_strokes = weekly_min_strokes
        self.daily_min_strokes = daily_min_strokes
        self.stop_loss_pct = stop_loss_pct
        self.exit_ratio = exit_ratio

        # 出场管理器
        self.exit_manager = DynamicExitManager(
            use_trailing_stop=use_trailing_stop,
            use_partial_profit=True
        )

        # 配置跟踪止损参数
        self.trailing_activation = trailing_activation
        self.trailing_offset = trailing_offset
        self.profit_targets = profit_targets or [
            (0.05, 0.3),   # 盈利5%卖30%
            (0.10, 0.3),   # 盈利10%卖30%
            (0.15, 0.4),   # 盈利15%卖剩余
        ]

        # 数据缓存
        self._weekly_data: Optional[pd.DataFrame] = None
        self._weekly_fractals: List[Fractal] = []
        self._weekly_strokes: List[Stroke] = []
        self._weekly_pivots: List[Pivot] = []
        self._weekly_first_buy: Optional[float] = None
        self._weekly_second_buy: Optional[float] = None

        self._daily_data: Optional[pd.DataFrame] = None
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_macd: Optional[MACD] = None
        self._daily_second_sell: bool = False

        self._last_weekly_count: int = 0
        self._last_daily_count: int = 0
        self._position_record: Dict[str, Dict] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金{capital:,.0f}")
        logger.info(f"  跟踪止损: {self.exit_manager.use_trailing_stop}, 激活阈值{self.trailing_activation:.0%}")
        logger.info(f"  分批止盈: {self.profit_targets}")

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
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

        # 已有持仓：检查出场
        if current_position > 0:
            return self._check_exit(symbol, current_price, bar, daily_df)

        # 无持仓：检查入场
        return self._check_entry(symbol, current_price, bar)

    def _convert_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
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
        if len(df) == self._last_weekly_count:
            return

        kline = KLine.from_dataframe(df, strict_mode=False)

        detector = FractalDetector(kline, confirm_required=False)
        self._weekly_fractals = detector.get_fractals()

        stroke_gen = StrokeGenerator(kline, self._weekly_fractals, min_bars=self.weekly_min_strokes)
        self._weekly_strokes = stroke_gen.get_strokes()

        pivot_detector = PivotDetector(kline, self._weekly_strokes)
        self._weekly_pivots = pivot_detector.get_pivots()

        self._find_weekly_buy_points()
        self._last_weekly_count = len(df)

    def _update_daily_analysis(self, df: pd.DataFrame) -> None:
        if len(df) == self._last_daily_count:
            return

        kline = KLine.from_dataframe(df, strict_mode=False)

        detector = FractalDetector(kline, confirm_required=False)
        self._daily_fractals = detector.get_fractals()

        stroke_gen = StrokeGenerator(kline, self._daily_fractals, min_bars=self.daily_min_strokes)
        self._daily_strokes = stroke_gen.get_strokes()

        self._daily_macd = MACD(df['close'])

        # 检测日线2卖
        self._daily_second_sell = self._detect_daily_second_sell()
        self._last_daily_count = len(df)

    def _find_weekly_buy_points(self) -> None:
        if not self._weekly_strokes:
            return

        down_strokes = [s for s in self._weekly_strokes if s.is_down]
        if down_strokes:
            self._weekly_first_buy = down_strokes[-1].low

        if len(self._weekly_strokes) >= 2:
            last = self._weekly_strokes[-1]
            if last.is_up:
                self._weekly_second_buy = last.start_value
            else:
                self._weekly_second_buy = None
        else:
            self._weekly_second_buy = None

    def _detect_daily_second_sell(self) -> bool:
        if len(self._daily_strokes) < 3:
            return False

        last = self._daily_strokes[-1]
        second_last = self._daily_strokes[-2]

        if last.is_down and second_last.is_up:
            if last.end_value < second_last.start_value * 0.98:
                return True

        return False

    def _check_entry(
        self,
        symbol: str,
        price: float,
        bar: pd.Series
    ) -> Optional[Signal]:
        # 使用原始周线2买逻辑
        if self._weekly_second_buy is None:
            return None

        # 价格在2买附近 (±5%范围，更严格控制)
        if not (self._weekly_second_buy * 0.95 <= price <= self._weekly_second_buy * 1.05):
            return None

        # 确认是向上笔
        if not self._weekly_strokes or not self._weekly_strokes[-1].is_up:
            return None

        # 检查是否有日线MACD顶背离（作为过滤）
        if self._check_daily_macd_divergence():
            return None  # 顶背离，不买入

        # 新增：检查日线MACD是否金叉（趋势确认）
        if not self._check_daily_macd_golden_cross():
            return None  # 未金叉，不买入

        # 新增：价格需要在2买点下方或附近（不在高位追涨）
        if price > self._weekly_second_buy * 1.05:
            return None

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            reason=f'周线2买 (1买:{self._weekly_first_buy:.2f}, 2买:{self._weekly_second_buy:.2f})',
            confidence=0.7
        )

    def _check_daily_macd_divergence(self) -> bool:
        if not self._daily_macd or len(self._daily_macd) < 20:
            return False

        has_divergence, _ = self._daily_macd.check_divergence(
            len(self._daily_macd) - 20,
            len(self._daily_macd) - 1,
            'up'
        )
        return has_divergence

    def _check_daily_macd_golden_cross(self) -> bool:
        """检查日线MACD是否金叉"""
        if not self._daily_macd or len(self._daily_macd) < 3:
            return False

        # 获取最近3个MACD值
        try:
            macd_values = self._daily_macd.get_macd()
            signal_values = self._daily_macd.get_signal()

            if len(macd_values) < 3 or len(signal_values) < 3:
                return False

            # 检查是否刚金叉：MACD从下向上穿过信号线
            recent_macd = macd_values[-3:]
            recent_signal = signal_values[-3:]

            # 前2个点：MACD < Signal
            # 最新点：MACD > Signal
            if (recent_macd[0] < recent_signal[0] and
                recent_macd[1] < recent_signal[1] and
                recent_macd[2] > recent_signal[2]):
                return True

            # 或者MACD和Signal都在0轴以下且向上
            if (recent_macd[2] > 0 and recent_signal[2] > 0 and
                recent_macd[2] > recent_macd[1]):
                return True

        except Exception:
            pass

        return False

    def _check_exit(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        current_qty = self.get_position(symbol)

        # 获取或创建持仓记录
        position = self.exit_manager.get_position(symbol)
        if not position:
            # 从_position_record创建
            record = self._position_record.get(symbol)
            if not record:
                entry_price = price
                initial_stop = (self._weekly_first_buy or price) * (1 - self.stop_loss_pct)
                record = {
                    'entry_price': entry_price,
                    'initial_stop': initial_stop,
                    'highest_price': entry_price,
                    'partial_exited': False,
                }
                self._position_record[symbol] = record
            else:
                entry_price = record['entry_price']

            position = self.exit_manager.open_position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=current_qty,
                initial_stop=record.get('initial_stop', price * 0.94),
                volatility=0.03  # 默认波动率
            )
            # 设置分批止盈目标
            position.profit_targets = [
                (entry_price * (1 + tgt_pct), tgt_ratio)
                for tgt_pct, tgt_ratio in self.profit_targets
            ]

        # 1. 检查跟踪止损
        if self.exit_manager.use_trailing_stop:
            # 更新最高价
            if price > position.highest_price:
                position.highest_price = price

            # 计算盈利比例
            profit_ratio = (price - position.entry_price) / position.entry_price

            if profit_ratio > self.trailing_activation:
                # 启动跟踪止损
                trailing_stop = position.highest_price * (1 - self.trailing_offset)
                if price <= trailing_stop:
                    return Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                        price=price,
                        quantity=current_qty,
                        reason=f'跟踪止损 @ {price:.2f} (最高{position.highest_price:.2f})',
                        confidence=1.0
                    )

        # 2. 检查分批止盈
        # 使用本地记录判断是否已分批
        record = self._position_record.get(symbol, {})
        has_partially_exited = record.get('partial_exited', False)

        if not has_partially_exited:
            entry_price = record.get('entry_price', position.entry_price)
            for i, (profit_pct, exit_ratio) in enumerate(self.profit_targets):
                target_price = entry_price * (1 + profit_pct)
                if price >= target_price:
                    exit_qty = int(current_qty * exit_ratio)
                    exit_qty = (exit_qty // 100) * 100

                    if exit_qty > 0:
                        # 标记已分批
                        if symbol in self._position_record:
                            self._position_record[symbol]['partial_exited'] = True
                        # 更新持仓
                        self.position[symbol] -= exit_qty

                        return Signal(
                            signal_type=SignalType.SELL,
                            symbol=symbol,
                            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                            price=price,
                            quantity=exit_qty,
                            reason=f'分批止盈 ({i+1}) 盈利{profit_pct:.0%} @ {price:.2f}',
                            confidence=0.9
                        )

        # 3. 检查传统止损
        if self._weekly_first_buy and price < self._weekly_first_buy * (1 - self.stop_loss_pct):
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'止损: 跌破1买{self._weekly_first_buy:.2f}',
                confidence=1.0
            )

        # 4. 检查日线2卖
        if self._daily_second_sell:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason='日线2卖',
                confidence=0.9
            )

        return None

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        symbol = signal.symbol

        if signal.is_buy():
            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

            # 初始化持仓记录
            self._position_record[symbol] = {
                'entry_price': executed_price,
                'initial_stop': self._weekly_first_buy * (1 - self.stop_loss_pct) if self._weekly_first_buy else executed_price * 0.94,
                'highest_price': executed_price,
                'partial_exited': False,
            }

        elif signal.is_sell():
            qty_to_sell = signal.quantity if signal.quantity else self.get_position(symbol)
            self.position[symbol] = self.position.get(symbol, 0) - qty_to_sell
            self.cash += executed_price * qty_to_sell

            # 清除记录
            if self.get_position(symbol) == 0:
                if symbol in self._position_record:
                    del self._position_record[symbol]
                self.exit_manager.close_position(symbol)
