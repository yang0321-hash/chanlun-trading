"""
智能缠论策略 - 整合所有改进模块

整合内容：
1. 市场环境识别
2. 改进的入场条件
3. 优化的出场机制
4. 多Agent辩论系统
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD

# 新增模块
from indicators.market_regime import (
    MarketRegimeDetector,
    VolumeAnalyzer,
    BullBearStrength,
    RangeDetector
)
from strategies.enhanced_entry_conditions import (
    EnhancedEntryFilter,
    ChanLunEntryValidator
)
from strategies.enhanced_exit_conditions import (
    DynamicExitManager,
    PositionRecord,
    ExitSignal
)
from agents.debate_system import (
    ChanLunDebateSystem,
    DebateResult,
    Decision
)


class IntelligentChanLunStrategy(Strategy):
    """
    智能缠论策略

    结合市场环境识别、改进的入场条件、优化的出场机制和多Agent辩论系统
    """

    def __init__(
        self,
        name: str = '智能缠论策略',
        # 缠论参数
        weekly_min_strokes: int = 3,
        daily_min_strokes: int = 3,

        # 辩论系统参数
        enable_debate: bool = True,
        debate_rounds: int = 2,
        min_decision_confidence: float = 0.55,

        # 市场过滤参数
        require_uptrend: bool = True,
        min_adx: float = 25,
        max_volatility: float = 0.06,

        # 出场参数
        use_trailing_stop: bool = True,
        use_partial_profit: bool = True,
        trailing_offset: float = 0.06,
        profit_targets: List[tuple] = None,
    ):
        super().__init__(name)

        self.weekly_min_strokes = weekly_min_strokes
        self.daily_min_strokes = daily_min_strokes
        self.enable_debate = enable_debate
        self.debate_rounds = debate_rounds
        self.min_decision_confidence = min_decision_confidence

        # 初始化模块
        self.market_detector = MarketRegimeDetector(adx_threshold=min_adx)
        self.entry_validator = ChanLunEntryValidator()
        self.exit_manager = DynamicExitManager(
            use_trailing_stop=use_trailing_stop,
            use_partial_profit=use_partial_profit
        )
        self.debate_system = ChanLunDebateSystem(max_rounds=debate_rounds) if enable_debate else None

        # 数据缓存
        self._weekly_data: Optional[pd.DataFrame] = None
        self._daily_data: Optional[pd.DataFrame] = None
        self._weekly_fractals: List[Fractal] = []
        self._weekly_strokes: List[Stroke] = []
        self._weekly_pivots: List[Pivot] = []
        self._weekly_first_buy: Optional[float] = None
        self._weekly_second_buy: Optional[float] = None

        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_macd: Optional[MACD] = None

        # 缓存计数
        self._last_weekly_count: int = 0
        self._last_daily_count: int = 0

        # 持仓管理
        self._positions: Dict[str, PositionRecord] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}")
        logger.info(f"  周线最小笔数: {self.weekly_min_strokes}")
        logger.info(f"  日线最小笔数: {self.daily_min_strokes}")
        logger.info(f"  辩论系统: {'启用' if self.enable_debate else '禁用'}")
        logger.info(f"  跟踪止损: {'启用' if self.exit_manager.use_trailing_stop else '禁用'}")
        logger.info(f"  分批止盈: {'启用' if self.exit_manager.use_partial_profit else '禁用'}")

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

        # 已有持仓：检查出场
        if current_position > 0:
            return self._check_exit(symbol, current_price, bar, daily_df)

        # 无持仓：检查入场
        return self._check_entry(symbol, current_price, bar, daily_df, weekly_df, index)

    def _convert_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """将日线转换为周线"""
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

        detector = FractalDetector(kline, confirm_required=False)
        self._weekly_fractals = detector.get_fractals()

        stroke_gen = StrokeGenerator(kline, self._weekly_fractals, min_bars=self.weekly_min_strokes)
        self._weekly_strokes = stroke_gen.get_strokes()

        pivot_detector = PivotDetector(kline, self._weekly_strokes)
        self._weekly_pivots = pivot_detector.get_pivots()

        self._find_weekly_buy_points()
        self._last_weekly_count = len(df)

    def _update_daily_analysis(self, df: pd.DataFrame) -> None:
        """更新日线分析"""
        if len(df) == self._last_daily_count:
            return

        kline = KLine.from_dataframe(df, strict_mode=False)

        detector = FractalDetector(kline, confirm_required=False)
        self._daily_fractals = detector.get_fractals()

        stroke_gen = StrokeGenerator(kline, self._daily_fractals, min_bars=self.daily_min_strokes)
        self._daily_strokes = stroke_gen.get_strokes()

        self._daily_macd = MACD(df['close'])
        self._last_daily_count = len(df)

    def _find_weekly_buy_points(self) -> None:
        """找出周线买卖点"""
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

    def _check_entry(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame,
        weekly_df: pd.DataFrame,
        index: int
    ) -> Optional[Signal]:
        """检查入场条件"""
        # 基础条件：需要周线2买
        if self._weekly_second_buy is None:
            return None

        # 价格不在2买区间
        if not (self._weekly_second_buy * 0.96 <= price <= self._weekly_second_buy * 1.05):
            return None

        # 构建上下文
        context = self._build_context(symbol, price, daily_df, weekly_df, index)

        # 方法1: 使用多Agent辩论系统
        if self.enable_debate:
            debate_result = self.debate_system.debate(symbol, daily_df, index, context)

            if debate_result.decision == Decision.BUY and debate_result.confidence >= self.min_decision_confidence:
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    reason=f"[辩论] {debate_result.final_reasoning}",
                    confidence=debate_result.confidence
                )
            else:
                logger.debug(f"辩论结果: {debate_result.decision.value}, 置信度{debate_result.confidence:.2f}, 不满足入场条件")

        # 方法2: 使用增强过滤器
        entry_signal = self.entry_validator.validate_second_buy(
            daily_df, index,
            self._weekly_first_buy or 0,
            self._weekly_second_buy,
            price
        )

        if entry_signal.should_enter:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                reason=f"[过滤] {entry_signal.reason}",
                confidence=entry_signal.confidence
            )

        return None

    def _check_exit(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查出场条件"""
        # 获取持仓记录
        position = self.exit_manager.get_position(symbol)
        if not position:
            # 首次检查，创建记录
            quantity = self.get_position(symbol)
            if quantity > 0:
                position = self.exit_manager.open_position(
                    symbol=symbol,
                    entry_price=position.entry_price if hasattr(position, 'entry_price') else price,
                    quantity=quantity,
                    initial_stop=self._weekly_first_buy * 0.98 if self._weekly_first_buy else price * 0.92
                )

        # 更新并检查出场信号
        exit_signals = self.exit_manager.update_position(symbol, price, datetime.now())

        if exit_signals:
            for signal in exit_signals:
                if signal.should_exit:
                    quantity = int(self.get_position(symbol) * signal.exit_ratio)
                    quantity = (quantity // 100) * 100

                    if quantity > 0:
                        # 处理出场
                        self.exit_manager.close_position(symbol, signal.exit_ratio)

                        return Signal(
                            signal_type=SignalType.SELL,
                            symbol=symbol,
                            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                            price=price,
                            quantity=quantity,
                            reason=signal.description,
                            confidence=1.0
                        )

        # 检查传统止损
        if self._weekly_first_buy and price < self._weekly_first_buy * 0.98:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=self.get_position(symbol),
                reason=f'止损: 跌破周线1买{self._weekly_first_buy:.2f}',
                confidence=1.0
            )

        return None

    def _build_context(
        self,
        symbol: str,
        price: float,
        daily_df: pd.DataFrame,
        weekly_df: pd.DataFrame,
        index: int
    ) -> Dict[str, Any]:
        """构建辩论上下文"""
        # 市场环境
        market_state = self.market_detector.detect(daily_df, index)

        context = {
            'symbol': symbol,
            'current_price': price,
            'market_regime': market_state.regime.value,
            'volatility': market_state.volatility,
            'trend_strength': market_state.trend_strength,
            'is_tradeable': market_state.is_tradeable,

            # 缠论信号
            'weekly_second_buy': self._weekly_second_buy is not None,
            'weekly_first_buy': self._weekly_first_buy,
            'fractals': self._weekly_fractals + self._daily_fractals,
            'strokes': self._weekly_strokes,
            'pivot': self._weekly_pivots[-1] if self._weekly_pivots else None,

            # 技术指标
            'current_index': index,
        }

        return context

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调"""
        symbol = signal.symbol

        if signal.is_buy():
            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

            # 记录到出场管理器
            self.exit_manager.open_position(
                symbol=symbol,
                entry_price=executed_price,
                quantity=executed_quantity,
                initial_stop=self._weekly_first_buy * 0.98 if self._weekly_first_buy else executed_price * 0.92
            )

        elif signal.is_sell():
            qty_to_sell = signal.quantity if signal.quantity else self.get_position(symbol)
            self.position[symbol] = self.position.get(symbol, 0) - qty_to_sell
            self.cash += executed_price * qty_to_sell

            # 记录学习
            if self.debate_system:
                self.debate_system.reflect(
                    symbol=symbol,
                    decision=Decision.BUY,  # 假设之前是买入
                    actual_outcome="sell",
                    profit_loss=0  # 需要计算
                )
