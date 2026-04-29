"""
增强版缠论策略

集成所有优化模块：
1. ATR动态仓位管理
2. 多重技术指标过滤
3. 多级止盈
4. 组合管理
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

# 导入优化模块
from backtest.position_sizing import (
    RiskParitySizer, KellySizer, AdaptivePositionSizer, calculate_atr
)
from backtest.filters import (
    FilterChain, VolumeFilter, RSIFilter, BollingerFilter,
    TrendFilter, MultiStageExit
)
from backtest.portfolio import PortfolioManager


class EnhancedChanLunStrategy(Strategy):
    """
    增强版缠论策略

    特点：
    1. 自适应仓位管理 (ATR风险平价)
    2. 多重过滤器确认 (趋势+成交量+RSI)
    3. 多级止盈策略
    4. 行业分散度控制
    """

    def __init__(
        self,
        name: str = '增强版缠论策略',

        # 缠论参数
        weekly_min_strokes: int = 3,
        daily_min_strokes: int = 3,

        # 仓位管理参数
        position_method: str = 'risk_parity',  # 'risk_parity', 'kelly', 'adaptive'
        risk_per_trade: float = 0.02,         # 每笔风险2%
        atr_multiplier: float = 2.0,           # ATR止损倍数

        # 过滤器参数
        use_volume_filter: bool = True,
        use_rsi_filter: bool = True,
        use_trend_filter: bool = True,
        use_bollinger_filter: bool = False,

        # 止盈参数
        use_multistage_exit: bool = True,
        exit_stages: List[tuple] = None,

        # 组合管理参数
        max_positions: int = 5,
        max_single_weight: float = 0.25,

        # 其他参数
        stop_loss_pct: float = 0.08,
    ):
        # 先设置实例属性
        self.weekly_min_strokes = weekly_min_strokes
        self.daily_min_strokes = daily_min_strokes
        self.position_method = position_method
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight

        # 过滤器配置
        self.use_volume_filter = use_volume_filter
        self.use_rsi_filter = use_rsi_filter
        self.use_trend_filter = use_trend_filter
        self.use_bollinger_filter = use_bollinger_filter

        # 止盈配置
        self.use_multistage_exit = use_multistage_exit
        self.exit_stages = exit_stages or [(0.10, 0.30), (0.20, 0.30), (0.30, 0.40)]

        # 初始化组合管理器（在调用super之前）
        self.portfolio_manager: Optional[PortfolioManager] = None

        # 现在调用父类初始化
        super().__init__(name)

        # 初始化仓位管理器
        self._init_position_sizer()

        # 初始化过滤器
        self._init_filters()

        # 初始化多级止盈
        self.multi_stage_exit = MultiStageExit(
            stages=self.exit_stages,
            use_trailing=True,
            trailing_atr_multiple=3.0,
        )

        # 初始化组合管理器
        self.portfolio_manager: Optional[PortfolioManager] = None

        # 数据缓存
        self._weekly_data: Optional[pd.DataFrame] = None
        self._weekly_fractals: List[Fractal] = []
        self._weekly_strokes: List[Stroke] = []
        self._weekly_pivots: List[Pivot] = []
        self._weekly_first_buy_price: Optional[float] = None
        self._weekly_second_buy_price: Optional[float] = None

        self._daily_data: Optional[pd.DataFrame] = None
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_macd: Optional[MACD] = None
        self._daily_atr: Optional[float] = None
        self._daily_second_sell_detected: bool = False

        # 缓存计数
        self._last_weekly_count: int = 0
        self._last_daily_count: int = 0

    def _init_position_sizer(self):
        """初始化仓位管理器"""
        self.position_sizer = AdaptivePositionSizer(
            initial_capital=100000,  # 初始化时设置，实际使用时会更新
            risk_per_trade=self.risk_per_trade,
        )

    def _init_filters(self):
        """初始化过滤器"""
        self.filters = []

        if self.use_trend_filter:
            self.filters.append(TrendFilter(fast_period=20, slow_period=60))

        if self.use_volume_filter:
            self.filters.append(VolumeFilter())

        if self.use_rsi_filter:
            self.filters.append(RSIFilter())

        if self.use_bollinger_filter:
            self.filters.append(BollingerFilter())

        self.filter_chain = FilterChain(self.filters, mode='all')

    def initialize(
        self,
        capital: float,
        symbols: List[str],
        industry_map: Optional[Dict[str, tuple]] = None,
    ) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)

        # 更新仓位管理器初始资金
        if self.position_method == 'risk_parity':
            self.position_sizer = RiskParitySizer(
                initial_capital=capital,
                risk_per_trade=self.risk_per_trade,
                atr_multiplier=self.atr_multiplier,
            )
        elif self.position_method == 'kelly':
            self.position_sizer = KellySizer(
                initial_capital=capital,
            )
        else:
            self.position_sizer = AdaptivePositionSizer(
                initial_capital=capital,
                risk_per_trade=self.risk_per_trade,
            )

        # 初始化组合管理器
        self.portfolio_manager = PortfolioManager(
            initial_capital=capital,
            max_positions=5,
            max_single_weight=self.max_single_weight,
        )

        if industry_map:
            self.portfolio_manager.set_industry_mapping(industry_map)

        logger.info(f"初始化{self.name}: 资金¥{capital:,.0f}")
        logger.info(f"  仓位管理: {self.position_method}")
        logger.info(f"  过滤器: {[f.name for f in self.filters]}")

    def reset(self) -> None:
        """重置策略"""
        super().reset()
        self._weekly_first_buy_price = None
        self._weekly_second_buy_price = None
        self.multi_stage_exit.positions = {}

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        daily_df = context['data'].get(symbol)
        if daily_df is None or len(daily_df) < 150:
            return None

        current_price = bar['close']

        # 更新组合管理器价格
        self.portfolio_manager.update_prices({symbol: current_price})

        # 生成周线数据
        weekly_df = self._convert_to_weekly(daily_df)

        # 更新分析
        self._update_weekly_analysis(weekly_df)
        self._update_daily_analysis(daily_df)

        # 判断趋势
        is_uptrend = self._check_trend_condition()

        # 已有持仓：检查卖出信号
        if symbol in self.portfolio_manager.positions:
            return self._check_exit_signals(symbol, current_price, bar, daily_df)

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar, is_uptrend, daily_df)

    def _convert_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """将日线数据转换为周线数据"""
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
        self._daily_atr = calculate_atr(df)

        self._daily_second_sell_detected = self._detect_daily_second_sell()
        self._last_daily_count = len(df)

    def _find_weekly_buy_points(self) -> None:
        """找出周线买卖点"""
        if not self._weekly_strokes:
            return

        down_strokes = [s for s in self._weekly_strokes if s.is_down]
        if down_strokes:
            self._weekly_first_buy_price = down_strokes[-1].low

        if len(self._weekly_strokes) >= 2:
            last = self._weekly_strokes[-1]
            if last.is_up:
                self._weekly_second_buy_price = last.start_value

    def _detect_daily_second_sell(self) -> bool:
        """检测日线第二类卖点"""
        if len(self._daily_strokes) < 3:
            return False

        last = self._daily_strokes[-1]
        second_last = self._daily_strokes[-2]

        if last.is_down and second_last.is_up:
            if last.end_value < second_last.start_value * 0.98:
                return True

        return False

    def _check_trend_condition(self) -> bool:
        """检查趋势条件"""
        if not self._weekly_strokes:
            return False

        last_segments = self._weekly_strokes[-3:]
        up_count = sum(1 for s in last_segments if s.is_up)
        return up_count >= 2

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        is_uptrend: bool,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查买入信号"""
        if not self._weekly_strokes:
            return None

        # 1. 趋势过滤
        if not is_uptrend:
            return None

        # 2. 周线2买位置确认
        if self._weekly_second_buy_price is None:
            return None

        # 价格在2买附近或突破前高
        in_zone = (
            self._weekly_second_buy_price * 0.90 <= price <=
            self._weekly_second_buy_price * 1.15
        )

        if not in_zone:
            return None

        # 3. 确认是向上笔
        if not self._weekly_strokes[-1].is_up:
            return None

        # 4. 过滤器确认
        filter_result = self.filter_chain.check(
            bar=bar,
            historical=daily_df,
            signal_type='buy'
        )

        if not filter_result.passed:
            logger.debug(f"[{symbol}] 过滤器未通过: {filter_result.reason}")
            return None

        # 5. 计算仓位
        position_result = self.position_sizer.calculate(
            price=price,
            cash=self.cash,
            atr=self._daily_atr,
            stop_price=self._weekly_first_buy_price,
        )

        if position_result.shares == 0:
            return None

        # 6. 组合管理检查
        can_buy, msg = self.portfolio_manager.can_buy(
            symbol, price, position_result.shares
        )

        if not can_buy:
            logger.debug(f"[{symbol}] {msg}")
            return None

        # 通过所有检查，生成买入信号
        reason_parts = [
            f"周线2买({self._weekly_second_buy_price:.2f})",
            filter_result.reason,
            f"仓位{position_result.shares}股",
        ]

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            quantity=position_result.shares,
            reason=' | '.join(reason_parts),
            confidence=filter_result.confidence,
        )

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查卖出信号"""
        position = self.portfolio_manager.positions.get(symbol)
        if not position:
            return None

        current_qty = position.shares

        # 1. 精确止损：跌破周线1买最低点
        if self._weekly_first_buy_price and price < self._weekly_first_buy_price:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'止损:跌破周线1买({self._weekly_first_buy_price:.2f})',
                confidence=1.0
            )

        # 2. ATR动态止损
        if symbol in self.multi_stage_exit.positions:
            exit_ratio = self.multi_stage_exit.update_price(symbol, price)
            if exit_ratio is not None:
                exit_qty = int(current_qty * exit_ratio / 100) * 100

                if exit_qty > 0:
                    return Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                        price=price,
                        quantity=exit_qty,
                        reason=f'多级止盈{exit_ratio:.0%}',
                        confidence=0.9
                    )
        else:
            # 添加到多级止盈跟踪
            self.multi_stage_exit.add_position(
                symbol, position.entry_price, self._daily_atr
            )

        # 3. 日线2卖
        if self._daily_second_sell_detected:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason='日线2卖',
                confidence=0.9
            )

        # 4. 过滤器卖出确认
        filter_result = self.filter_chain.check(
            bar=bar,
            historical=daily_df,
            signal_type='sell'
        )

        if filter_result.passed and filter_result.confidence > 0.8:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'过滤器卖出: {filter_result.reason}',
                confidence=filter_result.confidence
            )

        return None

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调"""
        if signal.is_buy():
            self.portfolio_manager.buy(
                symbol=signal.symbol,
                price=executed_price,
                shares=executed_quantity,
                date=signal.datetime,
                reason=signal.reason,
            )
            # 添加到多级止盈
            self.multi_stage_exit.add_position(
                signal.symbol, executed_price, self._daily_atr
            )

        elif signal.is_sell():
            self.portfolio_manager.sell(
                symbol=signal.symbol,
                price=executed_price,
                shares=executed_quantity,
                date=signal.datetime,
                reason=signal.reason,
            )

    @property
    def cash(self) -> float:
        """获取现金"""
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            return self.portfolio_manager.cash
        return 0

    @cash.setter
    def cash(self, value: float):
        """设置现金"""
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            self.portfolio_manager.cash = value

    @property
    def position(self) -> Dict[str, int]:
        """获取持仓"""
        if self.portfolio_manager:
            return {
                symbol: pos.shares
                for symbol, pos in self.portfolio_manager.positions.items()
            }
        return {}

    @position.setter
    def position(self, value: Dict[str, int]):
        """设置持仓"""
        pass  # 通过PortfolioManager管理

    def get_equity(self, prices: Dict[str, float]) -> float:
        """获取总权益"""
        if self.portfolio_manager:
            self.portfolio_manager.update_prices(prices)
            return self.portfolio_manager.get_total_value()
        return self.cash

    def get_position(self, symbol: str) -> int:
        """获取持仓数量"""
        if self.portfolio_manager and symbol in self.portfolio_manager.positions:
            return self.portfolio_manager.positions[symbol].shares
        return 0

    def get_cash(self) -> float:
        """获取可用现金"""
        return self.cash

    def get_portfolio_report(self) -> str:
        """获取组合报告"""
        if self.portfolio_manager:
            from backtest.portfolio import generate_portfolio_report
            return generate_portfolio_report(self.portfolio_manager)
        return "组合管理器未初始化"
