"""
优化版多级别缠论策略 - 周线+日线

优化内容：
1. 趋势过滤：使用MA系统和周线MACD判断上升趋势
2. 2买严格化：要求周线MACD金叉 + 日线MACD金叉双重确认
3. 动态止损：使用ATR动态调整止损位，适应波动率

交易规则：
1. 周线上升趋势 + 周线2买 + MACD金叉买入
2. ATR动态止损（1.5倍ATR）
3. 日线MACD顶背离减仓50%
4. 日线2卖卖出剩余
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


class OptimizedWeeklyDailyStrategy(Strategy):
    """
    优化版周线+日线 多级别缠论策略
    """

    def __init__(
        self,
        name: str = '优化周日线缠论策略',
        weekly_min_strokes: int = 3,
        daily_min_strokes: int = 3,
        ma_fast: int = 20,               # 快速均线
        ma_slow: int = 60,               # 慢速均线
        ma_trend: int = 120,             # 趋势均线
        atr_period: int = 14,            # ATR周期
        atr_multiplier: float = 2.0,     # ATR止损倍数 (放宽)
        exit_ratio: float = 0.5,         # 减仓50%
        require_macd_cross: bool = False, # 关闭MACD金叉强制要求
        trend_lookback: int = 5,         # 趋势回看周期
    ):
        super().__init__(name)
        self.weekly_min_strokes = weekly_min_strokes
        self.daily_min_strokes = daily_min_strokes
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_trend = ma_trend
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.exit_ratio = exit_ratio
        self.require_macd_cross = require_macd_cross
        self.trend_lookback = trend_lookback

        # 周线数据
        self._weekly_data: Optional[pd.DataFrame] = None
        self._weekly_fractals: List[Fractal] = []
        self._weekly_strokes: List[Stroke] = []
        self._weekly_pivots: List[Pivot] = []
        self._weekly_first_buy_price: Optional[float] = None
        self._weekly_second_buy_price: Optional[float] = None
        self._weekly_macd: Optional[MACD] = None
        self._weekly_ma_fast: Optional[pd.Series] = None
        self._weekly_ma_slow: Optional[pd.Series] = None
        self._weekly_ma_trend: Optional[pd.Series] = None

        # 日线数据
        self._daily_data: Optional[pd.DataFrame] = None
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_macd: Optional[MACD] = None
        self._daily_second_sell_detected: bool = False
        self._daily_atr: Optional[float] = None

        # 趋势状态
        self._is_uptrend: bool = False
        self._weekly_macd_golden_cross: bool = False
        self._daily_macd_golden_cross: bool = False

        # 缓存
        self._last_weekly_count: int = 0
        self._last_daily_count: int = 0

        # 持仓记录
        self._position_record: Dict[str, Dict] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金{capital:,.0f}")
        logger.info(f"  趋势过滤: MA{self.ma_fast}>MA{self.ma_slow}>MA{self.ma_trend}")
        logger.info(f"  MACD确认: {self.require_macd_cross}")
        logger.info(f"  动态止损: ATR{self.atr_period} x {self.atr_multiplier}")

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
        if daily_df is None or len(daily_df) < 150:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 生成周线数据
        weekly_df = self._convert_to_weekly(daily_df)

        # 更新分析
        self._update_weekly_analysis(weekly_df)
        self._update_daily_analysis(daily_df)

        # 判断趋势
        self._check_trend_condition()

        # 已有持仓：检查卖出信号
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar, daily_df)

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar)

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

        # 计算均线
        self._weekly_ma_fast = df['close'].rolling(self.ma_fast).mean()
        self._weekly_ma_slow = df['close'].rolling(self.ma_slow).mean()
        self._weekly_ma_trend = df['close'].rolling(self.ma_trend).mean()

        # 计算MACD
        self._weekly_macd = MACD(df['close'])

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

        # 检查MACD金叉
        self._check_weekly_macd_cross()

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

        # 计算ATR
        self._daily_atr = self._calculate_atr(df)

        # 检查日线MACD金叉
        self._check_daily_macd_cross()

        # 检测日线2卖
        self._daily_second_sell_detected = self._detect_daily_second_sell()

        self._last_daily_count = len(df)

    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> float:
        """计算ATR (Average True Range)"""
        if period is None:
            period = self.atr_period

        if len(df) < period + 1:
            return 0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_list.append(max(tr1, tr2, tr3))

        if len(tr_list) == 0:
            return 0

        return np.mean(tr_list[-period:])

    def _check_trend_condition(self) -> None:
        """检查趋势条件 - 优化版"""
        if self._weekly_ma_fast is None or self._weekly_ma_slow is None:
            self._is_uptrend = False
            return

        # 获取最新值
        ma_fast = self._weekly_ma_fast.iloc[-1] if not pd.isna(self._weekly_ma_fast.iloc[-1]) else 0
        ma_slow = self._weekly_ma_slow.iloc[-1] if not pd.isna(self._weekly_ma_slow.iloc[-1]) else 0

        # 检查最近N周的趋势
        if len(self._weekly_ma_fast) < self.trend_lookback:
            self._is_uptrend = False
            return

        # 计算均线斜率
        ma_fast_recent = self._weekly_ma_fast.iloc[-self.trend_lookback:]
        ma_slow_recent = self._weekly_ma_slow.iloc[-self.trend_lookback:]

        # 均线向上：最新值 > 最旧值
        ma_fast_rising = ma_fast_recent.iloc[-1] > ma_fast_recent.iloc[0]
        ma_slow_rising = ma_slow_recent.iloc[-1] > ma_slow_recent.iloc[0]

        # 上升趋势条件：
        # 1. 快线在慢线上方 (或者接近)
        # 2. 快线或慢线呈上升态势
        self._is_uptrend = (
            (ma_fast > ma_slow * 0.98) and  # 允许1%的容差
            (ma_fast_rising or ma_slow_rising)
        )

    def _check_weekly_macd_cross(self) -> None:
        """检查周线MACD金叉"""
        if not self._weekly_macd or len(self._weekly_macd) < 2:
            self._weekly_macd_golden_cross = False
            return

        # 使用MACD类的方法
        self._weekly_macd_golden_cross = self._weekly_macd.check_golden_cross()

    def _check_daily_macd_cross(self) -> None:
        """检查日线MACD金叉"""
        if not self._daily_macd or len(self._daily_macd) < 2:
            self._daily_macd_golden_cross = False
            return

        # 使用MACD类的方法
        self._daily_macd_golden_cross = self._daily_macd.check_golden_cross()

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

        last = self._daily_strokes[-1]
        second_last = self._daily_strokes[-2]

        if last.is_down and second_last.is_up:
            if last.end_value < second_last.start_value * 0.98:
                return True

        return False

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series
    ) -> Optional[Signal]:
        """检查买入信号 - 优化版周线2买"""
        if not self._weekly_strokes:
            return None

        # 确认是向上笔
        if not self._weekly_strokes[-1].is_up:
            return None

        # ========== 优化1: 趋势过滤 ==========
        if not self._is_uptrend:
            return None  # 不在上升趋势，不买入

        # 入场条件优化：有两种入场方式
        entry_reason = None
        confidence = 0.8

        # 方式1: 接近2买位置 (保守入场)
        if self._weekly_second_buy_price is not None:
            # 放宽到±10%
            if price > self._weekly_second_buy_price * 0.90 and price < self._weekly_second_buy_price * 1.10:
                entry_reason = f'2买入场 ({self._weekly_second_buy_price:.2f})'
                confidence = 0.9

        # 方式2: 突破前高 (趋势确认入场)
        if not entry_reason and len(self._weekly_strokes) >= 2:
            prev_stroke = self._weekly_strokes[-2]
            if prev_stroke and price > prev_stroke.end_value * 1.02:
                entry_reason = f'突破入场 ({prev_stroke.end_value:.2f})'
                confidence = 0.7

        if not entry_reason:
            return None

        # ========== 优化2: MACD金叉确认 (可选) ==========
        if self.require_macd_cross:
            # 需要周线或日线MACD金叉
            if not (self._weekly_macd_golden_cross or self._daily_macd_golden_cross):
                return None

        # 通过所有条件，生成买入信号
        reason_parts = [
            entry_reason,
            f'趋势向上' if self._is_uptrend else '',
        ]
        if self._weekly_macd_golden_cross:
            reason_parts.append('周线MACD金叉')
        if self._daily_macd_golden_cross:
            reason_parts.append('日线MACD金叉')

        reason = ' | '.join([p for p in reason_parts if p])

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            reason=reason,
            confidence=confidence
        )

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
            # ========== 优化3: 动态止损 ==========
            dynamic_stop_loss = self._calculate_dynamic_stop_loss(price)

            record = {
                'entry_price': price,
                'stop_loss': dynamic_stop_loss,
                'partial_exited': False,
                'highest_price': price,  # 跟踪最高价
            }
            self._position_record[symbol] = record

        # 更新最高价
        if price > record['highest_price']:
            record['highest_price'] = price
            # 更新动态止损（跟踪止损）
            if self._weekly_first_buy_price:
                # 使用1买低点和ATR的组合
                atr_stop = record['highest_price'] - self._daily_atr * self.atr_multiplier
                fixed_stop = self._weekly_first_buy_price
                record['stop_loss'] = max(atr_stop, fixed_stop)

        # 1. 动态止损
        if price < record['stop_loss']:
            stop_reason = f'止损({record["stop_loss"]:.2f})'
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=stop_reason,
                confidence=1.0
            )

        # 2. 日线MACD顶背离减仓50%
        if self._use_partial_exit and not record['partial_exited']:
            if self._check_daily_macd_divergence():
                record['partial_exited'] = True
                exit_qty = int(current_qty * self.exit_ratio)
                exit_qty = (exit_qty // 100) * 100

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

    def _calculate_dynamic_stop_loss(self, entry_price: float) -> float:
        """计算动态止损位"""
        # 方法1: 使用ATR
        atr_stop = entry_price - self._daily_atr * self.atr_multiplier if self._daily_atr else entry_price * 0.95

        # 方法2: 使用周线1买低点
        fixed_stop = self._weekly_first_buy_price if self._weekly_first_buy_price else entry_price * 0.95

        # 取两者中较高的作为止损位（较宽松）
        return max(atr_stop, fixed_stop)

    @property
    def _use_partial_exit(self) -> bool:
        """是否使用分批止盈"""
        return True

    def _check_daily_macd_divergence(self) -> bool:
        """检查日线MACD顶背驰"""
        if not self._daily_macd or len(self._daily_macd) < 20:
            return False

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
            dynamic_stop_loss = self._calculate_dynamic_stop_loss(executed_price)

            self._position_record[symbol] = {
                'entry_price': executed_price,
                'stop_loss': dynamic_stop_loss,
                'partial_exited': False,
                'highest_price': executed_price,
            }

        elif signal.is_sell():
            qty_to_sell = signal.quantity if signal.quantity else self.get_position(symbol)
            self.position[symbol] = self.position.get(symbol, 0) - qty_to_sell
            self.cash += executed_price * qty_to_sell

            # 如果全部卖出，清除记录
            if self.get_position(symbol) == 0:
                if symbol in self._position_record:
                    del self._position_record[symbol]
