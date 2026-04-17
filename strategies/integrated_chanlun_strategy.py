"""
整合缠论策略

整合现有最佳组件的缠论交易策略：
- 买卖点识别: core.buy_sell_points.BuySellPointDetector
- 入场过滤: strategies.enhanced_entry_conditions.EnhancedEntryFilter
- 仓位管理: backtest.position_sizing.PositionSizer
- 出场管理: strategies.enhanced_exit_conditions.DynamicExitManager

这是经过Agent辩论系统（组A方案组+组B审计组）5轮迭代后确定的最终方案。
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.segment import SegmentGenerator, Segment
from core.pivot import PivotDetector, Pivot
from core.buy_sell_points import BuySellPointDetector, BuySellPoint
from indicator.macd import MACD
from strategies.enhanced_exit_conditions import DynamicExitManager


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """计算ATR"""
    if len(df) < period + 1:
        return df['high'].max() - df['low'].min()

    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> float:
    """计算成交量均线"""
    if len(df) < period:
        return float(df['volume'].mean())
    return float(df['volume'].tail(period).mean())


class IntegratedChanLunStrategy(Strategy):
    """
    整合缠论策略

    将精确的买卖点识别与成熟的过滤、仓位、出场管理模块整合。
    """

    def __init__(
        self,
        name: str = '整合缠论策略',
        # 买卖点参数
        divergence_threshold: float = 0.3,
        min_confidence: float = 0.6,
        # 出场参数
        use_trailing_stop: bool = True,
        use_partial_profit: bool = True,
        trailing_activation: float = 0.05,
        trailing_offset: float = 0.08,
        profit_targets: List[tuple] = None,
        # 仓位参数
        max_position_pct: float = 0.25,
        min_unit: int = 100,
        # 过滤参数
        use_volume_confirm: bool = True,
        volume_ma_period: int = 20,
        min_volume_ratio: float = 1.2,
        # 冷却期
        cooldown_bars: int = 10,
    ):
        super().__init__(name)

        # 买卖点参数
        self.divergence_threshold = divergence_threshold
        self.min_confidence = min_confidence

        # 出场管理
        self.exit_manager = DynamicExitManager(
            use_trailing_stop=use_trailing_stop,
            use_partial_profit=use_partial_profit,
            use_time_stop=True,
        )
        self.trailing_activation = trailing_activation
        self.trailing_offset = trailing_offset
        self.profit_targets = profit_targets or [
            (0.05, 0.3),   # 盈利5%卖30%
            (0.10, 0.3),   # 盈利10%卖30%
            (0.15, 0.4),   # 盈利15%卖剩余
        ]

        # 仓位参数
        self.max_position_pct = max_position_pct
        self.min_unit = min_unit

        # 过滤参数
        self.use_volume_confirm = use_volume_confirm
        self.volume_ma_period = volume_ma_period
        self.min_volume_ratio = min_volume_ratio

        # 冷却期
        self.cooldown_bars = cooldown_bars
        self._last_sell_index: Dict[str, int] = {}

        # 当前bar索引追踪（用于on_order中记录冷却期）
        self._current_bar_index: Dict[str, int] = {}

        # 缓存
        self._last_bar_count: int = 0
        self._fractals: List[Fractal] = []
        self._strokes: List[Stroke] = []
        self._segments: List[Segment] = []
        self._pivots: List[Pivot] = []
        self._macd: Optional[MACD] = None
        self._detector: Optional[BuySellPointDetector] = None

        # 持仓记录
        self._positions_record: Dict[str, Dict] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金{capital:,}元")

    def reset(self) -> None:
        """重置策略"""
        super().reset()
        self._positions_record = {}
        self._last_sell_index = {}
        self._fractals = []
        self._strokes = []
        self._segments = []
        self._pivots = []

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        hist_df = context.get('data', {}).get(symbol) if isinstance(context.get('data'), dict) else context.get('data', {}).get(symbol)
        if hist_df is None or len(hist_df) < 60:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 记录当前bar索引
        self._current_bar_index[symbol] = index

        # 记录当前bar索引
        self._current_bar_index[symbol] = index

        # 更新缠论分析（有变化时才更新）
        if len(hist_df) != self._last_bar_count:
            self._update_analysis(hist_df)
            self._last_bar_count = len(hist_df)

        # 已有持仓：检查出场
        if current_position > 0:
            return self._check_exit(symbol, current_price, bar, hist_df)

        # 无持仓：检查入场
        return self._check_entry(symbol, current_price, bar, hist_df)

    def _update_analysis(self, df: pd.DataFrame) -> None:
        """更新缠论分析"""
        kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(kline, confirm_required=False)
        self._fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(kline, self._fractals)
        self._strokes = stroke_gen.get_strokes()

        # 生成线段
        seg_gen = SegmentGenerator(kline, self._strokes)
        self._segments = seg_gen.get_segments()

        # 识别中枢
        pivot_detector = PivotDetector(kline, self._strokes)
        self._pivots = pivot_detector.get_pivots()

        # 计算MACD
        self._macd = MACD(df['close'])

        # 初始化买卖点检测器
        self._detector = BuySellPointDetector(
            fractals=self._fractals,
            strokes=self._strokes,
            segments=self._segments,
            pivots=self._pivots,
            macd=self._macd,
            divergence_threshold=self.divergence_threshold,
        )

    def _check_entry(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame
    ) -> Optional[Signal]:
        """
        检查入场条件（结构化检测，非detect_latest_buy即时模式）

        检测逻辑：
        1. 必须有笔和中枢结构
        2. 最后一个分型必须是底分型
        3. 先尝试批量检测器查找近期买点
        4. 回退到简化缠论结构检测（1买/2买/3买）
        """
        if not self._strokes or not self._pivots:
            return None

        # 冷却期检查
        last_sell_idx = self._last_sell_index.get(symbol, -999)
        current_idx = len(df) - 1
        if current_idx - last_sell_idx < self.cooldown_bars:
            return None

        # 检查最后一个分型必须是底分型
        if not self._fractals:
            return None
        last_fractal = self._fractals[-1]
        if not last_fractal.is_bottom:
            return None

        # 尝试批量买卖点检测器获取买点（含置信度和止损）
        buy_point = None
        if self._detector is not None:
            self._detector.detect_all()
            buys = self._detector._buy_points

            if buys:
                # 取当前K线附近的买点（5根K线窗口）
                recent_buys = [b for b in buys if abs(b.index - current_idx) <= 5]
                if recent_buys:
                    best = max(recent_buys, key=lambda b: b.confidence)
                    if best.confidence >= self.min_confidence:
                        buy_point = best

        # 如果批量检测器没找到合格买点，使用简化结构检测
        if buy_point is None:
            buy_point = self._detect_structural_buy(price, current_idx)

        if buy_point is None:
            return None

        # 成交量确认
        if self.use_volume_confirm:
            vol_ma = calculate_volume_ma(df, self.volume_ma_period)
            current_vol = bar['volume']
            if current_vol < vol_ma * self.min_volume_ratio:
                return None

        # 计算仓位
        cash = self.get_cash()
        target_amount = cash * self.max_position_pct
        quantity = int(target_amount / price / self.min_unit) * self.min_unit
        if quantity <= 0:
            return None

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            quantity=quantity,
            reason=buy_point.reason,
            confidence=buy_point.confidence,
            metadata={
                'buy_point_type': buy_point.point_type,
                'stop_loss': buy_point.stop_loss,
                'divergence_ratio': getattr(buy_point, 'divergence_ratio', 0.0),
            }
        )

    def _detect_structural_buy(self, price: float, current_idx: int) -> Optional[BuySellPoint]:
        """
        简化缠论结构买点检测（不依赖批量检测器）

        依次检测：
        1. 第一类买点：底背驰（向下笔 + 中枢下 + MACD背驰）
        2. 第二类买点：回调不破前低
        3. 第三类买点：突破中枢回踩
        """
        # === 1买检测 ===
        first_buy = self._check_structural_first_buy(price)
        if first_buy is not None:
            return first_buy

        # === 2买检测 ===
        second_buy = self._check_structural_second_buy(price)
        if second_buy is not None:
            return second_buy

        # === 3买检测 ===
        third_buy = self._check_structural_third_buy(price)
        if third_buy is not None:
            return third_buy

        return None

    def _check_structural_first_buy(self, price: float) -> Optional[BuySellPoint]:
        """第一类买点：底背驰"""
        if len(self._strokes) < 3 or not self._pivots:
            return None

        down_strokes = [s for s in self._strokes if s.is_down]
        if len(down_strokes) < 2:
            return None

        last_down = down_strokes[-1]
        prev_down = down_strokes[-2]
        last_pivot = self._pivots[-1]

        # 最后一笔是向下笔
        if last_down != self._strokes[-1]:
            return None

        # 价格低于中枢下沿
        if last_down.end_value >= last_pivot.low:
            return None

        # MACD背驰检测
        divergence_ratio = 0.0
        if self._macd:
            has_div, divergence_ratio = self._macd.check_divergence(
                last_down.start_index, last_down.end_index, 'down',
                prev_start=prev_down.start_index,
                prev_end=prev_down.end_index
            )
            if not has_div:
                return None
        else:
            # 简单价格比较
            drop1 = abs(prev_down.price_change_pct)
            drop2 = abs(last_down.price_change_pct)
            if drop2 >= drop1 * 0.7:
                return None
            divergence_ratio = 1 - drop2 / drop1 if drop1 > 0 else 0

        confidence = 0.6 + min(divergence_ratio * 0.4, 0.4)
        stop_loss = last_down.low * 0.99

        return BuySellPoint(
            point_type='1buy',
            price=last_down.end_value,
            index=last_down.end_index,
            related_pivot=last_pivot,
            related_strokes=[prev_down, last_down],
            divergence_ratio=divergence_ratio,
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'1买: 底背驰(强度{divergence_ratio:.2f}), 中枢下沿{last_pivot.low:.2f}'
        )

    def _check_structural_second_buy(self, price: float) -> Optional[BuySellPoint]:
        """第二类买点：回调不破前低"""
        if not self._pivots or not self._strokes:
            return None

        last_pivot = self._pivots[-1]

        # 找中枢下方的向下笔（推断的1买低点）
        down_strokes = [s for s in self._strokes
                        if s.is_down and s.end_value < last_pivot.low]
        if not down_strokes:
            return None

        last_down = down_strokes[-1]
        implied_buy_price = last_down.end_value

        # 之后有向上笔
        up_after = [s for s in self._strokes
                    if s.is_up and s.start_index > last_down.end_index]
        if not up_after:
            return None

        # 之后又有向下笔回调
        last_up = up_after[-1]
        pullback = [s for s in self._strokes
                    if s.is_down and s.start_index > last_up.end_index]
        if not pullback:
            return None

        last_pullback = pullback[-1]

        # 回调不破推断的1买低点
        if last_pullback.end_value <= implied_buy_price:
            return None

        # 最后一笔必须是向下笔（回调中）
        if last_pullback != self._strokes[-1]:
            return None

        confidence = 0.6
        stop_loss = implied_buy_price * 0.99

        return BuySellPoint(
            point_type='2buy',
            price=last_pullback.end_value,
            index=last_pullback.end_index,
            related_pivot=last_pivot,
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'2买: 回调不破推断1买{implied_buy_price:.2f}'
        )

    def _check_structural_third_buy(self, price: float) -> Optional[BuySellPoint]:
        """第三类买点：突破中枢后回踩不进入中枢"""
        if not self._pivots:
            return None

        last_pivot = self._pivots[-1]

        # 价格在中枢上沿附近（回踩区间）
        if not (price > last_pivot.high * 0.99 and price < last_pivot.high * 1.08):
            return None

        # 有向上笔突破中枢
        breakout_strokes = [s for s in self._strokes
                            if s.is_up and s.end_value > last_pivot.high
                            and s.start_index >= last_pivot.end_index]
        if not breakout_strokes:
            return None

        last_breakout = breakout_strokes[-1]

        # 突破后有回调笔
        pullback_strokes = [s for s in self._strokes
                            if s.is_down and s.start_index > last_breakout.end_index]
        if not pullback_strokes:
            return None

        last_pullback = pullback_strokes[-1]

        # 回调低点仍在中枢上沿之上
        if last_pullback.end_value <= last_pivot.high:
            return None

        # 最后一笔是向下回调笔
        if last_pullback != self._strokes[-1]:
            return None

        margin = (last_pullback.end_value - last_pivot.high) / last_pivot.high
        confidence = 0.5 + min(margin * 10, 0.3)
        stop_loss = last_pivot.high * 0.99

        return BuySellPoint(
            point_type='3buy',
            price=last_pullback.end_value,
            index=last_pullback.end_index,
            related_pivot=last_pivot,
            related_strokes=[last_breakout, last_pullback],
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'3买: 回踩不破中枢上沿{last_pivot.high:.2f}'
        )

    def _check_exit(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查出场条件"""
        record = self._positions_record.get(symbol)
        current_qty = self.get_position(symbol)

        if not record:
            return None

        entry_price = record['entry_price']
        profit_pct = (price - entry_price) / entry_price

        # 1. 缠论止损（优先级最高）
        chan_stop = record.get('chan_stop_loss', 0)
        if chan_stop > 0 and price <= chan_stop:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'缠论止损: 跌破{chan_stop:.2f} ({profit_pct:.2%})',
                confidence=1.0
            )

        # 2. 固定止损兜底
        fixed_stop = record.get('fixed_stop_loss', entry_price * 0.95)
        if price <= fixed_stop:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'固定止损: 跌破{fixed_stop:.2f} ({profit_pct:.2%})',
                confidence=1.0
            )

        # 3. 跟踪止损
        highest = record.get('highest_price', entry_price)
        if price > highest:
            record['highest_price'] = price

        if profit_pct > self.trailing_activation:
            trailing_stop = highest * (1 - self.trailing_offset)
            if price <= trailing_stop:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=current_qty,
                    reason=f'跟踪止损: 最高{highest:.2f} 回撤{profit_pct:.2%}',
                    confidence=0.9
                )

        # 4. 分批止盈
        current_stage = record.get('exit_stage', 0)
        if current_stage < len(self.profit_targets):
            target_pct, exit_ratio = self.profit_targets[current_stage]
            if profit_pct >= target_pct:
                exit_qty = int(current_qty * exit_ratio)
                exit_qty = (exit_qty // self.min_unit) * self.min_unit
                exit_qty = max(exit_qty, self.min_unit)

                record['exit_stage'] = current_stage + 1

                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason=f'分批止盈({current_stage + 1}): 盈利{profit_pct:.2%}',
                    confidence=0.8
                )

        # 5. 缠论卖点信号（使用批量检测 + 结构化检测）
        sell_point = self._detect_structural_sell(price)
        if sell_point is not None:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'缠论{sell_point.reason}',
                confidence=sell_point.confidence
            )

        return None

    def _detect_structural_sell(self, price: float) -> Optional[BuySellPoint]:
        """
        缠论结构化卖点检测

        检测：
        1. 第一类卖点：顶背驰（向上笔 + 中枢上 + MACD背驰）
        2. 第二类卖点：反弹不破前高
        """
        if not self._fractals or not self._strokes:
            return None

        # 必须有顶分型
        last_fractal = self._fractals[-1]
        if not last_fractal.is_top:
            return None

        # === 1卖检测：顶背驰 ===
        if len(self._strokes) >= 3 and self._pivots:
            up_strokes = [s for s in self._strokes if s.is_up]
            if len(up_strokes) >= 2:
                last_up = up_strokes[-1]
                prev_up = up_strokes[-2]
                last_pivot = self._pivots[-1]

                # 最后一笔是向上笔
                if last_up == self._strokes[-1]:
                    # 价格高于中枢上沿
                    if last_up.end_value > last_pivot.high:
                        # MACD背驰检测
                        divergence_ratio = 0.0
                        has_div = False
                        if self._macd:
                            has_div, divergence_ratio = self._macd.check_divergence(
                                last_up.start_index, last_up.end_index, 'up',
                                prev_start=prev_up.start_index,
                                prev_end=prev_up.end_index
                            )

                        if has_div or not self._macd:
                            if not self._macd:
                                rise1 = prev_up.price_change_pct
                                rise2 = last_up.price_change_pct
                                if rise2 >= rise1 * 0.7:
                                    return self._check_structural_second_sell(price)
                                divergence_ratio = 1 - rise2 / rise1 if rise1 > 0 else 0

                            confidence = 0.6 + min(divergence_ratio * 0.4, 0.4)
                            return BuySellPoint(
                                point_type='1sell',
                                price=last_up.end_value,
                                index=last_up.end_index,
                                related_pivot=last_pivot,
                                related_strokes=[prev_up, last_up],
                                divergence_ratio=divergence_ratio,
                                confidence=confidence,
                                reason=f'1卖: 顶背驰(强度{divergence_ratio:.2f})'
                            )

        # === 2卖检测：反弹不破前高 ===
        return self._check_structural_second_sell(price)

    def _check_structural_second_sell(self, price: float) -> Optional[BuySellPoint]:
        """第二类卖点：反弹不破前高"""
        if not self._pivots or not self._strokes:
            return None

        last_pivot = self._pivots[-1]

        # 找中枢上方的向上笔（推断的1卖高点）
        up_strokes = [s for s in self._strokes
                      if s.is_up and s.end_value > last_pivot.high]
        if not up_strokes:
            return None

        last_up = up_strokes[-1]
        implied_sell_price = last_up.end_value

        # 之后有向下笔
        down_after = [s for s in self._strokes
                      if s.is_down and s.start_index > last_up.end_index]
        if not down_after:
            return None

        last_down = down_after[-1]

        # 之后又有向上笔反弹
        bounce = [s for s in self._strokes
                  if s.is_up and s.start_index > last_down.end_index]
        if not bounce:
            return None

        last_bounce = bounce[-1]

        # 反弹不破推断的1卖高点
        if last_bounce.end_value >= implied_sell_price:
            return None

        # 最后一笔必须是向上笔（反弹中）
        if last_bounce != self._strokes[-1]:
            return None

        confidence = 0.6

        return BuySellPoint(
            point_type='2sell',
            price=last_bounce.end_value,
            index=last_bounce.end_index,
            related_pivot=last_pivot,
            confidence=confidence,
            reason=f'2卖: 反弹不破推断1卖{implied_sell_price:.2f}'
        )

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调 - 修复双记账：只调用super，不手动更新position/cash"""
        super().on_order(signal, executed_price, executed_quantity)
        symbol = signal.symbol

        if signal.is_buy():
            # 计算止损位
            chan_stop = signal.metadata.get('stop_loss', 0) if signal.metadata else 0
            # ATR兜底止损
            fixed_stop = executed_price * 0.95

            # 取更严格的止损(R1修复: max取更高止损价=更严格)
            if chan_stop > 0:
                actual_stop = max(chan_stop, fixed_stop)
            else:
                actual_stop = fixed_stop

            self._positions_record[symbol] = {
                'entry_price': executed_price,
                'chan_stop_loss': chan_stop,
                'fixed_stop_loss': fixed_stop,
                'highest_price': executed_price,
                'exit_stage': 0,
            }

        elif signal.is_sell():
            # 全部清仓时移除记录
            remaining = self.get_position(symbol)
            if remaining <= 0:
                self._positions_record.pop(symbol, None)
                # 记录冷却期索引
                self._last_sell_index[symbol] = self._current_bar_index.get(symbol, 0)
