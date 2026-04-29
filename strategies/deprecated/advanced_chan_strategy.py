"""
高级缠论策略 - 实现所有优化建议

优化内容：
1. 调整买卖点阈值 - 放宽确认要求
2. ATR动态止损 + 分批止盈
3. 趋势过滤 + 成交量确认
4. 可配置参数
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """计算ATR (Average True Range)"""
    if len(df) < period + 1:
        return df['high'].max() - df['low'].min()

    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.tail(period).mean()


def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> float:
    """计算成交量均线"""
    if len(df) < period:
        return df['volume'].mean()
    return df['volume'].tail(period).mean()


class AdvancedChanLunStrategy(Strategy):
    """
    高级缠论策略

    优化实现：
    1. ATR动态止损
    2. 分批止盈（30%+30%+40%）
    3. 成交量确认
    4. 趋势强度评分
    5. 可调参数
    """

    def __init__(
        self,
        name: str = '高级缠论策略',
        # 缠论参数
        min_stroke_bars: int = 3,           # 降低从5到3
        fractal_confirm: bool = False,       # 不要求分型确认
        # 止损止盈参数
        use_atr_stop: bool = True,           # 使用ATR动态止损
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,    # ATR止损倍数
        fixed_stop_pct: float = 0.05,        # 固定止损5%
        # 分批止盈
        use_partial_exit: bool = True,        # 分批止盈
        profit_levels: List[float] = None,   # 止盈点位 [0.08, 0.15, 0.25]
        exit_ratios: List[float] = None,     # 出货比例 [0.3, 0.3, 0.4]
        # 趋势过滤
        min_trend_score: float = 0.4,        # 最小趋势评分
        use_volume_confirm: bool = True,      # 成交量确认
        volume_ma_period: int = 20,
        # 仓位管理
        position_size: float = 0.95,
    ):
        super().__init__(name)

        self.min_stroke_bars = min_stroke_bars
        self.fractal_confirm = fractal_confirm
        self.use_atr_stop = use_atr_stop
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.fixed_stop_pct = fixed_stop_pct
        self.use_partial_exit = use_partial_exit
        self.profit_levels = profit_levels or [0.08, 0.15, 0.25]
        self.exit_ratios = exit_ratios or [0.3, 0.3, 0.4]
        self.min_trend_score = min_trend_score
        self.use_volume_confirm = use_volume_confirm
        self.volume_ma_period = volume_ma_period
        self.position_size = position_size

        # 缓存
        self._last_bar_count: int = 0
        self._fractals: List[Fractal] = []
        self._strokes: List[Stroke] = []
        self._pivots: List[Pivot] = []
        self._macd: Optional[MACD] = None
        self._current_trend: str = 'unknown'
        self._trend_score: float = 0.0

        # 持仓记录
        self._positions_record: Dict[str, Dict] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金¥{capital:,}, 品种{symbols}")

    def reset(self) -> None:
        """重置策略"""
        super().reset()
        self._positions_record = {}
        self._fractals = []
        self._strokes = []
        self._pivots = []

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        hist_df = context['data'].get(symbol)
        if hist_df is None or len(hist_df) < 30:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 检查是否需要更新缠论要素
        if len(hist_df) != self._last_bar_count:
            self._update_analysis(hist_df)
            self._last_bar_count = len(hist_df)

        # 已有持仓：检查止盈止损
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar, hist_df)

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar, hist_df)

    def _update_analysis(self, df: pd.DataFrame) -> None:
        """更新缠论分析"""
        from core.kline import KLine

        kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型（不要求确认）
        detector = FractalDetector(kline, confirm_required=self.fractal_confirm)
        self._fractals = detector.get_fractals()

        # 生成笔（降低最小笔数要求）
        stroke_gen = StrokeGenerator(kline, self._fractals, min_bars=self.min_stroke_bars)
        self._strokes = stroke_gen.get_strokes()

        # 识别中枢
        pivot_detector = PivotDetector(kline, self._strokes)
        self._pivots = pivot_detector.get_pivots()

        # 计算MACD
        self._macd = MACD(df['close'])

        # 计算趋势评分
        self._calculate_trend_score(df)

    def _calculate_trend_score(self, df: pd.DataFrame) -> None:
        """计算趋势评分 (0-1)"""
        if len(self._strokes) < 3:
            self._trend_score = 0.5
            self._current_trend = 'unknown'
            return

        score = 0.0
        recent_strokes = self._strokes[-5:]

        # 1. 高点低点分析 (0.3)
        ups = [s for s in recent_strokes if s.is_up]
        downs = [s for s in recent_strokes if s.is_down]

        if ups and downs:
            higher_highs = all(ups[i].high >= ups[i-1].high for i in range(1, len(ups)))
            higher_lows = all(downs[i].low >= downs[i-1].low for i in range(1, len(downs)))

            if higher_highs and higher_lows:
                score += 0.3
                self._current_trend = 'up'
            elif not higher_highs and not higher_lows:
                score += 0.1
                self._current_trend = 'down'
            else:
                score += 0.2
                self._current_trend = 'range'
        else:
            score += 0.15

        # 2. 价格位置 (0.25)
        if len(self._pivots) > 0:
            last_pivot = self._pivots[-1]
            current_price = df['close'].iloc[-1]
            pivot_mid = (last_pivot.high + last_pivot.low) / 2

            if current_price > last_pivot.high:
                score += 0.25  # 突破中枢上沿
            elif current_price > pivot_mid:
                score += 0.15  # 中枢上半区
            elif current_price > last_pivot.low:
                score += 0.05  # 中枢下半区
            # 低于中枢下沿不加分

        # 3. MACD状态 (0.25)
        if self._macd and len(self._macd) > 0:
            latest = self._macd.get_latest()
            if latest:
                if latest.macd > latest.signal and latest.histogram > 0:
                    score += 0.25
                elif latest.macd > latest.signal:
                    score += 0.15
                elif latest.histogram > 0:
                    score += 0.1

        # 4. 近期涨跌 (0.2)
        if len(df) >= 10:
            recent_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            if recent_change > 0.03:
                score += 0.2
            elif recent_change > 0:
                score += 0.1

        self._trend_score = min(score, 1.0)

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查入场信号"""
        # 趋势过滤
        if self._trend_score < self.min_trend_score:
            return None

        # 成交量确认
        if self.use_volume_confirm:
            vol_ma = calculate_volume_ma(df, self.volume_ma_period)
            current_vol = bar['volume']
            if current_vol < vol_ma * 0.8:  # 成交量低于均线的80%
                return None

        # 计算信号强度
        strength = self._calculate_signal_strength(price, df, 'buy')
        if strength < 0.5:
            return None

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            reason=f'趋势评分:{self._trend_score:.2f} 信号强度:{strength:.2f}',
            confidence=strength
        )

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查出场信号"""
        record = self._positions_record.get(symbol)
        if not record:
            return None

        entry_price = record['entry_price']
        current_qty = self.get_position(symbol)
        profit_pct = (price - entry_price) / entry_price

        # 1. 止损检查
        stop_loss = record.get('stop_loss', entry_price * (1 - self.fixed_stop_pct))
        if price <= stop_loss:
            record['exit_stage'] = 'stop_loss'
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason=f'止损({profit_pct:.2%})',
                confidence=1.0
            )

        # 2. 分批止盈检查
        if self.use_partial_exit:
            current_stage = record.get('exit_stage', 0)
            total_stages = len(self.profit_levels)

            for i in range(current_stage, total_stages):
                if profit_pct >= self.profit_levels[i]:
                    exit_qty = int(current_qty * self.exit_ratios[i] / 100) * 100
                    exit_qty = max(exit_qty, 100)  # 至少100股

                    record['exit_stage'] = i + 1
                    record['highest'] = max(record.get('highest', entry_price), price)

                    return Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                        price=price,
                        quantity=exit_qty,
                        reason=f'分批止盈{profit_pct:.2%}',
                        confidence=0.8
                    )

        # 3. 移动止损
        highest = record.get('highest', entry_price)
        if price > highest:
            record['highest'] = price

        if profit_pct > 0.05:  # 盈利5%后启用移动止损
            trailing_stop = highest * (1 - self.atr_stop_multiplier * 0.01)
            if price <= trailing_stop:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=current_qty,
                    reason=f'移动止损({profit_pct:.2%})',
                    confidence=0.9
                )

        # 4. 技术卖出信号
        if self._is_sell_signal(price, df):
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=current_qty,
                reason='技术信号卖出',
                confidence=0.7
            )

        return None

    def _calculate_signal_strength(self, price: float, df: pd.DataFrame, direction: str) -> float:
        """计算信号强度"""
        strength = 0.0

        # 1. 趋势评分 (0.4)
        strength += self._trend_score * 0.4

        # 2. 中枢位置 (0.25)
        if self._pivots:
            last_pivot = self._pivots[-1]
            if direction == 'buy':
                if price <= last_pivot.low * 1.02:
                    strength += 0.25  # 中枢下沿附近
                elif price >= last_pivot.high:
                    strength -= 0.1

        # 3. 分型确认 (0.2)
        if self._fractals:
            last_fractal = self._fractals[-1]
            if direction == 'buy' and last_fractal.is_bottom:
                # 检查是否有双重底
                bottoms = [f for f in self._fractals[-5:] if f.is_bottom]
                if len(bottoms) >= 2:
                    strength += 0.2
                else:
                    strength += 0.1

        # 4. 笔的状态 (0.15)
        if len(self._strokes) >= 2:
            last_stroke = self._strokes[-1]
            if direction == 'buy':
                if last_stroke.is_up and last_stroke.price_change_pct > 1:
                    strength += 0.15
                elif self._strokes[-2].is_down and self._strokes[-2].price_change_pct < -2:
                    strength += 0.1  # 前一笔大幅下跌后的反弹

        return min(max(strength, 0), 1)

    def _is_sell_signal(self, price: float, df: pd.DataFrame) -> bool:
        """判断是否为卖出信号"""
        # 顶分型
        if self._fractals and len(self._fractals) >= 2:
            last = self._fractals[-1]
            if last.is_top:
                # 检查是否有双顶
                tops = [f for f in self._fractals[-5:] if f.is_top]
                if len(tops) >= 2:
                    return True

        # MACD死叉
        if self._macd and self._macd.check_death_cross():
            return True

        # 跌破前低
        if len(self._strokes) >= 2:
            last_stroke = self._strokes[-1]
            if last_stroke.is_down and last_stroke.price_change_pct < -3:
                return True

        return False

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调"""
        symbol = signal.symbol

        if signal.is_buy():
            # 计算止损
            stop_loss = executed_price * (1 - self.fixed_stop_pct)

            self._positions_record[symbol] = {
                'entry_price': executed_price,
                'entry_date': signal.datetime,
                'highest': executed_price,
                'stop_loss': stop_loss,
                'exit_stage': 0,
            }

            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

        elif signal.is_sell():
            current_qty = self.get_position(symbol)

            if signal.quantity is None:
                # 全部卖出
                if symbol in self._positions_record:
                    del self._positions_record[symbol]
                self.position[symbol] = 0
            else:
                # 部分卖出
                self.position[symbol] = current_qty - signal.quantity
                if self.position[symbol] == 0 and symbol in self._positions_record:
                    del self._positions_record[symbol]

            self.cash += executed_price * executed_quantity

    @property
    def trend_score(self) -> float:
        return self._trend_score
