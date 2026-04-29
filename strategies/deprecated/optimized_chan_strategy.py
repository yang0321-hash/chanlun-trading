"""
优化后的缠论交易策略

主要优化：
1. 避免重复计算缠论要素
2. 完善持仓管理和止损止盈
3. 改进买卖点判断逻辑
4. 添加仓位管理
5. 增加趋势过滤器
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.segment import SegmentGenerator
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD


class OptimizedChanLunStrategy(Strategy):
    """
    优化版缠论策略

    优化点：
    1. 缓存机制：只在新K线时重新计算
    2. 完善的止损止盈
    3. 仓位管理：根据信号强度调整仓位
    4. 趋势过滤：只在有利趋势中交易
    5. 多重确认：结合多个指标确认信号
    """

    def __init__(
        self,
        name: str = '优化缠论策略',
        use_macd: bool = True,
        use_stop_loss: bool = True,
        stop_loss_pct: float = 0.05,      # 5%止损
        take_profit_pct: float = 0.15,    # 15%止盈
        trailing_stop_pct: float = 0.03,  # 3%移动止损
        min_stroke_for_signal: int = 3,   # 最小笔数
        position_size: float = 0.95,      # 每次使用资金比例
    ):
        super().__init__(name)
        self.use_macd = use_macd
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.min_stroke_for_signal = min_stroke_for_signal
        self.position_size = position_size

        # 缓存
        self._last_bar_count: int = 0
        self._fractals: List[Fractal] = []
        self._strokes: List[Stroke] = []
        self._pivots: List[Pivot] = []
        self._macd: Optional[MACD] = None
        self._current_trend: str = 'unknown'

        # 持仓记录 {symbol: {'entry_price': float, 'entry_date': datetime, 'highest': float, 'stop_loss': float}}
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
        if hist_df is None or len(hist_df) < 50:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 检查是否需要更新缠论要素（缓存机制）
        if len(hist_df) != self._last_bar_count:
            self._update_analysis(hist_df)
            self._last_bar_count = len(hist_df)

        # 已有持仓：检查止盈止损
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar)

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar, hist_df)

    def _update_analysis(self, df: pd.DataFrame) -> None:
        """更新缠论分析（带缓存）"""
        from core.kline import KLine

        kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(kline, confirm_required=False)
        self._fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(kline, self._fractals, min_bars=5)
        self._strokes = stroke_gen.get_strokes()

        # 识别中枢
        pivot_detector = PivotDetector(kline, self._strokes)
        self._pivots = pivot_detector.get_pivots()

        # 计算MACD
        if self.use_macd:
            self._macd = MACD(df['close'])

        # 判断趋势
        self._update_trend()

    def _update_trend(self) -> None:
        """更新趋势判断"""
        if len(self._strokes) < 3:
            self._current_trend = 'unknown'
            return

        # 使用最近N笔判断趋势
        recent_strokes = self._strokes[-5:]

        # 计算高低点
        highs = [s.high for s in recent_strokes if s.is_up]
        lows = [s.low for s in recent_strokes if s.is_down]

        if not highs or not lows:
            self._current_trend = 'unknown'
            return

        # 波段高点是否递增
        higher_highs = all(highs[i] >= highs[i-1] for i in range(1, len(highs)))
        # 波段低点是否递增
        higher_lows = all(lows[i] >= lows[i-1] for i in range(1, len(lows)))

        if higher_highs and higher_lows:
            self._current_trend = 'up'
        elif not higher_highs and not higher_lows:
            self._current_trend = 'down'
        else:
            self._current_trend = 'range'

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查入场信号"""
        # 过滤条件：笔数不足
        if len(self._strokes) < self.min_stroke_for_signal:
            return None

        # 过滤条件：下降趋势中不做多（或等待更好的位置）
        if self._current_trend == 'down' and len(self._strokes) < 5:
            return None

        # 获取最后几个笔
        last_stroke = self._strokes[-1]
        second_last_stroke = self._strokes[-2] if len(self._strokes) >= 2 else None

        # 买入条件1：向上笔形成，且是底分型结束
        if last_stroke.is_up:
            # 检查是否形成有效底分型
            if self._is_valid_bottom_formed():
                strength = self._calculate_signal_strength(symbol, price, 'buy')
                if strength > 0.5:
                    return Signal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                        price=price,
                        reason=f'向上笔形成(强度:{strength:.2f})',
                        confidence=strength
                    )

        # 买入条件2：价格在中枢下沿附近企稳
        if self._pivots:
            last_pivot = self._pivots[-1]
            if price <= last_pivot.low * 1.02 and price >= last_pivot.low * 0.98:
                # 检查是否企稳（最近几根K线没有创新低）
                if self._is_price_stabilized(df, 3):
                    return Signal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                        price=price,
                        reason='中枢下沿企稳',
                        confidence=0.6
                    )

        # 买入条件3：MACD金叉
        if self._macd and self._macd.check_golden_cross():
            # 确认不在下降趋势中
            if self._current_trend != 'down':
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    reason='MACD金叉',
                    confidence=0.5
                )

        return None

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series
    ) -> Optional[Signal]:
        """检查出场信号"""
        record = self._positions_record.get(symbol)
        if not record:
            return None

        entry_price = record['entry_price']
        highest_price = record['highest']

        # 更新最高价
        if price > highest_price:
            record['highest'] = price

        # 计算收益率
        profit_pct = (price - entry_price) / entry_price

        # 1. 止损检查
        if self.use_stop_loss and price <= record.get('stop_loss', entry_price * (1 - self.stop_loss_pct)):
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                reason=f'止损({profit_pct:.2%})',
                confidence=1.0
            )

        # 2. 移动止损（追踪最高价）
        if profit_pct > 0.05:  # 盈利超过5%后才启用移动止损
            trailing_stop = highest_price * (1 - self.trailing_stop_pct)
            if price <= trailing_stop:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    reason=f'移动止损({profit_pct:.2%})',
                    confidence=0.9
                )

        # 3. 止盈检查
        if profit_pct >= self.take_profit_pct:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                reason=f'止盈({profit_pct:.2%})',
                confidence=0.8
            )

        # 4. 技术信号卖出
        if self._is_sell_signal_formed(price):
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                reason='技术信号卖出',
                confidence=0.7
            )

        return None

    def _is_valid_bottom_formed(self) -> bool:
        """检查是否形成有效底部"""
        if not self._fractals:
            return False

        # 最后一个分型是底分型
        last_fractal = self._fractals[-1]
        if not last_fractal.is_bottom:
            return False

        # 检查前面是否有更低的高点（形成下降趋势后的转折）
        if len(self._fractals) >= 3:
            recent_fractals = self._fractals[-3:]
            # 应该有顶-底-底的模式
            if recent_fractals[0].is_top and recent_fractals[1].is_bottom:
                # 第二个底分型不低于第一个底分型（形成双底或抬高的底）
                if recent_fractals[2].low >= recent_fractals[1].low * 0.98:
                    return True

        return len([f for f in self._fractals[-5:] if f.is_bottom]) >= 2

    def _is_price_stabilized(self, df: pd.DataFrame, lookback: int) -> bool:
        """检查价格是否企稳"""
        if len(df) < lookback + 1:
            return False

        recent = df.tail(lookback + 1)
        # 最近几根K线的低点没有创新低
        lows = recent['low'].values
        return all(lows[i] >= lows[i-1] * 0.99 for i in range(1, len(lows)))

    def _is_sell_signal_formed(self, price: float) -> bool:
        """检查卖出信号"""
        # 1. 顶分型形成
        if self._fractals and len(self._fractals) >= 2:
            last = self._fractals[-1]
            second_last = self._fractals[-2]

            if last.is_top and second_last.is_top:
                # 连续顶分型，可能形成顶部
                if last.high < second_last.high:
                    return True  # 降低的顶

        # 2. MACD死叉
        if self._macd and self._macd.check_death_cross():
            return True

        # 3. 价格跌破前一个低点
        if len(self._strokes) >= 2:
            last_stroke = self._strokes[-1]
            if last_stroke.is_down:
                prev_low = self._strokes[-2].low if self._strokes[-2].is_up else self._strokes[-2].end_value
                if price < prev_low:
                    return True

        return False

    def _calculate_signal_strength(self, symbol: str, price: float, direction: str) -> float:
        """计算信号强度"""
        strength = 0.0

        # 因素1：趋势一致性 (0.3)
        if direction == 'buy' and self._current_trend == 'up':
            strength += 0.3
        elif direction == 'sell' and self._current_trend == 'down':
            strength += 0.3
        elif self._current_trend == 'range':
            strength += 0.1

        # 因素2：中枢位置 (0.25)
        if self._pivots:
            last_pivot = self._pivots[-1]
            if direction == 'buy':
                # 在中枢下沿附近买入
                if price <= last_pivot.low * 1.03:
                    strength += 0.25
                elif price >= last_pivot.high:
                    strength -= 0.1  # 突破后不追高
        else:
            strength += 0.1  # 无中枢时给予基础分

        # 因素3：MACD确认 (0.25)
        if self._macd and len(self._macd) > 2:
            latest = self._macd.get_latest()
            if latest:
                if direction == 'buy' and latest.histogram > 0:
                    strength += 0.15
                elif direction == 'sell' and latest.histogram < 0:
                    strength += 0.15

        # 因素4：近期表现 (0.2)
        if len(self._strokes) >= 2:
            last_stroke = self._strokes[-1]
            change_pct = abs(last_stroke.price_change_pct)

            if direction == 'buy' and last_stroke.is_down:
                # 刚经历下跌，反弹机会
                if change_pct > 2:
                    strength += 0.1
                if change_pct > 5:
                    strength += 0.1

        return min(max(strength, 0), 1)

    def on_order(
        self,
        signal: Signal,
        executed_price: float,
        executed_quantity: int
    ) -> None:
        """订单成交回调"""
        symbol = signal.symbol

        if signal.is_buy():
            # 记录买入信息
            self._positions_record[symbol] = {
                'entry_price': executed_price,
                'entry_date': signal.datetime,
                'highest': executed_price,
                'stop_loss': executed_price * (1 - self.stop_loss_pct)
            }
            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

        elif signal.is_sell():
            # 清除买入记录
            if symbol in self._positions_record:
                del self._positions_record[symbol]
            self.position[symbol] = self.position.get(symbol, 0) - executed_quantity
            self.cash += executed_price * executed_quantity

    def get_position_size(self, symbol: str, price: float, confidence: float) -> int:
        """
        计算仓位大小

        根据信号强度和可用资金计算买入数量
        """
        available_cash = self.cash * self.position_size

        # 根据置信度调整仓位
        if confidence < 0.5:
            available_cash *= 0.5
        elif confidence > 0.8:
            available_cash *= 1.0
        else:
            available_cash *= 0.7

        # 计算股数（100的整数倍）
        shares = int(available_cash / price / 100) * 100

        return max(shares, 100)

    @property
    def fractals(self) -> List[Fractal]:
        return self._fractals

    @property
    def strokes(self) -> List[Stroke]:
        return self._strokes

    @property
    def pivots(self) -> List[Pivot]:
        return self._pivots

    @property
    def current_trend(self) -> str:
        return self._current_trend
