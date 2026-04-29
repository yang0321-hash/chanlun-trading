"""
缠论延长止盈策略

核心改进：
1. 提高目标位（20% → 30%）
2. 放宽移动止损（8% → 12%）
3. 延后启用移动止损（15% → 20%）
4. 2卖只减仓不平仓
5. 只在明确破位时才清仓
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field

from backtest.strategy import Strategy, Signal, SignalType
from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD
from indicator.volume import VolumeAnalyzer
from backtest.position_sizing import RiskParitySizer


@dataclass
class PositionRecord:
    """持仓记录"""
    symbol: str
    entry_price: float
    entry_date: datetime
    quantity: int
    stop_loss: float
    initial_stop: float
    target_price: float
    buy_point_type: str
    highest_price: float = field(default_factory=lambda: 0.0)
    partial_exit_done: bool = False
    trailing_stop_activated: bool = False
    exit_stage: int = 0  # 0=无, 1=第一次减仓, 2=第二次减仓


class ExtendedProfitChanLun(Strategy):
    """
    缠论延长止盈策略

    让利润充分奔跑：
    - 目标位20-30%
    - 移动止损12%
    - 盈利20%后才启用
    - 2卖只减仓50%
    """

    def __init__(
        self,
        name: str = '缠论延长止盈',
        # 基础参数
        max_risk_per_trade: float = 0.02,
        max_drawdown_pct: float = 0.20,
        # 买入参数
        enable_buy1: bool = False,
        enable_buy2: bool = True,
        enable_buy3: bool = False,
        min_confidence: float = 0.60,
        # 量能参数
        enable_volume_confirm: bool = True,
        min_volume_ratio: float = 1.2,
        enable_volume_divergence: bool = False,
        # 延长止盈参数
        target_profit_pct: float = 0.25,      # 目标25%
        trailing_stop_pct: float = 0.12,       # 12%移动止损
        trailing_activate_pct: float = 0.20,   # 盈利20%后启用
        # 分批减仓
        first_exit_pct: float = 0.12,          # 盈利12%第一次减仓
        first_exit_ratio: float = 0.30,        # 减仓30%
        second_exit_pct: float = 0.20,         # 盈利20%第二次减仓
        second_exit_ratio: float = 0.30,       # 减仓30%
    ):
        super().__init__(name)

        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct
        self.enable_buy1 = enable_buy1
        self.enable_buy2 = enable_buy2
        self.enable_buy3 = enable_buy3
        self.min_confidence = min_confidence
        self.enable_volume_confirm = enable_volume_confirm
        self.min_volume_ratio = min_volume_ratio
        self.enable_volume_divergence = enable_volume_divergence

        self.target_profit_pct = target_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activate_pct = trailing_activate_pct
        self.first_exit_pct = first_exit_pct
        self.first_exit_ratio = first_exit_ratio
        self.second_exit_pct = second_exit_pct
        self.second_exit_ratio = second_exit_ratio

        # 数据缓存
        self._daily_kline: Optional[KLine] = None
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_pivots: List[Pivot] = []
        self._daily_macd: Optional[MACD] = None
        self._volume_analyzer: Optional[VolumeAnalyzer] = None

        # 市场状态
        self._market_trend: str = 'unknown'
        self._trend_strength: float = 0.5

        # 持仓管理
        self._positions: Dict[str, PositionRecord] = {}

        # 缓存标识
        self._last_daily_count: int = 0

        # 风控
        self._is_paused: bool = False
        self._peak_equity: float = 0

    def initialize(self, capital: float, symbols: List[str]) -> None:
        self._peak_equity = capital
        logger.info(f"初始化{self.name}: 初始资金CNY{capital:,.0f}")

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        daily_df = context['data'].get(symbol)
        if daily_df is None or len(daily_df) < 60:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 更新权益
        equity = self.get_equity(context.get('current_prices', {symbol: current_price}))
        if equity > self._peak_equity:
            self._peak_equity = equity

        # 检查回撤暂停
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self.max_drawdown_pct:
            if not self._is_paused:
                logger.warning(f"最大回撤{drawdown:.1%}暂停交易")
                self._is_paused = True
            if current_position > 0:
                return self._check_stop_loss_only(symbol, current_price, bar)
            return None

        if drawdown < self.max_drawdown_pct * 0.5:
            if self._is_paused:
                self._is_paused = False

        # 更新分析
        self._update_analysis(daily_df)
        self._update_market_state(daily_df)

        # 已有持仓：检查出场
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar, daily_df)

        if self._is_paused:
            return None

        # 检查入场
        return self._check_entry_signals(symbol, current_price, bar, daily_df)

    def _update_analysis(self, df: pd.DataFrame) -> None:
        if len(df) == self._last_daily_count:
            return

        self._daily_kline = KLine.from_dataframe(df, strict_mode=False)
        detector = FractalDetector(self._daily_kline, confirm_required=False)
        self._daily_fractals = detector.get_fractals()
        stroke_gen = StrokeGenerator(self._daily_kline, self._daily_fractals, min_bars=3)
        self._daily_strokes = stroke_gen.get_strokes()
        pivot_detector = PivotDetector(self._daily_kline, self._daily_strokes)
        self._daily_pivots = pivot_detector.get_pivots()
        self._daily_macd = MACD(df['close'])

        if len(df) >= 30:
            self._volume_analyzer = VolumeAnalyzer(
                df['close'].values,
                df['volume'].values
            )

        self._last_daily_count = len(df)

    def _update_market_state(self, df: pd.DataFrame) -> None:
        if len(self._daily_strokes) < 5:
            return

        recent_strokes = self._daily_strokes[-10:]
        ups = [s for s in recent_strokes if s.is_up]
        downs = [s for s in recent_strokes if s.is_down]

        if not ups or not downs:
            return

        higher_highs = all(ups[i].end_value >= ups[i-1].end_value for i in range(1, len(ups)))
        higher_lows = all(downs[i].end_value >= downs[i-1].end_value for i in range(1, len(downs)))

        if higher_highs and higher_lows:
            self._market_trend = 'up'
            self._trend_strength = 0.8
        elif not higher_highs and not higher_lows:
            self._market_trend = 'down'
            self._trend_strength = 0.8
        else:
            self._market_trend = 'range'
            self._trend_strength = 0.4

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        # 趋势过滤
        if self._market_trend == 'down' and self._trend_strength > 0.7:
            return None

        if len(self._daily_strokes) < 3:
            return None

        last_stroke = self._daily_strokes[-1]
        if not last_stroke.is_up:
            return None

        # 检测2买
        prev_down_strokes = [s for s in self._daily_strokes[-5:] if s.is_down]
        if not prev_down_strokes or len(prev_down_strokes) < 2:
            return None

        last_down = prev_down_strokes[-1]
        prev_low = prev_down_strokes[-2].low

        # 2买条件
        if last_down.low < prev_low * 0.98:  # 创新低，不是2买
            return None

        confidence = 0.70
        reason = "2买回踩不创新低"

        # MACD金叉
        if self._daily_macd and self._daily_macd.check_golden_cross():
            confidence += 0.10
            reason += "+MACD金叉"

        # 量能确认
        if self.enable_volume_confirm and self._volume_analyzer:
            vol_confirmed, vol_reason = self._volume_analyzer.check_volume_confirmation(
                min_ratio=self.min_volume_ratio
            )
            if vol_confirmed:
                confidence += 0.10
                reason += f"+{vol_reason}"
            else:
                confidence -= 0.15

        # 趋势确认
        if self._market_trend == 'up':
            confidence += 0.10

        if confidence < self.min_confidence:
            return None

        # 计算止损
        stop_loss = last_down.low * 0.98

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            reason=f"{reason} (置信度:{confidence:.2f})",
            confidence=min(confidence, 0.95)
        )

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        record = self._positions.get(symbol)
        if not record:
            return None

        # 更新最高价
        if price > record.highest_price:
            record.highest_price = price

        profit_pct = (price - record.entry_price) / record.entry_price
        position = self.get_position(symbol)

        # 1. 硬止损（严格保护）
        if price <= record.stop_loss:
            return self._create_exit_signal(symbol, price, bar, position,
                                           f"止损: 亏损{profit_pct:.2%}")

        # 2. ATR动态止损（保护已实现利润）
        atr_stop = price * (1 - 0.08)  # 8%基础止损
        if profit_pct > 0.05:
            # 盈利>5%后，止损提高到成本价
            atr_stop = max(atr_stop, record.entry_price)
        if profit_pct > 0.10:
            # 盈利>10%后，止损提高到保本+5%
            atr_stop = max(atr_stop, record.entry_price * 1.05)

        if price <= atr_stop:
            return self._create_exit_signal(symbol, price, bar, position,
                                           f"动态止损: 盈利{profit_pct:.2%}")

        # 3. 分批减仓
        if record.exit_stage == 0 and profit_pct >= self.first_exit_pct:
            record.exit_stage = 1
            exit_qty = int(position * self.first_exit_ratio / 100) * 100
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=exit_qty,
                reason=f"第一目标({self.first_exit_pct:.0%})减仓{self.first_exit_ratio:.0%} (盈利{profit_pct:.2%})",
                confidence=0.9
            )

        if record.exit_stage == 1 and profit_pct >= self.second_exit_pct:
            record.exit_stage = 2
            exit_qty = int(position * self.second_exit_ratio / 100) * 100
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=exit_qty,
                reason=f"第二目标({self.second_exit_pct:.0%})减仓{self.second_exit_ratio:.0%} (盈利{profit_pct:.2%})",
                confidence=0.9
            )

        # 4. 宽松移动止损（只在盈利较大后启用）
        if profit_pct >= self.trailing_activate_pct:
            trailing_stop = record.highest_price * (1 - self.trailing_stop_pct)
            # 移动止损不能低于保本位
            trailing_stop = max(trailing_stop, record.entry_price * 1.05)
            if price <= trailing_stop:
                return self._create_exit_signal(symbol, price, bar, position,
                                               f"移动止损({self.trailing_stop_pct:.0%}): 盈利{profit_pct:.2%}")

        # 5. 目标位止盈
        if price >= record.target_price:
            return self._create_exit_signal(symbol, price, bar, position,
                                           f"目标止盈({self.target_profit_pct:.0%}): 盈利{profit_pct:.2%}")

        # 6. 强顶背离减仓（保护利润）
        if profit_pct > 0.15 and not record.partial_exit_done:
            if self._check_strong_divergence():
                record.partial_exit_done = True
                exit_qty = int(position * 0.50 / 100) * 100
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason=f"强顶背离减仓50% (盈利{profit_pct:.2%})",
                    confidence=0.8
                )

        # 7. 日线2卖减仓（不平仓，只减仓）
        if profit_pct > 0.10 and self._check_daily_second_sell():
            exit_qty = int(position * 0.50 / 100) * 100
            if exit_qty > 0:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason=f"日线2卖减仓50% (盈利{profit_pct:.2%})",
                    confidence=0.7
                )

        return None

    def _check_strong_divergence(self) -> bool:
        """检查强顶背离"""
        if not self._daily_macd or len(self._daily_macd) < 20:
            return False

        has_div, strength = self._daily_macd.check_divergence(
            max(0, len(self._daily_macd) - 20),
            len(self._daily_macd) - 1,
            'up'
        )

        # 只在背离强度较大时触发
        return has_div and strength > 0.3

    def _check_daily_second_sell(self) -> bool:
        """检查日线2卖"""
        if len(self._daily_strokes) < 3:
            return False

        last = self._daily_strokes[-1]
        second_last = self._daily_strokes[-2]

        if last.is_down and second_last.is_up:
            # 反弹明显不破前高才认为是2卖
            if last.end_value < second_last.start_value * 0.95:
                return True
        return False

    def _check_stop_loss_only(self, symbol: str, price: float, bar: pd.Series) -> Optional[Signal]:
        record = self._positions.get(symbol)
        if not record:
            return None
        if price <= record.stop_loss:
            return self._create_exit_signal(symbol, price, bar, self.get_position(symbol),
                                           "止损（风控暂停期）")
        return None

    def _create_exit_signal(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        quantity: int,
        reason: str
    ) -> Signal:
        return Signal(
            signal_type=SignalType.SELL,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            quantity=quantity,
            reason=reason,
            confidence=1.0
        )

    def on_order(
        self,
        signal: Signal,
        executed_price: float,
        executed_quantity: int
    ) -> None:
        symbol = signal.symbol

        if signal.is_buy():
            # 计算止损
            stop_loss = executed_price * 0.95
            if self._daily_strokes:
                prev_down = [s for s in self._daily_strokes[-5:] if s.is_down]
                if prev_down:
                    stop_loss = max(stop_loss, prev_down[-1].low * 0.98)

            self._positions[symbol] = PositionRecord(
                symbol=symbol,
                entry_price=executed_price,
                entry_date=signal.datetime,
                quantity=executed_quantity,
                stop_loss=stop_loss,
                initial_stop=stop_loss,
                target_price=executed_price * (1 + self.target_profit_pct),
                buy_point_type="2买",
                highest_price=executed_price,
            )

            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

            logger.info(
                f"买入 {symbol} @{executed_price:.2f} x {executed_quantity} "
                f"| 止损:{stop_loss:.2f} 目标:{self._positions[symbol].target_price:.2f} | {signal.reason}"
            )

        elif signal.is_sell():
            qty = signal.quantity if signal.quantity else self.get_position(symbol)
            self.position[symbol] = self.position.get(symbol, 0) - qty
            self.cash += executed_price * qty

            if symbol in self._positions:
                profit = (executed_price - self._positions[symbol].entry_price) * qty
                profit_pct = (executed_price - self._positions[symbol].entry_price) / self._positions[symbol].entry_price

                logger.info(
                    f"卖出 {symbol} @{executed_price:.2f} x {qty} "
                    f"| 盈亏:{profit:,.0f}({profit_pct:.2%}) | {signal.reason}"
                )

            if self.get_position(symbol) == 0:
                self._positions.pop(symbol, None)

    def get_system_state(self) -> Dict[str, Any]:
        return {
            'is_paused': self._is_paused,
            'market_trend': self._market_trend,
            'trend_strength': self._trend_strength,
            'positions': len(self._positions),
            'peak_equity': self._peak_equity,
        }
