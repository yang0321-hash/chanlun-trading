"""
缠论终极增强版交易系统

整合多重指标提升胜率：
1. 量能分析 - 放量确认
2. RSI共振 - 超卖区域
3. 波动率过滤 - ATR识别
4. 市场环境 - 震荡/趋势区分
5. 动态止损 - ATR自适应
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


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """计算RSI指标"""
    rsi = np.full_like(prices, np.nan, dtype=float)

    if len(prices) < period + 1:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(prices)):
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

        # 更新平均值
        if i < len(deltas):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """计算ATR（平均真实波幅）"""
    atr = np.full_like(close, np.nan, dtype=float)

    if len(close) < period + 1:
        return atr

    tr_list = []
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr_list.append(max(tr1, tr2, tr3))

    for i in range(period - 1, len(tr_list)):
        atr[i + 1] = np.mean(tr_list[i - period + 1:i + 1])

    return atr


@dataclass
class MarketRegime:
    """市场环境状态"""
    is_trending: bool = False        # 是否趋势市
    is_ranging: bool = False         # 是否震荡市
    volatility_level: str = 'normal'  # low, normal, high
    trend_direction: str = 'neutral'  # up, down, neutral
    strength: float = 0.5            # 趋势强度 0-1


class EnhancedChanLunSystem(Strategy):
    """
    缠论终极增强版

    核心改进：
    1. 量能确认（放量才买）
    2. RSI超卖共振（安全边际）
    3. 波动率过滤（高波动减仓）
    4. 市场环境识别（趋势vs震荡）
    5. ATR动态止损
    """

    def __init__(
        self,
        name: str = '缠论终极增强版',
        # 基础参数
        max_risk_per_trade: float = 0.02,
        max_drawdown_pct: float = 0.15,
        # 买入参数
        enable_buy1: bool = False,
        enable_buy2: bool = True,
        enable_buy3: bool = True,
        min_confidence: float = 0.6,
        # 止损参数
        trailing_stop_pct: float = 0.08,
        trailing_activate_pct: float = 0.15,
        # 量能参数
        enable_volume_confirm: bool = True,
        min_volume_ratio: float = 1.5,
        # RSI参数
        enable_rsi_filter: bool = True,
        rsi_oversold: float = 40,      # RSI超卖阈值
        rsi_max: float = 70,           # RSI超买阈值
        # 波动率参数
        enable_volatility_filter: bool = True,
        max_atr_pct: float = 0.05,     # 最大ATR占比（超过则降低仓位）
        # 市场环境参数
        enable_regime_filter: bool = True,
        ranging_position_scale: float = 0.5,  # 震荡市仓位系数
    ):
        super().__init__(name)

        # 基础参数
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct
        self.enable_buy1 = enable_buy1
        self.enable_buy2 = enable_buy2
        self.enable_buy3 = enable_buy3
        self.min_confidence = min_confidence
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activate_pct = trailing_activate_pct

        # 量能参数
        self.enable_volume_confirm = enable_volume_confirm
        self.min_volume_ratio = min_volume_ratio

        # RSI参数
        self.enable_rsi_filter = enable_rsi_filter
        self.rsi_oversold = rsi_oversold
        self.rsi_max = rsi_max

        # 波动率参数
        self.enable_volatility_filter = enable_volatility_filter
        self.max_atr_pct = max_atr_pct

        # 市场环境参数
        self.enable_regime_filter = enable_regime_filter
        self.ranging_position_scale = ranging_position_scale

        # 数据缓存
        self._daily_kline: Optional[KLine] = None
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_pivots: List[Pivot] = []
        self._daily_macd: Optional[MACD] = None
        self._volume_analyzer: Optional[VolumeAnalyzer] = None

        # 指标缓存
        self._rsi: np.ndarray = np.array([])
        self._atr: np.ndarray = np.array([])
        self._current_atr: float = 0

        # 市场环境
        self._market_regime = MarketRegime()

        # 持仓管理
        self._positions: Dict[str, Dict] = {}

        # 缓存标识
        self._last_daily_count: int = 0

        # 风控状态
        self._is_paused: bool = False
        self._peak_equity: float = 0

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化系统"""
        super().initialize(capital, symbols)
        self._peak_equity = capital
        logger.info(f"初始化{self.name}: 初始资金CNY{capital:,.0f}")

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        daily_df = context['data'].get(symbol)
        if daily_df is None or len(daily_df) < 60:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 更新权益峰值
        equity = self.get_equity(context.get('current_prices', {symbol: current_price}))
        if equity > self._peak_equity:
            self._peak_equity = equity

        # 检查最大回撤
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self.max_drawdown_pct:
            if not self._is_paused:
                logger.warning(f"最大回撤{drawdown:.1%}超过限制，暂停交易")
                self._is_paused = True
            if current_position > 0:
                return self._check_stop_loss_only(symbol, current_price, bar)
            return None

        if drawdown < self.max_drawdown_pct * 0.5:
            if self._is_paused:
                logger.info(f"回撤恢复至{drawdown:.1%}，恢复交易")
                self._is_paused = False

        # 更新分析
        self._update_analysis(daily_df)
        self._update_market_regime(daily_df)

        # 已有持仓：检查出场信号
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar, daily_df)

        # 暂停状态不开新仓
        if self._is_paused:
            return None

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar, daily_df)

    def _update_analysis(self, df: pd.DataFrame) -> None:
        """更新技术分析"""
        if len(df) == self._last_daily_count:
            return

        # 缠论分析
        self._daily_kline = KLine.from_dataframe(df, strict_mode=False)
        detector = FractalDetector(self._daily_kline, confirm_required=False)
        self._daily_fractals = detector.get_fractals()
        stroke_gen = StrokeGenerator(self._daily_kline, self._daily_fractals, min_bars=3)
        self._daily_strokes = stroke_gen.get_strokes()
        pivot_detector = PivotDetector(self._daily_kline, self._daily_strokes)
        self._daily_pivots = pivot_detector.get_pivots()
        self._daily_macd = MACD(df['close'])

        # 量能分析
        if len(df) >= 30:
            self._volume_analyzer = VolumeAnalyzer(
                df['close'].values,
                df['volume'].values
            )

        # RSI计算
        self._rsi = calculate_rsi(df['close'].values)

        # ATR计算
        self._atr = calculate_atr(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
        if len(self._atr) > 0 and not np.isnan(self._atr[-1]):
            self._current_atr = self._atr[-1]

        self._last_daily_count = len(df)

    def _update_market_regime(self, df: pd.DataFrame) -> None:
        """更新市场环境判断"""
        if len(self._daily_strokes) < 5:
            return

        # 判断趋势
        recent_strokes = self._daily_strokes[-10:]
        ups = [s for s in recent_strokes if s.is_up]
        downs = [s for s in recent_strokes if s.is_down]

        if not ups or not downs:
            self._market_regime.trend_direction = 'neutral'
            return

        # 高点低点分析
        higher_highs = all(ups[i].end_value >= ups[i-1].end_value for i in range(1, len(ups)))
        higher_lows = all(downs[i].end_value >= downs[i-1].end_value for i in range(1, len(downs)))

        if higher_highs and higher_lows:
            self._market_regime.trend_direction = 'up'
            self._market_regime.is_trending = True
            self._market_regime.is_ranging = False
            self._market_regime.strength = 0.8
        elif not higher_highs and not higher_lows:
            self._market_regime.trend_direction = 'down'
            self._market_regime.is_trending = True
            self._market_regime.is_ranging = False
            self._market_regime.strength = 0.8
        else:
            self._market_regime.trend_direction = 'neutral'
            self._market_regime.is_trending = False
            self._market_regime.is_ranging = True
            self._market_regime.strength = 0.4

        # 波动率水平
        if self._current_atr > 0:
            atr_pct = self._current_atr / df['close'].iloc[-1]
            if atr_pct > self.max_atr_pct * 1.5:
                self._market_regime.volatility_level = 'high'
            elif atr_pct < self.max_atr_pct * 0.5:
                self._market_regime.volatility_level = 'low'
            else:
                self._market_regime.volatility_level = 'normal'

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查买入信号"""
        # 1. 趋势过滤 - 只做多或震荡
        if self._market_regime.trend_direction == 'down' and self._market_regime.strength > 0.7:
            return None

        # 2. 笔数据检查
        if len(self._daily_strokes) < 3:
            return None

        last_stroke = self._daily_strokes[-1]
        if not last_stroke.is_up:
            return None

        # 3. 计算基础置信度
        buy_type, confidence, reason = self._classify_buy_point(price, bar)

        if not buy_type or confidence < self.min_confidence - 0.1:
            return None

        # 4. RSI过滤
        if self.enable_rsi_filter and len(self._rsi) > 0:
            current_rsi = self._rsi[-1]
            if not np.isnan(current_rsi):
                # 超买区域不买
                if current_rsi > self.rsi_max:
                    return None
                # 超卖区域加分
                if current_rsi < self.rsi_oversold:
                    confidence += 0.10
                    reason += f"+RSI({current_rsi:.0f})超卖"
                elif current_rsi > 60:
                    confidence -= 0.10  # RSI偏高降低置信度

        # 5. 量能确认
        if self.enable_volume_confirm and self._volume_analyzer:
            vol_confirmed, vol_reason = self._volume_analyzer.check_volume_confirmation(
                min_ratio=self.min_volume_ratio
            )
            if vol_confirmed:
                confidence += 0.10
                reason += f"+{vol_reason}"
            else:
                confidence -= 0.15  # 量能不足降低置信度

        # 6. 波动率过滤
        if self.enable_volatility_filter:
            if self._market_regime.volatility_level == 'high':
                confidence -= 0.10
                reason += "+高波动"

        # 7. 市场环境调整
        if self.enable_regime_filter and self._market_regime.is_ranging:
            confidence *= self.ranging_position_scale  # 震荡市降低仓位

        # 8. 最终置信度检查
        if confidence < self.min_confidence:
            return None

        # 9. 计算止损位（ATR动态）
        if self._current_atr > 0:
            stop_loss = price - self._current_atr * 2
        else:
            stop_loss = price * 0.95

        # 10. 检查前低
        prev_down_strokes = [s for s in self._daily_strokes[-5:] if s.is_down]
        if prev_down_strokes:
            stop_loss = max(stop_loss, prev_down_strokes[-1].low * 0.98)

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            reason=f"{buy_type} {reason} (置信度:{confidence:.2f})",
            confidence=min(confidence, 0.95)
        )

    def _classify_buy_point(
        self,
        price: float,
        bar: pd.Series
    ) -> Tuple[str, float, str]:
        """分类买点"""
        if len(self._daily_strokes) < 3:
            return '', 0.0, ''

        last_stroke = self._daily_strokes[-1]
        if not last_stroke.is_up:
            return '', 0.0, ''

        prev_down_strokes = [s for s in self._daily_strokes[-5:] if s.is_down]
        if not prev_down_strokes:
            return '', 0.0, ''

        last_down = prev_down_strokes[-1]

        # 检查2买
        is_buy2 = False
        if len(prev_down_strokes) >= 2:
            prev_low = prev_down_strokes[-2].low
            if last_down.low >= prev_low * 0.98:
                is_buy2 = True

        # 检查3买
        is_buy3 = False
        if self._daily_pivots:
            last_pivot = self._daily_pivots[-1]
            if last_stroke.start_value > last_pivot.high:
                if last_down.low >= last_pivot.low * 0.98:
                    is_buy3 = True

        confidence = 0.0
        reason = ""

        if is_buy2 and self.enable_buy2:
            confidence = 0.70
            reason = "回踩不创新低"

            # MACD金叉
            if self._daily_macd and self._daily_macd.check_golden_cross():
                confidence += 0.10
                reason += "+MACD金叉"

            # 趋势确认
            if self._market_regime.trend_direction == 'up':
                confidence += 0.10
            elif self._market_regime.is_ranging:
                confidence += 0.05

            return '2买', min(confidence, 0.90), reason

        if is_buy3 and self.enable_buy3:
            confidence = 0.65
            reason = "突破后回踩"

            if self._market_regime.trend_direction == 'up':
                confidence += 0.15

            return '3买', min(confidence, 0.85), reason

        # 基础向上笔
        if self.enable_buy1 or self.enable_buy3:
            confidence = 0.50
            reason = "向上笔"

            return '向上笔', min(confidence, 0.70), reason

        return '', 0.0, ''

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查卖出信号"""
        record = self._positions.get(symbol)
        if not record:
            return None

        entry_price = record['entry_price']
        profit_pct = (price - entry_price) / entry_price

        # 更新最高价
        if price > record.get('highest_price', entry_price):
            record['highest_price'] = price

        # 1. 止损检查（ATR动态）
        stop_loss = record.get('stop_loss', entry_price * 0.95)

        # 动态更新止损
        if self._current_atr > 0 and profit_pct > 0:
            atr_stop = price - self._current_atr * 2.5
            if atr_stop > stop_loss:
                stop_loss = atr_stop

        if price <= stop_loss:
            return self._create_exit_signal(
                symbol, price, bar, self.get_position(symbol),
                f"止损: 亏损{profit_pct:.2%}"
            )

        # 2. 移动止损
        highest = record.get('highest_price', entry_price)
        if profit_pct > self.trailing_activate_pct:
            trailing_stop = highest * (1 - self.trailing_stop_pct)
            if price <= trailing_stop:
                return self._create_exit_signal(
                    symbol, price, bar, self.get_position(symbol),
                    f"移动止损: 盈利{profit_pct:.2%}"
                )

        # 3. RSI超买减仓
        if not record.get('rsi_exit_done', False) and profit_pct > 0.08:
            if len(self._rsi) > 0:
                current_rsi = self._rsi[-1]
                if not np.isnan(current_rsi) and current_rsi > self.rsi_max:
                    record['rsi_exit_done'] = True
                    exit_qty = int(self.get_position(symbol) * 0.5 / 100) * 100
                    return Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                        price=price,
                        quantity=exit_qty,
                        reason=f"RSI({current_rsi:.0f})超买减仓50% (盈利{profit_pct:.2%})",
                        confidence=0.8
                    )

        # 4. MACD顶背离减仓
        if not record.get('macd_exit_done', False) and profit_pct > 0.08:
            if self._check_macd_divergence():
                record['macd_exit_done'] = True
                exit_qty = int(self.get_position(symbol) * 0.5 / 100) * 100
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason=f"MACD顶背离减仓50% (盈利{profit_pct:.2%})",
                    confidence=0.8
                )

        # 5. 日线2卖清仓
        if self._check_second_sell():
            return self._create_exit_signal(
                symbol, price, bar, self.get_position(symbol),
                f"日线2卖: 盈利{profit_pct:.2%}"
            )

        return None

    def _check_stop_loss_only(self, symbol: str, price: float, bar: pd.Series) -> Optional[Signal]:
        """仅检查止损"""
        record = self._positions.get(symbol)
        if not record:
            return None

        stop_loss = record.get('stop_loss', 0)
        if stop_loss > 0 and price <= stop_loss:
            return self._create_exit_signal(
                symbol, price, bar, self.get_position(symbol),
                "止损（风控暂停期）"
            )
        return None

    def _check_macd_divergence(self) -> bool:
        """检查MACD顶背离"""
        if not self._daily_macd or len(self._daily_macd) < 20:
            return False
        has_div, _ = self._daily_macd.check_divergence(
            max(0, len(self._daily_macd) - 20),
            len(self._daily_macd) - 1,
            'up'
        )
        return has_div

    def _check_second_sell(self) -> bool:
        """检查日线2卖"""
        if len(self._daily_strokes) < 3:
            return False

        last = self._daily_strokes[-1]
        second_last = self._daily_strokes[-2]

        if last.is_down and second_last.is_up:
            if last.end_value < second_last.start_value * 0.98:
                return True
        return False

    def _create_exit_signal(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        quantity: int,
        reason: str
    ) -> Signal:
        """创建退出信号"""
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
        """订单成交回调"""
        symbol = signal.symbol

        if signal.is_buy():
            # 计算止损
            stop_loss = executed_price * 0.95
            if self._current_atr > 0:
                stop_loss = executed_price - self._current_atr * 2

            # 使用笔的低点
            if self._daily_strokes:
                prev_down = [s for s in self._daily_strokes[-5:] if s.is_down]
                if prev_down:
                    stop_loss = max(stop_loss, prev_down[-1].low * 0.98)

            self._positions[symbol] = {
                'entry_price': executed_price,
                'entry_date': signal.datetime,
                'quantity': executed_quantity,
                'stop_loss': stop_loss,
                'highest_price': executed_price,
                'buy_point_type': signal.reason.split()[0] if signal.reason else '',
            }

            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

            logger.info(
                f"买入 {symbol} @ {executed_price:.2f} x {executed_quantity} "
                f"| 止损:{stop_loss:.2f} | {signal.reason}"
            )

        elif signal.is_sell():
            qty = signal.quantity if signal.quantity else self.get_position(symbol)
            self.position[symbol] = self.position.get(symbol, 0) - qty
            self.cash += executed_price * qty

            if symbol in self._positions:
                profit = (executed_price - self._positions[symbol]['entry_price']) * qty
                profit_pct = (executed_price - self._positions[symbol]['entry_price']) / self._positions[symbol]['entry_price']

                logger.info(
                    f"卖出 {symbol} @ {executed_price:.2f} x {qty} "
                    f"| 盈亏:{profit:,.0f}({profit_pct:.2%}) | {signal.reason}"
                )

            if self.get_position(symbol) == 0:
                self._positions.pop(symbol, None)

    def get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_paused': self._is_paused,
            'market_trend': self._market_regime.trend_direction,
            'trend_strength': self._market_regime.strength,
            'volatility_level': self._market_regime.volatility_level,
            'is_ranging': self._market_regime.is_ranging,
            'positions': len(self._positions),
            'peak_equity': self._peak_equity,
        }
