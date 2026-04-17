"""
稳定缠论策略 (StableChanLunStrategy)

核心逻辑：纯2买/3买入场 + 1买低点止损 + 2卖出场
- 入场：日线2买或3买 + 周线不能是下跌
- 止损：1买低点（2买）或ZG（3买）
- 出场：日线2卖（持仓>10根后）或 时间止损（60根）

仓位 = 风险金额 / 止损距离，每笔风险2%
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector, BuySellPoint
from indicator.macd import MACD


class StableChanLunStrategy(Strategy):
    """纯缠论买卖点策略 - 让利润奔跑"""

    def __init__(
        self,
        name: str = '稳定缠论策略',
        risk_per_trade: float = 0.02,
        max_positions: int = 3,
        max_drawdown_pct: float = 0.10,
        cooldown_bars: int = 5,
        time_stop_bars: int = 60,
        use_intraday_exit: bool = True,
        allowed_buy_types: Tuple[str, ...] = ('2buy',),
        min_hold_before_sell: int = 10,
        require_weekly_up: bool = True,
        max_stop_pct: float = 0.20,
    ):
        super().__init__(name)
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_bars = cooldown_bars
        self.time_stop_bars = time_stop_bars
        self.use_intraday_exit = use_intraday_exit
        self.allowed_buy_types = allowed_buy_types
        self.min_hold_before_sell = min_hold_before_sell
        self.require_weekly_up = require_weekly_up
        self.max_stop_pct = max_stop_pct

        # 缠论缓存
        self._fractals: List[Fractal] = []
        self._strokes: List = []
        self._segments: List = []
        self._pivots: List = []
        self._macd: Optional[MACD] = None
        self._detector: Optional[BuySellPointDetector] = None

        # 持仓追踪
        self._entry_prices: Dict[str, float] = {}
        self._entry_indices: Dict[str, int] = {}
        self._stop_losses: Dict[str, float] = {}
        self._highest_prices: Dict[str, float] = {}

        # 风控
        self._last_loss_index: Dict[str, int] = {}
        self._peak_equity: float = 0.0
        self._trading_halted: bool = False
        self._active_positions: int = 0

        self._raw_df: Optional[pd.DataFrame] = None

    def initialize(self, capital: float, symbols: List[str]) -> None:
        super().initialize(capital, symbols)
        self._peak_equity = capital
        logger.info(f"初始化{self.name}: 资金¥{capital:,.0f}")

    def on_bar(self, bar: pd.Series, symbol: str, index: int, context: Dict[str, Any]) -> Optional[Signal]:
        hist_df = context.get('data', {}).get(symbol)
        if hist_df is None or len(hist_df) < 60:
            return None

        try:
            from core.kline import KLine
            kline = KLine.from_dataframe(hist_df, strict_mode=True)
            self._raw_df = hist_df
            self._update_chanlun(kline)

            current_price = bar['close']
            position = self.get_position(symbol)
            self._check_portfolio_risk(current_price, symbol)

            if position > 0:
                return self._check_exit(current_price, index, symbol, context)
            else:
                return self._check_entry(current_price, index, symbol, hist_df)
        except Exception as e:
            logger.debug(f"分析错误 {symbol} @{index}: {e}")
            return None

    def _update_chanlun(self, kline) -> None:
        self._fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        self._strokes = StrokeGenerator(kline, self._fractals).get_strokes()
        self._segments = SegmentGenerator(kline, self._strokes).get_segments()
        self._pivots = PivotDetector(kline, self._strokes).get_pivots()
        df = kline.to_dataframe()
        self._macd = MACD(df['close'])

        if self._pivots and self._strokes:
            self._detector = BuySellPointDetector(
                fractals=self._fractals, strokes=self._strokes,
                segments=self._segments, pivots=self._pivots, macd=self._macd,
            )
            self._detector.detect_all()
        else:
            self._detector = None

    def _get_weekly_trend(self, hist_df: pd.DataFrame) -> str:
        if hist_df is None or len(hist_df) < 100:
            return 'neutral'
        try:
            weekly = hist_df.resample('W').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
            if len(weekly) < 20:
                return 'neutral'
            ma5 = weekly['close'].rolling(5).mean()
            ma20 = weekly['close'].rolling(20).mean()
            if pd.isna(ma20.iloc[-1]) or pd.isna(ma5.iloc[-1]):
                return 'neutral'
            if ma5.iloc[-1] > ma20.iloc[-1] * 1.003:
                return 'up'
            elif ma5.iloc[-1] < ma20.iloc[-1] * 0.997:
                return 'down'
            return 'neutral'
        except Exception:
            return 'neutral'

    # ==================== 入场 ====================

    def _check_entry(self, current_price: float, index: int, symbol: str, hist_df: pd.DataFrame) -> Optional[Signal]:
        if self._trading_halted or self._active_positions >= self.max_positions:
            return None
        if index - self._last_loss_index.get(symbol, -999) < self.cooldown_bars:
            return None
        if self._detector is None:
            return None

        # 周线过滤：下跌不做多；严格模式要求必须上涨
        weekly_trend = self._get_weekly_trend(hist_df)
        if weekly_trend == 'down':
            return None
        if self.require_weekly_up and weekly_trend != 'up':
            return None

        buy_point = self._detector.detect_latest_buy()
        if buy_point is None or buy_point.point_type not in self.allowed_buy_types:
            return None

        # 确定止损位
        if buy_point.point_type in ('2buy', 'quasi2buy'):
            # 2买/类2买: 止损 = 1买低点（如有）
            first_buy = next(
                (b for b in reversed(self._detector._buy_points) if b.point_type == '1buy'), None
            )
            stop_loss = first_buy.price if first_buy else buy_point.stop_loss
            if stop_loss <= 0:
                stop_loss = current_price * 0.90
        else:
            # 3买: 止损 = ZG（中枢上沿）
            stop_loss = buy_point.stop_loss if buy_point.stop_loss > 0 else current_price * 0.90

        stop_distance = current_price - stop_loss
        if stop_distance <= 0:
            return None

        # 止损距离过滤：拒绝过宽止损
        stop_pct = stop_distance / current_price
        if stop_pct > self.max_stop_pct:
            return None

        quantity = self._position_size(current_price, stop_distance)
        if quantity <= 0:
            return None

        logger.info(
            f"[{symbol}@{index}] {buy_point.point_type}入场: 价格={current_price:.2f}, "
            f"止损={stop_loss:.2f}({stop_distance/current_price*100:.1f}%), 仓位={quantity}股"
        )

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=hist_df.index[-1] if hasattr(hist_df, 'index') else pd.Timestamp.now(),
            price=current_price,
            quantity=quantity,
            reason=f'{buy_point.point_type}: {buy_point.reason}',
            confidence=buy_point.confidence,
            metadata={
                'entry_index': index,
                'stop_loss': stop_loss,
                'buy_point_type': buy_point.point_type,
            }
        )

    def _position_size(self, current_price: float, stop_distance: float) -> int:
        risk_amount = self.initial_capital * self.risk_per_trade
        shares = risk_amount / stop_distance
        quantity = int(shares // 100) * 100
        max_affordable = int(self.get_cash() * 0.95 / current_price)
        quantity = min(quantity, max_affordable)
        return max(quantity, 0)

    # ==================== 退出：只有3种 ====================

    def _check_exit(self, current_price: float, index: int, symbol: str, context: Dict[str, Any]) -> Optional[Signal]:
        position = self.get_position(symbol)
        if position <= 0:
            return None

        entry_price = self._entry_prices.get(symbol, current_price)
        stop_loss = self._stop_losses.get(symbol, entry_price * 0.90)
        highest = self._highest_prices.get(symbol, entry_price)
        entry_idx = self._entry_indices.get(symbol, index)

        if current_price > highest:
            self._highest_prices[symbol] = current_price
            highest = current_price

        profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        bars_held = index - entry_idx

        # === 1. 结构止损 ===
        if current_price <= stop_loss:
            return self._sell(symbol, position, current_price, index,
                f'结构止损(止损{stop_loss:.2f}被跌破, 亏损{profit_pct*100:.1f}%)')

        # === 2. 2卖信号（持仓足够久才检查）===
        if bars_held >= self.min_hold_before_sell and self._detector is not None:
            sell = self._detector.detect_latest_sell()
            if sell is not None and sell.point_type == '2sell':
                return self._sell(symbol, position, current_price, index,
                    f'日线2卖(盈利{profit_pct*100:.1f}%): {sell.reason}')

        # === 3. 时间止损 ===
        if bars_held >= self.time_stop_bars:
            return self._sell(symbol, position, current_price, index,
                f'时间止损(持仓{bars_held}根, 盈亏{profit_pct*100:.1f}%)')

        return None

    def _sell(self, symbol: str, qty: int, price: float, index: int, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.SELL, symbol=symbol,
            datetime=pd.Timestamp.now(), price=price, quantity=qty,
            reason=reason, confidence=1.0,
            metadata={'exit_index': index}
        )

    def _check_portfolio_risk(self, current_price: float, symbol: str) -> None:
        equity = self.get_equity({symbol: current_price})
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd >= self.max_drawdown_pct:
                self._trading_halted = True

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        super().on_order(signal, executed_price, executed_quantity)
        symbol = signal.symbol
        if signal.is_buy():
            self._entry_prices[symbol] = executed_price
            self._highest_prices[symbol] = executed_price
            self._active_positions += 1
            self._entry_indices[symbol] = signal.metadata.get('entry_index', 0)
            self._stop_losses[symbol] = signal.metadata.get('stop_loss', executed_price * 0.90)
        elif signal.is_sell():
            remaining = self.get_position(symbol) - executed_quantity
            if remaining <= 0:
                if executed_price < self._entry_prices.get(symbol, executed_price):
                    self._last_loss_index[symbol] = signal.metadata.get('exit_index', 0)
                for d in (self._entry_prices, self._highest_prices, self._entry_indices, self._stop_losses):
                    d.pop(symbol, None)
                self._active_positions = max(0, self._active_positions - 1)
