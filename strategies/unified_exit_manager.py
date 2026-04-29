"""
Unified Exit Manager

Priority-based exit system with dynamic take-profit:
1. Weekly trend reversal (emergency)
2. ChanLun structural stop
3. Fixed stop loss
4. ATR adaptive trailing stop (replaces fixed trailing)
5. Trend-adaptive partial profit (dynamic targets)
5.5. 30min 1卖减仓 (reduce 70%, tighten trailing for remainder)
6. Time stop
6.5. Structure-accelerated exit (NEW)
7. Signal reversal (ChanLun opposite signal)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .unified_config import ExitConfig


@dataclass
class ExitSignal:
    """Exit signal"""
    action: str          # 'sell', 'force_exit'
    quantity: int        # Quantity to sell
    reason: str          # Exit reason
    confidence: float    # Confidence
    exit_type: str       # Exit type identifier


@dataclass
class PositionRecord:
    """Position tracking record"""
    entry_price: float
    entry_index: int
    chan_stop_loss: float = 0.0
    fixed_stop_loss: float = 0.0
    highest_price: float = 0.0
    exit_stage: int = 0
    bars_held: int = 0
    buy_point_type: str = ''  # 买点类型: 1buy/2buy/3buy
    breakeven_raised: bool = False  # 止损是否已上移至保本
    sell_reduced: bool = False       # 30min 1卖是否已触发减仓
    sell_reduce_price: float = 0.0   # 1卖减仓时的价格


class UnifiedExitManager:
    """
    Unified exit manager with dynamic take-profit.

    Usage:
        mgr = UnifiedExitManager(config)
        mgr.on_buy(symbol, price, bar_index, chan_stop=...)
        signal = mgr.check_exit(symbol, price, qty, bar_index, ...)
    """

    def __init__(self, config: ExitConfig = None):
        self.config = config or ExitConfig()
        self._positions: Dict[str, PositionRecord] = {}

    def on_buy(
        self,
        symbol: str,
        price: float,
        bar_index: int,
        chan_stop: float = 0.0,
        buy_point_type: str = '',
    ):
        """Record buy event"""
        # 按买点类型选择止损比例
        if buy_point_type in ('3buy', 'quasi3buy') and self.config.fixed_stop_pct_3buy > 0:
            stop_pct = self.config.fixed_stop_pct_3buy
        else:
            stop_pct = self.config.fixed_stop_pct

        fixed_stop = price * (1 - stop_pct)

        # Use stricter stop
        actual_stop = fixed_stop
        if chan_stop > 0:
            actual_stop = max(chan_stop, fixed_stop)

        self._positions[symbol] = PositionRecord(
            entry_price=price,
            entry_index=bar_index,
            chan_stop_loss=chan_stop,
            fixed_stop_loss=actual_stop,
            highest_price=price,
            exit_stage=0,
            bars_held=0,
            buy_point_type=buy_point_type,
        )

    def on_sell(self, symbol: str):
        """Record sell event, clear position"""
        self._positions.pop(symbol, None)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def _get_dynamic_targets(self, trend_status: str) -> List[Tuple[float, float]]:
        """Get profit targets adapted to trend status"""
        if not self.config.use_dynamic_targets:
            return self.config.profit_targets

        if trend_status in ('STRONG_UP', 'strong_up'):
            return self.config.dynamic_targets_strong
        elif trend_status in ('WEAK_UP', 'weak_up', 'NEUTRAL', 'neutral'):
            return self.config.dynamic_targets_normal
        else:
            # Weak/down trend: use conservative targets
            return self.config.dynamic_targets_weak

    def check_exit(
        self,
        symbol: str,
        price: float,
        current_qty: int,
        bar_index: int,
        min_unit: int = 100,
        sell_signals: Optional[List] = None,
        weekly_bias: str = 'neutral',
        weekly_strength: float = 0.0,
        # Dynamic take-profit parameters
        atr_value: float = 0.0,
        trend_status: str = 'neutral',
        structure_warning: str = 'none',
        # 30min sell point for 1-sell reduce
        sell_point_30min: Optional[object] = None,
    ) -> Optional[ExitSignal]:
        """
        Check all exit conditions by priority.

        Args:
            symbol: Stock code
            price: Current price
            current_qty: Current position quantity
            bar_index: Current bar index
            min_unit: Minimum trade unit
            sell_signals: Daily sell point list
            weekly_bias: Weekly trend direction
            weekly_strength: Weekly trend strength
            atr_value: Current ATR value (for adaptive trailing)
            trend_status: Trend status (STRONG_UP/WEAK_UP/NEUTRAL/...)
            structure_warning: Structure warning (none/caution/danger)
            sell_point_30min: 30min sell point object (for 1-sell reduce)

        Returns:
            ExitSignal or None
        """
        record = self._positions.get(symbol)
        if not record:
            return None

        record.bars_held = bar_index - record.entry_index
        profit_pct = (price - record.entry_price) / record.entry_price

        # Update highest price
        if price > record.highest_price:
            record.highest_price = price

        # === 1. Emergency: Weekly trend reversal ===
        if weekly_bias == 'short' and weekly_strength > 0.7:
            return ExitSignal(
                action='force_exit',
                quantity=current_qty,
                reason=f'Weekly trend reversal to bearish (strength={weekly_strength:.2f})',
                confidence=1.0,
                exit_type='weekly_reversal',
            )

        # === 2. ChanLun stop loss ===
        if self.config.use_chanlun_stop and record.chan_stop_loss > 0:
            if price <= record.chan_stop_loss:
                return ExitSignal(
                    action='sell',
                    quantity=current_qty,
                    reason=f'ChanLun stop: below {record.chan_stop_loss:.2f} ({profit_pct:.2%})',
                    confidence=1.0,
                    exit_type='chanlun_stop',
                )

        # === 3. Fixed stop loss ===
        if self.config.use_fixed_stop and record.fixed_stop_loss > 0:
            if price <= record.fixed_stop_loss:
                return ExitSignal(
                    action='sell',
                    quantity=current_qty,
                    reason=f'Fixed stop: below {record.fixed_stop_loss:.2f} ({profit_pct:.2%})',
                    confidence=1.0,
                    exit_type='fixed_stop',
                )

        # === 3.5. 30min 1卖减仓 ===
        if self.config.use_1sell_reduce and not record.sell_reduced and sell_point_30min is not None:
            sp = sell_point_30min
            sp_type = getattr(sp, 'point_type', '')
            if sp_type in ('1sell',):
                reduce_pct = self.config.sell_reduce_pct
                exit_qty = int(current_qty * reduce_pct)
                exit_qty = (exit_qty // min_unit) * min_unit
                exit_qty = max(exit_qty, min_unit)
                if exit_qty > 0 and exit_qty < current_qty:
                    record.sell_reduced = True
                    record.sell_reduce_price = price
                    return ExitSignal(
                        action='sell',
                        quantity=exit_qty,
                        reason=f'30min 1卖减仓{reduce_pct:.0%}: {sp_type} conf={getattr(sp, "confidence", 0):.2f} at +{profit_pct:.2%}',
                        confidence=0.85,
                        exit_type='sell_1sell_reduce',
                    )

        # === 4. Trailing stop (ATR-adaptive or fixed) ===
        # 1卖减仓后始终启用紧跟踪(不受activation阈值限制)
        if record.sell_reduced and self.config.use_trailing_stop and self.config.sell_reduce_tight_tiers:
            max_p = record.highest_price
            for tg, trail in reversed(self.config.sell_reduce_tight_tiers):
                if (max_p - record.entry_price) / record.entry_price >= tg:
                    tp = record.entry_price * (1 + tg - trail)
                    if price <= tp:
                        return ExitSignal(
                            action='sell',
                            quantity=current_qty,
                            reason=f'Tight trailing after 1sell reduce: {tp:.2f} at +{profit_pct:.2%}',
                            confidence=0.9,
                            exit_type='trailing_stop',
                        )

        if self.config.use_trailing_stop and profit_pct > self.config.trailing_activation:
            if self.config.use_atr_trailing and atr_value > 0:
                # ATR-adaptive trailing: stop = highest - N * ATR
                # 强趋势中给更多空间（让利润跑更远）
                multiplier = self.config.atr_trailing_multiplier
                if trend_status in ('STRONG_UP', 'strong_up') and profit_pct > 0.15:
                    multiplier *= 1.5  # 强趋势且已有15%+利润，放宽50%
                atr_trailing_stop = record.highest_price - atr_value * multiplier
                if price <= atr_trailing_stop:
                    return ExitSignal(
                        action='sell',
                        quantity=current_qty,
                        reason=f'ATR trailing stop: highest {record.highest_price:.2f} - {self.config.atr_trailing_multiplier}xATR({atr_value:.2f}) = {atr_trailing_stop:.2f}',
                        confidence=0.9,
                        exit_type='trailing_stop',
                    )
            else:
                # Fallback: fixed percentage trailing
                trailing_stop = record.highest_price * (1 - self.config.trailing_offset)
                if price <= trailing_stop:
                    return ExitSignal(
                        action='sell',
                        quantity=current_qty,
                        reason=f'Fixed trailing stop: highest {record.highest_price:.2f} pulled back {profit_pct:.2%}',
                        confidence=0.9,
                        exit_type='trailing_stop',
                    )

        # === 5. Partial profit (trend-adaptive) ===
        if self.config.use_partial_profit:
            targets = self._get_dynamic_targets(trend_status)
            current_stage = record.exit_stage
            if current_stage < len(targets):
                target_pct, exit_ratio = targets[current_stage]
                if profit_pct >= target_pct:
                    exit_qty = int(current_qty * exit_ratio)
                    exit_qty = (exit_qty // min_unit) * min_unit
                    exit_qty = max(exit_qty, min_unit)

                    record.exit_stage = current_stage + 1

                    # 阶梯止损上移：每次部分止盈后，将止损提到已实现利润的一部分
                    # 公式：new_stop = entry * (1 + profit_pct * lock_ratio)
                    # lock_ratio: 锁定已实现利润的30%
                    lock_ratio = 0.30
                    new_stop = record.entry_price * (1 + profit_pct * lock_ratio)
                    record.fixed_stop_loss = max(record.fixed_stop_loss, new_stop)
                    record.breakeven_raised = True

                    trend_label = 'strong' if trend_status in ('STRONG_UP', 'strong_up') else \
                                  'weak' if trend_status in ('STRONG_DOWN', 'WEAK_DOWN', 'strong_down', 'weak_down') else 'normal'
                    return ExitSignal(
                        action='sell',
                        quantity=exit_qty,
                        reason=f'Partial profit ({current_stage + 1}, {trend_label}): +{profit_pct:.2%}',
                        confidence=0.8,
                        exit_type='partial_profit',
                    )

        # === 6. Time stop ===
        weak_time_stop = 3 if getattr(record, 'weak_market', False) else None
        time_bars = weak_time_stop or self.config.time_stop_bars
        if self.config.use_time_stop and record.bars_held >= time_bars:
            if profit_pct < 0.03 or weak_time_stop:
                return ExitSignal(
                    action='sell',
                    quantity=current_qty,
                    reason=f'Time stop: held {record.bars_held} bars'
                           f'{" (弱市3日强制)" if weak_time_stop else ""}'
                           f', profit {profit_pct:.2%}',
                    confidence=0.6,
                    exit_type='time_stop',
                )

        # === 6.5. Structure-accelerated exit ===
        if self.config.use_structure_exit and structure_warning == 'danger':
            # Danger: top fractal + MACD divergence detected, accelerate exit
            exit_qty = int(current_qty * self.config.structure_exit_ratio)
            exit_qty = (exit_qty // min_unit) * min_unit
            exit_qty = max(exit_qty, min_unit)
            if exit_qty > 0:
                return ExitSignal(
                    action='sell',
                    quantity=exit_qty,
                    reason=f'Structure exit (danger): top divergence at +{profit_pct:.2%}',
                    confidence=0.85,
                    exit_type='structure_exit',
                )
        elif self.config.use_structure_exit and structure_warning == 'caution':
            # Caution: trend track break, tighten trailing (already handled above)
            # Just log - the trailing stop will catch it
            pass

        # === 7. Signal reversal ===
        min_hold_bars = 5
        if record.bars_held < min_hold_bars:
            return None

        if sell_signals:
            best_sell = max(sell_signals, key=lambda s: s.confidence)
            type_thresholds = {
                '1sell': 0.60,
                '2sell': 0.70,
                '3sell': 0.80,
            }
            threshold = type_thresholds.get(best_sell.point_type, 0.75)

            if best_sell.confidence >= threshold:
                return ExitSignal(
                    action='sell',
                    quantity=current_qty,
                    reason=f'ChanLun {best_sell.point_type} signal reversal (conf={best_sell.confidence:.2f})',
                    confidence=best_sell.confidence,
                    exit_type='signal_reversal',
                )

        return None

    def get_stop_loss(self, symbol: str) -> float:
        """Get current stop loss level"""
        record = self._positions.get(symbol)
        if not record:
            return 0.0
        return record.fixed_stop_loss
