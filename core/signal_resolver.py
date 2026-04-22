"""
跨周期信号融合解析器

将周线方向、日线信号、30分钟确认融合为最终交易决策。

解析规则：
1. 周线偏空 → 忽略日线买信号（资金保护）
2. 日线买点 + 30分确认 → 全仓入场 (position_ratio=1.0)
3. 日线买点无30分确认 → 半仓入场 (position_ratio=0.5)
4. 日线卖信号 → 无条件执行（资金保护优先）
5. 周线趋势反转 → 强制退出
"""

from dataclasses import dataclass
from typing import Optional
from loguru import logger

from .buy_sell_points import BuySellPoint
from .multi_tf_analyzer import MultiTimeFrameAnalyzer, TimeFrameAnalysis
from .interval_entry import IntervalEntry


@dataclass
class ResolvedSignal:
    """最终融合信号"""
    direction: str                     # 'long', 'short', 'flat'
    action: str                        # 'buy', 'sell', 'hold', 'force_exit'
    entry_type: str                    # '1buy', '2buy', '3buy', '1sell', '2sell', '3sell', 'force_exit'
    price: float                       # 建议价格
    stop_loss: float                   # 止损价
    confidence: float                  # 综合置信度 0-1
    position_ratio: float              # 仓位比例 0-1 (1.0=全仓, 0.5=半仓)

    # 多周期信息
    weekly_bias: str                   # 周线方向
    weekly_strength: float             # 周线方向强度
    daily_signal: Optional[BuySellPoint] = None   # 日线信号
    min30_confirmation: Optional[IntervalEntry] = None  # 30分钟确认
    min30_confirmed: bool = False      # 30分钟是否确认

    reason: str = ''                   # 决策原因


class SignalResolver:
    """
    跨周期信号融合解析器

    使用方法：
        resolver = SignalResolver(analyzer)
        signal = resolver.resolve()
        if signal and signal.action == 'buy':
            print(f"买入: {signal.price}, 仓位: {signal.position_ratio}")
    """

    def __init__(
        self,
        analyzer: MultiTimeFrameAnalyzer,
        current_price: float = 0.0,
        current_position: int = 0,
        # 可选阈值
        min_daily_confidence: float = 0.5,
        min_30m_confirmed_ratio: float = 1.0,
        unconfirmed_ratio: float = 0.5,
    ):
        """
        Args:
            analyzer: 多周期分析器
            current_price: 当前价格
            current_position: 当前持仓量
            min_daily_confidence: 日线信号最低置信度
            min_30m_confirmed_ratio: 30分确认后的仓位比例
            unconfirmed_ratio: 无30分确认时的仓位比例
        """
        self.analyzer = analyzer
        self.current_price = current_price
        self.current_position = current_position
        self.min_daily_confidence = min_daily_confidence
        self.min_30m_confirmed_ratio = min_30m_confirmed_ratio
        self.unconfirmed_ratio = unconfirmed_ratio

    def resolve_buy(self) -> Optional[ResolvedSignal]:
        """
        解析买入信号

        Returns:
            ResolvedSignal 或 None
        """
        # 1. 获取周线方向
        weekly_bias, weekly_strength = self.analyzer.get_strategic_bias()

        # 2. 周线偏空 → 阻止买入
        if weekly_bias == 'short' and weekly_strength > 0.6:
            return ResolvedSignal(
                direction='flat',
                action='hold',
                entry_type='',
                price=self.current_price,
                stop_loss=0,
                confidence=0,
                position_ratio=0,
                weekly_bias=weekly_bias,
                weekly_strength=weekly_strength,
                reason=f'周线偏空(bias={weekly_bias}, strength={weekly_strength:.2f}), 禁止买入'
            )

        # 3. 获取日线买点
        buy_signals = self.analyzer.get_operational_signals()
        if not buy_signals:
            return None

        # 取置信度最高的买点
        best_buy = max(buy_signals, key=lambda b: b.confidence)
        if best_buy.confidence < self.min_daily_confidence:
            return None

        # 4. 30分钟精确入场确认
        min30_entry = self.analyzer.get_entry_timing(best_buy)
        min30_confirmed = (min30_entry is not None and min30_entry.confirmed)

        # 5. 决定仓位比例
        if min30_confirmed:
            position_ratio = self.min_30m_confirmed_ratio
        elif self.analyzer.has_min30():
            # 有30分钟数据但未确认 → 半仓
            position_ratio = self.unconfirmed_ratio
        else:
            # 无30分钟数据 → 使用日线信号全仓
            position_ratio = self.min_30m_confirmed_ratio

        # 6. 综合置信度
        base_conf = best_buy.confidence
        if min30_confirmed and min30_entry:
            confidence = min(1.0, base_conf * 0.6 + min30_entry.confidence * 0.4)
        else:
            confidence = base_conf

        # 周线同向加成
        if weekly_bias == 'long' and weekly_strength > 0.5:
            confidence = min(1.0, confidence + 0.1)

        # 7. 止损取最严格
        stop_loss = best_buy.stop_loss
        if min30_confirmed and min30_entry and min30_entry.stop_loss > stop_loss:
            stop_loss = min30_entry.stop_loss

        reason_parts = [
            f'日线{best_buy.point_type}(conf={best_buy.confidence:.2f})',
        ]
        if min30_confirmed:
            reason_parts.append(f'30分确认({min30_entry.confirm_type})')
        else:
            reason_parts.append('无30分确认')
        reason_parts.append(f'周线={weekly_bias}({weekly_strength:.2f})')

        return ResolvedSignal(
            direction='long',
            action='buy',
            entry_type=best_buy.point_type,
            price=self.current_price,
            stop_loss=stop_loss,
            confidence=confidence,
            position_ratio=position_ratio,
            weekly_bias=weekly_bias,
            weekly_strength=weekly_strength,
            daily_signal=best_buy,
            min30_confirmation=min30_entry,
            min30_confirmed=min30_confirmed,
            reason=' | '.join(reason_parts),
        )

    def resolve_sell(self) -> Optional[ResolvedSignal]:
        """
        解析卖出信号

        卖出无需30分钟确认（资金保护优先）
        """
        # 1. 周线趋势反转 → 强制退出
        weekly_bias, weekly_strength = self.analyzer.get_strategic_bias()
        if weekly_bias == 'short' and weekly_strength > 0.7:
            return ResolvedSignal(
                direction='flat',
                action='force_exit',
                entry_type='force_exit',
                price=self.current_price,
                stop_loss=0,
                confidence=1.0,
                position_ratio=1.0,
                weekly_bias=weekly_bias,
                weekly_strength=weekly_strength,
                reason=f'周线趋势反转为下跌(strength={weekly_strength:.2f}), 强制退出'
            )

        # 2. 日线卖点
        sell_signals = self.analyzer.get_sell_signals()
        if not sell_signals:
            return None

        best_sell = max(sell_signals, key=lambda s: s.confidence)

        return ResolvedSignal(
            direction='flat',
            action='sell',
            entry_type=best_sell.point_type,
            price=self.current_price,
            stop_loss=0,
            confidence=best_sell.confidence,
            position_ratio=1.0,  # 卖出全仓
            weekly_bias=weekly_bias,
            weekly_strength=weekly_strength,
            daily_signal=best_sell,
            reason=f'日线{best_sell.point_type}(conf={best_sell.confidence:.2f})'
        )
