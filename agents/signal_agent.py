"""
信号生成智能体
综合形态和趋势分析，生成交易信号
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd

from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.state import SignalType, TrendDirection


@dataclass
class SignalGenerationResult:
    """信号生成结果"""
    signal_type: SignalType = SignalType.NONE
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0

    position_size: int = 0  # 建议仓位（股数）
    position_ratio: float = 0.0  # 建议仓位比例

    confidence: float = 0.5
    reasoning: str = ""

    # 信号条件
    conditions_met: List[str] = field(default_factory=list)
    conditions_missed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "position_size": self.position_size,
            "position_ratio": self.position_ratio,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "conditions_met": self.conditions_met,
            "conditions_missed": self.conditions_missed,
        }


class SignalAgent(BaseAgent):
    """
    信号生成智能体
    根据形态和趋势生成交易信号
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("signal_agent", config)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.use_memory = self.config.get('use_memory', True)

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        综合分析并生成信号
        """
        # 获取前置智能体的结果
        pattern_result = input_data.previous_results.get('pattern_agent')
        trend_result = input_data.previous_results.get('trend_agent')

        if not pattern_result or not pattern_result.success:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning="缺少形态分析结果"
            )

        # 解析形态数据
        pattern_data = pattern_result.data
        trend_data = trend_result.data if trend_result else {}

        # 生成信号
        signal = self._generate_signal(
            pattern_data,
            trend_data,
            input_data
        )

        return AgentOutput(
            agent_name=self.name,
            success=signal.signal_type != SignalType.NONE,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            data=signal.to_dict()
        )

    def _generate_signal(self, pattern_data: Dict, trend_data: Dict,
                         input_data: AgentInput) -> SignalGenerationResult:
        """生成交易信号"""

        result = SignalGenerationResult()
        conditions_met = []
        conditions_missed = []

        # 获取当前价格
        df = input_data.ohlcv_data
        if df is None:
            return result
        current_price = df['close'].iloc[input_data.current_index]

        # 判断趋势方向
        trend_direction = TrendDirection(trend_data.get('direction', 'unknown'))

        # 检查是否有背驰
        divergence = pattern_data.get('divergence_detected', False)
        divergence_type = pattern_data.get('divergence_type', '')

        # 检查是否在中枢内
        in_pivot = pattern_data.get('in_pivot', False)
        current_pivot = pattern_data.get('current_pivot')

        # 1. 检测第一类买点（下跌+底背驰）
        if trend_direction == TrendDirection.DOWN and divergence and divergence_type == 'bullish':
            conditions_met.append("下跌趋势+底背驰")
            signal_type = SignalType.BUY_1
            entry_price = current_price
            stop_loss = current_pivot.get('low', current_price * 0.95) if current_pivot else current_price * 0.95

        # 2. 检测第二类买点（回抽不破前低）
        elif (trend_direction == TrendDirection.UP and
              pattern_data.get('current_fractal') and
              pattern_data['current_fractal'].get('type') == 'bottom'):
            conditions_met.append("上涨趋势+底分型确认")
            signal_type = SignalType.BUY_2
            entry_price = current_price
            stop_loss = pattern_data['current_fractal'].get('low', current_price * 0.98)

        # 3. 检测第三类买点（突破中枢回踩不破）
        elif (not in_pivot and
              current_pivot and
              current_price > current_pivot.get('high', current_price)):
            conditions_met.append("突破中枢上沿")
            signal_type = SignalType.BUY_3
            entry_price = current_price
            stop_loss = current_pivot.get('center', current_price * 0.97)

        # 4. 检测第一类卖点（上涨+顶背驰）
        elif trend_direction == TrendDirection.UP and divergence and divergence_type == 'bearish':
            conditions_met.append("上涨趋势+顶背驰")
            signal_type = SignalType.SELL_1
            entry_price = current_price
            stop_loss = current_pivot.get('high', current_price * 1.05) if current_pivot else current_price * 1.05

        # 5. 检测第二类卖点（反弹不破前高）
        elif (trend_direction == TrendDirection.DOWN and
              pattern_data.get('current_fractal') and
              pattern_data['current_fractal'].get('type') == 'top'):
            conditions_met.append("下跌趋势+顶分型确认")
            signal_type = SignalType.SELL_2
            entry_price = current_price
            stop_loss = pattern_data['current_fractal'].get('high', current_price * 1.02)

        # 6. 检测第三类卖点（跌破中枢反弹不破下沿）
        elif (not in_pivot and
              current_pivot and
              current_price < current_pivot.get('low', current_price)):
            conditions_met.append("跌破中枢下沿")
            signal_type = SignalType.SELL_3
            entry_price = current_price
            stop_loss = current_pivot.get('center', current_price * 1.03)

        else:
            conditions_missed.append("不满足任何买卖点条件")
            return SignalGenerationResult(
                signal_type=SignalType.NONE,
                confidence=0.0,
                reasoning="无交易信号",
                conditions_missed=conditions_missed
            )

        # 计算信心度
        pattern_confidence = pattern_data.get('confidence', 0.5)
        trend_confidence = trend_data.get('strength', 0.5)

        # 基础信心度
        base_confidence = (pattern_confidence + trend_confidence) / 2

        # 如果有记忆系统，根据历史调整
        if self.use_memory and hasattr(self, 'memory_manager'):
            pattern_id = f"{signal_type.value}_{input_data.symbol}"
            adjusted_confidence = self.memory_manager.get_adjusted_confidence(
                pattern_id, base_confidence
            )
        else:
            adjusted_confidence = base_confidence

        # 如果信心度不足，不生成信号
        if adjusted_confidence < self.min_confidence:
            conditions_missed.append(f"信心度{adjusted_confidence:.2f}低于阈值{self.min_confidence}")
            return SignalGenerationResult(
                signal_type=SignalType.NONE,
                confidence=adjusted_confidence,
                reasoning=f"信心度不足: {adjusted_confidence:.2f}",
                conditions_met=conditions_met,
                conditions_missed=conditions_missed
            )

        # 计算目标价格
        if signal_type.value.startswith("买"):
            target_price = entry_price * 1.15  # 15%目标
        else:
            target_price = entry_price * 0.85

        # 计算仓位
        position_ratio = min(0.95, adjusted_confidence * 1.5)  # 信心度越高仓位越大

        result = SignalGenerationResult(
            signal_type=signal_type,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            position_ratio=position_ratio,
            confidence=adjusted_confidence,
            reasoning=f"{'，'.join(conditions_met)}，信心度{adjusted_confidence:.2f}",
            conditions_met=conditions_met,
            conditions_missed=conditions_missed
        )

        return result

    def set_memory_manager(self, memory_manager):
        """设置记忆管理器"""
        self.memory_manager = memory_manager


class WeeklyDailySignalAgent(BaseAgent):
    """
    周线日线联合信号智能体
    实现周线日线级别的信号确认
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("weekly_daily_signal_agent", config)

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        周线日线联合分析
        """
        pattern_result = input_data.previous_results.get('multi_timeframe_pattern_agent')

        if not pattern_result or not pattern_result.success:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning="缺少多周期分析结果"
            )

        pattern_data = pattern_result.data

        # 检查周线和日线是否共振
        weekly = pattern_data.get('weekly', {})
        daily = pattern_data.get('daily', {})

        signal = self._check_resonance(weekly, daily, input_data)

        return AgentOutput(
            agent_name=self.name,
            success=signal['signal_type'] != SignalType.NONE,
            confidence=signal['confidence'],
            reasoning=signal['reasoning'],
            data=signal
        )

    def _check_resonance(self, weekly: Dict, daily: Dict,
                         input_data: AgentInput) -> Dict[str, Any]:
        """检查周线日线共振"""
        weekly_divergence = weekly.get('divergence_detected', False)
        daily_divergence = daily.get('divergence_detected', False)

        weekly_divergence_type = weekly.get('divergence_type', '')
        daily_divergence_type = daily.get('divergence_type', '')

        # 周线底背驰 + 日线底背驰 = 强买点
        if (weekly_divergence and weekly_divergence_type == 'bullish' and
            daily_divergence and daily_divergence_type == 'bullish'):
            return {
                'signal_type': SignalType.BUY_2,
                'confidence': 0.85,
                'reasoning': '周线日线双底背驰，强烈买入信号',
                'entry_signal': 'weekly_daily_bullish_divergence'
            }

        # 周线顶背驰 + 日线顶背驰 = 强卖点
        if (weekly_divergence and weekly_divergence_type == 'bearish' and
            daily_divergence and daily_divergence_type == 'bearish'):
            return {
                'signal_type': SignalType.SELL_2,
                'confidence': 0.85,
                'reasoning': '周线日线双顶背驰，强烈卖出信号',
                'entry_signal': 'weekly_daily_bearish_divergence'
            }

        # 周线看多 + 日线回调 = 2买
        if (not weekly_divergence and
            daily_divergence and daily_divergence_type == 'bullish'):
            return {
                'signal_type': SignalType.BUY_2,
                'confidence': 0.75,
                'reasoning': '周线上涨日线回调，2买机会',
                'entry_signal': 'weekly_up_daily_pullback'
            }

        return {
            'signal_type': SignalType.NONE,
            'confidence': 0.3,
            'reasoning': '周线日线未形成共振',
            'entry_signal': None
        }


class SignalFusionAgent(BaseAgent):
    """
    信号融合智能体
    融合多个信号源，生成最终交易决策
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("signal_fusion_agent", config)
        self.fusion_method = self.config.get('fusion_method', 'weighted_vote')

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        融合多个信号
        """
        signals = {}

        # 收集所有信号
        for key, value in input_data.previous_results.items():
            if 'signal' in key.lower() or hasattr(value, 'data'):
                data = value.data if isinstance(value.data, dict) else {}
                if 'signal_type' in data:
                    signals[key] = data

        if not signals:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning="无信号可融合"
            )

        # 融合信号
        return self._fuse_signals(signals)

    def _fuse_signals(self, signals: Dict[str, Dict]) -> AgentOutput:
        """融合多个信号"""
        if self.fusion_method == 'unanimous':
            return self._unanimous_fusion(signals)
        elif self.fusion_method == 'majority':
            return self._majority_fusion(signals)
        elif self.fusion_method == 'weighted':
            return self._weighted_fusion(signals)
        else:
            return self._weighted_fusion(signals)

    def _unanimous_fusion(self, signals: Dict) -> AgentOutput:
        """一致同意模式"""
        signal_types = [s.get('signal_type') for s in signals.values()]

        if len(set(signal_types)) == 1 and signal_types[0] != SignalType.NONE:
            # 所有信号一致
            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=sum(s.get('confidence', 0.5) for s in signals.values()) / len(signals),
                reasoning=f"所有信号一致: {signal_types[0]}",
                data={'fused_signal': signal_types[0]}
            )

        return AgentOutput(
            agent_name=self.name,
            success=False,
            confidence=0.0,
            reasoning="信号不一致",
            data={'fused_signal': SignalType.NONE}
        )

    def _majority_fusion(self, signals: Dict) -> AgentOutput:
        """多数表决模式"""
        buys = sum(1 for s in signals.values() if '买' in s.get('signal_type', ''))
        sells = sum(1 for s in signals.values() if '卖' in s.get('signal_type', ''))

        if buys > sells:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=0.7,
                reasoning=f"多数信号看多 ({buys}/{len(signals)})",
                data={'fused_signal': 'BUY', 'buy_votes': buys, 'sell_votes': sells}
            )
        elif sells > buys:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=0.7,
                reasoning=f"多数信号看空 ({sells}/{len(signals)})",
                data={'fused_signal': 'SELL', 'buy_votes': buys, 'sell_votes': sells}
            )

        return AgentOutput(
            agent_name=self.name,
            success=False,
            confidence=0.0,
            reasoning="买卖信号数量相等",
            data={'fused_signal': SignalType.NONE}
        )

    def _weighted_fusion(self, signals: Dict) -> AgentOutput:
        """加权融合模式"""
        buy_weight = 0
        sell_weight = 0

        for signal in signals.values():
            confidence = signal.get('confidence', 0.5)
            signal_type = signal.get('signal_type', '')

            if '买' in signal_type:
                buy_weight += confidence
            elif '卖' in signal_type:
                sell_weight += confidence

        if buy_weight > sell_weight:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=min(0.9, buy_weight / (buy_weight + sell_weight)),
                reasoning=f"加权看多 (买{buy_weight:.2f} vs 卖{sell_weight:.2f})",
                data={'fused_signal': 'BUY'}
            )
        elif sell_weight > buy_weight:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=min(0.9, sell_weight / (buy_weight + sell_weight)),
                reasoning=f"加权看空 (卖{sell_weight:.2f} vs 买{buy_weight:.2f})",
                data={'fused_signal': 'SELL'}
            )

        return AgentOutput(
            agent_name=self.name,
            success=False,
            confidence=0.0,
            reasoning="买卖权重相等",
            data={'fused_signal': SignalType.NONE}
        )
