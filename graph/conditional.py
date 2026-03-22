"""
条件边
实现基于状态的条件路由
"""
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass

from .chanlun_graph import ChanLunGraphState


class ConditionalEdge:
    """
    条件边
    根据状态决定下一步走向
    """

    def __init__(self, name: str):
        self.name = name
        self.conditions: List[Dict[str, Any]] = []

    def add_condition(self, condition: Callable[[ChanLunGraphState], bool],
                     target_node: str, priority: int = 0) -> "ConditionalEdge":
        """添加条件"""
        self.conditions.append({
            'condition': condition,
            'target': target_node,
            'priority': priority
        })
        # 按优先级排序
        self.conditions.sort(key=lambda x: x['priority'], reverse=True)
        return self

    def route(self, state: ChanLunGraphState) -> Optional[str]:
        """根据状态路由"""
        for cond in self.conditions:
            if cond['condition'](state):
                return cond['target']
        return None

    @staticmethod
    def has_signal(state: ChanLunGraphState) -> bool:
        """检查是否有信号"""
        signal_output = state.get('signal_output')
        return signal_output is not None and signal_output.success

    @staticmethod
    def is_low_risk(state: ChanLunGraphState) -> bool:
        """检查是否低风险"""
        risk_output = state.get('risk_output')
        if risk_output and risk_output.data:
            level = risk_output.data.get('risk_level', '')
            return level in ['低', '中']
        return False

    @staticmethod
    def is_uptrend(state: ChanLunGraphState) -> bool:
        """检查是否上涨趋势"""
        trend_output = state.get('trend_output')
        if trend_output and trend_output.data:
            return trend_output.data.get('direction') == 'up'
        return False

    @staticmethod
    def is_downtrend(state: ChanLunGraphState) -> bool:
        """检查是否下跌趋势"""
        trend_output = state.get('trend_output')
        if trend_output and trend_output.data:
            return trend_output.data.get('direction') == 'down'
        return False

    @staticmethod
    def confidence_above(threshold: float) -> Callable[[ChanLunGraphState], bool]:
        """信心度高于阈值"""
        def check(state: ChanLunGraphState) -> bool:
            signal_output = state.get('signal_output')
            if signal_output:
                return signal_output.data.get('confidence', 0) >= threshold
            return False
        return check

    @staticmethod
    def has_divergence(divergence_type: str) -> Callable[[ChanLunGraphState], bool]:
        """检查是否有特定类型的背驰"""
        def check(state: ChanLunGraphState) -> bool:
            pattern_output = state.get('pattern_output')
            if pattern_output and pattern_output.data:
                return (pattern_output.data.get('divergence_detected', False) and
                       pattern_output.data.get('divergence_type') == divergence_type)
            return False
        return check

    @staticmethod
    def in_pivot(state: ChanLunGraphState) -> bool:
        """检查是否在中枢内"""
        pattern_output = state.get('pattern_output')
        if pattern_output and pattern_output.data:
            return pattern_output.data.get('in_pivot', False)
        return False

    @staticmethod
    def signal_is_buy(state: ChanLunGraphState) -> bool:
        """检查是否买入信号"""
        signal_output = state.get('signal_output')
        if signal_output and signal_output.data:
            signal_type = signal_output.data.get('signal_type', '')
            return '买' in signal_type
        return False

    @staticmethod
    def signal_is_sell(state: ChanLunGraphState) -> bool:
        """检查是否卖出信号"""
        signal_output = state.get('signal_output')
        if signal_output and signal_output.data:
            signal_type = signal_output.data.get('signal_type', '')
            return '卖' in signal_type
        return False


# 预定义条件组合
class Conditions:
    """常用条件组合"""

    @staticmethod
    def bullish_setup() -> Callable[[ChanLunGraphState], bool]:
        """看多设置"""
        def check(state: ChanLunGraphState) -> bool:
            return (ConditionalEdge.is_uptrend(state) and
                   ConditionalEdge.has_divergence('bullish')(state))
        return check

    @staticmethod
    def bearish_setup() -> Callable[[ChanLunGraphState], bool]:
        """看空设置"""
        def check(state: ChanLunGraphState) -> bool:
            return (ConditionalEdge.is_downtrend(state) and
                   ConditionalEdge.has_divergence('bearish')(state))
        return check

    @staticmethod
    def buy_signal_safe() -> Callable[[ChanLunGraphState], bool]:
        """安全的买入信号"""
        def check(state: ChanLunGraphState) -> bool:
            return (ConditionalEdge.signal_is_buy(state) and
                   ConditionalEdge.is_low_risk(state) and
                   ConditionalEdge.confidence_above(0.6)(state))
        return check

    @staticmethod
    def sell_signal_safe() -> Callable[[ChanLunGraphState], bool]:
        """安全的卖出信号"""
        def check(state: ChanLunGraphState) -> bool:
            return (ConditionalEdge.signal_is_sell(state) and
                   ConditionalEdge.confidence_above(0.6)(state))
        return check
