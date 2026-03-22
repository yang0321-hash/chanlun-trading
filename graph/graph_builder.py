"""
图构建器
提供灵活的图构建API
"""
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .chanlun_graph import ChanLunGraph, GraphNode, ChanLunGraphState


@dataclass
class NodeConfig:
    """节点配置"""
    name: str
    agent_type: str  # agent类型名称
    config: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None
    enabled: bool = True


@dataclass
class EdgeConfig:
    """边配置"""
    from_node: str
    to_node: str
    condition: Optional[str] = None  # 条件表达式
    enabled: bool = True


class GraphBuilder:
    """
    图构建器
    提供流式API构建自定义决策图
    """

    def __init__(self, name: str = "chanlun_graph"):
        self.name = name
        self.node_configs: List[NodeConfig] = []
        self.edge_configs: List[EdgeConfig] = []
        self.config: Dict[str, Any] = {}
        self.start_node: Optional[str] = None
        self.end_node: Optional[str] = None

    def add_node(self, name: str, agent_type: str,
                 config: Optional[Dict[str, Any]] = None,
                 handler: Optional[Callable] = None) -> "GraphBuilder":
        """添加节点"""
        node_config = NodeConfig(
            name=name,
            agent_type=agent_type,
            config=config or {},
            handler=handler
        )
        self.node_configs.append(node_config)
        return self

    def add_edge(self, from_node: str, to_node: str,
                 condition: Optional[str] = None) -> "GraphBuilder":
        """添加边"""
        edge_config = EdgeConfig(
            from_node=from_node,
            to_node=to_node,
            condition=condition
        )
        self.edge_configs.append(edge_config)
        return self

    def set_start(self, node_name: str) -> "GraphBuilder":
        """设置起始节点"""
        self.start_node = node_name
        return self

    def set_end(self, node_name: str) -> "GraphBuilder":
        """设置结束节点"""
        self.end_node = node_name
        return self

    def set_config(self, config: Dict[str, Any]) -> "GraphBuilder":
        """设置全局配置"""
        self.config.update(config)
        return self

    def build(self) -> ChanLunGraph:
        """构建图"""
        graph = ChanLunGraph(self.config)

        # 添加节点
        for node_config in self.node_configs:
            if not node_config.enabled:
                continue

            agent = graph.agents.get(node_config.agent_type)
            if agent is None:
                raise ValueError(f"Unknown agent type: {node_config.agent_type}")

            # 更新agent配置
            if node_config.config:
                agent.update_config(node_config.config)

            graph.add_node(node_config.name, agent, node_config.handler)

        # 添加边
        for edge_config in self.edge_configs:
            if not edge_config.enabled:
                continue

            condition = None
            if edge_config.condition:
                condition = self._parse_condition(edge_config.condition)

            graph.add_edge(edge_config.from_node, edge_config.to_node, condition)

        # 设置起止节点
        if self.start_node:
            graph.set_start(self.start_node)
        if self.end_node:
            graph.set_end(self.end_node)

        return graph

    def _parse_condition(self, condition_str: str) -> Callable[[ChanLunGraphState], bool]:
        """解析条件表达式"""
        # 简化实现，支持几个常见条件
        def condition_func(state: ChanLunGraphState) -> bool:
            # 检查信号输出
            if condition_str == "has_signal":
                signal_output = state.get('signal_output')
                return signal_output is not None and signal_output.success

            # 检查风险等级
            if condition_str == "low_risk":
                risk_output = state.get('risk_output')
                if risk_output and risk_output.data:
                    return risk_output.data.get('risk_level') in ['低', '中']

            # 检查趋势方向
            if condition_str == "uptrend":
                trend_output = state.get('trend_output')
                if trend_output and trend_output.data:
                    return trend_output.data.get('direction') == 'up'

            if condition_str == "downtrend":
                trend_output = state.get('trend_output')
                if trend_output and trend_output.data:
                    return trend_output.data.get('direction') == 'down'

            # 默认返回True
            return True

        return condition_func


# 预定义的图模板
class GraphTemplates:
    """预定义图模板"""

    @staticmethod
    def basic_graph(config: Optional[Dict] = None) -> ChanLunGraph:
        """基础图 - 形态->趋势->信号"""
        builder = GraphBuilder("basic")
        builder.set_config(config or {})

        builder.add_node("pattern", "pattern_agent")
        builder.add_node("trend", "trend_agent")
        builder.add_node("signal", "signal_agent")

        builder.add_edge("pattern", "trend")
        builder.add_edge("trend", "signal")

        builder.set_start("pattern")
        builder.set_end("signal")

        return builder.build()

    @staticmethod
    def full_graph(config: Optional[Dict] = None) -> ChanLunGraph:
        """完整图 - 包含所有模块"""
        builder = GraphBuilder("full")
        builder.set_config(config or {})

        builder.add_node("pattern", "pattern_agent")
        builder.add_node("multi_tf", "multi_timeframe_agent")
        builder.add_node("trend", "trend_agent")
        builder.add_node("trend_confirm", "trend_confirm_agent")
        builder.add_node("signal", "signal_agent")
        builder.add_node("weekly_daily", "weekly_daily_signal_agent")
        builder.add_node("signal_fusion", "signal_fusion_agent")
        builder.add_node("risk", "risk_agent")
        builder.add_node("position", "position_agent")

        builder.add_edge("pattern", "multi_tf")
        builder.add_edge("multi_tf", "trend")
        builder.add_edge("trend", "trend_confirm")
        builder.add_edge("trend_confirm", "signal")
        builder.add_edge("signal", "weekly_daily")
        builder.add_edge("weekly_daily", "signal_fusion")
        builder.add_edge("signal_fusion", "risk")
        builder.add_edge("risk", "position")

        builder.set_start("pattern")
        builder.set_end("position")

        return builder.build()

    @staticmethod
    def weekly_daily_graph(config: Optional[Dict] = None) -> ChanLunGraph:
        """周线日线联合图"""
        builder = GraphBuilder("weekly_daily")
        builder.set_config(config or {})

        builder.add_node("multi_tf", "multi_timeframe_agent")
        builder.add_node("weekly_daily_signal", "weekly_daily_signal_agent")
        builder.add_node("risk", "risk_agent")
        builder.add_node("position", "position_agent")

        builder.add_edge("multi_tf", "weekly_daily_signal")
        builder.add_edge("weekly_daily_signal", "risk", condition="has_signal")
        builder.add_edge("risk", "position")

        builder.set_start("multi_tf")
        builder.set_end("position")

        return builder.build()

    @staticmethod
    def conservative_graph(config: Optional[Dict] = None) -> ChanLunGraph:
        """保守策略图 - 严格风险控制"""
        builder = GraphBuilder("conservative")
        builder.set_config(config or {})

        # 更严格的配置
        full_config = (config or {}).copy()
        full_config.setdefault('risk', {}).update({
            'max_position_pct': 0.5,
            'default_stop_loss': 0.03,
        })
        full_config.setdefault('signal', {}).update({
            'min_confidence': 0.7,
        })

        builder.set_config(full_config)

        builder.add_node("pattern", "pattern_agent")
        builder.add_node("trend", "trend_agent")
        builder.add_node("signal", "signal_agent")
        builder.add_node("risk", "risk_agent")
        builder.add_node("position", "position_agent")

        builder.add_edge("pattern", "trend")
        builder.add_edge("trend", "signal")
        builder.add_edge("signal", "risk")
        builder.add_edge("risk", "position", condition="low_risk")

        builder.set_start("pattern")
        builder.set_end("position")

        return builder.build()

    @staticmethod
    def aggressive_graph(config: Optional[Dict] = None) -> ChanLunGraph:
        """激进策略图 - 追求高收益"""
        builder = GraphBuilder("aggressive")
        builder.set_config(config or {})

        # 更激进的配置
        full_config = (config or {}).copy()
        full_config.setdefault('risk', {}).update({
            'max_position_pct': 0.95,
            'default_stop_loss': 0.08,
            'default_take_profit': 0.25,
        })
        full_config.setdefault('signal', {}).update({
            'min_confidence': 0.5,
        })

        builder.set_config(full_config)

        builder.add_node("pattern", "pattern_agent")
        builder.add_node("trend", "trend_agent")
        builder.add_node("signal", "signal_agent")
        builder.add_node("position", "position_agent")

        builder.add_edge("pattern", "trend")
        builder.add_edge("trend", "signal")
        builder.add_edge("signal", "position")

        builder.set_start("pattern")
        builder.set_end("position")

        return builder.build()
