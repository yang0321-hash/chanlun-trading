"""
缠论决策图
实现完整的缠论分析决策流程
"""
from typing import Any, Dict, List, Optional, Callable, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from agents.pattern_agent import PatternAgent, MultiTimeframePatternAgent
from agents.trend_agent import TrendAgent, TrendConfirmationAgent
from agents.signal_agent import SignalAgent, WeeklyDailySignalAgent, SignalFusionAgent
from agents.risk_agent import RiskAgent, PositionSizingAgent


class ChanLunGraphState(TypedDict):
    """缠论图状态"""
    # 输入数据
    symbol: str
    ohlcv_data: Optional[pd.DataFrame]
    current_index: int

    # 配置
    config: Dict[str, Any]

    # 各节点输出
    pattern_output: Optional[AgentOutput]
    trend_output: Optional[AgentOutput]
    signal_output: Optional[AgentOutput]
    risk_output: Optional[AgentOutput]
    position_output: Optional[AgentOutput]

    # 决策结果
    final_decision: Dict[str, Any]
    should_execute: bool

    # 元数据
    timestamp: datetime
    execution_log: List[str]


@dataclass
class GraphNode:
    """图节点"""
    name: str
    agent: BaseAgent
    handler: Optional[Callable] = None

    def execute(self, state: ChanLunGraphState) -> ChanLunGraphState:
        """执行节点逻辑"""
        if self.handler:
            return self.handler(state)

        # 默认执行 agent
        input_data = AgentInput(
            ohlcv_data=state['ohlcv_data'],
            current_index=state['current_index'],
            symbol=state['symbol'],
            config=state['config'],
            previous_results=self._get_previous_results(state)
        )

        output = self.agent.execute(input_data)
        state[f'{self.name}_output'] = output
        state['execution_log'].append(f"{self.name}: {output.reasoning}")

        return state

    def _get_previous_results(self, state: ChanLunGraphState) -> Dict[str, Any]:
        """获取前置节点的结果"""
        results = {}
        for key in state:
            if key.endswith('_output') and state[key] is not None:
                name = key.replace('_output', '')
                results[name] = state[key]
        return results


@dataclass
class GraphEdge:
    """图边"""
    from_node: str
    to_node: str
    condition: Optional[Callable[[ChanLunGraphState], bool]] = None


class ChanLunGraph:
    """
    缠论决策图
    编排各个智能体，实现完整的分析流程
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.start_node: Optional[str] = None
        self.end_node: Optional[str] = None

        # 初始化智能体
        self._init_agents()

    def _init_agents(self):
        """初始化所有智能体"""
        self.agents = {
            'pattern_agent': PatternAgent(self.config.get('pattern', {})),
            'multi_timeframe_agent': MultiTimeframePatternAgent(self.config.get('pattern', {})),
            'trend_agent': TrendAgent(self.config.get('trend', {})),
            'trend_confirm_agent': TrendConfirmationAgent(self.config.get('trend', {})),
            'signal_agent': SignalAgent(self.config.get('signal', {})),
            'weekly_daily_signal_agent': WeeklyDailySignalAgent(self.config.get('signal', {})),
            'signal_fusion_agent': SignalFusionAgent(self.config.get('signal', {})),
            'risk_agent': RiskAgent(self.config.get('risk', {})),
            'position_agent': PositionSizingAgent(self.config.get('position', {})),
        }

    def add_node(self, name: str, agent: BaseAgent,
                 handler: Optional[Callable] = None):
        """添加节点"""
        self.nodes[name] = GraphNode(name=name, agent=agent, handler=handler)

    def add_edge(self, from_node: str, to_node: str,
                 condition: Optional[Callable[[ChanLunGraphState], bool]] = None):
        """添加边"""
        self.edges.append(GraphEdge(from_node=from_node, to_node=to_node, condition=condition))

    def set_start(self, node_name: str):
        """设置起始节点"""
        self.start_node = node_name

    def set_end(self, node_name: str):
        """设置结束节点"""
        self.end_node = node_name

    def build_default_graph(self):
        """构建默认的缠论分析图"""
        # 添加节点
        self.add_node('pattern_analysis', self.agents['pattern_agent'])
        self.add_node('multi_timeframe', self.agents['multi_timeframe_agent'])
        self.add_node('trend_analysis', self.agents['trend_agent'])
        self.add_node('signal_generation', self.agents['signal_agent'])
        self.add_node('risk_assessment', self.agents['risk_agent'])
        self.add_node('position_sizing', self.agents['position_agent'])

        # 添加边（简单线性流程）
        self.add_edge('pattern_analysis', 'multi_timeframe')
        self.add_edge('multi_timeframe', 'trend_analysis')
        self.add_edge('trend_analysis', 'signal_generation')
        self.add_edge('signal_generation', 'risk_assessment')
        self.add_edge('risk_assessment', 'position_sizing')

        self.set_start('pattern_analysis')
        self.set_end('position_sizing')

    def execute(self, symbol: str, df: pd.DataFrame,
                current_index: int = -1) -> ChanLunGraphState:
        """
        执行图
        """
        if current_index == -1:
            current_index = len(df) - 1

        # 初始化状态
        state: ChanLunGraphState = {
            'symbol': symbol,
            'ohlcv_data': df,
            'current_index': current_index,
            'config': self.config,
            'pattern_output': None,
            'trend_output': None,
            'signal_output': None,
            'risk_output': None,
            'position_output': None,
            'final_decision': {},
            'should_execute': False,
            'timestamp': datetime.now(),
            'execution_log': [],
        }

        # 按照边的顺序执行
        current_node = self.start_node

        while current_node and current_node != self.end_node:
            if current_node not in self.nodes:
                break

            node = self.nodes[current_node]
            state = node.execute(state)

            # 找下一个节点
            next_node = None
            for edge in self.edges:
                if edge.from_node == current_node:
                    if edge.condition is None or edge.condition(state):
                        next_node = edge.to_node
                        break

            current_node = next_node

        # 执行结束节点
        if self.end_node and self.end_node in self.nodes:
            state = self.nodes[self.end_node].execute(state)

        # 生成最终决策
        state['final_decision'] = self._make_decision(state)

        return state

    def _make_decision(self, state: ChanLunGraphState) -> Dict[str, Any]:
        """生成最终决策"""
        decision = {
            'should_trade': False,
            'action': 'hold',
            'symbol': state['symbol'],
            'signal_type': None,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 0,
            'confidence': 0,
            'reasoning': '',
        }

        # 获取信号输出
        signal_output = state.get('signal_output')
        risk_output = state.get('risk_output')
        position_output = state.get('position_output')

        if not signal_output or not signal_output.success:
            decision['reasoning'] = '无有效交易信号'
            return decision

        signal_data = signal_output.data
        risk_data = risk_output.data if risk_output else {}
        position_data = position_output.data if position_output else {}

        # 检查风险等级
        risk_level = risk_data.get('risk_level', '中')
        if risk_level in ['高', '极高']:
            decision['reasoning'] = f'风险等级{risk_level}，暂不交易'
            return decision

        # 确定交易决策
        decision['signal_type'] = signal_data.get('signal_type')
        decision['entry_price'] = signal_data.get('entry_price')
        decision['stop_loss'] = signal_data.get('stop_loss', signal_data.get('entry_price', 0))
        decision['take_profit'] = signal_data.get('target_price', signal_data.get('entry_price', 0))
        decision['confidence'] = signal_data.get('confidence', 0)
        decision['position_size'] = position_data.get('shares', 0)
        decision['should_trade'] = decision['position_size'] > 0

        # 判断操作类型
        signal_type = signal_data.get('signal_type', '')
        if '买' in signal_type:
            decision['action'] = 'buy'
        elif '卖' in signal_type:
            decision['action'] = 'sell'
        else:
            decision['action'] = 'hold'

        # 组合理由
        reasons = []
        reasons.append(signal_output.reasoning)
        if risk_output:
            reasons.append(risk_output.reasoning)

        decision['reasoning'] = ' | '.join(reasons)

        return decision

    def visualize(self) -> str:
        """生成图的文本表示"""
        lines = ["缠论决策流程图:", ""]

        for edge in self.edges:
            lines.append(f"  {edge.from_node} -> {edge.to_node}")

        return "\n".join(lines)


def create_chanlun_graph(config: Optional[Dict[str, Any]] = None) -> ChanLunGraph:
    """
    创建缠论决策图的工厂函数
    """
    graph = ChanLunGraph(config)
    graph.build_default_graph()
    return graph
