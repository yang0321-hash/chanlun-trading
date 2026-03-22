"""
基础智能体类
定义所有智能体的通用接口和行为
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime
import pandas as pd


@dataclass
class AgentInput:
    """智能体输入"""
    # 数据
    ohlcv_data: Optional[pd.DataFrame] = None
    current_bar: Optional[Dict[str, Any]] = None
    current_index: int = 0

    # 上下文
    symbol: str = ""
    datetime: Optional[datetime] = None

    # 前置智能体的输出
    previous_results: Dict[str, Any] = field(default_factory=dict)

    # 配置
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """智能体输出"""
    agent_name: str
    success: bool = True
    confidence: float = 0.5
    reasoning: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseAgent(ABC):
    """
    基础智能体抽象类
    所有具体智能体都应继承此类
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._history: List[AgentOutput] = []

    @abstractmethod
    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        执行分析

        Args:
            input_data: 输入数据

        Returns:
            AgentOutput: 分析结果
        """
        pass

    def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        执行智能体逻辑，包含历史记录
        """
        output = self.analyze(input_data)
        output.agent_name = self.name
        self._history.append(output)
        return output

    def get_history(self, n: int = 10) -> List[AgentOutput]:
        """获取最近的历史输出"""
        return self._history[-n:]

    def clear_history(self):
        """清空历史"""
        self._history.clear()

    def update_config(self, config: Dict[str, Any]):
        """更新配置"""
        self.config.update(config)

    @property
    def confidence_level(self) -> float:
        """获取当前信心度（基于历史表现）"""
        if not self._history:
            return 0.5

        recent = self._history[-20:]  # 最近20次
        successful = sum(1 for h in recent if h.success)
        avg_confidence = sum(h.confidence for h in recent) / len(recent)

        # 成功率 * 平均信心度
        return (successful / len(recent)) * avg_confidence


class AgentChain:
    """
    智能体链
    按顺序执行多个智能体
    """

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def execute(self, input_data: AgentInput) -> Dict[str, AgentOutput]:
        """
        按顺序执行所有智能体
        每个智能体的输出会传递给下一个
        """
        results = {}
        current_input = input_data

        for agent in self.agents:
            output = agent.execute(current_input)
            results[agent.name] = output

            # 将当前输出添加到下一个智能体的输入中
            current_input = AgentInput(
                **{k: v for k, v in input_data.items() if k != 'previous_results'},
                previous_results={**current_input.previous_results, agent.name: output}
            )

        return results


class AgentEnsemble:
    """
    智能体集成
    多个智能体并行执行，结果融合
    """

    def __init__(self, agents: List[BaseAgent], fusion_method: str = "weighted_vote"):
        self.agents = agents
        self.fusion_method = fusion_method

    def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        并行执行所有智能体并融合结果
        """
        outputs = [agent.execute(input_data) for agent in self.agents]

        if self.fusion_method == "weighted_vote":
            return self._weighted_vote(outputs)
        elif self.fusion_method == "max_confidence":
            return self._max_confidence(outputs)
        elif self.fusion_method == "average":
            return self._average(outputs)
        else:
            return outputs[0]  # 默认返回第一个

    def _weighted_vote(self, outputs: List[AgentOutput]) -> AgentOutput:
        """加权投票"""
        total_weight = sum(o.confidence for o in outputs if o.success)
        if total_weight == 0:
            return AgentOutput(agent_name="ensemble", success=False, confidence=0.0)

        # 合并数据
        merged_data = {}
        for output in outputs:
            weight = output.confidence / total_weight
            for key, value in output.data.items():
                if isinstance(value, (int, float)):
                    merged_data[key] = merged_data.get(key, 0) + value * weight

        return AgentOutput(
            agent_name="ensemble",
            success=True,
            confidence=sum(o.confidence for o in outputs) / len(outputs),
            reasoning=f"融合了 {len(outputs)} 个智能体的结果",
            data=merged_data
        )

    def _max_confidence(self, outputs: List[AgentOutput]) -> AgentOutput:
        """取最高信心度"""
        return max(outputs, key=lambda x: x.confidence)

    def _average(self, outputs: List[AgentOutput]) -> AgentOutput:
        """平均"""
        successful = [o for o in outputs if o.success]
        if not successful:
            return AgentOutput(agent_name="ensemble", success=False)

        avg_confidence = sum(o.confidence for o in successful) / len(successful)
        return AgentOutput(
            agent_name="ensemble",
            success=True,
            confidence=avg_confidence,
            reasoning=f"平均了 {len(successful)} 个智能体的结果"
        )
