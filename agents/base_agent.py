"""
Agent 基类 - 智能决策者
Agent = 使用 Skill 完成任务的决策者，负责流程编排和决策
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import pandas as pd

from skills.base import BaseSkill, SkillResult


@dataclass
class AgentState:
    """Agent 状态"""
    agent_name: str
    current_datetime: datetime
    symbols: List[str] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    signals_generated: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def update_analysis(self, symbol: str, key: str, value: Any) -> None:
        """更新分析结果"""
        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        self.analysis_results[symbol][key] = value

    def get_analysis(self, symbol: str, key: str = None, default: Any = None) -> Any:
        """获取分析结果"""
        if symbol not in self.analysis_results:
            return default
        if key is None:
            return self.analysis_results[symbol]
        return self.analysis_results[symbol].get(key, default)

    def add_signal(self, signal: Dict[str, Any]) -> None:
        """添加信号"""
        self.signals_generated.append(signal)

    def get_signals(self, symbol: str = None) -> List[Dict]:
        """获取信号"""
        if symbol is None:
            return self.signals_generated
        return [s for s in self.signals_generated if s.get('symbol') == symbol]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'agent_name': self.agent_name,
            'current_datetime': self.current_datetime.isoformat(),
            'symbols': self.symbols,
            'analysis_results': self.analysis_results,
            'signals_generated': self.signals_generated,
            'metrics': self.metrics,
        }


@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str
    skills: List[str] = field(default_factory=list)
    skill_configs: Dict[str, Dict] = field(default_factory=dict)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def get_skill_config(self, skill_name: str) -> Dict:
        """获取 Skill 配置"""
        return self.skill_configs.get(skill_name, {})

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'skills': self.skills,
            'skill_configs': self.skill_configs,
            'agent_config': self.agent_config,
            'description': self.description,
        }


class BaseAgent(ABC):
    """
    Agent 基类

    职责:
    1. 管理 Skills 的注册和调用
    2. 编排分析流程
    3. 做出交易决策
    4. 维护自身状态
    """

    def __init__(self, config: AgentConfig):
        """
        初始化 Agent

        Args:
            config: Agent 配置
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        self.skills: Dict[str, BaseSkill] = {}
        self.state = AgentState(
            agent_name=self.name,
            current_datetime=datetime.now()
        )

        # 从配置中获取参数
        for key, value in config.agent_config.items():
            setattr(self, key, value)

        # 注册 Skills
        self._register_skills()

    @abstractmethod
    def _register_skills(self) -> None:
        """注册所需的 Skills"""
        pass

    @abstractmethod
    def analyze(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分析流程

        Args:
            symbol: 股票代码
            context: 上下文数据 (K线、指标等)

        Returns:
            分析结果字典
        """
        pass

    def register_skill(self, name: str, skill: BaseSkill) -> None:
        """注册 Skill"""
        self.skills[name] = skill

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """获取 Skill"""
        return self.skills.get(name)

    def execute_skill(
        self,
        skill_name: str,
        **kwargs
    ) -> Optional[SkillResult]:
        """
        执行 Skill

        Args:
            skill_name: Skill 名称
            **kwargs: Skill 参数

        Returns:
            SkillResult 或 None (如果 Skill 不存在)
        """
        skill = self.get_skill(skill_name)
        if skill is None:
            return None

        return skill._execute_with_tracking(**kwargs)

    def update_state(self, **kwargs) -> None:
        """更新 Agent 状态"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            else:
                self.state.context[key] = value

    def reset_state(self) -> None:
        """重置 Agent 状态"""
        self.state = AgentState(
            agent_name=self.name,
            current_datetime=datetime.now()
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取 Agent 统计信息"""
        skill_stats = {}
        for name, skill in self.skills.items():
            skill_stats[name] = skill.get_stats()

        return {
            'name': self.name,
            'state': self.state.to_dict(),
            'skills': skill_stats,
            'total_signals': len(self.state.signals_generated),
        }

    def log(self, message: str, level: str = "info") -> None:
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.name}] {message}")


class AgentRegistry:
    """
    Agent 注册表
    管理所有可用的 Agent
    """

    _agents: Dict[str, type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str, agent_class: type[BaseAgent]) -> None:
        """注册 Agent"""
        cls._agents[name] = agent_class

    @classmethod
    def create(cls, name: str, config: Optional[AgentConfig] = None) -> Optional[BaseAgent]:
        """创建 Agent 实例"""
        agent_class = cls._agents.get(name)
        if agent_class is None:
            return None
        return agent_class(config or AgentConfig(name=name))

    @classmethod
    def list_agents(cls) -> List[str]:
        """列出所有已注册的 Agent"""
        return list(cls._agents.keys())


def register_agent(name: str):
    """
    Agent 注册装饰器

    使用示例:
    @register_agent('chanlun')
    class ChanLunAgent(BaseAgent):
        pass
    """
    def decorator(agent_class: type[BaseAgent]) -> type[BaseAgent]:
        AgentRegistry.register(name, agent_class)
        return agent_class
    return decorator


class AgentOrchestrator:
    """
    Agent 编排器
    管理多个 Agent 的协作
    """

    def __init__(self, agents: List[BaseAgent]):
        """
        初始化编排器

        Args:
            agents: Agent 列表
        """
        self.agents = agents
        self.results: Dict[str, Dict] = {}

    def analyze_all(
        self,
        symbol: str,
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        让所有 Agent 执行分析

        Args:
            symbol: 股票代码
            context: 上下文数据

        Returns:
            各 Agent 的分析结果
        """
        self.results = {}

        for agent in self.agents:
            try:
                result = agent.analyze(symbol, context)
                self.results[agent.name] = result
            except Exception as e:
                self.results[agent.name] = {
                    'success': False,
                    'error': str(e)
                }

        return self.results

    def get_consensus(self) -> Dict[str, Any]:
        """
        获取各 Agent 的共识

        Returns:
            共识结果
        """
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0

        for result in self.results.values():
            if not result.get('success'):
                continue

            signal = result.get('signal')
            if signal == 'buy':
                buy_votes += 1
            elif signal == 'sell':
                sell_votes += 1
            else:
                hold_votes += 1

        total = buy_votes + sell_votes + hold_votes
        if total == 0:
            return {'signal': 'hold', 'confidence': 0}

        # 多数表决
        if buy_votes > sell_votes and buy_votes > hold_votes:
            return {
                'signal': 'buy',
                'confidence': buy_votes / total,
                'votes': {'buy': buy_votes, 'sell': sell_votes, 'hold': hold_votes}
            }
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            return {
                'signal': 'sell',
                'confidence': sell_votes / total,
                'votes': {'buy': buy_votes, 'sell': sell_votes, 'hold': hold_votes}
            }
        else:
            return {
                'signal': 'hold',
                'confidence': hold_votes / total,
                'votes': {'buy': buy_votes, 'sell': sell_votes, 'hold': hold_votes}
            }

    def get_best_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取最佳信号（最高置信度）

        Args:
            symbol: 股票代码

        Returns:
            最佳信号
        """
        best_signal = None
        best_confidence = 0

        for agent_name, result in self.results.items():
            if not result.get('success'):
                continue

            signals = result.get('signals', [])
            if symbol in signals:
                signal = signals[symbol]
                if signal.get('confidence', 0) > best_confidence:
                    best_confidence = signal['confidence']
                    best_signal = {
                        **signal,
                        'source_agent': agent_name
                    }

        return best_signal
