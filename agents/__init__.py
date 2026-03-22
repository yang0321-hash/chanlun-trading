"""
缠论智能体系统
参考 Tauric Research 的多智能体设计，实现分工协作的AI分析系统
"""
from .base_agent import BaseAgent, AgentInput, AgentOutput
from .pattern_agent import PatternAgent, PatternRecognitionResult
from .trend_agent import TrendAgent, TrendAnalysisResult
from .signal_agent import SignalAgent, SignalGenerationResult
from .risk_agent import RiskAgent, RiskAssessmentResult

__all__ = [
    "BaseAgent",
    "AgentInput",
    "AgentOutput",
    "PatternAgent",
    "PatternRecognitionResult",
    "TrendAgent",
    "TrendAnalysisResult",
    "SignalAgent",
    "SignalGenerationResult",
    "RiskAgent",
    "RiskAssessmentResult",
]
