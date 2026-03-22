"""
Agents - 智能决策者
Agent = 使用 Skill 完成任务的决策者
"""
# 新架构 (Agent + Skill)
from .base_agent import (
    BaseAgent,
    AgentConfig,
    AgentState,
    AgentRegistry,
    register_agent,
    AgentOrchestrator,
)
from .chanlun_agent import ChanLunAgent, MultiLevelAgent

# 旧架构 (兼容性保留)
try:
    from .pattern_agent import PatternAgent, PatternRecognitionResult
    PATTERN_AGENT_AVAILABLE = True
except ImportError:
    PatternAgent = None
    PatternRecognitionResult = None
    PATTERN_AGENT_AVAILABLE = False

try:
    from .trend_agent import TrendAgent, TrendAnalysisResult
    TREND_AGENT_AVAILABLE = True
except ImportError:
    TrendAgent = None
    TrendAnalysisResult = None
    TREND_AGENT_AVAILABLE = False

try:
    from .signal_agent import SignalAgent, SignalGenerationResult
    SIGNAL_AGENT_AVAILABLE = True
except ImportError:
    SignalAgent = None
    SignalGenerationResult = None
    SIGNAL_AGENT_AVAILABLE = False

try:
    from .risk_agent import RiskAgent, RiskAssessmentResult
    RISK_AGENT_AVAILABLE = True
except ImportError:
    RiskAgent = None
    RiskAssessmentResult = None
    RISK_AGENT_AVAILABLE = False


__all__ = [
    # 新架构
    "BaseAgent",
    "AgentConfig",
    "AgentState",
    "AgentRegistry",
    "register_agent",
    "AgentOrchestrator",
    "ChanLunAgent",
    "MultiLevelAgent",
]

# 如果旧架构可用，也导出
if PATTERN_AGENT_AVAILABLE:
    __all__.extend(["PatternAgent", "PatternRecognitionResult"])
if TREND_AGENT_AVAILABLE:
    __all__.extend(["TrendAgent", "TrendAnalysisResult"])
if SIGNAL_AGENT_AVAILABLE:
    __all__.extend(["SignalAgent", "SignalGenerationResult"])
if RISK_AGENT_AVAILABLE:
    __all__.extend(["RiskAgent", "RiskAssessmentResult"])
