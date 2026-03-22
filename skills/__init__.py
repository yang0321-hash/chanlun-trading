"""
Skills - 可复用的能力单元
参考 Claude Code 的 Skill 设计模式

Skill = 单一职责的能力单元
- 明确的输入输出
- 无状态或弱状态
- 可复用、可测试、可组合
"""
from .base import BaseSkill, SkillResult

__all__ = [
    "BaseSkill",
    "SkillResult",
]
