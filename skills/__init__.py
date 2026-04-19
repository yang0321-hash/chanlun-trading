"""
Skills - 可复用的能力单元
参考 Claude Code 的 Skill 设计模式

Skill = 单一职责的能力单元
- 明确的输入输出
- 无状态或弱状态
- 可复用、可测试、可组合

SKILL.md 支持:
- 与 Agent Skills 生态系统兼容
- 元数据驱动的技能发现
- YAML frontmatter + Markdown 格式
"""
from .base import (
    BaseSkill,
    SkillResult,
    SkillStatus,
    SkillRegistry,
    SkillChain,
    SkillPipeline,
    register_skill,
)

from .skill_loader import (
    SkillLoader,
    SkillMetadata,
    get_loader,
    list_skills,
    get_skill_metadata,
    print_skills,
)

# 子模块
from . import pattern
from . import signal
from . import risk
from . import workflow

__all__ = [
    # 基础
    "BaseSkill",
    "SkillResult",
    "SkillStatus",
    "SkillRegistry",
    "SkillChain",
    "SkillPipeline",
    "register_skill",
    # 技能加载器
    "SkillLoader",
    "SkillMetadata",
    "get_loader",
    "list_skills",
    "get_skill_metadata",
    "print_skills",
    # 模块
    "pattern",
    "signal",
    "risk",
    "workflow",
]
