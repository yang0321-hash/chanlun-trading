"""
策略模块

活跃策略：
- UnifiedChanLunStrategy: 统一缠论策略（推荐，整合信号评分+多周期+过滤）
- IntegratedChanLunStrategy: 整合缠论策略（原有推荐）
- ChanLunStrategy: 基础缠论策略
- WeeklyDailyChanLunStrategy: 周线+日线策略

已归档策略在 deprecated/ 目录中。
"""

from .chan_strategy import ChanLunStrategy
from .weekly_daily_strategy import WeeklyDailyChanLunStrategy
from .integrated_chanlun_strategy import IntegratedChanLunStrategy
from .unified_chanlun_strategy import UnifiedChanLunStrategy
from .unified_config import UnifiedStrategyConfig
from .stable_chanlun_strategy import StableChanLunStrategy

__all__ = [
    'ChanLunStrategy',
    'WeeklyDailyChanLunStrategy',
    'IntegratedChanLunStrategy',
    'UnifiedChanLunStrategy',
    'UnifiedStrategyConfig',
    'StableChanLunStrategy',
]
