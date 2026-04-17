"""
信号过滤器链

插件式过滤器，可配置开关组合使用。
"""

from .base_filter import SignalFilter
from .volume_filter import VolumeFilter
from .regime_filter import RegimeFilter
from .cooldown_filter import CooldownFilter
from .trend_filter import TrendAlignmentFilter
from .composite_filter import CompositeFilter

# Kronos AI 预测确认过滤器 (可选，需要 torch/kronos 依赖)
try:
    from strategies.kronos import KronosFilter
except ImportError:
    KronosFilter = None

__all__ = [
    'SignalFilter',
    'VolumeFilter',
    'RegimeFilter',
    'CooldownFilter',
    'TrendAlignmentFilter',
    'CompositeFilter',
    'KronosFilter',
]
