"""
Kronos AI 预测模型集成模块

将 Kronos (shiyu-coder/Kronos, AAAI 2026) 金融K线基础模型
集成到缠论交易系统中，提供信号确认和选股增强。

用法:
    # 信号确认 (作为过滤器)
    from strategies.kronos import KronosConfig, KronosPredictor, KronosFilter

    config = KronosConfig(enabled=True)
    predictor = KronosPredictor(config)
    kronos_filter = KronosFilter(predictor)

    # 选股增强 (批量扫描)
    from strategies.kronos import KronosScreenerMixin

    mixin = KronosScreenerMixin(KronosConfig(enabled=True))
    ranked = mixin.rank_with_kronos(candidates, data_map)
"""

from .kronos_config import KronosConfig
from .kronos_predictor import KronosPredictor
from .kronos_filter import KronosFilter
from .kronos_screener_mixin import KronosScreenerMixin

__all__ = [
    'KronosConfig',
    'KronosPredictor',
    'KronosFilter',
    'KronosScreenerMixin',
]
