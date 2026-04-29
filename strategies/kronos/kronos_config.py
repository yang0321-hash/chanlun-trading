"""
Kronos AI 预测模型配置

Kronos 是首个金融K线基础模型 (shiyu-coder/Kronos, AAAI 2026)。
本模块定义集成所需的所有参数。
"""

from dataclasses import dataclass, field
from typing import Tuple


# 模型名称到 HuggingFace 路径的映射
MODEL_REGISTRY = {
    'mini': {
        'model': 'NeoQuasar/Kronos-mini',
        'tokenizer': 'NeoQuasar/Kronos-Tokenizer-2k',
        'max_context': 2048,
        'params': '4.1M',
    },
    'small': {
        'model': 'NeoQuasar/Kronos-small',
        'tokenizer': 'NeoQuasar/Kronos-Tokenizer-base',
        'max_context': 512,
        'params': '24.7M',
    },
    'base': {
        'model': 'NeoQuasar/Kronos-base',
        'tokenizer': 'NeoQuasar/Kronos-Tokenizer-base',
        'max_context': 512,
        'params': '102.3M',
    },
}


@dataclass
class KronosConfig:
    """Kronos 集成配置"""

    # === 开关 ===
    enabled: bool = False                      # 默认关闭，需主动开启

    # === 模型选择 ===
    model_name: str = 'mini'                   # 'mini', 'small', 'base'
    model_path: str = ''                       # 覆盖模型路径 (空=用 HuggingFace 默认)
    tokenizer_path: str = ''                   # 覆盖 tokenizer 路径

    # === 预测参数 ===
    pred_len: int = 5                          # 预测未来K线根数
    temperature: float = 1.0                   # 采样温度
    top_p: float = 0.9                         # 核采样阈值
    sample_count: int = 1                      # 预测采样次数 (多次取均值)

    # === 信号确认参数 (KronosFilter) ===
    min_upside_pct: float = 0.01               # 买入确认最低预测涨幅 1%
    max_downside_pct: float = 0.02             # 买入确认最大预测回撤 2%
    exempt_types: Tuple[str, ...] = ('1buy',)  # 豁免的信号类型 (1买是抄底，Kronos 可能不确认)

    # === 缓存参数 ===
    cache_size: int = 100                      # 最大缓存条目数
    cache_ttl_seconds: int = 300               # 缓存有效期 (秒，盘中5分钟)

    # === 选股增强参数 (KronosScreenerMixin) ===
    screener_top_n: int = 50                   # 选股增强只预测 top N 候选
    screener_pred_len: int = 10                # 选股用更长预测周期

    # === 设备管理 ===
    device: str = 'auto'                       # 'auto', 'cuda', 'cpu'

    @property
    def model_hub_path(self) -> str:
        """获取 HuggingFace 模型路径"""
        return self.model_path or MODEL_REGISTRY[self.model_name]['model']

    @property
    def tokenizer_hub_path(self) -> str:
        """获取 HuggingFace tokenizer 路径"""
        return self.tokenizer_path or MODEL_REGISTRY[self.model_name]['tokenizer']

    @property
    def max_context(self) -> int:
        """获取模型最大上下文长度"""
        return MODEL_REGISTRY[self.model_name]['max_context']
