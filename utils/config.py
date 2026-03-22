"""
配置管理模块
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """配置管理类"""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """加载配置文件"""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._default_config()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """默认配置"""
        return {
            'data': {
                'source': 'akshare',
                'storage_path': './data/storage',
                'default_period': 'daily',
                'fq': 'qfq'
            },
            'chanlun': {
                'fractal_confirm': True,
                'min_stroke_bars': 5,
                'min_segment_strokes': 3,
                'strict_mode': True
            },
            'backtest': {
                'initial_capital': 100000,
                'commission': 0.0003,
                'slippage': 0.0001,
                'min_unit': 100
            },
            'logging': {
                'level': 'INFO',
                'file': './logs/chanlun.log',
                'rotation': '100 MB'
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @property
    def data_source(self) -> str:
        return self.get('data.source', 'akshare')

    @property
    def storage_path(self) -> str:
        return self.get('data.storage_path', './data/storage')

    @property
    def default_period(self) -> str:
        return self.get('data.default_period', 'daily')

    @property
    def fq(self) -> str:
        return self.get('data.fq', 'qfq')

    @property
    def fractal_confirm(self) -> bool:
        return self.get('chanlun.fractal_confirm', True)

    @property
    def min_stroke_bars(self) -> int:
        return self.get('chanlun.min_stroke_bars', 5)

    @property
    def min_segment_strokes(self) -> int:
        return self.get('chanlun.min_segment_strokes', 3)

    @property
    def strict_mode(self) -> bool:
        return self.get('chanlun.strict_mode', True)

    @property
    def initial_capital(self) -> float:
        return self.get('backtest.initial_capital', 100000)

    @property
    def commission(self) -> float:
        return self.get('backtest.commission', 0.0003)

    @property
    def slippage(self) -> float:
        return self.get('backtest.slippage', 0.0001)

    @property
    def min_unit(self) -> int:
        return self.get('backtest.min_unit', 100)
