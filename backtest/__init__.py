"""
回测系统模块
"""

from .engine import BacktestEngine, BacktestConfig, Trade
from .strategy import Strategy, Signal, SignalType
from .metrics import Metrics
from .report import ReportGenerator

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'Trade',
    'Strategy',
    'Signal',
    'SignalType',
    'Metrics',
    'ReportGenerator',
]
