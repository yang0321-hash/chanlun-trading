import importlib.util
import os
import sys

_engine_path = os.path.join(os.path.dirname(__file__), '..', '..', 'chanlun_system', 'code', 'signal_engine.py')
_engine_path = os.path.abspath(_engine_path)

_spec = importlib.util.spec_from_file_location("chanlun_engine", _engine_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SignalEngine = _mod.SignalEngine
_engine = SignalEngine()

def generate(data_map):
    """backtest框架入口: data_map={code: DataFrame} -> {code: Series}"""
    return _engine.generate(data_map)
