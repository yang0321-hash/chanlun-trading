"""
缠论系统配置管理
参考 Tauric Research 的配置设计，支持多策略、多参数配置
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class FractalConfig:
    """分型识别配置"""
    min_bars: int = 3
    require_confirmation: bool = True
    allow_inclusion: bool = True


@dataclass
class StrokeConfig:
    """笔生成配置"""
    min_fractal_distance: int = 3
    min_price_move: float = 0.01  # 1%
    strict_alternating: bool = True


@dataclass
class SegmentConfig:
    """线段配置"""
    min_stroke_count: int = 3
    break_confirm_bars: int = 1


@dataclass
class PivotConfig:
    """中枢配置"""
    min_overlap_strokes: int = 3
    extend_mode: bool = True


@dataclass
class StrategyConfig:
    """策略配置"""
    entry_confidence: float = 0.6
    exit_confidence: float = 0.5
    max_position_pct: float = 0.95  # 最大仓位比例
    stop_loss_pct: float = 0.05     # 止损比例
    take_profit_pct: float = 0.15   # 止盈比例


@dataclass
class ChanLunConfig:
    """
    缠论系统总配置
    所有参数可配置，便于实验和调优
    """
    # 识别参数
    fractal: FractalConfig = field(default_factory=FractalConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    pivot: PivotConfig = field(default_factory=PivotConfig)

    # 策略参数
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # 数据参数
    data_source: str = "akshare"  # akshare, tdx, yfinance
    use_adjusted: bool = True     # 使用复权数据

    # 回测参数
    initial_capital: float = 100000
    commission: float = 0.0003
    slippage: float = 0.0001

    # 输出参数
    verbose: bool = False
    save_plots: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "fractal": self.fractal.__dict__,
            "stroke": self.stroke.__dict__,
            "segment": self.segment.__dict__,
            "pivot": self.pivot.__dict__,
            "strategy": self.strategy.__dict__,
            "data_source": self.data_source,
            "use_adjusted": self.use_adjusted,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChanLunConfig":
        """从字典创建配置"""
        config = cls()

        if "fractal" in config_dict:
            config.fractal = FractalConfig(**config_dict["fractal"])
        if "stroke" in config_dict:
            config.stroke = StrokeConfig(**config_dict["stroke"])
        if "segment" in config_dict:
            config.segment = SegmentConfig(**config_dict["segment"])
        if "pivot" in config_dict:
            config.pivot = PivotConfig(**config_dict["pivot"])
        if "strategy" in config_dict:
            config.strategy = StrategyConfig(**config_dict["strategy"])

        return config

    @classmethod
    def get_preset(cls, preset_name: str) -> "ChanLunConfig":
        """获取预设配置"""
        presets = {
            "conservative": cls(  # 保守配置
                strategy=StrategyConfig(
                    entry_confidence=0.8,
                    exit_confidence=0.7,
                    max_position_pct=0.5,
                    stop_loss_pct=0.03,
                )
            ),
            "aggressive": cls(  # 激进配置
                strategy=StrategyConfig(
                    entry_confidence=0.5,
                    exit_confidence=0.4,
                    max_position_pct=0.95,
                    stop_loss_pct=0.08,
                )
            ),
            "balanced": cls(),  # 默认平衡配置
        }
        return presets.get(preset_name, cls())


# 预设配置快捷方式
CONSERVATIVE_CONFIG = ChanLunConfig.get_preset("conservative")
AGGRESSIVE_CONFIG = ChanLunConfig.get_preset("aggressive")
BALANCED_CONFIG = ChanLunConfig.get_preset("balanced")
