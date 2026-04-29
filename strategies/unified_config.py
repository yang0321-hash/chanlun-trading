"""
统一策略配置

所有参数集中在一个配置对象中，预设模式快速切换。
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TimeFrameConfig:
    """多周期配置"""
    use_weekly: bool = True          # 启用周线分析
    use_min30: bool = False          # 启用30分钟分析（需要分钟数据）
    min_bars_daily: int = 5          # 日线最小K线数
    min_bars_30m: int = 3            # 30分钟最小K线数


@dataclass
class ScoringThresholds:
    """评分阈值"""
    min_daily_confidence: float = 0.5    # 日线买点最低置信度
    min_buy_score: float = 0.5           # 评分器最低买入分
    min_sell_score: float = 0.4          # 评分器最低卖出分
    divergence_threshold: float = 0.3    # 背离阈值


@dataclass
class FilterConfig:
    """过滤器配置"""
    use_volume: bool = True              # 启用成交量过滤
    use_regime: bool = True              # 启用市场状态过滤
    use_cooldown: bool = True            # 启用冷却期
    use_trend_alignment: bool = True     # 启用趋势对齐过滤
    volume_min_ratio: float = 1.2        # 最低量比
    volume_ma_period: int = 20           # 量均线周期
    cooldown_bars: int = 10              # 冷却期K线数
    trend_ma_period: int = 20            # 趋势MA周期
    trend_require_macd_turn: bool = False # 是否要求MACD转正
    trend_strict_mode: bool = False      # 严格模式要求MA60也对齐

    # === 周线过滤 (30分钟策略) ===
    weekly_rise_min: float = 0.0         # 周线最低涨幅阈值 (0=不过滤, 0.20=20%)
    weekly_vol_mult: float = 1.0         # 周线量能倍数 (1.0=不加量能过滤)

    # === Kronos AI 预测确认 ===
    use_kronos: bool = False             # 启用 Kronos AI 预测确认
    kronos_model: str = 'mini'           # 模型: 'mini', 'small', 'base'
    kronos_pred_len: int = 5             # 预测K线根数
    kronos_min_upside: float = 0.01      # 买入确认最低预测涨幅 1%
    kronos_max_downside: float = 0.02    # 买入确认最大预测回撤 2%


@dataclass
class ExitConfig:
    """出场管理配置"""
    use_chanlun_stop: bool = True         # 缠论止损
    use_fixed_stop: bool = True          # 固定止损
    use_trailing_stop: bool = True       # 跟踪止损
    use_partial_profit: bool = True      # 分批止盈
    use_time_stop: bool = True           # 时间止损

    fixed_stop_pct: float = 0.05         # 固定止损比例 5%（1买/2买默认）
    fixed_stop_pct_3buy: float = 0.05   # 三买固定止损比例 5%（三买回踩确认，止损更紧）
    trailing_activation: float = 0.05    # 跟踪止损激活点 5%
    trailing_offset: float = 0.08        # 跟踪止损回撤 8%
    profit_targets: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.05, 0.3),   # 盈利5%卖30%
        (0.10, 0.3),   # 盈利10%卖30%
        (0.15, 0.4),   # 盈利15%卖剩余
    ])
    time_stop_bars: int = 60             # 时间止损K线数

    # === 动态止盈参数 ===
    use_atr_trailing: bool = True        # ATR自适应跟踪止损
    atr_trailing_multiplier: float = 3.0 # ATR跟踪止损倍数
    atr_period: int = 14                 # ATR计算周期

    use_dynamic_targets: bool = True     # 趋势自适应分批止盈
    # 趋势自适应止盈目标: (profit_pct, sell_ratio)
    dynamic_targets_strong: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.10, 0.10),  # 强趋势: 10%先卖10%（轻仓试探）
        (0.20, 0.15),  # 20%再卖15%
        (0.35, 0.25),  # 35%再卖25%
        # 剩余50%靠ATR跟踪止损（让利润跑更远）
    ])
    dynamic_targets_normal: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.05, 0.3),   # 正常: 5%卖30%
        (0.10, 0.3),   # 10%卖30%
        (0.15, 0.4),   # 15%卖剩余
    ])
    dynamic_targets_weak: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.03, 0.4),   # 弱势: 3%卖40%
        (0.06, 0.35),  # 6%卖35%
        (0.10, 0.25),  # 10%卖剩余
    ])

    use_structure_exit: bool = True      # 结构加速出场
    structure_exit_ratio: float = 0.5    # 结构出场卖出剩余仓位比例

    # === 30min 1卖减仓 (回测验证: PF 5.31, 胜率75%) ===
    use_1sell_reduce: bool = True           # 启用30min 1卖减仓
    sell_reduce_pct: float = 0.70           # 1卖触发时减仓比例 (70%)
    sell_reduce_tight_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.02, 0.008),  # 剩余仓位: 2%启动/0.8%回撤
        (0.04, 0.015),  # 4%启动/1.5%回撤
        (0.07, 0.030),  # 7%启动/3%回撤
    ])


@dataclass
class PositionConfig:
    """仓位管理配置"""
    max_position_pct: float = 0.25       # 单只最大仓位 25%
    min_unit: int = 100                  # 最小交易单位
    confirmed_ratio: float = 1.0         # 30分确认后仓位比例
    unconfirmed_ratio: float = 0.5       # 无30分确认仓位比例


@dataclass
class UnifiedStrategyConfig:
    """统一策略完整配置"""
    name: str = '统一缠论策略'
    timeframes: TimeFrameConfig = field(default_factory=TimeFrameConfig)
    scoring: ScoringThresholds = field(default_factory=ScoringThresholds)
    filters: FilterConfig = field(default_factory=FilterConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    position: PositionConfig = field(default_factory=PositionConfig)

    # 预设模式
    @classmethod
    def conservative(cls) -> 'UnifiedStrategyConfig':
        """保守模式：高门槛，严格止损"""
        config = cls(name='统一缠论策略(保守)')
        config.scoring.min_daily_confidence = 0.65
        config.scoring.min_buy_score = 0.6
        config.filters.volume_min_ratio = 1.5
        config.filters.trend_strict_mode = True  # 保守模式：要求MA60也对齐
        config.filters.kronos_min_upside = 0.02  # 保守模式：Kronos 需要 2% 预测涨幅
        config.exit.fixed_stop_pct = 0.04
        config.exit.trailing_activation = 0.04
        config.position.max_position_pct = 0.20
        config.position.unconfirmed_ratio = 0.3
        return config

    @classmethod
    def aggressive(cls) -> 'UnifiedStrategyConfig':
        """激进模式：低门槛，宽止损"""
        config = cls(name='统一缠论策略(激进)')
        config.scoring.min_daily_confidence = 0.4
        config.scoring.min_buy_score = 0.35
        config.filters.volume_min_ratio = 0.5   # 允许较低成交量（适合3买回踩）
        config.filters.use_regime = False
        config.filters.use_trend_alignment = False  # 激进模式不限制趋势
        config.filters.kronos_min_upside = 0.005  # 激进模式：Kronos 仅需 0.5% 预测涨幅
        config.exit.fixed_stop_pct = 0.08
        config.exit.trailing_activation = 0.07
        config.exit.trailing_offset = 0.12
        config.position.max_position_pct = 0.35
        config.position.unconfirmed_ratio = 0.7
        return config

    @classmethod
    def multi_tf(cls) -> 'UnifiedStrategyConfig':
        """多周期模式：启用30分钟"""
        config = cls(name='统一缠论策略(多周期)')
        config.timeframes.use_min30 = True
        config.timeframes.use_weekly = True
        return config

    @classmethod
    def single_daily(cls) -> 'UnifiedStrategyConfig':
        """单日线模式（等价IntegratedChanLunStrategy）"""
        config = cls(name='统一缠论策略(单日线)')
        config.timeframes.use_weekly = False
        config.timeframes.use_min30 = False
        config.filters.use_regime = False
        config.filters.volume_min_ratio = 0.8  # 放宽量能要求（比默认1.2宽松）
        config.position.unconfirmed_ratio = 1.0  # 无30分确认则全仓
        return config

    @classmethod
    def v3a_30min(cls, mode: str = 'trend') -> 'UnifiedStrategyConfig':
        """v3a 30分钟缠论策略"""
        config = cls(name=f'v3a 30分钟策略({mode})')
        config.timeframes.use_min30 = True
        config.timeframes.use_weekly = False
        config.exit.fixed_stop_pct = 0.12
        config.exit.trailing_activation = 0.05
        config.exit.trailing_offset = 0.03
        config.exit.use_atr_trailing = False
        config.exit.time_stop_bars = 80
        config.exit.profit_targets = [(0.03, 0.5)]  # 3%盈利卖半仓
        config.position.max_position_pct = 0.30
        return config

    @classmethod
    def min30_optimized(cls) -> 'UnifiedStrategyConfig':
        """30分钟优化策略 — 基于150只×144组参数网格搜索最优结果

        回测验证: 胜率62.6%, 2买合计胜率71%, 均盈亏+3.69%
        参数来源: signals/grid_search_30min.json
        """
        config = cls(name='30分钟优化策略(网格搜索)')
        config.timeframes.use_min30 = True
        config.timeframes.use_weekly = True
        # 过滤: 周线≥20%涨幅, 不加额外量能过滤
        config.filters.weekly_rise_min = 0.20
        config.filters.weekly_vol_mult = 1.0
        config.filters.use_regime = True
        # 出场: 优化后止盈参数
        config.exit.trailing_activation = 0.04   # 4%启动 (原6%)
        config.exit.trailing_offset = 0.03       # 3%回撤 (原4%)
        config.exit.time_stop_bars = 200          # 200根K线时间止损
        config.exit.use_atr_trailing = False      # 不用ATR跟踪
        config.exit.use_dynamic_targets = False   # 不用分批止盈, 纯跟踪
        config.exit.fixed_stop_pct = 0.05         # 固定止损5%
        # 仓位
        config.position.max_position_pct = 0.25
        config.position.unconfirmed_ratio = 0.5
        return config
