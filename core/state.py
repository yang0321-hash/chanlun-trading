"""
缠论状态管理
参考 Tauric Research 的 AgentState 设计，统一管理缠论分析状态
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Any
from enum import Enum


class TrendDirection(Enum):
    """趋势方向"""
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


class SignalType(Enum):
    """信号类型"""
    BUY_1 = "1买"      # 第一类买点
    BUY_2 = "2买"      # 第二类买点
    BUY_3 = "3买"      # 第三类买点
    SELL_1 = "1卖"     # 第一类卖点
    SELL_2 = "2卖"     # 第二类卖点
    SELL_3 = "3卖"     # 第三类卖点
    HOLD = "持有"
    NONE = "无信号"


@dataclass
class FractalInfo:
    """分型信息"""
    index: int
    fractal_type: str  # "top" or "bottom"
    high: float
    low: float
    confirmed: bool = False


@dataclass
class StrokeInfo:
    """笔信息"""
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    direction: str  # "up" or "down"


@dataclass
class SignalInfo:
    """交易信号信息"""
    signal_type: SignalType
    datetime: datetime
    price: float
    confidence: float = 0.5
    reason: str = ""
    pattern_id: str = ""


@dataclass
class ChanLunState:
    """
    缠论分析状态
    类似 Tauric 的 AgentState，作为各模块间传递的统一状态对象
    """
    # 基本信息
    symbol: str = ""
    current_datetime: Optional[datetime] = None

    # 原始数据
    ohlcv_data: Optional[Any] = None  # pandas DataFrame

    # 识别结果 - 分层存储
    fractals: List[FractalInfo] = field(default_factory=list)
    strokes: List[StrokeInfo] = field(default_factory=list)
    segments: List[StrokeInfo] = field(default_factory=list)  # 复用 StrokeInfo
    pivots: List[dict] = field(default_factory=list)

    # 趋势判断
    trend_direction: TrendDirection = TrendDirection.UNKNOWN
    trend_strength: float = 0.0  # 0-1

    # 背驰信息
    macd_divergence: bool = False
    divergence_type: str = ""  # "bullish", "bearish"

    # 当前持仓
    position: int = 0  # 正数做多，负数做空
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # 交易决策
    current_signal: Optional[SignalInfo] = None
    signal_history: List[SignalInfo] = field(default_factory=list)

    # 分析元数据
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    confidence_level: float = 0.5  # 整体信心度

    def add_signal(self, signal_type: SignalType, price: float,
                   confidence: float = 0.5, reason: str = ""):
        """添加新信号"""
        signal = SignalInfo(
            signal_type=signal_type,
            datetime=self.current_datetime or datetime.now(),
            price=price,
            confidence=confidence,
            reason=reason
        )
        self.current_signal = signal
        self.signal_history.append(signal)
        return signal

    def get_latest_buys(self) -> List[SignalInfo]:
        """获取最近的买入信号"""
        return [s for s in self.signal_history if s.signal_type in [
            SignalType.BUY_1, SignalType.BUY_2, SignalType.BUY_3
        ]]

    def get_latest_sells(self) -> List[SignalInfo]:
        """获取最近的卖出信号"""
        return [s for s in self.signal_history if s.signal_type in [
            SignalType.SELL_1, SignalType.SELL_2, SignalType.SELL_3
        ]]

    def is_long_position(self) -> bool:
        """是否持有多头仓位"""
        return self.position > 0

    def is_short_position(self) -> bool:
        """是否持有空头仓位"""
        return self.position < 0

    def has_position(self) -> bool:
        """是否有持仓"""
        return self.position != 0

    def __repr__(self) -> str:
        return (f"ChanLunState(symbol={self.symbol}, "
                f"trend={self.trend_direction.value}, "
                f"position={self.position}, "
                f"signal={self.current_signal.signal_type.value if self.current_signal else 'None'})")
