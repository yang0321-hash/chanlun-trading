"""
策略基类和信号定义
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd


class SignalType(Enum):
    """信号类型"""
    BUY = 'buy'          # 买入信号
    SELL = 'sell'        # 卖出信号
    SHORT = 'short'       # 做空信号（A股不支持）
    COVER = 'cover'       # 平空信号
    HOLD = 'hold'        # 持有
    CLOSE = 'close'       # 平仓
    NONE = 'none'        # 无信号


@dataclass
class Signal:
    """
    交易信号

    Attributes:
        signal_type: 信号类型
        symbol: 股票代码
        datetime: 信号时间
        price: 信号价格
        quantity: 建议数量（可选）
        reason: 信号原因
        confidence: 信号置信度 (0-1)
        metadata: 额外信息
    """
    signal_type: SignalType
    symbol: str
    datetime: datetime
    price: float
    quantity: Optional[int] = None
    reason: str = ''
    confidence: float = 0.5
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def is_buy(self) -> bool:
        """是否买入信号"""
        return self.signal_type == SignalType.BUY

    def is_sell(self) -> bool:
        """是否卖出信号"""
        return self.signal_type == SignalType.SELL

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'type': self.signal_type.value,
            'symbol': self.symbol,
            'datetime': self.datetime,
            'price': self.price,
            'quantity': self.quantity,
            'reason': self.reason,
            'confidence': self.confidence
        }


class Strategy:
    """
    策略基类

    所有策略需要继承此类并实现 on_bar 方法
    """

    def __init__(self, name: str = 'Strategy'):
        """
        初始化策略

        Args:
            name: 策略名称
        """
        self.name = name
        self.signals: List[Signal] = []
        self.position: Dict[str, int] = {}  # symbol -> quantity
        self.cash: float = 0
        self.initial_capital: float = 0

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """
        初始化策略

        Args:
            capital: 初始资金
            symbols: 交易品种列表
        """
        self.initial_capital = capital
        self.cash = capital
        for symbol in symbols:
            self.position[symbol] = 0

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        处理单根K线

        子类必须实现此方法

        Args:
            bar: K线数据
            symbol: 股票代码
            index: K线索引
            context: 上下文信息（包含历史数据、指标等）

        Returns:
            交易信号，None表示无操作
        """
        raise NotImplementedError("子类必须实现 on_bar 方法")

    def on_order(
        self,
        signal: Signal,
        executed_price: float,
        executed_quantity: int
    ) -> None:
        """
        订单成交回调

        Args:
            signal: 原始信号
            executed_price: 成交价格
            executed_quantity: 成交数量
        """
        symbol = signal.symbol

        if signal.is_buy():
            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity
        elif signal.is_sell():
            self.position[symbol] = self.position.get(symbol, 0) - executed_quantity
            self.cash += executed_price * executed_quantity

    def get_position(self, symbol: str) -> int:
        """获取持仓"""
        return self.position.get(symbol, 0)

    def get_cash(self) -> float:
        """获取可用资金"""
        return self.cash

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """
        计算当前权益

        Args:
            current_prices: 当前价格字典

        Returns:
            总权益
        """
        equity = self.cash
        for symbol, quantity in self.position.items():
            if quantity != 0 and symbol in current_prices:
                equity += quantity * current_prices[symbol]
        return equity

    def reset(self) -> None:
        """重置策略状态"""
        self.signals = []
        self.position = {}
        self.cash = self.initial_capital
