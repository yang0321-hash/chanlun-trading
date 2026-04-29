"""
优化的出场机制模块

包含：
1. 跟踪止损（Trailing Stop）
2. 分批止盈（Partial Profit Taking）
3. 时间止损（Time-based Exit）
4. 动态止损调整
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np


class ExitReason(Enum):
    """出场原因"""
    TRAILING_STOP = "trailing_stop"      # 跟踪止损
    PARTIAL_PROFIT = "partial_profit"    # 分批止盈
    TIME_STOP = "time_stop"              # 时间止损
    SIGNAL_REVERSAL = "signal_reversal"  # 信号反转
    VOLATILITY_STOP = "volatility_stop"  # 波动率止损
    TARGET_REACHED = "target_reached"    # 目标达成
    EMERGENCY = "emergency"              # 紧急出场


@dataclass
class ExitSignal:
    """出场信号"""
    should_exit: bool
    exit_ratio: float  # 0-1, 出场比例
    reason: ExitReason
    description: str
    stop_price: Optional[float] = None


@dataclass
class PositionRecord:
    """持仓记录"""
    symbol: str
    entry_price: float
    entry_date: datetime
    quantity: int
    initial_stop: float
    current_stop: float

    # 跟踪止损相关
    highest_price: float = 0
    lowest_after_entry: float = 0
    trailing_stop_offset: float = 0.05  # 跟踪止损回撤比例

    # 分批止盈相关
    profit_targets: List[Tuple[float, float]] = field(default_factory=list)  # (价格, 比例)
    exited_stages: List[int] = field(default_factory=list)

    # 时间止损
    max_hold_days: int = 30
    entry_datetime: Optional[datetime] = None

    # 状态
    is_trailing: bool = False
    is_active: bool = True

    def __post_init__(self):
        if self.entry_datetime is None:
            self.entry_datetime = datetime.now()
        if self.highest_price == 0:
            self.highest_price = self.entry_price
        if self.lowest_after_entry == 0:
            self.lowest_after_entry = self.entry_price

    def update_trail(self, current_price: float):
        """更新跟踪止损"""
        if not self.is_active:
            return

        # 更新最高价
        if current_price > self.highest_price:
            self.highest_price = current_price
            # 盈利后启动跟踪止损
            if current_price > self.entry_price * 1.05:
                self.is_trailing = True
                # 新止损 = 最高价 * (1 - 回撤比例)
                new_stop = self.highest_price * (1 - self.trailing_stop_offset)
                self.current_stop = max(self.current_stop, new_stop)

        # 更新最低价（用于止损后的恢复判断）
        if current_price < self.lowest_after_entry:
            self.lowest_after_entry = current_price

    def check_partial_exit(self, current_price: float) -> Optional[float]:
        """
        检查分批止盈

        Returns:
            应该卖出的比例 (0-1)，None表示不卖出
        """
        if not self.profit_targets:
            return None

        for i, (target_price, ratio) in enumerate(self.profit_targets):
            if i not in self.exited_stages and current_price >= target_price:
                self.exited_stages.append(i)
                return ratio

        return None

    def check_time_exit(self, current_date: datetime) -> bool:
        """检查时间止损"""
        if not self.max_hold_days or not self.entry_datetime:
            return False

        hold_duration = (current_date - self.entry_datetime).days
        return hold_duration >= self.max_hold_days


class TrailingStopManager:
    """
    跟踪止损管理器

    根据价格移动动态调整止损位
    """

    def __init__(
        self,
        initial_offset: float = 0.05,    # 初始止损回撤 5%
        trailing_offset: float = 0.05,    # 跟踪止损回撤 5%
        activation_profit: float = 0.03,  # 激活跟踪止损的最小盈利 3%
    ):
        self.initial_offset = initial_offset
        self.trailing_offset = trailing_offset
        self.activation_profit = activation_profit

    def calculate_stop(
        self,
        position: PositionRecord,
        current_price: float
    ) -> float:
        """
        计算跟踪止损价格

        Args:
            position: 持仓记录
            current_price: 当前价格

        Returns:
            止损价格
        """
        # 计算盈利比例
        profit_ratio = (current_price - position.entry_price) / position.entry_price

        # 盈利未达到激活阈值，使用初始止损
        if profit_ratio < self.activation_profit:
            return position.entry_price * (1 - self.initial_offset)

        # 盈利后使用跟踪止损
        # 止损 = 最高价 * (1 - 跟踪回撤)
        trailing_stop = position.highest_price * (1 - self.trailing_offset)

        # 确保止损不低于保本位
        breakeven_stop = position.entry_price * 1.002  # 保本+手续费

        return max(trailing_stop, breakeven_stop)

    def should_exit(
        self,
        position: PositionRecord,
        current_price: float
    ) -> Tuple[bool, float]:
        """
        判断是否应该止损

        Returns:
            (是否止损, 止损价格)
        """
        stop_price = self.calculate_stop(position, current_price)
        return current_price <= stop_price, stop_price


class PartialProfitManager:
    """
    分批止盈管理器

    在不同价格分批卖出锁定利润
    """

    def __init__(
        self,
        targets: List[Tuple[float, float]] = None,
        auto_adjust: bool = True,
    ):
        """
        Args:
            targets: 止盈目标列表 [(价格比例, 卖出比例), ...]
                    如 [(0.05, 0.3), (0.10, 0.3), (0.15, 0.4)]
                    表示盈利5%卖30%, 10%卖30%, 15%卖40%
            auto_adjust: 是否根据波动率自动调整目标
        """
        self.targets = targets or [
            (0.05, 0.25),   # 盈利5%卖25%
            (0.10, 0.25),   # 盈利10%卖25%
            (0.15, 0.25),   # 盈利15%卖25%
            (0.20, 0.25),   # 盈利20%卖剩余
        ]
        self.auto_adjust = auto_adjust

    def get_targets(
        self,
        entry_price: float,
        volatility: float = None
    ) -> List[Tuple[float, float]]:
        """
        获取止盈目标价格

        Args:
            entry_price: 入场价格
            volatility: 波动率（用于调整）

        Returns:
            [(目标价格, 卖出比例), ...]
        """
        targets = []
        remaining_ratio = 1.0

        for profit_ratio, exit_ratio in self.targets:
            actual_ratio = min(exit_ratio, remaining_ratio)
            if actual_ratio <= 0:
                continue

            # 根据波动率调整
            if self.auto_adjust and volatility:
                # 高波动时放宽目标
                adjustment = 1 + min(volatility * 10, 0.5)
                adjusted_profit = profit_ratio * adjustment
            else:
                adjusted_profit = profit_ratio

            target_price = entry_price * (1 + adjusted_profit)
            targets.append((target_price, actual_ratio))
            remaining_ratio -= actual_ratio

        # 确保最后一批卖出剩余
        if targets and remaining_ratio > 0:
            last_price, _ = targets[-1]
            targets[-1] = (last_price, targets[-1][1] + remaining_ratio)

        return targets

    def check_exit(
        self,
        position: PositionRecord,
        current_price: float,
        volatility: float = None
    ) -> Optional[float]:
        """
        检查是否需要分批止盈

        Returns:
            应该卖出的比例，None表示不卖出
        """
        if not position.profit_targets:
            return None

        for i, (target_price, ratio) in enumerate(position.profit_targets):
            if i in position.exited_stages:
                continue

            if current_price >= target_price:
                position.exited_stages.append(i)
                return ratio

        return None


class TimeStopManager:
    """
    时间止损管理器

    根据持仓时间判断是否出场
    """

    def __init__(
        self,
        max_hold_days: int = 30,
        profit_time_adjustment: bool = True,
    ):
        self.max_hold_days = max_hold_days
        self.profit_time_adjustment = profit_time_adjustment

    def should_exit(
        self,
        position: PositionRecord,
        current_price: float,
        current_date: datetime = None
    ) -> Tuple[bool, str]:
        """
        判断是否应该时间止损

        有盈利时可以延长时间
        """
        if current_date is None:
            current_date = datetime.now()

        if not position.entry_datetime:
            return False, ""

        hold_duration = (current_date - position.entry_datetime).days
        profit_ratio = (current_price - position.entry_price) / position.entry_price

        # 有盈利时延长时间
        if self.profit_time_adjustment and profit_ratio > 0:
            # 每盈利1%，延长2天
            extended_days = self.max_hold_days + int(profit_ratio * 100 * 2)
            should_exit = hold_duration >= extended_days
        else:
            should_exit = hold_duration >= self.max_hold_days

        if should_exit:
            return True, f"持仓{hold_duration}天(限制{self.max_hold_days}天), 盈利{profit_ratio:.2%}"

        return False, ""


class DynamicExitManager:
    """
    动态出场管理器

    综合多种出场条件，动态调整策略
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        use_trailing_stop: bool = True,
        use_partial_profit: bool = True,
        use_time_stop: bool = True,
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_trailing_stop = use_trailing_stop
        self.use_partial_profit = use_partial_profit
        self.use_time_stop = use_time_stop

        self.trailing_manager = TrailingStopManager()
        self.profit_manager = PartialProfitManager()
        self.time_manager = TimeStopManager()

        self.positions: Dict[str, PositionRecord] = {}

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        initial_stop: float = None,
        volatility: float = None
    ) -> PositionRecord:
        """开仓记录"""
        if initial_stop is None:
            # 默认止损
            initial_stop = entry_price * 0.95

        # 获取分批止盈目标
        profit_targets = self.profit_manager.get_targets(entry_price, volatility)

        position = PositionRecord(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            quantity=quantity,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            profit_targets=profit_targets,
            trailing_stop_offset=0.05 if volatility and volatility < 0.3 else 0.08
        )

        self.positions[symbol] = position
        return position

    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_date: datetime = None
    ) -> List[ExitSignal]:
        """
        更新持仓并检查出场信号

        Returns:
            出场信号列表
        """
        if current_date is None:
            current_date = datetime.now()

        position = self.positions.get(symbol)
        if not position or not position.is_active:
            return []

        signals = []

        # 1. 跟踪止损检查
        if self.use_trailing_stop:
            should_exit, stop_price = self.trailing_manager.should_exit(position, current_price)
            if should_exit:
                signals.append(ExitSignal(
                    should_exit=True,
                    exit_ratio=1.0,  # 全部卖出
                    reason=ExitReason.TRAILING_STOP,
                    description=f"跟踪止损触发 @ {current_price:.2f}, 止损位{stop_price:.2f}",
                    stop_price=stop_price
                ))

        # 2. 分批止盈检查
        if self.use_partial_profit and not signals:
            exit_ratio = self.profit_manager.check_exit(position, current_price)
            if exit_ratio is not None:
                profit_pct = (current_price - position.entry_price) / position.entry_price
                signals.append(ExitSignal(
                    should_exit=True,
                    exit_ratio=exit_ratio,
                    reason=ExitReason.PARTIAL_PROFIT,
                    description=f"分批止盈: 盈利{profit_pct:.2%}, 卖出{exit_ratio:.0%}",
                    stop_price=current_price
                ))

        # 3. 时间止损检查
        if self.use_time_stop and not signals:
            should_exit, desc = self.time_manager.should_exit(position, current_price, current_date)
            if should_exit:
                signals.append(ExitSignal(
                    should_exit=True,
                    exit_ratio=1.0,
                    reason=ExitReason.TIME_STOP,
                    description=desc,
                    stop_price=current_price
                ))

        # 更新跟踪止损
        if self.use_trailing_stop:
            position.update_trail(current_price)

        return signals

    def close_position(
        self,
        symbol: str,
        exit_ratio: float = 1.0
    ):
        """平仓处理"""
        position = self.positions.get(symbol)
        if position:
            if exit_ratio >= 1.0:
                position.is_active = False
                # 全部平仓，移除记录
                if symbol in self.positions:
                    del self.positions[symbol]
            else:
                # 部分平仓，减少数量
                position.quantity = int(position.quantity * (1 - exit_ratio))

    def get_position(self, symbol: str) -> Optional[PositionRecord]:
        """获取持仓记录"""
        return self.positions.get(symbol)


class VolatilityBasedExit:
    """
    基于波动率的出场

    当波动率异常时及时出场
    """

    def __init__(
        self,
        volatility_period: int = 20,
        volatility_threshold: float = 0.05,
        volatility_multiplier: float = 2.0,
    ):
        self.volatility_period = volatility_period
        self.volatility_threshold = volatility_threshold
        self.volatility_multiplier = volatility_multiplier

    def calculate_volatility(self, df: pd.DataFrame, index: int) -> float:
        """计算历史波动率"""
        if len(df) < self.volatility_period:
            return 0.02

        lookback = df.iloc[max(0, index - self.volatility_period):index + 1]
        returns = lookback['close'].pct_change().dropna()

        if len(returns) == 0:
            return 0.02

        vol = returns.std() * np.sqrt(252)  # 年化
        return vol

    def should_exit(
        self,
        df: pd.DataFrame,
        index: int,
        position: PositionRecord
    ) -> Tuple[bool, str]:
        """
        检查波动率是否过高需要出场

        条件：
        1. 当前波动率是历史均值的2倍以上
        2. 当前价格在不利方向移动
        """
        current_vol = self.calculate_volatility(df, index)
        current_price = df['close'].iloc[index]

        # 检查是否持仓亏损
        is_loss = current_price < position.entry_price

        # 波动率激增且亏损，及时止损
        if current_vol > self.volatility_threshold * self.volatility_multiplier and is_loss:
            return True, f"波动率激增: {current_vol:.2%} > {self.volatility_threshold * self.volatility_multiplier:.2%}"

        return False, ""
