"""
日内交易回测引擎

支持：
- 分钟级K线数据
- 日内平仓（不持仓过夜）
- T+1交易规则（当天买入次日才能卖出）
- 手续费和滑点
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Dict, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np

from core.kline import KLine


class SignalType(Enum):
    """信号类型"""
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class Signal:
    """交易信号"""
    type: SignalType
    datetime: datetime
    price: float
    reason: str = ""
    confidence: float = 1.0  # 信号置信度


@dataclass
class Trade:
    """成交记录"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    quantity: int
    direction: str  # 'long' or 'short'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None

    @property
    def duration_minutes(self) -> Optional[int]:
        if not self.is_closed:
            return None
        return int((self.exit_time - self.entry_time).total_seconds() / 60)


@dataclass
class IntradayConfig:
    """日内交易配置"""
    initial_capital: float = 100000.0  # 初始资金
    commission_rate: float = 0.0003    # 手续费率 (万分之3)
    slippage: float = 0.001            # 滑点 (千分之一)
    min_unit: int = 100                # 最小交易单位(股)
    max_position_pct: float = 0.95     # 最大仓位比例
    force_close_time: time = time(14, 50)  # 强制平仓时间
    allow_short: bool = False          # 是否允许做空
    t1_rule: bool = True               # T+1规则 (当天买入次日才能卖出)


class IntradayEngine:
    """
    日内交易回测引擎

    特点：
    1. 日内交易，不持仓过夜
    2. 支持多周期分析
    3. 支持止损和止盈
    4. 详细的交易记录
    """

    def __init__(self, config: IntradayConfig = None):
        """
        初始化回测引擎

        Args:
            config: 交易配置
        """
        self.config = config or IntradayConfig()
        self.cash = self.config.initial_capital
        self.position = 0  # 持仓数量
        self.position_price = 0.0  # 持仓成本

        # 交易记录
        self.trades: List[Trade] = []
        self.signals: List[Dict] = []

        # 当前持仓
        self.current_trade: Optional[Trade] = None
        self.pending_buy_date: Optional[datetime] = None  # T+1规则下可卖出的日期

        # 统计数据
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

    def reset(self):
        """重置引擎状态"""
        self.cash = self.config.initial_capital
        self.position = 0
        self.position_price = 0.0
        self.trades = []
        self.signals = []
        self.current_trade = None
        self.pending_buy_date = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

    def run(
        self,
        df: pd.DataFrame,
        strategy: Callable,
        daily_df: pd.DataFrame = None
    ) -> Dict:
        """
        运行回测

        Args:
            df: 分钟级K线数据
            strategy: 策略函数 (bar, context, daily_context) -> Signal
            daily_df: 日线数据 (用于多周期分析)

        Returns:
            回测结果统计
        """
        self.reset()

        # 确保数据按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)

        # 当前日期
        current_date = None
        daily_context = None

        for idx, row in df.iterrows():
            bar_time = row['datetime']
            bar_date = bar_time.date()

            # 新的一天，检查是否需要强制平仓
            if current_date != bar_date:
                if self.current_trade and not self.current_trade.is_closed:
                    self._close_position(
                        bar_time,
                        row['open'],
                        reason="新交易日强制平仓"
                    )

                # 更新日线上下文
                if daily_df is not None:
                    daily_context = self._get_daily_context(bar_date, daily_df)

                current_date = bar_date

            # 检查强制平仓时间
            if bar_time.time() >= self.config.force_close_time:
                if self.current_trade and not self.current_trade.is_closed:
                    self._close_position(
                        bar_time,
                        row['close'],
                        reason="强制平仓"
                    )
                    continue

            # 构建上下文
            context = {
                'datetime': bar_time,
                'index': idx,
                'cash': self.cash,
                'position': self.position,
                'position_price': self.position_price,
                'total_value': self._calculate_total_value(row['close']),
                'date': bar_date,
                'can_sell': self._can_sell(bar_date)
            }

            # 调用策略获取信号
            signal = strategy(row, context, daily_context)

            if signal:
                self._process_signal(signal, row)

        # 最后强制平仓
        if self.current_trade and not self.current_trade.is_closed:
            last_row = df.iloc[-1]
            self._close_position(
                last_row['datetime'],
                last_row['close'],
                reason="回测结束平仓"
            )

        return self._calculate_results(df)

    def _get_daily_context(self, date, daily_df: pd.DataFrame) -> Dict:
        """获取日线上下文"""
        if daily_df is None:
            return None

        # 找到该日期之前的最后一根日线
        daily_df['date'] = pd.to_datetime(daily_df.index).date
        before_date = daily_df[daily_df['date'] <= date]

        if before_date.empty:
            return None

        return {
            'trend': 'up' if before_date.iloc[-1]['close'] >= before_date.iloc[-1]['open'] else 'down',
            'last_close': before_date.iloc[-1]['close'],
            'last_high': before_date.iloc[-1]['high'],
            'last_low': before_date.iloc[-1]['low']
        }

    def _can_sell(self, current_date) -> bool:
        """检查是否可以卖出 (T+1规则)"""
        if not self.config.t1_rule:
            return True

        if self.pending_buy_date is None:
            return True

        return current_date > self.pending_buy_date.date()

    def _process_signal(self, signal: Signal, bar):
        """处理交易信号"""
        signal_dict = {
            'datetime': signal.datetime,
            'type': signal.type.value,
            'price': signal.price,
            'reason': signal.reason
        }
        self.signals.append(signal_dict)

        if signal.type == SignalType.BUY:
            if self.current_trade is None:  # 没有持仓
                self._open_position(signal, bar)
        elif signal.type == SignalType.SELL:
            if self.current_trade and self._can_sell(signal.datetime):
                self._close_position(
                    signal.datetime,
                    signal.price,
                    reason=signal.reason
                )

    def _open_position(self, signal: Signal, bar):
        """开仓"""
        price = signal.price * (1 + self.config.slippage)

        # 计算可买数量
        available_cash = self.cash * self.config.max_position_pct
        max_qty = int(available_cash / price / self.config.min_unit) * self.config.min_unit

        if max_qty < self.config.min_unit:
            return  # 资金不足

        # 计算手续费
        cost = price * max_qty + price * max_qty * self.config.commission_rate

        if cost > self.cash:
            return  # 资金不足

        # 开仓
        self.cash -= cost
        self.position = max_qty
        self.position_price = price

        self.current_trade = Trade(
            entry_time=signal.datetime,
            entry_price=price,
            exit_time=None,
            exit_price=None,
            quantity=max_qty,
            direction='long'
        )

        self.pending_buy_date = signal.datetime

    def _close_position(self, datetime: datetime, price: float, reason: str):
        """平仓"""
        if self.current_trade is None:
            return

        trade = self.current_trade

        # 考虑滑点
        exit_price = price * (1 - self.config.slippage)

        # 计算手续费
        commission = exit_price * trade.quantity * self.config.commission_rate

        # 计算盈亏
        revenue = exit_price * trade.quantity - commission
        cost = trade.entry_price * trade.quantity
        pnl = revenue - cost
        pnl_pct = (pnl / cost) * 100

        # 更新资金
        self.cash += revenue

        # 更新交易记录
        trade.exit_time = datetime
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.exit_reason = reason

        self.trades.append(trade)
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1

        self.total_pnl += pnl

        # 清空持仓
        self.position = 0
        self.position_price = 0.0
        self.current_trade = None

    def _calculate_total_value(self, current_price: float) -> float:
        """计算总资产"""
        value = self.cash
        if self.position > 0:
            value += self.position * current_price
        return value

    def _calculate_results(self, df: pd.DataFrame) -> Dict:
        """计算回测结果"""
        # 默认值
        win_rate = 0
        profit_loss_ratio = 0
        avg_duration = 0
        max_drawdown = 0

        if not self.trades:
            return {
                'initial_capital': self.config.initial_capital,
                'final_capital': self.cash,
                'total_return': (self.cash / self.config.initial_capital - 1) * 100,
                'total_pnl': self.total_pnl,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'avg_duration': 0,
                'max_drawdown': 0,
                'trades': []
            }

        # 胜率
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # 总收益率
        total_return = (self.cash / self.config.initial_capital - 1) * 100

        # 盈亏比
        winning_pnl = sum(t.pnl for t in self.trades if t.pnl > 0)
        losing_pnl = sum(abs(t.pnl) for t in self.trades if t.pnl < 0)
        profit_loss_ratio = (winning_pnl / losing_pnl) if losing_pnl > 0 else 0

        # 平均持仓时间
        durations = [t.duration_minutes for t in self.trades if t.duration_minutes is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # 最大回撤
        equity_curve = self._calculate_equity_curve(df)
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        return {
            'initial_capital': self.config.initial_capital,
            'final_capital': self.cash,
            'total_return': total_return,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_duration': avg_duration,
            'max_drawdown': max_drawdown,
            'trades': self.trades
        }

    def _calculate_equity_curve(self, df: pd.DataFrame) -> List[float]:
        """计算资金曲线"""
        curve = [self.config.initial_capital]

        for _, row in df.iterrows():
            total_value = self._calculate_total_value(row['close'])
            # 检查是否有交易发生
            for trade in self.trades:
                if trade.exit_time == row['datetime']:
                    total_value = self.cash + self.position * row['close']
            curve.append(total_value)

        return curve

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """计算最大回撤"""
        if not equity_curve:
            return 0.0

        max_value = equity_curve[0]
        max_drawdown = 0.0

        for value in equity_curve:
            if value > max_value:
                max_value = value

            drawdown = (max_value - value) / max_value * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def print_results(self, results: Dict):
        """打印回测结果"""
        print("\n" + "=" * 50)
        print("回测结果")
        print("=" * 50)
        print(f"初始资金: {results['initial_capital']:,.2f}")
        print(f"最终资金: {results['final_capital']:,.2f}")
        print(f"总收益: {results['total_pnl']:,.2f} ({results['total_return']:.2f}%)")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print()
        print(f"总交易次数: {results['total_trades']}")
        print(f"盈利交易: {results['winning_trades']}")
        print(f"亏损交易: {results['losing_trades']}")
        print(f"胜率: {results['win_rate']:.2f}%")
        print(f"盈亏比: {results['profit_loss_ratio']:.2f}")
        print(f"平均持仓时间: {results['avg_duration']:.0f} 分钟")
        print("=" * 50)
