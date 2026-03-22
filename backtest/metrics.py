"""
绩效指标计算模块
"""

from typing import List, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np
from datetime import datetime

from .strategy import SignalType

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from .engine import Trade


class Metrics:
    """
    绩效指标计算类

    计算策略回测的各项绩效指标
    """

    def __init__(
        self,
        initial_capital: float,
        equity_curve: pd.Series,
        trades: Optional[List['Trade']] = None,
        risk_free_rate: float = 0.03
    ):
        """
        初始化绩效计算

        Args:
            initial_capital: 初始资金
            equity_curve: 权益曲线
            trades: 成交记录
            risk_free_rate: 无风险利率（年化）
        """
        self.initial_capital = initial_capital
        self.equity_curve = equity_curve
        self.trades = trades or []
        self.risk_free_rate = risk_free_rate

    @property
    def final_equity(self) -> float:
        """最终权益"""
        if len(self.equity_curve) > 0:
            return self.equity_curve.iloc[-1]
        return self.initial_capital

    @property
    def total_return(self) -> float:
        """总收益率"""
        return (self.final_equity - self.initial_capital) / self.initial_capital

    @property
    def annual_return(self) -> float:
        """年化收益率"""
        if len(self.equity_curve) < 2:
            return 0

        # 计算时间跨度（年）
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]

        # 处理不同类型的索引
        if hasattr(start_date, 'to_pydatetime'):
            start_dt = start_date.to_pydatetime()
            end_dt = end_date.to_pydatetime()
        elif isinstance(start_date, (int, float)):
            # 整数索引，假设是交易日（每年约252天）
            years = len(self.equity_curve) / 252
            if years <= 0:
                return 0
            total_return = self.total_return
            return (1 + total_return) ** (1 / years) - 1
        else:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

        years = (end_dt - start_dt).days / 365.25

        if years <= 0:
            return 0

        # 年化收益率
        total_return = self.total_return
        return (1 + total_return) ** (1 / years) - 1

    @property
    def sharpe_ratio(self) -> float:
        """夏普比率"""
        if len(self.equity_curve) < 2:
            return 0

        # 计算日收益率
        returns = self.equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return 0

        # 年化波动率
        daily_vol = returns.std()
        if daily_vol == 0:
            return 0

        annual_vol = daily_vol * np.sqrt(252)

        # 超额收益
        excess_return = self.annual_return - self.risk_free_rate

        return excess_return / annual_vol

    @property
    def max_drawdown(self) -> float:
        """最大回撤"""
        if len(self.equity_curve) < 2:
            return 0

        # 计算累计最高点
        cummax = self.equity_curve.cummax()

        # 计算回撤
        drawdown = (self.equity_curve - cummax) / cummax

        return drawdown.min()

    @property
    def max_drawdown_duration(self) -> int:
        """最大回撤持续天数"""
        if len(self.equity_curve) < 2:
            return 0

        cummax = self.equity_curve.cummax()
        drawdown = self.equity_curve < cummax

        # 计算连续回撤的最大长度
        max_duration = 0
        current_duration = 0

        for is_dd in drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    @property
    def win_rate(self) -> float:
        """胜率"""
        if not self.trades:
            return 0

        # 配对买卖交易
        trades_by_symbol = {}
        for trade in self.trades:
            if trade.symbol not in trades_by_symbol:
                trades_by_symbol[trade.symbol] = []
            trades_by_symbol[trade.symbol].append(trade)

        # 简单计算：每笔卖出与之前买入的配对
        profit_trades = 0
        total_trades = 0

        position = {}  # symbol -> (entry_price, quantity)

        for trade in self.trades:
            if trade.signal_type == SignalType.BUY:
                if trade.symbol not in position:
                    position[trade.symbol] = []
                position[trade.symbol].append((trade.price, trade.quantity))

            elif trade.signal_type == SignalType.SELL:
                if trade.symbol in position and position[trade.symbol]:
                    entry_price, entry_qty = position[trade.symbol].pop(0)
                    profit = (trade.price - entry_price) * entry_qty - trade.commission
                    total_trades += 1
                    if profit > 0:
                        profit_trades += 1

        if total_trades == 0:
            return 0

        return profit_trades / total_trades

    @property
    def profit_loss_ratio(self) -> float:
        """盈亏比"""
        if not self.trades:
            return 0

        profits = []
        losses = []
        position = {}

        for trade in self.trades:
            if trade.signal_type == SignalType.BUY:
                if trade.symbol not in position:
                    position[trade.symbol] = []
                position[trade.symbol].append((trade.price, trade.quantity))

            elif trade.signal_type == SignalType.SELL:
                if trade.symbol in position and position[trade.symbol]:
                    entry_price, entry_qty = position[trade.symbol].pop(0)
                    profit = (trade.price - entry_price) * entry_qty - trade.commission

                    if profit > 0:
                        profits.append(profit)
                    else:
                        losses.append(abs(profit))

        if not profits or not losses:
            return 0

        avg_profit = np.mean(profits)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return float('inf') if avg_profit > 0 else 0

        return avg_profit / avg_loss

    @property
    def total_trades(self) -> int:
        """总交易次数"""
        return len(self.trades)

    @property
    def profitable_trades(self) -> int:
        """盈利交易次数"""
        return self._count_profitable_trades()

    def _count_profitable_trades(self) -> int:
        """计算盈利交易次数"""
        count = 0
        position = {}

        for trade in self.trades:
            if trade.signal_type == SignalType.BUY:
                if trade.symbol not in position:
                    position[trade.symbol] = []
                position[trade.symbol].append((trade.price, trade.quantity))

            elif trade.signal_type == SignalType.SELL:
                if trade.symbol in position and position[trade.symbol]:
                    entry_price, entry_qty = position[trade.symbol].pop(0)
                    profit = (trade.price - entry_price) * entry_qty - trade.commission
                    if profit > 0:
                        count += 1

        return count

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.final_equity,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'win_rate': self.win_rate,
            'profit_loss_ratio': self.profit_loss_ratio,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades
        }

    def summary(self) -> str:
        """生成绩效摘要"""
        d = self.to_dict()

        summary = f"""
=== 回测绩效摘要 ===

初始资金: ¥{d['initial_capital']:,.2f}
最终权益: ¥{d['final_equity']:,.2f}

收益率:
  总收益率: {d['total_return']:.2%}
  年化收益: {d['annual_return']:.2%}

风险指标:
  夏普比率: {d['sharpe_ratio']:.2f}
  最大回撤: {d['max_drawdown']:.2%}
  回撤天数: {d['max_drawdown_duration']}

交易统计:
  总交易次数: {d['total_trades']}
  盈利次数: {d['profitable_trades']}
  胜率: {d['win_rate']:.2%}
  盈亏比: {d['profit_loss_ratio']:.2f}
"""
        return summary
