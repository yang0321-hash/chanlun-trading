"""
回测引擎
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
from loguru import logger

from .strategy import Strategy, Signal, SignalType
from .metrics import Metrics


@dataclass
class Trade:
    """成交记录"""
    symbol: str
    datetime: datetime
    signal_type: SignalType
    price: float
    quantity: int
    commission: float
    reason: str = ''


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000
    commission: float = 0.0003  # 手续费率
    slippage: float = 0.0001    # 滑点
    min_unit: int = 100         # 最小交易单位
    position_limit: float = 0.95 # 仓位上限


class BacktestEngine:
    """
    回测引擎

    提供策略回测功能
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测引擎

        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        self.strategy: Optional[Strategy] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple] = []
        self.signals_generated: List[Signal] = []

    def add_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        添加数据

        Args:
            symbol: 股票代码
            df: K线数据
        """
        # 确保按时间排序
        df = df.sort_index().copy()
        self.data[symbol] = df

    def set_strategy(self, strategy: Strategy) -> None:
        """
        设置策略

        Args:
            strategy: 策略对象
        """
        self.strategy = strategy

    def run(self) -> Dict[str, Any]:
        """
        运行回测

        Returns:
            回测结果字典
        """
        if not self.data:
            raise ValueError("没有数据，请先添加数据")

        if not self.strategy:
            raise ValueError("没有设置策略")

        # 初始化策略
        self.strategy.initialize(
            capital=self.config.initial_capital,
            symbols=list(self.data.keys())
        )

        # 重置状态
        self.trades = []
        self.equity_curve = []
        self.signals_generated = []

        # 找出所有日期并合并
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        logger.info(f"开始回测，共 {len(all_dates)} 个交易日")

        # 逐日回测
        for i, date in enumerate(all_dates):
            self._on_bar(date, i)

        # 计算绩效
        results = self._calculate_results()

        logger.info(f"回测完成，总收益率: {results['total_return']:.2%}")

        return results

    def _on_bar(self, date: pd.Timestamp, bar_index: int) -> None:
        """处理单个交易日"""
        current_prices = {}
        context = {
            'date': date,
            'bar_index': bar_index,
            'data': {}
        }

        # 获取当日价格和构建上下文
        for symbol, df in self.data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']
                # 截止到当前日期的历史数据
                context['data'][symbol] = df[df.index <= date]

        if not current_prices:
            return

        # 为每个品种触发策略
        for symbol in self.data.keys():
            if date not in self.data[symbol].index:
                continue

            bar = self.data[symbol].loc[date]

            # 调用策略
            try:
                signal = self.strategy.on_bar(
                    bar=bar,
                    symbol=symbol,
                    index=bar_index,
                    context=context
                )

                if signal:
                    self.signals_generated.append(signal)
                    self._execute_signal(signal, current_prices.get(symbol, bar['close']))

            except Exception as e:
                logger.error(f"策略执行错误 {symbol} @ {date}: {e}")

        # 记录权益曲线
        equity = self.strategy.get_equity(current_prices)
        self.equity_curve.append((date, equity))

    def _execute_signal(self, signal: Signal, current_price: float) -> None:
        """执行信号"""
        if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
            return

        symbol = signal.symbol

        # 考虑滑点
        if signal.is_buy():
            execution_price = current_price * (1 + self.config.slippage)
        else:
            execution_price = current_price * (1 - self.config.slippage)

        # 计算数量
        if signal.quantity is None:
            # 根据资金计算数量
            if signal.is_buy():
                available_cash = self.strategy.get_cash() * self.config.position_limit
                # 先计算能买多少股，然后取整到min_unit
                shares = int(available_cash / execution_price)
                quantity = (shares // self.config.min_unit) * self.config.min_unit
            else:
                quantity = self.strategy.get_position(symbol)
        else:
            quantity = signal.quantity

        # 检查最小交易单位
        if quantity < self.config.min_unit:
            return

        # 计算手续费
        commission = execution_price * quantity * self.config.commission
        commission = max(commission, 5)  # 最低5元

        # 执行交易
        try:
            self.strategy.on_order(signal, execution_price, quantity)

            # 记录交易
            trade = Trade(
                symbol=symbol,
                datetime=signal.datetime,
                signal_type=signal.signal_type,
                price=execution_price,
                quantity=quantity,
                commission=commission,
                reason=signal.reason
            )
            self.trades.append(trade)

            logger.debug(
                f"{signal.signal_type.value.upper()} {symbol} "
                f"@ {execution_price:.2f} x {quantity}"
            )

        except Exception as e:
            logger.error(f"执行订单失败: {e}")

    def _calculate_results(self) -> Dict[str, Any]:
        """计算回测结果"""
        if not self.equity_curve:
            return {}

        # 转换为DataFrame
        df_equity = pd.DataFrame(
            self.equity_curve,
            columns=['date', 'equity']
        ).set_index('date')

        # 计算绩效指标
        metrics = Metrics(
            initial_capital=self.config.initial_capital,
            equity_curve=df_equity['equity'],
            trades=self.trades
        )

        return {
            'total_return': metrics.total_return,
            'annual_return': metrics.annual_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'profit_loss_ratio': metrics.profit_loss_ratio,
            'total_trades': metrics.total_trades,
            'final_equity': metrics.final_equity,
            'equity_curve': df_equity,
            'trades': self.trades,
            'signals': self.signals_generated
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线"""
        if self.equity_curve:
            return pd.DataFrame(
                self.equity_curve,
                columns=['date', 'equity']
            ).set_index('date')
        return pd.DataFrame()

    def get_trades(self) -> List[Trade]:
        """获取成交记录"""
        return self.trades

    def get_signals(self) -> List[Signal]:
        """获取信号记录"""
        return self.signals_generated
