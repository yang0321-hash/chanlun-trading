"""
多周期回测引擎

继承BacktestEngine，增加多周期数据支持。
支持周线/日线/60分钟/30分钟数据。
"""

from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger

from .engine import BacktestEngine, BacktestConfig
from .strategy import Strategy


class MultiTFBacktestEngine(BacktestEngine):
    """
    多周期回测引擎

    扩展BacktestEngine以支持周线/日线/60分钟/30分钟数据。

    使用方法：
        engine = MultiTFBacktestEngine()
        engine.add_multi_tf_data('sh600519', daily_df, min60_df=min60_df)
        engine.set_strategy(strategy)
        result = engine.run()
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        super().__init__(config)
        self.min30_data: Dict[str, pd.DataFrame] = {}
        self.min60_data: Dict[str, pd.DataFrame] = {}
        self.weekly_data: Dict[str, pd.DataFrame] = {}

    def add_multi_tf_data(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        weekly_df: Optional[pd.DataFrame] = None,
        min60_df: Optional[pd.DataFrame] = None,
        min30_df: Optional[pd.DataFrame] = None,
    ):
        """
        添加多周期数据

        Args:
            symbol: 股票代码
            daily_df: 日线数据（必须）
            weekly_df: 周线数据（可选，不提供则从日线resample）
            min60_df: 60分钟数据（可选）
            min30_df: 30分钟数据（可选）
        """
        # 日线数据走父类
        self.add_data(symbol, daily_df)

        # 周线数据
        if weekly_df is not None:
            self.weekly_data[symbol] = weekly_df.sort_index().copy()
        else:
            self.weekly_data[symbol] = self._resample_weekly(daily_df)

        # 60分钟数据
        if min60_df is not None:
            self.min60_data[symbol] = self._normalize_intraday(min60_df)

        # 30分钟数据
        if min30_df is not None:
            self.min30_data[symbol] = self._normalize_intraday(min30_df)

        logger.debug(
            f"多周期数据已加载: {symbol} "
            f"日线={len(daily_df)}, "
            f"周线={len(self.weekly_data.get(symbol, []))}, "
            f"60分={len(min60_df) if min60_df is not None else 0}, "
            f"30分={len(min30_df) if min30_df is not None else 0}"
        )

        # 如果策略已设置，预加载数据
        if self.strategy and hasattr(self.strategy, 'set_multi_tf_data'):
            self.strategy.set_multi_tf_data(
                symbol, daily_df,
                min60_df=self.min60_data.get(symbol),
                min30_df=self.min30_data.get(symbol),
            )

    def _resample_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """日线→周线"""
        if len(daily_df) == 0:
            return daily_df
        weekly = daily_df.resample('W').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
        if 'amount' in daily_df.columns:
            weekly['amount'] = daily_df.resample('W').agg({'amount': 'sum'}).dropna()['amount']
        return weekly

    def _normalize_intraday(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化分钟数据"""
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ['date', 'datetime', 'time']:
                if col in df.columns:
                    df.index = pd.to_datetime(df[col])
                    break
        return df.sort_index()

    def set_strategy(self, strategy: Strategy) -> None:
        """设置策略（重写以注入多周期数据）"""
        super().set_strategy(strategy)

        # 如果策略支持多周期，注入已有数据
        if hasattr(strategy, 'set_multi_tf_data'):
            for symbol in self.data:
                strategy.set_multi_tf_data(
                    symbol, self.data[symbol],
                    min60_df=self.min60_data.get(symbol),
                    min30_df=self.min30_data.get(symbol),
                )

    def run(self) -> Dict[str, Any]:
        """运行多周期回测"""
        if not self.strategy:
            raise ValueError("未设置策略")
        if not self.data:
            raise ValueError("未添加数据")

        # 初始化
        symbols = list(self.data.keys())
        self.strategy.initialize(self.config.initial_capital, symbols)

        self.trades = []
        self.equity_curve = []
        self.signals_generated = []

        # 按日线迭代
        for symbol in symbols:
            df = self.data[symbol]

            for i in range(len(df)):
                bar = df.iloc[i]
                price = bar['close']

                # 更新持仓市值
                current_position = self.strategy.get_position(symbol)

                # 构建上下文
                context = {
                    'data': {symbol: df.iloc[:i + 1]},
                    'bar_index': i,
                    'min60_slice': None,
                    'min30_slice': None,
                }

                daily_date = df.index[i]
                date_str = pd.Timestamp(daily_date).strftime('%Y-%m-%d')

                # 提供60分钟切片
                min60_df = self.min60_data.get(symbol)
                if min60_df is not None and len(min60_df) > 0:
                    try:
                        mask = min60_df.index.strftime('%Y-%m-%d') == date_str
                        context['min60_slice'] = min60_df[mask]
                    except Exception:
                        pass

                # 提供30分钟切片
                min30_df = self.min30_data.get(symbol)
                if min30_df is not None and len(min30_df) > 0:
                    try:
                        mask = min30_df.index.strftime('%Y-%m-%d') == date_str
                        context['min30_slice'] = min30_df[mask]
                    except Exception:
                        pass

                # 触发策略
                signal = self.strategy.on_bar(bar, symbol, i, context)

                if signal:
                    self._execute_signal(signal, price)

                # 记录净值
                equity = self.strategy.get_equity({symbol: price})
                self.equity_curve.append((df.index[i], equity))

        return self._calculate_results()
