"""
智能策略选择器

根据股票特性自动选择最合适的交易策略：

1. 波动率分析 - 高波动用原版，低波动用优化版
2. 趋势强度 - 用ADX判断趋势强度
3. 价格区间 - 分析股票所处的价格区间

决策逻辑:
- 高波动 + 强趋势 → 原版策略 (捕捉更多机会)
- 低波动 + 弱趋势 → 优化策略 (减少止损)
- 震荡市 → 优化策略 (趋势过滤)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger

from backtest.strategy import Strategy, Signal
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy
from strategies.optimized_weekly_daily_strategy import OptimizedWeeklyDailyStrategy


class StockCharacteristics:
    """股票特征分析"""

    @staticmethod
    def analyze(df: pd.DataFrame, period: int = 60) -> Dict[str, Any]:
        """分析股票特征"""
        if len(df) < period:
            period = len(df) // 2

        recent = df.tail(period)

        # 1. 波动率 (标准差/均值)
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率

        # 2. 趋势强度 (线性回归R²)
        x = np.arange(len(recent['close']))
        y = recent['close'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        y_hat = p(x)
        y_bar = np.mean(y)
        ss_tot = np.sum((y - y_bar) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 3. 价格区间位置 (相对最低点的位置)
        period_low = recent['close'].min()
        period_high = recent['close'].max()
        current = recent['close'].iloc[-1]
        price_position = (current - period_low) / (period_high - period_low) if period_high != period_low else 0.5

        # 4. ATR (平均真实波幅)
        atr = StockCharacteristics.calculate_atr(recent)

        # 5. 交易活跃度 (成交量)
        volume_avg = recent['volume'].mean()
        volume_recent = recent['volume'].iloc[-10:].mean()
        volume_ratio = volume_recent / volume_avg if volume_avg > 0 else 1

        return {
            'volatility': volatility,
            'r_squared': r_squared,
            'price_position': price_position,
            'atr': atr,
            'volume_ratio': volume_ratio,
            'trend_strength': r_squared,  # R²作为趋势强度
            'is_high_volatility': volatility > 0.30,  # 年化波动率30%以上
            'is_strong_trend': r_squared > 0.7,     # R²>0.7为强趋势
            'is_ranging': r_squared < 0.3,         # R²<0.3为震荡
        }

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_list.append(max(tr1, tr2, tr3))

        if len(tr_list) == 0:
            return 0

        return np.mean(tr_list[-period:]) if len(tr_list) >= period else np.mean(tr_list)


class StrategySelector:
    """
    策略选择器

    根据股票特征自动选择最合适的交易策略
    """

    def __init__(
        self,
        default_strategy: str = 'original',  # 默认策略
        volatility_threshold: float = 0.30,   # 波动率阈值
        trend_threshold: float = 0.70,        # 趋势强度阈值
    ):
        self.default_strategy = default_strategy
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.selection_history: Dict[str, str] = {}

    def select_strategy(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> tuple:
        """
        选择策略

        Returns:
            (strategy_instance, strategy_name, reason)
        """
        # 分析股票特征
        chara = StockCharacteristics.analyze(df)

        strategy = None
        strategy_name = ""
        reason_parts = []

        # 决策逻辑
        is_ranging = chara['is_ranging']
        is_strong_trend = chara['is_strong_trend']
        is_high_volatility = chara['is_high_volatility']

        # 规则1: 震荡市 → 优化策略 (趋势过滤减少无效交易)
        if is_ranging:
            strategy = OptimizedWeeklyDailyStrategy(name='优化策略(震荡)')
            strategy_name = 'optimized'
            reason_parts.append(f"震荡市(R²={chara['r_squared']:.2f})")

        # 规则2: 强趋势 + 高波动 → 原版策略 (捕捉更多机会)
        elif is_strong_trend and is_high_volatility:
            strategy = WeeklyDailyChanLunStrategy(name='原版策略(趋势)')
            strategy_name = 'original'
            reason_parts.append(f"强趋势高波动(R²={chara['r_squared']:.2f}, 波动={chara['volatility']:.2%})")

        # 规则3: 弱趋势 + 低波动 → 优化策略
        elif not is_strong_trend and not is_high_volatility:
            strategy = OptimizedWeeklyDailyStrategy(name='优化策略(稳健)')
            strategy_name = 'optimized'
            reason_parts.append(f"稳健型(R²={chara['r_squared']:.2f}, 波动={chara['volatility']:.2%})")

        # 规则4: 默认 → 原版策略
        else:
            strategy = WeeklyDailyChanLunStrategy(name='原版策略(默认)')
            strategy_name = 'original'
            reason_parts.append("默认选择")

        # 记录选择历史
        self.selection_history[symbol] = strategy_name

        reason = " | ".join(reason_parts)

        logger.info(f"{symbol} 策略选择: {strategy_name} ({reason})")

        return strategy, strategy_name, reason

    def get_selection_summary(self) -> Dict[str, int]:
        """获取选择汇总"""
        summary = {}
        for strategy in self.selection_history.values():
            summary[strategy] = summary.get(strategy, 0) + 1
        return summary


class AdaptiveChanLunStrategy(Strategy):
    """
    自适应缠论策略

    自动根据股票特征选择原版或优化版策略
    """

    def __init__(
        self,
        name: str = '自适应缠论策略',
        selector: Optional[StrategySelector] = None,
    ):
        super().__init__(name)
        self.selector = selector or StrategySelector()
        self.active_strategy: Optional[Strategy] = None
        self.active_strategy_name: str = ""

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{name}: 资金{capital:,.0f}")

    def reset(self) -> None:
        """重置策略"""
        super().reset()
        if self.active_strategy:
            self.active_strategy.reset()

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线 - 委托给选中的策略"""
        from backtest.strategy import Signal

        # 首次运行时选择策略
        if self.active_strategy is None:
            df = context['data'].get(symbol)
            if df is not None:
                # 只在数据足够时才分析
                if len(df) >= 100:
                    strategy, strategy_name, reason = self.selector.select_strategy(symbol, df)
                    self.active_strategy = strategy
                    self.active_strategy_name = strategy_name
                    # 初始化策略
                    self.active_strategy.initialize(capital, symbols)
        else:
            # 定期重新评估策略 (每100根K线)
            if index % 100 == 0 and index > 0:
                df = context['data'].get(symbol)
                if df is not None and len(df) >= 100:
                    _, strategy_name, _ = self.selector.select_strategy(symbol, df)
                    if strategy_name != self.active_strategy_name:
                        # 策略切换
                        logger.info(f"{symbol} 策略切换: {self.active_strategy_name} → {strategy_name}")
                        self.active_strategy.reset()
                        new_strategy, _, _ = self.selector.select_strategy(symbol, df)
                        self.active_strategy = new_strategy
                        self.active_strategy_name = strategy_name
                        self.active_strategy.initialize(capital, symbols)

        # 委托给活动策略处理
        if self.active_strategy:
            signal = self.active_strategy.on_bar(bar, symbol, index, context)
            return signal

        return None

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调 - 委托给活动策略"""
        if self.active_strategy:
            self.active_strategy.on_order(signal, executed_price, executed_quantity)

    @property
    def cash(self) -> float:
        """获取现金"""
        return self.cash

    @cash.setter
    def cash(self, value: float):
        """设置现金"""
        self.cash = value

    @property
    def position(self) -> Dict[str, int]:
        """获取持仓"""
        return self.position

    @position.setter
    def position(self, value: Dict[str, int]):
        """设置持仓"""
        self.position = value
