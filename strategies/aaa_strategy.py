#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AAA 交易策略
A-share Analysis All-in-One Trading Strategy

基于 AAA 评分系统的完整量化交易策略。
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtest.strategy import Strategy, Signal, SignalType
from indicators.aaa_indicators import AAAIndicatorEngine
from indicators.aaa_scorer import AAAScores, AAAScorer
from strategies.aaa_signal_generator import AAASignalGenerator, AAASignal


class AAAStrategy(Strategy):
    """
    AAA 交易策略

    基于 AAA 多维度评分系统的量化交易策略。

    信号规则：
    - 买入: 评分 >= 70 且评分变化 >= 10
    - 卖出: 评分 <= 30 或评分下降 > 20 或触发止损止盈

    仓位管理：
    - 评分 85+: 30% 最大仓位
    - 评分 70-84: 18% 最大仓位
    - 评分 55-69: 12% 最大仓位
    """

    def __init__(
        self,
        name: str = 'AAA策略',
        buy_threshold: float = 70,
        sell_threshold: float = 30,
        max_position_ratio: float = 0.3,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        enable_score_change_filter: bool = True,
        min_score_change: float = 10
    ):
        """
        初始化 AAA 策略

        Args:
            name: 策略名称
            buy_threshold: 买入评分阈值
            sell_threshold: 卖出评分阈值
            max_position_ratio: 最大单只股票仓位比例
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
            enable_score_change_filter: 是否启用评分变化过滤
            min_score_change: 最小评分变化要求
        """
        super().__init__(name)

        # 参数
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_position_ratio = max_position_ratio
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.enable_score_change_filter = enable_score_change_filter
        self.min_score_change = min_score_change

        # 组件
        self.signal_generator = AAASignalGenerator(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            min_score_change=min_score_change,
            enable_score_change_filter=enable_score_change_filter
        )

        # 状态跟踪
        self.entry_prices: Dict[str, float] = {}
        self.entry_scores: Dict[str, float] = {}
        self.stop_losses: Dict[str, float] = {}
        self.take_profits: Dict[str, float] = {}
        self.score_history: Dict[str, List[AAAScores]] = {}
        self.indicator_cache: Dict[str, pd.DataFrame] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """
        初始化策略

        Args:
            capital: 初始资金
            symbols: 交易品种列表
        """
        super().initialize(capital, symbols)

        # 初始化状态
        for symbol in symbols:
            self.entry_prices[symbol] = 0
            self.entry_scores[symbol] = 0
            self.stop_losses[symbol] = 0
            self.take_profits[symbol] = 0
            self.score_history[symbol] = []
            self.indicator_cache[symbol] = pd.DataFrame()

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        处理单根K线

        Args:
            bar: K线数据
            symbol: 股票代码
            index: K线索引
            context: 上下文信息

        Returns:
            交易信号，None表示无操作
        """
        # 获取历史数据
        hist_df = context.get('data', {}).get(symbol)
        if hist_df is None or len(hist_df) < 60:
            return None

        # 截取到当前K线
        current_df = hist_df.iloc[:index+1].copy()
        if len(current_df) < 60:
            return None

        # 计算技术指标
        try:
            indicator_engine = AAAIndicatorEngine(current_df)
            current_df = indicator_engine.calculate_all()
        except Exception as e:
            return None

        # 获取当前持仓
        position = self.get_position(symbol)

        # 生成 AAA 信号
        aaa_signal = self.signal_generator.generate_signal(
            symbol=symbol,
            df=current_df,
            current_bar=bar,
            index=index,
            position=position,
            entry_price=self.entry_prices.get(symbol),
            entry_score=self.entry_scores.get(symbol)
        )

        if aaa_signal is None:
            return None

        # 记录评分历史
        self.score_history[symbol].append(aaa_signal.scores)
        if len(self.score_history[symbol]) > 200:
            self.score_history[symbol] = self.score_history[symbol][-200:]

        # 转换 AAA 信号为回测信号
        return self._convert_signal(aaa_signal, symbol)

    def _convert_signal(self, aaa_signal: AAASignal, symbol: str) -> Optional[Signal]:
        """
        将 AAA 信号转换为回测信号

        Args:
            aaa_signal: AAA 信号
            symbol: 股票代码

        Returns:
            回测信号
        """
        # 确定信号类型
        signal_type = self._map_signal_type(aaa_signal.signal_type)
        if signal_type is None:
            return None

        # 计算交易数量
        quantity = self._calculate_quantity(aaa_signal, symbol)

        # 构建元数据
        metadata = {
            'aaa_score': aaa_signal.scores.total,
            'aaa_level': aaa_signal.scores.level,
            'stop_loss': aaa_signal.stop_loss,
            'take_profit': aaa_signal.take_profit,
            'scores': {
                'trend': aaa_signal.scores.trend,
                'momentum': aaa_signal.scores.momentum,
                'rsi': aaa_signal.scores.rsi,
                'volume': aaa_signal.scores.volume,
                'boll': aaa_signal.scores.boll,
                'kdj': aaa_signal.scores.kdj
            }
        }

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            datetime=aaa_signal.datetime,
            price=aaa_signal.price,
            quantity=quantity,
            reason=aaa_signal.reason,
            confidence=aaa_signal.confidence,
            metadata=metadata
        )

    def _map_signal_type(self, aaa_signal_type: AAASignal) -> Optional[SignalType]:
        """
        映射 AAA 信号类型到回测信号类型

        Args:
            aaa_signal_type: AAA 信号类型

        Returns:
            回测信号类型，None 表示不生成信号
        """
        mapping = {
            AAASignalType.STRONG_BUY: SignalType.BUY,
            AAASignalType.BUY: SignalType.BUY,
            AAASignalType.WEAK_BUY: SignalType.BUY,
            AAASignalType.HOLD: None,  # 不生成信号
            AAASignalType.WEAK_SELL: None,  # 不生成信号
            AAASignalType.SELL: SignalType.SELL,
            AAASignalType.STOP_LOSS: SignalType.SELL,
            AAASignalType.TAKE_PROFIT: SignalType.SELL,
            AAASignalType.TRAILING_STOP: SignalType.SELL
        }

        return mapping.get(aaa_signal_type)

    def _calculate_quantity(self, aaa_signal: AAASignal, symbol: str) -> int:
        """
        计算交易数量

        Args:
            aaa_signal: AAA 信号
            symbol: 股票代码

        Returns:
            交易数量（100股的整数倍）
        """
        if aaa_signal.signal_type in [AAASignalType.SELL, AAASignalType.STOP_LOSS,
                                       AAASignalType.TAKE_PROFIT, AAASignalType.TRAILING_STOP]:
            # 卖出信号：卖出全部持仓
            return self.get_position(symbol)

        # 买入信号：根据评分计算仓位
        score = aaa_signal.scores.total
        cash = self.get_cash()

        # 计算目标仓位金额
        if score >= 85:
            target_ratio = self.max_position_ratio
        elif score >= 80:
            target_ratio = self.max_position_ratio * 0.8
        elif score >= 70:
            target_ratio = self.max_position_ratio * 0.6
        elif score >= 60:
            target_ratio = self.max_position_ratio * 0.4
        elif score >= 55:
            target_ratio = self.max_position_ratio * 0.2
        else:
            return 0

        # 计算可买股数
        target_amount = cash * target_ratio
        quantity = int(target_amount / aaa_signal.price / 100) * 100

        return max(0, quantity)

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

        # 调用基类方法更新持仓和资金
        super().on_order(signal, executed_price, executed_quantity)

        # 更新入场信息
        if signal.is_buy():
            current_pos = self.get_position(symbol)
            if current_pos > 0:
                # 计算平均成本
                old_cost = self.entry_prices.get(symbol, 0) * self.entry_prices.get(symbol, 0)
                new_cost = executed_price * executed_quantity
                avg_cost = (old_cost + new_cost) / current_pos

                self.entry_prices[symbol] = avg_cost

                # 更新评分（加权平均）
                old_score = self.entry_scores.get(symbol, 0) * self.entry_scores.get(symbol, 0)
                new_score = signal.metadata.get('aaa_score', 0) * executed_quantity
                avg_score = (old_score + new_score) / current_pos
                self.entry_scores[symbol] = avg_score
            else:
                self.entry_prices[symbol] = executed_price
                self.entry_scores[symbol] = signal.metadata.get('aaa_score', 0)

            # 设置止损止盈
            self.stop_losses[symbol] = signal.metadata.get('stop_loss',
                    executed_price * (1 - self.stop_loss_pct))
            self.take_profits[symbol] = signal.metadata.get('take_profit',
                    executed_price * (1 + self.take_profit_pct))

        elif signal.is_sell():
            # 如果全部卖出，清空状态
            if self.get_position(symbol) == 0:
                self.entry_prices[symbol] = 0
                self.entry_scores[symbol] = 0
                self.stop_losses[symbol] = 0
                self.take_profits[symbol] = 0

    def get_score_history(self, symbol: str) -> List[AAAScores]:
        """获取评分历史"""
        return self.score_history.get(symbol, [])

    def get_latest_score(self, symbol: str) -> Optional[AAAScores]:
        """获取最新评分"""
        history = self.score_history.get(symbol)
        if history:
            return history[-1]
        return None

    def get_state_summary(self, symbol: str) -> Dict[str, Any]:
        """
        获取策略状态摘要

        Args:
            symbol: 股票代码

        Returns:
            状态摘要字典
        """
        latest_score = self.get_latest_score(symbol)

        return {
            'symbol': symbol,
            'position': self.get_position(symbol),
            'entry_price': self.entry_prices.get(symbol, 0),
            'entry_score': self.entry_scores.get(symbol, 0),
            'stop_loss': self.stop_losses.get(symbol, 0),
            'take_profit': self.take_profits.get(symbol, 0),
            'latest_score': latest_score.total if latest_score else 0,
            'latest_level': latest_score.level if latest_score else '-'
        }


if __name__ == '__main__':
    # 简单测试
    strategy = AAAStrategy()
    strategy.initialize(capital=100000, symbols=['sz002600'])

    print(f"策略名称: {strategy.name}")
    print(f"初始资金: {strategy.initial_capital}")
    print(f"买入阈值: {strategy.buy_threshold}")
    print(f"卖出阈值: {strategy.sell_threshold}")
