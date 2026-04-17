#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AAA 信号生成器
A-share Analysis All-in-One Signal Generator

基于 AAA 评分生成买卖信号，包含：
- 买入信号生成
- 卖出信号生成
- 仓位大小建议
- 止损止盈计算
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from indicators.aaa_scorer import AAAScores, AAAScorer


class AAASignalType(Enum):
    """AAA 信号类型"""
    STRONG_BUY = "strong_buy"      # 强买入 (score >= 85)
    BUY = "buy"                    # 买入 (70 <= score < 85)
    WEAK_BUY = "weak_buy"          # 弱买入 (55 <= score < 70)
    HOLD = "hold"                  # 持有 (45 <= score < 55)
    WEAK_SELL = "weak_sell"        # 弱卖出 (30 <= score < 45)
    SELL = "sell"                  # 卖出 (score < 30)
    STOP_LOSS = "stop_loss"        # 止损
    TAKE_PROFIT = "take_profit"    # 止盈
    TRAILING_STOP = "trailing_stop"  # 移动止损


@dataclass
class AAASignal:
    """AAA 交易信号"""
    signal_type: AAASignalType
    symbol: str
    datetime: datetime
    price: float
    scores: AAAScores
    reason: str
    confidence: float
    suggested_size: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class AAASignalGenerator:
    """
    AAA 信号生成器

    信号生成规则：
    - 买入: 评分 >= 70 且评分变化 >= 10
    - 卖出: 评分 <= 30 或评分下降 > 20 或触发止损
    """

    def __init__(
        self,
        buy_threshold: float = 70,
        sell_threshold: float = 30,
        min_score_change: float = 10,
        score_drop_threshold: float = 20,
        enable_score_change_filter: bool = True
    ):
        """
        初始化信号生成器

        Args:
            buy_threshold: 买入阈值
            sell_threshold: 卖出阈值
            min_score_change: 最小评分变化要求
            score_drop_threshold: 评分下降阈值（触发卖出）
            enable_score_change_filter: 是否启用评分变化过滤
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_score_change = min_score_change
        self.score_drop_threshold = score_drop_threshold
        self.enable_score_change_filter = enable_score_change_filter

        # 评分历史（用于检测变化）
        self.score_history: Dict[str, List[AAAScores]] = {}

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_bar: pd.Series,
        index: int,
        position: int,
        entry_price: Optional[float] = None,
        entry_score: Optional[float] = None
    ) -> Optional[AAASignal]:
        """
        生成交易信号

        Args:
            symbol: 股票代码
            df: 包含所有指标的数据
            current_bar: 当前K线
            index: 当前索引
            position: 当前持仓数量
            entry_price: 入场价格（如有持仓）
            entry_score: 入场时的评分（如有持仓）

        Returns:
            AAASignal 对象，如果没有信号则返回 None
        """
        # 计算当前评分
        scorer = AAAScorer()
        current_score = scorer.score(df)

        # 记录评分历史
        if symbol not in self.score_history:
            self.score_history[symbol] = []
        self.score_history[symbol].append(current_score)
        if len(self.score_history[symbol]) > 100:
            self.score_history[symbol] = self.score_history[symbol][-100:]

        # 获取前一次评分
        prev_score = self.score_history[symbol][-2] if len(self.score_history[symbol]) >= 2 else None

        # 有持仓时，检查卖出信号
        if position > 0:
            return self._check_sell_signal(
                symbol, current_bar, current_score, prev_score,
                position, entry_price, entry_score
            )

        # 无持仓时，检查买入信号
        return self._check_buy_signal(
            symbol, current_bar, current_score, prev_score
        )

    def _check_buy_signal(
        self,
        symbol: str,
        current_bar: pd.Series,
        current_score: AAAScores,
        prev_score: Optional[AAAScores]
    ) -> Optional[AAASignal]:
        """检查买入信号"""

        # 评分必须达到买入阈值
        if current_score.total < self.buy_threshold:
            return None

        # 检查评分变化要求
        if self.enable_score_change_filter and prev_score:
            score_change = current_score.total - prev_score.total
            if score_change < self.min_score_change:
                return None

        # 确定信号强度
        if current_score.total >= 85:
            signal_type = AAASignalType.STRONG_BUY
            reason = f"强买入信号: 评分 {current_score.total:.0f} >= 85"
        else:
            signal_type = AAASignalType.BUY
            reason = f"买入信号: 评分 {current_score.total:.0f} >= {self.buy_threshold}"

        # 建议仓位大小
        suggested_size = self._calculate_position_size(current_score.total)

        # 计算止损止盈
        stop_loss = self._calculate_stop_loss(current_bar['close'], current_score.total)
        take_profit = self._calculate_take_profit(current_bar['close'], stop_loss, current_score.total)

        return AAASignal(
            signal_type=signal_type,
            symbol=symbol,
            datetime=pd.to_datetime(current_bar.name) if hasattr(current_bar, 'name') else datetime.now(),
            price=float(current_bar['close']),
            scores=current_score,
            reason=reason,
            confidence=current_score.confidence,
            suggested_size=suggested_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _check_sell_signal(
        self,
        symbol: str,
        current_bar: pd.Series,
        current_score: AAAScores,
        prev_score: Optional[AAAScores],
        position: int,
        entry_price: Optional[float],
        entry_score: Optional[float]
    ) -> Optional[AAASignal]:
        """检查卖出信号"""

        current_price = current_bar['close']
        reason = ""

        # 1. 检查评分阈值
        if current_score.total <= self.sell_threshold:
            signal_type = AAASignalType.SELL
            reason = f"卖出信号: 评分 {current_score.total:.0f} <= {self.sell_threshold}"
        # 2. 检查评分大幅下降
        elif entry_score and prev_score:
            score_drop = entry_score - current_score.total
            if score_drop >= self.score_drop_threshold:
                signal_type = AAASignalType.SELL
                reason = f"卖出信号: 评分下降 {score_drop:.0f} 分"
        # 3. 检查止损
        elif entry_price and current_price <= entry_price * 0.95:
            signal_type = AAASignalType.STOP_LOSS
            reason = f"止损: 价格 {current_price:.2f} <= 入场价 {entry_price:.2f} * 0.95"
        # 4. 检查止盈
        elif entry_price and current_price >= entry_price * 1.10:
            signal_type = AAASignalType.TAKE_PROFIT
            reason = f"止盈: 价格 {current_price:.2f} >= 入场价 {entry_price:.2f} * 1.10"
        else:
            return None

        return AAASignal(
            signal_type=signal_type,
            symbol=symbol,
            datetime=pd.to_datetime(current_bar.name) if hasattr(current_bar, 'name') else datetime.now(),
            price=float(current_price),
            scores=current_score,
            reason=reason,
            confidence=current_score.confidence
        )

    def _calculate_position_size(
        self,
        score: float,
        max_position_ratio: float = 0.3
    ) -> int:
        """
        根据评分计算建议仓位大小

        评分与仓位映射：
        - 90-100: 100% 最大仓位
        - 80-89:  80% 最大仓位
        - 70-79:  60% 最大仓位
        - 60-69:  40% 最大仓位
        - 55-59:  20% 最大仓位

        Args:
            score: AAA 评分
            max_position_ratio: 最大仓位比例（相对于总资金）

        Returns:
            建议股数（100股的整数倍）
        """
        if score >= 90:
            position_ratio = 1.0
        elif score >= 80:
            position_ratio = 0.8
        elif score >= 70:
            position_ratio = 0.6
        elif score >= 60:
            position_ratio = 0.4
        elif score >= 55:
            position_ratio = 0.2
        else:
            return 0

        # 实际仓位比例
        actual_ratio = max_position_ratio * position_ratio

        # 返回比例表示
        return actual_ratio

    def _calculate_stop_loss(
        self,
        entry_price: float,
        score: float,
        default_stop_pct: float = 0.05
    ) -> float:
        """
        计算止损价格

        评分越高，止损越紧

        Args:
            entry_price: 入场价格
            score: AAA 评分
            default_stop_pct: 默认止损比例

        Returns:
            止损价格
        """
        if score >= 85:
            stop_pct = default_stop_pct * 0.8  # 更紧
        elif score >= 70:
            stop_pct = default_stop_pct
        else:
            stop_pct = default_stop_pct * 1.2  # 更宽

        return entry_price * (1 - stop_pct)

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        score: float,
        default_rr_ratio: float = 2.0
    ) -> float:
        """
        计算止盈价格

        使用盈亏比计算

        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            score: AAA 评分
            default_rr_ratio: 默认盈亏比

        Returns:
            止盈价格
        """
        risk = entry_price - stop_loss

        if score >= 85:
            rr_ratio = default_rr_ratio * 1.5  # 更高目标
        elif score >= 70:
            rr_ratio = default_rr_ratio
        else:
            rr_ratio = default_rr_ratio * 0.8

        return entry_price + risk * rr_ratio

    def reset_history(self, symbol: Optional[str] = None):
        """
        重置评分历史

        Args:
            symbol: 指定股票代码，None 表示重置全部
        """
        if symbol:
            self.score_history.pop(symbol, None)
        else:
            self.score_history.clear()


if __name__ == '__main__':
    # 测试代码
    import sys
    sys.path.insert(0, 'D:/新建文件夹/claude')

    from indicators.aaa_indicators import AAAIndicatorEngine
    from data.akshare_source import AKShareSource
    from datetime import datetime

    # 获取测试数据
    source = AKShareSource()
    df = source.get_kline('sz002600', start_date=datetime(2024, 1, 1))

    # 计算指标
    engine = AAAIndicatorEngine(df)
    df = engine.calculate_all()

    # 创建信号生成器
    generator = AAASignalGenerator()

    # 模拟生成信号
    for i in range(60, len(df)):
        current_bar = df.iloc[i]
        hist_df = df.iloc[:i+1]

        signal = generator.generate_signal(
            symbol='sz002600',
            df=hist_df,
            current_bar=current_bar,
            index=i,
            position=0
        )

        if signal:
            print(f"\n{i}: {signal.datetime}")
            print(f"  类型: {signal.signal_type.value}")
            print(f"  价格: {signal.price:.2f}")
            print(f"  评分: {signal.scores.total:.0f}")
            print(f"  原因: {signal.reason}")
            if signal.stop_loss:
                print(f"  止损: {signal.stop_loss:.2f}")
            if signal.take_profit:
                print(f"  止盈: {signal.take_profit:.2f}")
