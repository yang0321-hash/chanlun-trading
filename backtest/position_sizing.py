"""
仓位管理模块

提供多种仓位计算方法：
1. FixedPercent - 固定百分比
2. RiskParity - 风险平价 (基于ATR)
3. Kelly - 凯利公式
4. VolatilityTarget - 波动率目标
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class PositionResult:
    """仓位计算结果"""
    quantity: int          # 建议数量
    shares: int            # 股数
    reason: str            # 原因说明
    risk_amount: float     # 风险金额
    position_value: float  # 仓位价值
    risk_pct: float        # 风险百分比


class PositionSizer:
    """仓位管理基类"""

    def __init__(
        self,
        initial_capital: float,
        min_unit: int = 100,
        max_position_pct: float = 0.30,  # 单股最大30%
    ):
        self.initial_capital = initial_capital
        self.min_unit = min_unit
        self.max_position_pct = max_position_pct

    def calculate(
        self,
        price: float,
        cash: float,
        atr: Optional[float] = None,
        stop_price: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss: Optional[float] = None,
        **kwargs
    ) -> PositionResult:
        """计算仓位"""
        raise NotImplementedError


class FixedPercentSizer(PositionSizer):
    """固定百分比仓位管理器"""

    def __init__(
        self,
        initial_capital: float,
        position_pct: float = 0.95,
        min_unit: int = 100,
        max_position_pct: float = 0.30,
    ):
        super().__init__(initial_capital, min_unit, max_position_pct)
        self.position_pct = position_pct

    def calculate(
        self,
        price: float,
        cash: float,
        atr: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> PositionResult:
        """固定百分比计算"""
        # 使用现金的固定比例
        available_cash = cash * self.position_pct

        # 最大仓位限制
        max_value = self.initial_capital * self.max_position_pct
        available_cash = min(available_cash, max_value)

        shares = int(available_cash / price / self.min_unit) * self.min_unit

        return PositionResult(
            quantity=shares,
            shares=shares,
            reason=f"固定{self.position_pct*100:.0f}%仓位",
            risk_amount=shares * price * 0.05,  # 假设5%止损
            position_value=shares * price,
            risk_pct=0.05
        )


class RiskParitySizer(PositionSizer):
    """
    风险平价仓位管理器

    基于ATR计算仓位，使每笔交易的风险金额相等
    """

    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = 0.02,  # 每笔交易风险2%
        atr_period: int = 14,
        atr_multiplier: float = 2.0,   # 止损距离 = ATR * multiplier
        min_unit: int = 100,
        max_position_pct: float = 0.30,
    ):
        super().__init__(initial_capital, min_unit, max_position_pct)
        self.risk_per_trade = risk_per_trade
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate(
        self,
        price: float,
        cash: float,
        atr: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> PositionResult:
        """基于ATR风险平价计算"""
        if atr is None or atr <= 0:
            # 回退到固定比例
            return PositionResult(
                quantity=0,
                shares=0,
                reason="ATR无效",
                risk_amount=0,
                position_value=0,
                risk_pct=0
            )

        # 止损距离
        stop_distance = atr * self.atr_multiplier

        # 风险金额
        risk_amount = self.initial_capital * self.risk_per_trade

        # 根据风险金额计算股数
        shares_by_risk = int(risk_amount / stop_distance / self.min_unit) * self.min_unit

        # 根据可用现金计算
        shares_by_cash = int(cash * 0.95 / price / self.min_unit) * self.min_unit

        # 取较小值
        shares = min(shares_by_risk, shares_by_cash)

        # 最大仓位限制
        max_shares = int(self.initial_capital * self.max_position_pct / price / self.min_unit) * self.min_unit
        shares = min(shares, max_shares)

        actual_risk = shares * stop_distance / self.initial_capital

        return PositionResult(
            quantity=shares,
            shares=shares,
            reason=f"风险平价 ATR={atr:.2f} 止损距离={stop_distance:.2f}",
            risk_amount=shares * stop_distance,
            position_value=shares * price,
            risk_pct=actual_risk
        )


class KellySizer(PositionSizer):
    """
    凯利公式仓位管理器

    f = (p*b - q) / b
    其中:
    - f = 仓位比例
    - p = 胜率
    - q = 败率 = 1-p
    - b = 盈亏比
    """

    def __init__(
        self,
        initial_capital: float,
        win_rate: float = 0.45,      # 默认胜率45%
        profit_loss_ratio: float = 2.0,  # 默认盈亏比2:1
        kelly_fraction: float = 0.5, # 半凯利（更保守）
        min_unit: int = 100,
        max_position_pct: float = 0.30,
    ):
        super().__init__(initial_capital, min_unit, max_position_pct)
        self.win_rate = win_rate
        self.profit_loss_ratio = profit_loss_ratio
        self.kelly_fraction = kelly_fraction

    def calculate(
        self,
        price: float,
        cash: float,
        atr: Optional[float] = None,
        stop_price: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss: Optional[float] = None,
        **kwargs
    ) -> PositionResult:
        """凯利公式计算"""
        # 使用传入的胜率或默认值
        p = win_rate if win_rate is not None else self.win_rate
        q = 1 - p
        b = avg_win_loss if avg_win_loss is not None else self.profit_loss_ratio

        # 凯利公式
        if b <= 0:
            kelly_pct = 0
        else:
            kelly_pct = (p * b - q) / b

        # 半凯利（更保守）
        kelly_pct = kelly_pct * self.kelly_fraction

        # 限制在合理范围
        kelly_pct = max(0, min(kelly_pct, 0.50))  # 最多50%

        # 计算股数
        max_value = self.initial_capital * min(kelly_pct, self.max_position_pct)
        shares = int(max_value / price / self.min_unit) * self.min_unit

        # 不能超过可用现金
        shares = min(shares, int(cash * 0.95 / price / self.min_unit) * self.min_unit)

        # 预期风险（假设止损为凯利亏损）
        expected_loss = max_value * q if kelly_pct > 0 else 0

        return PositionResult(
            quantity=shares,
            shares=shares,
            reason=f"凯利公式 p={p:.2%} b={b:.2f} kelly={kelly_pct:.2%}",
            risk_amount=expected_loss,
            position_value=shares * price,
            risk_pct=kelly_pct
        )


class VolatilityTargetSizer(PositionSizer):
    """
    波动率目标仓位管理器

    根据历史波动率调整仓位：
    - 低波动 → 加大仓位
    - 高波动 → 减小仓位
    """

    def __init__(
        self,
        initial_capital: float,
        target_volatility: float = 0.15,  # 目标年化波动率15%
        lookback_period: int = 60,        # 回看60天
        min_unit: int = 100,
        max_position_pct: float = 0.30,
    ):
        super().__init__(initial_capital, min_unit, max_position_pct)
        self.target_volatility = target_volatility
        self.lookback_period = lookback_period

    def calculate(
        self,
        price: float,
        cash: float,
        historical_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> PositionResult:
        """波动率目标计算"""
        if historical_data is None or len(historical_data) < self.lookback_period:
            # 无历史数据，使用默认50%仓位
            target_pct = 0.50
        else:
            # 计算历史波动率
            recent = historical_data.tail(self.lookback_period)
            returns = recent['close'].pct_change().dropna()

            if len(returns) < 10:
                target_pct = 0.50
            else:
                # 年化波动率
                hist_vol = returns.std() * np.sqrt(252)

                # 目标仓位 = 目标波动率 / 实际波动率
                if hist_vol > 0:
                    target_pct = self.target_volatility / hist_vol
                else:
                    target_pct = 1.0

                # 限制范围
                target_pct = max(0.20, min(target_pct, 1.0))

        # 计算股数
        max_value = self.initial_capital * min(target_pct, self.max_position_pct)
        shares = int(max_value / price / self.min_unit) * self.min_unit

        # 不能超过可用现金
        shares = min(shares, int(cash * 0.95 / price / self.min_unit) * self.min_unit)

        return PositionResult(
            quantity=shares,
            shares=shares,
            reason=f"波动率目标 目标={target_pct:.0%}仓位",
            risk_amount=shares * price * 0.05,
            position_value=shares * price,
            risk_pct=0.05
        )


class AdaptivePositionSizer(PositionSizer):
    """
    自适应仓位管理器

    结合多种方法，根据市场状态自动选择：
    - 趋势市 → 风险平价
    - 震荡市 → 固定百分比（保守）
    - 高波动 → 波动率目标
    """

    def __init__(
        self,
        initial_capital: float,
        min_unit: int = 100,
        max_position_pct: float = 0.30,
        risk_per_trade: float = 0.02,
    ):
        super().__init__(initial_capital, min_unit, max_position_pct)

        # 创建子管理器
        self.risk_parity = RiskParitySizer(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            min_unit=min_unit,
            max_position_pct=max_position_pct,
        )
        self.fixed = FixedPercentSizer(
            initial_capital=initial_capital,
            position_pct=0.70,  # 震荡市用70%
            min_unit=min_unit,
            max_position_pct=max_position_pct,
        )
        self.vol_target = VolatilityTargetSizer(
            initial_capital=initial_capital,
            min_unit=min_unit,
            max_position_pct=max_position_pct,
        )

    def calculate(
        self,
        price: float,
        cash: float,
        atr: Optional[float] = None,
        historical_data: Optional[pd.DataFrame] = None,
        trend_strength: float = 0.5,  # 0-1, >0.7为强趋势
        **kwargs
    ) -> PositionResult:
        """自适应计算"""

        # 判断市场状态
        if trend_strength > 0.7:
            # 强趋势：使用风险平价
            result = self.risk_parity.calculate(
                price=price, cash=cash, atr=atr, **kwargs
            )
            result.reason = f"[趋势] {result.reason}"

        elif trend_strength < 0.3:
            # 震荡市：使用固定比例（保守）
            result = self.fixed.calculate(
                price=price, cash=cash, **kwargs
            )
            result.reason = f"[震荡] {result.reason}"

        else:
            # 中等：根据ATR选择
            if atr and atr > 0:
                result = self.risk_parity.calculate(
                    price=price, cash=cash, atr=atr, **kwargs
                )
                result.reason = f"[中等] {result.reason}"
            else:
                result = self.fixed.calculate(
                    price=price, cash=cash, **kwargs
                )
                result.reason = f"[中等] {result.reason}"

        return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """计算ATR"""
    if len(df) < period + 1:
        return 0

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

    return np.mean(tr_list[-period:])


def calculate_volatility(df: pd.DataFrame, period: int = 60) -> float:
    """计算年化波动率"""
    if len(df) < period:
        period = len(df) // 2

    if period < 10:
        return 0.20  # 默认20%

    returns = df['close'].tail(period).pct_change().dropna()
    if len(returns) < 5:
        return 0.20

    return returns.std() * np.sqrt(252)
