"""
日频组合管理模块

桥接 signal_engine 的目标仓位输出与回测引擎:
1. 接收 signal_engine 输出的目标仓位权重 (如 0.15 = 15%)
2. _align(): 仓位归一化 (仅当 sum(|pos|) > 1 时才缩放)
3. _portfolio_equity(): 组合净值计算 (A股非对称佣金模型)

关键设计:
- signal_engine 输出绝对目标权重 (0=空仓, 0.10~0.30=持仓权重)
- _align() 保留 signal_engine 的绝对权重语义, 不做无条件归一化
- _portfolio_equity() 区分买入/卖出佣金, 模拟A股印花税
"""

from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class CommissionConfig:
    """A股佣金配置

    买入: 佣金(万2.5) + 过户费(万0.1) + 证管费(万0.2) ≈ 0.028%
          简化为 commission_buy = 0.0003 (万3)
    卖出: 佣金(万2.5) + 过户费(万0.1) + 证管费(万0.2) + 印花税(千1) ≈ 0.128%
          简化为 commission_sell = 0.0013 (千1.3)
    """
    commission_buy: float = 0.0003    # 买入综合费率 (万3)
    commission_sell: float = 0.0013   # 卖出综合费率 (千1.3, 含印花税千1)
    min_commission: float = 5.0       # 最低佣金 (元)


class DailyPortfolio:
    """
    日频组合管理器

    接收 signal_engine 的目标仓位信号, 管理组合的逐日再平衡:
    1. 信号对齐: _align() 确保仓位权重之和不超过1
    2. 仓位调整: 计算目标权重与当前权重的差异, 生成交易
    3. 净值计算: _portfolio_equity() 使用A股非对称佣金模型

    使用示例:
        portfolio = DailyPortfolio(initial_capital=1_000_000)
        # signal_engine 输出: {'sh600519': 0.15, 'sz000858': 0.15, 'sh600036': 0.15}
        aligned = portfolio.align_positions(target_weights)
        # aligned = {'sh600519': 0.15, 'sz000858': 0.15, 'sh600036': 0.15}
        # (sum=0.45 <= 1, 不归一化)

        # 当 sum > 1 时才缩放:
        # target = {'A': 0.5, 'B': 0.4, 'C': 0.3}  sum=1.2 > 1
        # aligned = {'A': 0.417, 'B': 0.333, 'C': 0.250}  缩放到 sum=1
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_config: Optional[CommissionConfig] = None,
        position_limit: float = 0.95,  # 仓位上限 (留5%现金)
    ):
        self.initial_capital = initial_capital
        self.commission = commission_config or CommissionConfig()
        self.position_limit = position_limit

        # 当前状态
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}   # symbol -> market_value
        self.weights: Dict[str, float] = {}     # symbol -> weight

    def align_positions(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """对齐目标仓位 (公共接口)

        调用 _align() 进行仓位归一化, 仅在 sum(|w|) > 1 时缩放.

        Args:
            target_weights: signal_engine 输出的目标仓位权重
                {symbol: weight}, 如 {'sh600519': 0.15, 'sz000858': 0.20}

        Returns:
            对齐后的仓位权重, 保持 signal_engine 的绝对权重语义
        """
        pos = pd.Series(target_weights, dtype=float)
        aligned = self._align(pos)
        return aligned.to_dict()

    def _align(self, pos: pd.Series) -> pd.Series:
        """仓位归一化

        修复: 仅在 sum(|pos|) > 1 时才做归一化缩放.
        当 sum(|pos|) <= 1 时, 直接返回原始仓位, 保留 signal_engine 的绝对权重语义.

        原问题:
            旧逻辑: pos = pos.div(pos.abs().sum())  # 无条件归一化
            当3只股同时持仓0.15时, 归一化后每只变0.33, 风控逻辑被框架覆盖.

        修复后:
            sum(0.15, 0.15, 0.15) = 0.45 <= 1 → 不归一化, 每只保持0.15
            sum(0.5, 0.4, 0.3) = 1.2 > 1 → 缩放到 0.417, 0.333, 0.250

        Args:
            pos: 目标仓位 pd.Series, index=symbol, values=权重

        Returns:
            对齐后的仓位 pd.Series
        """
        if pos.empty:
            return pos

        scale = pos.abs().sum()

        if scale <= 1.0:
            # sum(|w|) <= 1: 不需要归一化, 保留绝对权重
            return pos
        else:
            # sum(|w|) > 1: 需要等比缩放, 使 sum = 1
            logger.debug(
                f"仓位归一化: sum(|pos|)={scale:.4f} > 1, "
                f"缩放比例={1.0/scale:.4f}"
            )
            return pos.div(scale)

    def _portfolio_equity(
        self,
        prev_equity: float,
        prev_weights: Dict[str, float],
        target_weights: Dict[str, float],
        price_returns: Dict[str, float],
    ) -> float:
        """计算组合净值 (A股非对称佣金模型)

        修复: 区分买入和卖出的佣金率, 模拟A股印花税.

        A股费用结构:
        - 买入: 佣金(万2.5) + 过户费(万0.1) ≈ 0.03%
        - 卖出: 佣金(万2.5) + 过户费(万0.1) + 印花税(千1) ≈ 0.13%

        佣金 = 仓位变化量 × 对应费率
        - 仓位增加(买入): 用 commission_buy
        - 仓位减少(卖出): 用 commission_sell

        Args:
            prev_equity: 前一日组合净值
            prev_weights: 前一日持仓权重 {symbol: weight}
            target_weights: 当日目标权重 (已对齐) {symbol: weight}
            price_returns: 当日各股票收益率 {symbol: return}

        Returns:
            当日组合净值
        """
        # 1. 计算持仓市值变化 (价格变动带来的收益)
        portfolio_return = 0.0
        for symbol, weight in prev_weights.items():
            ret = price_returns.get(symbol, 0.0)
            portfolio_return += weight * ret

        # 2. 计算再平衡交易成本 (非对称佣金)
        rebalance_cost = 0.0
        all_symbols = set(prev_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            prev_w = prev_weights.get(symbol, 0.0)
            # 目标权重需要先经过价格调整
            # 如果之前持有, 目标权重已被价格变动影响
            if symbol in prev_weights and symbol in price_returns:
                adjusted_prev = prev_w * (1 + price_returns[symbol])
            else:
                adjusted_prev = prev_w

            target_w = target_weights.get(symbol, 0.0)
            weight_change = target_w - adjusted_prev

            if abs(weight_change) < 1e-8:
                continue

            trade_value = abs(weight_change) * prev_equity

            if weight_change > 0:
                # 仓位增加 = 买入, 使用买入费率
                cost = trade_value * self.commission.commission_buy
            else:
                # 仓位减少 = 卖出, 使用卖出费率 (含印花税)
                cost = trade_value * self.commission.commission_sell

            # 最低佣金检查
            cost = max(cost, self.commission.min_commission) if trade_value > 0 else 0.0
            rebalance_cost += cost

        # 3. 计算净值
        cost_pct = rebalance_cost / prev_equity if prev_equity > 0 else 0.0
        current_equity = prev_equity * (1 + portfolio_return - cost_pct)

        return current_equity

    def update_positions(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """更新持仓到目标权重

        Args:
            target_weights: 目标权重 (已对齐)
            prices: 当前价格 {symbol: price}

        Returns:
            实际交易后的权重
        """
        total_value = self.cash + sum(self.positions.values())

        # 计算每只股票的目标市值
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = total_value * weight

        # 执行交易
        for symbol, target_val in target_values.items():
            current_val = self.positions.get(symbol, 0.0)
            diff = target_val - current_val

            if symbol not in prices:
                continue

            price = prices[symbol]
            if price <= 0:
                continue

            if diff > 0:
                # 买入
                cost = diff + diff * self.commission.commission_buy
                cost = max(cost, diff + self.commission.min_commission) if diff > 0 else 0
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = target_val
            elif diff < 0:
                # 卖出
                sell_value = abs(diff)
                proceeds = sell_value - sell_value * self.commission.commission_sell
                proceeds = max(proceeds, sell_value - self.commission.min_commission) if sell_value > 0 else 0
                self.cash += proceeds
                if target_val < 1.0:  # 接近0, 清仓
                    del self.positions[symbol]
                else:
                    self.positions[symbol] = target_val

        # 更新权重
        total_value = self.cash + sum(self.positions.values())
        self.weights = {
            sym: val / total_value
            for sym, val in self.positions.items()
            if total_value > 0
        }

        return self.weights
