"""
组合管理模块

提供多股票组合管理功能：
1. 行业分散度控制
2. 相关性限制
3. 动态再平衡
4. 风险归因分析
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    shares: int
    entry_price: float
    current_price: float
    market_value: float
    weight: float  # 占比
    entry_date: datetime
    industry: str = ""
    sector: str = ""

    @property
    def pnl(self) -> float:
        """浮动盈亏"""
        return (self.current_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        """浮动盈亏比例"""
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass
class PortfolioMetrics:
    """组合指标"""
    total_value: float
    cash: float
    total_position_value: float
    total_pnl: float
    daily_pnl: float
    daily_pnl_pct: float
    weights: Dict[str, float]
    industry_weights: Dict[str, float]
    sector_weights: Dict[str, float]
    concentration: float  # 集中度 (HHI)


class PortfolioManager:
    """
    组合管理器

    管理多股票投资组合的构建、调整和风险控制
    """

    def __init__(
        self,
        initial_capital: float,
        max_positions: int = 10,         # 最大持仓数
        max_single_weight: float = 0.20,  # 单股最大权重
        max_industry_weight: float = 0.40, # 单行业最大权重
        max_sector_weight: float = 0.60,   # 单板块最大权重
        min_unit: int = 100,
        rebalance_threshold: float = 0.10, # 再平衡阈值
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        self.max_industry_weight = max_industry_weight
        self.max_sector_weight = max_sector_weight
        self.min_unit = min_unit
        self.rebalance_threshold = rebalance_threshold

        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.industry_map: Dict[str, str] = {}  # symbol -> industry
        self.sector_map: Dict[str, str] = {}    # symbol -> sector

        # 历史记录
        self.trades: List[dict] = []
        self.daily_values: List[Tuple[datetime, float]] = []

    def set_industry_mapping(self, mapping: Dict[str, Tuple[str, str]]):
        """
        设置行业映射

        Args:
            mapping: {symbol: (industry, sector)}
        """
        for symbol, (industry, sector) in mapping.items():
            self.industry_map[symbol] = industry
            self.sector_map[symbol] = sector

    def get_industry(self, symbol: str) -> str:
        """获取股票行业"""
        return self.industry_map.get(symbol, "未知")

    def get_sector(self, symbol: str) -> str:
        """获取股票板块"""
        return self.sector_map.get(symbol, "未知")

    def can_buy(
        self,
        symbol: str,
        price: float,
        shares: int
    ) -> Tuple[bool, str]:
        """
        检查是否可以买入

        Returns:
            (是否可以, 原因)
        """
        # 1. 检查持仓数量限制
        if symbol not in self.positions and len(self.positions) >= self.max_positions:
            return False, f"已达最大持仓数 {self.max_positions}"

        # 2. 检查资金
        required = price * shares
        if required > self.cash:
            return False, f"资金不足 需要{required:.2f} 可用{self.cash:.2f}"

        # 3. 计算买入后的权重
        current_weight = 0
        if symbol in self.positions:
            current_weight = self.positions[symbol].market_value / self.get_total_value()

        additional_value = price * shares
        new_weight = (self.positions[symbol].market_value + additional_value) / self.get_total_value() \
            if symbol in self.positions else additional_value / self.get_total_value()

        # 4. 检查单股权重
        if new_weight > self.max_single_weight:
            return False, f"单股权重超限 {new_weight:.2%} > {self.max_single_weight:.2%}"

        # 5. 检查行业权重
        industry = self.get_industry(symbol)
        industry_value = sum(
            p.market_value for p in self.positions.values()
            if self.get_industry(p.symbol) == industry
        )
        if symbol in self.positions:
            industry_value += self.positions[symbol].market_value

        new_industry_weight = (industry_value + additional_value) / self.get_total_value()
        if new_industry_weight > self.max_industry_weight:
            return False, f"行业权重超限 {new_industry_weight:.2%} > {self.max_industry_weight:.2%}"

        # 6. 检查板块权重
        sector = self.get_sector(symbol)
        sector_value = sum(
            p.market_value for p in self.positions.values()
            if self.get_sector(p.symbol) == sector
        )
        if symbol in self.positions:
            sector_value += self.positions[symbol].market_value

        new_sector_weight = (sector_value + additional_value) / self.get_total_value()
        if new_sector_weight > self.max_sector_weight:
            return False, f"板块权重超限 {new_sector_weight:.2%} > {self.max_sector_weight:.2%}"

        return True, "可以买入"

    def buy(
        self,
        symbol: str,
        price: float,
        shares: int,
        date: datetime,
        reason: str = ""
    ) -> bool:
        """买入"""
        can_buy, msg = self.can_buy(symbol, price, shares)
        if not can_buy:
            logger.warning(f"[{symbol}] {msg}")
            return False

        # 扣减资金
        cost = price * shares
        self.cash -= cost

        # 更新持仓
        if symbol in self.positions:
            # 加仓 - 计算新的平均成本
            old_pos = self.positions[symbol]
            total_shares = old_pos.shares + shares
            avg_price = (old_pos.entry_price * old_pos.shares + price * shares) / total_shares

            self.positions[symbol].shares = total_shares
            self.positions[symbol].entry_price = avg_price
            self.positions[symbol].current_price = price
            self.positions[symbol].market_value = total_shares * price
        else:
            # 新建持仓
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                current_price=price,
                market_value=shares * price,
                weight=0,
                entry_date=date,
                industry=self.get_industry(symbol),
                sector=self.get_sector(symbol),
            )

        # 记录交易
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'buy',
            'price': price,
            'shares': shares,
            'value': cost,
            'reason': reason,
        })

        # 更新权重
        self._update_weights()

        logger.info(f"[{symbol}] 买入 {shares}股 @ {price:.2f}")

        return True

    def sell(
        self,
        symbol: str,
        price: float,
        shares: Optional[int] = None,
        date: datetime = None,
        reason: str = ""
    ) -> bool:
        """卖出"""
        if symbol not in self.positions:
            logger.warning(f"[{symbol}] 无持仓")
            return False

        position = self.positions[symbol]

        # 默认全部卖出
        if shares is None:
            shares = position.shares

        if shares > position.shares:
            shares = position.shares

        # 增加资金
        proceeds = price * shares
        self.cash += proceeds

        # 更新持仓
        position.shares -= shares
        position.current_price = price
        position.market_value = position.shares * price

        if position.shares == 0:
            del self.positions[symbol]

        # 记录交易
        self.trades.append({
            'date': date or datetime.now(),
            'symbol': symbol,
            'action': 'sell',
            'price': price,
            'shares': shares,
            'value': proceeds,
            'reason': reason,
        })

        # 更新权重
        self._update_weights()

        logger.info(f"[{symbol}] 卖出 {shares}股 @ {price:.2f}")

        return True

    def get_total_value(self) -> float:
        """获取总资产"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    def get_position_value(self) -> float:
        """获取持仓市值"""
        return sum(p.market_value for p in self.positions.values())

    def _update_weights(self):
        """更新权重"""
        total_value = self.get_total_value()

        for pos in self.positions.values():
            pos.weight = pos.market_value / total_value if total_value > 0 else 0

    def update_prices(self, prices: Dict[str, float]):
        """更新价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
                self.positions[symbol].market_value = price * self.positions[symbol].shares

        self._update_weights()

    def get_metrics(self) -> PortfolioMetrics:
        """获取组合指标"""
        total_value = self.get_total_value()
        position_value = self.get_position_value()

        # 计算盈亏
        total_pnl = sum(p.pnl for p in self.positions.values())

        # 计算权重
        weights = {symbol: pos.weight for symbol, pos in self.positions.items()}

        # 计算行业权重
        industry_weights = defaultdict(float)
        for pos in self.positions.values():
            industry_weights[pos.industry] += pos.weight

        # 计算板块权重
        sector_weights = defaultdict(float)
        for pos in self.positions.values():
            sector_weights[pos.sector] += pos.weight

        # 计算集中度 (HHI)
        hhi = sum(w ** 2 for w in weights.values())

        # 计算日盈亏
        daily_pnl = 0
        daily_pnl_pct = 0
        if len(self.daily_values) >= 2:
            prev_value = self.daily_values[-1][1]
            daily_pnl = total_value - prev_value
            daily_pnl_pct = daily_pnl / prev_value if prev_value > 0 else 0

        return PortfolioMetrics(
            total_value=total_value,
            cash=self.cash,
            total_position_value=position_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weights=weights,
            industry_weights=dict(industry_weights),
            sector_weights=dict(sector_weights),
            concentration=hhi,
        )

    def get_rebalance_suggestions(self) -> List[dict]:
        """
        获取再平衡建议

        Returns:
            建议列表 [{'symbol':, 'action':, 'target_weight':, 'current_weight':}]
        """
        suggestions = []
        metrics = self.get_metrics()

        for symbol, pos in self.positions.items():
            target_weight = 1.0 / len(self.positions)  # 等权重
            current_weight = pos.weight

            # 偏离超过阈值
            if abs(current_weight - target_weight) > self.rebalance_threshold:
                if current_weight > target_weight:
                    suggestions.append({
                        'symbol': symbol,
                        'action': 'reduce',
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'reduce_to': target_weight,
                    })
                else:
                    suggestions.append({
                        'symbol': symbol,
                        'action': 'increase',
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'increase_to': target_weight,
                    })

        return suggestions

    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float]):
        """
        执行再平衡

        Args:
            target_weights: 目标权重 {symbol: weight}
            prices: 当前价格
        """
        total_value = self.get_total_value()
        suggestions = []

        for symbol, target_weight in target_weights.items():
            if symbol not in self.positions:
                continue

            current_value = self.positions[symbol].market_value
            target_value = total_value * target_weight
            diff = target_value - current_value

            if abs(diff) / total_value > 0.05:  # 偏离超过5%
                price = prices.get(symbol, self.positions[symbol].current_price)

                if diff > 0:
                    # 需要加仓
                    shares = int(diff / price / self.min_unit) * self.min_unit
                    suggestions.append(('buy', symbol, shares))
                else:
                    # 需要减仓
                    shares = int(abs(diff) / price / self.min_unit) * self.min_unit
                    suggestions.append(('sell', symbol, shares))

        return suggestions

    def calculate_correlation(
        self,
        returns_data: Dict[str, pd.Series],
        period: int = 60
    ) -> pd.DataFrame:
        """
        计算持仓相关性

        Args:
            returns_data: {symbol: return_series}
            period: 计算周期
        """
        symbols = list(self.positions.keys())

        if len(symbols) < 2:
            return pd.DataFrame()

        # 获取各股票的收益率
        series_dict = {}
        for symbol in symbols:
            if symbol in returns_data:
                series_dict[symbol] = returns_data[symbol].tail(period)

        if len(series_dict) < 2:
            return pd.DataFrame()

        # 构建收益率矩阵
        df = pd.DataFrame(series_dict)

        # 计算相关系数
        return df.corr()

    def get_risk_attribution(
        self,
        returns_data: Dict[str, pd.Series],
        period: int = 60
    ) -> Dict[str, float]:
        """
        风险归因分析

        计算各持仓对组合风险的贡献
        """
        metrics = self.get_metrics()
        corr_matrix = self.calculate_correlation(returns_data, period)

        if corr_matrix.empty or len(corr_matrix) < 2:
            return {symbol: weight for symbol, weight in metrics.weights.items()}

        weights = np.array([metrics.weights.get(s, 0) for s in corr_matrix.columns])
        cov_matrix = corr_matrix.values  # 简化：直接用相关系数

        # 组合方差 = w' * Σ * w
        portfolio_var = weights @ cov_matrix @ weights.T

        # 边际风险贡献 = (Σ * w)_i / portfolio_var
        marginal_contrib = (cov_matrix @ weights.T) / portfolio_var if portfolio_var > 0 else np.zeros_like(weights)

        # 风险贡献 = weight_i * marginal_contrib_i
        risk_contrib = weights * marginal_contrib

        return dict(zip(corr_matrix.columns, risk_contrib.tolist()))


def generate_portfolio_report(manager: PortfolioManager) -> str:
    """生成组合报告"""
    metrics = manager.get_metrics()

    report = f"""
=== 组合报告 ===

总资产: ¥{metrics.total_value:,.2f}
  现金: ¥{metrics.cash:,.2f} ({metrics.cash/metrics.total_value:.1%})
  持仓: ¥{metrics.total_position_value:,.2f} ({metrics.total_position_value/metrics.total_value:.1%})

盈亏: ¥{metrics.total_pnl:,.2f}
日盈亏: ¥{metrics.daily_pnl:,.2f} ({metrics.daily_pnl_pct:.2%})

持仓明细:
"""

    for pos in sorted(manager.positions.values(), key=lambda p: p.market_value, reverse=True):
        report += f"  {pos.symbol}: {pos.shares}股 @ {pos.entry_price:.2f} "
        report += f"市值¥{pos.market_value:,.2f} ({pos.weight:.1%}) "
        report += f"盈亏{pos.pnl:+,.2f} ({pos.pnl_pct:+.2%}) "
        report += f"[{pos.industry}]\n"

    report += f"\n行业分布:\n"
    for industry, weight in sorted(metrics.industry_weights.items(), key=lambda x: -x[1]):
        report += f"  {industry}: {weight:.1%}\n"

    report += f"\n板块分布:\n"
    for sector, weight in sorted(metrics.sector_weights.items(), key=lambda x: -x[1]):
        report += f"  {sector}: {weight:.1%}\n"

    report += f"\n集中度 (HHI): {metrics.concentration:.4f}\n"

    return report


class SectorRotationStrategy:
    """
    行业轮动策略

    基于行业强度进行配置
    """

    def __init__(
        self,
        lookback_period: int = 20,
        top_n_sectors: int = 3,
    ):
        self.lookback_period = lookback_period
        self.top_n_sectors = top_n_sectors

    def calculate_sector_strength(
        self,
        sector_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        计算行业强度

        Returns:
            {sector: strength_score}
        """
        strength = {}

        for sector, returns in sector_returns.items():
            if len(returns) < self.lookback_period:
                continue

            recent_returns = returns.tail(self.lookback_period)

            # 综合得分：
            # 1. 累计收益率 (40%)
            # 2. 夏普比率 (30%)
            # 3. 胜率 (30%)
            cum_return = (1 + recent_returns).prod() - 1
            mean_return = recent_returns.mean()
            std_return = recent_returns.std()
            sharpe = mean_return / std_return if std_return > 0 else 0
            win_rate = (recent_returns > 0).sum() / len(recent_returns)

            score = (
                cum_return * 0.4 +
                sharpe * 0.3 +
                win_rate * 0.3
            )

            strength[sector] = score

        return strength

    def get_allocation(
        self,
        sector_strength: Dict[str, float]
    ) -> Dict[str, float]:
        """
        获取行业配置建议

        Returns:
            {sector: target_weight}
        """
        # 按强度排序
        sorted_sectors = sorted(
            sector_strength.items(),
            key=lambda x: -x[1]
        )

        # 配置前N个行业
        allocation = {}
        if len(sorted_sectors) > 0:
            top_sectors = sorted_sectors[:self.top_n_sectors]

            # 等权重配置
            weight = 1.0 / len(top_sectors)
            for sector, _ in top_sectors:
                allocation[sector] = weight

        return allocation


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    计算最大回撤

    Returns:
        (最大回撤, 开始日期, 结束日期)
    """
    if len(equity_curve) < 2:
        return 0, None, None

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    # 找到回撤开始的最高点
    max_dd_value = cummax.loc[max_dd_date]
    start_date = equity_curve[equity_curve == max_dd_value].index[0]

    return max_dd, start_date, max_dd_date
