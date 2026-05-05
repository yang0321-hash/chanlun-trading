"""
投资组合优化模块

基于 PyPortfolioOpt 实现:
  - 均值-方差优化 (Markowitz)
  - 最大化Sharpe比率
  - 最小化波动率
  - 风险平价 (Risk Parity)
  - 行业约束 + 个股上限
  - 与委员会BUY推荐集成
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False


@dataclass
class AllocationResult:
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    shares: Optional[Dict[str, int]] = None
    leftover_cash: float = 0

    def to_dict(self) -> dict:
        d = {
            'method': self.method,
            'expected_return': f'{self.expected_return:.2%}',
            'expected_volatility': f'{self.expected_volatility:.2%}',
            'sharpe_ratio': f'{self.sharpe_ratio:.2f}',
            'weights': {k: f'{v:.1%}' for k, v in self.weights.items() if v > 0.001},
        }
        if self.shares:
            d['shares'] = self.shares
            d['leftover_cash'] = f'¥{self.leftover_cash:,.0f}'
        return d

    def summary(self) -> str:
        lines = [
            f'=== 组合优化 ({self.method}) ===',
            f'预期收益: {self.expected_return:.2%} | 波动率: {self.expected_volatility:.2%} | '
            f'Sharpe: {self.sharpe_ratio:.2f}',
            '权重分配:',
        ]
        for sym, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            if w > 0.001:
                line = f'  {sym}: {w:.1%}'
                if self.shares and sym in self.shares:
                    line += f' ({self.shares[sym]}股)'
                lines.append(line)
        if self.leftover_cash > 0:
            lines.append(f'  剩余资金: ¥{self.leftover_cash:,.0f}')
        return '\n'.join(lines)


class PortfolioOptimizer:

    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def _prepare(self, price_df: pd.DataFrame):
        mu = expected_returns.mean_historical_return(price_df)
        S = risk_models.sample_cov(price_df)
        return mu, S

    def max_sharpe(self, price_df: pd.DataFrame,
                   symbols: Optional[List[str]] = None,
                   max_weight: float = 0.3,
                   sector_constraints: Optional[Dict[str, List[str]]] = None,
                   sector_max: float = 0.5,
                   total_capital: float = 0,
                   latest_prices: Optional[Dict[str, float]] = None) -> AllocationResult:
        if not HAS_PYPFOPT:
            return self._fallback_equal_weight(price_df, symbols, total_capital, latest_prices, 'max_sharpe')

        if symbols:
            cols = [c for c in symbols if c in price_df.columns]
            price_df = price_df[cols]

        mu, S = self._prepare(price_df)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))

        if sector_constraints:
            for sector, syms in sector_constraints.items():
                present = [s for s in syms if s in price_df.columns]
                if present:
                    ef.add_sector(sector, {s: (0, sector_max) for s in present})

        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        clean_weights = ef.clean_weights()
        ret, vol, sr = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)

        result = AllocationResult(weights=clean_weights, expected_return=ret,
                                   expected_volatility=vol, sharpe_ratio=sr, method='max_sharpe')
        if total_capital > 0 and latest_prices:
            result = self._discrete_allocate(result, total_capital, latest_prices)
        return result

    def min_volatility(self, price_df: pd.DataFrame,
                       symbols: Optional[List[str]] = None,
                       max_weight: float = 0.3,
                       total_capital: float = 0,
                       latest_prices: Optional[Dict[str, float]] = None) -> AllocationResult:
        if not HAS_PYPFOPT:
            return self._fallback_equal_weight(price_df, symbols, total_capital, latest_prices, 'min_volatility')

        if symbols:
            cols = [c for c in symbols if c in price_df.columns]
            price_df = price_df[cols]

        mu, S = self._prepare(price_df)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        weights = ef.min_volatility()
        clean_weights = ef.clean_weights()
        ret, vol, sr = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)

        result = AllocationResult(weights=clean_weights, expected_return=ret,
                                   expected_volatility=vol, sharpe_ratio=sr, method='min_volatility')
        if total_capital > 0 and latest_prices:
            result = self._discrete_allocate(result, total_capital, latest_prices)
        return result

    def risk_parity(self, price_df: pd.DataFrame,
                    symbols: Optional[List[str]] = None,
                    total_capital: float = 0,
                    latest_prices: Optional[Dict[str, float]] = None) -> AllocationResult:
        if symbols:
            cols = [c for c in symbols if c in price_df.columns]
            price_df = price_df[cols]

        returns = price_df.pct_change().dropna()
        vols = returns.std()
        inv_vol = 1.0 / vols.replace(0, np.nan).dropna()
        weights = (inv_vol / inv_vol.sum()).to_dict()

        port_ret = (returns.mean() * pd.Series(weights)).sum() * 252
        w_arr = pd.Series(weights)
        port_vol = np.sqrt(w_arr @ returns.cov().values @ w_arr.values) * np.sqrt(252)
        sr = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        result = AllocationResult(weights=weights, expected_return=port_ret,
                                   expected_volatility=port_vol, sharpe_ratio=sr, method='risk_parity')
        if total_capital > 0 and latest_prices:
            result = self._discrete_allocate(result, total_capital, latest_prices)
        return result

    def efficient_frontier_points(self, price_df: pd.DataFrame,
                                  n_points: int = 20,
                                  max_weight: float = 0.3) -> List[Dict[str, float]]:
        if not HAS_PYPFOPT:
            return []
        mu, S = self._prepare(price_df)
        targets = np.linspace(mu.min(), mu.max(), n_points)
        points = []
        for target in targets:
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
                ef.efficient_return(target)
                ret, vol, sr = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
                points.append({'return': ret, 'volatility': vol, 'sharpe': sr})
            except Exception:
                continue
        return points

    def allocate_for_committee(self, buy_signals: List[dict],
                               price_df: pd.DataFrame,
                               total_capital: float,
                               latest_prices: Dict[str, float],
                               max_weight: float = 0.25,
                               method: str = 'max_sharpe',
                               sector_map: Optional[Dict[str, str]] = None,
                               sector_max: float = 0.4) -> AllocationResult:
        if not buy_signals:
            return AllocationResult({}, 0, 0, 0, method)

        symbols = [s['symbol'] for s in buy_signals if 'symbol' in s]
        symbols = [s for s in symbols if s in price_df.columns]
        if not symbols:
            return AllocationResult({}, 0, 0, 0, method)

        sector_constraints = None
        if sector_map:
            sectors = {}
            for sym in symbols:
                sec = sector_map.get(sym, 'other')
                sectors.setdefault(sec, []).append(sym)
            if sectors:
                sector_constraints = sectors

        kwargs = dict(price_df=price_df, symbols=symbols,
                      total_capital=total_capital, latest_prices=latest_prices)

        if method == 'max_sharpe':
            return self.max_sharpe(**kwargs, max_weight=max_weight,
                                   sector_constraints=sector_constraints, sector_max=sector_max)
        elif method == 'min_volatility':
            return self.min_volatility(**kwargs, max_weight=max_weight)
        elif method == 'risk_parity':
            return self.risk_parity(**kwargs)
        return self.max_sharpe(**kwargs, max_weight=max_weight,
                               sector_constraints=sector_constraints, sector_max=sector_max)

    def _discrete_allocate(self, result: AllocationResult,
                           total_capital: float,
                           latest_prices: Dict[str, float]) -> AllocationResult:
        if not HAS_PYPFOPT:
            return result
        try:
            prices = pd.Series(latest_prices)
            da = DiscreteAllocation(result.weights, prices, total_portfolio_value=total_capital)
            alloc, leftover = da.greedy_portfolio()
            result.shares = alloc
            result.leftover_cash = leftover
        except Exception:
            pass
        return result

    def _fallback_equal_weight(self, price_df, symbols, total_capital, latest_prices, method):
        if symbols:
            cols = [c for c in symbols if c in price_df.columns]
            price_df = price_df[cols]
        n = len(price_df.columns)
        if n == 0:
            return AllocationResult({}, 0, 0, 0, method)
        weights = {s: 1.0 / n for s in price_df.columns}
        returns = price_df.pct_change().dropna()
        port_ret = returns.mean().sum() / n * 252
        port_vol = returns.std().mean() / np.sqrt(n) * np.sqrt(252)
        sr = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        result = AllocationResult(weights, port_ret, port_vol, sr, f'{method}_equal_weight')
        if total_capital > 0 and latest_prices:
            result = self._discrete_allocate(result, total_capital, latest_prices)
        return result
