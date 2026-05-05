"""
VaR / CVaR 风险管理模块

支持:
  - 历史模拟 VaR (Historical VaR)
  - 参数 VaR (Parametric VaR)
  - Cornish-Fisher VaR (调整偏度/峰度)
  - CVaR / Expected Shortfall
  - Monte Carlo VaR
  - 组合级风险
  - 压力测试
  - 风险报告生成
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scipy import stats


@dataclass
class VaRResult:
    """VaR计算结果"""
    method: str
    confidence: float
    horizon: int
    var_value: float
    cvar_value: float
    var_pct: float
    cvar_pct: float

    def to_dict(self) -> dict:
        return {
            'method': self.method,
            'confidence': self.confidence,
            'horizon': f'{self.horizon}d',
            'VaR': f'{self.var_value:,.0f} ({self.var_pct:.2%})',
            'CVaR': f'{self.cvar_value:,.0f} ({self.cvar_pct:.2%})',
        }


@dataclass
class PortfolioRiskReport:
    """组合风险报告"""
    total_value: float
    position_count: int
    var_results: List[VaRResult] = field(default_factory=list)
    concentration: Dict[str, float] = field(default_factory=dict)
    stress_tests: Dict[str, float] = field(default_factory=dict)
    risk_rating: str = ''
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'total_value': f'{self.total_value:,.0f}',
            'position_count': self.position_count,
            'var': [v.to_dict() for v in self.var_results],
            'concentration': self.concentration,
            'stress_tests': {k: f'{v:,.0f}' for k, v in self.stress_tests.items()},
            'risk_rating': self.risk_rating,
            'warnings': self.warnings,
        }

    def summary(self) -> str:
        lines = [
            '=== 组合风险报告 ===',
            f'持仓市值: {self.total_value:,.0f} | 持仓数: {self.position_count}',
            '',
        ]
        for v in self.var_results:
            lines.append(
                f'  [{v.method}] {v.confidence:.0%} VaR({v.horizon}d) = '
                f'{v.var_value:,.0f} ({v.var_pct:.2%}) | '
                f'CVaR = {v.cvar_value:,.0f} ({v.cvar_pct:.2%})'
            )
        if self.stress_tests:
            lines.append('')
            lines.append('压力测试:')
            for scenario, loss in self.stress_tests.items():
                pct = loss / self.total_value * 100 if self.total_value else 0
                lines.append(f'  {scenario}: -{loss:,.0f} ({pct:.1f}%)')
        if self.concentration:
            lines.append('')
            lines.append('行业集中度:')
            for sector, pct in sorted(self.concentration.items(), key=lambda x: -x[1]):
                if pct > 0.01:
                    lines.append(f'  {sector}: {pct:.1%}')
        if self.warnings:
            lines.append('')
            lines.append('风险警告:')
            for w in self.warnings:
                lines.append(f'  - {w}')
        lines.append(f'\n风险评级: {self.risk_rating.upper()}')
        return '\n'.join(lines)


class RiskManager:
    """风险管理器"""

    STRESS_SCENARIOS = {
        '2015股灾(6月)': {'start': '2015-06-12', 'end': '2015-07-08'},
        '2020疫情(2月)': {'start': '2020-01-20', 'end': '2020-02-03'},
        '2022熊市': {'start': '2022-01-04', 'end': '2022-04-27'},
        '2024微盘股崩盘': {'start': '2024-01-05', 'end': '2024-02-05'},
    }

    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def historical_var(self, returns: pd.Series, confidence: float = 0.95,
                       horizon: int = 1, position_value: float = 0) -> VaRResult:
        if len(returns) < 30:
            return VaRResult('historical', confidence, horizon, 0, 0, 0, 0)
        var_daily = -np.percentile(returns, (1 - confidence) * 100)
        cvar_daily = -returns[returns <= -var_daily].mean() if (returns <= -var_daily).any() else var_daily
        var_h, cvar_h = var_daily * np.sqrt(horizon), cvar_daily * np.sqrt(horizon)
        return VaRResult('historical', confidence, horizon,
                         var_h * position_value, cvar_h * position_value, var_h, cvar_h)

    def parametric_var(self, returns: pd.Series, confidence: float = 0.95,
                       horizon: int = 1, position_value: float = 0) -> VaRResult:
        if len(returns) < 30:
            return VaRResult('parametric', confidence, horizon, 0, 0, 0, 0)
        mu, sigma = returns.mean(), returns.std()
        z = stats.norm.ppf(confidence)
        var_daily = z * sigma - mu
        cvar_daily = sigma * stats.norm.pdf(z) / (1 - confidence) - mu
        var_h, cvar_h = var_daily * np.sqrt(horizon), cvar_daily * np.sqrt(horizon)
        return VaRResult('parametric', confidence, horizon,
                         var_h * position_value, cvar_h * position_value, var_h, cvar_h)

    def cornish_fisher_var(self, returns: pd.Series, confidence: float = 0.95,
                           horizon: int = 1, position_value: float = 0) -> VaRResult:
        if len(returns) < 30:
            return VaRResult('cornish_fisher', confidence, horizon, 0, 0, 0, 0)
        mu, sigma = returns.mean(), returns.std()
        skew, kurt = returns.skew(), returns.kurtosis()
        z = stats.norm.ppf(confidence)
        z_cf = (z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24
                - (2 * z**3 - 5 * z) * skew**2 / 36)
        var_daily = z_cf * sigma - mu
        tail = returns[returns <= -var_daily]
        cvar_daily = -tail.mean() if len(tail) > 0 else var_daily
        var_h, cvar_h = var_daily * np.sqrt(horizon), cvar_daily * np.sqrt(horizon)
        return VaRResult('cornish_fisher', confidence, horizon,
                         var_h * position_value, cvar_h * position_value, var_h, cvar_h)

    def monte_carlo_var(self, returns: pd.Series, confidence: float = 0.95,
                        horizon: int = 1, position_value: float = 0,
                        n_simulations: int = 10000) -> VaRResult:
        if len(returns) < 30:
            return VaRResult('monte_carlo', confidence, horizon, 0, 0, 0, 0)
        mu, sigma = returns.mean(), returns.std()
        np.random.seed(42)
        sim = mu * horizon + sigma * np.sqrt(horizon) * np.random.standard_normal(n_simulations)
        var_pct = -np.percentile(sim, (1 - confidence) * 100)
        tail = sim[sim <= -var_pct]
        cvar_pct = -np.mean(tail) if len(tail) > 0 else var_pct
        return VaRResult('monte_carlo', confidence, horizon,
                         var_pct * position_value, cvar_pct * position_value, var_pct, cvar_pct)

    def portfolio_var(self, returns_dict: Dict[str, pd.Series],
                      weights: Optional[Dict[str, float]] = None,
                      confidence: float = 0.95, horizon: int = 1,
                      total_value: float = 0) -> VaRResult:
        symbols = list(returns_dict.keys())
        if not symbols:
            return VaRResult('portfolio_historical', confidence, horizon, 0, 0, 0, 0)
        if weights is None:
            weights = {s: 1.0 / len(symbols) for s in symbols}
        df = pd.DataFrame(returns_dict).dropna()
        if len(df) < 30:
            return VaRResult('portfolio_historical', confidence, horizon, 0, 0, 0, 0)
        w = np.array([weights.get(s, 0) for s in df.columns])
        w = w / w.sum()
        port_ret = pd.Series(df.values @ w, index=df.index)
        return self.historical_var(port_ret, confidence, horizon, total_value)

    def parametric_portfolio_var(self, returns_dict: Dict[str, pd.Series],
                                  weights: Optional[Dict[str, float]] = None,
                                  confidence: float = 0.95, horizon: int = 1,
                                  total_value: float = 0) -> VaRResult:
        symbols = list(returns_dict.keys())
        if not symbols:
            return VaRResult('portfolio_parametric', confidence, horizon, 0, 0, 0, 0)
        if weights is None:
            weights = {s: 1.0 / len(symbols) for s in symbols}
        df = pd.DataFrame(returns_dict).dropna()
        if len(df) < 30:
            return VaRResult('portfolio_parametric', confidence, horizon, 0, 0, 0, 0)
        w = np.array([weights.get(s, 0) for s in df.columns])
        w = w / w.sum()
        port_vol = np.sqrt(w @ df.cov().values @ w)
        port_mean = (df.mean().values * w).sum()
        z = stats.norm.ppf(confidence)
        var_daily = z * port_vol - port_mean
        cvar_daily = port_vol * stats.norm.pdf(z) / (1 - confidence) - port_mean
        var_h, cvar_h = var_daily * np.sqrt(horizon), cvar_daily * np.sqrt(horizon)
        return VaRResult('portfolio_parametric', confidence, horizon,
                         var_h * total_value, cvar_h * total_value, var_h, cvar_h)

    def simple_stress_test(self, position_values: Dict[str, float],
                           shocks: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if shocks is None:
            shocks = {'温和回调(-5%)': -0.05, '中度回调(-10%)': -0.10,
                      '严重回调(-20%)': -0.20, '极端危机(-30%)': -0.30}
        total = sum(position_values.values())
        return {name: abs(pct) * total for name, pct in shocks.items()}

    def generate_report(self, returns_dict: Dict[str, pd.Series],
                        position_values: Dict[str, float],
                        position_sectors: Optional[Dict[str, str]] = None,
                        confidence: float = 0.95, horizon: int = 1) -> PortfolioRiskReport:
        total_value = sum(position_values.values())
        n = len(position_values)
        symbols = [s for s in position_values if s in returns_dict]
        if not symbols:
            return PortfolioRiskReport(total_value, n, risk_rating='unknown',
                                       warnings=['无足够历史数据计算风险'])

        report = PortfolioRiskReport(total_value=total_value, position_count=n)
        weights = {s: v / total_value for s, v in position_values.items() if s in returns_dict}
        sub_returns = {s: returns_dict[s] for s in symbols}

        report.var_results = [
            self.portfolio_var(sub_returns, weights, confidence, horizon, total_value),
            self.parametric_portfolio_var(sub_returns, weights, confidence, horizon, total_value),
        ]

        if position_sectors:
            sv = {}
            for s, v in position_values.items():
                sv[position_sectors.get(s, '未知')] = sv.get(position_sectors.get(s, '未知'), 0) + v
            report.concentration = {k: v / total_value for k, v in sv.items()}

        report.stress_tests = self.simple_stress_test(position_values)

        worst = max(v.var_pct for v in report.var_results)
        report.risk_rating = 'low' if worst < 0.02 else ('medium' if worst < 0.05 else ('high' if worst < 0.10 else 'extreme'))

        warnings = []
        for sector, pct in report.concentration.items():
            if pct > 0.4:
                warnings.append(f'行业"{sector}"集中度 {pct:.0%} 过高')
        if n <= 2 and total_value > 100000:
            warnings.append(f'仅{n}只持仓，分散度不足')
        report.warnings = warnings

        return report

    def stock_risk_metrics(self, returns: pd.Series) -> dict:
        if len(returns) < 30:
            return {'error': '数据不足'}
        mu, sigma = returns.mean() * 252, returns.std() * np.sqrt(252)
        neg = (returns < 0).astype(int)
        max_losing, cur = 0, 0
        for v in neg:
            cur = cur + 1 if v else 0
            max_losing = max(max_losing, cur)
        return {
            'annual_return': f'{mu:.2%}', 'annual_volatility': f'{sigma:.2%}',
            'sharpe': f'{(mu - self.risk_free_rate) / sigma:.2f}' if sigma > 0 else 'N/A',
            'max_daily_loss': f'{returns.min():.2%}', 'max_losing_streak': f'{max_losing}天',
            'downside_dev': f'{returns[returns < 0].std() * np.sqrt(252):.2%}',
        }
