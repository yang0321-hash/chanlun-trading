"""组合风险管理模块 — VaR/CVaR 计算"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VaRResult:
    var_pct: float
    var_value: float
    cvar_pct: float = 0.0
    cvar_value: float = 0.0


@dataclass
class RiskReport:
    total_value: float
    confidence: float
    horizon: int
    var_result: VaRResult = None
    sector_concentration: Dict[str, float] = field(default_factory=dict)
    position_risks: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f'\n=== 组合风险评估 (置信度{self.confidence:.0%}, {self.horizon}日) ===', '']
        if self.var_result:
            lines.append(f'  组合VaR: {self.var_result.var_pct:.2%} (¥{self.var_result.var_value:,.0f})')
            lines.append(f'  CVaR: {self.var_result.cvar_pct:.2%} (¥{self.var_result.cvar_value:,.0f})')
            if self.var_result.var_pct > 0.05:
                lines.append('  ⚠ VaR超过5%，建议减仓或对冲')
            elif self.var_result.var_pct > 0.03:
                lines.append('  ⚡ VaR偏高，注意仓位控制')
            else:
                lines.append('  ✓ VaR正常')
        if self.sector_concentration:
            lines.append('')
            lines.append('  行业集中度:')
            for sector, pct in sorted(self.sector_concentration.items(), key=lambda x: -x[1]):
                flag = ' ⚠' if pct > 0.3 else ''
                lines.append(f'    {sector}: {pct:.1%}{flag}')
        return '\n'.join(lines)


class RiskManager:
    """组合风险计算"""

    def portfolio_var(self, returns_dict: Dict[str, np.ndarray],
                      weights: Dict[str, float],
                      confidence: float = 0.95,
                      horizon: int = 1,
                      portfolio_value: float = 1.0) -> VaRResult:
        """计算组合VaR (参数化法)"""
        codes = list(returns_dict.keys())
        if len(codes) < 1:
            return VaRResult(0, 0)

        min_len = min(len(returns_dict[c]) for c in codes)
        returns_matrix = np.column_stack([
            (returns_dict[c].values if hasattr(returns_dict[c], 'values') else returns_dict[c])[-min_len:]
            for c in codes
        ])

        w = np.array([weights.get(c, 0) for c in codes])
        port_returns = returns_matrix @ w

        mu = port_returns.mean()
        sigma = port_returns.std()
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)

        var_daily = -(mu + z * sigma)
        cvar_daily = -(mu - sigma * norm.pdf(z) / (1 - confidence))

        var_h = var_daily * np.sqrt(horizon)
        cvar_h = cvar_daily * np.sqrt(horizon)

        return VaRResult(
            var_pct=var_h,
            var_value=var_h * portfolio_value,
            cvar_pct=cvar_h,
            cvar_value=cvar_h * portfolio_value,
        )

    def generate_report(self, returns_dict: Dict[str, np.ndarray],
                        position_values: Dict[str, float],
                        position_sectors: Dict[str, str] = None,
                        confidence: float = 0.95,
                        horizon: int = 1) -> RiskReport:
        """生成完整风险报告"""
        total_value = sum(position_values.values())
        if total_value <= 0:
            return RiskReport(total_value=0, confidence=confidence, horizon=horizon)

        weights = {c: v / total_value for c, v in position_values.items()}
        var_result = self.portfolio_var(returns_dict, weights, confidence, horizon, total_value)

        sector_conc = {}
        if position_sectors:
            for code, sector in position_sectors.items():
                sector_conc[sector] = sector_conc.get(sector, 0) + position_values.get(code, 0)
            sector_conc = {s: v / total_value for s, v in sector_conc.items()}

        return RiskReport(
            total_value=total_value,
            confidence=confidence,
            horizon=horizon,
            var_result=var_result,
            sector_concentration=sector_conc,
        )
