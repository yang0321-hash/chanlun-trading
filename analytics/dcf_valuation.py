"""
DCF基本面估值模块

支持:
  - DCF折现现金流 (FCFF模型)
  - 股利折现模型 (DDM / Gordon Growth)
  - 相对估值: PE/PB/PS历史分位 + PEG
  - 综合估值报告 + 评级
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DCFResult:
    intrinsic_value: float
    current_price: float
    margin_of_safety: float
    valuation: str
    assumptions: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'intrinsic_value': f'{self.intrinsic_value:.2f}',
            'current_price': f'{self.current_price:.2f}',
            'margin_of_safety': f'{self.margin_of_safety:.1%}',
            'valuation': self.valuation,
        }

    def summary(self) -> str:
        return (f'内在价值: {self.intrinsic_value:.2f} | 现价: {self.current_price:.2f} | '
                f'安全边际: {self.margin_of_safety:.1%} | {self.valuation}')


@dataclass
class RelativeValuation:
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    ps_ttm: Optional[float] = None
    peg: Optional[float] = None
    pe_percentile: Optional[float] = None
    pb_percentile: Optional[float] = None

    def to_dict(self) -> dict:
        d = {}
        if self.pe_ttm: d['PE(TTM)'] = f'{self.pe_ttm:.1f}'
        if self.pb: d['PB'] = f'{self.pb:.2f}'
        if self.peg: d['PEG'] = f'{self.peg:.2f}'
        if self.pe_percentile is not None: d['PE分位'] = f'{self.pe_percentile:.0%}'
        if self.pb_percentile is not None: d['PB分位'] = f'{self.pb_percentile:.0%}'
        return d


@dataclass
class ValuationReport:
    dcf: Optional[DCFResult] = None
    relative: Optional[RelativeValuation] = None
    overall_rating: str = ''
    score: float = 0
    recommendation: str = ''

    def to_dict(self) -> dict:
        d = {'rating': self.overall_rating, 'score': f'{self.score:.0f}/100',
             'recommendation': self.recommendation}
        if self.dcf:
            d['dcf'] = self.dcf.to_dict()
        if self.relative:
            d['relative'] = self.relative.to_dict()
        return d

    def summary(self) -> str:
        lines = ['=== 综合估值报告 ===']
        if self.dcf:
            lines.append(self.dcf.summary())
        if self.relative:
            lines.append(f'相对估值: {self.relative.to_dict()}')
        lines.append(f'评级: {self.overall_rating} ({self.score:.0f}/100)')
        lines.append(f'建议: {self.recommendation}')
        return '\n'.join(lines)


class DCFValuation:
    """DCF估值器"""

    def __init__(self, risk_free_rate: float = 0.03, equity_risk_premium: float = 0.06):
        self.risk_free_rate = risk_free_rate
        self.equity_risk_premium = equity_risk_premium

    def wacc(self, risk_free_rate: float, beta: float, cost_of_debt: float = 0.05,
             tax_rate: float = 0.25, debt_ratio: float = 0.3) -> float:
        cost_of_equity = risk_free_rate + beta * self.equity_risk_premium
        return (1 - debt_ratio) * cost_of_equity + debt_ratio * cost_of_debt * (1 - tax_rate)

    def dcf_fcf(self, current_fcf: float, growth_rate: float, terminal_growth: float,
                wacc: float, n_years: int = 10, shares_outstanding: float = 1,
                net_debt: float = 0, current_price: float = 0) -> DCFResult:
        fcf = current_fcf
        pv_fcf = 0
        for i in range(n_years):
            g = growth_rate if i < 5 else growth_rate * 0.7 + terminal_growth * 0.3
            fcf = fcf * (1 + g)
            pv_fcf += fcf / (1 + wacc) ** (i + 1)

        terminal_fcf = fcf * (1 + terminal_growth)
        pv_terminal = (terminal_fcf / (wacc - terminal_growth)) / (1 + wacc) ** n_years

        ev = pv_fcf + pv_terminal
        intrinsic = (ev - net_debt) / shares_outstanding if shares_outstanding > 0 else 0
        margin = (intrinsic - current_price) / current_price if current_price > 0 else 0

        if margin > 0.3:
            val = 'undervalued'
        elif margin > 0.1:
            val = 'slightly_undervalued'
        elif margin > -0.1:
            val = 'fair'
        elif margin > -0.3:
            val = 'slightly_overvalued'
        else:
            val = 'overvalued'

        return DCFResult(intrinsic, current_price, margin, val,
                         {'growth_rate': growth_rate, 'wacc': wacc, 'terminal_growth': terminal_growth},
                         {'pv_fcf': pv_fcf, 'pv_terminal': pv_terminal, 'ev': ev})

    def ddm(self, current_dividend: float, growth_rate: float,
            required_return: float, current_price: float = 0) -> DCFResult:
        if required_return <= growth_rate:
            return DCFResult(0, current_price, -1, 'invalid_ddm')
        d1 = current_dividend * (1 + growth_rate)
        intrinsic = d1 / (required_return - growth_rate)
        margin = (intrinsic - current_price) / current_price if current_price > 0 else 0
        return DCFResult(intrinsic, current_price, margin,
                         'undervalued' if margin > 0.15 else ('fair' if margin > -0.1 else 'overvalued'))

    def relative_valuation(self, pe_ttm: float, pb: float, eps_growth: float = 0,
                           pe_history: Optional[List[float]] = None,
                           pb_history: Optional[List[float]] = None) -> RelativeValuation:
        rv = RelativeValuation(pe_ttm=pe_ttm, pb=pb)
        if eps_growth > 0:
            rv.peg = pe_ttm / (eps_growth * 100)
        if pe_history and len(pe_history) >= 20:
            rv.pe_percentile = np.mean([1 if pe_ttm >= v else 0 for v in pe_history])
        if pb_history and len(pb_history) >= 20:
            rv.pb_percentile = np.mean([1 if pb >= v else 0 for v in pb_history])
        return rv

    def full_report(self, current_price: float, shares_outstanding: float,
                    current_fcf: float, net_debt: float, growth_rate: float,
                    beta: float = 1.0, terminal_growth: float = 0.03,
                    pe_ttm: float = 0, pb: float = 0, eps_growth: float = 0,
                    pe_history: Optional[List[float]] = None,
                    pb_history: Optional[List[float]] = None,
                    dividend_per_share: float = 0, dividend_growth: float = 0) -> ValuationReport:
        report = ValuationReport()
        w = self.wacc(self.risk_free_rate, beta)

        report.dcf = self.dcf_fcf(current_fcf, growth_rate, terminal_growth, w,
                                   10, shares_outstanding, net_debt, current_price)

        if dividend_per_share > 0 and dividend_growth > 0:
            req = self.risk_free_rate + beta * self.equity_risk_premium
            ddm_r = self.ddm(dividend_per_share, dividend_growth, req, current_price)
            if ddm_r.valuation != 'invalid_ddm':
                report.dcf.intrinsic_value = (report.dcf.intrinsic_value + ddm_r.intrinsic_value) / 2
                report.dcf.margin_of_safety = (report.dcf.intrinsic_value - current_price) / current_price

        if pe_ttm > 0 or pb > 0:
            report.relative = self.relative_valuation(pe_ttm, pb, eps_growth, pe_history, pb_history)

        score = 50
        if report.dcf:
            score += min(25, report.dcf.margin_of_safety * 50)
        if report.relative:
            if report.relative.pe_percentile is not None:
                score += (1 - report.relative.pe_percentile) * 25
            if report.relative.peg is not None:
                score += 10 if report.relative.peg < 0.5 else (-10 if report.relative.peg > 2 else 0)

        score = max(0, min(100, score))
        report.score = score

        if score >= 80:
            report.overall_rating, report.recommendation = 'deep_value', '显著低估'
        elif score >= 65:
            report.overall_rating, report.recommendation = 'value', '合理偏低'
        elif score >= 40:
            report.overall_rating, report.recommendation = 'fair', '估值合理'
        elif score >= 25:
            report.overall_rating, report.recommendation = 'expensive', '偏贵'
        else:
            report.overall_rating, report.recommendation = 'bubble', '明显高估'

        return report
