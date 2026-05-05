"""
因子IC/IR分析模块

评估因子对股票收益的预测能力:
  - IC (Information Coefficient): Spearman秩相关系数
  - IR (Information Ratio): IC均值 / IC标准差
  - 分层回测: Top/Bottom组合收益差
  - 因子衰减分析: 不同滞后期IC变化
  - 多因子相关性矩阵
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scipy import stats


@dataclass
class FactorICTimeSeries:
    dates: List[str]
    ic_values: List[float]
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_rate: float
    t_stat: float

    def to_dict(self) -> dict:
        return {
            'IC_mean': f'{self.ic_mean:.4f}',
            'IC_std': f'{self.ic_std:.4f}',
            'IR': f'{self.ir:.2f}',
            'IC_positive_rate': f'{self.ic_positive_rate:.1%}',
            't_stat': f'{self.t_stat:.2f}',
            'periods': len(self.dates),
        }


@dataclass
class LayerTestResult:
    layer_names: List[str]
    layer_returns: List[float]
    top_return: float
    bottom_return: float
    spread: float

    def to_dict(self) -> dict:
        return {
            'layers': dict(zip(self.layer_names, [f'{r:.2%}' for r in self.layer_returns])),
            'spread': f'{self.spread:.2%}',
        }


@dataclass
class FactorReport:
    factor_name: str
    ic_analysis: Optional[FactorICTimeSeries] = None
    layer_test: Optional[LayerTestResult] = None
    decay_analysis: Optional[Dict[int, float]] = None
    effectiveness: str = ''

    def to_dict(self) -> dict:
        d = {'factor': self.factor_name, 'effectiveness': self.effectiveness}
        if self.ic_analysis:
            d['ic'] = self.ic_analysis.to_dict()
        if self.layer_test:
            d['layer_test'] = self.layer_test.to_dict()
        if self.decay_analysis:
            d['decay'] = {f'lag_{k}d': f'{v:.4f}' for k, v in self.decay_analysis.items()}
        return d

    def summary(self) -> str:
        lines = [f'=== 因子分析: {self.factor_name} ===']
        if self.ic_analysis:
            ic = self.ic_analysis
            lines.append(f'  IC均值={ic.ic_mean:.4f} | IR={ic.ir:.2f} | '
                         f'IC>0比率={ic.ic_positive_rate:.1%} | t={ic.t_stat:.2f}')
        if self.layer_test:
            lt = self.layer_test
            lines.append(f'  Top={lt.top_return:.2%} | Bottom={lt.bottom_return:.2%} | '
                         f'Spread={lt.spread:.2%}')
        if self.decay_analysis:
            decay_str = ' | '.join(f'{k}d={v:.3f}' for k, v in self.decay_analysis.items())
            lines.append(f'  衰减: {decay_str}')
        lines.append(f'  有效性: {self.effectiveness}')
        return '\n'.join(lines)


class FactorAnalyzer:
    """因子分析器 — IC/IR + 分层回测 + 衰减分析"""

    IC_THRESHOLDS = {
        'strong': 0.05,
        'moderate': 0.03,
        'weak': 0.01,
    }

    def calc_ic_series(self, factor_df: pd.DataFrame,
                       return_df: pd.DataFrame,
                       date_col: str = 'date',
                       asset_col: str = 'asset',
                       factor_col: str = 'factor',
                       return_col: str = 'return',
                       method: str = 'spearman') -> FactorICTimeSeries:
        merged = pd.merge(factor_df[[date_col, asset_col, factor_col]],
                          return_df[[date_col, asset_col, return_col]],
                          on=[date_col, asset_col])
        dates = sorted(merged[date_col].unique())
        ic_values = []
        for dt in dates:
            sub = merged[merged[date_col] == dt]
            if len(sub) < 5:
                ic_values.append(np.nan)
                continue
            if method == 'spearman':
                corr, _ = stats.spearmanr(sub[factor_col], sub[return_col])
            else:
                corr, _ = stats.pearsonr(sub[factor_col], sub[return_col])
            ic_values.append(corr)

        ic_series = pd.Series(ic_values, index=dates).dropna()
        if len(ic_series) == 0:
            return FactorICTimeSeries([], [], 0, 0, 0, 0, 0)

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        n = len(ic_series)
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0

        return FactorICTimeSeries(
            dates=[str(d) for d in ic_series.index],
            ic_values=ic_series.tolist(),
            ic_mean=ic_mean, ic_std=ic_std, ir=ir,
            ic_positive_rate=(ic_series > 0).mean(), t_stat=t_stat,
        )

    def layer_backtest(self, factor_df: pd.DataFrame,
                       return_df: pd.DataFrame,
                       date_col: str = 'date', asset_col: str = 'asset',
                       factor_col: str = 'factor', return_col: str = 'return',
                       n_layers: int = 5) -> LayerTestResult:
        merged = pd.merge(factor_df[[date_col, asset_col, factor_col]],
                          return_df[[date_col, asset_col, return_col]],
                          on=[date_col, asset_col])
        dates = sorted(merged[date_col].unique())
        layer_returns = {i: [] for i in range(n_layers)}

        for dt in dates:
            sub = merged[merged[date_col] == dt].copy()
            if len(sub) < n_layers * 2:
                continue
            sub['layer'] = pd.qcut(sub[factor_col], n_layers, labels=False, duplicates='drop')
            for i in range(n_layers):
                layer_ret = sub[sub['layer'] == i][return_col].mean()
                if not np.isnan(layer_ret):
                    layer_returns[i].append(layer_ret)

        avg_returns = [np.mean(layer_returns[i]) if layer_returns[i] else 0 for i in range(n_layers)]
        return LayerTestResult(
            layer_names=[f'L{i+1}' for i in range(n_layers)],
            layer_returns=avg_returns,
            top_return=avg_returns[-1],
            bottom_return=avg_returns[0],
            spread=avg_returns[-1] - avg_returns[0],
        )

    def decay_analysis(self, factor_df: pd.DataFrame,
                       return_df: pd.DataFrame,
                       lags: List[int] = None,
                       date_col: str = 'date', asset_col: str = 'asset',
                       factor_col: str = 'factor', return_col: str = 'return') -> Dict[int, float]:
        if lags is None:
            lags = [1, 3, 5, 10, 20]

        ret_sorted = return_df[[date_col, asset_col, return_col]].sort_values([asset_col, date_col])
        results = {}
        for lag in lags:
            ret_shifted = ret_sorted.copy()
            ret_shifted[return_col] = ret_shifted.groupby(asset_col)[return_col].shift(-lag)
            ret_shifted = ret_shifted.dropna(subset=[return_col])
            ic_ts = self.calc_ic_series(factor_df, ret_shifted,
                                        date_col, asset_col, factor_col, return_col)
            results[lag] = ic_ts.ic_mean
        return results

    def full_report(self, factor_df: pd.DataFrame,
                    return_df: pd.DataFrame,
                    factor_name: str = '',
                    date_col: str = 'date', asset_col: str = 'asset',
                    factor_col: str = 'factor', return_col: str = 'return',
                    n_layers: int = 5) -> FactorReport:
        report = FactorReport(factor_name=factor_name or factor_col)

        report.ic_analysis = self.calc_ic_series(
            factor_df, return_df, date_col, asset_col, factor_col, return_col)

        report.layer_test = self.layer_backtest(
            factor_df, return_df, date_col, asset_col, factor_col, return_col, n_layers)

        report.decay_analysis = self.decay_analysis(
            factor_df, return_df, date_col=date_col, asset_col=asset_col,
            factor_col=factor_col, return_col=return_col)

        ic = report.ic_analysis
        if abs(ic.ic_mean) >= self.IC_THRESHOLDS['strong'] and abs(ic.ir) >= 0.5:
            report.effectiveness = 'strong'
        elif abs(ic.ic_mean) >= self.IC_THRESHOLDS['moderate'] and abs(ic.ir) >= 0.3:
            report.effectiveness = 'moderate'
        elif abs(ic.ic_mean) >= self.IC_THRESHOLDS['weak']:
            report.effectiveness = 'weak'
        else:
            report.effectiveness = 'invalid'

        return report

    def factor_correlation_matrix(self, factor_dict: Dict[str, pd.DataFrame],
                                  date_col: str = 'date',
                                  asset_col: str = 'asset',
                                  factor_col: str = 'factor') -> pd.DataFrame:
        names = list(factor_dict.keys())
        n = len(names)
        corr_matrix = pd.DataFrame(np.eye(n), index=names, columns=names)
        for i in range(n):
            for j in range(i + 1, n):
                df_i = factor_dict[names[i]].rename(columns={factor_col: 'f_i'})
                df_j = factor_dict[names[j]].rename(columns={factor_col: 'f_j'})
                merged = pd.merge(df_i[[date_col, asset_col, 'f_i']],
                                  df_j[[date_col, asset_col, 'f_j']],
                                  on=[date_col, asset_col])
                if len(merged) > 10:
                    corr, _ = stats.spearmanr(merged['f_i'], merged['f_j'])
                    corr_matrix.iloc[i, j] = corr
                    corr_matrix.iloc[j, i] = corr
        return corr_matrix
