"""
参数优化模块

提供多种参数优化方法：
1. GridSearch - 网格搜索
2. WalkForward - Walk-forward分析
3. Genetic - 遗传算法
4. Bayesian - 贝叶斯优化
"""

from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from itertools import product
from loguru import logger
import json
from datetime import datetime
from pathlib import Path


@dataclass
class OptimizationResult:
    """优化结果"""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    trades: int
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float

    def to_dict(self) -> dict:
        return {
            'params': self.params,
            'metrics': self.metrics,
            'trades': self.trades,
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
        }

    @classmethod
    def from_backtest(cls, params: Dict[str, Any], backtest_result: dict) -> 'OptimizationResult':
        """从回测结果创建"""
        return cls(
            params=params,
            metrics=backtest_result,
            trades=backtest_result.get('total_trades', 0),
            sharpe_ratio=backtest_result.get('sharpe_ratio', 0),
            total_return=backtest_result.get('total_return', 0),
            max_drawdown=backtest_result.get('max_drawdown', 0),
            win_rate=backtest_result.get('win_rate', 0),
        )


@dataclass
class ParameterGrid:
    """参数网格定义"""
    name: str
    values: List[Any]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class GridSearchOptimizer:
    """
    网格搜索优化器

    遍历所有参数组合，找出最优参数
    """

    def __init__(
        self,
        objective: str = 'sharpe_ratio',  # 优化目标
        maximize: bool = True,
        n_jobs: int = 1,  # 并行数
    ):
        self.objective = objective
        self.maximize = maximize
        self.n_jobs = n_jobs
        self.results: List[OptimizationResult] = []
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float('-inf') if maximize else float('inf')

    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        backtest_func: Callable[[Dict[str, Any]], dict],
        data: pd.DataFrame,
        min_trades: int = 10,
    ) -> OptimizationResult:
        """
        执行网格搜索

        Args:
            param_grid: 参数网格 {参数名: [值列表]}
            backtest_func: 回测函数，接收参数字典返回结果
            data: 回测数据
            min_trades: 最小交易次数
        """
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        total_combinations = len(all_combinations)
        logger.info(f"网格搜索: {total_combinations} 种参数组合")

        self.results = []

        # 执行搜索
        if self.n_jobs > 1:
            self._parallel_search(
                all_combinations, param_names, backtest_func, data, min_trades
            )
        else:
            self._sequential_search(
                all_combinations, param_names, backtest_func, data, min_trades
            )

        # 找出最优参数
        if self.results:
            sorted_results = sorted(
                self.results,
                key=lambda x: x.__getattribute__(self.objective),
                reverse=self.maximize
            )
            best = sorted_results[0]
            self.best_params = best.params
            self.best_score = best.__getattribute__(self.objective)

            logger.info(f"最优参数: {self.best_params}")
            logger.info(f"{self.objective}: {self.best_score:.4f}")

            return best
        else:
            logger.warning("没有有效的优化结果")
            return None

    def _sequential_search(
        self,
        combinations: List[Tuple],
        param_names: List[str],
        backtest_func: Callable,
        data: pd.DataFrame,
        min_trades: int,
    ):
        """顺序执行搜索"""
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                result = backtest_func(params)

                # 检查最小交易次数
                if result.get('total_trades', 0) < min_trades:
                    logger.debug(f"[{i+1}/{len(combinations)}] {params} - 交易次数不足")
                    continue

                opt_result = OptimizationResult.from_backtest(params, result)
                self.results.append(opt_result)

                score = opt_result.__getattribute__(self.objective)
                logger.info(
                    f"[{i+1}/{len(combinations)}] {params} - "
                    f"{self.objective}={score:.4f}"
                )

            except Exception as e:
                logger.error(f"参数 {params} 回测失败: {e}")

    def _parallel_search(
        self,
        combinations: List[Tuple],
        param_names: List[str],
        backtest_func: Callable,
        data: pd.DataFrame,
        min_trades: int,
    ):
        """并行执行搜索"""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                future = executor.submit(self._run_backtest, backtest_func, params)
                futures[future] = (i, params)

            for future in as_completed(futures):
                i, params = futures[future]
                try:
                    result = future.result()

                    if result.get('total_trades', 0) < min_trades:
                        continue

                    opt_result = OptimizationResult.from_backtest(params, result)
                    self.results.append(opt_result)

                    score = opt_result.__getattribute__(self.objective)
                    logger.info(
                        f"[{i+1}/{len(combinations)}] {params} - "
                        f"{self.objective}={score:.4f}"
                    )

                except Exception as e:
                    logger.error(f"参数 {params} 回测失败: {e}")

    @staticmethod
    def _run_backtest(backtest_func: Callable, params: Dict[str, Any]) -> dict:
        """运行回测（用于并行执行）"""
        return backtest_func(params)

    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        """获取前N个结果"""
        sorted_results = sorted(
            self.results,
            key=lambda x: x.__getattribute__(self.objective),
            reverse=self.maximize
        )
        return sorted_results[:n]

    def save_results(self, filepath: str):
        """保存结果"""
        data = {
            'objective': self.objective,
            'maximize': self.maximize,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'results': [r.to_dict() for r in self.results],
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"结果已保存到 {filepath}")

    def load_results(self, filepath: str):
        """加载结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.objective = data['objective']
        self.maximize = data['maximize']
        self.best_params = data['best_params']
        self.best_score = data['best_score']
        self.results = [
            OptimizationResult(
                params=r['params'],
                metrics=r['metrics'],
                trades=r['trades'],
                sharpe_ratio=r['sharpe_ratio'],
                total_return=r['total_return'],
                max_drawdown=r['max_drawdown'],
                win_rate=r['win_rate'],
            )
            for r in data['results']
        ]

        logger.info(f"已加载 {len(self.results)} 条结果")


class WalkForwardOptimizer:
    """
    Walk-forward优化器

    滚动窗口优化，模拟实盘参数更新
    """

    def __init__(
        self,
        in_sample_period: int = 252,    # 样本内期间（天）
        out_sample_period: int = 63,    # 样本外期间（天）
        step_period: int = 21,          # 滚动步长（天）
        min_trades: int = 10,
    ):
        self.in_sample_period = in_sample_period
        self.out_sample_period = out_sample_period
        self.step_period = step_period
        self.min_trades = min_trades

        self.walk_results: List[dict] = []

    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        backtest_func: Callable[[Dict[str, Any], pd.DataFrame], dict],
        data: pd.DataFrame,
        objective: str = 'sharpe_ratio',
        maximize: bool = True,
    ) -> Dict[str, Any]:
        """
        执行Walk-forward分析

        Returns:
            汇总结果字典
        """
        if len(data) < self.in_sample_period + self.out_sample_period:
            raise ValueError("数据长度不足以进行Walk-forward分析")

        all_dates = data.index
        start_idx = 0

        self.walk_results = []

        while True:
            # 计算窗口边界
            in_sample_end = start_idx + self.in_sample_period
            out_sample_end = in_sample_end + self.out_sample_period

            if out_sample_end > len(data):
                break

            # 分割数据
            in_sample_data = data.iloc[start_idx:in_sample_end]
            out_sample_data = data.iloc[in_sample_end:out_sample_end]

            in_sample_dates = (all_dates[start_idx], all_dates[in_sample_end - 1])
            out_sample_dates = (all_dates[in_sample_end], all_dates[out_sample_end - 1])

            logger.info(f"\n=== Walk-forward 窗口 {len(self.walk_results) + 1} ===")
            logger.info(f"样本内: {in_sample_dates[0]} ~ {in_sample_dates[1]}")
            logger.info(f"样本外: {out_sample_dates[0]} ~ {out_sample_dates[1]}")

            # 样本内优化
            logger.info("样本内优化中...")
            grid_optimizer = GridSearchOptimizer(objective=objective, maximize=maximize)

            def in_sample_backtest(params):
                return backtest_func(params, in_sample_data)

            best_in_sample = grid_optimizer.optimize(
                param_grid,
                in_sample_backtest,
                in_sample_data,
                min_trades=self.min_trades,
            )

            if best_in_sample is None:
                logger.warning("样本内优化失败，跳过此窗口")
                start_idx += self.step_period
                continue

            # 样本外验证
            logger.info(f"最优参数: {best_in_sample.params}")
            logger.info("样本外验证中...")

            out_sample_result = backtest_func(best_in_sample.params, out_sample_data)

            # 记录结果
            walk_result = {
                'window': len(self.walk_results) + 1,
                'in_sample_dates': in_sample_dates,
                'out_sample_dates': out_sample_dates,
                'best_params': best_in_sample.params,
                'in_sample': {
                    'sharpe_ratio': best_in_sample.sharpe_ratio,
                    'total_return': best_in_sample.total_return,
                    'max_drawdown': best_in_sample.max_drawdown,
                    'trades': best_in_sample.trades,
                },
                'out_sample': {
                    'sharpe_ratio': out_sample_result.get('sharpe_ratio', 0),
                    'total_return': out_sample_result.get('total_return', 0),
                    'max_drawdown': out_sample_result.get('max_drawdown', 0),
                    'trades': out_sample_result.get('total_trades', 0),
                },
            }
            self.walk_results.append(walk_result)

            logger.info(f"样本内 Sharpe: {best_in_sample.sharpe_ratio:.4f}")
            logger.info(f"样本外 Sharpe: {out_sample_result.get('sharpe_ratio', 0):.4f}")

            # 移动窗口
            start_idx += self.step_period

        return self._summarize_results()

    def _summarize_results(self) -> Dict[str, Any]:
        """汇总Walk-forward结果"""
        if not self.walk_results:
            return {}

        in_sample_sharpe = [r['in_sample']['sharpe_ratio'] for r in self.walk_results]
        out_sample_sharpe = [r['out_sample']['sharpe_ratio'] for r in self.walk_results]

        in_sample_return = [r['in_sample']['total_return'] for r in self.walk_results]
        out_sample_return = [r['out_sample']['total_return'] for r in self.walk_results]

        summary = {
            'windows': len(self.walk_results),
            'in_sample': {
                'mean_sharpe': np.mean(in_sample_sharpe),
                'std_sharpe': np.std(in_sample_sharpe),
                'mean_return': np.mean(in_sample_return),
                'std_return': np.std(in_sample_return),
            },
            'out_sample': {
                'mean_sharpe': np.mean(out_sample_sharpe),
                'std_sharpe': np.std(out_sample_sharpe),
                'mean_return': np.mean(out_sample_return),
                'std_return': np.std(out_sample_return),
            },
            'stability_ratio': np.mean(out_sample_sharpe) / np.mean(in_sample_sharpe)
            if np.mean(in_sample_sharpe) != 0 else 0,
        }

        logger.info("\n=== Walk-forward 汇总 ===")
        logger.info(f"窗口数: {summary['windows']}")
        logger.info(f"样本内平均Sharpe: {summary['in_sample']['mean_sharpe']:.4f} "
                   f"(±{summary['in_sample']['std_sharpe']:.4f})")
        logger.info(f"样本外平均Sharpe: {summary['out_sample']['mean_sharpe']:.4f} "
                   f"(±{summary['out_sample']['std_sharpe']:.4f})")
        logger.info(f"稳定性比率: {summary['stability_ratio']:.2%}")

        return summary

    def get_results_dataframe(self) -> pd.DataFrame:
        """获取结果DataFrame"""
        if not self.walk_results:
            return pd.DataFrame()

        rows = []
        for r in self.walk_results:
            rows.append({
                '窗口': r['window'],
                '样本开始': r['in_sample_dates'][0],
                '样本结束': r['in_sample_dates'][1],
                '验证开始': r['out_sample_dates'][0],
                '验证结束': r['out_sample_dates'][1],
                '最优参数': str(r['best_params']),
                '样本内Sharpe': r['in_sample']['sharpe_ratio'],
                '样本外Sharpe': r['out_sample']['sharpe_ratio'],
                '样本内收益率': r['in_sample']['total_return'],
                '样本外收益率': r['out_sample']['total_return'],
            })

        return pd.DataFrame(rows)

    def save_results(self, filepath: str):
        """保存结果"""
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Walk-forward结果已保存到 {filepath}")

        # 同时保存JSON格式
        json_path = Path(filepath).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.walk_results, f, ensure_ascii=False, indent=2, default=str)


class ParameterStabilityAnalyzer:
    """
    参数稳定性分析器

    分析参数对结果的影响程度
    """

    def __init__(self):
        self.sensitivity_results: Dict[str, dict] = {}

    def analyze(
        self,
        param_grid: Dict[str, List[Any]],
        backtest_func: Callable,
        data: pd.DataFrame,
        metric: str = 'sharpe_ratio',
    ) -> Dict[str, dict]:
        """
        分析参数敏感性

        Returns:
            各参数的敏感性分析结果
        """
        results = {}

        for param_name, param_values in param_grid.items():
            logger.info(f"分析参数: {param_name}")

            param_results = []

            for value in param_values:
                params = {param_name: value}

                try:
                    backtest_result = backtest_func(params, data)
                    metric_value = backtest_result.get(metric, 0)

                    param_results.append({
                        'value': value,
                        'metric': metric_value,
                    })

                    logger.info(f"  {value} -> {metric}={metric_value:.4f}")

                except Exception as e:
                    logger.error(f"  {value} 失败: {e}")

            if param_results:
                metric_values = [r['metric'] for r in param_results]

                results[param_name] = {
                    'values': param_values,
                    'metric_values': metric_values,
                    'min': float(np.min(metric_values)),
                    'max': float(np.max(metric_values)),
                    'mean': float(np.mean(metric_values)),
                    'std': float(np.std(metric_values)),
                    'range': float(np.max(metric_values) - np.min(metric_values)),
                    'coefficient_of_variation': float(np.std(metric_values) / np.abs(np.mean(metric_values)))
                    if np.mean(metric_values) != 0 else float('inf'),
                }

        self.sensitivity_results = results
        return results

    def get_most_sensitive(self) -> List[Tuple[str, float]]:
        """获取最敏感的参数（按变异系数排序）"""
        sorted_params = sorted(
            self.sensitivity_results.items(),
            key=lambda x: x[1].get('coefficient_of_variation', 0),
            reverse=True
        )
        return [(name, data['coefficient_of_variation']) for name, data in sorted_params]

    def plot_sensitivity(self, save_path: Optional[str] = None):
        """绘制参数敏感性图"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                len(self.sensitivity_results),
                1,
                figsize=(10, 4 * len(self.sensitivity_results))
            )

            if len(self.sensitivity_results) == 1:
                axes = [axes]

            for ax, (param_name, data) in zip(axes, self.sensitivity_results.items()):
                values = data['values']
                metric_values = data['metric_values']

                ax.plot(values, metric_values, marker='o', linewidth=2)
                ax.set_xlabel(param_name)
                ax.set_ylabel('Metric Value')
                ax.set_title(f'{param_name} 敏感性分析')
                ax.grid(True, alpha=0.3)

                # 标注最优点
                best_idx = int(np.argmax(metric_values))
                ax.scatter(
                    [values[best_idx]],
                    [metric_values[best_idx]],
                    color='red',
                    s=100,
                    zorder=5
                )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"敏感性图已保存到 {save_path}")

            return fig

        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")
            return None


def create_default_param_grid() -> Dict[str, List[Any]]:
    """创建默认参数网格"""
    return {
        # 周线参数
        'weekly_min_strokes': [2, 3, 4, 5],
        'stop_loss_pct': [0.05, 0.08, 0.10, 0.12],

        # 趋势过滤参数
        'ma_fast': [10, 20, 30],
        'ma_slow': [40, 60, 80],

        # ATR参数
        'atr_period': [10, 14, 20],
        'atr_multiplier': [1.5, 2.0, 2.5, 3.0],

        # 仓位参数
        'position_pct': [0.70, 0.80, 0.90, 0.95],
        'risk_per_trade': [0.01, 0.015, 0.02, 0.025],
    }
