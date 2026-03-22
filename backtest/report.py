"""
回测报告生成模块
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


class ReportGenerator:
    """
    回测报告生成器
    """

    def __init__(self, results: Dict[str, Any]):
        """
        初始化报告生成器

        Args:
            results: 回测结果字典
        """
        self.results = results

    def generate_text_report(self) -> str:
        """生成文本报告"""
        from .metrics import Metrics

        metrics = Metrics(
            initial_capital=self.results.get('initial_capital', 100000),
            equity_curve=self.results.get('equity_curve', pd.Series()),
            trades=self.results.get('trades', [])
        )

        return metrics.summary()

    def save_html_report(
        self,
        filepath: Optional[str] = None
    ) -> str:
        """
        保存HTML格式报告

        Args:
            filepath: 文件路径，None则使用默认路径

        Returns:
            保存的文件路径
        """
        if filepath is None:
            filepath = './reports/backtest_report.html'

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self._generate_html()

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(path)

    def _generate_html(self) -> str:
        """生成HTML内容"""
        from .metrics import Metrics

        metrics = Metrics(
            initial_capital=self.results.get('initial_capital', 100000),
            equity_curve=self.results.get('equity_curve', pd.Series()),
            trades=self.results.get('trades', [])
        )

        d = metrics.to_dict()

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>缠论策略回测报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .positive {{
            color: #4CAF50;
        }}
        .negative {{
            color: #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>缠论策略回测报告</h1>

        <h2>核心指标</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">总收益率</div>
                <div class="metric-value {'positive' if d['total_return'] > 0 else 'negative'}">
                    {d['total_return']:.2%}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">年化收益</div>
                <div class="metric-value {'positive' if d['annual_return'] > 0 else 'negative'}">
                    {d['annual_return']:.2%}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">
                    {d['sharpe_ratio']:.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value negative">
                    {d['max_drawdown']:.2%}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">胜率</div>
                <div class="metric-value">
                    {d['win_rate']:.2%}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">盈亏比</div>
                <div class="metric-value">
                    {d['profit_loss_ratio']:.2f}
                </div>
            </div>
        </div>

        <h2>资金统计</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">初始资金</div>
                <div class="metric-value">¥{d['initial_capital']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最终权益</div>
                <div class="metric-value">¥{d['final_equity']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">总交易次数</div>
                <div class="metric-value">{d['total_trades']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">盈利次数</div>
                <div class="metric-value positive">{d['profitable_trades']}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html
