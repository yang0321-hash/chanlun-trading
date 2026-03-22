"""
可视化模块

绘制K线图并标注缠论要素
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .kline import KLine
from .fractal import Fractal, FractalType
from .stroke import Stroke
from .segment import Segment
from .pivot import Pivot


class ChanLunPlotter:
    """
    缠论可视化类

    绘制K线图并标注分型、笔、线段、中枢等要素
    """

    def __init__(self, kline: KLine):
        """
        初始化绘图器

        Args:
            kline: K线对象
        """
        self.kline = kline
        self.df = kline.to_dataframe()

    def plot_kline(
        self,
        title: str = 'K线图',
        height: int = 600,
        style: str = 'plotly'
    ):
        """
        绘制基础K线图

        Args:
            title: 图表标题
            height: 图表高度
            style: 绘图风格 ('plotly' 或 'matplotlib')
        """
        if style == 'plotly' and PLOTLY_AVAILABLE:
            return self._plotly_kline(title, height)
        elif MATPLOTLIB_AVAILABLE:
            return self._matplotlib_kline(title)
        else:
            raise ImportError("需要安装 plotly 或 matplotlib")

    def _plotly_kline(self, title: str, height: int) -> 'go.Figure':
        """使用Plotly绘制K线图"""
        fig = go.Figure(data=[go.Candlestick(
            x=self.df.index,
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name='K线'
        )])

        fig.update_layout(
            title=title,
            yaxis_title='价格',
            height=height,
            xaxis_rangeslider_visible=False
        )

        return fig

    def _matplotlib_kline(self, title: str):
        """使用Matplotlib绘制K线图"""
        fig, ax = plt.subplots(figsize=(14, 7))

        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 绘制K线
        for i, (idx, row) in enumerate(self.df.iterrows()):
            # 计算颜色
            color = 'red' if row['close'] >= row['open'] else 'green'

            # 绘制实体
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            rect = Rectangle(
                (i - 0.3, body_bottom),
                0.6, body_height,
                facecolor=color, edgecolor=color
            )
            ax.add_patch(rect)

            # 绘制影线
            ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)

        ax.set_title(title)
        ax.set_ylabel('价格')

        # 旋转日期标签
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_with_fractals(
        self,
        fractals: List[Fractal],
        title: str = '缠论分型',
        height: int = 700
    ) -> 'go.Figure':
        """
        绘制带分型标注的K线图

        Args:
            fractals: 分型列表
            title: 标题
            height: 高度

        Returns:
            Plotly Figure对象
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("需要安装 plotly")

        fig = self._plotly_kline(title, height)

        # 添加分型标注
        for fractal in fractals:
            idx = fractal.index
            if idx >= len(self.df):
                continue

            dt = self.df.index[idx]

            if fractal.is_top:
                # 顶分型 - 红色倒三角
                fig.add_trace(go.Scatter(
                    x=[dt],
                    y=[fractal.high],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name='顶分型',
                    showlegend=False
                ))
                fig.add_annotation(
                    x=dt,
                    y=fractal.high,
                    text='顶',
                    showarrow=False,
                    yshift=10,
                    font=dict(color='red', size=10)
                )
            else:
                # 底分型 - 绿色正三角
                fig.add_trace(go.Scatter(
                    x=[dt],
                    y=[fractal.low],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name='底分型',
                    showlegend=False
                ))
                fig.add_annotation(
                    x=dt,
                    y=fractal.low,
                    text='底',
                    showarrow=False,
                    yshift=-10,
                    font=dict(color='green', size=10)
                )

        return fig

    def plot_with_strokes(
        self,
        strokes: List[Stroke],
        fractals: Optional[List[Fractal]] = None,
        title: str = '缠论-笔',
        height: int = 800
    ) -> 'go.Figure':
        """
        绘制带笔标注的K线图

        Args:
            strokes: 笔列表
            fractals: 分型列表（可选）
            title: 标题
            height: 高度

        Returns:
            Plotly Figure对象
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("需要安装 plotly")

        fig = self._plotly_kline(title, height)

        # 绘制笔
        for stroke in strokes:
            start_idx = stroke.start_index
            end_idx = min(stroke.end_index, len(self.df) - 1)

            if start_idx >= len(self.df) or end_idx >= len(self.df):
                continue

            start_dt = self.df.index[start_idx]
            end_dt = self.df.index[end_idx]

            # 笔的颜色
            color = 'red' if stroke.is_up else 'green'

            # 绘制笔线
            fig.add_trace(go.Scatter(
                x=[start_dt, end_dt],
                y=[stroke.start_value, stroke.end_value],
                mode='lines',
                line=dict(color=color, width=2),
                name='向上笔' if stroke.is_up else '向下笔',
                showlegend=False
            ))

            # 标注起止点
            fig.add_trace(go.Scatter(
                x=[start_dt, end_dt],
                y=[stroke.start_value, stroke.end_value],
                mode='markers',
                marker=dict(size=8, color=color),
                showlegend=False
            ))

        # 添加分型标注
        if fractals:
            for fractal in fractals:
                idx = fractal.index
                if idx >= len(self.df):
                    continue

                dt = self.df.index[idx]
                color = 'red' if fractal.is_top else 'green'

                fig.add_trace(go.Scatter(
                    x=[dt],
                    y=[fractal.value],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down' if fractal.is_top else 'triangle-up',
                        size=12,
                        color=color
                    ),
                    showlegend=False
                ))

        return fig

    def plot_with_pivots(
        self,
        pivots: List[Pivot],
        title: str = '缠论-中枢',
        height: int = 800
    ) -> 'go.Figure':
        """
        绘制带中枢标注的K线图

        Args:
            pivots: 中枢列表
            title: 标题
            height: 高度

        Returns:
            Plotly Figure对象
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("需要安装 plotly")

        fig = self._plotly_kline(title, height)

        # 绘制中枢区间
        for i, pivot in enumerate(pivots):
            start_idx = pivot.start_index
            end_idx = min(pivot.end_index, len(self.df) - 1)

            if start_idx >= len(self.df) or end_idx >= len(self.df):
                continue

            start_dt = self.df.index[start_idx]
            end_dt = self.df.index[end_idx]

            # 绘制中枢矩形区域
            fig.add_hrect(
                y0=pivot.low,
                y1=pivot.high,
                x0=start_dt,
                x1=end_dt,
                fillcolor='blue',
                opacity=0.2,
                line_width=0,
                layer='below'
            )

            # 绘制中枢边界线
            fig.add_hline(
                y=pivot.high,
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.5
            )
            fig.add_hline(
                y=pivot.low,
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.5
            )

            # 标注中枢
            fig.add_annotation(
                x=start_dt,
                y=pivot.high,
                text=f'中枢{chr(65 + i % 26)}',
                showarrow=False,
                yshift=5,
                font=dict(color='blue', size=10)
            )

        return fig

    def plot_full_analysis(
        self,
        fractals: List[Fractal],
        strokes: List[Stroke],
        pivots: Optional[List[Pivot]] = None,
        title: str = '缠论完整分析',
        height: int = 900
    ) -> 'go.Figure':
        """
        绘制完整的缠论分析图

        Args:
            fractals: 分型列表
            strokes: 笔列表
            pivots: 中枢列表（可选）
            title: 标题
            height: 高度

        Returns:
            Plotly Figure对象
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("需要安装 plotly")

        fig = self._plotly_kline(title, height)

        # 绘制中枢
        if pivots:
            for pivot in pivots:
                start_idx = pivot.start_index
                end_idx = min(pivot.end_index, len(self.df) - 1)

                if start_idx >= len(self.df) or end_idx >= len(self.df):
                    continue

                start_dt = self.df.index[start_idx]
                end_dt = self.df.index[end_idx]

                fig.add_hrect(
                    y0=pivot.low,
                    y1=pivot.high,
                    x0=start_dt,
                    x1=end_dt,
                    fillcolor='blue',
                    opacity=0.15,
                    line_width=0,
                    layer='below'
                )

        # 绘制笔
        for stroke in strokes:
            start_idx = stroke.start_index
            end_idx = min(stroke.end_index, len(self.df) - 1)

            if start_idx >= len(self.df) or end_idx >= len(self.df):
                continue

            start_dt = self.df.index[start_idx]
            end_dt = self.df.index[end_idx]

            color = 'red' if stroke.is_up else 'green'

            fig.add_trace(go.Scatter(
                x=[start_dt, end_dt],
                y=[stroke.start_value, stroke.end_value],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))

        # 绘制分型
        for fractal in fractals:
            idx = fractal.index
            if idx >= len(self.df):
                continue

            dt = self.df.index[idx]

            if fractal.is_top:
                fig.add_trace(go.Scatter(
                    x=[dt],
                    y=[fractal.high],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    text='顶',
                    textposition='top center',
                    textfont=dict(color='red', size=9),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[dt],
                    y=[fractal.low],
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    text='底',
                    textposition='bottom center',
                    textfont=dict(color='green', size=9),
                    showlegend=False
                ))

        return fig

    def show(self, fig: 'go.Figure') -> None:
        """显示图表"""
        if PLOTLY_AVAILABLE:
            fig.show()

    def save(self, fig: 'go.Figure', filepath: str) -> None:
        """保存图表"""
        if PLOTLY_AVAILABLE:
            fig.write_html(filepath)
        else:
            raise ImportError("需要安装 plotly")


def plot_kline(kline: KLine, **kwargs) -> 'go.Figure':
    """便捷函数：绘制K线图"""
    plotter = ChanLunPlotter(kline)
    return plotter.plot_kline(**kwargs)


def plot_fractals(kline: KLine, fractals: List[Fractal], **kwargs) -> 'go.Figure':
    """便捷函数：绘制带分型的K线图"""
    plotter = ChanLunPlotter(kline)
    return plotter.plot_with_fractals(fractals, **kwargs)


def plot_strokes(kline: KLine, strokes: List[Stroke], **kwargs) -> 'go.Figure':
    """便捷函数：绘制带笔的K线图"""
    plotter = ChanLunPlotter(kline)
    return plotter.plot_with_strokes(strokes, **kwargs)


def plot_pivots(kline: KLine, pivots: List[Pivot], **kwargs) -> 'go.Figure':
    """便捷函数：绘制带中枢的K线图"""
    plotter = ChanLunPlotter(kline)
    return plotter.plot_with_pivots(pivots, **kwargs)
