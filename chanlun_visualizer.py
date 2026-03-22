"""
缠论可视化脚本
绘制K线图、分型、笔、中枢
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_chanlun_results(json_file: str = '.claude/temp/chanlun_analysis.json'):
    """加载缠论分析结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_kline_data(code: str = 'sz002600', json_dir: str = '.claude/temp'):
    """加载K线数据"""
    import os
    json_path = f"{json_dir}/{code}.day.json"
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        kline_data = json.load(f)

    df = pd.DataFrame(kline_data)
    df['datetime'] = pd.to_datetime(df['date'])
    return df


def plot_chanlun_chart(df: pd.DataFrame, chanlun_data: dict,
                       start_date: str = None, end_date: str = None,
                       save_path: str = None):
    """
    绘制缠论分析图

    参数:
        df: K线数据
        chanlun_data: 缠论分析结果
        start_date: 起始日期
        end_date: 结束日期
        save_path: 保存路径
    """

    # 筛选日期范围
    if start_date:
        df = df[df['datetime'] >= start_date]
    if end_date:
        df = df[df['datetime'] <= end_date]

    df = df.reset_index(drop=True)

    if len(df) == 0:
        print("没有数据")
        return

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                     height_ratios=[3, 1],
                                     sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # ==================== K线图 ====================
    dates = df['datetime']

    # 绘制K线
    for i, row in df.iterrows():
        color = 'red' if row['close'] >= row['open'] else 'green'
        # 影线
        ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        # 实体
        ax1.plot([i, i], [row['open'], row['close']], color=color, linewidth=3)

    ax1.set_xlim(-1, len(df))
    ax1.set_ylabel('价格', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('缠论分析图', fontsize=14, fontweight='bold')

    # ==================== 绘制中枢 ====================
    if 'centers' in chanlun_data and chanlun_data['centers']:
        for center in chanlun_data['centers']:
            start_dt = pd.to_datetime(center['start_date'])
            end_dt = pd.to_datetime(center['end_date'])

            # 找到对应的索引范围
            start_idx = df[df['datetime'] >= start_dt].index
            end_idx = df[df['datetime'] <= end_dt].index

            if len(start_idx) > 0 and len(end_idx) > 0:
                left = start_idx[0]
                right = end_idx[-1]
                upper = center['upper']
                lower = center['lower']

                # 绘制中枢区域
                rect = Rectangle((left, lower), right - left, upper - lower,
                                facecolor='blue', alpha=0.15, edgecolor='yellow',
                                linewidth=1, linestyle='--')
                ax1.add_patch(rect)

                # 中枢上下轨线
                ax1.plot([left, right], [upper, upper], 'y--', linewidth=1, alpha=0.7)
                ax1.plot([left, right], [lower, lower], 'y--', linewidth=1, alpha=0.7)

    # ==================== 绘制笔 ====================
    if 'strokes' in chanlun_data and chanlun_data['strokes']:
        for stroke in chanlun_data['strokes']:
            start_dt = pd.to_datetime(stroke['start_date'])
            end_dt = pd.to_datetime(stroke['end_date'])

            # 找到对应的索引和价格
            start_rows = df[df['datetime'] == start_dt]
            end_rows = df[df['datetime'] == end_dt]

            if len(start_rows) > 0 and len(end_rows) > 0:
                x0 = start_rows.index[0]
                x1 = end_rows.index[0]

                if stroke['type'] == 'up':
                    # 向上笔：从底分型到顶分型
                    y0 = stroke['low']   # 起点低点
                    y1 = stroke['high']  # 终点高点
                    color = 'red'
                else:
                    # 向下笔：从顶分型到底分型
                    y0 = stroke['high']  # 起点高点
                    y1 = stroke['low']   # 终点低点
                    color = 'green'

                ax1.plot([x0, x1], [y0, y1], color=color, linewidth=1.5, alpha=0.6)

    # ==================== 绘制分型 ====================
    if 'fractals' in chanlun_data and chanlun_data['fractals']:
        for fractal in chanlun_data['fractals']:
            dt = pd.to_datetime(fractal['date'])
            rows = df[df['datetime'] == dt]

            if len(rows) > 0:
                idx = rows.index[0]

                if fractal['type'] == 'top':
                    # 顶分型：在上方标记
                    ax1.scatter(idx, fractal['high'], marker='v',
                              s=100, color='green', zorder=5)
                    ax1.text(idx, fractal['high'] * 1.01, '顶',
                            fontsize=8, color='green', ha='center')
                else:
                    # 底分型：在下方标记
                    ax1.scatter(idx, fractal['low'], marker='^',
                              s=100, color='red', zorder=5)
                    ax1.text(idx, fractal['low'] * 0.99, '底',
                            fontsize=8, color='red', ha='center')

    # ==================== 成交量 ====================
    colors = ['red' if df.loc[i, 'close'] >= df.loc[i, 'open'] else 'green'
              for i in range(len(df))]
    ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # X轴日期标签
    def format_date(x, pos):
        if x < 0 or x >= len(df):
            return ''
        return df.iloc[int(x)]['datetime'].strftime('%Y-%m-%d')

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ==================== 图例 ====================
    legend_elements = [
        mpatches.Patch(color='red', label='阳线/向上笔/底分型'),
        mpatches.Patch(color='green', label='阴线/向下笔/顶分型'),
        mpatches.Patch(color='blue', alpha=0.3, label='中枢区间'),
        mpatches.Patch(color='yellow', linestyle='--', label='中枢上下轨'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()


def plot_recent_chart(df: pd.DataFrame, chanlun_data: dict,
                      bars: int = 200, save_path: str = None):
    """绘制最近N根K线的缠论图"""
    if len(df) > bars:
        df_recent = df.tail(bars).copy()
        start_date = df_recent['datetime'].min()
    else:
        df_recent = df
        start_date = None

    plot_chanlun_chart(df_recent, chanlun_data,
                       start_date=start_date,
                       save_path=save_path)


def plot_center_detail(df: pd.DataFrame, chanlun_data: dict,
                      center_index: int = -1, save_path: str = None):
    """绘制某个中枢的详细图"""
    if 'centers' not in chanlun_data or not chanlun_data['centers']:
        print("没有中枢数据")
        return

    centers = chanlun_data['centers']
    center = centers[center_index]

    print(f"绘制中枢: {center['start_date']} 至 {center['end_date']}")
    print(f"区间: [{center['lower']:.2f}, {center['upper']:.2f}]")

    # 扩展日期范围，多显示一些
    start_dt = pd.to_datetime(center['start_date'])
    end_dt = pd.to_datetime(center['end_date'])

    # 向前扩展20根，向后扩展20根
    start_rows = df[df['datetime'] >= start_dt]
    if len(start_rows) > 0:
        start_idx = max(0, start_rows.index[0] - 20)

    end_rows = df[df['datetime'] <= end_dt]
    if len(end_rows) > 0:
        end_idx = min(len(df) - 1, end_rows.index[-1] + 20)

    df_subset = df.iloc[start_idx:end_idx+1].copy()

    plot_chanlun_chart(df_subset, chanlun_data,
                       start_date=df_subset['datetime'].min(),
                       end_date=df_subset['datetime'].max(),
                       save_path=save_path)


def create_html_chart(df: pd.DataFrame, chanlun_data: dict,
                      output_file: str = '.claude/temp/chanlun_chart.html'):
    """创建交互式HTML图表"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('缠论分析图', '成交量')
    )

    # K线图
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线',
        increasing_line_color='red',
        decreasing_line_color='green'
    ), row=1, col=1)

    # 绘制中枢
    if 'centers' in chanlun_data and chanlun_data['centers']:
        for i, center in enumerate(chanlun_data['centers']):
            start_dt = pd.to_datetime(center['start_date'])
            end_dt = pd.to_datetime(center['end_date'])

            # 添加中枢区域
            fig.add_vrect(
                x0=start_dt, x1=end_dt,
                fillcolor="blue", opacity=0.1,
                layer="below", line_width=0,
                row=1, col=1
            )

            # 添加中枢上下轨线
            fig.add_hline(
                y=center['upper'],
                line=dict(color='yellow', width=1, dash='dash'),
                annotation_text=f"上轨:{center['upper']:.2f}",
                annotation_position="right",
                row=1, col=1
            )
            fig.add_hline(
                y=center['lower'],
                line=dict(color='yellow', width=1, dash='dash'),
                annotation_text=f"下轨:{center['lower']:.2f}",
                annotation_position="right",
                row=1, col=1
            )

    # 绘制分型标记
    if 'fractals' in chanlun_data and chanlun_data['fractals']:
        top_fractals = [f for f in chanlun_data['fractals'] if f['type'] == 'top']
        bottom_fractals = [f for f in chanlun_data['fractals'] if f['type'] == 'bottom']

        if top_fractals:
            top_df = pd.DataFrame(top_fractals)
            top_df['date'] = pd.to_datetime(top_df['date'])
            fig.add_trace(go.Scatter(
                x=top_df['date'],
                y=top_df['high'],
                mode='markers+text',
                name='顶分型',
                marker=dict(symbol='triangle-down', size=10, color='green'),
                text=['顶'] * len(top_df),
                textposition='top center',
                textfont=dict(size=8, color='green')
            ), row=1, col=1)

        if bottom_fractals:
            bot_df = pd.DataFrame(bottom_fractals)
            bot_df['date'] = pd.to_datetime(bot_df['date'])
            fig.add_trace(go.Scatter(
                x=bot_df['date'],
                y=bot_df['low'],
                mode='markers+text',
                name='底分型',
                marker=dict(symbol='triangle-up', size=10, color='red'),
                text=['底'] * len(bot_df),
                textposition='bottom center',
                textfont=dict(size=8, color='red')
            ), row=1, col=1)

    # 成交量
    df_reset = df.reset_index(drop=True)
    colors = ['red' if df_reset.loc[i, 'close'] >= df_reset.loc[i, 'open'] else 'green'
              for i in range(len(df_reset))]

    fig.add_trace(go.Bar(
        x=df_reset['datetime'],
        y=df_reset['volume'],
        name='成交量',
        marker_color=colors,
        opacity=0.6
    ), row=2, col=1)

    # 布局设置
    fig.update_layout(
        title='缠论分析图 - 交互式',
        xaxis_rangeslider_visible=False,
        height=800,
        hovermode='x unified',
        template='plotly_dark'
    )

    fig.update_xaxes(title_text='日期', row=2, col=1)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='成交量', row=2, col=1)

    # 保存HTML
    fig.write_html(output_file)
    print(f"交互式图表已保存到: {output_file}")


def main():
    """主函数"""
    code = 'sz002600'

    print("=" * 60)
    print("缠论可视化")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    df = load_kline_data(code)
    chanlun_data = load_chanlun_results()

    if df is None:
        print("K线数据不存在，请先运行 parse_sz002600.py")
        return

    print(f"K线数据: {len(df)}条")
    print(f"分型数量: {len(chanlun_data.get('fractals', []))}")
    print(f"笔数量: {len(chanlun_data.get('strokes', []))}")
    print(f"中枢数量: {len(chanlun_data.get('centers', []))}")

    # 1. 绘制最近200根K线
    print("\n生成最近200根K线图...")
    plot_recent_chart(df, chanlun_data, bars=200,
                     save_path='.claude/temp/chanlun_recent.png')

    # 2. 绘制最后一个中枢详情
    if chanlun_data.get('centers'):
        print("\n生成最后一个中枢详情图...")
        plot_center_detail(df, chanlun_data, center_index=-1,
                          save_path='.claude/temp/chanlun_center_detail.png')

    # 3. 创建交互式HTML图表（最近100根）
    print("\n生成交互式HTML图表...")
    df_recent = df.tail(100).copy()
    create_html_chart(df_recent, chanlun_data,
                     output_file='.claude/temp/chanlun_chart.html')

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  1. .claude/temp/chanlun_recent.png - 最近200根K线静态图")
    print("  2. .claude/temp/chanlun_center_detail.png - 最后中枢详情")
    print("  3. .claude/temp/chanlun_chart.html - 交互式图表")


if __name__ == '__main__':
    main()
