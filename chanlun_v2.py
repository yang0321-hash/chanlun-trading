"""
缠论分析器 v2 - 修复版
核心改进：
1. 正确识别笔（顶分型连底分型，底分型连顶分型）
2. 正确识别中枢（连续3笔的重叠区间）
3. 可视化更清晰（分层显示）
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Point:
    """点位：分型点"""
    index: int          # K线索引
    date: pd.Timestamp   # 日期
    price: float        # 价格（顶分型=high，底分型=low）
    f_type: str         # 'top' 或 'bottom'

    def __repr__(self):
        mark = "顶" if self.f_type == "top" else "底"
        return f"{mark}@{self.date.strftime('%Y-%m-%d')}({self.price:.2f})"


@dataclass
class Stroke:
    """笔：连接相邻的两个分型"""
    start: Point        # 起始分型
    end: Point          # 结束分型
    s_type: str         # 'up' 或 'down'

    @property
    def high(self):
        return max(self.start.price, self.end.price)

    @property
    def low(self):
        return min(self.start.price, self.end.price)

    def __repr__(self):
        arrow = "↑" if self.s_type == "up" else "↓"
        return f"{arrow}笔:{self.start}->{self.end}"


@dataclass
class Center:
    """中枢：连续笔的重叠区间"""
    strokes: List[Stroke]
    start_idx: int
    end_idx: int
    start_date: pd.Timestamp = None
    end_date: pd.Timestamp = None

    def __post_init__(self):
        """初始化后设置日期"""
        if self.strokes:
            if self.start_date is None:
                self.start_date = self.strokes[0].start.date
            if self.end_date is None:
                self.end_date = self.strokes[-1].end.date

    @property
    def upper(self):
        """中枢上沿 = 下笔中最低的高点"""
        down_strokes = [s for s in self.strokes if s.s_type == 'down']
        if not down_strokes:
            return max(s.high for s in self.strokes)
        return min(s.high for s in down_strokes)

    @property
    def lower(self):
        """中枢下沿 = 上笔中最高的低点"""
        up_strokes = [s for s in self.strokes if s.s_type == 'up']
        if not up_strokes:
            return min(s.low for s in self.strokes)
        return max(s.low for s in up_strokes)

    def __repr__(self):
        return (f"中枢[{self.lower:.2f},{self.upper:.2f}]@"
                f"{self.start_date.strftime('%y/%m/%d')}-"
                f"{self.end_date.strftime('%m/%d')}")

    def __repr__(self):
        return f"中枢[{self.lower:.2f},{self.upper:.2f}]@{self.start_date.strftime('%m/%d')}-{self.end_date.strftime('%m/%d')}"


class ChanLunV2:
    """缠论分析器 v2"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values('datetime').reset_index(drop=True)
        self.points: List[Point] = []
        self.strokes: List[Stroke] = []
        self.centers: List[Center] = []

    def identify_fractals(self) -> List[Point]:
        """
        识别分型
        顶分型：中间K线高点最高、低点最高
        底分型：中间K线低点最低、高点最低
        """
        self.points = []

        for i in range(1, len(self.df) - 1):
            curr = self.df.iloc[i]
            prev = self.df.iloc[i - 1]
            next_k = self.df.iloc[i + 1]

            # 顶分型
            if (curr['high'] > prev['high'] and curr['high'] > next_k['high'] and
                curr['low'] > prev['low'] and curr['low'] > next_k['low']):
                self.points.append(Point(
                    index=i,
                    date=curr['datetime'],
                    price=curr['high'],
                    f_type='top'
                ))

            # 底分型
            elif (curr['low'] < prev['low'] and curr['low'] < next_k['low'] and
                  curr['high'] < prev['high'] and curr['high'] < next_k['high']):
                self.points.append(Point(
                    index=i,
                    date=curr['datetime'],
                    price=curr['low'],
                    f_type='bottom'
                ))

        return self.points

    def identify_strokes(self, min_gap: int = 3) -> List[Stroke]:
        """
        识别笔
        规则：顶分型和底分型必须交替出现，且间隔>=min_gap根K线
        """
        self.strokes = []
        if len(self.points) < 2:
            return self.strokes

        # 按时间顺序连接分型
        i = 0
        while i < len(self.points) - 1:
            curr = self.points[i]
            next_p = self.points[i + 1]

            # 必须是不同类型的分型
            if curr.f_type == next_p.f_type:
                i += 1
                continue

            # 检查间隔
            gap = next_p.index - curr.index
            if gap < min_gap:
                i += 1
                continue

            # 确定笔的方向
            if curr.f_type == 'bottom' and next_p.f_type == 'top':
                stroke = Stroke(curr, next_p, 'up')
            else:
                stroke = Stroke(curr, next_p, 'down')

            self.strokes.append(stroke)
            i += 1

        return self.strokes

    def identify_centers(self, min_strokes: int = 3) -> List[Center]:
        """
        识别中枢
        规则：至少min_strokes笔连续重叠
        """
        self.centers = []
        if len(self.strokes) < min_strokes:
            return self.centers

        i = 0
        while i <= len(self.strokes) - min_strokes:
            # 尝试扩展中枢
            for j in range(i + min_strokes, len(self.strokes) + 1):
                candidate = self.strokes[i:j]

                if not self._check_overlap(candidate):
                    # 不重叠了，形成中枢
                    if j - i >= min_strokes:
                        center = Center(
                            strokes=self.strokes[i:j-1],
                            start_idx=i,
                            end_idx=j-2,
                            start_date=self.strokes[i].start.date,
                            end_date=self.strokes[j-2].end.date
                        )
                        self.centers.append(center)
                    i = j - 1
                    break
            else:
                # 到最后都重叠
                if len(self.strokes) - i >= min_strokes:
                    center = Center(
                        strokes=self.strokes[i:],
                        start_idx=i,
                        end_idx=len(self.strokes)-1,
                        start_date=self.strokes[i].start.date,
                        end_date=self.strokes[-1].end.date
                    )
                    self.centers.append(center)
                break
            i += 1

        return self.centers

    def _check_overlap(self, strokes: List[Stroke]) -> bool:
        """检查一组笔是否有重叠"""
        if not strokes:
            return False

        up_strokes = [s for s in strokes if s.s_type == 'up']
        down_strokes = [s for s in strokes if s.s_type == 'down']

        if not up_strokes or not down_strokes:
            return False

        # 中枢下沿：上笔中最高的低点
        lower = max(s.low for s in up_strokes)
        # 中枢上沿：下笔中最低的高点
        upper = min(s.high for s in down_strokes)

        return upper > lower

    def analyze(self):
        """执行完整分析"""
        print("1. 识别分型...")
        self.identify_fractals()
        print(f"   顶分型: {sum(1 for p in self.points if p.f_type=='top')}个")
        print(f"   底分型: {sum(1 for p in self.points if p.f_type=='bottom')}个")

        print("\n2. 识别笔...")
        self.identify_strokes(min_gap=3)
        print(f"   向上笔: {sum(1 for s in self.strokes if s.s_type=='up')}笔")
        print(f"   向下笔: {sum(1 for s in self.strokes if s.s_type=='down')}笔")

        print("\n3. 识别中枢...")
        self.identify_centers(min_strokes=3)
        print(f"   中枢数量: {len(self.centers)}个")

        return {
            'points': self.points,
            'strokes': self.strokes,
            'centers': self.centers
        }


def visualize_clean(df: pd.DataFrame, analyzer: ChanLunV2,
                    bars: int = 150, save_path: str = None):
    """
    清晰的可视化
    - K线作为背景
    - 分型用三角形标记
    - 笔用连线
    - 中枢用矩形框
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle

    # 只显示最近bars根K线
    if len(df) > bars:
        df_plot = df.tail(bars).copy().reset_index(drop=True)
        start_idx = len(df) - bars
    else:
        df_plot = df.copy().reset_index(drop=True)
        start_idx = 0

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_facecolor('#1a1a2e')

    # 绘制K线
    for i, row in df_plot.iterrows():
        color = '#ff4757' if row['close'] >= row['open'] else '#2ed573'
        # 影线
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1, alpha=0.8)
        # 实体
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        ax.add_patch(Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                              facecolor=color, edgecolor=color, alpha=0.7))

    # 筛选在这个范围内的数据
    plot_points = [p for p in analyzer.points if p.index >= start_idx]

    # 绘制笔的连线
    for stroke in analyzer.strokes:
        # 检查笔是否在显示范围内
        if (stroke.end.index < start_idx or
            stroke.start.index >= start_idx + len(df_plot)):
            continue

        # 计算在显示图中的位置
        x0 = stroke.start.index - start_idx
        x1 = stroke.end.index - start_idx

        # 裁剪到显示范围
        if x0 < 0:
            x0 = 0
        if x1 > len(df_plot) - 1:
            x1 = len(df_plot) - 1

        if x0 > len(df_plot) - 1 or x1 < 0:
            continue

        y0 = stroke.start.price
        y1 = stroke.end.price

        color = '#ff6b6b' if stroke.s_type == 'up' else '#4ecdc4'
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2, alpha=0.8)

    # 绘制中枢
    for center in analyzer.centers:
        start_dt = center.start_date
        end_dt = center.end_date

        # 找到在显示图中的位置
        start_rows = df_plot[df_plot['datetime'] >= start_dt]
        end_rows = df_plot[df_plot['datetime'] <= end_dt]

        if len(start_rows) > 0 and len(end_rows) > 0:
            x0 = start_rows.index[0]
            x1 = end_rows.index[-1]

            if x0 > len(df_plot) or x1 < 0:
                continue

            x0 = max(0, x0)
            x1 = min(len(df_plot) - 1, x1)

            # 绘制中枢矩形
            height = center.upper - center.lower
            rect = Rectangle((x0 - 0.5, center.lower), x1 - x0 + 1, height,
                            facecolor='#3742fa', edgecolor='#ffa502',
                            linewidth=1.5, linestyle='--', alpha=0.25)
            ax.add_patch(rect)

            # 标注中枢
            mid_x = (x0 + x1) / 2
            ax.text(mid_x, center.upper + 0.2,
                   f'[{center.lower:.2f},{center.upper:.2f}]',
                   color='#ffa502', fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                            edgecolor='#ffa502', alpha=0.7))

    # 绘制分型标记
    for p in plot_points:
        x = p.index - start_idx
        if x < 0 or x >= len(df_plot):
            continue

        if p.f_type == 'top':
            ax.scatter(x, p.price, marker='v', s=120, color='#4ecdc4',
                      edgecolors='white', linewidth=1.5, zorder=5)
        else:
            ax.scatter(x, p.price, marker='^', s=120, color='#ff6b6b',
                      edgecolors='white', linewidth=1.5, zorder=5)

    # 设置坐标轴
    ax.set_xlim(-2, len(df_plot))
    ax.grid(True, alpha=0.15, color='white')
    ax.set_ylabel('价格', color='white', fontsize=12)
    ax.tick_params(colors='white')

    # X轴日期
    step = max(1, len(df_plot) // 10)
    x_ticks = list(range(0, len(df_plot), step))
    x_labels = [df_plot.iloc[i]['datetime'].strftime('%m-%d') if i < len(df_plot) else ''
                for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=0, color='white')

    # 标题
    ax.set_title('缠论分析图', color='white', fontsize=16, fontweight='bold', pad=15)

    # 图例
    legend_elements = [
        mpatches.Patch(color='#ff4757', label='阳线'),
        mpatches.Patch(color='#2ed573', label='阴线'),
        mpatches.Patch(color='#ff6b6b', label='向上笔/底分型'),
        mpatches.Patch(color='#4ecdc4', label='向下笔/顶分型'),
        mpatches.Patch(color='#3742fa', alpha=0.5, label='中枢区间'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              facecolor='#1a1a2e', edgecolor='white',
              labelcolor='white', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, facecolor='#1a1a2e',
                   edgecolor='none', bbox_inches='tight')
        print(f"已保存: {save_path}")
    else:
        plt.show()


def save_for_tdx(df: pd.DataFrame, analyzer: ChanLunV2,
                 output_file: str = '.claude/temp/chanlun_tdx.txt'):
    """
    导出为通达信可识别的格式
    """
    lines = []
    lines.append("{缠论数据}")
    lines.append("{注: 复制到通达信公式管理器测试}")

    # 导出中枢数据（转为通达信公式格式）
    if analyzer.centers:
        last_center = analyzer.centers[-1]
        lines.append(f"\n{{最新中枢}}")
        lines.append(f"ZU := {last_center.upper:.2f};")
        lines.append(f"ZD := {last_center.lower:.2f};")
        lines.append(f"STICKLINE(1, ZU, ZD, 4, 1), COLORBLUE;")

    # 导出分型位置
    lines.append(f"\n{{分型位置}}")
    for p in analyzer.points[-10:]:  # 最近10个
        if p.f_type == 'top':
            lines.append(f"DRAWTEXT_FIX(ISLASTBAR AND CURRBARSCOUNT={len(df)-p.index}, "
                        f"H*1.02, '顶'), COLORGREEN;")
        else:
            lines.append(f"DRAWTEXT_FIX(ISLASTBAR AND CURRBARSCOUNT={len(df)-p.index}, "
                        f"L*0.98, '底'), COLORRED;")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"已导出通达信格式: {output_file}")


def main():
    code = 'sz002600'

    # 加载数据
    print(f"加载 {code} 数据...")
    df = pd.read_json('.claude/temp/sz002600.day.json')
    df['datetime'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    # 创建分析器
    analyzer = ChanLunV2(df)
    analyzer.analyze()

    # 打印最近的中枢
    if analyzer.centers:
        print("\n【最近5个中枢】")
        for c in analyzer.centers[-5:]:
            print(f"  {c}")

    # 生成可视化
    print("\n生成图表...")
    visualize_clean(df, analyzer, bars=120,
                   save_path='.claude/temp/chanlun_v2.png')

    # 导出通达信格式
    save_for_tdx(df, analyzer)

    print("\n完成!")


if __name__ == '__main__':
    main()
