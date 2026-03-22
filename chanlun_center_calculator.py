"""
缠论中枢计算器
根据缠论原著定义计算分型、笔、线段、中枢
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json


class ChanLunKLine:
    """经过包含处理后的K线"""

    def __init__(self, datetime: pd.Timestamp, open: float, high: float, low: float, close: float, volume: float):
        self.datetime = datetime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.index = 0  # 在处理后的序列中的索引

    def __repr__(self):
        return f"KLine({self.datetime}, O={self.open:.2f}, H={self.high:.2f}, L={self.low:.2f}, C={self.close:.2f})"


class Fractal:
    """分型：顶分型或底分型"""

    def __init__(self, kline: ChanLunKLine, fractal_type: str, index: int):
        self.kline = kline
        self.type = fractal_type  # 'top' 或 'bottom'
        self.index = index
        self.high = kline.high
        self.low = kline.low

    def __repr__(self):
        mark = "顶" if self.type == "top" else "底"
        return f"{mark}分型@{self.kline.datetime.strftime('%Y-%m-%d')} H={self.high:.2f} L={self.low:.2f}"


class Stroke:
    """笔：连接相邻的顶分型和底分型"""

    def __init__(self, start_fractal: Fractal, end_fractal: Fractal, stroke_type: str):
        self.start = start_fractal
        self.end = end_fractal
        self.type = stroke_type  # 'up' (向上笔) 或 'down' (向下笔)
        self.high = max(start_fractal.high, end_fractal.high)
        self.low = min(start_fractal.low, end_fractal.low)

    def __repr__(self):
        direction = "↑" if self.type == "up" else "↓"
        return f"{direction}笔: {self.start.kline.datetime.strftime('%Y-%m-%d')} -> {self.end.kline.datetime.strftime('%Y-%m-%d')} ({self.low:.2f}-{self.high:.2f})"


class Center:
    """中枢：至少3笔连续重叠形成"""

    def __init__(self, strokes: List[Stroke], center_type: str):
        self.strokes = strokes
        self.type = center_type  # 'up' (上涨中枢) 或 'down' (下跌中枢)
        self.start_date = strokes[0].start.kline.datetime
        self.end_date = strokes[-1].end.kline.datetime

        # 计算中枢区间
        # 中枢上沿：构成中枢的向下笔中，最低的高点
        # 中枢下沿：构成中枢的向上笔中，最高的低点
        down_strokes = [s for s in strokes if s.type == 'down']
        up_strokes = [s for s in strokes if s.type == 'up']

        if down_strokes:
            self.upper = min(s.high for s in down_strokes)
        else:
            self.upper = max(s.high for s in strokes)

        if up_strokes:
            self.lower = max(s.low for s in up_strokes)
        else:
            self.lower = min(s.low for s in strokes)

        self.high = self.upper  # 中枢高点
        self.low = self.lower   # 中枢低点
        self.mid = (self.upper + self.lower) / 2

    def __repr__(self):
        direction = "上涨中枢" if self.type == "up" else "下跌中枢"
        return f"{direction}@{self.start_date.strftime('%Y-%m-%d')}-{self.end_date.strftime('%Y-%m-%d')} 区间:[{self.lower:.2f}, {self.upper:.2f}]"


class ChanLunAnalyzer:
    """缠论分析器"""

    def __init__(self, df: pd.DataFrame):
        """
        初始化分析器
        df: 包含 datetime, open, high, low, close, volume 列的DataFrame
        """
        self.df = df.copy()
        self.df = self.df.sort_values('datetime').reset_index(drop=True)

        # 转换为KLine对象
        self.klines = []
        for _, row in self.df.iterrows():
            kline = ChanLunKLine(
                datetime=row['datetime'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            kline.index = len(self.klines)
            self.klines.append(kline)

        self.fractals: List[Fractal] = []
        self.strokes: List[Stroke] = []
        self.centers: List[Center] = []

    def process_inclusion(self):
        """
        处理K线包含关系
        方向判断：当前K线与前一根比较
        - 向上趋势中：取高高（high中取max，low中取max）
        - 向下趋势中：取低低（high中取min，low中取min）
        """
        if len(self.klines) < 2:
            return

        # 简化处理：只处理相邻K线的包含关系
        # 实际应该根据趋势方向决定合并方式
        processed = [self.klines[0]]

        for i in range(1, len(self.klines)):
            curr = self.klines[i]
            prev = processed[-1]

            # 检查包含关系
            curr_contains_prev = (curr.high >= prev.high and curr.low <= prev.low)
            prev_contains_curr = (prev.high >= curr.high and prev.low <= curr.low)

            if curr_contains_prev or prev_contains_curr:
                # 有包含关系，需要合并
                # 判断趋势方向：比较当前K线与前一非包含K线
                direction = self._get_trend_direction(processed, curr)

                if direction == 'up':
                    # 向上：取高高
                    merged_high = max(curr.high, prev.high)
                    merged_low = max(curr.low, prev.low)
                else:
                    # 向下：取低低
                    merged_high = min(curr.high, prev.high)
                    merged_low = min(curr.low, prev.low)

                # 创建合并后的K线
                merged = ChanLunKLine(
                    datetime=curr.datetime,
                    open=curr.open,  # 开盘价保持原值
                    high=merged_high,
                    low=merged_low,
                    close=curr.close,  # 收盘价保持原值
                    volume=curr.volume + prev.volume
                )
                merged.index = curr.index
                processed[-1] = merged  # 替换前一根
            else:
                processed.append(curr)

        self.klines = processed

    def _get_trend_direction(self, processed: List[ChanLunKLine], curr: ChanLunKLine) -> str:
        """判断当前趋势方向"""
        if len(processed) < 2:
            return 'down'  # 默认向下

        # 比较当前K线中点与最近几根K线的关系
        curr_mid = (curr.high + curr.low) / 2
        recent = processed[-min(3, len(processed)):]

        highs = [k.high for k in recent]
        lows = [k.low for k in recent]

        if curr_mid > sum(highs) / len(highs):
            return 'up'
        elif curr_mid < sum(lows) / len(lows):
            return 'down'
        else:
            # 根据前一根K线的位置判断
            prev = processed[-1]
            if curr.high > prev.high:
                return 'up'
            else:
                return 'down'

    def identify_fractals(self):
        """
        识别分型
        顶分型：中间K线高点最高、低点最高
        底分型：中间K线低点最低、高点最低
        """
        self.fractals = []

        for i in range(1, len(self.klines) - 1):
            prev = self.klines[i - 1]
            curr = self.klines[i]
            next_k = self.klines[i + 1]

            # 顶分型
            if (curr.high > prev.high and curr.high > next_k.high and
                curr.low > prev.low and curr.low > next_k.low):
                self.fractals.append(Fractal(curr, 'top', i))

            # 底分型
            elif (curr.low < prev.low and curr.low < next_k.low and
                  curr.high < prev.high and curr.high < next_k.high):
                self.fractals.append(Fractal(curr, 'bottom', i))

    def identify_strokes(self, min_bars: int = 3):
        """
        识别笔
        笔的条件：
        1. 顶分型和底分型交替出现
        2. 两个分型之间至少间隔min_bars根K线
        """
        self.strokes = []
        if len(self.fractals) < 2:
            return

        for i in range(len(self.fractals) - 1):
            curr = self.fractals[i]
            next_f = self.fractals[i + 1]

            # 必须是一顶一底交替
            if curr.type == next_f.type:
                continue

            # 检查间隔
            bar_gap = next_f.index - curr.index
            if bar_gap < min_bars:
                continue

            # 确定笔的方向
            if curr.type == 'bottom' and next_f.type == 'top':
                # 向上笔
                stroke = Stroke(curr, next_f, 'up')
                self.strokes.append(stroke)
            elif curr.type == 'top' and next_f.type == 'bottom':
                # 向下笔
                stroke = Stroke(curr, next_f, 'down')
                self.strokes.append(stroke)

    def identify_centers(self, min_strokes: int = 3):
        """
        识别中枢
        中枢条件：至少min_strokes笔连续重叠

        重叠判断：
        - 多笔之间有共同的价格区间
        - 中枢上沿 = 下笔中最低的高点
        - 中枢下沿 = 上笔中最高的低点
        """
        self.centers = []

        if len(self.strokes) < min_strokes:
            return

        i = 0
        while i <= len(self.strokes) - min_strokes:
            # 尝试从中枢开始
            for j in range(i + min_strokes, len(self.strokes) + 1):
                candidate_strokes = self.strokes[i:j]

                # 检查是否有重叠
                if self._has_overlap(candidate_strokes):
                    # 尝试扩展中枢
                    continue
                else:
                    # 形成一个中枢
                    if j - i >= min_strokes:
                        center_strokes = self.strokes[i:j-1]
                        center_type = self._determine_center_type(center_strokes)
                        self.centers.append(Center(center_strokes, center_type))
                        i = j - 1
                        break
            else:
                # 到最后都有重叠
                if len(self.strokes) - i >= min_strokes:
                    center_strokes = self.strokes[i:]
                    center_type = self._determine_center_type(center_strokes)
                    self.centers.append(Center(center_strokes, center_type))
                break

            i += 1

    def _has_overlap(self, strokes: List[Stroke]) -> bool:
        """检查一组笔是否有共同重叠区间"""
        if not strokes:
            return False

        # 计算所有笔的共同区间
        # 上笔的低点作为下限
        up_strokes = [s for s in strokes if s.type == 'up']
        down_strokes = [s for s in strokes if s.type == 'down']

        if not up_strokes or not down_strokes:
            return False

        # 中枢下沿：上笔中最高的低点
        lower = max(s.low for s in up_strokes)
        # 中枢上沿：下笔中最低的高点
        upper = min(s.high for s in down_strokes)

        # 有重叠则上沿 > 下沿
        return upper > lower

    def _determine_center_type(self, strokes: List[Stroke]) -> str:
        """判断中枢类型"""
        # 看进入中枢的笔
        if not strokes:
            return 'up'

        first_stroke = strokes[0]
        return 'down' if first_stroke.type == 'down' else 'up'

    def analyze(self):
        """执行完整分析"""
        print("=" * 60)
        print("缠论分析开始")
        print("=" * 60)

        print(f"原始K线数量: {len(self.df)}")

        # 1. 处理包含关系
        print("\n1. 处理K线包含关系...")
        self.process_inclusion()
        print(f"   处理后K线数量: {len(self.klines)}")

        # 2. 识别分型
        print("\n2. 识别分型...")
        self.identify_fractals()
        top_count = sum(1 for f in self.fractals if f.type == 'top')
        bot_count = sum(1 for f in self.fractals if f.type == 'bottom')
        print(f"   顶分型: {top_count}个, 底分型: {bot_count}个")

        # 3. 识别笔
        print("\n3. 识别笔...")
        self.identify_strokes(min_bars=3)
        up_count = sum(1 for s in self.strokes if s.type == 'up')
        down_count = sum(1 for s in self.strokes if s.type == 'down')
        print(f"   向上笔: {up_count}个, 向下笔: {down_count}个")

        # 4. 识别中枢
        print("\n4. 识别中枢...")
        self.identify_centers(min_strokes=3)
        print(f"   中枢数量: {len(self.centers)}个")

        return {
            'klines': self.klines,
            'fractals': self.fractals,
            'strokes': self.strokes,
            'centers': self.centers
        }

    def print_results(self):
        """打印分析结果"""
        print("\n" + "=" * 60)
        print("分析结果")
        print("=" * 60)

        if self.fractals:
            print("\n【分型】")
            for f in self.fractals[:10]:  # 只显示前10个
                print(f"  {f}")
            if len(self.fractals) > 10:
                print(f"  ... 共{len(self.fractals)}个分型")

        if self.strokes:
            print("\n【笔】")
            for s in self.strokes[:10]:
                print(f"  {s}")
            if len(self.strokes) > 10:
                print(f"  ... 共{len(self.strokes)}笔")

        if self.centers:
            print("\n【中枢】")
            for c in self.centers:
                print(f"  {c}")
                print(f"    包含笔数: {len(c.strokes)}笔")
                print(f"    区间宽度: {c.upper - c.lower:.2f} ({(c.upper - c.lower) / c.lower * 100:.2f}%)")

    def export_to_json(self, filename: str):
        """导出结果到JSON文件"""
        data = {
            'fractals': [
                {
                    'date': str(f.kline.datetime),
                    'type': f.type,
                    'high': f.high,
                    'low': f.low
                }
                for f in self.fractals
            ],
            'strokes': [
                {
                    'start_date': str(s.start.kline.datetime),
                    'end_date': str(s.end.kline.datetime),
                    'type': s.type,
                    'high': s.high,
                    'low': s.low
                }
                for s in self.strokes
            ],
            'centers': [
                {
                    'start_date': str(c.start_date),
                    'end_date': str(c.end_date),
                    'type': c.type,
                    'upper': c.upper,
                    'lower': c.lower,
                    'stroke_count': len(c.strokes)
                }
                for c in self.centers
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已导出到: {filename}")

    def get_current_centers(self) -> List[Center]:
        """获取当前有效的中枢（最近的）"""
        if not self.centers:
            return []
        # 返回最近的中枢
        return [self.centers[-1]]


def load_tdx_json(code: str, json_dir: str = '.claude/temp') -> pd.DataFrame:
    """加载通达信解析后的JSON数据"""
    import os
    json_path = f"{json_dir}/{code}.day.json"
    if not os.path.exists(json_path):
        print(f"未找到数据文件: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    return df


def main():
    """主函数"""
    import sys

    # 示例：分析sz002600
    code = "sz002600"

    print(f"正在分析 {code}...")

    df = load_tdx_json(code)
    if df is None:
        print("请先运行 parse_sz002600.py 生成数据文件")
        return

    print(f"数据范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"数据量: {len(df)}条")

    # 创建分析器
    analyzer = ChanLunAnalyzer(df)

    # 执行分析
    analyzer.analyze()

    # 打印结果
    analyzer.print_results()

    # 导出结果
    analyzer.export_to_json('.claude/temp/chanlun_analysis.json')


if __name__ == '__main__':
    main()
