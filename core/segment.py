"""
线段识别模块

根据笔生成线段，使用缠论第71课的特征序列算法

缠论线段划分标准：
1. 将笔序列抽象为特征序列（每笔取端点值，构成虚拟K线）
2. 对特征序列处理包含关系（同K线包含处理规则）
3. 在处理后的特征序列上找分型来判断线段端点

线段破坏分两类：
- 第一类破坏：特征序列分型中，第一元素和第二元素之间无缺口
  → 线段直接在分型极值点结束
- 第二类破坏：特征序列分型中，第一元素和第二元素之间有缺口
  → 需要额外确认（看右侧是否形成反向分型）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from .kline import KLine, KLineData
from .fractal import Fractal, FractalType, FractalDetector
from .stroke import Stroke, StrokeType, StrokeGenerator


@dataclass
class Segment:
    """
    线段数据

    线段是由至少3笔组成的更高级别的走势结构

    Attributes:
        strokes: 组成线段的笔列表
        start_index: 起始K线索引
        end_index: 结束K线索引
        start_value: 线段起始值
        end_value: 线段结束值
        high: 线段最高价
        low: 线段最低价
        direction: 线段方向 ('up' 或 'down')
        broken: 是否被破坏
        break_type: 破坏类型 ('type1' 第一类 / 'type2' 第二类)
    """
    strokes: List[Stroke]
    start_index: int
    end_index: int
    start_value: float
    end_value: float
    high: float
    low: float
    direction: str
    broken: bool = False
    break_type: str = ''

    @property
    def is_up(self) -> bool:
        """是否向上线段"""
        return self.direction == 'up'

    @property
    def is_down(self) -> bool:
        """是否向下线段"""
        return self.direction == 'down'

    @property
    def length(self) -> int:
        """线段包含的笔数量"""
        return len(self.strokes)

    @property
    def bars_count(self) -> int:
        """线段包含的K线数量"""
        if not self.strokes:
            return 0
        return self.strokes[-1].end_index - self.strokes[0].start_index + 1

    @property
    def price_change(self) -> float:
        """价格变化"""
        return self.end_value - self.start_value

    @property
    def price_change_pct(self) -> float:
        """价格变化百分比"""
        if self.start_value == 0:
            return 0
        return (self.end_value - self.start_value) / self.start_value * 100

    @property
    def start_datetime(self) -> datetime:
        """起始时间"""
        return self.strokes[0].start_datetime

    @property
    def end_datetime(self) -> datetime:
        """结束时间"""
        return self.strokes[-1].end_datetime

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'direction': self.direction,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'start_value': self.start_value,
            'end_value': self.end_value,
            'high': self.high,
            'low': self.low,
            'length': self.length,
            'bars_count': self.bars_count,
            'price_change': self.price_change,
            'price_change_pct': self.price_change_pct,
            'start_datetime': self.start_datetime,
            'end_datetime': self.end_datetime,
            'broken': self.broken,
            'break_type': self.break_type
        }


@dataclass
class CharElement:
    """
    特征序列元素

    将每笔抽象为特征序列中的一个元素，相当于虚拟K线。
    向上笔取 [low, high] = [start_value, end_value]
    向下笔取 [low, high] = [end_value, start_value]

    Attributes:
        index: 在笔序列中的索引
        stroke: 原始笔
        high: 元素高点
        low: 元素低点
        is_up: 对应的笔是否向上
    """
    index: int
    stroke: Stroke
    high: float
    low: float
    is_up: bool

    @property
    def direction(self) -> str:
        return 'up' if self.is_up else 'down'


class SegmentGenerator:
    """
    线段生成器（特征序列算法）

    使用缠论第71课的特征序列方法划分线段：
    1. 构建特征序列：将笔的端点值抽象为虚拟K线
    2. 处理特征序列的包含关系
    3. 在特征序列上检测分型
    4. 区分第一类/第二类破坏
    5. 第一类破坏：直接在分型极值点结束线段
    6. 第二类破坏：需确认右侧是否形成反向分型
    """

    def __init__(
        self,
        kline: KLine,
        strokes: Optional[List[Stroke]] = None,
        min_strokes: int = 3
    ):
        """
        初始化线段生成器

        Args:
            kline: K线对象
            strokes: 笔列表，如果为None则自动生成
            min_strokes: 线段最小笔数
        """
        self.kline = kline
        self.min_strokes = min_strokes

        if strokes is None:
            stroke_gen = StrokeGenerator(kline)
            self.strokes = stroke_gen.get_strokes()
        else:
            self.strokes = strokes

        self.segments: List[Segment] = []
        self._generate()

    def _build_char_sequence(self, strokes: List[Stroke]) -> List[CharElement]:
        """
        构建特征序列

        每笔对应一个特征序列元素：
        - 向上笔：high = end_value, low = start_value
        - 向下笔：high = start_value, low = end_value

        注意：特征序列中相邻元素方向一定交替（笔的方向交替）
        """
        elements = []
        for i, s in enumerate(strokes):
            if s.is_up:
                elements.append(CharElement(
                    index=i, stroke=s,
                    high=s.end_value, low=s.start_value,
                    is_up=True
                ))
            else:
                elements.append(CharElement(
                    index=i, stroke=s,
                    high=s.start_value, low=s.end_value,
                    is_up=False
                ))
        return elements

    def _process_inclusion(self, elements: List[CharElement]) -> List[CharElement]:
        """
        处理特征序列的包含关系

        缠论原文：特征序列的包含关系处理与K线相同。
        方向由前一个已处理元素决定（看其对应的笔方向）：
        - 前一元素对应笔向上（阳线）→ 向上合并：取高high、高low
        - 前一元素对应笔向下（阴线）→ 向下合并：取低high、低low
        """
        if len(elements) <= 1:
            return elements.copy()

        processed = [elements[0]]

        for i in range(1, len(elements)):
            curr = elements[i]
            prev = processed[-1]

            # 检查包含关系：一方完全包含另一方
            if (curr.high >= prev.high and curr.low <= prev.low) or \
               (prev.high >= curr.high and prev.low <= curr.low):
                # 有包含关系，按方向合并
                if prev.is_up:
                    # 向上合并：取高high、高low
                    new_high = max(prev.high, curr.high)
                    new_low = max(prev.low, curr.low)
                else:
                    # 向下合并：取低high、低low
                    new_high = min(prev.high, curr.high)
                    new_low = min(prev.low, curr.low)

                # 更新已处理元素
                merged = CharElement(
                    index=prev.index,
                    stroke=prev.stroke,
                    high=new_high,
                    low=new_low,
                    is_up=prev.is_up
                )
                processed[-1] = merged
            else:
                processed.append(curr)

        return processed

    def _detect_char_fractals(self, elements: List[CharElement]) -> List[Tuple[int, FractalType]]:
        """
        在特征序列上检测分型

        特征序列分型定义与K线分型相同：
        - 顶分型：中间元素的high和low都是三者中最高的
        - 底分型：中间元素的high和low都是三者中最低的

        Returns:
            [(元素索引, 分型类型), ...]
        """
        fractals = []
        if len(elements) < 3:
            return fractals

        for i in range(len(elements) - 2):
            e1, e2, e3 = elements[i], elements[i + 1], elements[i + 2]

            # 顶分型检查
            if (e2.high >= e1.high and e2.high >= e3.high and
                    e2.low >= e1.low and e2.low >= e3.low):
                fractals.append((i + 1, FractalType.TOP))
                continue

            # 底分型检查
            if (e2.low <= e1.low and e2.low <= e3.low and
                    e2.high <= e1.high and e2.high <= e3.high):
                fractals.append((i + 1, FractalType.BOTTOM))

        return fractals

    def _has_gap(self, e1: CharElement, e2: CharElement) -> bool:
        """
        判断两个特征序列元素之间是否有缺口

        缺口定义：e1和e2的价格区间不重叠
        - e1.high < e2.low → 向上缺口
        - e1.low > e2.high → 向下缺口
        """
        return e1.high < e2.low or e1.low > e2.high

    def _generate(self) -> None:
        """使用特征序列算法生成线段"""
        if len(self.strokes) < self.min_strokes:
            return

        # Step 1: 构建特征序列
        char_seq = self._build_char_sequence(self.strokes)

        # Step 2: 处理包含关系
        processed_seq = self._process_inclusion(char_seq)

        # Step 3: 检测分型
        char_fractals = self._detect_char_fractals(processed_seq)

        if not char_fractals:
            return

        # Step 4: 根据分型划分线段
        self._divide_segments(processed_seq, char_fractals)

    def _divide_segments(
        self,
        elements: List[CharElement],
        char_fractals: List[Tuple[int, FractalType]]
    ) -> None:
        """
        根据特征序列分型划分线段

        线段划分规则（缠论第71课）：
        1. 向上线段的结束 = 特征序列出现顶分型
        2. 向下线段的结束 = 特征序列出现底分型
        3. 第一类破坏（无缺口）：直接确认线段结束
        4. 第二类破坏（有缺口）：需要后续出现反向分型确认
        """
        # 确定初始方向（第一笔的方向）
        if not elements:
            return

        # 线段起点从第0个元素开始
        seg_start_elem = 0
        seg_direction = 'up' if elements[0].is_up else 'down'

        i = 0
        while i < len(char_fractals):
            frac_idx, frac_type = char_fractals[i]

            # 检查这个分型是否构成当前线段的结束
            # 向上线段需要顶分型结束，向下线段需要底分型结束
            expected_type = FractalType.TOP if seg_direction == 'up' else FractalType.BOTTOM

            if frac_type != expected_type:
                i += 1
                continue

            # 检查分型位置是否在当前线段范围内
            if frac_idx <= seg_start_elem:
                i += 1
                continue

            # 确保线段至少有min_strokes笔
            # 计算实际笔数：从seg_start_elem到frac_idx对应的原始笔数
            start_stroke_idx = elements[seg_start_elem].index
            end_elem_idx = frac_idx
            # 分型的中间元素对应线段结束笔
            end_stroke_idx = elements[end_elem_idx].stroke_index if hasattr(elements[end_elem_idx], 'stroke_index') else elements[end_elem_idx].index

            stroke_count = end_stroke_idx - start_stroke_idx + 1
            if stroke_count < self.min_strokes:
                i += 1
                continue

            # 判断破坏类型（第一类/第二类）
            break_type = 'type1'
            if frac_idx >= 1:
                e1 = elements[frac_idx - 1]
                e2 = elements[frac_idx]
                if self._has_gap(e1, e2):
                    break_type = 'type2'
                    # 第二类破坏需要额外确认：
                    # 缠论第71课要求：缺口后的反向走势必须有效完成
                    # 即反向特征序列分型之后，价格确实走出了反向走势
                    confirmed = False
                    for j in range(i + 1, len(char_fractals)):
                        next_frac_idx, next_frac_type = char_fractals[j]
                        # 反向分型确认
                        reverse_type = FractalType.BOTTOM if seg_direction == 'up' else FractalType.TOP
                        if next_frac_type == reverse_type and next_frac_idx > frac_idx:
                            # 进一步验证：反向分型后的元素必须有效突破
                            # 向上线段结束（顶分型）后需要底分型，且底分型对应的
                            # 笔端点确实突破了之前的高点/低点
                            if next_frac_idx < len(elements):
                                # 验证反向走势力度：特征序列值确实创了新高/新低
                                break_elem = elements[frac_idx]
                                if seg_direction == 'up':
                                    # 向上线段结束，需要确认后续下跌力度
                                    # 反向底分型的低点应低于顶分型前一个元素的低点
                                    target_elem = elements[next_frac_idx]
                                    if target_elem.low < break_elem.low:
                                        confirmed = True
                                        break
                                else:
                                    # 向下线段结束，需要确认后续上涨力度
                                    target_elem = elements[next_frac_idx]
                                    if target_elem.high > break_elem.high:
                                        confirmed = True
                                        break
                            # 如果没有力度确认，不能直接确认第二类破坏
                            break
                    if not confirmed:
                        # 未确认的第二类破坏，暂不结束线段
                        i += 1
                        continue

            # 创建线段
            seg_strokes = self.strokes[start_stroke_idx:end_stroke_idx + 1]
            if len(seg_strokes) < self.min_strokes:
                i += 1
                continue

            seg_high = max(s.high for s in seg_strokes)
            seg_low = min(s.low for s in seg_strokes)

            # 线段起止值: 按时间顺序连接区间的关键极值点
            # 线段从第一个笔的start到最后一个笔的end, 按时间从左到右
            kline_data = self.kline.data
            si = seg_strokes[0].start_index
            ei = seg_strokes[-1].end_index
            seg_bars = kline_data[si:ei + 1]
            if seg_bars:
                seg_start_value = seg_strokes[0].start_value
                seg_end_value = seg_strokes[-1].end_value
                seg_start_index = si
                seg_end_index = ei
            else:
                seg_start_value = seg_strokes[0].start_value
                seg_end_value = seg_strokes[-1].end_value
                seg_start_index = seg_strokes[0].start_index
                seg_end_index = seg_strokes[-1].end_index

            segment = Segment(
                strokes=seg_strokes,
                start_index=seg_start_index,
                end_index=seg_end_index,
                start_value=seg_start_value,
                end_value=seg_end_value,
                high=seg_high,
                low=seg_low,
                direction=seg_direction,
                broken=True,
                break_type=break_type
            )
            self.segments.append(segment)

            # 新线段从分型位置开始
            seg_start_elem = frac_idx
            seg_direction = 'down' if seg_direction == 'up' else 'up'
            i += 1

        # 处理最后一个未完成线段（剩余的笔）
        if seg_start_elem < len(elements):
            start_stroke_idx = elements[seg_start_elem].index
            end_stroke_idx = len(self.strokes) - 1
            seg_strokes = self.strokes[start_stroke_idx:end_stroke_idx + 1]

            if len(seg_strokes) >= self.min_strokes:
                seg_high = max(s.high for s in seg_strokes)
                seg_low = min(s.low for s in seg_strokes)

                kline_data = self.kline.data
                si = seg_strokes[0].start_index
                ei = seg_strokes[-1].end_index
                seg_bars = kline_data[si:ei + 1]
                if seg_bars:
                    seg_start_value = seg_strokes[0].start_value
                    seg_end_value = seg_strokes[-1].end_value
                    seg_start_index = si
                    seg_end_index = ei
                else:
                    seg_start_value = seg_strokes[0].start_value
                    seg_end_value = seg_strokes[-1].end_value
                    seg_start_index = seg_strokes[0].start_index
                    seg_end_index = seg_strokes[-1].end_index

                segment = Segment(
                    strokes=seg_strokes,
                    start_index=seg_start_index,
                    end_index=seg_end_index,
                    start_value=seg_start_value,
                    end_value=seg_end_value,
                    high=seg_high,
                    low=seg_low,
                    direction=seg_direction,
                    broken=False,
                    break_type=''
                )
                self.segments.append(segment)

    def get_segments(
        self,
        direction: Optional[str] = None
    ) -> List[Segment]:
        """
        获取线段列表

        Args:
            direction: 方向过滤 ('up'/'down'/None)

        Returns:
            线段列表
        """
        if direction is None:
            return self.segments.copy()

        return [s for s in self.segments if s.direction == direction]

    def get_up_segments(self) -> List[Segment]:
        """获取所有向上线段"""
        return self.get_segments('up')

    def get_down_segments(self) -> List[Segment]:
        """获取所有向下线段"""
        return self.get_segments('down')

    def get_last_segment(self) -> Optional[Segment]:
        """获取最后一条线段"""
        if self.segments:
            return self.segments[-1]
        return None

    def get_segment_at(self, index: int) -> Optional[Segment]:
        """
        获取指定K线位置所在的线段

        Args:
            index: K线索引

        Returns:
            线段对象
        """
        for segment in self.segments:
            if segment.start_index <= index <= segment.end_index:
                return segment
        return None

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Segment:
        return self.segments[index]


def generate_segments(
    kline: KLine,
    strokes: Optional[List[Stroke]] = None,
    min_strokes: int = 3
) -> List[Segment]:
    """
    便捷函数：生成K线序列的所有线段

    Args:
        kline: K线对象
        strokes: 笔列表，None则自动生成
        min_strokes: 最小笔数

    Returns:
        线段列表
    """
    generator = SegmentGenerator(kline, strokes, min_strokes)
    return generator.get_segments()
