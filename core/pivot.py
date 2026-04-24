"""
中枢识别模块

识别缠论中的中枢结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from .kline import KLine
from .fractal import Fractal
from .stroke import Stroke, StrokeType
from .segment import Segment


class PivotLevel(Enum):
    """中枢级别"""
    MIN_1 = '1m'       # 1分钟级别
    MIN_5 = '5m'       # 5分钟级别
    MIN_30 = '30m'     # 30分钟级别
    DAY = 'day'        # 日线级别
    WEEK = 'week'      # 周线级别
    MONTH = 'month'    # 月线级别


class PivotEvolution(Enum):
    """中枢演化状态"""
    FORMING = "forming"       # 形成中
    NORMAL = "normal"         # 标准中枢 (N=3)
    EXTENDING = "extending"   # 延伸中（ZG/ZD不变，段数递增）
    EXPANDING = "expanding"   # 扩张中（ZG/ZD扩大，级别不变）
    UPGRADED = "upgraded"     # 已升级（9段/两中枢重叠 → 级别+1）
    ESCAPED = "escaped"       # 已脱离（3买/3卖确认）


@dataclass
class Pivot:
    """
    中枢数据

    中枢是价格在一定区间内的震荡区域

    缠论原文定义：
    - ZG = min(g₁, g₂) 前两次级别走势高点的较小值（中枢上沿，核心重叠区上界）
    - ZD = max(d₁, d₂) 前两次级别走势低点的较大值（中枢下沿，核心重叠区下界）
    - GG = max(gₙ) 所有构成中枢走势高点的最大值（中枢波动最高点）
    - DD = min(dₙ) 所有构成中枢走势低点的最小值（中枢波动最低点）

    关系：DD ≤ ZD ≤ ZG ≤ GG
    ZG/ZD由初始构成笔确定后固定不变，扩展不影响ZG/ZD

    Attributes:
        level: 中枢级别
        start_index: 起始K线索引
        end_index: 结束K线索引
        high: 中枢上沿（ZG，固定不变）
        low: 中枢下沿（ZD，固定不变）
        zg: 中枢上沿 ZG = min(前两笔high)
        zd: 中枢下沿 ZD = max(前两笔low)
        gg: 中枢波动最高点 GG = max(所有笔high)
        dd: 中枢波动最低点 DD = min(所有笔low)
        strokes: 组成中枢的笔列表
        direction: 中枢方向 ('up'表示上涨中枢, 'down'表示下跌中枢)
        confirmed: 中枢是否确认
        extended: 中枢是否延伸
        segments: 关联的线段
    """
    level: PivotLevel
    start_index: int
    end_index: int
    high: float
    low: float
    zg: float = 0.0
    zd: float = 0.0
    gg: float = 0.0
    dd: float = 0.0
    strokes: List[Stroke] = field(default_factory=list)
    direction: str = ''
    confirmed: bool = False
    extended: bool = False
    segments: List[Segment] = field(default_factory=list)
    # --- 中枢演化字段 ---
    evolution: PivotEvolution = PivotEvolution.NORMAL
    segment_count: int = 0       # 总段数 N（含延伸/扩张）
    upgrade_reason: str = ''     # 升级原因
    sub_level: bool = False      # True=次级别中枢（DIF未穿越0轴）
    dif_crossed_zero: bool = False  # DIF是否穿越0轴
    evolution_level: int = 1     # 演化级别（升级后+1）

    @property
    def range_value(self) -> float:
        """中枢区间宽度"""
        return self.high - self.low

    @property
    def width(self) -> float:
        """ZG-ZD宽度（演化相关）"""
        return self.zg - self.zd

    @property
    def middle(self) -> float:
        """中枢中点"""
        return (self.high + self.low) / 2

    @property
    def strokes_count(self) -> int:
        """组成中枢的笔数量"""
        return len(self.strokes)

    @property
    def is_expanded(self) -> bool:
        """中枢是否扩张 (≥6笔)。扩张中枢代表更强震荡，后续突破更有力"""
        return len(self.strokes) >= 6

    @property
    def start_datetime(self) -> datetime:
        """起始时间"""
        if self.strokes:
            return self.strokes[0].start_datetime
        return datetime.now()

    @property
    def end_datetime(self) -> datetime:
        """结束时间"""
        if self.strokes:
            return self.strokes[-1].end_datetime
        return datetime.now()

    @property
    def quality_score(self) -> float:
        """
        中枢结构质量评分 (0-1)

        因子：
        - 笔画数：更多笔 = 经过更多测试 = 越可靠 (最多0.3)
        - 紧凑度：区间越窄相对价格 = 共识越强 = 越可靠 (最多0.5)
        - 确认状态：已确认中枢更可靠 (0.2)
        """
        # 笔画数评分：7笔及以上为满分
        stroke_score = min(self.strokes_count / 7.0, 1.0) * 0.3

        # 紧凑度评分：区间宽度占中枢中点的百分比
        tightness = 0.0
        if self.middle > 0:
            range_pct = self.range_value / self.middle
            # 区间越窄越好，10%以下为满分
            tightness = max(0.0, 1.0 - range_pct / 0.10) * 0.5

        # 确认加分
        confirmation = 0.2 if self.confirmed else 0.0

        return min(stroke_score + tightness + confirmation, 1.0)

    def gravity_index(self, price: float) -> float:
        """
        中枢引力指数

        量化中枢对价格的"引力"。引力越强，价格越可能被中枢拉回或在中枢附近获得支撑。

        计算方式：
            gravity = width / distance_from_boundary

        - width = ZG - ZD（中枢宽度）
        - distance = 价格到中枢最近边界的距离

        返回值越大，引力越强：
        - >5.0: 极强引力（价格紧贴中枢）
        - 2.0-5.0: 强引力（价格靠近中枢）
        - 0.5-2.0: 中等引力
        - <0.5: 弱引力（价格远离中枢）

        Args:
            price: 当前价格

        Returns:
            引力指数 (≥0)
        """
        w = self.width
        if w <= 0:
            return 0.0

        # 价格到中枢最近边界的距离
        if price > self.zg:
            distance = price - self.zg
        elif price < self.zd:
            distance = self.zd - price
        else:
            # 价格在中枢内部，引力最强
            return 10.0

        if distance <= 0:
            return 10.0

        return w / distance

    def check_consolidation(self) -> Tuple[str, float]:
        """
        检测中枢收敛/发散状态

        通过分析构成笔的波动幅度趋势，判断中枢正在收缩（收敛）
        还是扩张（发散）。

        收敛 = 即将突破的预兆，对3买有利
        发散 = 趋势延续，对顺势交易有利

        Returns:
            ('converging'/'diverging'/'neutral', 收敛/发散强度 0-1)
        """
        if len(self.strokes) < 4:
            return ('neutral', 0.0)

        # 将笔按对分组（上笔+下笔=一对），计算每对的振幅
        pairs = []
        i = 0
        while i < len(self.strokes) - 1:
            s1 = self.strokes[i]
            s2 = self.strokes[i + 1]
            pair_range = abs(s1.high - s1.low) + abs(s2.high - s2.low)
            pair_width = max(s1.high, s2.high) - min(s1.low, s2.low)
            pairs.append(pair_width)
            i += 2

        if len(pairs) < 2:
            return ('neutral', 0.0)

        # 检查振幅趋势：前半 vs 后半
        half = len(pairs) // 2
        first_half_avg = sum(pairs[:half]) / half if half > 0 else 0
        second_half_avg = sum(pairs[half:]) / (len(pairs) - half)

        if first_half_avg <= 0:
            return ('neutral', 0.0)

        ratio = second_half_avg / first_half_avg

        if ratio < 0.7:
            # 后半振幅 < 前半70% → 收敛
            strength = min(1.0 - ratio, 1.0)
            return ('converging', strength)
        elif ratio > 1.3:
            # 后半振幅 > 前半130% → 发散
            strength = min(ratio - 1.0, 1.0)
            return ('diverging', strength)

        return ('neutral', 0.0)

    def contains(self, price: float) -> bool:
        """
        判断价格是否在中枢区间内

        Args:
            price: 价格

        Returns:
            是否在中枢内
        """
        return self.low <= price <= self.high

    def contains_fuzzy(self, price: float, tolerance: float = 0.005) -> bool:
        """
        判断价格是否在模糊中枢区间内

        Args:
            price: 价格
            tolerance: 模糊容忍度（百分比，默认0.5%）

        Returns:
            是否在模糊中枢内
        """
        fuzzy_zg = self.zg * (1 + tolerance) if self.zg > 0 else self.high
        fuzzy_zd = self.zd * (1 - tolerance) if self.zd > 0 else self.low
        return fuzzy_zd <= price <= fuzzy_zg

    def is_above(self, price: float) -> bool:
        """判断价格是否在中枢上方"""
        return price > self.high

    def is_below(self, price: float) -> bool:
        """判断价格是否在中枢下方"""
        return price < self.low

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'level': self.level.value,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'high': self.high,
            'low': self.low,
            'zg': self.zg,
            'zd': self.zd,
            'gg': self.gg,
            'dd': self.dd,
            'range': self.range_value,
            'width': round(self.width, 2),
            'middle': self.middle,
            'strokes_count': self.strokes_count,
            'segment_count': self.segment_count or self.strokes_count,
            'direction': self.direction,
            'confirmed': self.confirmed,
            'extended': self.extended,
            'evolution': self.evolution.value,
            'evolution_level': self.evolution_level,
            'upgrade_reason': self.upgrade_reason,
            'sub_level': self.sub_level,
            'dif_crossed_zero': self.dif_crossed_zero,
            'start_datetime': self.start_datetime,
            'end_datetime': self.end_datetime
        }


class PivotDetector:
    """
    中枢识别器

    识别K线序列中的中枢结构

    中枢定义（缠论）：
    1. 至少由3笔组成，且存在重叠区间
    2. 中枢区间：高点中的最低点 到 低点中的最高点
    3. 后续笔在中枢区间内震荡则为中枢延伸
    4. 突破中枢后回抽不破则为第三类买卖点
    """

    def __init__(
        self,
        kline: KLine,
        strokes: Optional[List[Stroke]] = None,
        level: PivotLevel = PivotLevel.DAY
    ):
        """
        初始化中枢识别器

        Args:
            kline: K线对象
            strokes: 笔列表，如果为None则自动生成
            level: 中枢级别
        """
        self.kline = kline
        self.level = level

        if strokes is None:
            from .stroke import StrokeGenerator
            stroke_gen = StrokeGenerator(kline)
            self.strokes = stroke_gen.get_strokes()
        else:
            self.strokes = strokes

        self.pivots: List[Pivot] = []
        self._detect()

    def _detect(self) -> None:
        """检测所有中枢"""
        if len(self.strokes) < 3:
            return

        i = 0
        while i <= len(self.strokes) - 3:
            pivot = self._try_create_pivot(i)

            if pivot is not None:
                self.pivots.append(pivot)
                # 跳过已使用的笔
                i += pivot.strokes_count - 1
            else:
                i += 1

        # 检查中枢延伸
        self._check_pivot_extension()

        # 检查中枢演化状态 + DIF级别
        self._check_evolution()

    def _try_create_pivot(self, start_idx: int) -> Optional[Pivot]:
        """
        尝试从指定位置创建中枢

        缠论原文：中枢由至少3笔重叠区间构成
        - ZG = min(g1, g2) 由前2笔高点确定，固定不变
        - ZD = max(d1, d2) 由前2笔低点确定，固定不变
        - 第3笔必须与 [ZD, ZG] 有重叠才确认中枢成立
        - 扩展条件：新笔与 [ZD, ZG] 有重叠（ZG/ZD不变）
        - GG/DD 取所有笔的极值

        Args:
            start_idx: 起始笔索引

        Returns:
            中枢对象，如果不能创建则返回None
        """
        if start_idx + 2 >= len(self.strokes):
            return None

        # 获取3笔
        s1 = self.strokes[start_idx]
        s2 = self.strokes[start_idx + 1]
        s3 = self.strokes[start_idx + 2]

        # 检查方向是否交替
        if s1.type == s2.type or s2.type == s3.type:
            return None

        # 计算中枢核心区间 (ZG/ZD)
        # 缠论定义：ZG = min(H1, H2, H3) 三笔高点的最小值
        #           ZD = max(L1, L2, L3) 三笔低点的最大值
        zg = min(s1.high, s2.high, s3.high)
        zd = max(s1.low, s2.low, s3.low)

        # 检查是否有有效的重叠区间
        if zg <= zd:
            return None

        # 第3笔必须与前两笔的重叠区间 [ZD, ZG] 有交集才构成中枢
        if not (s3.high >= zd and s3.low <= zg):
            return None

        pivot_strokes = [s1, s2, s3]

        # 确定中枢方向
        direction = 'up' if s1.is_up else 'down'

        start_index = s1.start_index
        end_index = s3.end_index

        # 尝试扩展中枢（ZG/ZD保持不变，只检查重叠）
        idx = start_idx + 3
        while idx < len(self.strokes):
            next_stroke = self.strokes[idx]

            # 扩展条件：新笔与 [ZD, ZG] 有重叠
            if next_stroke.low <= zg and next_stroke.high >= zd:
                pivot_strokes.append(next_stroke)
                end_index = next_stroke.end_index
                idx += 1
            else:
                break

        # GG/DD 取所有构成笔的极值
        gg = max(s.high for s in pivot_strokes)
        dd = min(s.low for s in pivot_strokes)

        # 判断演化状态
        n_strokes = len(pivot_strokes)
        if n_strokes >= 9:
            evolution = PivotEvolution.UPGRADED
            upgrade_reason = "9段延伸升级"
        elif n_strokes > 5:
            evolution = PivotEvolution.EXTENDING
        else:
            evolution = PivotEvolution.NORMAL

        return Pivot(
            level=self.level,
            start_index=start_index,
            end_index=end_index,
            high=zg,
            low=zd,
            zg=zg,
            zd=zd,
            gg=gg,
            dd=dd,
            strokes=pivot_strokes,
            direction=direction,
            confirmed=True,
            evolution=evolution,
            segment_count=n_strokes,
            upgrade_reason=upgrade_reason if evolution == PivotEvolution.UPGRADED else '',
            evolution_level=2 if evolution == PivotEvolution.UPGRADED else 1,
        )

    def _is_within_pivot(
        self,
        stroke: Stroke,
        pivot_high: float,
        pivot_low: float
    ) -> bool:
        """
        判断笔是否与中枢区间有交集（缠论定义）

        缠论中枢延伸条件：后续笔与中枢区间有重叠即可，
        即笔的低点 <= 中枢高点 且 笔的高点 >= 中枢低点。
        不要求笔完全包含在中枢内。

        Args:
            stroke: 笔对象
            pivot_high: 中枢高点
            pivot_low: 中枢低点

        Returns:
            是否与中枢有交集
        """
        return stroke.low <= pivot_high and stroke.high >= pivot_low

    def _check_pivot_extension(self) -> None:
        """合并有重叠区间的中枢 (中枢扩展)

        检查所有中枢对(含非相邻)的ZG/ZD重叠，用Union-Find分组，
        传递性合并: A重叠B, B重叠C → A,B,C合并为一个中枢。
        合并后的中枢包含组内所有笔画，ZG/ZD取交集，GG/DD取全域极值。
        """
        n = len(self.pivots)
        if n < 2:
            return

        # Union-Find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # 检查所有中枢对: ZG/ZD核心区间是否有重叠
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = self.pivots[i], self.pivots[j]
                if p1.zg >= p2.zd and p2.zg >= p1.zd:
                    union(i, j)

        # 按根节点分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # 合并每组
        new_pivots = []
        for indices in groups.values():
            if len(indices) == 1:
                new_pivots.append(self.pivots[indices[0]])
                continue

            # 收集组内所有笔画 + 中间过渡笔画
            first = self.pivots[indices[0]]
            last = self.pivots[indices[-1]]
            start_idx = first.start_index
            end_idx = last.end_index

            combined = []
            seen = {id(s) for p in (self.pivots[i] for i in indices) for s in p.strokes}
            for s in self.pivots[indices[0]].strokes:
                combined.append(s)
            for idx in indices[1:]:
                for s in self.pivots[idx].strokes:
                    if id(s) not in seen:
                        seen.add(id(s))
                        combined.append(s)

            # 补充中间过渡笔画 (属于同一段K线范围但未被任何中枢收录)
            for s in self.strokes:
                if s.start_index >= start_idx and s.end_index <= end_idx and id(s) not in seen:
                    seen.add(id(s))
                    combined.append(s)

            combined.sort(key=lambda s: s.start_index)

            # ZG/ZD: 取组内所有中枢ZG/ZD的交集
            new_zg = min(self.pivots[i].zg for i in indices)
            new_zd = max(self.pivots[i].zd for i in indices)

            if new_zg <= new_zd:
                # 交集无效, 用前2笔重算
                if len(combined) >= 2:
                    new_zg = min(combined[0].high, combined[1].high)
                    new_zd = max(combined[0].low, combined[1].low)
                if new_zg <= new_zd:
                    # 无法构成有效中枢, 保留原始中枢不合并
                    for i in indices:
                        new_pivots.append(self.pivots[i])
                    continue

            new_gg = max(s.high for s in combined)
            new_dd = min(s.low for s in combined)

            merged = Pivot(
                level=first.level,
                start_index=start_idx,
                end_index=end_idx,
                high=new_zg,
                low=new_zd,
                zg=new_zg,
                zd=new_zd,
                gg=new_gg,
                dd=new_dd,
                strokes=combined,
                direction=first.direction,
                confirmed=True,
                extended=True,
                evolution=PivotEvolution.EXTENDING,
                segment_count=len(combined),
                upgrade_reason="两中枢重叠合并" if len(indices) > 1 else '',
                evolution_level=2 if len(combined) >= 9 else 1,
            )

            for i in indices:
                for seg in self.pivots[i].segments:
                    if seg not in merged.segments:
                        merged.segments.append(seg)

            new_pivots.append(merged)

        self.pivots = new_pivots

    def _check_dif_level(self, pivot: Pivot) -> None:
        """
        检查中枢区间内MACD DIF是否穿越0轴

        规则：
        - DIF从负变正 或 从正变负 → 本级别中枢 (sub_level=False)
        - DIF始终同侧 → 次级别中枢 (sub_level=True)

        趋势中的回调中枢（DIF不穿0轴）= 次级别
        真正的转折中枢（DIF穿0轴）= 本级别
        """
        closes = [kd.close for kd in self.kline.data]
        if len(closes) < 35:
            return

        first_stroke = pivot.strokes[0]
        last_stroke = pivot.strokes[-1]

        # 找中枢对应的K线索引区间
        start_idx = None
        end_idx = None
        for i, kd in enumerate(self.kline.data):
            dt = kd.datetime if hasattr(kd, 'datetime') else getattr(kd, 'date', None)
            if start_idx is None and dt is not None:
                try:
                    if dt >= first_stroke.start_datetime:
                        start_idx = max(0, i - 1)
                except TypeError:
                    pass
            if dt is not None:
                try:
                    if dt <= last_stroke.end_datetime:
                        end_idx = i
                except TypeError:
                    pass

        if start_idx is None or end_idx is None or end_idx - start_idx < 10:
            return

        macd_start = max(0, start_idx - 35)
        closes_seg = closes[macd_start:end_idx + 1]
        if len(closes_seg) < 35:
            return

        # 计算DIF (标准MACD参数 12, 26)
        ema12 = ema26 = closes_seg[0]
        dif_values = []
        for c in closes_seg:
            ema12 = ema12 * 11 / 12 + c / 12
            ema26 = ema26 * 25 / 26 + c / 26
            dif_values.append(ema12 - ema26)

        pivot_offset = start_idx - macd_start
        pivot_dif = dif_values[pivot_offset:]

        if len(pivot_dif) < 3:
            return

        has_positive = any(d > 0 for d in pivot_dif)
        has_negative = any(d < 0 for d in pivot_dif)

        if has_positive and has_negative:
            pivot.sub_level = False
            pivot.dif_crossed_zero = True
        else:
            pivot.sub_level = True
            pivot.dif_crossed_zero = False

    def _check_evolution(self) -> None:
        """
        对所有中枢进行完整的演化状态检查

        状态转换链：
        FORMING → NORMAL(3笔确认) → EXTENDING(>5笔,ZG/ZD不变)
                                  → EXPANDING(ZG/ZD扩大)
                                  → UPGRADED(≥9段/两中枢重叠)
                                  → ESCAPED(3买/3卖确认脱离)

        每个状态的实战意义：
        - FORMING: 中枢尚未确认，不作为买卖点参考
        - NORMAL: 标准中枢，3买/3卖可直接使用
        - EXTENDING: 延伸中枢，震荡充分，后续突破更有力
        - EXPANDING: 扩张中枢，波动加大，方向不明，需谨慎
        - UPGRADED: 已升级到高级别，应视为更高级别的中枢
        - ESCAPED: 已脱离，不再构成支撑/压力
        """
        for p in self.pivots:
            # DIF级别判定
            self._check_dif_level(p)

            n = p.segment_count or len(p.strokes)

            # 扩张检测：GG/DD是否持续扩大（后段波动>前段波动）
            is_expanding = False
            if n >= 6 and len(p.strokes) >= 6:
                half = n // 2
                first_half = p.strokes[:half]
                second_half = p.strokes[half:]
                first_range = max(s.high for s in first_half) - min(s.low for s in first_half)
                second_range = max(s.high for s in second_half) - min(s.low for s in second_half)
                if first_range > 0 and second_range > first_range * 1.3:
                    is_expanding = True

            # 状态转换（优先级从高到低）
            if n >= 9 and p.evolution != PivotEvolution.ESCAPED:
                p.evolution = PivotEvolution.UPGRADED
                p.upgrade_reason = "9段延伸升级"
                p.evolution_level = 2
            elif is_expanding and p.evolution not in (
                PivotEvolution.UPGRADED, PivotEvolution.ESCAPED
            ):
                p.evolution = PivotEvolution.EXPANDING
            elif n > 5 and p.evolution not in (
                PivotEvolution.UPGRADED, PivotEvolution.ESCAPED,
                PivotEvolution.EXPANDING,
            ):
                p.evolution = PivotEvolution.EXTENDING
            # NORMAL 保持默认值

    def get_pivots(self) -> List[Pivot]:
        """获取所有中枢"""
        return self.pivots.copy()

    def get_pivot_at(self, index: int) -> Optional[Pivot]:
        """
        获取指定K线位置所在的中枢

        Args:
            index: K线索引

        Returns:
            中枢对象
        """
        for pivot in self.pivots:
            if pivot.start_index <= index <= pivot.end_index:
                return pivot
        return None

    def get_latest_pivot(self) -> Optional[Pivot]:
        """获取最新的中枢"""
        if self.pivots:
            return self.pivots[-1]
        return None

    def check_pivot_breakout(self, pivot: Pivot, current_price: float) -> Tuple[bool, str]:
        """
        检查是否突破中枢

        Args:
            pivot: 中枢对象
            current_price: 当前价格

        Returns:
            (是否突破, 突破方向: 'up'/'down'/None)
        """
        if current_price > pivot.high:
            return (True, 'up')
        elif current_price < pivot.low:
            return (True, 'down')
        return (False, '')

    @staticmethod
    def mark_escaped(pivot: Pivot, direction: str = 'up') -> None:
        """
        标记中枢已脱离（3买/3卖确认后调用）

        Args:
            pivot: 中枢对象
            direction: 脱离方向 ('up'=3买, 'down'=3卖)
        """
        pivot.evolution = PivotEvolution.ESCAPED
        pivot.upgrade_reason = f"3买脱离" if direction == 'up' else "3卖脱离"

    def __len__(self) -> int:
        return len(self.pivots)

    def __getitem__(self, index: int) -> Pivot:
        return self.pivots[index]


def detect_pivots(
    kline: KLine,
    strokes: Optional[List[Stroke]] = None,
    level: PivotLevel = PivotLevel.DAY
) -> List[Pivot]:
    """
    便捷函数：检测K线序列的所有中枢

    Args:
        kline: K线对象
        strokes: 笔列表，None则自动生成
        level: 中枢级别

    Returns:
        中枢列表
    """
    detector = PivotDetector(kline, strokes, level)
    return detector.get_pivots()


def detect_segment_pivots(
    kline: KLine,
    segments: List[Segment],
    level: PivotLevel = PivotLevel.DAY
) -> List[Pivot]:
    """
    从线段构建中枢（线段级别中枢）

    线段级别中枢比笔级别中枢范围更精确，用于验证3买/3卖信号。
    复用3元素重叠区间逻辑，但以线段为基本单位。

    Args:
        kline: K线对象
        segments: 线段列表
        level: 中枢级别

    Returns:
        线段级别中枢列表
    """
    if len(segments) < 3:
        return []

    pivots = []
    i = 0
    while i <= len(segments) - 3:
        s1, s2, s3 = segments[i], segments[i + 1], segments[i + 2]

        # 检查方向交替
        if s1.direction == s2.direction or s2.direction == s3.direction:
            i += 1
            continue

        # ZG/ZD 由三个线段确定
        zg = min(s1.high, s2.high, s3.high)
        zd = max(s1.low, s2.low, s3.low)

        # 有效重叠区间
        if zg <= zd:
            i += 1
            continue

        # 第3个线段参与ZG/ZD计算，验证重叠已隐含在zg>zd检查中
        if zg <= zd:
            i += 1
            continue

        seg_list = [s1, s2, s3]
        direction = 'up' if s1.is_up else 'down'
        start_index = s1.start_index
        end_index = s3.end_index

        # 扩展中枢
        idx = i + 3
        while idx < len(segments):
            next_seg = segments[idx]
            if next_seg.low <= zg and next_seg.high >= zd:
                seg_list.append(next_seg)
                end_index = next_seg.end_index
                idx += 1
            else:
                break

        gg = max(s.high for s in seg_list)
        dd = min(s.low for s in seg_list)

        pivot = Pivot(
            level=level,
            start_index=start_index,
            end_index=end_index,
            high=zg,
            low=zd,
            zg=zg,
            zd=zd,
            gg=gg,
            dd=dd,
            strokes=[],  # 线段中枢不含笔列表
            direction=direction,
            confirmed=True,
            segment_count=len(seg_list),
        )
        # 存储线段引用（附加属性）
        pivot.segments = seg_list
        pivots.append(pivot)

        i += len(seg_list) - 1

    return pivots
    """
    多级别中枢分析

    同时分析不同级别的中枢结构
    """

    def __init__(self, kline_dict: dict):
        """
        初始化多级别中枢分析

        Args:
            kline_dict: 字典，key为级别名称，value为KLine对象
                {
                    '1m': KLine(...),
                    '5m': KLine(...),
                    '30m': KLine(...),
                    'day': KLine(...)
                }
        """
        self.kline_dict = kline_dict
        self.detectors = {}
        self._analyze()

    def _analyze(self) -> None:
        """分析所有级别的中枢"""
        level_map = {
            '1m': PivotLevel.MIN_1,
            '5m': PivotLevel.MIN_5,
            '30m': PivotLevel.MIN_30,
            'day': PivotLevel.DAY,
            'week': PivotLevel.WEEK,
            'month': PivotLevel.MONTH
        }

        for level_name, kline in self.kline_dict.items():
            level = level_map.get(level_name, PivotLevel.DAY)
            self.detectors[level_name] = PivotDetector(kline, level=level)

    def get_pivots_by_level(self, level: str) -> List[Pivot]:
        """
        获取指定级别的中枢

        Args:
            level: 级别名称 ('1m', '5m', '30m', 'day', 'week', 'month')

        Returns:
            中枢列表
        """
        if level in self.detectors:
            return self.detectors[level].get_pivots()
        return []

    def get_aligned_pivots(self) -> dict:
        """
        获取各级别对齐的中枢

        Returns:
            字典，key为级别，value为中枢列表
        """
        return {level: detector.get_pivots()
                for level, detector in self.detectors.items()}
