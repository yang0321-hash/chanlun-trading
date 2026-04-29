# -*- coding: utf-8 -*-
"""
缠论引擎飞书 v1.0 - 中枢演化量化实现
基于:缠论交易系统 v2.0 附录A(中枢演化量化规则)

模块:
 1. K线包含处理
 2. 顶底分型识别
 3. 笔划分
 4. 中枢识别(含延伸/扩张/升级)
 5. 买卖点初步检测
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import json


# ============================================================
# 数据结构
# ============================================================

@dataclass
class KLine:
    """原始K线"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0

    def __repr__(self):
        return f"K({self.date} O={self.open:.2f} H={self.high:.2f} L={self.low:.2f} C={self.close:.2f})"


@dataclass
class MergedKLine:
    """包含处理后的K线"""
    date_start: str
    date_end: str
    high: float
    low: float
    direction: int  # 1=向上, -1=向下, 0=初始
    raw_count: int = 1  # 合并了几根原始K线
    high_date: str = ""  # 创新高那根K线的日期
    low_date: str = ""   # 创新低那根K线的日期

    def __repr__(self):
        return f"MK({self.date_start}-{self.date_end} H={self.high:.2f}@{self.high_date} L={self.low:.2f}@{self.low_date} d={self.direction})"


class FractalType(Enum):
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class StrokeConfig:
    """
    笔划分配置(对齐通达信缠论指标选项)

    new_pen:         新笔规则,顶底分型之间至少4根K线(含分型本身)
    gap_as_pen:      跳空成笔,有缺口时不满足最小K线数也可成笔
    sub_extreme:     次高(低)点成笔,过滤幅度不足的次级分型
    sub_extreme_pct: 次高点过滤阈值,回调/反弹幅度 < 前一笔的此比例则忽略
    """
    new_pen: bool = True
    gap_as_pen: bool = True
    sub_extreme: bool = True
    sub_extreme_pct: float = 0.30  # 30%回调才算有效分型


@dataclass
class FractalStrength(Enum):
    STRONG = "strong"      # 强势分型（转折分型）：第3根突破第1根
    NORMAL = "normal"      # 标准分型
    WEAK = "weak"          # 弱势分型：第3根未突破第1根

@dataclass
class Fractal:
    """分型"""
    index: int  # 在合并K线序列中的位置
    ftype: FractalType
    value: float  # 顶=高点, 底=低点
    date: str
    strength: str = "normal"  # 分型强度: strong/normal/weak

    def __repr__(self):
        s = {'strong':'强','weak':'弱'}.get(self.strength, '')
        return f"{'顶' if self.ftype == FractalType.TOP else '底'}{s}({self.date} {self.value:.2f})"


class StrokeType(Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class Stroke:
    """笔"""
    index: int  # 笔序号
    stype: StrokeType
    start_date: str
    end_date: str
    start_value: float
    end_value: float
    high: float
    low: float
    length: int = 0  # 包含的合并K线数
    start_strength: str = "normal"  # 起始分型强度: strong/normal/weak
    end_strength: str = "normal"    # 结束分型强度: strong/normal/weak

    @property
    def amplitude(self):
        return abs(self.end_value - self.start_value)

    @property
    def strength_score(self) -> float:
        """笔的综合强度分数 (0~1)"""
        scores = {"strong": 1.0, "normal": 0.7, "weak": 0.4}
        return (scores.get(self.start_strength, 0.7) + scores.get(self.end_strength, 0.7)) / 2

    def __repr__(self):
        arrow = "↑" if self.stype == StrokeType.UP else "↓"
        return f"笔{self.index}{arrow}({self.start_date}~{self.end_date} {self.start_value:.2f}→{self.end_value:.2f})"


class PivotEvolution(Enum):
    FORMING = "forming"       # 形成中
    NORMAL = "normal"         # 标准中枢 (N=3)
    EXTENDING = "extending"   # 延伸中
    EXPANDING = "expanding"   # 扩张中
    UPGRADED = "upgraded"     # 已升级
    ESCAPED = "escaped"       # 已脱离(3买/3卖确认)


# 中枢级别判断DIF/DEA回贴0轴阈值（可由外部覆盖用于AB测试）
DIF_PROX_THRESHOLD = 0.25


@dataclass
class Pivot:
    """中枢"""
    index: int
    level: int = 1  # 1=本级别, 2=高一级别...
    sub_level: bool = True  # 默认次级别，需DIF+DEA双回贴0轴才确认本级别
    strokes: List[Stroke] = field(default_factory=list)  # 构成中枢的笔
    zg: float = 0.0
    zd: float = 0.0
    gg: float = 0.0
    dd: float = 0.0
    segment_count: int = 0  # 总段数 N
    evolution: PivotEvolution = PivotEvolution.FORMING
    upgrade_reason: str = ""  # 升级原因
    dif_crossed_zero: bool = False  # DIF是否穿越0轴

    def __repr__(self):
        lv = f"Lv{self.level}"
        if self.sub_level:
            lv += "(次级别)"
        return (f"中枢{self.index}({lv} ZG={self.zg:.2f} ZD={self.zd:.2f} "
                f"N={self.segment_count} {self.evolution.value})")

    def width(self):
        return self.zg - self.zd

    def contains_price(self, price: float) -> bool:
        return self.zd <= price <= self.zg

    def to_dict(self):
        return {
            "index": self.index,
            "level": self.level,
            "sub_level": self.sub_level,
            "zg": round(self.zg, 2),
            "zd": round(self.zd, 2),
            "gg": round(self.gg, 2),
            "dd": round(self.dd, 2),
            "width": round(self.width(), 2),
            "segment_count": self.segment_count,
            "evolution": self.evolution.value,
            "upgrade_reason": self.upgrade_reason,
            "strokes": [str(s) for s in self.strokes],
        }


@dataclass
class BuySellPoint:
    """买卖点"""
    index: int
    bp_type: str  # "1B","2B","3B","sub1B","quasi2B","subQuasi2B","pz1B","2B3B","q2B","xzd1B","1S","2S","3S"
    date: str
    price: float
    confidence: float = 0.0
    stop_loss: float = 0.0
    pivot: Optional[Pivot] = None
    reason: str = ""

    def __repr__(self):
        return f"{self.bp_type}({self.date} {self.price:.2f} conf={self.confidence:.2f})"


# ============================================================
# 1. K线包含处理
# ============================================================

def process_inclusion(klines: List[KLine]) -> List[MergedKLine]:
    if len(klines) < 2:
        return [MergedKLine(k.date, k.date, k.high, k.low, 0,
                            high_date=k.date, low_date=k.date) for k in klines]

    merged = []
    k0 = klines[0]
    merged.append(MergedKLine(k0.date, k0.date, k0.high, k0.low, 0,
                              high_date=k0.date, low_date=k0.date))

    for i in range(1, len(klines)):
        prev = merged[-1]
        curr = klines[i]

        if _is_included(prev, curr):
            direction = prev.direction
            if direction == 0:
                if len(merged) >= 2:
                    pp = merged[-2]
                    direction = 1 if pp.high < prev.high else -1
                else:
                    direction = 1 if curr.close >= curr.open else -1

            if direction == 1:
                new_high = max(prev.high, curr.high)
                new_low = max(prev.low, curr.low)
            else:
                new_high = min(prev.high, curr.high)
                new_low = min(prev.low, curr.low)

            new_high_date = prev.high_date if new_high == prev.high else curr.date
            new_low_date = prev.low_date if new_low == prev.low else curr.date
            if new_high == prev.high and new_high != curr.high:
                new_high_date = prev.high_date
            elif new_high == curr.high and new_high != prev.high:
                new_high_date = curr.date
            else:
                new_high_date = prev.high_date

            if new_low == prev.low and new_low != curr.low:
                new_low_date = prev.low_date
            elif new_low == curr.low and new_low != prev.low:
                new_low_date = curr.date
            else:
                new_low_date = prev.low_date

            merged[-1] = MergedKLine(
                prev.date_start, curr.date,
                new_high, new_low, direction,
                prev.raw_count + 1,
                high_date=new_high_date, low_date=new_low_date
            )
        else:
            if curr.close > prev.high:
                direction = 1
            elif curr.close < prev.low:
                direction = -1
            else:
                direction = 1 if curr.high > prev.high else -1
            merged.append(MergedKLine(
                curr.date, curr.date,
                curr.high, curr.low, direction,
                high_date=curr.date, low_date=curr.date
            ))

    return merged


def _is_included(a: MergedKLine, b: KLine) -> bool:
    return (a.high >= b.high and a.low <= b.low) or \
           (b.high >= a.high and b.low <= a.low)


# ============================================================
# 2. 分型识别
# ============================================================

def find_fractals(merged: List[MergedKLine]) -> List[Fractal]:
    """分型识别 + 强弱分类
    
    强势顶分型（转折）：第3根低点 < 第1根低点（向下突破力度大）
    弱势顶分型：第3根低点 > 第1根低点（反转力度弱）
    强势底分型（转折）：第3根高点 > 第1根高点（向上突破力度大）
    弱势底分型：第3根高点 < 第1根高点（反转力度弱）
    """
    fractals = []
    for i in range(1, len(merged) - 1):
        prev, curr, next_ = merged[i - 1], merged[i], merged[i + 1]

        if curr.high > prev.high and curr.high > next_.high and \
           curr.low > prev.low and curr.low > next_.low:
            # 顶分型：判断强度
            if next_.low < prev.low:
                strength = "strong"  # 第3根低点破第1根低点 → 强势顶
            elif next_.low > prev.low:
                strength = "weak"    # 第3根低点高于第1根 → 弱势顶
            else:
                strength = "normal"
            fractals.append(Fractal(i, FractalType.TOP, curr.high,
                                    curr.high_date or curr.date_end, strength))

        elif curr.low < prev.low and curr.low < next_.low and \
             curr.high < prev.high and curr.high < next_.high:
            # 底分型：判断强度
            if next_.high > prev.high:
                strength = "strong"  # 第3根高点破第1根高点 → 强势底（转折分型）
            elif next_.high < prev.high:
                strength = "weak"    # 第3根高点低于第1根 → 弱势底
            else:
                strength = "normal"
            fractals.append(Fractal(i, FractalType.BOTTOM, curr.low,
                                    curr.low_date or curr.date_end, strength))

    return fractals


def build_strokes(fractals: List[Fractal], merged: List[MergedKLine],
                  config: StrokeConfig = None) -> List[Stroke]:
    if config is None:
        config = StrokeConfig()
    if len(fractals) < 2:
        return []
    filtered = _filter_fractals_v2(fractals, merged, config)
    strokes = []
    for i in range(len(filtered) - 1):
        f1, f2 = filtered[i], filtered[i + 1]
        if f1.ftype == FractalType.BOTTOM and f2.ftype == FractalType.TOP:
            stype = StrokeType.UP
        elif f1.ftype == FractalType.TOP and f2.ftype == FractalType.BOTTOM:
            stype = StrokeType.DOWN
        else:
            continue
        start_val = f1.value
        end_val = f2.value
        start_idx = f1.index
        end_idx = f2.index
        high = max(merged[j].high for j in range(start_idx, end_idx + 1))
        low = min(merged[j].low for j in range(start_idx, end_idx + 1))
        strokes.append(Stroke(
            index=len(strokes), stype=stype,
            start_date=f1.date, end_date=f2.date,
            start_value=start_val, end_value=end_val,
            high=high, low=low,
            length=end_idx - start_idx + 1,
            start_strength=f1.strength,
            end_strength=f2.strength
        ))
    return strokes


def _has_gap(merged: List[MergedKLine], idx1: int, idx2: int) -> bool:
    if idx2 <= idx1 + 1:
        return False
    for j in range(idx1 + 1, idx2 + 1):
        if merged[j].low > merged[j - 1].high:
            return True
        if merged[j].high < merged[j - 1].low:
            return True
    return False


def _filter_fractals_v2(fractals: List[Fractal], merged: List[MergedKLine],
                         config: StrokeConfig) -> List[Fractal]:
    if not fractals:
        return []

    result = [fractals[0]]
    for f in fractals[1:]:
        last = result[-1]
        if f.ftype == last.ftype:
            if f.ftype == FractalType.TOP and f.value > last.value:
                result[-1] = f
            elif f.ftype == FractalType.BOTTOM and f.value < last.value:
                result[-1] = f
            continue
        kline_gap = f.index - last.index
        min_gap = 3 if config.new_pen else 1
        has_gap = False
        if config.gap_as_pen and kline_gap < min_gap:
            has_gap = _has_gap(merged, last.index, f.index)
        if kline_gap >= min_gap or has_gap:
            result.append(f)

    if config.sub_extreme and len(result) > 2:
        result = _filter_sub_extreme(result, config.sub_extreme_pct)
    return result


def _filter_sub_extreme(fractals: List[Fractal], threshold: float) -> List[Fractal]:
    if len(fractals) < 3:
        return fractals
    filtered = [fractals[0], fractals[1]]
    for i in range(2, len(fractals)):
        curr = fractals[i]
        prev = filtered[-1]
        prev_prev = filtered[-2]
        prev_stroke_amp = abs(prev.value - prev_prev.value)
        curr_stroke_amp = abs(curr.value - prev.value)
        if prev_stroke_amp > 0:
            ratio = curr_stroke_amp / prev_stroke_amp
            if ratio < threshold:
                if curr.ftype == prev.ftype:
                    if curr.ftype == FractalType.TOP and curr.value > prev.value:
                        filtered[-1] = curr
                    elif curr.ftype == FractalType.BOTTOM and curr.value < prev.value:
                        filtered[-1] = curr
                continue
        filtered.append(curr)
    return filtered


# ============================================================
# 4. 中枢识别(含延伸/扩张/升级)
# ============================================================

class PivotDetector:
    def __init__(self):
        self.pivots: List[Pivot] = []
        self.current_pivot: Optional[Pivot] = None
        self.pivot_index = 0
        self.raw_klines: List[KLine] = []

    def detect(self, strokes: List[Stroke], klines: List[KLine] = None) -> List[Pivot]:
        """检测中枢（缠论正确定义）
        中枢 = 进入段之后的3笔重叠
        上涨进入段(UP)后 → DN-UP-DN 三笔构成中枢
        下跌进入段(DN)后 → UP-DN-UP 三笔构成中枢
        进入段不属于中枢
        """
        if len(strokes) < 4:  # 至少需要: 进入段 + 3笔中枢
            return self.pivots

        self.raw_klines = klines or self.raw_klines

        i = 0  # i 指向进入段(entry stroke)
        while i + 3 < len(strokes):  # 需要 strokes[i+1], [i+2], [i+3] 三笔
            # strokes[i] = 进入段（不纳入中枢）
            # strokes[i+1], [i+2], [i+3] = 中枢候选3笔
            s1, s2, s3 = strokes[i + 1], strokes[i + 2], strokes[i + 3]
            zg = min(s1.high, s2.high, s3.high)
            zd = max(s1.low, s2.low, s3.low)

            if zd < zg:
                if self.current_pivot is None:
                    self._create_pivot(zg, zd, [s1, s2, s3])
                    # 如果中枢创建时就已脱离(ESCAPED)，不需要延伸
                    if self.current_pivot is not None:
                        i = self._extend_pivot(strokes, i + 4)
                    else:
                        # ESCAPED中枢已关闭，离开段可作为下一中枢的进入段
                        # 缠论中：离开段=进入段，同一笔可同时属于两个中枢
                        i = i + 3  # 离开段(strokes[i+3])作为下一中枢的进入段
                else:
                    i = self._handle_new_segment(strokes, i + 1, zg, zd, [s1, s2, s3])
            else:
                i += 1

        if self.current_pivot and self.current_pivot.evolution not in (
            PivotEvolution.UPGRADED, PivotEvolution.ESCAPED
        ):
            self.current_pivot.evolution = PivotEvolution.NORMAL if \
                self.current_pivot.segment_count <= 5 else PivotEvolution.EXTENDING

        return self.pivots

    def _create_pivot(self, zg, zd, initial_strokes):
        self.pivot_index += 1
        p = Pivot(
            index=self.pivot_index, level=1,
            strokes=list(initial_strokes),
            zg=zg, zd=zd,
            gg=max(s.high for s in initial_strokes),
            dd=min(s.low for s in initial_strokes),
            segment_count=3,
            evolution=PivotEvolution.NORMAL
        )
        # 中枢初始3笔不检查escaped——它们是构成中枢的笔画
        # 只有后续延伸的笔才检查是否脱离
        # 旧逻辑错误地检查第3笔是否突破ZG/ZD，导致中枢刚创建就被标记ESCAPED
        self._check_dif_level(p)
        self.current_pivot = p
        self.pivots.append(p)
        return p

    def _check_dif_level(self, pivot: Pivot):
        """中枢级别判断：DIF和DEA双线回贴0轴
        
        缠论标准：上涨趋势中，中枢（下上下）期间DIF和DEA都应回到0轴附近。
        双线贴近0轴 → 本级别中枢（多空真正平衡）。
        双线远离0轴 → 次级别（趋势中的小波动）。
        """
        if not self.raw_klines or len(self.raw_klines) < 35:
            return
        first_date = pivot.strokes[0].start_date
        last_date = pivot.strokes[-1].end_date
        start_idx = end_idx = None
        for i, k in enumerate(self.raw_klines):
            if start_idx is None and k.date >= first_date:
                start_idx = max(0, i - 1)
            if k.date <= last_date:
                end_idx = i
        if start_idx is None or end_idx is None or end_idx - start_idx < 10:
            return
        macd_start = max(0, start_idx - 35)
        closes = [k.close for k in self.raw_klines[macd_start:end_idx + 1]]
        if len(closes) < 35:
            return
        # 计算DIF
        ema12 = closes[0]
        ema26 = closes[0]
        dif_values = []
        for c in closes:
            ema12 = ema12 * 11 / 12 + c / 12
            ema26 = ema26 * 25 / 26 + c / 26
            dif_values.append(ema12 - ema26)
        # 计算DEA = EMA(DIF, 9)
        dea_values = [dif_values[0]]
        for d in dif_values[1:]:
            dea_values.append(dea_values[-1] * 8 / 9 + d / 9)
        
        pivot_offset = start_idx - macd_start
        pivot_dif = dif_values[pivot_offset:]
        pivot_dea = dea_values[pivot_offset:]
        if len(pivot_dif) < 3:
            return
        
        # 中枢级别判断逻辑 v2：
        # 
        # 缠论标准：本级别中枢形成时，DIF和DEA应回抽0轴附近。
        # 数学上等价于：中枢期间DIF/DEA应穿越0轴（有正有负）。
        #
        # v1 bug回顾：用 min_abs/max_abs 作为 proximity 指标，
        # 当DIF从0.13单边涨到0.79时，0.13/0.79=0.16被判为"贴近0轴"
        # 但0.13根本不接近0轴。P#10(600666)因此被误判为本级别。
        #
        # 正确方法：检查DIF/DEA是否在中枢期间穿越0轴（符号变化）
        # 穿越0轴 = 从正变负或从负变正 = 真正回贴0轴 = 本级别中枢
        # 全程同侧 = 没有回抽 = 趋势中的小波动 = 次级别
        
        dif_has_positive = any(d > 0 for d in pivot_dif)
        dif_has_negative = any(d < 0 for d in pivot_dif)
        dea_has_positive = any(d > 0 for d in pivot_dea)
        dea_has_negative = any(d < 0 for d in pivot_dea)
        
        dif_crossed = dif_has_positive and dif_has_negative
        dea_crossed = dea_has_positive and dea_has_negative
        
        if dif_crossed and dea_crossed:
            pivot.sub_level = False
            pivot.dif_crossed_zero = True
        else:
            pivot.sub_level = True
            pivot.dif_crossed_zero = False

    def _extend_pivot(self, strokes: List[Stroke], start_idx: int) -> int:
        p = self.current_pivot
        if p is None:
            return start_idx
        i = start_idx
        while i < len(strokes):
            s = strokes[i]
            if s.high >= p.zd and s.low <= p.zg:
                # 笔落入中枢区间 → 延伸
                p.strokes.append(s)
                p.segment_count += 1
                p.gg = max(p.gg, s.high)
                p.dd = min(p.dd, s.low)
                # 中枢扩张：ZG/ZD随新笔动态更新
                # ZG = min(所有笔高点), ZD = max(所有笔低点)
                p.zg = min(st.high for st in p.strokes)
                p.zd = max(st.low for st in p.strokes)
                
                if p.segment_count >= 9:
                    width_ratio = (p.zg - p.zd) / p.zg if p.zg > 0 else 0
                    if width_ratio >= 0.05:
                        self._upgrade_pivot(p, "9段延伸升级")
                        return i + 1
                if p.segment_count <= 5:
                    p.evolution = PivotEvolution.NORMAL
                else:
                    p.evolution = PivotEvolution.EXTENDING
                i += 1
            else:
                # 笔未完全落入[ZD,ZG] → 检查是否有部分重叠
                has_overlap = (s.high >= p.zd and s.low <= p.zg)
                if has_overlap:
                    # 部分重叠 → 扩张：纳入该笔，更新ZG/ZD
                    p.strokes.append(s)
                    p.segment_count += 1
                    p.gg = max(p.gg, s.high)
                    p.dd = min(p.dd, s.low)
                    p.zg = min(st.high for st in p.strokes)
                    p.zd = max(st.low for st in p.strokes)
                    p.evolution = PivotEvolution.EXPANDING
                    
                    if p.segment_count >= 9:
                        width_ratio = (p.zg - p.zd) / p.zg if p.zg > 0 else 0
                        if width_ratio >= 0.05:
                            self._upgrade_pivot(p, "9段延伸升级(扩张后)")
                            return i + 1
                    i += 1
                else:
                    # 无重叠 → 脱离中枢
                    if s.high < p.zd:
                        p.evolution = PivotEvolution.ESCAPED
                    elif s.low > p.zg:
                        p.evolution = PivotEvolution.ESCAPED
                    self.current_pivot = None
                    return i
        return i

    def _check_expansion(self, p: Pivot, s: Stroke) -> bool:
        """已合并到_extend_pivot中，保留空方法避免报错"""
        return False

    def _upgrade_pivot(self, p: Pivot, reason: str):
        p.evolution = PivotEvolution.UPGRADED
        p.level += 1
        p.upgrade_reason = reason
        previous_upgraded = [pp for pp in self.pivots
                             if pp.level == p.level and pp.index != p.index
                             and pp.evolution == PivotEvolution.UPGRADED]
        for prev in previous_upgraded:
            if prev.zg >= p.zd:
                prev.zg = min(prev.zg, p.zg)
                prev.zd = max(prev.zd, p.zd)
                prev.gg = max(prev.gg, p.gg)
                prev.dd = min(prev.dd, p.dd)
                prev.segment_count += p.segment_count
                prev.level += 1
                prev.upgrade_reason += " + 中枢重叠合并"
                p.evolution = PivotEvolution.UPGRADED
                break

    def _handle_new_segment(self, strokes, i, zg, zd, seg_strokes):
        p = self.current_pivot
        # 双向重叠检查：当前中枢的ZG>=新ZD 且 新ZG>=当前中枢ZD
        has_overlap = p and p.zg >= zd and zg >= p.zd
        # 已升级的中枢不再合并，视为已脱离
        if has_overlap and p.evolution != PivotEvolution.UPGRADED:
            self._upgrade_pivot(p, "两中枢重叠合并")
            new_p = self._create_pivot(
                min(p.zg, zg), max(p.zd, zd),
                p.strokes + seg_strokes
            )
            new_p.level = p.level + 1
            new_p.segment_count = p.segment_count + 3
            new_p.upgrade_reason = "两中枢重叠合并"
            return self._extend_pivot(strokes, i + 3)
        else:
            if p:
                p.evolution = PivotEvolution.ESCAPED
            self.current_pivot = None
            new_p = self._create_pivot(zg, zd, seg_strokes)
            return self._extend_pivot(strokes, i + 3)


# ============================================================
# 4.5 MACD面积背驰计算
# ============================================================

class MACDDivergence:
    @staticmethod
    def calc_ema(values, n):
        if len(values) < n:
            return [values[0]] * len(values)
        k = 2.0 / (n + 1)
        ema = [sum(values[:n]) / n]
        for v in values[1:]:
            ema.append(v * k + ema[-1] * (1 - k))
        return ema

    @staticmethod
    def calc_macd(closes, fast=12, slow=26, signal=9):
        if len(closes) < slow + signal:
            return None, None, None
        ema_fast = MACDDivergence.calc_ema(closes, fast)
        ema_slow = MACDDivergence.calc_ema(closes, slow)
        dif = [ema_fast[i] - ema_slow[i] for i in range(len(closes))]
        dea = MACDDivergence.calc_ema(dif, signal)
        macd_bar = [(dif[i] - dea[i]) * 2 for i in range(len(closes))]
        return dif, dea, macd_bar

    @staticmethod
    def stroke_macd_area(stroke, klines):
        start_idx = end_idx = None
        for i, k in enumerate(klines):
            if k.date == stroke.start_date and start_idx is None:
                start_idx = i
            if k.date == stroke.end_date:
                end_idx = i
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            return 0.0
        if start_idx < 30:
            return stroke.amplitude
        closes = [k.close for k in klines[:end_idx + 1]]
        _, _, macd_bar = MACDDivergence.calc_macd(closes)
        if macd_bar is None:
            return stroke.amplitude
        area = sum(macd_bar[start_idx:end_idx + 1])
        return abs(area)

    @staticmethod
    def check_divergence(curr_stroke, prev_stroke, klines):
        result = {"divergence": False, "area_ratio": 1.0, "amp_ratio": 1.0,
                  "macd_area_curr": 0.0, "macd_area_prev": 0.0, "method": "amplitude"}
        if prev_stroke.amplitude <= 0:
            return result
        amp_ratio = curr_stroke.amplitude / prev_stroke.amplitude
        result["amp_ratio"] = round(amp_ratio, 4)
        area_curr = MACDDivergence.stroke_macd_area(curr_stroke, klines)
        area_prev = MACDDivergence.stroke_macd_area(prev_stroke, klines)
        result["macd_area_curr"] = round(area_curr, 4)
        result["macd_area_prev"] = round(area_prev, 4)
        if area_prev > 0:
            area_ratio = area_curr / area_prev
            result["area_ratio"] = round(area_ratio, 4)
            if area_ratio < 1.0:
                result["divergence"] = True
                result["method"] = "macd"
        elif amp_ratio < 1.0:
            result["divergence"] = True
            result["method"] = "amplitude"
        return result


# ============================================================
# 5. 量价动力学
# ============================================================

@dataclass
class VolumeDivergenceResult:
    has_volume_divergence: bool = False
    vol_ratio: float = 1.0
    price_new_extreme: bool = False
    vol_shrink_pct: float = 0.0
    score: float = 0.0
    stroke1_vol: float = 0.0
    stroke2_vol: float = 0.0

class VolumeDynamics:
    SHRINK_THRESHOLD = 0.80
    SEVERE_THRESHOLD = 0.50

    @staticmethod
    def stroke_volume(stroke, klines):
        start_idx = end_idx = None
        for i, k in enumerate(klines):
            if k.date == stroke.start_date and start_idx is None:
                start_idx = i
            if k.date == stroke.end_date:
                end_idx = i
        if start_idx is None or end_idx is None or start_idx > end_idx:
            return 0.0
        return sum(k.volume for k in klines[start_idx:end_idx + 1])

    @staticmethod
    def stroke_avg_volume(stroke, klines):
        start_idx = end_idx = None
        for i, k in enumerate(klines):
            if k.date == stroke.start_date and start_idx is None:
                start_idx = i
            if k.date == stroke.end_date:
                end_idx = i
        if start_idx is None or end_idx is None or start_idx > end_idx:
            return 0.0
        days = end_idx - start_idx + 1
        if days <= 0:
            return 0.0
        return sum(k.volume for k in klines[start_idx:end_idx + 1]) / days

    @staticmethod
    def check_volume_divergence(curr_stroke, prev_stroke, klines):
        result = VolumeDivergenceResult()
        vol1 = VolumeDynamics.stroke_volume(prev_stroke, klines)
        vol2 = VolumeDynamics.stroke_volume(curr_stroke, klines)
        result.stroke1_vol = round(vol1, 2)
        result.stroke2_vol = round(vol2, 2)
        if vol1 <= 0 or vol2 <= 0:
            return result
        vol_ratio = vol2 / vol1
        result.vol_ratio = round(vol_ratio, 4)
        result.vol_shrink_pct = round(1.0 - vol_ratio, 4) if vol_ratio < 1.0 else 0.0
        if curr_stroke.stype == StrokeType.DOWN:
            result.price_new_extreme = curr_stroke.low < prev_stroke.low
        elif curr_stroke.stype == StrokeType.UP:
            result.price_new_extreme = curr_stroke.high > prev_stroke.high
        if result.price_new_extreme and vol_ratio < VolumeDynamics.SHRINK_THRESHOLD:
            result.has_volume_divergence = True
            raw_score = 1.0 - vol_ratio
            result.score = round(min(max(raw_score, 0.0), 1.0), 4)
        return result

    @staticmethod
    def check_pivot_shrink(pivot, klines):
        result = {"shrinking": False, "vol_sequence": [], "shrink_ratio": 1.0, "direction": "mixed"}
        if len(pivot.strokes) < 4:
            return result
        vol_seq = [VolumeDynamics.stroke_volume(s, klines) for s in pivot.strokes]
        vol_seq = [v for v in vol_seq if v > 0]
        result["vol_sequence"] = [round(v, 2) for v in vol_seq]
        if len(vol_seq) < 3:
            return result
        first_vol, last_vol = vol_seq[0], vol_seq[-1]
        if first_vol > 0:
            result["shrink_ratio"] = round(last_vol / first_vol, 4)
        decreasing_count = sum(1 for i in range(1, len(vol_seq)) if vol_seq[i] < vol_seq[i-1])
        decrease_ratio = decreasing_count / (len(vol_seq) - 1)
        if decrease_ratio >= 0.7 and result["shrink_ratio"] < 0.7:
            result["shrinking"] = True
            result["direction"] = "converging"
        elif decrease_ratio <= 0.3 and result["shrink_ratio"] > 1.3:
            result["direction"] = "expanding"
        return result


# ============================================================
# 6. 综合动力学评估器
# ============================================================

class DynamicsEvaluator:
    @staticmethod
    def evaluate(curr_stroke, prev_stroke, klines):
        result = {"diverged": False, "macd": None, "volume": None, "combined_score": 0.0, "signals": []}
        macd_result = MACDDivergence.check_divergence(curr_stroke, prev_stroke, klines)
        result["macd"] = macd_result
        vol_result = VolumeDynamics.check_volume_divergence(curr_stroke, prev_stroke, klines)
        result["volume"] = vol_result
        macd_score = 0.0
        if macd_result["divergence"]:
            macd_score = 1.0 - macd_result["area_ratio"]
            macd_score = min(max(macd_score, 0.0), 1.0)
            result["signals"].append(f"MACD背驰(面积比={macd_result['area_ratio']:.2f})")
        vol_score = vol_result.score
        if vol_result.has_volume_divergence:
            result["signals"].append(f"量价背驰(量能比={vol_result.vol_ratio:.2f}, 衰减={vol_result.vol_shrink_pct:.1%})")
        if macd_score > 0 and vol_score > 0:
            result["combined_score"] = round(min((macd_score * 0.5 + vol_score * 0.5) + 0.15, 1.0), 4)
            result["signals"].append("⚡ MACD+量价双重背驰")
        elif macd_score > 0:
            result["combined_score"] = round(macd_score * 0.7, 4)
        elif vol_score > 0:
            result["combined_score"] = round(vol_score * 0.6, 4)
        result["diverged"] = macd_result["divergence"] or vol_result.has_volume_divergence
        return result


# ============================================================
# 6b. 中枢引力场
# ============================================================

@dataclass
class GravityResult:
    pivot_index: int
    pivot_zg: float
    pivot_zd: float
    pivot_width: float
    current_price: float
    distance: float
    gravity_index: float
    status: str
    pull_probability: float

class ZhongshuGravity:
    STRONG_GRAVITY = 2.0
    MEDIUM_GRAVITY = 1.0
    WEAK_GRAVITY = 0.5

    @staticmethod
    def calc_gravity(pivot, price):
        zg, zd, width = pivot.zg, pivot.zd, pivot.width()
        if zd <= price <= zg:
            return GravityResult(pivot.index, round(zg,4), round(zd,4), round(width,4), round(price,4), 0.0, float('inf'), "inside", 0.0)
        distance = (price - zg) if price > zg else (zd - price)
        gravity = width / distance if distance > 0 else float('inf')
        if gravity >= ZhongshuGravity.STRONG_GRAVITY:
            status, pull_prob = "near", 0.7
        elif gravity >= ZhongshuGravity.MEDIUM_GRAVITY:
            status, pull_prob = "near", 0.45
        elif gravity >= ZhongshuGravity.WEAK_GRAVITY:
            status, pull_prob = "escaping", 0.25
        else:
            status, pull_prob = "escaped", 0.10
        return GravityResult(pivot.index, round(zg,4), round(zd,4), round(width,4), round(price,4), round(distance,4), round(gravity,4), status, pull_prob)

    @staticmethod
    def validate_3buy(pivot, pullback_price, current_price):
        gravity = ZhongshuGravity.calc_gravity(pivot, current_price)
        entered = pullback_price <= pivot.zd
        result = {"is_valid_3buy": False, "entered_pivot": entered, "gravity": gravity, "confidence_modifier": 0.0}
        if entered:
            result["confidence_modifier"] = -0.15
        elif gravity.status == "escaped":
            result["is_valid_3buy"] = True
            result["confidence_modifier"] = 0.10
        elif gravity.status == "escaping":
            result["is_valid_3buy"] = True
        else:
            result["confidence_modifier"] = -0.05
        return result

    @staticmethod
    def validate_3sell(pivot, pullback_price, current_price):
        gravity = ZhongshuGravity.calc_gravity(pivot, current_price)
        entered = pullback_price >= pivot.zg
        result = {"is_valid_3sell": False, "entered_pivot": entered, "gravity": gravity, "confidence_modifier": 0.0}
        if entered:
            result["confidence_modifier"] = -0.15
        elif gravity.status == "escaped":
            result["is_valid_3sell"] = True
            result["confidence_modifier"] = 0.10
        elif gravity.status == "escaping":
            result["is_valid_3sell"] = True
        else:
            result["confidence_modifier"] = -0.05
        return result


# ============================================================
# 6c. 盘整收敛/发散检测
# ============================================================

@dataclass
class ConsolidationResult:
    pivot_index: int
    stroke_count: int
    amplitude_sequence: list
    volume_sequence: list
    amp_trend: str
    vol_trend: str
    convergence_score: float
    breakout_direction: str
    breakout_imminent: bool

class ConsolidationDetector:
    @staticmethod
    def detect(pivot, klines):
        strokes = pivot.strokes
        n = len(strokes)
        result = ConsolidationResult(pivot.index, n, [], [], "mixed", "mixed", 0.0, "unclear", False)
        if n < 4:
            return result
        amp_seq = [round(s.amplitude, 4) for s in strokes]
        vol_seq = [round(VolumeDynamics.stroke_volume(s, klines), 2) for s in strokes]
        result.amplitude_sequence = amp_seq
        result.volume_sequence = vol_seq
        result.amp_trend = ConsolidationDetector._trend(amp_seq)
        result.vol_trend = ConsolidationDetector._trend(vol_seq)
        score = 0.0
        if result.amp_trend == "converging":
            score += 0.35
            if amp_seq[0] > 0:
                score += max(0, (1.0 - amp_seq[-1] / amp_seq[0]) * 0.15)
        if result.vol_trend == "converging":
            score += 0.30
            if vol_seq[0] > 0:
                score += max(0, (1.0 - vol_seq[-1] / vol_seq[0]) * 0.10)
        if result.amp_trend == "converging" and result.vol_trend == "converging":
            score += 0.10
        result.convergence_score = round(min(score, 1.0), 4)
        if n >= 3:
            last_3 = strokes[-3:]
            up_count = sum(1 for s in last_3 if s.stype == StrokeType.UP)
            down_count = sum(1 for s in last_3 if s.stype == StrokeType.DOWN)
            last_price = strokes[-1].end_value
            mid = (pivot.zg + pivot.zd) / 2
            if up_count > down_count and last_price >= mid:
                result.breakout_direction = "up"
            elif down_count > up_count and last_price <= mid:
                result.breakout_direction = "down"
        result.breakout_imminent = result.convergence_score >= 0.5 and result.amp_trend == "converging"
        return result

    @staticmethod
    def _trend(sequence):
        if len(sequence) < 3:
            return "mixed"
        decreasing = sum(1 for i in range(1, len(sequence)) if sequence[i] < sequence[i-1])
        increasing = sum(1 for i in range(1, len(sequence)) if sequence[i] > sequence[i-1])
        total = len(sequence) - 1
        if total > 0:
            if decreasing / total >= 0.7:
                return "converging"
            elif increasing / total >= 0.7:
                return "expanding"
        return "mixed"


# ============================================================
# 6d. 中枢位置系数 (P1)
# ============================================================

class PivotPosition(Enum):
    BELOW_PIVOT = "below"       # 中枢下方
    INSIDE_PIVOT = "inside"     # 中枢内部
    LEAVING_UP = "leaving_up"   # 向上离开
    THREE_BUY = "three_buy"     # 三买确认
    LEAVING_DOWN = "leaving_down"  # 向下离开
    THREE_SELL = "three_sell"   # 三卖确认

def get_pivot_position(pivot, price, strokes=None):
    """判断价格相对中枢的位置,返回 (PivotPosition, 买点系数)"""
    if pivot is None:
        return PivotPosition.INSIDE_PIVOT, 0.5

    zg, zd = pivot.zg, pivot.zd

    if zd <= price <= zg:
        return PivotPosition.INSIDE_PIVOT, 0.5

    if price > zg:
        # 价格在中枢上方 - 检查是否三买
        if strokes:
            for i, s in enumerate(strokes):
                if s.stype == StrokeType.UP and s.high > zg:
                    # 找到了向上突破的笔,检查后续回调
                    for j in range(i + 1, len(strokes)):
                        ns = strokes[j]
                        if ns.stype == StrokeType.DOWN:
                            if ns.end_value > zg:
                                return PivotPosition.THREE_BUY, 1.5
                            elif ns.end_value > zd:
                                return PivotPosition.LEAVING_UP, 1.0
                            break
                    break
        # 没找到确认笔,按距离判断
        margin = (price - zg) / zg if zg > 0 else 0
        if margin > 0.03:
            return PivotPosition.THREE_BUY, 1.5
        return PivotPosition.LEAVING_UP, 1.0

    if price < zd:
        # 价格在中枢下方 - 检查是否三卖
        if strokes:
            for i, s in enumerate(strokes):
                if s.stype == StrokeType.DOWN and s.low < zd:
                    for j in range(i + 1, len(strokes)):
                        ns = strokes[j]
                        if ns.stype == StrokeType.UP:
                            if ns.end_value < zd:
                                return PivotPosition.THREE_SELL, 1.5
                            elif ns.end_value < zg:
                                return PivotPosition.LEAVING_DOWN, 1.0
                            break
                    break
        margin = (zd - price) / zd if zd > 0 else 0
        if margin > 0.03:
            return PivotPosition.THREE_SELL, 1.5
        return PivotPosition.LEAVING_DOWN, 1.0

    return PivotPosition.INSIDE_PIVOT, 0.5


# ============================================================
# 6e. 区间套多级别分析器 v2 (P1)
# ============================================================

@dataclass
class MultiLevelResult:
    daily_modifier: float = 1.0       # 日线中枢位置系数
    trend_modifier: float = 1.0       # 趋势背驰系数
    sub_level_modifier: float = 1.0   # 子级别(30分)系数
    weekly_modifier: float = 1.0      # 周线方向系数
    final_modifier: float = 1.0       # 综合系数 = daily × trend × sub × weekly
    daily_position: str = ""          # 日线位置描述
    trend_stage: str = ""             # 趋势阶段描述
    cross_level_note: str = ""        # 跨级别验证备注

class MultiLevelAnalyzer:
    """
    区间套分析器 v2 - 逐层嵌套乘系数

    Step 0:  日线中枢位置 → 全局系数 (daily_modifier)
    Step 0b: 趋势背驰检测 → 趋势系数 (trend_modifier)
    Step 1:  日线买卖点 × daily × trend
    Step 2:  子级别区间套 → 子级别买点在日线中枢位置 → sub_level_modifier
    Step 3:  周线趋势确认 → weekly_modifier
    """

    @staticmethod
    def analyze(engine, weekly_engine=None, sub_engine=None):
        """
        engine: 日线 ChanLunEngine
        weekly_engine: 周线 ChanLunEngine (可选)
        sub_engine: 30分钟 ChanLunEngine (可选)
        """
        result = MultiLevelResult()

        # --- Step 0: 日线中枢位置 ---
        if engine.pivots and engine.strokes:
            last_pivot = engine.pivots[-1]
            last_price = engine.strokes[-1].end_value
            pos, dm = get_pivot_position(last_pivot, last_price, engine.strokes)
            result.daily_modifier = dm
            result.daily_position = f"{pos.value} (ZG={last_pivot.zg:.2f} ZD={last_pivot.zd:.2f})"
        else:
            result.daily_modifier = 0.8
            result.daily_position = "无有效中枢"

        # --- Step 0b: 趋势背驰 ---
        trend = TrendDivergenceDetector.detect(engine)
        trend_mod = TrendDivergenceDetector.get_buy_modifier(trend)
        result.trend_modifier = trend_mod
        result.trend_stage = trend.note

        # --- Step 2: 子级别(30分)区间套 ---
        if sub_engine is not None:
            sub_result = MultiLevelAnalyzer._sub_level_nesting(
                engine, sub_engine, trend
            )
            result.sub_level_modifier = sub_result[0]
            result.cross_level_note = sub_result[1]
        else:
            result.sub_level_modifier = 1.0
            result.cross_level_note = "无子级别数据"

        # --- Step 3: 周线确认 ---
        if weekly_engine is not None:
            weekly_trend = TrendDivergenceDetector.detect(weekly_engine)
            if weekly_trend.trend_direction == "up" and weekly_trend.pivot_count <= 2:
                result.weekly_modifier = 1.2
            elif weekly_trend.trend_direction == "down":
                result.weekly_modifier = 0.7
            else:
                result.weekly_modifier = 1.0
        else:
            result.weekly_modifier = 1.0

        # --- 综合系数 ---
        result.final_modifier = round(
            result.daily_modifier * result.trend_modifier *
            result.sub_level_modifier * result.weekly_modifier, 4
        )

        return result

    @staticmethod
    def _sub_level_nesting(daily_engine, sub_engine, daily_trend):
        """
        子级别买点在日线中枢中的位置 → 修正系数
        逻辑:
        - 30分买点 + 日线在中枢下方 → 30分只是反弹(×0.6)
        - 30分买点 + 日线三买已确认 → 30分是接力(×1.3)
        - 30分买点 + 日线在中枢内部 → 30分是噪音(×0.5)
        """
        if not daily_engine.pivots or not sub_engine.strokes:
            return 1.0, "数据不足"

        daily_pivot = daily_engine.pivots[-1]
        sub_price = sub_engine.strokes[-1].end_value

        _, daily_mod = get_pivot_position(daily_pivot, sub_price)

        # 跨级别趋势修正
        sub_trend = TrendDivergenceDetector.detect(sub_engine)
        sub_buy_mod, sub_sell_mod, cross_note = TrendDivergenceDetector.cross_level_validate(
            sub_trend, daily_trend
        )

        # 综合修正
        modifier = daily_mod * sub_buy_mod
        modifier = round(min(max(modifier, 0.3), 1.5), 4)  # 钳制在 [0.3, 1.5]

        note = f"日线位置系数={daily_mod:.2f} | 30分趋势修正={sub_buy_mod:.2f} | {cross_note}"
        return modifier, note


# ============================================================
# 7. 买卖点检测(含卖点 + 综合动力学)
# ============================================================

class BuySellDetector:
    def __init__(self, merged_klines=None, raw_klines=None):
        self.points = []
        self.point_index = 0
        self._used_dates = {}
        self.raw_klines = raw_klines or []

    def detect(self, strokes, pivots):
        if len(strokes) < 5 or not pivots:
            return self.points
        for p in pivots:
            if p.segment_count < 3 or len(p.strokes) < 3:
                continue
            # 统一走势方向过滤:上涨趋势中的回调不应触发任何买点
            # 上涨趋势 = 最近3个有效中枢ZG递增
            uptrend = self._is_uptrend_pullback(p, pivots)
            if not uptrend:
                self._detect_1buy(strokes, p, pivots)
                self._detect_panzheng_buy(strokes, p, pivots)
                self._detect_xiaozhuanda(strokes, p, pivots)
            # 以下买点可在上涨趋势中触发(二买/三买本身就是回调买点)
            self._detect_2buy(strokes, p)
            self._detect_3buy(strokes, p)
            self._detect_quasi2buy(strokes, p)
            self._detect_buy2_buy3_overlap(strokes, p)
            # q2B已移除(全量回测205信号, 20d胜率51.2%, 盈亏比差)
            # self._detect_quasi2buy_confirmed(strokes, p)
            # 卖点不受方向限制
            self._detect_1sell(strokes, p, pivots)
            self._detect_2sell(strokes, p)
            self._detect_3sell(strokes, p)
        self._deduplicate()
        # P2: 时间窗共振(对所有已检测买点加分)
        for p in pivots:
            if p.segment_count < 3 or len(p.strokes) < 3:
                continue
            self._detect_time_window(strokes, p)
        return self.points

    def _check_divergence(self, curr, prev):
        if not self.raw_klines:
            if prev.amplitude <= 0:
                return None
            ratio = curr.amplitude / prev.amplitude
            return {"diverged": ratio < 1.0, "ratio": round(ratio, 4), "method": "amplitude",
                    "vol_ratio": 1.0, "combined_score": round(1.0 - ratio, 4) if ratio < 1.0 else 0.0}
        dynamics = DynamicsEvaluator.evaluate(curr, prev, self.raw_klines)
        if dynamics["macd"]["method"] == "macd":
            primary_ratio = dynamics["macd"]["area_ratio"]
        else:
            primary_ratio = dynamics["macd"]["amp_ratio"]
        macd_div = dynamics["macd"]["divergence"]
        vol_div = dynamics["volume"].has_volume_divergence
        if macd_div and vol_div:
            method = "combined"
        elif vol_div:
            method = "volume"
        elif macd_div:
            method = dynamics["macd"]["method"]
        else:
            method = "none"
        return {"diverged": dynamics["diverged"], "ratio": round(primary_ratio, 4),
                "method": method, "vol_ratio": dynamics["volume"].vol_ratio,
                "combined_score": dynamics["combined_score"], "signals": dynamics["signals"]}

    def _is_duplicate(self, bp_type, pivot_index, date, price=None):
        key = (bp_type, pivot_index)
        if key in self._used_dates and self._used_dates[key] == date:
            return True
        # 价格去重：同类型同价格视为重复（用于3B多个P1产生同一P2.ZD）
        if price is not None:
            for bp in self.points:
                if bp.bp_type == bp_type and abs(bp.price - price) < 0.01:
                    return True
        return False

    def _mark_used(self, bp_type, pivot_index, date):
        self._used_dates[(bp_type, pivot_index)] = date

    def _deduplicate(self):
        seen = {}
        for bp in self.points:
            key = (bp.bp_type, bp.date)
            if key not in seen or bp.confidence > seen[key].confidence:
                seen[key] = bp
        self.points = sorted(seen.values(), key=lambda x: x.date)

    def _add_point(self, bp_type, date, price, stop_loss, pivot, confidence, reason):
        self.point_index += 1
        self.points.append(BuySellPoint(
            index=self.point_index, bp_type=bp_type, date=date,
            price=price, confidence=round(min(confidence, 1.0), 2),
            stop_loss=round(stop_loss, 2), pivot=pivot, reason=reason
        ))

    # --- 走势方向判断 ---
    def _is_uptrend_pullback(self, pivot, all_pivots):
        """判断当前中枢是否处于上涨趋势中
        上涨趋势定义:前面存在2个以上中枢,且中枢ZG/ZD序列递增
        如果是上涨趋势中的下跌段,不应触发1B/pz1B/xzd1B

        例: 中枢1(ZG=30) < 中枢2(ZG=40) < 中枢3(ZG=50) = 上涨趋势
        中枢3后的下跌只是回调,不是下跌趋势背驰

        注意:有效中枢<2时不做判断(数据不足),返回False
        """
        prev_pivots = sorted([pp for pp in all_pivots
                            if pp.index < pivot.index
                            and pp.evolution not in (PivotEvolution.UPGRADED,)
                            and not pp.sub_level],
                           key=lambda p: p.index)

        # 数据不足:中枢太少无法判断方向,保守返回False
        if len(prev_pivots) < 2:
            return False

        # 取最近3个中枢(含当前),检查ZG是否递增
        recent = prev_pivots[-2:] + [pivot]
        zgs = [p.zg for p in recent]
        zds = [p.zd for p in recent]

        # ZG严格递增 = 上涨趋势
        if zgs[0] < zgs[1] < zgs[2]:
            return True
        # ZG递增但允许最后一中枢略低(上涨趋势末端回调)
        if zgs[0] < zgs[1] and zds[0] < zds[1]:
            # 前两个中枢ZG递增,且当前中枢不破第一个中枢高点
            return True

        return False

    # --- 1买 (P0: sub1B区分) ---
    def _detect_1buy(self, strokes, pivot, all_pivots):
        if self._is_duplicate("1B", pivot.index, ""):
            return

        has_trend = any(pp.zd <= pivot.zd and pp.index < pivot.index
                        for pp in all_pivots if pp.index != pivot.index)
        
        # 检查是否在中枢震荡内部：如果存在更大的中枢包含当前买点区间，
        # 或者买点价格在中枢的[DD, ZD]区间内（中枢下沿），则是次级别波动
        is_inside_larger_pivot = False
        for pp in all_pivots:
            if pp.index == pivot.index:
                continue
            # 检查pivot是否被pp完全包含（pivot的范围在pp的[DD,GG]内）
            if pivot.zd >= pp.dd and pivot.zg <= pp.gg and len(pp.strokes) >= 6:
                is_inside_larger_pivot = True
                break
        
        # 额外检查：买点的DOWN笔低点是否仅略低于ZD但仍在中枢DD之上
        # 如果是，说明只是中枢内部的次级别回调，不是真正的趋势背驰破位
        is_pivot_internal = False

        for i, s in enumerate(strokes):
            if s.stype != StrokeType.DOWN:
                continue
            if not (s.low < pivot.zd and s.end_date > pivot.strokes[-1].end_date):
                continue
            # 检查买点DOWN笔是否仍在中枢DD之上（中枢内部回调，非趋势破位）
            if s.low >= pivot.dd:
                is_pivot_internal = True
            if i < 2:
                continue
            prev_down = None
            for j in range(i - 1, -1, -1):
                if strokes[j].stype == StrokeType.DOWN:
                    prev_down = strokes[j]
                    break
            if not prev_down or prev_down.amplitude <= 0:
                continue
            div = self._check_divergence(s, prev_down)
            if div is None or not div["diverged"]:
                continue
            if i + 1 >= len(strokes) or strokes[i + 1].stype != StrokeType.UP:
                continue
            if strokes[i + 1].amplitude <= 0:
                continue
            dyn_score = div.get("combined_score", 0.0)
            base_conf = (0.50 + min(dyn_score * 0.5, 0.40)) if dyn_score > 0 else (0.50 + min((1.0 - div["ratio"]) * 0.5, 0.35))
            # P0: 区分本级别1B和次级别1B
            # 优先级：大中枢包含 > 中枢内部回调 > has_trend > 默认sub1B
            if is_inside_larger_pivot or is_pivot_internal:
                bp_type = "sub1B"
                base_conf -= 0.08
            elif has_trend:
                bp_type = "1B"
                base_conf += 0.08
            else:
                bp_type = "sub1B"
                base_conf -= 0.12
            vol_bonus = 0.0
            if div.get("vol_ratio", 1.0) < 0.7:
                vol_bonus = 0.05
            if div.get("vol_ratio", 1.0) < 0.5:
                vol_bonus = 0.10
            base_conf += vol_bonus
            
            # 分型强度加权：买点DOWN笔+反弹UP笔的端点分型强度
            buy_stroke = s
            rebound_stroke = strokes[i + 1] if i + 1 < len(strokes) else None
            strength_scores = [buy_stroke.strength_score]
            if rebound_stroke:
                strength_scores.append(rebound_stroke.strength_score)
            avg_strength = sum(strength_scores) / len(strength_scores)
            strength_mod_1b = (avg_strength - 0.7) * 0.3
            strength_mod_1b = max(-0.05, min(0.08, strength_mod_1b))
            base_conf += strength_mod_1b
            
            stop_loss = max(s.low * 0.99, s.end_value * 0.95)
            reason_parts = [f"背驰比={div['ratio']:.2f}({div['method']})"]
            if vol_bonus > 0:
                reason_parts.append(f"量能衰减={1.0-div.get('vol_ratio',1.0):.0%}")
            if div.get("signals"):
                reason_parts.extend(div["signals"])
            if is_inside_larger_pivot:
                reason_parts.append("大中枢内部次级别背驰")
            elif is_pivot_internal:
                reason_parts.append("中枢下沿次级别背驰(未破DD)")
            elif has_trend:
                reason_parts.append("趋势确认")
            else:
                reason_parts.append("次级别盘整背驰(非本级别1B)")
            self._add_point(bp_type, s.end_date, s.end_value, stop_loss, pivot, base_conf, " ".join(reason_parts))
            self._mark_used(bp_type, pivot.index, s.end_date)
            break

    # --- 2买 ---
    def _detect_2buy(self, strokes, pivot):
        buy1s = [p for p in self.points if p.bp_type in ("1B", "sub1B") and p.pivot == pivot]
        if not buy1s:
            return
        buy1 = buy1s[-1]
        buy1_idx = None
        for i, s in enumerate(strokes):
            if s.end_date == buy1.date:
                buy1_idx = i
                break
        if buy1_idx is None:
            for i, s in enumerate(strokes):
                if s.start_date >= buy1.date:
                    buy1_idx = i
                    break
        if buy1_idx is None:
            return
        bounce_idx = None
        for i in range(buy1_idx, len(strokes)):
            s = strokes[i]
            if s.stype == StrokeType.UP and s.start_date >= buy1.date:
                if s.amplitude > buy1.price * 0.01:
                    bounce_idx = i
                    break
        if bounce_idx is None:
            return
        bounce_stroke = strokes[bounce_idx]
        callback_idx = None
        for i in range(bounce_idx + 1, len(strokes)):
            if strokes[i].stype == StrokeType.DOWN:
                callback_idx = i
                break
        if callback_idx is None:
            return
        callback_stroke = strokes[callback_idx]
        if callback_stroke.end_value <= buy1.price:
            return
        callback_ratio = (callback_stroke.amplitude / bounce_stroke.amplitude
                          if bounce_stroke.amplitude > 0 else 999)
        if callback_ratio < 0.10:
            return
        if callback_idx + 1 >= len(strokes):
            return
        if strokes[callback_idx + 1].stype != StrokeType.UP:
            return
        margin_from_1buy = (callback_stroke.end_value - buy1.price) / buy1.price
        base_conf = 0.45 + min(margin_from_1buy * 5, 0.15)
        if 0.3 <= callback_ratio <= 0.7:
            base_conf += 0.05
        elif callback_ratio > 0.80:
            base_conf -= 0.08
        elif callback_ratio < 0.15:
            base_conf -= 0.05
        if callback_stroke.end_value > pivot.zg:
            base_conf += 0.05
        bounce_pct = bounce_stroke.amplitude / buy1.price
        if bounce_pct > 0.10:
            base_conf += 0.05
        if bounce_pct > 0.20:
            base_conf += 0.05
        if callback_stroke.length < bounce_stroke.length * 0.8:
            base_conf += 0.03
        if bounce_stroke.length > callback_stroke.length * 1.2:
            base_conf += 0.05
        # P0: 二买位置判断 (above/inside/below 最后下跌中枢)
        position_mod = self._buy2_position_modifier(callback_stroke, pivot)
        base_conf += position_mod["modifier"]
        pos_tag = position_mod["tag"]

        # P1: 黄金分割回调比例
        fib_tag = ""
        if bounce_stroke.amplitude > 0 and buy1.price > 0:
            fib_ratio = callback_stroke.amplitude / (bounce_stroke.end_value - buy1.price) if (bounce_stroke.end_value - buy1.price) > 0 else 999
            if fib_ratio < 0.382:
                fib_tag = "极浅回调(最强)"
                base_conf += 0.08
            elif fib_ratio < 0.5:
                fib_tag = "浅回调(强)"
                base_conf += 0.05
            elif fib_ratio < 0.618:
                fib_tag = "标准回调"
            elif fib_ratio > 0.618:
                fib_tag = "深回调(弱)"
                base_conf -= 0.05

        # 分型强度加权：反弹UP笔+回调DOWN笔的强度
        _2b_scores = [bounce_stroke.strength_score, callback_stroke.strength_score]
        _2b_avg = sum(_2b_scores) / len(_2b_scores)
        _2b_mod = (_2b_avg - 0.7) * 0.3
        _2b_mod = max(-0.05, min(0.08, _2b_mod))
        base_conf += _2b_mod

        # 止损=二买低点×0.99（跌破回调低点=二买失效）
        stop_loss = callback_stroke.end_value * 0.99
        reason_parts = [f"回调深度={margin_from_1buy:.2%}", f"回调/反弹={callback_ratio:.0%}"]
        if callback_stroke.end_value > pivot.zg:
            reason_parts.append("2买3买重叠")
        if pos_tag:
            reason_parts.append(pos_tag)
        if fib_tag:
            reason_parts.append(f"斐波那契:{fib_tag}")
        self._add_point("2B", callback_stroke.end_date, callback_stroke.end_value, stop_loss, pivot, base_conf, " | ".join(reason_parts))

    # --- 3买 ---
    def _detect_3buy(self, strokes, pivot):
        """缠论三买（原文定义）
        
        原文：一个次级别走势离开中枢后，另一个次级别走势回抽，
        其低点不跌破ZG，则构成第三类买点。
        
        检测逻辑：
        1. 找到向上离开中枢P.ZG的UP笔（离开段）
        2. 找到离开之后的回调DN笔
        3. 回调低点 > P.ZG → 三买确认
        4. 买点 = 回调低点
        5. 止损 = P.ZG（跌破ZG = 三买失效）
        """
        if pivot.evolution in (PivotEvolution.UPGRADED,):
            return
        if pivot.sub_level:
            return

        pivot_last_date = pivot.strokes[-1].end_date

        # 1. 找离开段：中枢最后一笔之后的UP笔，high > P.ZG
        escape_idx = None
        for i, s in enumerate(strokes):
            if s.stype == StrokeType.UP and s.high > pivot.zg and s.start_date > pivot_last_date:
                escape_idx = i
                break

        if escape_idx is None:
            return

        # 2. 离开段之后，找回调DN笔
        # 回调DN笔：离开段之后的第一个DN笔，且其低点是候选买点
        for j in range(escape_idx + 1, len(strokes)):
            s = strokes[j]
            if s.stype != StrokeType.DOWN:
                continue
            
            # 回调DN笔低点 vs P.ZG
            if s.end_value > pivot.zg:
                # 回调低点 > P.ZG → 三买确认！
                
                # 确认回调后有反弹（否则可能还在下跌中）
                if j + 1 >= len(strokes):
                    break
                rebound = strokes[j + 1]
                if rebound.stype != StrokeType.UP:
                    break
                
                # 去重
                if self._is_duplicate("3B", pivot.index, s.end_date, price=s.end_value):
                    return
                
                buy_price = s.end_value
                # 止损取结构止损和价格止损的较高者
                struct_stop = pivot.zg * 0.99  # 跌破ZG=三买结构失效
                price_stop = buy_price * 0.99  # 跌破买入价=实际亏损
                stop_loss = max(struct_stop, price_stop)
                margin = (buy_price - pivot.zg) / pivot.zg if pivot.zg > 0 else 0
                
                base_conf = 0.65 + min(margin * 5, 0.25)
                
                # 分型强度加权
                strength_scores = [s.strength_score, rebound.strength_score]
                avg_str = sum(strength_scores) / len(strength_scores)
                str_mod = (avg_str - 0.7) * 0.4
                str_mod = max(-0.05, min(0.10, str_mod))
                base_conf += str_mod
                
                # 引力辅助评分
                gravity_info = ""
                if self.raw_klines:
                    gv = ZhongshuGravity.validate_3buy(pivot, s.low, s.end_value)
                    base_conf += gv["confidence_modifier"]
                    if gv["confidence_modifier"] < 0:
                        gravity_info = f" 引力压制={gv['gravity'].gravity_index:.2f}"
                    elif gv["confidence_modifier"] > 0:
                        gravity_info = f" 脱离引力={gv['gravity'].gravity_index:.2f}"
                
                reason = "三买 回调低={:.2f}>ZG={:.2f} margin={:.2%}".format(
                    buy_price, pivot.zg, margin)
                
                self._add_point("3B", s.end_date, buy_price, stop_loss, pivot,
                                base_conf, reason + gravity_info)
                return
            
            elif s.end_value <= pivot.zd:
                # 跌破ZD → 完全回到中枢 → 不是三买
                break
            else:
                # 低点在[ZD, ZG)区间 → 回到中枢内部 → 不是三买
                break

    def _check_pivot_segment_level(self, first_stroke, last_stroke):
        """检查P2中枢期间DIF/DEA是否回贴0轴
        
        用于_detect_3buy中临时构建的P2候选的级别确认。
        返回True=本级别中枢（DIF+DEA双回贴0轴），False=次级别。
        """
        if not self.raw_klines or len(self.raw_klines) < 35:
            return False  # 数据不足，保守判断为次级别
        
        first_date = first_stroke.start_date
        last_date = last_stroke.end_date
        start_idx = end_idx = None
        for i, k in enumerate(self.raw_klines):
            if start_idx is None and k.date >= first_date:
                start_idx = max(0, i - 1)
            if k.date <= last_date:
                end_idx = i
        
        if start_idx is None or end_idx is None:
            return False
        if end_idx - start_idx < 5:
            return False  # P2太短，无法确认级别
        
        macd_start = max(0, start_idx - 35)
        closes = [k.close for k in self.raw_klines[macd_start:end_idx + 1]]
        if len(closes) < 35:
            return False
        
        # 计算DIF
        ema12 = closes[0]
        ema26 = closes[0]
        dif_values = []
        for c in closes:
            ema12 = ema12 * 11 / 12 + c / 12
            ema26 = ema26 * 25 / 26 + c / 26
            dif_values.append(ema12 - ema26)
        # 计算DEA
        dea_values = [dif_values[0]]
        for d in dif_values[1:]:
            dea_values.append(dea_values[-1] * 8 / 9 + d / 9)
        
        pivot_offset = start_idx - macd_start
        pivot_dif = dif_values[pivot_offset:]
        pivot_dea = dea_values[pivot_offset:]
        
        if len(pivot_dif) < 3:
            return False
        
        # proximity = min_abs / max_abs（中枢期间DIF/DEA贴近0轴的程度）
        dif_min_abs = min(abs(d) for d in pivot_dif)
        dea_min_abs = min(abs(d) for d in pivot_dea)
        dif_max_abs = max(abs(d) for d in pivot_dif)
        dea_max_abs = max(abs(d) for d in pivot_dea)
        
        if dif_max_abs <= 0: dif_max_abs = 0.001
        if dea_max_abs <= 0: dea_max_abs = 0.001
        
        dif_proximity = dif_min_abs / dif_max_abs
        dea_proximity = dea_min_abs / dea_max_abs
        
        return dif_proximity < DIF_PROX_THRESHOLD and dea_proximity < DIF_PROX_THRESHOLD

    def _check_breakout_3buy(self, strokes, pivot, escape_idx):
        """已移除：缠论三买=第二中枢法(P2.ZD>P1.ZG)，无P2中枢不构成三买。
        仅突破GG不符合缠论定义，止损也无结构性锚定点。"""
        pass

    # 已合并到_detect_3buy（缠论原文定义：离开中枢后回调不破ZG）

    # --- 类2买 (P0: subQuasi2B区分) ---
    def _detect_quasi2buy(self, strokes, pivot):
        p_strokes = pivot.strokes
        is_sub_level = pivot.sub_level
        for i in range(1, len(p_strokes)):
            s = p_strokes[i]
            if s.stype == StrokeType.DOWN:
                for j in range(i - 1, -1, -1):
                    if p_strokes[j].stype == StrokeType.DOWN:
                        prev_low = p_strokes[j].low
                        if s.low > prev_low * 0.97 and s.low >= prev_low:
                            margin = (s.low - prev_low) / prev_low if prev_low > 0 else 0
                            if is_sub_level:
                                bp_type = "subQuasi2B"
                                base_conf = round(0.35 + min(margin * 10, 0.15), 2)
                                level_tag = "次级别中枢内的类2买"
                            else:
                                bp_type = "quasi2B"
                                base_conf = round(0.45 + min(margin * 10, 0.20), 2)
                                level_tag = "类2买"
                            if not self._is_duplicate(bp_type, pivot.index, s.end_date):
                                self._add_point(bp_type, s.end_date, s.end_value, s.low * 0.99, pivot, base_conf, f"{level_tag} margin={margin:.2%}")
                                self._mark_used(bp_type, pivot.index, s.end_date)
                        break

    # --- 1卖（缠论原文定义）---
    def _detect_1sell(self, strokes, pivot, all_pivots):
        """缠论一卖（原文定义）
        
        上涨趋势中，最后一个中枢之后的离开段UP笔出现背驰。
        - 前提：趋势（有2+中枢）→ 本级别1S
        - 盘整中单中枢背驰 → sub1S（次级别卖点）
        - 卖点 = 背驰UP笔的高点
        - 止损 = 卖出价×1.01
        """
        if pivot.sub_level:
            return
        if pivot.evolution in (PivotEvolution.UPGRADED,):
            return

        pivot_last_date = pivot.strokes[-1].end_date

        # 找离开段：中枢后的UP笔，high > P.ZG
        for i, s in enumerate(strokes):
            if s.stype != StrokeType.UP:
                continue
            if not (s.high > pivot.zg and s.start_date > pivot_last_date):
                continue
            if i < 2:
                continue

            # 找前一个UP笔做背驰比较
            prev_up = None
            for j in range(i - 1, -1, -1):
                if strokes[j].stype == StrokeType.UP:
                    prev_up = strokes[j]
                    break
            if not prev_up or prev_up.amplitude <= 0:
                continue

            # 背驰检测
            div = self._check_divergence(s, prev_up)
            if div is None or not div["diverged"]:
                continue

            # 确认：背驰UP笔之后必须有回调DN笔
            if i + 1 >= len(strokes) or strokes[i + 1].stype != StrokeType.DOWN:
                continue
            if strokes[i + 1].amplitude <= 0:
                continue

            # 趋势判断
            has_trend = any(pp.zg >= pivot.zg and pp.index < pivot.index
                           for pp in all_pivots if pp.index != pivot.index)

            sig_type = "1S" if has_trend else "sub1S"

            # 置信度
            dyn_score = div.get("combined_score", 0.0)
            base_conf = (0.50 + min(dyn_score * 0.5, 0.40)) if dyn_score > 0 else (0.50 + min((1.0 - div["ratio"]) * 0.5, 0.35))
            base_conf += 0.08 if has_trend else -0.12

            # 量能衰减加分
            vol_bonus = 0.0
            if div.get("vol_ratio", 1.0) < 0.7:
                vol_bonus = 0.05
            if div.get("vol_ratio", 1.0) < 0.5:
                vol_bonus = 0.10
            base_conf += vol_bonus

            # 分型强度加权
            dn_after = strokes[i + 1]
            str_scores = [s.strength_score, dn_after.strength_score]
            avg_str = sum(str_scores) / len(str_scores)
            str_mod = (avg_str - 0.7) * 0.3
            str_mod = max(-0.05, min(0.10, str_mod))
            base_conf += str_mod

            sell_price = s.high
            stop_loss = sell_price * 1.01

            reason_parts = [f"{sig_type} 背驰比={div['ratio']:.2f}({div['method']})"]
            if vol_bonus > 0:
                reason_parts.append(f"量能衰减={1.0-div.get('vol_ratio',1.0):.0%}")
            if div.get("signals"):
                reason_parts.extend(div["signals"])
            reason_parts.append("趋势确认" if has_trend else "盘整背驰")

            self._add_point(sig_type, s.end_date, sell_price, stop_loss, pivot, base_conf, " ".join(reason_parts))
            # 卖点操作分级：1S/sub1S=减仓
            self.points[-1].action = '减仓'
            break

    # --- 2卖（缠论原文定义）---
    def _detect_2sell(self, strokes, pivot):
        """缠论二卖（原文定义）
        
        一卖之后的下跌反弹不再创新高。
        - 卖点 = 反弹UP笔的高点
        - 止损 = 反弹高点×1.01
        """
        # 找本中枢的1S（包括sub1S）
        sell1s = [p for p in self.points if p.bp_type in ("1S", "sub1S") and p.pivot == pivot]
        if not sell1s:
            return
        sell1 = sell1s[-1]

        # 找1S在strokes中的位置
        sell1_idx = None
        for i, s in enumerate(strokes):
            if s.end_date == sell1.date:
                sell1_idx = i
                break
        if sell1_idx is None:
            for i, s in enumerate(strokes):
                if s.start_date >= sell1.date:
                    sell1_idx = i
                    break
        if sell1_idx is None:
            return

        # 1S之后找回调DN笔
        callback_idx = None
        for i in range(sell1_idx, len(strokes)):
            s = strokes[i]
            if s.stype == StrokeType.DOWN and s.start_date >= sell1.date:
                if s.amplitude > sell1.price * 0.01:
                    callback_idx = i
                    break
        if callback_idx is None:
            return
        callback_stroke = strokes[callback_idx]

        # 回调后找反弹UP笔
        bounce_idx = None
        for i in range(callback_idx + 1, len(strokes)):
            if strokes[i].stype == StrokeType.UP:
                bounce_idx = i
                break
        if bounce_idx is None:
            return
        bounce_stroke = strokes[bounce_idx]

        # 二卖核心：反弹高点 < 1S高点
        if bounce_stroke.end_value >= sell1.price:
            return

        # 反弹后必须有回调确认
        if bounce_idx + 1 >= len(strokes):
            return
        if strokes[bounce_idx + 1].stype != StrokeType.DOWN:
            return

        bounce_ratio = (bounce_stroke.amplitude / callback_stroke.amplitude
                        if callback_stroke.amplitude > 0 else 999)
        if bounce_ratio < 0.10:
            return

        margin = (sell1.price - bounce_stroke.end_value) / sell1.price
        base_conf = 0.45 + min(margin * 5, 0.15)
        if 0.3 <= bounce_ratio <= 0.7:
            base_conf += 0.05
        elif bounce_ratio > 0.80:
            base_conf -= 0.08

        if bounce_stroke.end_value < pivot.zd:
            base_conf += 0.05

        # 分型强度加权
        _2s_scores = [bounce_stroke.strength_score, callback_stroke.strength_score]
        _2s_avg = sum(_2s_scores) / len(_2s_scores)
        _2s_mod = (_2s_avg - 0.7) * 0.3
        _2s_mod = max(-0.05, min(0.08, _2s_mod))
        base_conf += _2s_mod

        sell_price = bounce_stroke.high
        stop_loss = sell_price * 1.01
        reason_parts = [f"反弹深度={margin:.2%}", f"反弹/回调={bounce_ratio:.0%}"]
        if bounce_stroke.end_value < pivot.zd:
            reason_parts.append("2卖3卖重叠")
        self._add_point("2S", bounce_stroke.end_date, sell_price, stop_loss, pivot, base_conf, " | ".join(reason_parts))
        # 卖点操作分级：2S=清仓
        self.points[-1].action = '清仓'

    # --- 3卖（缠论原文定义）---
    def _detect_3sell(self, strokes, pivot):
        """缠论三卖（原文定义）
        
        原文：一个次级别走势离开中枢后，另一个次级别走势回抽，
        其高点不升破ZD，则构成第三类卖点。
        
        检测逻辑（与三买镜像对称）：
        1. 找到向下离开中枢P.ZD的DN笔（离开段）
        2. 找到离开之后的反弹UP笔
        3. 反弹高点 < P.ZD → 三卖确认
        4. 卖点 = 反弹高点
        5. 止损 = min(ZD×1.01, 卖出价×1.01)
        """
        if pivot.evolution in (PivotEvolution.UPGRADED,):
            return
        if pivot.sub_level:
            return

        pivot_last_date = pivot.strokes[-1].end_date

        # 1. 找离开段：中枢最后一笔之后的DN笔，low < P.ZD
        escape_idx = None
        for i, s in enumerate(strokes):
            if s.stype == StrokeType.DOWN and s.low < pivot.zd and s.start_date > pivot_last_date:
                escape_idx = i
                break

        if escape_idx is None:
            return

        # 2. 离开段之后，找反弹UP笔
        for j in range(escape_idx + 1, len(strokes)):
            s = strokes[j]
            if s.stype != StrokeType.UP:
                continue

            # 反弹UP笔高点 vs P.ZD
            if s.end_value < pivot.zd:
                # 反弹高点 < ZD → 三卖确认！

                # 确认反弹后有回调（否则可能还在反弹中）
                if j + 1 >= len(strokes):
                    break
                pullback = strokes[j + 1]
                if pullback.stype != StrokeType.DOWN:
                    break

                # 去重
                if self._is_duplicate("3S", pivot.index, s.end_date, price=s.end_value):
                    return

                sell_price = s.high
                # 止损取结构止损和价格止损的较低者
                struct_stop = pivot.zd * 1.01  # 涨破ZD=三卖结构失效
                price_stop = sell_price * 1.01  # 涨破卖出价=实际亏损
                stop_loss = min(struct_stop, price_stop)
                margin = (pivot.zd - sell_price) / pivot.zd if pivot.zd > 0 else 0

                base_conf = 0.65 + min(margin * 5, 0.25)

                # 分型强度加权
                strength_scores = [s.strength_score, pullback.strength_score]
                avg_str = sum(strength_scores) / len(strength_scores)
                str_mod = (avg_str - 0.7) * 0.4
                str_mod = max(-0.05, min(0.10, str_mod))
                base_conf += str_mod

                # 引力辅助评分
                gravity_info = ""
                if self.raw_klines:
                    gv = ZhongshuGravity.validate_3sell(pivot, s.low, s.end_value)
                    base_conf += gv["confidence_modifier"]
                    if gv["confidence_modifier"] < 0:
                        gravity_info = f" 引力拉回={gv['gravity'].gravity_index:.2f}"
                    elif gv["confidence_modifier"] > 0:
                        gravity_info = f" 脱离引力={gv['gravity'].gravity_index:.2f}"

                reason = "三卖 反弹高={:.2f}<ZD={:.2f} margin={:.2%}".format(
                    sell_price, pivot.zd, margin)

                self._add_point("3S", s.end_date, sell_price, stop_loss, pivot,
                                base_conf, reason + gravity_info)
                # 卖点操作分级：3S=清仓（已脱离中枢下方）
                self.points[-1].action = '清仓'
                return

            elif s.end_value >= pivot.zg:
                # 反弹超过ZG → 完全回到中枢 → 不是三卖
                break
            else:
                # 高点在(ZD, ZG]区间 → 回到中枢内部 → 不是三卖
                break

    # --- P0: 盘整背驰买点 ---
    def _detect_panzheng_buy(self, strokes, pivot, all_pivots):
        """盘整背驰:单中枢走势 a+A+b,b段力度<a段的80%
        条件:仅1个中枢 + 离开段MACD面积<进入段的80% + 回调形成底分型
        """
        # 仅在单中枢(盘整走势)时触发
        active_pivots = [p for p in all_pivots
                        if p.evolution not in (PivotEvolution.UPGRADED,)
                        and p.level == 1]
        if len(active_pivots) > 1:
            return  # 多中枢=趋势走势,不走盘整背驰
        if pivot.sub_level:
            return
        if len(pivot.strokes) < 3:
            return

        p_strokes = pivot.strokes
        # a段 = 进入中枢的第一段下跌(中枢前的下跌笔)
        # b段 = 离开中枢的下跌笔(中枢后最后一段下跌)

        # 找中枢后的下跌笔
        after_down = None
        for i in range(len(strokes) - 1, -1, -1):
            s = strokes[i]
            if s.stype == StrokeType.DOWN and s.end_date > p_strokes[-1].end_date:
                after_down = s
                break
        if after_down is None:
            return

        # 找中枢前或中枢内的下跌笔(作为a段参考)
        before_down = None
        for s in p_strokes:
            if s.stype == StrokeType.DOWN:
                before_down = s
        if before_down is None or before_down.amplitude <= 0:
            return

        # 背驰判断:b段力度 / a段力度
        if self.raw_klines:
            dynamics = DynamicsEvaluator.evaluate(after_down, before_down, self.raw_klines)
            if not dynamics["diverged"]:
                return
            div_ratio = dynamics["macd"]["area_ratio"] if dynamics["macd"]["method"] == "macd" else dynamics["macd"]["amp_ratio"]
        else:
            amp_ratio = after_down.amplitude / before_down.amplitude
            if amp_ratio >= 0.8:
                return
            div_ratio = amp_ratio

        # 确认:离开段后必须有上升笔确认底分型
        after_idx = None
        for i, s in enumerate(strokes):
            if s.end_date == after_down.end_date and s.stype == StrokeType.DOWN:
                after_idx = i
                break
        if after_idx is None or after_idx + 1 >= len(strokes):
            return
        if strokes[after_idx + 1].stype != StrokeType.UP:
            return

        # 不与已有1买/2买重复
        if self._is_duplicate("pz1B", pivot.index, after_down.end_date):
            return

        base_conf = 0.45 + min((1.0 - div_ratio) * 0.3, 0.20)
        reason = f"盘整背驰 a+A+b 面积比={div_ratio:.2f}"

        self._add_point("pz1B", after_down.end_date, after_down.end_value,
                        after_down.end_value * 0.97, pivot, base_conf, reason)
        self._mark_used("pz1B", pivot.index, after_down.end_date)

    # --- P0: 二买三买重合 ---
    def _detect_buy2_buy3_overlap(self, strokes, pivot):
        """二买三买重合:一买低点<中枢ZD + 反弹突破ZG + 回踩不碰ZG + 不破一买
        严格定义(4条件全满足)
        """
        if pivot.sub_level:
            return
        if pivot.evolution in (PivotEvolution.UPGRADED,):
            return

        # 找一买
        buy1s = [p for p in self.points if p.bp_type in ("1B", "sub1B") and p.pivot == pivot]
        if not buy1s:
            return
        buy1 = buy1s[-1]
        buy1_price = buy1.price

        zg, zd = pivot.zg, pivot.zd

        # 条件1:一买低点 < 中枢ZD
        if buy1_price >= zd:
            return

        # 找一买后的走势:上涨笔 → 回调笔
        buy1_idx = None
        for i, s in enumerate(strokes):
            if s.end_date == buy1.date:
                buy1_idx = i
                break
        if buy1_idx is None:
            return

        # 找一买后的上涨笔(必须突破ZG)
        up_idx = None
        for i in range(buy1_idx + 1, len(strokes)):
            s = strokes[i]
            if s.stype == StrokeType.UP and s.start_date >= buy1.date:
                if s.end_value > zg:  # 条件2:凌厉突破ZG
                    up_idx = i
                    break
        if up_idx is None:
            return

        up_stroke = strokes[up_idx]

        # 找上涨后的回调笔
        cb_idx = None
        for i in range(up_idx + 1, len(strokes)):
            if strokes[i].stype == StrokeType.DOWN:
                cb_idx = i
                break
        if cb_idx is None:
            return

        cb_stroke = strokes[cb_idx]
        cb_low = cb_stroke.end_value

        # 条件3:回调不碰ZG(严格,不含容差)
        if cb_low < zg:
            return

        # 条件4:回调不破一买
        if cb_low <= buy1_price:
            return

        # 确认:回调后必须有上升笔
        if cb_idx + 1 >= len(strokes):
            return
        if strokes[cb_idx + 1].stype != StrokeType.UP:
            return

        if self._is_duplicate("2B3B", pivot.index, cb_stroke.end_date):
            return

        # 四条件全满足 → 二买三买重合
        base_conf = 0.75  # 高置信度
        if cb_low > zg * 1.02:  # 回调在ZG上方2%以上
            base_conf += 0.05
        reason = (f"二买三买重合: 一买{buy1_price:.2f}<ZD{zd:.2f}, "
                  f"突破{up_stroke.end_value:.2f}>ZG{zg:.2f}, "
                  f"回踩{cb_low:.2f}在ZG之上, 不破一买")

        self._add_point("2B3B", cb_stroke.end_date, cb_low,
                        buy1_price * 1.01, pivot, base_conf, reason)
        self._mark_used("2B3B", pivot.index, cb_stroke.end_date)

    # --- P0: 二买位置修正 ---
    def _buy2_position_modifier(self, callback_stroke, pivot):
        """二买回调低点相对最后下跌中枢的位置 → 置信度修正"""
        result = {"position": "unknown", "modifier": 0.0, "tag": ""}
        cb_low = callback_stroke.end_value
        zg, zd = pivot.zg, pivot.zd

        if cb_low > zg:
            result["position"] = "above"
            result["modifier"] = 0.08  # 中枢上方=强
            result["tag"] = "二买在中枢上方(强)"
        elif cb_low >= zd:
            result["position"] = "inside"
            result["modifier"] = -0.03  # 中枢内部=力度不确定
            result["tag"] = "二买在中枢内部"
        else:
            result["position"] = "below"
            result["modifier"] = -0.08  # 中枢下方=力度可疑
            result["tag"] = "二买在中枢下方(弱)"
        return result

    # --- P1: 类二买(二次回踩确认) ---
    def _detect_quasi2buy_confirmed(self, strokes, pivot):
        """类二买:一买→反弹→二买→上涨→二次回踩不破二买低点
        笔序列:down(一买) → up → down(二买) → up → down(类二买)
        与 _detect_quasi2buy 的区别:这个是中枢外的二次回踩确认
        """
        if pivot.sub_level:
            return
        p_strokes = pivot.strokes
        if len(p_strokes) < 5:
            return

        # 取中枢内最后5笔,检查 down-up-down-up-down 结构
        last5 = p_strokes[-5:] if len(p_strokes) >= 5 else p_strokes
        dirs = [s.stype for s in last5]
        expected = [StrokeType.DOWN, StrokeType.UP, StrokeType.DOWN, StrokeType.UP, StrokeType.DOWN]

        if dirs != expected:
            return

        buy1_low = last5[0].end_value  # 一买低点
        buy2_low = last5[2].end_value  # 二买低点
        quasi_low = last5[4].end_value  # 类二买低点

        # 条件1:二买不破一买
        if buy2_low <= buy1_low:
            return
        # 条件2:类二买不破二买
        if quasi_low <= buy2_low:
            return

        if self._is_duplicate("q2B", pivot.index, last5[4].end_date):
            return

        # 黄金分割回调比例
        up_high = last5[3].end_value  # 反弹高点
        fib_ratio = (up_high - quasi_low) / (up_high - buy2_low) if (up_high - buy2_low) > 0 else 1.0
        if fib_ratio < 0.382:
            fib_tag = "极浅回调(最强)"
            base_conf = 0.60
        elif fib_ratio < 0.5:
            fib_tag = "浅回调(强)"
            base_conf = 0.55
        elif fib_ratio < 0.618:
            fib_tag = "标准回调"
            base_conf = 0.50
        else:
            fib_tag = "深回调(弱)"
            base_conf = 0.42

        # P2+: 缩量确认 - 二次回踩笔成交量 < 反弹笔的80%
        vol_tag = ""
        if self.raw_klines:
            # 二次回踩笔(最后down) vs 反弹笔(倒数第二up)
            vol_shrink = self._check_volume_shrink(last5[4], strokes)
            if vol_shrink:
                vol_tag = " 缩量确认"
                base_conf += 0.05
            else:
                # 不缩量 → 适当降低置信度
                base_conf -= 0.03

        reason = f"类二买(二次回踩确认) 一买{buy1_low:.2f}→二买{buy2_low:.2f}→类二买{quasi_low:.2f} {fib_tag}{vol_tag}"

        self._add_point("q2B", last5[4].end_date, quasi_low,
                        buy2_low * 0.99, pivot, base_conf, reason)
        self._mark_used("q2B", pivot.index, last5[4].end_date)


    # --- P2: 三买10因子评分 ---
    def _score_buy3(self, pivot, callback_stroke, strokes, cb_idx, margin):
        """三买10因子评分体系(参考外部监控脚本check_buy3_bonus_scored)
        每项0或1分,总分0-10映射到置信度0.45-0.95
        """
        scores = {}
        total = 0
        details = []

        # 1. 回调幅度:margin > 3% = 1
        if margin > 0.03:
            scores["margin"] = 1; details.append(f"margin={margin:.1%}")
        # 2. 回调在中枢GG上方(强三买)
        if callback_stroke.end_value > pivot.gg:
            scores["above_gg"] = 1; total += 1; details.append("GG上方")
        # 3. 中枢宽度适中(不过宽,5-15笔)
        if 3 <= pivot.segment_count <= 15:
            scores["pivot_width"] = 1; total += 1; details.append(f"中枢N={pivot.segment_count}")
        # 4. 中枢引力确认脱离
        if self.raw_klines:
            gv = ZhongshuGravity.validate_3buy(pivot, callback_stroke.low, callback_stroke.end_value)
            if gv["confidence_modifier"] > 0:
                scores["gravity"] = 1; total += 1; details.append("脱离引力")
        # 5. 回调后有上升笔确认
        if cb_idx + 1 < len(strokes) and strokes[cb_idx + 1].stype == StrokeType.UP:
            scores["confirmed"] = 1; total += 1; details.append("确认笔")
        # 6. 量能:回调缩量(如果有volume数据)
        if self.raw_klines:
            vol_score = self._check_volume_shrink(callback_stroke, strokes)
            if vol_score:
                scores["volume"] = 1; total += 1; details.append("缩量回调")
        # 7. 突破力度:突破笔振幅 > 中枢宽度 * 0.5
        breakout_stroke = None
        for s in strokes:
            if s.stype == StrokeType.UP and s.high > pivot.zg and s.start_date > pivot.strokes[-1].end_date:
                breakout_stroke = s
                break
        if breakout_stroke and pivot.width() > 0:
            if breakout_stroke.amplitude > pivot.width() * 0.5:
                scores["breakout_strength"] = 1; total += 1; details.append("强突破")
        # 8. 中枢震荡收敛(后段宽度 < 前段宽度的80%)
        if len(pivot.strokes) >= 6:
            first_half = pivot.strokes[:len(pivot.strokes)//2]
            second_half = pivot.strokes[len(pivot.strokes)//2:]
            first_range = max(s.high for s in first_half) - min(s.low for s in first_half)
            second_range = max(s.high for s in second_half) - min(s.low for s in second_half)
            if first_range > 0 and second_range < first_range * 0.8:
                scores["convergence"] = 1; total += 1; details.append("收敛")
        # 9. 不碰ZG(回调低点 > ZG * 1.01)
        if callback_stroke.end_value > pivot.zg * 1.01:
            scores["clean_break"] = 1; total += 1; details.append("干净突破")
        # 10. margin也算分
        if margin > 0.03:
            total += 1

        # 映射: 0-10 → 0.45-0.95
        base_conf = 0.45 + total * 0.05
        if callback_stroke.end_value > pivot.gg:
            tag = f"强三买({total}/10)"
        else:
            tag = f"三买({total}/10)"

        return {"confidence": min(base_conf, 0.95), "tag": tag, "detail": " ".join(details)}

    def _check_volume_shrink(self, stroke, strokes):
        """检查回调笔是否缩量"""
        if not self.raw_klines:
            return False
        # 找回调笔对应的K线
        stroke_vols = [k.volume for k in self.raw_klines
                       if k.date >= stroke.start_date and k.date <= stroke.end_date and k.volume > 0]
        if not stroke_vols:
            return False
        avg_vol = sum(stroke_vols) / len(stroke_vols)
        # 找前一笔(上升笔)的量
        prev_vols = []
        for i, s in enumerate(strokes):
            if s.end_date == stroke.end_date and s.stype == stroke.stype:
                if i > 0:
                    ps = strokes[i-1]
                    prev_vols = [k.volume for k in self.raw_klines
                                if k.date >= ps.start_date and k.date <= ps.end_date and k.volume > 0]
                break
        if not prev_vols:
            return False
        prev_avg = sum(prev_vols) / len(prev_vols)
        return avg_vol < prev_avg * 0.8  # 回调量 < 反弹量的80%

    # --- P2: 小转大检测 ---
    def _detect_xiaozhuanda(self, strokes, pivot, all_pivots):
        """小转大:次级别趋势背驰引发本级别转折
        A+B+C结构,C段力度<A段的70%,且C段低点<本中枢ZD(或高点>ZG)
        """
        if pivot.sub_level:
            return
        # 只对最后一个中枢检测小转大——如果后面已有新中枢，当前中枢已"过期"
        if all_pivots and pivot.index < all_pivots[-1].index:
            return
            return
        p_strokes = pivot.strokes
        if len(p_strokes) < 5:
            return

        # 找中枢前的3段:down(进入) → up(内) → down(离开)
        # 要求离开段力度 < 进入段的70%
        before_down = None
        after_down = None
        before_up = None

        # 中枢前的下跌笔
        for s in strokes:
            if s.end_date < p_strokes[0].start_date and s.stype == StrokeType.DOWN:
                before_down = s
        # 中枢后的下跌笔
        for s in reversed(strokes):
            if s.start_date > p_strokes[-1].end_date and s.stype == StrokeType.DOWN:
                after_down = s
                break
        # 中枢前最后一个上升笔
        for s in strokes:
            if s.end_date < p_strokes[0].start_date and s.stype == StrokeType.UP:
                before_up = s

        if not before_down or not after_down:
            return

        # 背驰判断
        if before_down.amplitude > 0:
            ratio = after_down.amplitude / before_down.amplitude
        else:
            return

        if ratio >= 0.7:
            return

        # P2+: DIF线也必须背驰（力度衰减确认）
        dif_shrink = False
        if self.raw_klines:
            area_before = MACDDivergence.stroke_macd_area(before_down, self.raw_klines)
            area_after = MACDDivergence.stroke_macd_area(after_down, self.raw_klines)
            if area_before and area_after and isinstance(area_before, (int, float)) and isinstance(area_after, (int, float)):
                area_before = abs(area_before)
                area_after = abs(area_after)
                if area_before > 0 and area_after < area_before:
                    dif_shrink = True
                    dif_ratio = area_after / area_before
                else:
                    return
            else:
                dif_shrink = False
        else:
            dif_shrink = False

        # P2+: 价格必须低于中枢ZD(严格小转大定义)
        if after_down.end_value >= pivot.zd:
            return

        # 确认:C段后必须有上升笔
        after_idx = None
        for i, s in enumerate(strokes):
            if s.end_date == after_down.end_date and s.stype == StrokeType.DOWN:
                after_idx = i
                break
        if after_idx is None or after_idx + 1 >= len(strokes):
            return
        if strokes[after_idx + 1].stype != StrokeType.UP:
            return

        if self._is_duplicate("xzd1B", pivot.index, after_down.end_date):
            return

        base_conf = 0.55 + min((1.0 - ratio) * 0.3, 0.25)
        dif_tag = " DIF背驰确认" if dif_shrink else ""
        reason = f"小转大 A+B+C 力度比={ratio:.2f} 进入段={before_down.amplitude:.2f} 离开段={after_down.amplitude:.2f} 价格<{pivot.zd:.2f}{dif_tag}"

        self._add_point("xzd1B", after_down.end_date, after_down.end_value,
                        after_down.end_value * 0.97, pivot, base_conf, reason)
        self._mark_used("xzd1B", pivot.index, after_down.end_date)

    # --- P2: 时间窗共振 ---
    def _detect_time_window(self, strokes, pivot):
        """时间窗共振:对称时间窗 + 斐波那契时间窗
        检查买卖点日期是否落在关键时间窗口上
        """
        if len(pivot.strokes) < 3:
            return

        p_strokes = pivot.strokes
        # 中枢持续时间
        pivot_start = p_strokes[0].start_date
        pivot_end = p_strokes[-1].end_date
        try:
            ps = int(str(pivot_start))
            pe = int(str(pivot_end))
            pivot_bars = pe - ps  # 简化:用日期差近似
        except:
            return

        # 斐波那契关键天数: 8, 13, 21, 34, 55
        fib_days = [8, 13, 21, 34, 55]
        # 对称窗口: 中枢持续时间的0.5x, 1x, 1.5x, 2x
        sym_multipliers = [0.5, 1.0, 1.5, 2.0]

        # 对所有已检测到的买点,检查是否在时间窗上
        for bp in self.points:
            if bp.pivot != pivot:
                continue
            if bp.bp_type in ("1S", "2S", "3S"):
                continue

            try:
                bp_date = int(str(bp.date))
                pe_val = int(str(pivot_end))
            except:
                continue

            dist = bp_date - pe_val
            if dist <= 0:
                continue

            # 检查斐波那契窗口
            fib_hit = None
            for fd in fib_days:
                if abs(dist - fd) <= 2:  # ±2天容差
                    fib_hit = fd
                    break

            # 检查对称窗口
            sym_hit = None
            if pivot_bars > 0:
                for m in sym_multipliers:
                    target = int(pivot_bars * m)
                    if abs(dist - target) <= 3:
                        sym_hit = f"{m:.1f}x中枢"
                        break

            # 时间窗共振加分
            if fib_hit or sym_hit:
                tags = []
                if fib_hit:
                    tags.append(f"斐波那契{fib_hit}日")
                    bp.confidence = min(bp.confidence + 0.03, 0.98)
                if sym_hit:
                    tags.append(f"对称{sym_hit}")
                    bp.confidence = min(bp.confidence + 0.03, 0.98)
                bp.reason += f" [时间窗:{'+'.join(tags)}]"


# ============================================================
# 8. 出场系统 (P2: BIAS+ATR+跟踪止盈)
# ============================================================

class ExitSignal(Enum):
    HOLD = "hold"
    STOP_LOSS = "stop_loss"         # 硬止损
    BIAS_EXTREME = "bias_extreme"   # BIAS极值
    ATR_TRAILING = "atr_trailing"   # ATR跟踪止盈
    TRAILING_5TO3 = "trailing_5to3" # 5%→3%跟踪止盈
    SELL_POINT = "sell_point"       # 缠论卖点
    TP50 = "tp50"                   # 第1笔UP反弹完成, 止盈50%
    TP50_ZG_STOP = "tp50_zg_stop"  # 止盈50%后跌破ZG, 清仓剩余
    TP50_NO_DIV = "tp50_no_div"    # 止盈50%后3笔回调无背驰, 不回补
    TP50_REENTER = "tp50_reenter"  # 止盈50%后3笔回调背驰, 回补50%

@dataclass
class ExitResult:
    signal: ExitSignal = ExitSignal.HOLD
    exit_price: float = 0.0
    exit_date: str = ""
    reason: str = ""
    profit_pct: float = 0.0
    confidence: float = 0.0

class ExitSystem:
    """出场系统
    参数来自外部监控脚本回测优化:
    - 硬止损: 5%
    - 跟踪止盈: 盈利5%后,回撤3%即出
    - BIAS6 < -3%: 重仓20%买回信号(已做)
    - ATR止损: ATR(14) × 1.5
    """
    HARD_STOP = 0.05
    TRAILING_TRIGGER = 0.05
    TRAILING_STEP = 0.03
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 1.5
    BIAS_PERIOD = 6
    BIAS_EXTREME_SELL = 8.0   # BIAS6 > 8% → 超买出场
    BIAS_EXTREME_BUY = -3.0   # BIAS6 < -3% → 超卖出场(已有)

    @staticmethod
    def evaluate(buy_price, buy_date, stop_loss, klines, peak_price=None):
        """评估当前是否应该出场
        klines: K线列表 [KLine, ...]
        返回: ExitResult
        """
        if not klines:
            return ExitResult()

        current = klines[-1]
        current_price = current.close
        profit_pct = (current_price - buy_price) / buy_price if buy_price > 0 else 0

        result = ExitResult()

        # 1. 硬止损
        if current_price <= buy_price * (1 - ExitSystem.HARD_STOP):
            result.signal = ExitSignal.STOP_LOSS
            result.exit_price = current_price
            result.exit_date = current.date
            result.profit_pct = profit_pct
            result.reason = f"硬止损: 跌幅{profit_pct:.1%} > {ExitSystem.HARD_STOP:.0%}"
            return result

        # 2. 原始止损
        if current_price <= stop_loss:
            result.signal = ExitSignal.STOP_LOSS
            result.exit_price = current_price
            result.exit_date = current.date
            result.profit_pct = profit_pct
            result.reason = f"触及止损{stop_loss:.2f}"
            return result

        # 3. BIAS极值出场
        bias = ExitSystem._calc_bias(klines, ExitSystem.BIAS_PERIOD)
        if bias is not None and bias > ExitSystem.BIAS_EXTREME_SELL:
            result.signal = ExitSignal.BIAS_EXTREME
            result.exit_price = current_price
            result.exit_date = current.date
            result.profit_pct = profit_pct
            result.confidence = min((bias - ExitSystem.BIAS_EXTREME_SELL) / 4.0 + 0.5, 0.9)
            result.reason = f"BIAS{ExitSystem.BIAS_PERIOD}={bias:.1f}% > {ExitSystem.BIAS_EXTREME_SELL}% 超买"
            return result

        # 4. 跟踪止盈: 5%→3%
        if profit_pct >= ExitSystem.TRAILING_TRIGGER:
            if peak_price is None:
                peak_price = max(k.close for k in klines if k.date >= buy_date)
            drawdown = (peak_price - current_price) / peak_price if peak_price > 0 else 0
            if drawdown >= ExitSystem.TRAILING_STEP:
                result.signal = ExitSignal.TRAILING_5TO3
                result.exit_price = current_price
                result.exit_date = current.date
                result.profit_pct = profit_pct
                result.confidence = 0.7
                result.reason = f"跟踪止盈: 最高{peak_price:.2f} 回撤{drawdown:.1%} ≥ {ExitSystem.TRAILING_STEP:.0%}"
                return result

        # 5. ATR跟踪止损
        atr = ExitSystem._calc_atr(klines, ExitSystem.ATR_PERIOD)
        if atr is not None and atr > 0:
            if peak_price is None:
                peak_price = max(k.close for k in klines if k.date >= buy_date)
            atr_stop = peak_price - atr * ExitSystem.ATR_MULTIPLIER
            if current_price <= atr_stop and profit_pct > 0:
                result.signal = ExitSignal.ATR_TRAILING
                result.exit_price = current_price
                result.exit_date = current.date
                result.profit_pct = profit_pct
                result.confidence = 0.65
                result.reason = f"ATR止损: {atr_stop:.2f} (ATR={atr:.2f}×{ExitSystem.ATR_MULTIPLIER})"
                return result

        return result

    @staticmethod
    def _calc_bias(klines, period=6):
        """计算BIAS指标"""
        closes = [k.close for k in klines]
        if len(closes) < period + 1:
            return None
        ma = sum(closes[-period:]) / period
        if ma == 0:
            return None
        return (closes[-1] - ma) / ma * 100

    @staticmethod
    def _calc_atr(klines, period=14):
        """计算ATR"""
        if len(klines) < period + 1:
            return None
        trs = []
        for i in range(len(klines) - period, len(klines)):
            k = klines[i]
            prev_close = klines[i-1].close
            tr = max(k.high - k.low, abs(k.high - prev_close), abs(k.low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs)

    # ===== 分批止盈+背驰回补 (AB测试验证: 胜率99%, 均收+14.15%) =====
    
    @staticmethod
    def evaluate_staged_exit(buy_price, buy_date, zg, klines, eng_strokes, 
                              position=1.0, phase='holding'):
        """分批止盈+背驰回补评估
        
        策略逻辑(AB测试验证):
        1. 买入100%, 硬止损=跌破ZG清仓
        2. 第1笔UP反弹完成 → 止盈50% (锁利)
        3. 等3笔回调(DN-UP-DN):
           - MACD背驰(后段面积<前段) → 回补50%
           - 无背驰 → 不回补, 保持50%仓位
        4. 回补后再破ZG → 清仓
        
        Args:
            buy_price: 3B买入价
            buy_date: 买入日期
            zg: 中枢ZG(结构止损位)
            klines: 全部K线
            eng_strokes: 引擎的笔列表
            position: 当前仓位(0~1.0)
            phase: 当前阶段 'holding'/'tp50'/'reloaded'/'no_div'
        
        Returns:
            dict: {
                'signal': ExitSignal,
                'action': 'hold'/'sell50'/'sell_all'/'buy50',
                'price': 建议价格,
                'reason': 说明,
                'new_phase': 新阶段,
                'new_position': 新仓位,
                'cash_profit': 已锁定的现金收益(%),
            }
        """
        stop_loss = max(zg * 0.99, buy_price * 0.99)
        current_price = klines[-1].close
        current_date = klines[-1].date
        
        result = {
            'signal': ExitSignal.HOLD,
            'action': 'hold',
            'price': current_price,
            'reason': '',
            'new_phase': phase,
            'new_position': position,
            'cash_profit': 0.0,
        }
        
        # 找买入DN笔的索引
        buy_dn_idx = None
        for si, s in enumerate(eng_strokes):
            if s.end_date == buy_date and s.stype == StrokeType.DOWN:
                buy_dn_idx = si
                break
        
        if buy_dn_idx is None:
            return result
        
        # 后续笔: rebound(DN+1) → dn1(DN+2) → up2(DN+3) → dn2(DN+4)
        rebound_s = eng_strokes[buy_dn_idx + 1] if buy_dn_idx + 1 < len(eng_strokes) else None
        dn1_s = eng_strokes[buy_dn_idx + 2] if buy_dn_idx + 2 < len(eng_strokes) else None
        up2_s = eng_strokes[buy_dn_idx + 3] if buy_dn_idx + 3 < len(eng_strokes) else None
        dn2_s = eng_strokes[buy_dn_idx + 4] if buy_dn_idx + 4 < len(eng_strokes) else None
        
        # 验证笔方向
        if rebound_s and rebound_s.stype != StrokeType.UP: rebound_s = None
        if dn1_s and dn1_s.stype != StrokeType.DOWN: dn1_s = None
        if up2_s and up2_s.stype != StrokeType.UP: up2_s = None
        if dn2_s and dn2_s.stype != StrokeType.DOWN: dn2_s = None
        
        # === 阶段1: holding ===
        if phase == 'holding':
            # 检查止损
            if current_price <= stop_loss:
                result.update({
                    'signal': ExitSignal.STOP_LOSS,
                    'action': 'sell_all',
                    'price': max(stop_loss, current_price),
                    'reason': 'holding阶段跌破ZG止损, 清仓100%',
                    'new_position': 0.0,
                })
                return result
            
            # 检查第1笔UP反弹是否完成
            if rebound_s and current_date >= rebound_s.end_date:
                tp_price = rebound_s.high
                tp_ret = (tp_price / buy_price - 1) * 100
                result.update({
                    'signal': ExitSignal.TP50,
                    'action': 'sell50',
                    'price': tp_price,
                    'reason': '第1笔UP反弹完成(%.2f→%.2f, +%d%%), 止盈50%%锁利' % (
                        buy_price, tp_price, int(tp_ret)),
                    'new_phase': 'tp50',
                    'new_position': 0.5,
                    'cash_profit': tp_ret * 0.5,
                })
                return result
        
        # === 阶段2: tp50 (已止盈50%, 等3笔回调) ===
        elif phase == 'tp50':
            # 检查止损(清仓剩余50%)
            if current_price <= stop_loss:
                result.update({
                    'signal': ExitSignal.TP50_ZG_STOP,
                    'action': 'sell_all',
                    'price': max(stop_loss, current_price),
                    'reason': '止盈50%%后跌破ZG, 清仓剩余50%%',
                    'new_position': 0.0,
                })
                return result
            
            # 检查3笔回调是否完成
            if dn2_s and current_date >= dn2_s.end_date and dn1_s and up2_s:
                # 计算背驰: 后半段MACD面积 vs 前半段
                is_divergence = ExitSystem._check_pullback_divergence(
                    klines, buy_date, dn1_s, up2_s, dn2_s)
                
                reentry_price = dn2_s.end_value
                reentry_ret = (reentry_price / buy_price - 1) * 100
                
                if is_divergence:
                    result.update({
                        'signal': ExitSignal.TP50_REENTER,
                        'action': 'buy50',
                        'price': reentry_price,
                        'reason': '3笔回调背驰(面积缩小), 回补50%% @%.2f (%+.1f%%)' % (
                            reentry_price, reentry_ret),
                        'new_phase': 'reloaded',
                        'new_position': 1.0,
                    })
                else:
                    result.update({
                        'signal': ExitSignal.TP50_NO_DIV,
                        'action': 'hold',
                        'price': current_price,
                        'reason': '3笔回调无背驰, 不回补, 保持50%%仓位',
                        'new_phase': 'no_div',
                    })
                return result
        
        # === 阶段3: reloaded (回补后满仓) ===
        elif phase == 'reloaded':
            if current_price <= stop_loss:
                result.update({
                    'signal': ExitSignal.TP50_ZG_STOP,
                    'action': 'sell_all',
                    'price': max(stop_loss, current_price),
                    'reason': '回补后跌破ZG, 清仓100%',
                    'new_position': 0.0,
                })
                return result
        
        # === 阶段4: no_div (未回补, 保持50%) ===
        elif phase == 'no_div':
            if current_price <= stop_loss:
                result.update({
                    'signal': ExitSignal.TP50_ZG_STOP,
                    'action': 'sell_all',
                    'price': max(stop_loss, current_price),
                    'reason': '未回补状态跌破ZG, 清仓剩余50%',
                    'new_position': 0.0,
                })
                return result
        
        return result
    
    @staticmethod
    def _check_pullback_divergence(klines, buy_date, dn1, up2, dn2):
        """检查3笔回调是否有背驰
        
        比较两段MACD面积:
        - 第1段: 买入点到dn1结束
        - 第2段: up2开始到dn2结束
        后段面积 < 前段面积 = 背驰
        """
        # 找K线索引
        s1_idx = s2_start = s2_end = None
        for ki, k in enumerate(klines):
            if s1_idx is None and k.date >= buy_date:
                s1_idx = ki
            if s2_start is None and k.date >= up2.start_date:
                s2_start = ki
            if k.date >= dn2.end_date:
                s2_end = ki
                break
        
        if s1_idx is None or s2_start is None or s2_end is None:
            return False
        if s2_end - s1_idx < 10:
            return False
        
        # 计算MACD
        macd_start = max(0, s1_idx - 35)
        closes = [k.close for k in klines[macd_start:s2_end + 1]]
        if len(closes) < 35:
            return False
        
        ema12 = closes[0]
        ema26 = closes[0]
        dif = []
        for c in closes:
            ema12 = ema12 * 11/12 + c/12
            ema26 = ema26 * 25/26 + c/26
            dif.append(ema12 - ema26)
        dea = [dif[0]]
        for d in dif[1:]:
            dea.append(dea[-1] * 8/9 + d/9)
        macd_bar = [(dif[i] - dea[i]) * 2 for i in range(len(dif))]
        
        offset = macd_start
        a1_start = s1_idx - offset
        mid = s2_start - offset
        a2_end = s2_end - offset
        
        if a1_start >= len(macd_bar) or a2_end >= len(macd_bar):
            return False
        
        area_first = sum(abs(m) for m in macd_bar[a1_start:mid+1])
        area_second = sum(abs(m) for m in macd_bar[mid+1:a2_end+1])
        
        return area_second < area_first and area_first > 0



# ============================================================
# 7b. 趋势背驰检测器 (P0: 新增)
# ============================================================

class TrendStage(Enum):
    NO_TREND = "no_trend"
    SINGLE_PIVOT = "single_pivot"
    MULTI_PIVOT = "multi_pivot"
    EXHAUSTION = "exhaustion"

@dataclass
class TrendDivergenceResult:
    stage: TrendStage = TrendStage.NO_TREND
    pivot_count: int = 0
    trend_direction: str = ""
    macd_shrinking: bool = False
    exhaustion_risk: float = 0.0
    note: str = ""

class TrendDivergenceDetector:
    @staticmethod
    def detect(engine):
        result = TrendDivergenceResult()
        if not engine.pivots or len(engine.pivots) < 1:
            result.stage = TrendStage.NO_TREND
            result.note = "中枢不足,无趋势"
            return result
        pivots = engine.pivots
        last_pivot = pivots[-1]
        if engine.strokes:
            last_stroke = engine.strokes[-1]
            trend_dir = "up" if last_stroke.stype == StrokeType.UP else "down"
        else:
            trend_dir = ""
        result.trend_direction = trend_dir
        same_dir_count = 0
        for p in pivots:
            p_strokes = p.strokes
            if not p_strokes or len(p_strokes) < 2:
                continue
            last_p_stroke = p_strokes[-1]
            for s in engine.strokes:
                if s.index > last_p_stroke.index:
                    if trend_dir == "up" and s.stype == StrokeType.UP:
                        same_dir_count += 1
                    elif trend_dir == "down" and s.stype == StrokeType.DOWN:
                        same_dir_count += 1
                    break
        if same_dir_count == 0:
            same_dir_count = len(pivots)
        result.pivot_count = same_dir_count

        # MACD衰减检测
        up_strokes = [s for s in engine.strokes if s.stype == StrokeType.UP]
        down_strokes = [s for s in engine.strokes if s.stype == StrokeType.DOWN]
        if trend_dir == "up" and len(up_strokes) >= 2:
            if up_strokes[-2].amplitude > 0 and up_strokes[-1].amplitude < up_strokes[-2].amplitude * 0.8:
                result.macd_shrinking = True
        elif trend_dir == "down" and len(down_strokes) >= 2:
            if down_strokes[-2].amplitude > 0 and down_strokes[-1].amplitude < down_strokes[-2].amplitude * 0.8:
                result.macd_shrinking = True

        if same_dir_count <= 1:
            result.stage = TrendStage.SINGLE_PIVOT
            result.note = f"单中枢趋势初期({trend_dir})"
        elif same_dir_count == 2:
            if result.macd_shrinking:
                result.stage = TrendStage.EXHAUSTION
                result.exhaustion_risk = 0.6
                result.note = f"双中枢趋势({trend_dir}) + MACD衰减 → 背驰预警"
            else:
                result.stage = TrendStage.MULTI_PIVOT
                result.exhaustion_risk = 0.1
                result.note = f"双中枢趋势确认({trend_dir}),方向明确"
        else:
            if result.macd_shrinking:
                result.stage = TrendStage.EXHAUSTION
                result.exhaustion_risk = 0.8
                result.note = f"多中枢({same_dir_count}个)趋势({trend_dir}) + MACD衰减 → 强背驰预警"
            else:
                result.stage = TrendStage.MULTI_PIVOT
                result.exhaustion_risk = 0.3
                result.note = f"多中枢({same_dir_count}个)趋势确认({trend_dir})"
        if result.macd_shrinking:
            result.exhaustion_risk = min(result.exhaustion_risk + 0.2, 1.0)
            result.note += " | 末段幅度衰减"
        return result

    @staticmethod
    def get_buy_modifier(trend_result):
        count = trend_result.pivot_count
        if trend_result.trend_direction == "up":
            if count <= 1: return 1.0
            elif count == 2: return 0.6 if trend_result.macd_shrinking else 1.2
            elif count == 3: return 0.4 if trend_result.macd_shrinking else 0.9
            else: return 0.3 if trend_result.macd_shrinking else 0.7
        elif trend_result.trend_direction == "down":
            if count <= 1: return 1.0
            elif count == 2: return 1.3 if trend_result.macd_shrinking else 1.1
            else: return 1.5 if trend_result.macd_shrinking else 1.2
        return 1.0

    @staticmethod
    def get_sell_modifier(trend_result):
        count = trend_result.pivot_count
        if trend_result.trend_direction == "up":
            if count <= 1: return 1.0
            elif count == 2: return 1.5 if trend_result.macd_shrinking else 0.8
            else: return 1.8 if trend_result.macd_shrinking else 0.9
        elif trend_result.trend_direction == "down":
            if count <= 1: return 1.0
            elif count == 2: return 0.6 if trend_result.macd_shrinking else 0.9
            else: return 0.4 if trend_result.macd_shrinking else 0.8
        return 1.0

    @staticmethod
    def cross_level_validate(sub_trend, parent_trend):
        base_buy = TrendDivergenceDetector.get_buy_modifier(sub_trend)
        base_sell = TrendDivergenceDetector.get_sell_modifier(sub_trend)
        if parent_trend is None:
            return base_buy, base_sell, "无父级别趋势数据"
        if parent_trend.pivot_count <= 1:
            health_factor = 0.5
            parent_note = "日线趋势初期/健康"
        elif parent_trend.pivot_count == 2 and not parent_trend.macd_shrinking:
            health_factor = 0.5
            parent_note = "日线趋势中后期但无背驰"
        elif parent_trend.pivot_count >= 2 and parent_trend.macd_shrinking:
            health_factor = 1.5
            parent_note = "日线MACD衰减,背驰风险"
        else:
            health_factor = 1.0
            parent_note = "日线趋势中性"
        if sub_trend.pivot_count >= 2:
            adjusted_buy = base_buy + (1.0 - base_buy) * (1.0 - health_factor)
            adjusted_sell = base_sell * health_factor
            note = f"{parent_note} | 30分{sub_trend.pivot_count}中枢 买{base_buy:.2f}→{adjusted_buy:.2f} 卖{base_sell:.2f}→{adjusted_sell:.2f}"
        else:
            adjusted_buy, adjusted_sell = base_buy, base_sell
            note = f"{parent_note} | 30分单中枢,无需跨级别修正"
        return adjusted_buy, adjusted_sell, note


# ============================================================
# 引擎主类
# ============================================================

class ChanLunEngine:
    def __init__(self):
        self.merged_klines = []
        self.fractals = []
        self.strokes = []
        self.pivots = []
        self.buy_sell_points = []
        self.multi_level = None  # 区间套分析结果

    def analyze(self, klines, weekly_klines=None, sub_klines=None):
        self.merged_klines = process_inclusion(klines)
        self.fractals = find_fractals(self.merged_klines)
        self.stroke_config = StrokeConfig()
        self.strokes = build_strokes(self.fractals, self.merged_klines, self.stroke_config)
        detector = PivotDetector()
        self.pivots = detector.detect(self.strokes, klines=klines)
        bs_detector = BuySellDetector(raw_klines=klines)
        self.buy_sell_points = bs_detector.detect(self.strokes, self.pivots)

        # P1: 区间套多级别分析
        weekly_engine = None
        sub_engine = None
        if weekly_klines:
            weekly_engine = ChanLunEngine.__new__(ChanLunEngine)
            weekly_engine.__init__()
            weekly_engine.merged_klines = process_inclusion(weekly_klines)
            weekly_engine.fractals = find_fractals(weekly_engine.merged_klines)
            weekly_engine.strokes = build_strokes(weekly_engine.fractals, weekly_engine.merged_klines, StrokeConfig())
            weekly_engine.pivots = PivotDetector().detect(weekly_engine.strokes, klines=weekly_klines)
            weekly_engine.buy_sell_points = []
        if sub_klines:
            sub_engine = ChanLunEngine.__new__(ChanLunEngine)
            sub_engine.__init__()
            sub_engine.merged_klines = process_inclusion(sub_klines)
            sub_engine.fractals = find_fractals(sub_engine.merged_klines)
            sub_engine.strokes = build_strokes(sub_engine.fractals, sub_engine.merged_klines, StrokeConfig())
            sub_engine.pivots = PivotDetector().detect(sub_engine.strokes, klines=sub_klines)
            sub_engine.buy_sell_points = []

        self.multi_level = MultiLevelAnalyzer.analyze(
            self, weekly_engine=weekly_engine, sub_engine=sub_engine
        )

        return self.get_result()

    def get_result(self):
        result = {
            "summary": {
                "raw_klines": len(self.merged_klines),
                "merged_klines": sum(1 for m in self.merged_klines),
                "fractals": len(self.fractals),
                "strokes": len(self.strokes),
                "pivots": len(self.pivots),
                "buy_sell_points": len(self.buy_sell_points),
            },
            "pivots": [p.to_dict() for p in self.pivots],
            "buy_sell_points": [
                {"type": bp.bp_type, "date": bp.date, "price": bp.price,
                 "confidence": bp.confidence, "stop_loss": bp.stop_loss,
                 "reason": bp.reason, "action": getattr(bp, 'action', ''),
                 "pivot": bp.pivot.to_dict() if bp.pivot else None}
                for bp in self.buy_sell_points
            ],
            "strokes": [
                {"index": s.index, "type": s.stype.value,
                 "start": f"{s.start_date} {s.start_value:.2f}",
                 "end": f"{s.end_date} {s.end_value:.2f}",
                 "amplitude": round(s.amplitude, 2)}
                for s in self.strokes
            ],
        }
        if self.multi_level:
            ml = self.multi_level
            result["multi_level"] = {
                "daily_modifier": ml.daily_modifier,
                "trend_modifier": ml.trend_modifier,
                "sub_level_modifier": ml.sub_level_modifier,
                "weekly_modifier": ml.weekly_modifier,
                "final_modifier": ml.final_modifier,
                "daily_position": ml.daily_position,
                "trend_stage": ml.trend_stage,
                "cross_level_note": ml.cross_level_note,
            }
        return result


# ============================================================
# 辅助函数
# ============================================================

def klines_from_tuples(data, has_ohlc=False):
    klines = []
    for d in data:
        if len(d) == 2:
            klines.append(KLine(date=d[0], open=d[1], high=d[1], low=d[1], close=d[1]))
        elif len(d) == 5:
            klines.append(KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4]))
        elif len(d) >= 6:
            klines.append(KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4], volume=d[5]))
    return klines


def print_analysis(result):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("缠论分析结果")
    print("=" * 60)
    s = result["summary"]
    print(f"\n[统计] 原始K线{s['raw_klines']}根 | 分型{s['fractals']}个 | 笔{s['strokes']}段 | 中枢{s['pivots']}个 | 买卖点{s['buy_sell_points']}个")
    for bp in result.get("buy_sell_points", []):
        tag = {"1B": "[1买]", "2B": "[2买]", "3B": "[3买]", "sub1B": "[次级别1买]",
               "quasi2B": "[类2买]", "subQuasi2B": "[次级别类2买]",
               "pz1B": "[盘整背驰]", "2B3B": "[二买三买重合]", "q2B": "[类2买确认]", "xzd1B": "[小转大]",
               "1S": "[1卖]", "2S": "[2卖]", "3S": "[3卖]"}.get(bp["type"], "[?]")
        print(f"  {tag} {bp['type']} | {bp['date']} | 价格{bp['price']:.2f} | 置信度{bp['confidence']:.2f} | 止损{bp['stop_loss']:.2f}")
        print(f"     原因: {bp['reason']}")
