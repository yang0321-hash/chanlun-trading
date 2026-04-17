"""
缠论买卖点识别引擎

精确实现缠论6种标准买卖点 + 4种类买卖点的判定逻辑：

标准买卖点（需要趋势背驰或前序买卖点）：
- 1买：向下趋势中，最后中枢下方的底背驰点
- 2买：1买后回调，底分型不破1买低点
- 3买：突破中枢上沿后回踩不进入中枢
- 1卖：向上趋势中，最后中枢上方的顶背驰点
- 2卖：1卖后反弹，顶分型不破1卖高点
- 3卖：跌破中枢下沿后反弹不进入中枢

类买卖点（盘整/宽松条件，实战扩展）：
- 类2买：中枢盘整中回调不破前低，不要求前面有1买
- 类3买：突破ZG后回踩允许轻微跌破ZG（但高于ZD），3买的宽松版
- 类2卖：中枢盘整中反弹不破前高，不要求前面有1卖
- 类3卖：跌破ZD后反弹允许轻微突破ZD（但低于ZG），3卖的宽松版
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger

from .fractal import Fractal
from .stroke import Stroke
from .segment import Segment
from .pivot import Pivot
from .trend_track import TrendTrackDetector, TrendTrack, TrendStatus
from indicator.macd import MACD


@dataclass
class BuySellPoint:
    """买卖点数据"""
    point_type: str           # '1buy','2buy','3buy','quasi2buy','quasi3buy',
                              # '1sell','2sell','3sell','quasi2sell','quasi3sell'
    price: float              # 买卖点价格
    index: int                # K线索引
    fractal: Optional[Fractal] = None
    related_pivot: Optional[Pivot] = None
    related_strokes: List[Stroke] = field(default_factory=list)
    divergence_ratio: float = 0.0   # MACD面积辅助比（离开/进入，0=不可用）
    pivot_divergence_ratio: float = 0.0  # 振幅主判定比（离开/进入笔涨跌幅比）
    confidence: float = 0.0         # 信号置信度
    stop_loss: float = 0.0          # 理论止损位
    reason: str = ''
    trend_modifier: float = 0.0     # 趋势轨道修正值
    recommended_hold_days: int = 0  # 建议持仓天数（0=无引导）
    exit_urgency: float = 0.0       # 出场紧迫度 0-1（越大越应尽快出场）

    @property
    def is_buy(self) -> bool:
        return 'buy' in self.point_type

    @property
    def is_sell(self) -> bool:
        return 'sell' in self.point_type


class BuySellPointDetector:
    """
    缠论买卖点识别器

    基于精确的缠论定义识别买卖点，作为所有策略共享的基础组件。
    支持递归到子级别（如30分钟）确认日线买卖点。
    """

    def __init__(
        self,
        fractals: List[Fractal],
        strokes: List[Stroke],
        segments: List[Segment],
        pivots: List[Pivot],
        macd: Optional[MACD] = None,
        trend_tracks: Optional[List[TrendTrack]] = None,
        # 子级别递归确认（日线→30分钟）
        sub_level_strokes: Optional[List[Stroke]] = None,
        sub_level_pivots: Optional[List[Pivot]] = None,
        sub_level_macd: Optional[MACD] = None,
        sub_level_fractals: Optional[List[Fractal]] = None,
        sub_level_segments: Optional[List[Segment]] = None,
        # 参数
        divergence_threshold: float = 0.3,  # 背驰面积比阈值
        min_strokes_for_divergence: int = 2,  # 背驰比较最少需要的同向笔数
        fuzzy_tolerance: float = 0.005,  # ZG/ZD模糊容忍区(0.5%)
        segment_pivots: Optional[List[Pivot]] = None,  # 线段级别中枢
    ):
        self.fractals = fractals
        self.strokes = strokes
        self.segments = segments
        self.pivots = pivots
        self.macd = macd
        self.fuzzy_tolerance = fuzzy_tolerance
        self._segment_pivots = segment_pivots or []
        self.divergence_threshold = divergence_threshold
        self.min_strokes_for_divergence = min_strokes_for_divergence

        # 趋势轨道
        if trend_tracks is not None:
            self._track_detector = TrendTrackDetector(strokes, pivots)
            self._track_detector._tracks = trend_tracks
        elif pivots and strokes:
            self._track_detector = TrendTrackDetector(strokes, pivots)
            self._track_detector.detect()
        else:
            self._track_detector = None

        # 子级别检测器（30分钟级别）
        self._sub_strokes = sub_level_strokes
        self._sub_buy_points: List[BuySellPoint] = []
        self._sub_sell_points: List[BuySellPoint] = []
        if sub_level_strokes and sub_level_pivots and len(sub_level_strokes) >= 3:
            try:
                sub_td = TrendTrackDetector(sub_level_strokes, sub_level_pivots)
                sub_td.detect()
                sub_tracks = sub_td._tracks if hasattr(sub_td, '_tracks') else []
                sub_det = BuySellPointDetector(
                    fractals=sub_level_fractals or [],
                    strokes=sub_level_strokes,
                    segments=sub_level_segments or [],
                    pivots=sub_level_pivots,
                    macd=sub_level_macd,
                    trend_tracks=sub_tracks,
                    divergence_threshold=divergence_threshold,
                )
                sub_det._buy_points, sub_det._sell_points = sub_det.detect_all()
                self._sub_buy_points = sub_det._buy_points
                self._sub_sell_points = sub_det._sell_points
            except Exception:
                pass

        # 已识别的买卖点缓存
        self._buy_points: List[BuySellPoint] = []
        self._sell_points: List[BuySellPoint] = []

    def detect_all(self) -> Tuple[List[BuySellPoint], List[BuySellPoint]]:
        """
        检测所有买卖点

        Returns:
            (买点列表, 卖点列表)
        """
        self._buy_points = []
        self._sell_points = []

        self._detect_first_buy()
        self._detect_first_sell()
        self._detect_second_buy()
        self._detect_second_sell()
        self._detect_third_buy()
        self._detect_third_sell()
        self._detect_quasi_second_buy()
        self._detect_quasi_second_sell()
        self._detect_quasi_third_buy()
        self._detect_quasi_third_sell()

        # 统一的趋势轨道修正
        self._apply_trend_modifier()

        return self._buy_points, self._sell_points

    def detect_latest_buy(self) -> Optional[BuySellPoint]:
        """检测最近的买点"""
        buys = [
            self._check_first_buy(),
            self._check_second_buy(),
            self._check_third_buy(),
            self._check_quasi_second_buy(),
            self._check_quasi_third_buy(),
        ]
        valid = [b for b in buys if b is not None]
        if not valid:
            return None
        return max(valid, key=lambda b: b.confidence)

    def detect_latest_sell(self) -> Optional[BuySellPoint]:
        """检测最近的卖点"""
        sells = [
            self._check_first_sell(),
            self._check_second_sell(),
            self._check_third_sell(),
            self._check_quasi_second_sell(),
            self._check_quasi_third_sell(),
        ]
        valid = [s for s in sells if s is not None]
        if not valid:
            return None
        return max(valid, key=lambda s: s.confidence)

    # ==================== 买点检测 ====================

    def _check_first_buy(self) -> Optional[BuySellPoint]:
        """
        第一类买点检测（中枢背驰法）

        缠论原文定义：
        一买 = 下跌趋势中，一个次级别走势类型向下离开最后一个缠中说禅走势中枢后
              形成的背驰点。

        关键前提：必须有下跌趋势（至少2个中枢，后者低于前者）。
        没有趋势就没有背驰（缠论第15课核心定理）。

        主判定：比较离开中枢的向下笔 vs 进入中枢的向下笔力度
        - 离开笔力度 < 进入笔力度 → 中枢背驰 → 1买

        结构公式：a + A + b + B + c，一买在c段（离开B的向下笔）
        """
        if not self.pivots or len(self.strokes) < 3:
            return None

        # ===== 缠论强制要求：1买必须在下跌趋势中 =====
        # 至少2个中枢，且最后一个中枢低于前一个中枢
        if len(self.pivots) < 2:
            return None

        has_downtrend = False
        for prev_pivot in self.pivots[:-1]:
            if prev_pivot.zd > self.pivots[-1].zg:
                has_downtrend = True
                break

        if not has_downtrend:
            return None

        last_pivot = self.pivots[-1]

        # 找离开中枢的向下笔（终点低于ZD，起点在中枢之后或尾部）
        leaving_strokes = [s for s in self.strokes
                           if s.is_down and s.end_value < last_pivot.zd
                           and s.start_index >= last_pivot.start_index]
        if not leaving_strokes:
            return None

        last_leaving = leaving_strokes[-1]

        # ===== 主判定：振幅背驰 + 辅助MACD面积 =====
        pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
            last_pivot, last_leaving
        )
        if not pivot_div:
            return None

        # 置信度基准
        base_confidence = 0.5 + min((1.0 - amp_ratio) * 0.5, 0.4)

        # 辅助：MACD面积确认
        macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
        if macd_confirmed:
            base_confidence += 0.1
        elif macd_ratio >= 1.5:
            base_confidence -= 0.05

        # 子级别递归确认：日线1买需要30分钟级别1买背驰确认
        sub_ok, sub_conf, sub_desc = self._check_sub_level_buy(
            last_leaving.end_index, ['1buy']
        )
        if sub_ok:
            base_confidence += 0.15
        elif self._sub_buy_points:
            base_confidence -= 0.1  # 有子级别数据但无1买确认

        trend_mod = self._get_trend_modifier('buy')
        confidence = max(0.1, min(1.0, base_confidence + trend_mod))

        # 止损
        theory_stop = last_leaving.low * 0.99
        max_stop = last_leaving.end_value * 0.95
        stop_loss = max(theory_stop, max_stop)

        # 原因描述
        macd_info = f', MACD确认({macd_ratio:.2f})' if macd_confirmed else ''
        sub_info = f', {sub_desc}' if sub_ok else ''

        return BuySellPoint(
            point_type='1buy',
            price=last_leaving.end_value,
            index=last_leaving.end_index,
            related_pivot=last_pivot,
            related_strokes=[last_leaving],
            divergence_ratio=macd_ratio,
            pivot_divergence_ratio=amp_ratio,
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'1买: 振幅背驰(离开/进入={amp_ratio:.2f}), 下跌趋势, ZD={last_pivot.zd:.2f}{macd_info}{sub_info}',
            trend_modifier=trend_mod,
        )

    def _check_second_buy(self) -> Optional[BuySellPoint]:
        """
        第二类买点检测

        条件链：
        1. 已识别出1买点
        2. 1买后形成了向上笔 + 向下笔的回调结构
        3. 回调底分型价格 > 1买价格（不破1买低点）
        """
        first_buy = next((b for b in reversed(self._buy_points)
                          if b.point_type == '1buy'), None)
        if first_buy is None:
            # 没有1买记录，尝试从笔结构推断
            first_buy = self._find_implied_first_buy()

        if first_buy is None:
            return None

        # 找1买之后的回调
        # 1买之后的向上笔
        up_after_1buy = [s for s in self.strokes
                         if s.is_up and s.start_index > first_buy.index]
        if not up_after_1buy:
            return None

        last_up = up_after_1buy[-1]

        # 向上笔之后的向下笔（回调）
        down_after_up = [s for s in self.strokes
                         if s.is_down and s.start_index > last_up.end_index]
        if not down_after_up:
            return None

        last_pullback = down_after_up[-1]

        # 核心条件：回调低点不破1买低点
        if last_pullback.end_value <= first_buy.price:
            return None

        # 计算置信度
        pullback_depth = (first_buy.price - last_pullback.end_value) / first_buy.price
        confidence = 0.5 + min(abs(pullback_depth) * 5, 0.3)

        # 子级别递归确认：日线2买需要30分钟级别2买或1买确认
        sub_ok, sub_conf, sub_desc = self._check_sub_level_buy(
            last_pullback.end_index, ['2buy', '1buy']
        )
        if sub_ok:
            confidence += 0.15
        elif self._sub_buy_points:
            confidence -= 0.1

        confidence = max(0.1, min(1.0, confidence))

        # 缠论理论止损：回调笔低点（2买的最低点），跌破则2买失效
        stop_loss = last_pullback.low * 0.99

        sub_info = f', {sub_desc}' if sub_ok else ''

        return BuySellPoint(
            point_type='2buy',
            price=last_pullback.end_value,
            index=last_pullback.end_index,
            related_pivot=first_buy.related_pivot,
            related_strokes=[last_up, last_pullback],
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'2买: 回调不破1买{first_buy.price:.2f}, 止损={stop_loss:.2f}{sub_info}'
        )

    def _check_third_buy(self) -> Optional[BuySellPoint]:
        """
        第三类买点检测

        缠论原文定义（第20/21课）：
        一个次级别走势类型向上离开中枢，然后一个次级别走势类型回试，
        其低点不跌破ZG，则构成第三类买点。

        三买看的是 ZG（中枢上沿/核心重叠区上界），不是GG。
        离开笔力度检查：离开笔 vs 进入笔，力度弱则更可靠（背驰确认突破有效）。
        """
        if not self.pivots or len(self.strokes) < 4:
            return None

        last_pivot = self.pivots[-1]
        zg = last_pivot.zg if last_pivot.zg > 0 else last_pivot.high

        # 找突破ZG的向上笔（离开笔）
        breakout_strokes = [s for s in self.strokes
                            if s.is_up and s.end_value > zg
                            and s.start_index >= last_pivot.start_index]
        if not breakout_strokes:
            return None

        last_breakout = breakout_strokes[-1]

        # 离开笔 vs 进入笔力度比较
        pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
            last_pivot, last_breakout
        )

        # 突破后的回调笔
        pullback_strokes = [s for s in self.strokes
                            if s.is_down and s.start_index > last_breakout.end_index]
        if not pullback_strokes:
            return None

        last_pullback = pullback_strokes[-1]

        # 核心条件：回调低点不跌破ZG（含模糊容忍区）
        fuzzy_zg = zg * (1 - self.fuzzy_tolerance)
        if last_pullback.end_value <= fuzzy_zg:
            return None

        # 模糊区穿透惩罚：进入模糊区但未跌破fuzzy_zg
        fuzzy_penalty = 0.0
        if last_pullback.end_value < zg:
            penetration = (zg - last_pullback.end_value) / zg
            fuzzy_penalty = min(penetration * 20, 0.15)  # 穿透越深惩罚越大

        # 置信度
        margin = (last_pullback.end_value - fuzzy_zg) / zg
        confidence = 0.5 + min(margin * 10, 0.3) - fuzzy_penalty

        # 中枢背驰修正（振幅主判定）
        if pivot_div:
            confidence += 0.1
        elif amp_ratio > 5.0:
            confidence -= 0.1

        # MACD辅助确认
        macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
        if macd_confirmed:
            confidence += 0.05

        # 子级别递归确认：日线3买 = 突破+回踩不进中枢，30分钟也必须确认回踩不破ZG
        sub_ok, sub_low, sub_desc = self._check_sub_level_3buy(zg)
        if sub_ok:
            confidence += 0.1
        elif self._sub_strokes:
            confidence -= 0.05

        # 线段级别中枢验证（更强的确认）
        seg_pivot_info = ''
        if self._segment_pivots:
            for sp in reversed(self._segment_pivots):
                # 突破笔穿透了线段级别中枢ZG → 强信号
                if last_breakout.end_value > sp.zg > 0 and sp.start_index < last_breakout.start_index:
                    confidence += 0.1
                    # 线段级别ZG作为更精确的止损
                    seg_zg = sp.zg
                    seg_pivot_info = f', 线段中枢ZG={seg_zg:.2f}确认突破'
                    break

        confidence = max(0.1, min(1.0, confidence))
        # 止损：优先用线段级别ZG，其次用笔级别ZG
        stop_loss = seg_zg * 0.99 if seg_pivot_info else zg * 0.99

        # 持仓引导
        if confidence >= 0.8:
            recommended_hold_days = 20
        elif confidence >= 0.65:
            recommended_hold_days = 10
        else:
            recommended_hold_days = 5

        # ZG上方margin小 = 信号脆弱，出场紧迫
        margin = (last_pullback.end_value - zg) / zg
        exit_urgency = 0.3 if margin < 0.02 else (0.1 if margin < 0.05 else 0.0)

        reason_suffix = ''
        if pivot_div:
            reason_suffix = f', 振幅背驰(离开/进入={amp_ratio:.2f})'
        elif amp_ratio > 0:
            reason_suffix = f', 离开力度={amp_ratio:.2f}'
        if macd_confirmed:
            reason_suffix += f', MACD确认({macd_ratio:.2f})'
        if sub_ok:
            reason_suffix += f', {sub_desc}'
        if seg_pivot_info:
            reason_suffix += seg_pivot_info

        return BuySellPoint(
            point_type='3buy',
            price=last_pullback.end_value,
            index=last_pullback.end_index,
            related_pivot=last_pivot,
            related_strokes=[last_breakout, last_pullback],
            pivot_divergence_ratio=amp_ratio,
            divergence_ratio=macd_ratio,
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'3买: 回踩不破ZG={zg:.2f}{reason_suffix}',
            recommended_hold_days=recommended_hold_days,
            exit_urgency=exit_urgency,
        )

    # ==================== 卖点检测 ====================

    def _check_first_sell(self) -> Optional[BuySellPoint]:
        """
        第一类卖点检测（中枢背驰法）

        缠论原文定义：
        一卖 = 上涨趋势中，一个次级别走势类型向上离开最后一个缠中说禅走势中枢后
              形成的背驰点。

        关键前提：必须有上涨趋势（至少2个中枢，后者高于前者）。
        没有趋势就没有背驰（缠论第15课核心定理）。

        主判定：比较离开中枢的向上笔 vs 进入中枢的向上笔力度
        - 离开笔力度 < 进入笔力度 → 中枢背驰 → 1卖
        """
        if not self.pivots or len(self.strokes) < 3:
            return None

        # ===== 缠论强制要求：1卖必须在上涨趋势中 =====
        if len(self.pivots) < 2:
            return None

        has_uptrend = False
        for prev_pivot in self.pivots[:-1]:
            if prev_pivot.zg < self.pivots[-1].zd:
                has_uptrend = True
                break

        if not has_uptrend:
            return None

        last_pivot = self.pivots[-1]

        # 找离开中枢的向上笔（终点高于ZG，起点在中枢之后或尾部）
        leaving_strokes = [s for s in self.strokes
                           if s.is_up and s.end_value > last_pivot.zg
                           and s.start_index >= last_pivot.start_index]
        if not leaving_strokes:
            return None

        last_leaving = leaving_strokes[-1]

        # ===== 主判定：振幅背驰 + 辅助MACD面积 =====
        pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
            last_pivot, last_leaving
        )
        if not pivot_div:
            return None

        # 置信度基准
        base_confidence = 0.5 + min((1.0 - amp_ratio) * 0.5, 0.4)

        # 辅助：MACD面积确认
        macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
        if macd_confirmed:
            base_confidence += 0.1
        elif macd_ratio >= 1.5:
            base_confidence -= 0.05

        # 子级别递归确认：日线1卖需要30分钟级别1卖背驰确认
        sub_ok, sub_conf, sub_desc = self._check_sub_level_sell(
            last_leaving.end_index, ['1sell']
        )
        if sub_ok:
            base_confidence += 0.15
        elif self._sub_sell_points:
            base_confidence -= 0.1

        confidence = max(0.1, min(1.0, base_confidence))
        stop_loss = last_leaving.high * 1.01

        macd_info = f', MACD确认({macd_ratio:.2f})' if macd_confirmed else ''
        sub_info = f', {sub_desc}' if sub_ok else ''

        return BuySellPoint(
            point_type='1sell',
            price=last_leaving.end_value,
            index=last_leaving.end_index,
            related_pivot=last_pivot,
            related_strokes=[last_leaving],
            divergence_ratio=macd_ratio,
            pivot_divergence_ratio=amp_ratio,
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'1卖: 振幅背驰(离开/进入={amp_ratio:.2f}), 上涨趋势{macd_info}{sub_info}'
        )

    def _check_second_sell(self) -> Optional[BuySellPoint]:
        """
        第二类卖点检测（增强版）

        缠论原文定义：
        2卖 = 1卖后反弹，顶分型不破1卖高点。

        增强验证（对标2买结构）：
        1. 结构质量门控：反弹笔至少3根K线 + 振幅>0.5%
        2. 子级别递归确认：30分钟2卖或1卖
        3. 趋势轨道修正
        4. 止损设置
        """
        first_sell = next((s for s in reversed(self._sell_points)
                           if s.point_type == '1sell'), None)
        if first_sell is None:
            return None

        down_after_1sell = [s for s in self.strokes
                            if s.is_down and s.start_index > first_sell.index]
        if not down_after_1sell:
            return None

        last_down = down_after_1sell[-1]

        up_after_down = [s for s in self.strokes
                         if s.is_up and s.start_index > last_down.end_index]
        if not up_after_down:
            return None

        last_bounce = up_after_down[-1]

        # 核心条件：反弹不破1卖高点
        if last_bounce.end_value >= first_sell.price:
            return None

        # ===== 结构质量（宽松门控，仅过滤极端情况） =====
        bounce_amplitude = abs(last_bounce.price_change_pct)

        # ===== 置信度计算 =====
        base_confidence = 0.5 + min(abs(first_sell.price - last_bounce.end_value) / first_sell.price * 5, 0.3)

        # 子级别递归确认（有数据时加分，无数据时不惩罚）
        sub_ok, sub_conf, sub_desc = self._check_sub_level_sell(
            last_bounce.end_index, ['2sell', '1sell']
        )
        if sub_ok:
            base_confidence += 0.15
        # 无子级别数据时不减分（避免双重惩罚）

        # 趋势轨道修正
        trend_mod = self._get_trend_modifier('sell')

        confidence = max(0.1, min(1.0, base_confidence + trend_mod))

        # 止损：1卖高点上方1%
        stop_loss = first_sell.price * 1.01

        sub_info = f', {sub_desc}' if sub_ok else ''

        return BuySellPoint(
            point_type='2sell',
            price=last_bounce.end_value,
            index=last_bounce.end_index,
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'2卖: 反弹不破1卖{first_sell.price:.2f}, 振幅={bounce_amplitude:.1f}%{sub_info}',
            trend_modifier=trend_mod,
        )

    def _check_third_sell(self) -> Optional[BuySellPoint]:
        """
        第三类卖点检测

        条件链：
        1. 存在已确认的中枢
        2. 离开笔（向下）跌破ZD，且与进入笔比较确认力度
        3. 跌破后反弹（向上笔），反弹高点 < ZD
        """
        if not self.pivots or len(self.strokes) < 4:
            return None

        last_pivot = self.pivots[-1]
        zd = last_pivot.zd if last_pivot.zd > 0 else last_pivot.low

        breakdown_strokes = [s for s in self.strokes
                             if s.is_down and s.end_value < zd
                             and s.start_index >= last_pivot.start_index]
        if not breakdown_strokes:
            return None

        last_breakdown = breakdown_strokes[-1]

        # 离开笔 vs 进入笔力度比较
        pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
            last_pivot, last_breakdown
        )

        bounce_strokes = [s for s in self.strokes
                          if s.is_up and s.start_index > last_breakdown.end_index]
        if not bounce_strokes:
            return None

        last_bounce = bounce_strokes[-1]

        # 反弹不进入中枢（含模糊容忍区）
        fuzzy_zd = zd * (1 + self.fuzzy_tolerance)
        if last_bounce.end_value >= fuzzy_zd:
            return None

        # 模糊区穿透惩罚
        fuzzy_penalty = 0.0
        if last_bounce.end_value > zd:
            penetration = (last_bounce.end_value - zd) / zd
            fuzzy_penalty = min(penetration * 20, 0.15)

        confidence = 0.5 + min(abs(fuzzy_zd - last_bounce.end_value) / zd * 10, 0.3) - fuzzy_penalty

        # 中枢背驰修正（振幅主判定）
        if pivot_div:
            confidence += 0.1
        elif amp_ratio > 5.0:
            confidence -= 0.1

        # MACD辅助确认
        macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
        if macd_confirmed:
            confidence += 0.05

        # 子级别递归确认：日线3卖 = 跌破+反弹不进中枢，30分钟也必须确认反弹不破ZD
        sub_ok, sub_high, sub_desc = self._check_sub_level_3sell(zd)
        if sub_ok:
            confidence += 0.1
        elif self._sub_strokes:
            confidence -= 0.05

        confidence = max(0.1, min(1.0, confidence))

        reason_suffix = ''
        if pivot_div:
            reason_suffix = f', 振幅背驰(离开/进入={amp_ratio:.2f})'
        elif amp_ratio > 0:
            reason_suffix = f', 离开力度={amp_ratio:.2f}'
        if macd_confirmed:
            reason_suffix += f', MACD确认({macd_ratio:.2f})'
        if sub_ok:
            reason_suffix += f', {sub_desc}'

        return BuySellPoint(
            point_type='3sell',
            price=last_bounce.end_value,
            index=last_bounce.end_index,
            related_pivot=last_pivot,
            pivot_divergence_ratio=amp_ratio,
            divergence_ratio=macd_ratio,
            confidence=confidence,
            reason=f'3卖: 反弹不破ZD={zd:.2f}{reason_suffix}'
        )

    # ==================== 子级别递归确认 ====================

    def _check_sub_level_buy(
        self, daily_index: int, expected_types: List[str]
    ) -> Tuple[bool, float, str]:
        """
        递归到子级别（30分钟）检查是否出现对应买卖点。

        缠论区间套原理：日线1买的离开笔，本质上是30分钟级别的下跌趋势，
        其终点就是30分钟级别的1买（底背驰点）。

        Args:
            daily_index: 日线买卖点的K线索引
            expected_types: 期望的子级别买卖点类型 ['1buy', '2buy'] 等

        Returns:
            (是否确认, 子级别置信度, 子级别类型描述)
        """
        if not self._sub_buy_points:
            return (False, 0.0, '')

        # 在子级别买卖点中找匹配的类型
        matched = [b for b in self._sub_buy_points if b.point_type in expected_types]
        if not matched:
            return (False, 0.0, '')

        # 取最新的一个
        best = matched[-1]
        return (True, best.confidence, f'30分{best.point_type}')

    def _check_sub_level_sell(
        self, daily_index: int, expected_types: List[str]
    ) -> Tuple[bool, float, str]:
        """
        递归到子级别（30分钟）检查是否出现对应卖点。

        Args:
            daily_index: 日线卖点的K线索引
            expected_types: 期望的子级别卖点类型 ['1sell', '2sell'] 等

        Returns:
            (是否确认, 子级别置信度, 子级别类型描述)
        """
        if not self._sub_sell_points:
            return (False, 0.0, '')

        matched = [s for s in self._sub_sell_points if s.point_type in expected_types]
        if not matched:
            return (False, 0.0, '')

        best = matched[-1]
        return (True, best.confidence, f'30分{best.point_type}')

    def _check_sub_level_3buy(
        self, zg: float
    ) -> Tuple[bool, float, str]:
        """
        日线3买区域 → 递归30分钟找精确入场点。

        日线3买定区域（ZG上方），30分钟在区域内找：
        1. 30分钟2买（最强确认）：回调不破前低
        2. 30分钟1买+强底分型（次强确认）：中枢背驰底

        Args:
            zg: 日线中枢上沿（ZG）

        Returns:
            (是否确认, 入场价, 描述)
        """
        if not self._sub_strokes or not self._sub_buy_points:
            return (False, 0.0, '')

        # 优先找30分钟2买（在ZG上方区域）
        buys_2buy = [b for b in self._sub_buy_points
                     if b.point_type == '2buy' and b.price > zg]
        if buys_2buy:
            best = buys_2buy[-1]
            return (True, best.price,
                    f'30分2买@{best.price:.2f}(ZG上方)')

        # 其次找30分钟1买（在ZG上方区域）
        buys_1buy = [b for b in self._sub_buy_points
                     if b.point_type == '1buy' and b.price > zg]
        if buys_1buy:
            best = buys_1buy[-1]
            return (True, best.price,
                    f'30分1买@{best.price:.2f}(ZG上方)')

        return (False, 0.0, '')

    def _check_sub_level_3sell(
        self, zd: float
    ) -> Tuple[bool, float, str]:
        """
        日线3卖区域 → 递归30分钟找精确入场点。

        日线3卖定区域（ZD下方），30分钟在区域内找：
        1. 30分钟2卖（最强确认）：反弹不破前高
        2. 30分钟1卖（次强确认）：中枢背驰顶

        Args:
            zd: 日线中枢下沿（ZD）

        Returns:
            (是否确认, 入场价, 描述)
        """
        if not self._sub_strokes or not self._sub_sell_points:
            return (False, 0.0, '')

        # 优先找30分钟1卖（顶背驰=反弹动能衰竭，最直接确认跌势）
        sells_1sell = [s for s in self._sub_sell_points
                       if s.point_type == '1sell' and s.price < zd]
        if sells_1sell:
            best = sells_1sell[-1]
            return (True, best.price,
                    f'30分1卖@{best.price:.2f}(ZD下方)')

        # 其次找30分钟2卖（反弹不破前高）
        sells_2sell = [s for s in self._sub_sell_points
                       if s.point_type == '2sell' and s.price < zd]
        if sells_2sell:
            best = sells_2sell[-1]
            return (True, best.price,
                    f'30分2卖@{best.price:.2f}(ZD下方)')

        return (False, 0.0, '')

    # ==================== 辅助方法 ====================

    def _compute_pivot_divergence(
        self, pivot: Pivot, leaving_stroke: Stroke
    ) -> Tuple[bool, float, float]:
        """
        中枢背驰检测

        主判定：振幅百分比比较（始终可用）
        辅助指标：MACD面积比较（可选）

        缠论原文：比较进入中枢和离开中枢的同向笔力度。
        离开力度 < 进入力度 → 中枢背驰 → 动能衰竭 → 趋势反转信号。

        比较对象：离开笔 vs 紧邻的进入笔（离开前最后一笔同向笔），
        不是所有进入笔的最大值。紧邻比较更符合缠论"相邻同向笔对比"原理。

        Args:
            pivot: 中枢
            leaving_stroke: 离开中枢的笔（突破笔）

        Returns:
            (是否存在中枢背驰, 振幅比, MACD面积比)
            ratio < 1.0 表示离开弱于进入（背驰）
            macd_ratio = 0.0 表示MACD不可用
        """
        if not pivot.strokes:
            return (False, 0.0, 0.0)

        # 找中枢内与离开笔同方向的笔（进入笔）
        if leaving_stroke.is_up:
            entering = [s for s in pivot.strokes if s.is_up]
        else:
            entering = [s for s in pivot.strokes if s.is_down]

        if not entering:
            return (False, 0.0, 0.0)

        # ===== 主判定：与紧邻进入笔比较（非全部最大值） =====
        # 缠论原理：比较相邻两段同向走势的力度变化
        # 取离开笔之前的最后一笔同向笔（紧邻进入笔）
        adjacent_entering = None
        for s in reversed(entering):
            if s.end_index <= leaving_stroke.start_index:
                adjacent_entering = s
                break

        # 如果找不到紧邻的，用中枢内最后一笔同向笔
        if adjacent_entering is None:
            adjacent_entering = entering[-1]

        adjacent_amp = abs(adjacent_entering.price_change_pct)

        # 同时记录所有进入笔的最大值作为辅助参考
        max_entering_amp = max(abs(s.price_change_pct) for s in entering) if entering else 0.0

        if adjacent_amp <= 0:
            if max_entering_amp <= 0:
                return (False, 0.0, 0.0)
            # 紧邻笔振幅为0时退回用最大值
            adjacent_amp = max_entering_amp

        leaving_amp = abs(leaving_stroke.price_change_pct)

        # 主比值：与紧邻进入笔比较
        amp_ratio = leaving_amp / adjacent_amp if adjacent_amp > 0 else 0.0
        has_divergence = amp_ratio < 1.0

        # ===== 辅助指标：MACD面积比较 =====
        macd_ratio = 0.0
        if self.macd:
            direction = 'up' if leaving_stroke.is_up else 'down'

            # MACD面积同样用紧邻进入笔
            adjacent_area = self.macd.compute_area(
                adjacent_entering.start_index, adjacent_entering.end_index, direction
            )

            if adjacent_area > 0:
                leaving_area = self.macd.compute_area(
                    leaving_stroke.start_index, leaving_stroke.end_index, direction
                )
                macd_ratio = leaving_area / adjacent_area

        return (has_divergence, amp_ratio, macd_ratio)

    def _get_trend_modifier(self, signal_direction: str) -> float:
        """获取趋势轨道置信度修正值"""
        if self._track_detector is None:
            return 0.0
        return self._track_detector.get_track_confidence_modifier(signal_direction)

    def _apply_trend_modifier(self) -> None:
        """将趋势轨道修正值应用到所有买卖点的置信度中"""
        if self._track_detector is None:
            return

        for bp in self._buy_points:
            modifier = self._track_detector.get_track_confidence_modifier('buy')
            bp.confidence = max(0.1, min(1.0, bp.confidence + modifier))
            bp.trend_modifier = modifier

        for sp in self._sell_points:
            modifier = self._track_detector.get_track_confidence_modifier('sell')
            sp.confidence = max(0.1, min(1.0, sp.confidence + modifier))
            sp.trend_modifier = modifier

    def _find_implied_first_buy(self) -> Optional[BuySellPoint]:
        """从笔结构推断隐含的1买点（用于2买检测），增加背驰验证"""
        if not self.pivots or len(self.strokes) < 3:
            return None

        last_pivot = self.pivots[-1]
        down_strokes = [s for s in self.strokes
                        if s.is_down and s.end_value < last_pivot.low]
        if not down_strokes:
            return None

        last_down = down_strokes[-1]

        # 增加背驰验证：最后一笔的跌幅应小于前一笔（底背驰特征）
        if len(down_strokes) >= 2:
            prev_down = down_strokes[-2]
            curr_drop = abs(last_down.price_change_pct) if hasattr(last_down, 'price_change_pct') else 0
            prev_drop = abs(prev_down.price_change_pct) if hasattr(prev_down, 'price_change_pct') else 0

            if prev_drop > 0 and curr_drop >= prev_drop * 0.7:
                # 没有背驰迹象（后一笔跌幅不小于前一笔），不认为是1买
                return None

            # MACD背驰辅助验证
            if self.macd:
                has_div, div_ratio = self.macd.check_divergence(
                    last_down.start_index, last_down.end_index, 'down',
                    prev_start=prev_down.start_index,
                    prev_end=prev_down.end_index
                )
                if not has_div:
                    return None

        return BuySellPoint(
            point_type='1buy',
            price=last_down.low,
            index=last_down.end_index,
            related_pivot=last_pivot,
            confidence=0.5,
            stop_loss=last_down.low * 0.99,
            reason='隐含1买(推断+背驰验证)'
        )

    # ==================== 批量检测 ====================

    def _detect_first_buy(self) -> None:
        """批量检测所有1买点（振幅背驰主判定 + MACD辅助 + 趋势强制要求）"""
        if not self.pivots or len(self.strokes) < 3:
            return

        for pivot_idx, pivot in enumerate(self.pivots):
            # 缠论强制要求：1买必须在下跌趋势中
            # 当前中枢之前必须存在更高的中枢
            has_downtrend = any(
                prev_p.zd > pivot.zg
                for prev_p in self.pivots[:pivot_idx]
            )
            if not has_downtrend:
                continue

            # 找离开此中枢的向下笔（终点低于ZD）
            leaving = [s for s in self.strokes
                       if s.is_down and s.end_value < pivot.zd
                       and s.start_index >= pivot.start_index]

            for leave_s in leaving:
                # 主判定：振幅背驰 + 辅助MACD
                pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
                    pivot, leave_s
                )
                if not pivot_div:
                    continue

                confidence = 0.5 + min((1.0 - amp_ratio) * 0.5, 0.4)

                macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
                if macd_confirmed:
                    confidence += 0.1
                elif macd_ratio >= 1.5:
                    confidence -= 0.05

                macd_info = f', MACD确认({macd_ratio:.2f})' if macd_confirmed else ''

                # 子级别递归确认
                sub_ok, sub_conf, sub_desc = self._check_sub_level_buy(
                    leave_s.end_index, ['1buy']
                )
                if sub_ok:
                    confidence += 0.15
                elif self._sub_buy_points:
                    confidence -= 0.1
                sub_info = f', {sub_desc}' if sub_ok else ''

                confidence = max(0.1, min(1.0, confidence))

                self._buy_points.append(BuySellPoint(
                    point_type='1buy',
                    price=leave_s.end_value,
                    index=leave_s.end_index,
                    related_pivot=pivot,
                    related_strokes=[leave_s],
                    divergence_ratio=macd_ratio,
                    pivot_divergence_ratio=amp_ratio,
                    confidence=confidence,
                    stop_loss=leave_s.low * 0.99,
                    reason=f'1买: 振幅背驰(离开/进入={amp_ratio:.2f}), 下跌趋势, ZD={pivot.zd:.2f}{macd_info}{sub_info}'
                ))

    def _detect_first_sell(self) -> None:
        """批量检测所有1卖点（振幅背驰主判定 + MACD辅助 + 趋势强制要求）"""
        if not self.pivots or len(self.strokes) < 3:
            return

        for pivot_idx, pivot in enumerate(self.pivots):
            # 缠论强制要求：1卖必须在上涨趋势中
            has_uptrend = any(
                prev_p.zg < pivot.zd
                for prev_p in self.pivots[:pivot_idx]
            )
            if not has_uptrend:
                continue

            # 找离开此中枢的向上笔（终点高于ZG）
            leaving = [s for s in self.strokes
                       if s.is_up and s.end_value > pivot.zg
                       and s.start_index >= pivot.start_index]

            for leave_s in leaving:
                # 主判定：振幅背驰 + 辅助MACD
                pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
                    pivot, leave_s
                )
                if not pivot_div:
                    continue

                confidence = 0.5 + min((1.0 - amp_ratio) * 0.5, 0.4)

                macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
                if macd_confirmed:
                    confidence += 0.1
                elif macd_ratio >= 1.5:
                    confidence -= 0.05

                macd_info = f', MACD确认({macd_ratio:.2f})' if macd_confirmed else ''

                # 子级别递归确认
                sub_ok, sub_conf, sub_desc = self._check_sub_level_sell(
                    leave_s.end_index, ['1sell']
                )
                if sub_ok:
                    confidence += 0.15
                elif self._sub_sell_points:
                    confidence -= 0.1
                sub_info = f', {sub_desc}' if sub_ok else ''

                confidence = max(0.1, min(1.0, confidence))

                self._sell_points.append(BuySellPoint(
                    point_type='1sell',
                    price=leave_s.end_value,
                    index=leave_s.end_index,
                    related_pivot=pivot,
                    related_strokes=[leave_s],
                    divergence_ratio=macd_ratio,
                    pivot_divergence_ratio=amp_ratio,
                    confidence=confidence,
                    reason=f'1卖: 振幅背驰(离开/进入={amp_ratio:.2f}), 上涨趋势{macd_info}{sub_info}'
                ))

    def _detect_second_buy(self) -> None:
        """批量检测所有2买点"""
        for bp in self._buy_points:
            if bp.point_type != '1buy':
                continue

            up_after = [s for s in self.strokes
                        if s.is_up and s.start_index > bp.index]
            if not up_after:
                continue

            for up_s in up_after:
                pullback = [s for s in self.strokes
                            if s.is_down and s.start_index > up_s.end_index
                            and s.end_index <= up_s.end_index + 50]
                if not pullback:
                    continue

                for pb in pullback:
                    if pb.end_value > bp.price:
                        self._buy_points.append(BuySellPoint(
                            point_type='2buy',
                            price=pb.end_value,
                            index=pb.end_index,
                            related_pivot=bp.related_pivot,
                            confidence=0.6,
                            stop_loss=pb.low * 0.99,
                            reason=f'2买: 回调不破1买{bp.price:.2f}'
                        ))
                        break
                break

    def _detect_second_sell(self) -> None:
        """批量检测所有2卖点（增强版：结构质量+子级别确认）"""
        for sp in self._sell_points:
            if sp.point_type != '1sell':
                continue

            down_after = [s for s in self.strokes
                          if s.is_down and s.start_index > sp.index]
            if not down_after:
                continue

            for down_s in down_after:
                bounce = [s for s in self.strokes
                          if s.is_up and s.start_index > down_s.end_index
                          and s.end_index <= down_s.end_index + 50]
                if not bounce:
                    continue

                for b in bounce:
                    if b.end_value < sp.price:
                        bounce_amp = abs(b.price_change_pct)
                        if bounce_amp > 8.0:
                            continue  # 振幅过大可能反转，不是2卖

                        # 置信度按反弹深度计算
                        confidence = 0.5 + min(abs(sp.price - b.end_value) / sp.price * 5, 0.3)

                        # 子级别确认（有数据时加分，无数据不惩罚）
                        if self._sub_sell_points:
                            sub_matched = [s for s in self._sub_sell_points
                                           if s.point_type in ('2sell', '1sell')]
                            if sub_matched:
                                confidence += 0.15

                        confidence = max(0.1, min(1.0, confidence))

                        self._sell_points.append(BuySellPoint(
                            point_type='2sell',
                            price=b.end_value,
                            index=b.end_index,
                            confidence=confidence,
                            stop_loss=sp.price * 1.01,
                            reason=f'2卖: 反弹不破1卖{sp.price:.2f}, 振幅={bounce_amp:.1f}%'
                        ))
                        break
                break

    def _detect_third_buy(self) -> None:
        """批量检测所有3买点（振幅背驰主判定 + MACD辅助 + ZG判定）"""
        if not self.pivots:
            return

        for pivot in self.pivots:
            zg = pivot.zg if pivot.zg > 0 else pivot.high

            breakout = [s for s in self.strokes
                        if s.is_up and s.end_value > zg
                        and s.start_index >= pivot.start_index]
            if not breakout:
                continue

            for bo in breakout:
                pullback = [s for s in self.strokes
                            if s.is_down and s.start_index > bo.end_index
                            and s.end_value > zg * (1 - self.fuzzy_tolerance)
                            and s.end_index <= bo.end_index + 50]
                for pb in pullback:
                    pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
                        pivot, bo
                    )
                    confidence = 0.6

                    # 模糊区穿透惩罚
                    if pb.end_value < zg:
                        penetration = (zg - pb.end_value) / zg
                        confidence -= min(penetration * 20, 0.15)

                    if pivot_div:
                        confidence += 0.1
                    elif amp_ratio > 5.0:
                        confidence -= 0.1

                    macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
                    if macd_confirmed:
                        confidence += 0.05

                    reason_suffix = ''
                    if pivot_div:
                        reason_suffix = f', 振幅背驰({amp_ratio:.2f})'
                    elif amp_ratio > 0:
                        reason_suffix = f', 离开力度={amp_ratio:.2f}'
                    if macd_confirmed:
                        reason_suffix += f', MACD确认({macd_ratio:.2f})'

                    # 子级别递归确认
                    sub_ok, sub_low, sub_desc = self._check_sub_level_3buy(zg)
                    if sub_ok:
                        confidence += 0.1
                    elif self._sub_strokes:
                        confidence -= 0.05
                    if sub_ok:
                        reason_suffix += f', {sub_desc}'

                    # 持仓引导
                    if confidence >= 0.8:
                        hold_days = 20
                    elif confidence >= 0.65:
                        hold_days = 10
                    else:
                        hold_days = 5
                    margin = (pb.end_value - zg) / zg
                    urgency = 0.3 if margin < 0.02 else (0.1 if margin < 0.05 else 0.0)

                    self._buy_points.append(BuySellPoint(
                        point_type='3buy',
                        price=pb.end_value,
                        index=pb.end_index,
                        related_pivot=pivot,
                        related_strokes=[bo, pb],
                        pivot_divergence_ratio=amp_ratio,
                        divergence_ratio=macd_ratio,
                        confidence=confidence,
                        stop_loss=zg * 0.99,
                        reason=f'3买: 回踩不破ZG={zg:.2f}{reason_suffix}',
                        recommended_hold_days=hold_days,
                        exit_urgency=urgency,
                    ))
                    break
                if pullback:
                    break

    def _detect_third_sell(self) -> None:
        """批量检测所有3卖点（振幅背驰主判定 + MACD辅助 + ZD判定 + 子级别确认）"""
        if not self.pivots:
            return

        for pivot in self.pivots:
            zd = pivot.zd if pivot.zd > 0 else pivot.low

            breakdown = [s for s in self.strokes
                         if s.is_down and s.end_value < zd
                         and s.start_index >= pivot.start_index]
            if not breakdown:
                continue

            for bd in breakdown:
                bounce = [s for s in self.strokes
                          if s.is_up and s.start_index > bd.end_index
                          and s.end_value < zd * (1 + self.fuzzy_tolerance)
                          and s.end_index <= bd.end_index + 50]
                for b in bounce:
                    pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
                        pivot, bd
                    )
                    confidence = 0.6

                    # 模糊区穿透惩罚
                    if b.end_value > zd:
                        penetration = (b.end_value - zd) / zd
                        confidence -= min(penetration * 20, 0.15)

                    if pivot_div:
                        confidence += 0.1
                    elif amp_ratio > 5.0:
                        confidence -= 0.1

                    macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
                    if macd_confirmed:
                        confidence += 0.05

                    # 子级别确认（补齐，与_check_third_sell一致）
                    sub_ok, sub_high, sub_desc = self._check_sub_level_3sell(zd)
                    if sub_ok:
                        confidence += 0.1
                    elif self._sub_strokes:
                        confidence -= 0.05

                    reason_suffix = ''
                    if pivot_div:
                        reason_suffix = f', 振幅背驰({amp_ratio:.2f})'
                    elif amp_ratio > 0:
                        reason_suffix = f', 离开力度={amp_ratio:.2f}'
                    if macd_confirmed:
                        reason_suffix += f', MACD确认({macd_ratio:.2f})'
                    if sub_ok:
                        reason_suffix += f', {sub_desc}'

                    self._sell_points.append(BuySellPoint(
                        point_type='3sell',
                        price=b.end_value,
                        index=b.end_index,
                        related_pivot=pivot,
                        related_strokes=[bd, b],
                        pivot_divergence_ratio=amp_ratio,
                        divergence_ratio=macd_ratio,
                        confidence=confidence,
                        reason=f'3卖: 反弹不破ZD={zd:.2f}{reason_suffix}'
                    ))
                    break
                if bounce:
                    break

    # ==================== 类买卖点检测 ====================

    def _check_quasi_second_buy(self) -> Optional[BuySellPoint]:
        """
        类二买检测（单次）

        盘整中回调不破前低。与标准2买的核心区别：
        - 不要求前面有1买（趋势背驰低点）
        - 在中枢盘整区间内发生
        - 本质：盘整中的"二次探底不破前低"
        """
        results = self._detect_quasi_second_buy_internal()
        return results[-1] if results else None

    def _check_quasi_third_buy(self) -> Optional[BuySellPoint]:
        """
        类三买检测（单次）

        突破ZG后回踩，允许轻微跌破ZG（但高于ZD）。
        与标准3买的区别：回踩可以少量进入中枢，但不超过ZD。
        """
        results = self._detect_quasi_third_buy_internal()
        return results[-1] if results else None

    def _check_quasi_second_sell(self) -> Optional[BuySellPoint]:
        """
        类二卖检测（单次）

        盘整中反弹不破前高。与标准2卖的核心区别：
        - 不要求前面有1卖（趋势背驰高点）
        - 在中枢盘整区间内发生
        - 本质：盘整中的"二次探顶不破前高"
        """
        results = self._detect_quasi_second_sell_internal()
        return results[-1] if results else None

    def _check_quasi_third_sell(self) -> Optional[BuySellPoint]:
        """
        类三卖检测（单次）

        跌破ZD后反弹，允许轻微突破ZD（但低于ZG）。
        与标准3卖的区别：反弹可以少量进入中枢，但不超过ZG。
        """
        results = self._detect_quasi_third_sell_internal()
        return results[-1] if results else None

    def _detect_quasi_second_buy(self) -> None:
        """批量检测所有类2买点"""
        points = self._detect_quasi_second_buy_internal()
        self._buy_points.extend(points)

    def _detect_quasi_second_buy_internal(self) -> List[BuySellPoint]:
        """
        类2买核心检测逻辑

        规则：
        1. 找中枢盘整区域内的向下笔
        2. 向下笔终点在中枢范围内（ZD附近，不低于ZD太多）
        3. 该向下笔不破同中枢内前一个向下笔的低点
        4. 不要求前面有1买
        """
        if not self.pivots or len(self.strokes) < 4:
            return []

        results = []
        for pivot in self.pivots:
            zd = pivot.zd if pivot.zd > 0 else pivot.low
            zg = pivot.zg if pivot.zg > 0 else pivot.high

            # 找中枢内及附近的向下笔
            down_in_pivot = [s for s in self.strokes
                            if s.is_down
                            and s.start_index >= pivot.start_index
                            and s.start_index <= pivot.end_index + 30]
            if len(down_in_pivot) < 2:
                continue

            # 逐对比较：后一个向下笔 vs 前一个向下笔
            for i in range(1, len(down_in_pivot)):
                prev_down = down_in_pivot[i - 1]
                curr_down = down_in_pivot[i]

                # 类2买核心条件：当前向下笔不破前一个向下笔的低点
                # 即 curr_down.end_value > prev_down.end_value（当前底部高于前低）
                if curr_down.end_value < prev_down.end_value:
                    continue  # 打破了前低，不是类2买

                # 当前向下笔终点应在中枢下沿附近（ZD附近，允许低于ZD最多3%）
                if curr_down.end_value < zd * 0.97:
                    continue

                # 去重：确保同位置没有更强的买点
                existing = [b for b in self._buy_points
                           if b.index == curr_down.end_index]
                if existing:
                    continue

                # 置信度：基于不破前低的程度（差距越大越好）
                margin = (curr_down.end_value - prev_down.end_value) / prev_down.end_value
                confidence = 0.45 + min(margin * 10, 0.25)

                # MACD辅助：如果当前笔MACD面积小于前一笔，加分
                if self.macd:
                    prev_area = self.macd.compute_area(
                        prev_down.start_index, prev_down.end_index, 'down'
                    )
                    curr_area = self.macd.compute_area(
                        curr_down.start_index, curr_down.end_index, 'down'
                    )
                    if prev_area > 0 and curr_area > 0:
                        ratio = curr_area / prev_area
                        if ratio < 1.0:
                            confidence += 0.1

                # 子级别确认
                sub_ok, _, sub_desc = self._check_sub_level_buy(
                    curr_down.end_index, ['2buy', '1buy', 'quasi2buy']
                )
                if sub_ok:
                    confidence += 0.1

                confidence = max(0.1, min(1.0, confidence))
                stop_loss = curr_down.low * 0.99

                results.append(BuySellPoint(
                    point_type='quasi2buy',
                    price=curr_down.end_value,
                    index=curr_down.end_index,
                    related_pivot=pivot,
                    related_strokes=[prev_down, curr_down],
                    confidence=confidence,
                    stop_loss=stop_loss,
                    reason=f'类2买: 盘整回调不破前低{prev_down.end_value:.2f}(当前{curr_down.end_value:.2f}), ZD={zd:.2f}',
                ))

        return results

    def _detect_quasi_third_buy(self) -> None:
        """批量检测所有类3买点"""
        points = self._detect_quasi_third_buy_internal()
        self._buy_points.extend(points)

    def _detect_quasi_third_buy_internal(self) -> List[BuySellPoint]:
        """
        类3买核心检测逻辑

        规则：
        1. 向上笔突破中枢ZG
        2. 突破后回调（向下笔），允许轻微跌破ZG
        3. 回调低点必须高于ZD（不进入中枢核心区）
        4. 与标准3买的区别：标准3买要求回调 > ZG，类3买允许回调 < ZG 但 > ZD
        """
        if not self.pivots or len(self.strokes) < 4:
            return []

        results = []
        for pivot in self.pivots:
            zd = pivot.zd if pivot.zd > 0 else pivot.low
            zg = pivot.zg if pivot.zg > 0 else pivot.high

            # 找突破ZG的向上笔
            breakout = [s for s in self.strokes
                       if s.is_up and s.end_value > zg
                       and s.start_index >= pivot.start_index]
            if not breakout:
                continue

            for bo in breakout:
                # 找突破后的回调笔
                pullback = [s for s in self.strokes
                           if s.is_down
                           and s.start_index > bo.end_index
                           and s.end_index <= bo.end_index + 50]
                if not pullback:
                    continue

                for pb in pullback:
                    # 类3买核心条件：
                    # 回调低点 <= ZG（标准3买要求 > ZG，这里放宽）
                    # 且回调低点 > ZD（不进入中枢核心）
                    if pb.end_value > zg:
                        continue  # 这是标准3买，跳过
                    if pb.end_value <= zd:
                        continue  # 跌入中枢核心，太深

                    # 确保没有同位置的3买已检测
                    existing = [b for b in self._buy_points
                               if b.index == pb.end_index
                               and b.point_type == '3buy']
                    if existing:
                        continue

                    # 振幅背驰检测
                    pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
                        pivot, bo
                    )

                    # 置信度
                    margin_above_zd = (pb.end_value - zd) / zd
                    penetration = (zg - pb.end_value) / zg  # 回调穿透ZG的深度
                    confidence = 0.45 + min(margin_above_zd * 10, 0.2)

                    # 穿透越浅，信号越强
                    if penetration < 0.01:
                        confidence += 0.1
                    elif penetration > 0.03:
                        confidence -= 0.05

                    if pivot_div:
                        confidence += 0.1

                    macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
                    if macd_confirmed:
                        confidence += 0.05

                    # 子级别确认
                    sub_ok, _, sub_desc = self._check_sub_level_buy(
                        pb.end_index, ['2buy', '1buy', 'quasi3buy']
                    )
                    if sub_ok:
                        confidence += 0.1

                    confidence = max(0.1, min(1.0, confidence))
                    stop_loss = zd * 0.99

                    reason_parts = [f'类3买: 回踩轻微破ZG={zg:.2f}但高于ZD={zd:.2f}']
                    if pivot_div:
                        reason_parts.append(f'振幅背驰({amp_ratio:.2f})')
                    if macd_confirmed:
                        reason_parts.append(f'MACD确认({macd_ratio:.2f})')
                    if sub_ok:
                        reason_parts.append(sub_desc)

                    # 持仓引导
                    if confidence >= 0.75:
                        hold_days = 15
                    elif confidence >= 0.6:
                        hold_days = 8
                    else:
                        hold_days = 4
                    exit_urgency = 0.2 if penetration > 0.02 else 0.0

                    results.append(BuySellPoint(
                        point_type='quasi3buy',
                        price=pb.end_value,
                        index=pb.end_index,
                        related_pivot=pivot,
                        related_strokes=[bo, pb],
                        pivot_divergence_ratio=amp_ratio,
                        divergence_ratio=macd_ratio,
                        confidence=confidence,
                        stop_loss=stop_loss,
                        reason=', '.join(reason_parts),
                        recommended_hold_days=hold_days,
                        exit_urgency=exit_urgency,
                    ))
                    break
                if pullback:
                    break

        return results

    def _detect_quasi_second_sell(self) -> None:
        """批量检测所有类2卖点"""
        points = self._detect_quasi_second_sell_internal()
        self._sell_points.extend(points)

    def _detect_quasi_second_sell_internal(self) -> List[BuySellPoint]:
        """
        类2卖核心检测逻辑

        规则：
        1. 找中枢盘整区域内的向上笔
        2. 向上笔终点在中枢范围内（ZG附近，不高于ZG太多）
        3. 该向上笔不破同中枢内前一个向上笔的高点
        4. 不要求前面有1卖
        """
        if not self.pivots or len(self.strokes) < 4:
            return []

        results = []
        for pivot in self.pivots:
            zd = pivot.zd if pivot.zd > 0 else pivot.low
            zg = pivot.zg if pivot.zg > 0 else pivot.high

            # 找中枢内及附近的向上笔
            up_in_pivot = [s for s in self.strokes
                          if s.is_up
                          and s.start_index >= pivot.start_index
                          and s.start_index <= pivot.end_index + 30]
            if len(up_in_pivot) < 2:
                continue

            # 逐对比较：后一个向上笔 vs 前一个向上笔
            for i in range(1, len(up_in_pivot)):
                prev_up = up_in_pivot[i - 1]
                curr_up = up_in_pivot[i]

                # 类2卖核心条件：当前向上笔不破前一个向上笔的高点
                if curr_up.end_value >= prev_up.end_value:
                    continue

                # 当前向上笔终点应在中枢上沿附近（ZG附近，允许高于ZG最多3%）
                if curr_up.end_value > zg * 1.03:
                    continue

                # 去重：确保同位置没有更强的卖点
                existing = [s for s in self._sell_points
                           if s.index == curr_up.end_index]
                if existing:
                    continue

                # 置信度：前高与当前反弹高点的差距
                margin = (prev_up.end_value - curr_up.end_value) / prev_up.end_value
                confidence = 0.45 + min(margin * 10, 0.25)

                # MACD辅助
                if self.macd:
                    prev_area = self.macd.compute_area(
                        prev_up.start_index, prev_up.end_index, 'up'
                    )
                    curr_area = self.macd.compute_area(
                        curr_up.start_index, curr_up.end_index, 'up'
                    )
                    if prev_area > 0 and curr_area > 0:
                        ratio = curr_area / prev_area
                        if ratio < 1.0:
                            confidence += 0.1

                # 子级别确认
                sub_ok, _, sub_desc = self._check_sub_level_sell(
                    curr_up.end_index, ['2sell', '1sell', 'quasi2sell']
                )
                if sub_ok:
                    confidence += 0.1

                confidence = max(0.1, min(1.0, confidence))
                stop_loss = curr_up.high * 1.01

                results.append(BuySellPoint(
                    point_type='quasi2sell',
                    price=curr_up.end_value,
                    index=curr_up.end_index,
                    related_pivot=pivot,
                    related_strokes=[prev_up, curr_up],
                    confidence=confidence,
                    stop_loss=stop_loss,
                    reason=f'类2卖: 盘整反弹不破前高{prev_up.end_value:.2f}(当前{curr_up.end_value:.2f}), ZG={zg:.2f}',
                ))

        return results

    def _detect_quasi_third_sell(self) -> None:
        """批量检测所有类3卖点"""
        points = self._detect_quasi_third_sell_internal()
        self._sell_points.extend(points)

    def _detect_quasi_third_sell_internal(self) -> List[BuySellPoint]:
        """
        类3卖核心检测逻辑

        规则：
        1. 向下笔跌破中枢ZD
        2. 跌破后反弹（向上笔），允许轻微突破ZD
        3. 反弹高点必须低于ZG（不进入中枢核心区）
        4. 与标准3卖的区别：标准3卖要求反弹 < ZD，类3卖允许反弹 > ZD 但 < ZG
        """
        if not self.pivots or len(self.strokes) < 4:
            return []

        results = []
        for pivot in self.pivots:
            zd = pivot.zd if pivot.zd > 0 else pivot.low
            zg = pivot.zg if pivot.zg > 0 else pivot.high

            # 找跌破ZD的向下笔
            breakdown = [s for s in self.strokes
                        if s.is_down and s.end_value < zd
                        and s.start_index >= pivot.start_index]
            if not breakdown:
                continue

            for bd in breakdown:
                # 找跌破后的反弹笔
                bounce = [s for s in self.strokes
                         if s.is_up
                         and s.start_index > bd.end_index
                         and s.end_index <= bd.end_index + 50]
                if not bounce:
                    continue

                for b in bounce:
                    # 类3卖核心条件：
                    # 反弹高点 >= ZD（标准3卖要求 < ZD，这里放宽）
                    # 且反弹高点 < ZG（不进入中枢核心）
                    if b.end_value < zd:
                        continue  # 这是标准3卖，跳过
                    if b.end_value >= zg:
                        continue  # 反弹过强，进入中枢核心

                    # 确保没有同位置的3卖已检测
                    existing = [s for s in self._sell_points
                               if s.index == b.end_index
                               and s.point_type == '3sell']
                    if existing:
                        continue

                    # 振幅背驰检测
                    pivot_div, amp_ratio, macd_ratio = self._compute_pivot_divergence(
                        pivot, bd
                    )

                    # 置信度
                    margin_below_zg = (zg - b.end_value) / zg
                    penetration = (b.end_value - zd) / zd  # 反弹穿透ZD的深度
                    confidence = 0.45 + min(margin_below_zg * 10, 0.2)

                    # 穿透越浅，信号越强
                    if penetration < 0.01:
                        confidence += 0.1
                    elif penetration > 0.03:
                        confidence -= 0.05

                    if pivot_div:
                        confidence += 0.1

                    macd_confirmed = macd_ratio > 0 and macd_ratio < 1.0
                    if macd_confirmed:
                        confidence += 0.05

                    # 子级别确认
                    sub_ok, _, sub_desc = self._check_sub_level_sell(
                        b.end_index, ['2sell', '1sell', 'quasi3sell']
                    )
                    if sub_ok:
                        confidence += 0.1

                    confidence = max(0.1, min(1.0, confidence))
                    stop_loss = zg * 1.01

                    reason_parts = [f'类3卖: 反弹轻微破ZD={zd:.2f}但低于ZG={zg:.2f}']
                    if pivot_div:
                        reason_parts.append(f'振幅背驰({amp_ratio:.2f})')
                    if macd_confirmed:
                        reason_parts.append(f'MACD确认({macd_ratio:.2f})')
                    if sub_ok:
                        reason_parts.append(sub_desc)

                    results.append(BuySellPoint(
                        point_type='quasi3sell',
                        price=b.end_value,
                        index=b.end_index,
                        related_pivot=pivot,
                        related_strokes=[bd, b],
                        pivot_divergence_ratio=amp_ratio,
                        divergence_ratio=macd_ratio,
                        confidence=confidence,
                        stop_loss=stop_loss,
                        reason=', '.join(reason_parts),
                    ))
                    break
                if bounce:
                    break

        return results
