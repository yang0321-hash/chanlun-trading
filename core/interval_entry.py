"""
区间套精确入场模块

缠论核心技巧：大级别定方向，小级别精确入场。

逻辑：
1. 日线级别检测到买点（1买/2买/3买）
2. 递归到30分钟级别，找到对应的回调区域
3. 在30分钟级别等待"第二个底分型"确认反转
4. 第二个底分型出现 → 入场信号确认

为什么是"第二个底分型"？
- 第一个底分型：可能是下跌中继，后续继续下跌
- 第二个底分型：价格再次探底不破前低 → 双底确认 → 反转概率大
- 这本质上对应30分钟级别的2买逻辑
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger

from .kline import KLine
from .fractal import Fractal, FractalDetector, FractalType
from .stroke import Stroke, StrokeGenerator
from .segment import Segment, SegmentGenerator
from .pivot import Pivot, PivotDetector
from .buy_sell_points import BuySellPoint, BuySellPointDetector
from .trend_track import TrendTrackDetector
from indicator.macd import MACD

import pandas as pd


@dataclass
class IntervalEntry:
    """区间套入场信号"""
    # 大级别（日线）信息
    daily_buy: BuySellPoint          # 日线买点
    daily_trend: str                 # 日线趋势

    # 小级别（30分钟）确认信息
    confirmed: bool = False          # 是否已确认
    confirm_type: str = ''           # 确认类型: 'second_bottom_fractal' / '2buy_30m' / 'divergence_30m'
    confirm_price: float = 0.0       # 确认入场价
    confirm_index: int = 0           # 确认K线在30分钟中的索引

    # 30分钟结构
    fractals_30m: List[Fractal] = field(default_factory=list)
    strokes_30m: List[Stroke] = field(default_factory=list)
    bottom_fractals_30m: List[Fractal] = field(default_factory=list)  # 30分钟底分型列表
    second_bottom: Optional[Fractal] = None  # 第二个底分型

    # 风控
    stop_loss: float = 0.0           # 最终止损（取日线止损和30分钟止损的较高者）
    stop_loss_source: str = ''       # 止损来源

    # 评分
    confidence: float = 0.0          # 综合置信度
    reason: str = ''

    def to_dict(self) -> dict:
        return {
            'daily_type': self.daily_buy.point_type,
            'daily_price': self.daily_buy.price,
            'daily_confidence': self.daily_buy.confidence,
            'daily_trend': self.daily_trend,
            'confirmed': self.confirmed,
            'confirm_type': self.confirm_type,
            'confirm_price': self.confirm_price,
            'stop_loss': self.stop_loss,
            'stop_loss_source': self.stop_loss_source,
            'confidence': self.confidence,
            'reason': self.reason,
        }


class IntervalEntryDetector:
    """
    区间套入场检测器

    使用方法:
        detector = IntervalEntryDetector(daily_result, df_30m)
        entries = detector.detect()
        for entry in entries:
            if entry.confirmed:
                print(f"入场: {entry.confirm_price} 止损: {entry.stop_loss}")
    """

    def __init__(self, daily_buy: BuySellPoint, daily_trend: str,
                 df_30m: pd.DataFrame, daily_stop_loss: float = 0.0):
        """
        Args:
            daily_buy: 日线级别买点
            daily_trend: 日线趋势状态字符串
            df_30m: 30分钟K线DataFrame (columns: date,open,close,high,low,volume)
            daily_stop_loss: 日线级别止损位（可选，用于取较高者）
        """
        self.daily_buy = daily_buy
        self.daily_trend = daily_trend
        self.df_30m = df_30m
        self.daily_stop_loss = daily_stop_loss or daily_buy.stop_loss

    def detect(self) -> Optional[IntervalEntry]:
        """
        检测区间套入场信号

        Returns:
            IntervalEntry 或 None
        """
        if self.df_30m is None or len(self.df_30m) < 30:
            return None

        # Step 1: 30分钟缠论分析
        analysis = self._analyze_30m()
        if analysis is None:
            return None

        kline_30m, fractals, strokes, segments, pivots, macd, td_detector = analysis

        # Step 2: 获取30分钟底分型序列
        bottom_fractals = [f for f in fractals if f.is_bottom]

        entry = IntervalEntry(
            daily_buy=self.daily_buy,
            daily_trend=self.daily_trend,
            fractals_30m=fractals,
            strokes_30m=strokes,
            bottom_fractals_30m=bottom_fractals,
        )

        # Step 3: 三层确认检查（优先级从高到低）
        # 3a: 30分钟级别出现2买信号（最强确认）
        confirmed_2buy = self._check_30m_2buy(entry, fractals, strokes, segments, pivots, macd, td_detector)
        if confirmed_2buy:
            return self._finalize_entry(entry)

        # 3b: 第二个底分型不破第一个底分型低点（双底确认）
        confirmed_second = self._check_second_bottom_fractal(entry, bottom_fractals, strokes, kline_30m)
        if confirmed_second:
            return self._finalize_entry(entry)

        # 3c: 30分钟MACD底背离 + 底分型（弱势确认）
        confirmed_div = self._check_30m_divergence(entry, fractals, strokes, macd, kline_30m)
        if confirmed_div:
            return self._finalize_entry(entry)

        # 未确认，返回entry但confirmed=False
        entry.stop_loss = self.daily_stop_loss
        entry.stop_loss_source = 'daily'
        entry.confidence = self.daily_buy.confidence * 0.6  # 未确认降低置信度
        entry.reason = f"日线{self.daily_buy.point_type}未获30分钟确认"
        return entry

    def _analyze_30m(self):
        """30分钟缠论完整分析"""
        try:
            kline = KLine.from_dataframe(self.df_30m, strict_mode=True)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
            if len(strokes) < 3:
                return None
            segments = SegmentGenerator(kline, strokes).get_segments()
            pivots = PivotDetector(kline, strokes).get_pivots()
            close_s = pd.Series([k.close for k in kline])
            macd = MACD(close_s)
            td = TrendTrackDetector(strokes, pivots)
            td.detect()
            return kline, fractals, strokes, segments, pivots, macd, td
        except Exception:
            return None

    def _check_30m_2buy(self, entry, fractals, strokes, segments, pivots, macd, td_detector):
        """检查30分钟级别是否出现2买（最强确认）"""
        if not pivots:
            return False

        try:
            tracks = td_detector._tracks if hasattr(td_detector, '_tracks') else []
            detector = BuySellPointDetector(fractals, strokes, segments, pivots, macd,
                                            trend_tracks=tracks)
            buys, _ = detector.detect_all()
        except Exception:
            return False

        klen = len(fractals) if fractals else 0
        # 找最近的2买信号
        recent_2buys = [b for b in buys if b.point_type in ('2buy', 'quasi2buy')]

        if not recent_2buys:
            return False

        # 取最近的一个2买
        last_2buy = recent_2buys[-1]

        # 验证：当前价应在2买价附近（不能跌破太多）
        latest_price = self.df_30m['close'].iloc[-1]
        if latest_price < last_2buy.stop_loss:
            return False

        entry.confirmed = True
        entry.confirm_type = '2buy_30m'
        entry.confirm_price = last_2buy.price
        entry.confirm_index = last_2buy.index
        entry.stop_loss = max(last_2buy.stop_loss, self.daily_stop_loss)
        entry.stop_loss_source = '30m_2buy' if last_2buy.stop_loss > self.daily_stop_loss else 'daily'
        entry.confidence = min(1.0, self.daily_buy.confidence * 0.7 + last_2buy.confidence * 0.3)
        entry.reason = (f"日线{self.daily_buy.point_type}(conf={self.daily_buy.confidence:.2f}) "
                       f"+ 30分钟2买@{last_2buy.price:.2f}(conf={last_2buy.confidence:.2f})")
        return True

    def _check_second_bottom_fractal(self, entry, bottom_fractals, strokes, kline_30m):
        """
        检查第二个底分型确认

        核心逻辑：
        - 找最近的两个底分型
        - 第二个底分型低点 >= 第一个底分型低点（不破前低）
        - 第二个底分型之后出现向上笔（确认反转）

        这就是缠论中"回调不破前低"的30分钟级别体现
        """
        if len(bottom_fractals) < 2:
            return False

        klen_30m = len(self.df_30m)
        latest_price = self.df_30m['close'].iloc[-1]

        # 取最近20根30分钟K线内的底分型
        recent_bottoms = [f for f in bottom_fractals if f.index >= klen_30m - 40]

        if len(recent_bottoms) < 2:
            # 放宽到50根
            recent_bottoms = [f for f in bottom_fractals if f.index >= klen_30m - 60]

        if len(recent_bottoms) < 2:
            return False

        first = recent_bottoms[-2]
        second = recent_bottoms[-1]

        # 核心：第二个底分型不破第一个底分型的低点
        if second.low < first.low:
            return False

        # 第二个底分型之后必须有向上走势（至少收盘价高于底分型高点）
        if latest_price < second.high:
            return False

        # 计算双底深度差
        depth_diff = (second.low - first.low) / first.low * 100  # 百分比

        # 止损取第二个底分型低点
        sl_30m = second.low * 0.99

        entry.confirmed = True
        entry.confirm_type = 'second_bottom_fractal'
        entry.confirm_price = latest_price  # 当前价入场（确认后）
        entry.confirm_index = klen_30m - 1
        entry.second_bottom = second
        entry.stop_loss = max(sl_30m, self.daily_stop_loss)
        entry.stop_loss_source = '30m_second_bottom' if sl_30m > self.daily_stop_loss else 'daily'

        # 置信度计算
        base = self.daily_buy.confidence * 0.6
        # 双底越浅（depth_diff越大），说明下方支撑越强
        depth_bonus = min(depth_diff * 5, 0.2)
        # 两个底分型间隔适中（不太多也不太少）
        gap = second.index - first.index
        gap_bonus = 0.1 if 5 <= gap <= 25 else 0.05
        entry.confidence = min(1.0, base + depth_bonus + gap_bonus)

        entry.reason = (f"日线{self.daily_buy.point_type} + 30分钟双底确认 "
                       f"(底1={first.low:.2f} 底2={second.low:.2f} 深度差={depth_diff:.2f}%)")
        return True

    def _check_30m_divergence(self, entry, fractals, strokes, macd, kline_30m):
        """检查30分钟MACD底背离 + 底分型（最弱确认）"""
        if not macd.values or len(strokes) < 2:
            return False

        klen_30m = len(self.df_30m)
        latest_price = self.df_30m['close'].iloc[-1]

        # 最近60根K线内的下跌笔
        recent_down = [s for s in strokes if not s.is_up and s.end_index >= klen_30m - 60]
        if len(recent_down) < 2:
            return False

        s1, s2 = recent_down[-2], recent_down[-1]

        # 价格创新低
        if s2.end_value >= s1.end_value:
            return False

        # MACD DIF未创新低（底背离）
        offset = macd._kline_offset
        idx1 = min(s1.end_index - offset, len(macd.values) - 1)
        idx2 = min(s2.end_index - offset, len(macd.values) - 1)
        if idx1 < 0 or idx2 < 0:
            return False
        dif1 = macd.values[idx1].macd
        dif2 = macd.values[idx2].macd

        if dif2 <= dif1:
            return False

        # 必须有底分型配合
        bottom_after = [f for f in fractals if f.is_bottom and f.index > s2.end_index]
        if not bottom_after:
            return False

        sl_30m = s2.low * 0.99

        entry.confirmed = True
        entry.confirm_type = 'divergence_30m'
        entry.confirm_price = latest_price
        entry.confirm_index = klen_30m - 1
        entry.stop_loss = max(sl_30m, self.daily_stop_loss)
        entry.stop_loss_source = '30m_divergence' if sl_30m > self.daily_stop_loss else 'daily'
        entry.confidence = min(1.0, self.daily_buy.confidence * 0.6 + 0.15)
        entry.reason = (f"日线{self.daily_buy.point_type} + 30分钟MACD底背离确认 "
                       f"(DIF: {dif1:.4f}→{dif2:.4f})")
        return True

    def _finalize_entry(self, entry: IntervalEntry) -> IntervalEntry:
        """最终校验入场信号"""
        if not entry.confirmed:
            return entry

        latest_price = self.df_30m['close'].iloc[-1]

        # 安全校验：当前价不能低于止损
        if latest_price <= entry.stop_loss:
            entry.confirmed = False
            entry.reason += " [CANCELLED: price below stop_loss]"
            return entry

        # 安全校验：确认价不能是0
        if entry.confirm_price <= 0:
            entry.confirm_price = latest_price

        # 计算风险收益比
        risk = latest_price - entry.stop_loss
        if risk <= 0:
            entry.confirmed = False
            return entry

        return entry
