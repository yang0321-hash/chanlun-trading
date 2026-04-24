"""
多周期缠论分析器

对每个周期独立运行完整缠论分析流程，
提供统一查询接口。

分析流程：KLine → 分型 → 笔 → 线段 → 中枢 → MACD → 趋势

三个核心方法：
1. get_strategic_bias()  → 周线定方向
2. get_operational_signals() → 日线买卖点候选
3. get_entry_timing() → 30分钟精确入场
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd
from loguru import logger

from .kline import KLine
from .fractal import Fractal, FractalDetector
from .stroke import Stroke, StrokeGenerator
from .segment import Segment, SegmentGenerator
from .pivot import Pivot, PivotDetector, PivotLevel
from .buy_sell_points import BuySellPoint, BuySellPointDetector
from .trend_track import TrendTrackDetector, TrendStatus
from .interval_entry import IntervalEntry, IntervalEntryDetector
from .level_mapping import recursive_level_analysis, LevelMapping
from indicator.macd import MACD
from indicator.enhanced_divergence import EnhancedDivergenceDetector


@dataclass
class TimeFrameAnalysis:
    """单个周期的完整缠论分析结果"""
    period: str                    # 'weekly', 'daily', '30min'
    kline: KLine
    fractals: List[Fractal]
    strokes: List[Stroke]
    segments: List[Segment]
    pivots: List[Pivot]
    macd: MACD
    trend_status: TrendStatus = TrendStatus.NEUTRAL
    trend_direction: str = 'neutral'

    @property
    def has_structure(self) -> bool:
        """是否有完整的缠论结构"""
        return len(self.strokes) >= 3 and len(self.pivots) >= 1


class MultiTimeFrameAnalyzer:
    """
    多周期缠论分析器

    使用方法：
        analyzer = MultiTimeFrameAnalyzer(weekly_df, daily_df, min30_df)
        bias, strength = analyzer.get_strategic_bias()
        signals = analyzer.get_operational_signals()
        timing = analyzer.get_entry_timing(signal)
    """

    def __init__(
        self,
        weekly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        min30_df: Optional[pd.DataFrame] = None,
        min_bars_daily: int = 5,
        min_bars_30m: int = 3,
        divergence_threshold: float = 0.3,
    ):
        self.divergence_threshold = divergence_threshold

        # 各周期分析
        self.weekly = self._analyze(weekly_df, 'weekly', min_bars_daily, PivotLevel.WEEK)
        self.daily = self._analyze(daily_df, 'daily', min_bars_daily, PivotLevel.DAY)
        self.min30 = self._analyze(min30_df, '30min', min_bars_30m, PivotLevel.MIN_30) if min30_df is not None else None

    def _analyze(
        self,
        df: pd.DataFrame,
        period: str,
        min_bars: int,
        level: PivotLevel,
    ) -> Optional[TimeFrameAnalysis]:
        """对单个周期运行完整缠论分析"""
        if df is None or len(df) < 30:
            return None

        try:
            kline = KLine.from_dataframe(df, strict_mode=False)

            # 分型
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 2:
                return None

            # 笔
            strokes = StrokeGenerator(kline, fractals, min_bars=min_bars).get_strokes()
            if len(strokes) < 2:
                return None

            # 线段
            segments = SegmentGenerator(kline, strokes).get_segments()

            # 中枢
            pivots = PivotDetector(kline, strokes, level=level).get_pivots()

            # MACD
            close_s = pd.Series([k.close for k in kline])
            macd = MACD(close_s)

            # 趋势跟踪
            trend_direction = 'neutral'
            trend_status = TrendStatus.NEUTRAL
            try:
                td = TrendTrackDetector(strokes, pivots)
                td.detect()
                trend_status = td.get_trend_status()
                trend_direction = trend_status.value if hasattr(trend_status, 'value') else str(trend_status)
            except Exception:
                pass

            return TimeFrameAnalysis(
                period=period,
                kline=kline,
                fractals=fractals,
                strokes=strokes,
                segments=segments,
                pivots=pivots,
                macd=macd,
                trend_status=trend_status,
                trend_direction=trend_direction,
            )

        except Exception as e:
            logger.debug(f"{period}分析失败: {e}")
            return None

    def get_recursive_weekly(self) -> Optional[LevelMapping]:
        """
        日线笔 → 周线递归结构

        比直接用周线K线更精确的周线级别分析。
        日线笔作为虚拟K线，保留更多结构细节。
        """
        if self.daily is None or len(self.daily.strokes) < 10:
            return None
        return recursive_level_analysis(
            self.daily.strokes, 'daily', 'weekly', PivotLevel.WEEK
        )

    def get_recursive_daily(self) -> Optional[LevelMapping]:
        """
        30min笔 → 日线递归结构

        当30min数据充足时（>2000根），递归日线结构比直接日线更精确。
        """
        if self.min30 is None or len(self.min30.strokes) < 10:
            return None
        return recursive_level_analysis(
            self.min30.strokes, '30min', 'daily', PivotLevel.DAY
        )

    def get_strategic_bias(self) -> Tuple[str, float]:
        """
        周线级别战略方向

        Returns:
            ('long'/'short'/'neutral', bias_strength 0-1)

        逻辑：
        - 周线趋势方向 = 主方向
        - 最新周线中枢突破方向 = 辅助
        - 两者一致 = 强偏多/空
        """
        if self.weekly is None or not self.weekly.has_structure:
            return ('neutral', 0.0)

        # 主方向：趋势跟踪
        direction = 'neutral'
        if self.weekly.trend_direction in ('strong_up', 'up'):
            direction = 'long'
        elif self.weekly.trend_direction in ('strong_down', 'down'):
            direction = 'short'

        # 辅助：最新中枢突破方向
        if self.weekly.pivots:
            last_pivot = self.weekly.pivots[-1]
            if self.weekly.strokes:
                last_stroke = self.weekly.strokes[-1]
                if last_stroke.end_value > last_pivot.high:
                    if direction == 'long':
                        return ('long', 0.9)
                    else:
                        return ('neutral', 0.3)
                elif last_stroke.end_value < last_pivot.low:
                    if direction == 'short':
                        return ('short', 0.9)
                    else:
                        return ('neutral', 0.3)

        strength = 0.6 if direction != 'neutral' else 0.3

        # 辅助：递归周线中枢确认
        try:
            recursive = self.get_recursive_weekly()
            if recursive and recursive.pivot_count > 0:
                last_rp = recursive.recursive_pivots[-1]
                last_rs = recursive.recursive_strokes[-1] if recursive.recursive_strokes else None
                if last_rs:
                    if last_rs.end_value > last_rp.zg:
                        if direction == 'long':
                            strength = min(strength + 0.15, 1.0)
                    elif last_rs.end_value < last_rp.zd:
                        if direction == 'short':
                            strength = min(strength + 0.15, 1.0)
        except Exception:
            pass

        return (direction, strength)

    def get_operational_signals(self) -> List[BuySellPoint]:
        """
        日线级别买卖点候选

        使用BuySellPointDetector检测所有买点。
        """
        if self.daily is None or not self.daily.has_structure:
            return []

        try:
            # 获取趋势轨道（如果有）
            trend_tracks = []
            try:
                td = TrendTrackDetector(self.daily.strokes, self.daily.pivots)
                td.detect()
                trend_tracks = td._tracks if hasattr(td, '_tracks') else []
            except Exception:
                pass

            detector = BuySellPointDetector(
                fractals=self.daily.fractals,
                strokes=self.daily.strokes,
                segments=self.daily.segments,
                pivots=self.daily.pivots,
                macd=self.daily.macd,
                divergence_threshold=self.divergence_threshold,
                trend_tracks=trend_tracks,
            )
            detector.detect_all()
            return detector._buy_points

        except Exception as e:
            logger.debug(f"日线买卖点检测失败: {e}")
            return []

    def get_sell_signals(self) -> List[BuySellPoint]:
        """获取日线级别卖点"""
        if self.daily is None or not self.daily.has_structure:
            return []

        try:
            trend_tracks = []
            try:
                td = TrendTrackDetector(self.daily.strokes, self.daily.pivots)
                td.detect()
                trend_tracks = td._tracks if hasattr(td, '_tracks') else []
            except Exception:
                pass

            detector = BuySellPointDetector(
                fractals=self.daily.fractals,
                strokes=self.daily.strokes,
                segments=self.daily.segments,
                pivots=self.daily.pivots,
                macd=self.daily.macd,
                divergence_threshold=self.divergence_threshold,
                trend_tracks=trend_tracks,
            )
            detector.detect_all()
            return detector._sell_points

        except Exception as e:
            logger.debug(f"日线卖点检测失败: {e}")
            return []

    def get_entry_timing(
        self, daily_buy: BuySellPoint
    ) -> Optional[IntervalEntry]:
        """
        30分钟精确入场时机

        复用 core/interval_entry.py 的 IntervalEntryDetector

        Args:
            daily_buy: 日线买点

        Returns:
            IntervalEntry 或 None
        """
        if self.min30 is None:
            return None

        try:
            detector = IntervalEntryDetector(
                daily_buy=daily_buy,
                daily_trend=self.daily.trend_direction if self.daily else 'neutral',
                df_30m=self.min30.kline.to_dataframe() if hasattr(self.min30.kline, 'to_dataframe') else pd.DataFrame({
                    'open': [k.open for k in self.min30.kline],
                    'high': [k.high for k in self.min30.kline],
                    'low': [k.low for k in self.min30.kline],
                    'close': [k.close for k in self.min30.kline],
                    'volume': [k.volume for k in self.min30.kline],
                }),
                daily_stop_loss=daily_buy.stop_loss,
            )
            return detector.detect()
        except Exception as e:
            logger.debug(f"30分钟入场检测失败: {e}")
            return None

    def get_enhanced_divergence(self, direction: str, period: str = 'daily') -> Optional[object]:
        """
        获取增强背离检测结果

        Args:
            direction: 'up' 或 'down'
            period: 'daily' 或 'weekly'

        Returns:
            DivergenceResult
        """
        analysis = self.daily if period == 'daily' else self.weekly
        if analysis is None:
            return None

        detector = EnhancedDivergenceDetector(analysis.macd, analysis.strokes)
        return detector.detect_trend_divergence(direction)

    def interval_nesting_check(
        self, daily_buy_point: BuySellPoint
    ) -> Tuple[bool, float, str]:
        """
        区间套精确入场检测

        缠论核心方法论：父级别定方向，子级别确认入场。

        逻辑：
        1. 周线（父级）方向必须支持买点方向
        2. 30min（子级）必须在对应区域出现买卖点确认

        区间套共振 = 高胜率信号

        Args:
            daily_buy_point: 日线买点

        Returns:
            (是否确认, 共振强度0-1, 描述)
        """
        if self.weekly is None or self.daily is None:
            return (False, 0.0, '缺少周线/日线数据')

        # Step 1: 周线方向检查
        weekly_dir, weekly_strength = self.get_strategic_bias()

        if weekly_dir == 'neutral':
            return (False, 0.0, '周线方向不明')

        # 买点必须与周线方向一致
        if daily_buy_point.is_buy and weekly_dir != 'long':
            return (False, 0.0, f'周线{weekly_dir}，与买点方向不一致')

        # Step 2: 递归周线中枢确认
        recursive_weekly = self.get_recursive_weekly()
        recursive_info = ''
        if recursive_weekly and recursive_weekly.pivot_count > 0:
            rp = recursive_weekly.recursive_pivots[-1]
            if weekly_dir == 'long' and daily_buy_point.price > rp.zg:
                weekly_strength = min(weekly_strength + 0.1, 1.0)
                recursive_info = ', 递归周线中枢确认(价格>ZG)'

        # Step 3: 30min子级别确认
        sub_strength = 0.0
        sub_info = ''
        if self.min30 is not None and self.min30.has_structure:
            # 30min也必须有买点
            if self.min30.macd:
                from .buy_sell_points import BuySellPointDetector
                try:
                    sub_det = BuySellPointDetector(
                        fractals=self.min30.fractals,
                        strokes=self.min30.strokes,
                        segments=self.min30.segments,
                        pivots=self.min30.pivots,
                        macd=self.min30.macd,
                    )
                    sub_buys, _ = sub_det.detect_all()

                    # 找30min在日线买点价格区域附近的买点
                    matching_sub = [b for b in sub_buys
                                    if b.is_buy
                                    and abs(b.price - daily_buy_point.price) / daily_buy_point.price < 0.05]

                    if matching_sub:
                        best_sub = max(matching_sub, key=lambda b: b.confidence)
                        sub_strength = best_sub.confidence
                        sub_info = f', 30min{best_sub.point_type}确认(置信{best_sub.confidence:.2f})'
                    else:
                        sub_info = ', 30min无对应买点'
                except Exception:
                    sub_info = ', 30min分析失败'
        else:
            sub_info = ', 无30min数据'

        # 综合评分
        if sub_strength > 0:
            resonance = (weekly_strength * 0.4 + sub_strength * 0.6)
            desc = (f'区间套确认: 周线{weekly_dir}({weekly_strength:.2f})'
                    f'{sub_info}{recursive_info}')
            return (True, resonance, desc)

        # 30min无确认，但周线方向明确
        if weekly_strength > 0.6:
            return (True, weekly_strength * 0.4,
                    f'仅周线确认({weekly_dir}, {weekly_strength:.2f}), 30min未确认')

        return (False, 0.0, f'周线强度不足({weekly_strength:.2f}){sub_info}')
