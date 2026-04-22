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
