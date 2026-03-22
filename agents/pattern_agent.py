"""
形态识别智能体
识别分型、笔、线段、中枢等缠论形态
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime

from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.state import ChanLunState, TrendDirection


@dataclass
class PatternRecognitionResult:
    """形态识别结果"""
    # 分型
    fractals: List[Dict] = field(default_factory=list)
    current_fractal: Optional[Dict] = None

    # 笔
    strokes: List[Dict] = field(default_factory=list)
    current_stroke: Optional[Dict] = None

    # 线段
    segments: List[Dict] = field(default_factory=list)
    current_segment: Optional[Dict] = None

    # 中枢
    pivots: List[Dict] = field(default_factory=list)
    current_pivot: Optional[Dict] = None
    in_pivot: bool = False

    # 背驰
    divergence_detected: bool = False
    divergence_type: str = ""  # "bullish", "bearish"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fractals": self.fractals,
            "current_fractal": self.current_fractal,
            "strokes": self.strokes,
            "current_stroke": self.current_stroke,
            "segments": self.segments,
            "current_segment": self.current_segment,
            "pivots": self.pivots,
            "current_pivot": self.current_pivot,
            "in_pivot": self.in_pivot,
            "divergence_detected": self.divergence_detected,
            "divergence_type": self.divergence_type,
        }


class PatternAgent(BaseAgent):
    """
    形态识别智能体
    负责识别所有缠论形态
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("pattern_agent", config)
        # 导入核心算法
        try:
            from core.fractal import FractalDetector
            from core.stroke import StrokeGenerator
            from core.segment import SegmentGenerator
            from core.pivot import PivotDetector
            from indicator.macd import MACD

            self.FractalDetector = FractalDetector
            self.StrokeGenerator = StrokeGenerator
            self.SegmentGenerator = SegmentGenerator
            self.PivotDetector = PivotDetector
            self.MACD = MACD

            self.has_modules = True
        except ImportError as e:
            print(f"Warning: Could not import core modules: {e}")
            self.has_modules = False

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        分析并识别缠论形态
        """
        if not self.has_modules or input_data.ohlcv_data is None:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning="缺少数据或核心模块"
            )

        try:
            df = input_data.ohlcv_data
            current_index = input_data.current_index

            # 创建 KLine 对象
            from core.kline import KLine
            kline = KLine.from_dataframe(df)

            # 识别分型
            fractal_detector = self.FractalDetector(kline)
            fractals = fractal_detector.get_fractals()
            current_fractal = self._get_current_fractal(fractals, current_index, df)

            # 生成笔
            stroke_gen = self.StrokeGenerator(kline)
            strokes = stroke_gen.get_strokes()
            current_stroke = self._get_current_stroke(strokes, current_index, df)

            # 生成线段
            segment_gen = self.SegmentGenerator(strokes)
            segments = segment_gen.get_segments()
            current_segment = self._get_current_segment(segments, current_index)

            # 识别中枢
            pivot_detector = self.PivotDetector(strokes)
            pivots = pivot_detector.get_pivots()
            current_pivot = self._get_current_pivot(pivots, current_index, df)

            # 检测背驰
            divergence_info = self._detect_divergence(df, current_index)

            # 构建结果
            result = PatternRecognitionResult(
                fractals=[f.to_dict() for f in fractals],
                current_fractal=current_fractal,
                strokes=[s.to_dict() for s in strokes],
                current_stroke=current_stroke,
                segments=[s.to_dict() for s in segments],
                current_segment=current_segment,
                pivots=[p.to_dict() for p in pivots],
                current_pivot=current_pivot,
                in_pivot=current_pivot is not None,
                divergence_detected=divergence_info['detected'],
                divergence_type=divergence_info['type']
            )

            # 计算信心度
            confidence = self._calculate_confidence(result, current_index, len(df))

            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=confidence,
                reasoning=self._generate_reasoning(result),
                data=result.to_dict()
            )

        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning=f"分析出错: {str(e)}"
            )

    def _get_current_fractal(self, fractals, current_index, df):
        """获取当前分型"""
        if not fractals:
            return None

        # 找到最近的已确认分型
        for f in reversed(fractals):
            if f.index <= current_index and f.confirmed:
                return {
                    'index': f.index,
                    'type': f.fractal_type,
                    'high': f.high,
                    'low': f.low,
                    'datetime': df.iloc[f.index]['date'] if 'date' in df.columns else f.index
                }
        return None

    def _get_current_stroke(self, strokes, current_index, df):
        """获取当前笔"""
        if not strokes:
            return None

        for s in reversed(strokes):
            if s.end_index <= current_index:
                return {
                    'start_index': s.start_index,
                    'end_index': s.end_index,
                    'start_price': s.start_price,
                    'end_price': s.end_price,
                    'direction': s.direction,
                }
        return None

    def _get_current_segment(self, segments, current_index):
        """获取当前线段"""
        if not segments:
            return None

        for seg in reversed(segments):
            if seg.end_index <= current_index:
                return {
                    'start_index': seg.start_index,
                    'end_index': seg.end_index,
                    'direction': seg.direction,
                }
        return None

    def _get_current_pivot(self, pivots, current_index, df):
        """获取当前中枢"""
        if not pivots:
            return None

        current_price = df.iloc[current_index]['close']

        for p in pivots:
            if (p.start_index <= current_index and
                p.low <= current_price <= p.high):
                return {
                    'start_index': p.start_index,
                    'end_index': p.end_index,
                    'high': p.high,
                    'low': p.low,
                    'center': (p.high + p.low) / 2,
                }
        return None

    def _detect_divergence(self, df, current_index):
        """检测背驰"""
        try:
            macd = self.MACD(df['close'])
            macd_values = macd.calculate()[-20:]  # 最近20个值

            prices = df['close'].values[-20:]

            # 简单背驰检测
            if len(macd_values) < 5:
                return {'detected': False, 'type': ''}

            # 底背驰：价格创新低但MACD没有
            price_low = min(prices[-5:])
            macd_low = min(macd_values[-5:])

            if prices[-1] == price_low and macd_values[-1] > macd_low:
                return {'detected': True, 'type': 'bullish'}

            # 顶背驰：价格创新高但MACD没有
            price_high = max(prices[-5:])
            macd_high = max(macd_values[-5:])

            if prices[-1] == price_high and macd_values[-1] < macd_high:
                return {'detected': True, 'type': 'bearish'}

        except Exception:
            pass

        return {'detected': False, 'type': ''}

    def _calculate_confidence(self, result: PatternRecognitionResult,
                             current_index: int, total_len: int) -> float:
        """计算识别信心度"""
        confidence = 0.5

        # 有分型加分
        if result.current_fractal:
            confidence += 0.1

        # 有笔加分
        if result.current_stroke:
            confidence += 0.15

        # 有线段加分
        if result.current_segment:
            confidence += 0.15

        # 有中枢加分
        if result.current_pivot:
            confidence += 0.1

        return min(0.95, confidence)

    def _generate_reasoning(self, result: PatternRecognitionResult) -> str:
        """生成分析理由"""
        parts = []

        if result.current_fractal:
            parts.append(f"发现{result.current_fractal['type']}分型")

        if result.current_stroke:
            direction = "向上" if result.current_stroke['direction'] == 'up' else "向下"
            parts.append(f"当前笔方向{direction}")

        if result.current_segment:
            parts.append("线段结构完整")

        if result.in_pivot:
            parts.append("价格在中枢区间内")

        if result.divergence_detected:
            direction = "底" if result.divergence_type == 'bullish' else "顶"
            parts.append(f"检测到{direction}背驰")

        return "；".join(parts) if parts else "形态不明显"


class MultiTimeframePatternAgent(BaseAgent):
    """
    多周期形态识别智能体
    同时分析日线、周线、月线
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("multi_timeframe_pattern_agent", config)
        self.daily_agent = PatternAgent()
        self.weekly_agent = PatternAgent()

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        分析多个时间周期
        """
        results = {}

        # 日线分析
        daily_output = self.daily_agent.analyze(input_data)
        results['daily'] = daily_output

        # 周线分析 (如果数据足够)
        if input_data.ohlcv_data is not None and len(input_data.ohlcv_data) >= 50:
            # 转换为周线
            weekly_df = self._to_weekly(input_data.ohlcv_data)
            weekly_input = AgentInput(
                ohlcv_data=weekly_df,
                current_index=len(weekly_df) - 1,
                symbol=input_data.symbol,
                config=input_data.config
            )
            weekly_output = self.weekly_agent.analyze(weekly_input)
            results['weekly'] = weekly_output

        # 融合结果
        return self._merge_results(results)

    def _to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """将日线数据转换为周线"""
        df = daily_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        weekly.reset_index(inplace=True)
        return weekly

    def _merge_results(self, results: Dict[str, AgentOutput]) -> AgentOutput:
        """融合多周期结果"""
        daily = results.get('daily')
        weekly = results.get('weekly')

        if not daily:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                reasoning="无法获取日线数据"
            )

        reasoning_parts = [f"日线: {daily.reasoning}"]

        if weekly and weekly.success:
            reasoning_parts.append(f"周线: {weekly.reasoning}")

        # 综合信心度
        confidence = daily.confidence
        if weekly:
            confidence = (daily.confidence + weekly.confidence) / 2

        return AgentOutput(
            agent_name=self.name,
            success=True,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            data={k: v.to_dict() for k, v in results.items()}
        )
