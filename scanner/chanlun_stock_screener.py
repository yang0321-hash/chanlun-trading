"""
缠论选股器

筛选符合缠论买入条件的股票：
1. 周线趋势向上或底部
2. 日线出现2买或3买信号
3. MACD金叉或底背离
4. 成交量配合
5. 价格在合理位置
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass

from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD


@dataclass
class StockSignal:
    """股票信号"""
    symbol: str
    name: str
    signal_type: str  # '1买', '2买', '3买'
    price: float
    confidence: float
    reason: str
    weekly_trend: str
    macd_status: str
    volume_ratio: float
    risk_reward: float


class ChanLunStockScreener:
    """
    缠论选股器

    扫描股票池，找出符合缠论买入条件的股票
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        require_weekly_up: bool = True,
        require_macd_golden: bool = False,
        min_volume_ratio: float = 1.0,
        max_stocks: int = 50,
    ):
        self.min_confidence = min_confidence
        self.require_weekly_up = require_weekly_up
        self.require_macd_golden = require_macd_golden
        self.min_volume_ratio = min_volume_ratio
        self.max_stocks = max_stocks

    def screen(
        self,
        data_map: Dict[str, pd.DataFrame],
        name_map: Optional[Dict[str, str]] = None
    ) -> List[StockSignal]:
        """
        扫描股票池

        Args:
            data_map: {symbol: DataFrame} 股票数据
            name_map: {symbol: name} 股票名称映射

        Returns:
            符合条件的股票信号列表
        """
        results = []

        for symbol, df in data_map.items():
            try:
                signal = self._analyze_stock(symbol, df)
                if signal and signal.confidence >= self.min_confidence:
                    if name_map and symbol in name_map:
                        signal.name = name_map[symbol]
                    results.append(signal)
            except Exception as e:
                logger.debug(f"分析{symbol}失败: {e}")
                continue

        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)

        # 限制数量
        return results[:self.max_stocks]

    def _analyze_stock(self, symbol: str, df: pd.DataFrame) -> Optional[StockSignal]:
        """分析单只股票"""
        if len(df) < 60:
            return None

        # 生成周线
        weekly_df = self._to_weekly(df)
        if len(weekly_df) < 30:
            return None

        # 分析周线
        weekly_trend, weekly_strength = self._analyze_trend(weekly_df)

        # 周线趋势过滤
        if self.require_weekly_up and weekly_trend == 'down':
            return None

        # 分析日线
        daily_kline = KLine.from_dataframe(df, strict_mode=False)
        detector = FractalDetector(daily_kline, confirm_required=False)
        fractals = detector.get_fractals()

        stroke_gen = StrokeGenerator(daily_kline, fractals, min_bars=3)
        strokes = stroke_gen.get_strokes()

        if len(strokes) < 3:
            return None

        pivot_detector = PivotDetector(daily_kline, strokes)
        pivots = pivot_detector.get_pivots()

        macd = MACD(df['close'])

        # 识别买点
        buy_point = self._identify_buy_point(strokes, pivots, macd, df)

        if not buy_point:
            return None

        signal_type, confidence, reason, stop_loss, target = buy_point

        # MACD金叉过滤
        if self.require_macd_golden:
            if not self._check_macd_golden(macd):
                return None

        # 成交量检查
        volume_ratio = self._check_volume(df)

        # 综合评分
        final_confidence = confidence
        if weekly_trend == 'up':
            final_confidence += 0.1
        if volume_ratio > 1.5:
            final_confidence += 0.05
        if self._check_macd_divergence(macd, 'bottom'):
            final_confidence += 0.1

        final_confidence = min(final_confidence, 0.95)

        # 计算盈亏比
        risk_reward = abs(target - df['close'].iloc[-1]) / abs(df['close'].iloc[-1] - stop_loss)
        if risk_reward < 1.5:
            final_confidence -= 0.1

        if final_confidence < self.min_confidence:
            return None

        return StockSignal(
            symbol=symbol,
            name='',
            signal_type=signal_type,
            price=df['close'].iloc[-1],
            confidence=final_confidence,
            reason=reason,
            weekly_trend=weekly_trend,
            macd_status=self._get_macd_status(macd),
            volume_ratio=volume_ratio,
            risk_reward=risk_reward
        )

    def _to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """转换为周线"""
        return daily_df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

    def _analyze_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """分析趋势"""
        kline = KLine.from_dataframe(df, strict_mode=False)
        detector = FractalDetector(kline, confirm_required=False)
        fractals = detector.get_fractals()

        stroke_gen = StrokeGenerator(kline, fractals, min_bars=3)
        strokes = stroke_gen.get_strokes()

        if len(strokes) < 5:
            return 'unknown', 0.5

        recent = strokes[-10:]
        ups = [s for s in recent if s.is_up]
        downs = [s for s in recent if s.is_down]

        if not ups or not downs:
            return 'unknown', 0.5

        higher_highs = all(ups[i].end_value >= ups[i-1].end_value for i in range(1, len(ups)))
        higher_lows = all(downs[i].end_value >= downs[i-1].end_value for i in range(1, len(downs)))

        if higher_highs and higher_lows:
            strength = min(0.5 + len(ups) * 0.05, 1.0)
            return 'up', strength
        elif not higher_highs and not higher_lows:
            strength = min(0.5 + len(downs) * 0.05, 1.0)
            return 'down', strength
        else:
            return 'range', 0.4

    def _identify_buy_point(
        self,
        strokes: List[Stroke],
        pivots: List[Pivot],
        macd: MACD,
        df: pd.DataFrame
    ) -> Optional[Tuple[str, float, str, float, float]]:
        """识别买点"""
        if len(strokes) < 3:
            return None

        current_price = df['close'].iloc[-1]
        last_stroke = strokes[-1]

        # 检查2买
        if last_stroke.is_up and len(strokes) >= 3:
            # 寻找前一个低点
            prev_down = [s for s in strokes[-4:-1] if s.is_down]
            if prev_down:
                prev_low = min(s.low for s in prev_down)
                # 当前向上笔起点不低于前低
                if last_stroke.start_value >= prev_low * 0.98:
                    stop_loss = prev_low * 0.97
                    target = last_stroke.end_value * 1.15
                    return ('2买', 0.7, f'2买:回踩{prev_low:.2f}不破', stop_loss, target)

        # 检查3买
        if pivots and last_stroke.is_up:
            last_pivot = pivots[-1]
            if last_stroke.start_value > last_pivot.high:
                # 突破后回踩
                if len(strokes) >= 2 and strokes[-2].is_down:
                    if strokes[-2].low >= last_pivot.low * 0.98:
                        stop_loss = last_pivot.low * 0.97
                        target = current_price * 1.2
                        return ('3买', 0.65, f'3买:突破{last_pivot.high:.2f}确认', stop_loss, target)

        # 检查1买
        down_strokes = [s for s in strokes[-5:] if s.is_down]
        if down_strokes and len(pivots) >= 2:
            # 中枢下移
            if pivots[-1].low < pivots[-2].low:
                if self._check_macd_divergence(macd, 'bottom'):
                    stop_loss = down_strokes[-1].low * 0.95
                    target = current_price * 1.25
                    return ('1买', 0.5, f'1买:底背离', stop_loss, target)

        return None

    def _check_macd_golden(self, macd: MACD) -> bool:
        """检查MACD金叉"""
        if len(macd) < 3:
            return False
        hist = macd.histogram
        return hist[-2] <= 0 and hist[-1] > 0

    def _check_macd_divergence(self, macd: MACD, direction: str) -> bool:
        """检查MACD背离"""
        if len(macd) < 20:
            return False
        has_div, _ = macd.check_divergence(
            len(macd) - 20,
            len(macd) - 1,
            'down' if direction == 'bottom' else 'up'
        )
        return has_div

    def _get_macd_status(self, macd: MACD) -> str:
        """获取MACD状态"""
        if len(macd) < 2:
            return 'unknown'

        hist = macd.histogram[-1]
        dif = macd.dif[-1]
        dea = macd.dea[-1]

        if dif > dea and hist > 0:
            return '金叉'
        elif dif < dea and hist < 0:
            return '死叉'
        elif dif > dea:
            return '多头'
        else:
            return '空头'

    def _check_volume(self, df: pd.DataFrame, period: int = 5) -> float:
        """检查成交量"""
        if len(df) < period * 2:
            return 1.0

        recent_vol = df['volume'].tail(period).mean()
        past_vol = df['volume'].tail(period * 3).head(period * 2).mean()

        if past_vol == 0:
            return 1.0

        return recent_vol / past_vol


def format_screen_results(signals: List[StockSignal]) -> str:
    """格式化选股结果"""
    if not signals:
        return "未找到符合条件的股票"

    lines = [
        f"共找到 {len(signals)} 只符合条件的股票\n",
        "=" * 120,
        f"{'代码':<10}{'名称':<12}{'信号':<8}{'价格':<10}{'置信度':<10}{'周线':<8}{'MACD':<8}{'量比':<8}{'盈亏比':<10}{'理由':<30}",
        "-" * 120
    ]

    for s in signals:
        lines.append(
            f"{s.symbol:<10}{s.name:<12}{s.signal_type:<8}"
            f"{s.price:<10.2f}{s.confidence:<10.1%}{s.weekly_trend:<8}"
            f"{s.macd_status:<8}{s.volume_ratio:<8.2f}{s.risk_reward:<10.2f}{s.reason:<30}"
        )

    lines.append("=" * 120)
    return "\n".join(lines)
