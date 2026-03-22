"""
缠论交易策略

基于缠论买卖点的交易策略实现
"""

from typing import List, Optional, Dict, Any
import pandas as pd
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD


class BuySellPoint:
    """买卖点类型"""
    FIRST_BUY = '1buy'      # 第一类买点
    FIRST_SELL = '1sell'    # 第一类卖点
    SECOND_BUY = '2buy'     # 第二类买点
    SECOND_SELL = '2sell'   # 第二类卖点
    THIRD_BUY = '3buy'      # 第三类买点
    THIRD_SELL = '3sell'    # 第三类卖点


class ChanLunStrategy(Strategy):
    """
    缠论交易策略

    买卖点定义：
    - 第一类买点：下跌趋势中，最后中枢下方的底背驰点
    - 第一类卖点：上涨趋势中，最后中枢上方的顶背驰点
    - 第二类买点：第一类买点后，回抽不破前低的点
    - 第二类卖点：第一类卖点后，反弹不破前高的点
    - 第三类买点：上涨中，回抽不进中枢的点（介入点）
    - 第三类卖点：下跌中，反弹不进中枢的点（卖出点）
    """

    def __init__(
        self,
        name: str = '缠论策略',
        use_macd: bool = True,
        min_pivot_count: int = 1
    ):
        """
        初始化缠论策略

        Args:
            name: 策略名称
            use_macd: 是否使用MACD判断背驰
            min_pivot_count: 最小中枢数量
        """
        super().__init__(name)
        self.use_macd = use_macd
        self.min_pivot_count = min_pivot_count

        # 缠论要素缓存
        self._fractals: List[Fractal] = []
        self._strokes: List = []
        self._segments: List = []
        self._pivots: List = []
        self._macd: Optional[MACD] = None

        # 趋势判断
        self._current_trend = ''  # 'up', 'down', 'unknown'

        # 最后的买卖点
        self._last_buy_point: Optional[tuple] = None
        self._last_sell_point: Optional[tuple] = None

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金¥{capital:,}, 品种{symbols}")

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        处理K线

        Args:
            bar: 当前K线
            symbol: 股票代码
            index: K线索引
            context: 上下文

        Returns:
            交易信号
        """
        # 获取历史数据
        hist_df = context['data'].get(symbol)
        if hist_df is None or len(hist_df) < 30:
            return None

        try:
            # 转换为KLine对象
            from core.kline import KLine
            kline = KLine.from_dataframe(hist_df, strict_mode=True)

            # 更新缠论要素
            self._update_chanlun_elements(kline)

            # 分析当前状态
            current_price = bar['close']
            position = self.get_position(symbol)

            # 持仓逻辑
            if position > 0:
                signal = self._check_sell_signal(current_price, index, symbol)
            else:
                signal = self._check_buy_signal(current_price, index, symbol)

            return signal

        except Exception as e:
            logger.debug(f"分析错误 {symbol} @ {index}: {e}")
            return None

    def _update_chanlun_elements(self, kline) -> None:
        """更新缠论要素"""
        # 识别分型
        detector = FractalDetector(kline, confirm_required=False)
        self._fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(kline, self._fractals)
        self._strokes = stroke_gen.get_strokes()

        # 生成线段
        seg_gen = SegmentGenerator(kline, self._strokes)
        self._segments = seg_gen.get_segments()

        # 识别中枢
        pivot_detector = PivotDetector(kline, self._strokes)
        self._pivots = pivot_detector.get_pivots()

        # 计算MACD
        if self.use_macd:
            df = kline.to_dataframe()
            self._macd = MACD(df['close'])

        # 判断趋势
        self._update_trend()

    def _update_trend(self) -> None:
        """更新当前趋势"""
        if not self._segments:
            self._current_trend = 'unknown'
            return

        # 检查最后几个线段的方向
        last_segments = self._segments[-3:]
        up_count = sum(1 for s in last_segments if s.direction == 'up')
        down_count = len(last_segments) - up_count

        if up_count > down_count:
            self._current_trend = 'up'
        elif down_count > up_count:
            self._current_trend = 'down'
        else:
            # 检查最后一个线段
            if self._segments:
                self._current_trend = self._segments[-1].direction
            else:
                self._current_trend = 'unknown'

    def _check_buy_signal(
        self,
        current_price: float,
        index: int,
        symbol: str
    ) -> Optional[Signal]:
        """检查买入信号"""
        if not self._fractals or not self._strokes:
            return None

        # 检查最后是否有底分型
        last_fractal = self._fractals[-1] if self._fractals else None
        if not last_fractal or last_fractal.is_top:
            return None

        # 检查第一类买点：底背驰
        if self._check_first_buy_point(current_price, index):
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason=f'第一类买点: 底背驰',
                confidence=0.8
            )

        # 检查第二类买点：回抽不破前低
        if self._check_second_buy_point(current_price, index):
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason='第二类买点: 回抽不破前低',
                confidence=0.7
            )

        # 检查第三类买点：突破中枢后回踩
        if self._check_third_buy_point(current_price, index):
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason='第三类买点: 突破中枢回踩',
                confidence=0.6
            )

        return None

    def _check_sell_signal(
        self,
        current_price: float,
        index: int,
        symbol: str
    ) -> Optional[Signal]:
        """检查卖出信号"""
        if not self._fractals or not self._strokes:
            return None

        # 检查最后是否有顶分型
        last_fractal = self._fractals[-1] if self._fractals else None
        if not last_fractal or last_fractal.is_bottom:
            return None

        # 检查第一类卖点：顶背驰
        if self._check_first_sell_point(current_price, index):
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason=f'第一类卖点: 顶背驰',
                confidence=0.8
            )

        # 检查第二类卖点：反弹不破前高
        if self._check_second_sell_point(current_price, index):
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason='第二类卖点: 反弹不破前高',
                confidence=0.7
            )

        # 止损
        entry_price = self._get_entry_price(symbol)
        if entry_price:
            stop_loss = entry_price * 0.95  # 5%止损
            if current_price < stop_loss:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=last_fractal.datetime,
                    price=current_price,
                    reason=f'止损: 价格跌破{stop_loss:.2f}',
                    confidence=1.0
                )

        return None

    def _check_first_buy_point(self, current_price: float, index: int) -> bool:
        """检查第一类买点（底背驰）"""
        if len(self._strokes) < 3:
            return False

        # 检查是否有中枢
        if not self._pivots:
            return False

        last_pivot = self._pivots[-1]
        last_stroke = self._strokes[-1]

        # 检查是否是向下笔
        if last_stroke.is_up:
            return False

        # 检查价格是否低于中枢下沿
        if current_price >= last_pivot.low:
            return False

        # 检查MACD背驰
        if self.use_macd and self._macd:
            if len(self._macd) > 20:
                has_divergence, _ = self._macd.check_divergence(
                    len(self._macd) - 20,
                    len(self._macd) - 1,
                    'down'
                )
                return has_divergence

        return True

    def _check_second_buy_point(self, current_price: float, index: int) -> bool:
        """检查第二类买点"""
        if self._last_buy_point is None:
            return False

        buy_idx, buy_price = self._last_buy_point

        # 回抽不破前低
        return current_price > buy_price * 0.98

    def _check_third_buy_point(self, current_price: float, index: int) -> bool:
        """检查第三类买点"""
        if not self._pivots:
            return False

        last_pivot = self._pivots[-1]

        # 价格突破中枢后回踩不破中枢上沿
        return (current_price > last_pivot.high * 0.99 and
                current_price < last_pivot.high * 1.02)

    def _check_first_sell_point(self, current_price: float, index: int) -> bool:
        """检查第一类卖点（顶背驰）"""
        if len(self._strokes) < 3:
            return False

        # 检查是否有中枢
        if not self._pivots:
            return False

        last_pivot = self._pivots[-1]
        last_stroke = self._strokes[-1]

        # 检查是否是向上笔
        if last_stroke.is_down:
            return False

        # 检查价格是否高于中枢上沿
        if current_price <= last_pivot.high:
            return False

        # 检查MACD背驰
        if self.use_macd and self._macd:
            if len(self._macd) > 20:
                has_divergence, _ = self._macd.check_divergence(
                    len(self._macd) - 20,
                    len(self._macd) - 1,
                    'up'
                )
                return has_divergence

        return True

    def _check_second_sell_point(self, current_price: float, index: int) -> bool:
        """检查第二类卖点"""
        if self._last_sell_point is None:
            return False

        sell_idx, sell_price = self._last_sell_point

        # 反弹不破前高
        return current_price < sell_price * 1.02

    def _get_entry_price(self, symbol: str) -> Optional[float]:
        """获取入场价格"""
        # 简化实现，可以从trades记录中获取
        return None

    @property
    def fractals(self) -> List[Fractal]:
        """获取分型列表"""
        return self._fractals

    @property
    def strokes(self) -> List:
        """获取笔列表"""
        return self._strokes

    @property
    def pivots(self) -> List:
        """获取中枢列表"""
        return self._pivots

    @property
    def current_trend(self) -> str:
        """获取当前趋势"""
        return self._current_trend
