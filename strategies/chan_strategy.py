"""
缠论交易策略

基于缠论买卖点的交易策略实现
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD
from indicator.atr import ATR
from indicator.volume_price import VolumePriceAnalyzer
from risk.triple_barrier import TripleBarrierManager, BarrierType
from risk.position_sizer import PositionSizer


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
        use_volume_price: bool = True,
        min_pivot_count: int = 1,
        # 三重屏障参数
        use_triple_barrier: bool = True,
        atr_period: int = 14,
        atr_multiplier_tp: float = 3.0,
        atr_multiplier_sl: float = 1.5,
        max_holding_bars: int = 20,
        # P1: 动态仓位 + 市场状态过滤
        use_position_sizer: bool = False,
        use_regime_filter: bool = False,
        # P2: 多周期共振 + 量价过滤
        use_multi_tf: bool = False,
        use_volume_filter: bool = False,
    ):
        """
        初始化缠论策略

        Args:
            name: 策略名称
            use_macd: 是否使用MACD判断背驰
            use_volume_price: 是否启用量价分析确认
            min_pivot_count: 最小中枢数量
            use_triple_barrier: 是否使用三重屏障退出法
            atr_period: ATR计算周期
            atr_multiplier_tp: 止盈ATR倍数
            atr_multiplier_sl: 止损ATR倍数
            max_holding_bars: 最大持仓K线数
            use_position_sizer: 是否使用ATR动态仓位管理
            use_regime_filter: 是否启用市场状态过滤
            use_multi_tf: 是否启用多周期共振过滤
            use_volume_filter: 是否启用增强量价过滤
        """
        super().__init__(name)
        self.use_macd = use_macd
        self.use_volume_price = use_volume_price
        self.min_pivot_count = min_pivot_count

        # 缠论要素缓存
        self._fractals: List[Fractal] = []
        self._strokes: List = []
        self._segments: List = []
        self._pivots: List = []
        self._macd: Optional[MACD] = None
        self._volume_price: Optional[VolumePriceAnalyzer] = None

        # 三重屏障退出系统
        self.use_triple_barrier = use_triple_barrier
        self.atr_period = atr_period
        self._triple_barrier = TripleBarrierManager(
            atr_multiplier_tp=atr_multiplier_tp,
            atr_multiplier_sl=atr_multiplier_sl,
            max_holding_bars=max_holding_bars,
        )
        self._atr: Optional[ATR] = None
        self._last_buy_point_type: Dict[str, str] = {}  # '1buy', '2buy', '3buy'

        # P1: 动态仓位管理
        self.use_position_sizer = use_position_sizer
        self._position_sizer = PositionSizer() if use_position_sizer else None

        # P1: 市场状态过滤
        self.use_regime_filter = use_regime_filter
        self._current_regime: str = 'sideways'  # 'bull', 'sideways', 'bear'

        # P2: 多周期共振
        self.use_multi_tf = use_multi_tf
        self._weekly_trend: str = 'neutral'  # 'up', 'down', 'neutral'

        # P2: 增强量价过滤
        self.use_volume_filter = use_volume_filter

        # 趋势判断
        self._current_trend = ''  # 'up', 'down', 'unknown'

        # 最后的买卖点
        self._last_buy_point: Optional[tuple] = None
        self._last_sell_point: Optional[tuple] = None

        # 入场价格追踪
        self._entry_prices: Dict[str, float] = {}

        # 跟踪止盈：最高价记录
        self._highest_prices: Dict[str, float] = {}

        # 冷却期：卖出后N根K线不买入
        self._cooldown_bars: int = 10
        self._last_sell_index: Dict[str, int] = {}

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

            # 保存原始DataFrame供量价分析使用
            self._raw_df = hist_df

            # 更新缠论要素
            self._update_chanlun_elements(kline)

            # 分析当前状态
            current_price = bar['close']
            position = self.get_position(symbol)

            # 持仓逻辑
            if position > 0:
                signal = self._check_sell_signal(current_price, index, symbol, bar=bar)
            else:
                signal = self._check_buy_signal(current_price, index, symbol)

            # 修复Bug #5：记录买卖点，让2买2卖逻辑生效
            if signal is not None:
                if signal.signal_type == SignalType.BUY:
                    self._last_buy_point = (index, current_price)
                elif signal.signal_type == SignalType.SELL:
                    self._last_sell_point = (index, current_price)
                    # 记录冷却期
                    self._last_sell_index[symbol] = index

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

        # 计算ATR（用于三重屏障止损 + 动态仓位）
        need_atr = self.use_triple_barrier or self.use_position_sizer
        if need_atr and self._raw_df is not None:
            raw = self._raw_df
            if all(c in raw.columns for c in ['high', 'low', 'close']):
                self._atr = ATR(raw['high'], raw['low'], raw['close'], self.atr_period)

        # 计算量价分析（使用原始DataFrame，避免KLine合并后index不匹配）
        if self.use_volume_price and self._raw_df is not None:
            raw = self._raw_df
            if 'close' in raw.columns and 'volume' in raw.columns:
                self._volume_price = VolumePriceAnalyzer(
                    raw['close'].values,
                    raw['volume'].values,
                    raw
                )

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

    def _detect_market_regime(self) -> str:
        """
        检测市场状态 (牛/熊/震荡)

        使用均线排列 + MACD柱线判断:
        - 牛市: MA5 > MA20 > MA60 且 MACD柱 > 0
        - 熊市: MA5 < MA20 < MA60
        - 震荡: 其他情况

        Returns:
            'bull', 'bear', 'sideways'
        """
        if self._raw_df is None or len(self._raw_df) < 60:
            return 'sideways'

        close = self._raw_df['close']
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()

        latest_ma5 = ma5.iloc[-1]
        latest_ma20 = ma20.iloc[-1]
        latest_ma60 = ma60.iloc[-1]

        macd_hist = 0.0
        if self._macd is not None:
            latest = self._macd.get_latest()
            if latest is not None:
                macd_hist = latest.histogram

        if latest_ma5 > latest_ma20 > latest_ma60 and macd_hist > 0:
            return 'bull'
        elif latest_ma5 < latest_ma20 < latest_ma60:
            return 'bear'
        else:
            return 'sideways'

    def _calculate_position_size(
        self,
        current_price: float,
        confidence: float
    ) -> Optional[int]:
        """
        使用ATR计算动态仓位大小

        Args:
            current_price: 当前价格
            confidence: 信号置信度

        Returns:
            股数 (None表示回退到引擎默认)
        """
        if self._atr is None or self._position_sizer is None:
            return None

        latest_atr = self._atr.get_latest()
        if latest_atr is None or latest_atr.atr <= 0:
            return None

        equity = self.get_cash()
        risk_pct = self._position_sizer.get_risk_pct_for_confidence(confidence)

        # 震荡市减半仓位
        if self.use_regime_filter and self._current_regime == 'sideways':
            risk_pct *= 0.5

        quantity = self._position_sizer.calculate(
            equity=equity,
            current_price=current_price,
            atr_value=latest_atr.atr,
            min_unit=100,
            available_cash=self.get_cash(),
            risk_pct=risk_pct,
        )
        return quantity if quantity > 0 else None

    def _get_weekly_trend(self, daily_df: pd.DataFrame) -> str:
        """
        获取周线趋势方向 (用于多周期共振)

        通过MA5/MA20判断周线级别趋势:
        - 'up': MA5 > MA20, 周线上升趋势
        - 'down': MA5 < MA20, 周线下降趋势
        - 'neutral': 数据不足或无明显趋势

        Args:
            daily_df: 日线数据

        Returns:
            趋势方向
        """
        if daily_df is None or len(daily_df) < 100:
            return 'neutral'

        try:
            weekly = daily_df.resample('W').agg({'close': 'last'}).dropna()
            if len(weekly) < 20:
                return 'neutral'

            ma5 = weekly['close'].rolling(5).mean()
            ma20 = weekly['close'].rolling(20).mean()

            if pd.isna(ma5.iloc[-1]) or pd.isna(ma20.iloc[-1]):
                return 'neutral'

            if ma5.iloc[-1] > ma20.iloc[-1] * 1.005:  # 0.5% margin to avoid noise
                return 'up'
            elif ma5.iloc[-1] < ma20.iloc[-1] * 0.995:
                return 'down'
            return 'neutral'
        except Exception:
            return 'neutral'

    def _check_volume_surge(self, index: int) -> bool:
        """
        检测当前K线是否放量 (成交量 > 2倍均量)

        Args:
            index: K线索引

        Returns:
            是否放量
        """
        if self._raw_df is None or 'volume' not in self._raw_df.columns:
            return False

        vol = self._raw_df['volume']
        if len(vol) < 20:
            return False

        try:
            ma20_vol = vol.iloc[-20:].mean()
            current_vol = vol.iloc[-1]
            return current_vol > ma20_vol * 2.0
        except Exception:
            return False

    def _check_volume_shrink_before_surge(self, lookback: int = 5) -> bool:
        """
        检测"缩量洗盘→放量突破"形态

        在最近N根K线中，是否出现过缩量(<0.5倍均量)后放量的过程。

        Args:
            lookback: 回看K线数

        Returns:
            是否检测到缩量→放量形态
        """
        if self._raw_df is None or 'volume' not in self._raw_df.columns:
            return False

        vol = self._raw_df['volume']
        if len(vol) < 25:
            return False

        try:
            ma20_vol = vol.iloc[-25:-5].mean()  # 使用更早的数据算均量
            recent = vol.iloc[-lookback:]

            has_shrink = any(v < ma20_vol * 0.5 for v in recent)
            current_surge = vol.iloc[-1] > ma20_vol * 1.5

            return has_shrink and current_surge
        except Exception:
            return False

    def _apply_buy_enhancements(
        self,
        confidence: float,
        buy_type: str,
    ) -> float:
        """
        应用P2买入信号增强: 多周期共振 + 量价过滤

        Args:
            confidence: 当前置信度
            buy_type: 买点类型 ('1buy', '2buy', '3buy')

        Returns:
            调整后的置信度
        """
        # 多周期共振
        if self.use_multi_tf and self._raw_df is not None:
            self._weekly_trend = self._get_weekly_trend(self._raw_df)
            if self._weekly_trend == 'up':
                confidence = min(1.0, confidence + 0.05)

        # 增强量价过滤
        if self.use_volume_filter:
            if self._check_volume_shrink_before_surge():
                confidence = min(1.0, confidence + 0.10)
            elif self._check_volume_surge(0):
                confidence = min(1.0, confidence + 0.05)

        return confidence

    def _check_buy_signal(
        self,
        current_price: float,
        index: int,
        symbol: str
    ) -> Optional[Signal]:
        """检查买入信号"""
        if not self._fractals or not self._strokes:
            return None

        # 市场状态过滤
        if self.use_regime_filter:
            self._current_regime = self._detect_market_regime()
            if self._current_regime == 'bear':
                return None

        # 冷却期检查：卖出后N根K线不买入
        last_sell_idx = self._last_sell_index.get(symbol, -999)
        if index - last_sell_idx < self._cooldown_bars:
            return None

        # 检查最后是否有底分型
        last_fractal = self._fractals[-1] if self._fractals else None
        if not last_fractal or last_fractal.is_top:
            return None

        # 检查第一类买点：底背驰
        if self._check_first_buy_point(current_price, index):
            # P2: 周线下降趋势中不做1买（避免接飞刀）
            if self.use_multi_tf:
                self._weekly_trend = self._get_weekly_trend(self._raw_df)
                if self._weekly_trend == 'down':
                    return None

            base_confidence = 0.8
            reason = '第一类买点: 底背驰'
            confidence, reason = self._apply_volume_price_confirmation(
                index, base_confidence, reason, '1buy'
            )
            # P2: 多周期共振 + 量价增强
            confidence = self._apply_buy_enhancements(confidence, '1buy')
            if self.use_regime_filter and self._current_regime == 'sideways' and confidence < 0.65:
                return None
            if confidence <= 0:
                return None
            self._last_buy_point_type[symbol] = '1buy'
            quantity = self._calculate_position_size(current_price, confidence) if self.use_position_sizer else None
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                quantity=quantity,
                reason=reason,
                confidence=confidence,
                metadata={'index': index, 'buy_point_type': '1buy'}
            )

        # 检查第二类买点：回抽不破前低
        if self._check_second_buy_point(current_price, index):
            base_confidence = 0.7
            reason = '第二类买点: 回抽不破前低'
            confidence, reason = self._apply_volume_price_confirmation(
                index, base_confidence, reason, '2buy'
            )
            confidence = self._apply_buy_enhancements(confidence, '2buy')
            if self.use_regime_filter and self._current_regime == 'sideways' and confidence < 0.65:
                return None
            if confidence <= 0:
                return None
            self._last_buy_point_type[symbol] = '2buy'
            quantity = self._calculate_position_size(current_price, confidence) if self.use_position_sizer else None
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                quantity=quantity,
                reason=reason,
                confidence=confidence,
                metadata={'index': index, 'buy_point_type': '2buy'}
            )

        # 检查第三类买点：突破中枢后回踩
        if self._check_third_buy_point(current_price, index):
            base_confidence = 0.6
            reason = '第三类买点: 突破中枢回踩'
            confidence, reason = self._apply_volume_price_confirmation(
                index, base_confidence, reason, '3buy'
            )
            confidence = self._apply_buy_enhancements(confidence, '3buy')
            if self.use_regime_filter and self._current_regime == 'sideways' and confidence < 0.65:
                return None
            if confidence <= 0:
                return None
            self._last_buy_point_type[symbol] = '3buy'
            quantity = self._calculate_position_size(current_price, confidence) if self.use_position_sizer else None
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                quantity=quantity,
                reason=reason,
                confidence=confidence,
                metadata={'index': index, 'buy_point_type': '3buy'}
            )

        return None

    def _check_sell_signal(
        self,
        current_price: float,
        index: int,
        symbol: str,
        bar: pd.Series = None
    ) -> Optional[Signal]:
        """
        检查卖出信号

        三优先级退出系统:
        1. 三重屏障 (ATR动态止损/止盈/时间退出)
        2. 中枢边界止损 (缠论中枢结构止损)
        3. 缠论分型卖出信号 (1卖/2卖)
        """
        entry_price = self._get_entry_price(symbol)

        # === 优先级1: 三重屏障退出 ===
        if self.use_triple_barrier and entry_price:
            current_high = bar['high'] if bar is not None else current_price
            current_low = bar['low'] if bar is not None else current_price
            exit_result = self._triple_barrier.check_exit(
                symbol=symbol,
                current_price=current_price,
                current_index=index,
                current_high=current_high,
                current_low=current_low
            )
            if exit_result is not None and exit_result.triggered:
                confidence = 1.0 if exit_result.barrier_type == BarrierType.LOWER else 0.9
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=pd.Timestamp.now(),
                    price=current_price,
                    reason=exit_result.reason,
                    confidence=confidence
                )

        # === 优先级2: 中枢边界止损 ===
        if entry_price and self._pivots:
            pivot_stop = self._calculate_pivot_stop_loss(symbol, entry_price)
            if pivot_stop is not None and current_price <= pivot_stop:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=pd.Timestamp.now(),
                    price=current_price,
                    reason=f'中枢止损: 价格{current_price:.2f}跌破中枢边界{pivot_stop:.2f}',
                    confidence=1.0
                )

        # === 优先级3: 缠论分型卖出信号 ===
        if not self._fractals or not self._strokes:
            return None

        # 检查最后是否有顶分型
        last_fractal = self._fractals[-1] if self._fractals else None
        if not last_fractal or last_fractal.is_bottom:
            return None

        # 检查第一类卖点：顶背驰
        if self._check_first_sell_point(current_price, index):
            base_confidence = 0.8
            reason = '第一类卖点: 顶背驰'
            confidence, reason = self._apply_volume_price_confirmation(
                index, base_confidence, reason, sell=True
            )
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason=reason,
                confidence=confidence
            )

        # 检查第二类卖点：反弹不破前高
        if self._check_second_sell_point(current_price, index):
            base_confidence = 0.7
            reason = '第二类卖点: 反弹不破前高'
            confidence, reason = self._apply_volume_price_confirmation(
                index, base_confidence, reason, sell=True
            )
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason=reason,
                confidence=confidence
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

        # 检查MACD背驰：使用笔对齐而非固定窗口
        if self.use_macd and self._macd:
            down_strokes = [s for s in self._strokes if s.is_down]
            if len(down_strokes) >= 2:
                prev_stroke = down_strokes[-2]
                curr_stroke = down_strokes[-1]
                has_divergence, _ = self._macd.check_divergence(
                    curr_stroke.start_index,
                    curr_stroke.end_index,
                    'down',
                    prev_start=prev_stroke.start_index,
                    prev_end=prev_stroke.end_index
                )
                return has_divergence

        return True

    def _check_second_buy_point(self, current_price: float, index: int) -> bool:
        """检查第二类买点"""
        if self._last_buy_point is None:
            # 没有1买记录，尝试从中枢推断
            return self._check_second_buy_from_pivot(current_price, index)

        buy_idx, buy_price = self._last_buy_point

        # 回抽不破前低
        return current_price > buy_price * 0.98

    def _check_second_buy_from_pivot(self, current_price: float, index: int) -> bool:
        """从中枢推断2买：中枢下方有底分型，价格在中枢范围内"""
        if not self._pivots or not self._strokes:
            return False

        last_pivot = self._pivots[-1]

        # 找中枢下方的向下笔（作为推断的1买低点）
        down_strokes = [s for s in self._strokes
                        if s.is_down and s.end_value < last_pivot.low]
        if not down_strokes:
            return False

        last_down = down_strokes[-1]
        implied_buy_price = last_down.end_value

        # 之后有向上笔和向下笔回调
        up_after = [s for s in self._strokes
                    if s.is_up and s.start_index > last_down.end_index]
        if not up_after:
            return False

        # 回调不破推断的1买低点
        return current_price > implied_buy_price * 0.98 and current_price < last_pivot.low

    def _check_third_buy_point(self, current_price: float, index: int) -> bool:
        """检查第三类买点"""
        if not self._pivots:
            return False

        last_pivot = self._pivots[-1]

        # 价格突破中枢后回踩不破中枢上沿（放宽到1.08）
        return (current_price > last_pivot.high * 0.99 and
                current_price < last_pivot.high * 1.08)

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

        # 检查MACD背驰：使用笔对齐而非固定窗口
        if self.use_macd and self._macd:
            up_strokes = [s for s in self._strokes if s.is_up]
            if len(up_strokes) >= 2:
                prev_stroke = up_strokes[-2]
                curr_stroke = up_strokes[-1]
                has_divergence, _ = self._macd.check_divergence(
                    curr_stroke.start_index,
                    curr_stroke.end_index,
                    'up',
                    prev_start=prev_stroke.start_index,
                    prev_end=prev_stroke.end_index
                )
                return has_divergence

        return True

    def _apply_volume_price_confirmation(
        self,
        index: int,
        base_confidence: float,
        reason: str,
        buy_point_type: str = '',
        sell: bool = False
    ) -> tuple:
        """
        应用量价分析确认，调整信号置信度

        Args:
            index: K线索引
            base_confidence: 基础置信度
            reason: 原始原因
            buy_point_type: 买点类型 ('1buy', '2buy', '3buy')
            sell: 是否为卖出信号

        Returns:
            (调整后置信度, 增强原因)
            当置信度低于 min_confidence 阈值时返回 (0, reason) 表示过滤
        """
        if not self.use_volume_price or self._volume_price is None:
            return base_confidence, reason

        try:
            if sell:
                adj, vp_reason = self._volume_price.check_sell_confirmation(
                    index, self._strokes
                )
            else:
                adj, vp_reason = self._volume_price.check_buy_confirmation(
                    index, self._strokes, buy_point_type
                )

            confidence = max(0.1, min(1.0, base_confidence + adj))
            if vp_reason and vp_reason != '无量价确认':
                reason = f"{reason} | {vp_reason}"

            # 量价确认过滤：置信度过低则返回0（过滤信号）
            min_confidence = 0.4
            if confidence < min_confidence:
                logger.debug(f"量价过滤: {reason}, confidence={confidence:.2f} < {min_confidence}")
                return 0.0, f"[已过滤] {reason}"

            return confidence, reason
        except Exception as e:
            logger.debug(f"量价分析异常: {e}")
            return base_confidence, reason

    def _check_second_sell_point(self, current_price: float, index: int) -> bool:
        """检查第二类卖点"""
        if self._last_sell_point is None:
            return False

        sell_idx, sell_price = self._last_sell_point

        # 反弹不破前高
        return current_price < sell_price * 1.02

    def _calculate_pivot_stop_loss(
        self,
        symbol: str,
        entry_price: float
    ) -> Optional[float]:
        """
        基于中枢边界计算止损价格

        止损规则:
        - 1买/2买后: 止损于 pivot.dd (中枢波动最低点) 或 pivot.low * 0.99
        - 3买后: 止损于 pivot.low (中枢下边界ZD)
        - 最大止损不超过入场价的7%

        Args:
            symbol: 股票代码
            entry_price: 入场价格

        Returns:
            止损价格, None表示无法计算
        """
        if not self._pivots:
            return None

        last_pivot = self._pivots[-1]
        buy_type = self._last_buy_point_type.get(symbol, '')

        # 根据买点类型确定止损位
        if buy_type in ('3buy',):
            # 3买: 止损于中枢下边界 (ZD = pivot.low)
            pivot_stop = last_pivot.low
        else:
            # 1买/2买/未知: 止损于中枢波动最低点
            pivot_stop = max(last_pivot.dd, last_pivot.low * 0.99)

        # 止损价必须低于入场价
        if pivot_stop >= entry_price:
            return None

        # 最大止损不超过7% (防止中枢过宽)
        max_stop = entry_price * 0.93
        pivot_stop = max(pivot_stop, max_stop)

        return pivot_stop

    def _get_entry_price(self, symbol: str) -> Optional[float]:
        """获取入场价格"""
        return self._entry_prices.get(symbol)

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调 - 记录入场/出场价格"""
        super().on_order(signal, executed_price, executed_quantity)
        symbol = signal.symbol
        if signal.is_buy():
            self._entry_prices[symbol] = executed_price
            self._highest_prices[symbol] = executed_price

            # 注册三重屏障
            if self.use_triple_barrier and self._atr is not None:
                latest_atr = self._atr.get_latest()
                if latest_atr is not None and latest_atr.atr > 0:
                    entry_index = signal.metadata.get('index', 0) if signal.metadata else 0
                    self._triple_barrier.open_trade(
                        symbol=symbol,
                        entry_price=executed_price,
                        entry_index=entry_index,
                        atr_value=latest_atr.atr
                    )

        elif signal.is_sell():
            self._entry_prices.pop(symbol, None)
            self._highest_prices.pop(symbol, None)
            self._last_buy_point_type.pop(symbol, None)
            # 记录卖出时的K线索引用于冷却期
            self._last_sell_index[symbol] = getattr(signal, '_index', 0)
            # 关闭三重屏障
            if self.use_triple_barrier:
                self._triple_barrier.close_trade(symbol)

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
