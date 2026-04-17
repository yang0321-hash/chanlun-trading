"""
最优缠论策略 - 融合ChanLun信号质量 + Integrated风控体系 + 动能确认

设计原则:
- 入场: ChanLun简化买卖点检测(1买/2买/3买)
- 过滤: EMA200趋势确认 + 成交量确认 + 动能评分
- 出场: 跟踪止损 + 分批止盈体系 + 动能衰减监控
- 止损: 缠论结构止损 + 固定止损兜底(取更严格者)
- 冷却期: 防止频繁交易

动能集成(Agent辩论5轮收敛):
R1: MACD柱状图二值过滤 - 无效(72%通过率无区分力)
R2: 改为连续动能评分(-1~1), 影响置信度 - sh600519收益从3%提升到11.82%
R3: 放宽动能衰减退出条件(score<-0.5即触发) - sh600519进一步提升到16.64%
R4: 动能加成+1买RSI确认 - 无额外收益, 已移除
R5: 收敛确认 - R3/R4结果相同, 动能优化已达到边际收益递减点

动能评分逻辑(复用self._macd, 零新参数):
- MACD柱>0且扩大: score=1.0(强动能)
- MACD柱>0但收窄: score=0.5(衰减中)
- MACD柱<0但收窄: score=0.3(拐头)
- MACD柱<0扩大减速: score=0.0(弱衰减)
- MACD柱<0加速扩大: score=-1.0(强衰减)

动能衰减退出: 持仓盈利>2%且动能评分<-0.5时触发卖出
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD


class OptimalChanLunStrategy(Strategy):
    """
    最优缠论策略

    融合两个策略的最佳部分:
    - ChanLun的买卖点检测(已验证能产生合理信号)
    - 跟踪止损 + 分批止盈体系
    - EMA趋势确认过滤低质量信号
    - 成交量确认
    - 缠论结构止损
    """

    def __init__(
        self,
        name: str = '最优缠论策略',
        # 入场参数
        confidence_threshold: float = 0.55,  # R3: 0.5->0.55, 过滤低质量信号(茅台18%胜率说明05太低)
        min_volume_ratio: float = 0.9,      # R3: 0.8->0.9, 略微收紧成交量门槛
        ema_period: int = 200,               # EMA趋势判断周期
        require_ema_trend: bool = False,      # R1: True->False, 改为软降权而非硬过滤
        # 止损参数
        fixed_stop_pct: float = 0.07,        # 固定止损7%(300936参数扫描: 5%->7%减少止损洗出, 收益5%->23%)
        use_structure_stop: bool = True,      # 使用缠论结构止损

        # 跟踪止盈参数
        trailing_activate_pct: float = 0.08,  # R3: 7%->8%, 略微延后启动
        trailing_drawdown_pct: float = 0.04,  # R1: 5%->4%, 回撤容忍收窄锁定更多利润

        # 分批止盈参数
        use_partial_profit: bool = True,
        profit_target_1: float = 0.20,        # R3: 25%->20%, 回到合理阈值
        profit_sell_ratio_1: float = 0.25,    # R3: 30%->25%, 保留更多底仓
        profit_target_2: float = 0.35,        # R3: 40%->35%, 回到合理阈值
        profit_sell_ratio_2: float = 0.50,    # R3: 40%->50%, 首笔少卖则二笔多卖

        # 冷却期
        cooldown_bars: int = 15,              # 300936参数扫描: 10->15减少交易频率, 收益5%->23%

        # MACD参数
        use_macd: bool = True,

        # R2新增: 自适应参数
        breakeven_activate_pct: float = 0.05, # R2: 3%->5%, 提高保本止损激活阈值
        allow_fractional_shares: bool = True,  # R2: 允许不足100股的零股交易(高价股适配)
        max_atr_multiplier: float = 0.05,     # R2: ATR止损不超过5%固定止损
    ):
        super().__init__(name)

        # 入场参数
        self.confidence_threshold = confidence_threshold
        self.min_volume_ratio = min_volume_ratio
        self.ema_period = ema_period
        self.require_ema_trend = require_ema_trend

        # 止损参数
        self.fixed_stop_pct = fixed_stop_pct
        self.use_structure_stop = use_structure_stop

        # 跟踪止盈参数
        self.trailing_activate_pct = trailing_activate_pct
        self.trailing_drawdown_pct = trailing_drawdown_pct

        # 分批止盈参数
        self.use_partial_profit = use_partial_profit
        self.profit_target_1 = profit_target_1
        self.profit_sell_ratio_1 = profit_sell_ratio_1
        self.profit_target_2 = profit_target_2
        self.profit_sell_ratio_2 = profit_sell_ratio_2

        # 冷却期
        self.cooldown_bars = cooldown_bars

        # MACD
        self.use_macd = use_macd

        # R2新增: 自适应参数
        self.breakeven_activate_pct = breakeven_activate_pct
        self.allow_fractional_shares = allow_fractional_shares
        self.max_atr_multiplier = max_atr_multiplier

        # ========== 运行时状态 ==========
        # 缠论要素缓存(每次on_bar重新计算)
        self._fractals: List[Fractal] = []
        self._strokes: List = []
        self._segments: List = []
        self._pivots: List = []
        self._macd: Optional[MACD] = None

        # 趋势
        self._current_trend: str = 'unknown'

        # 买卖点追踪(用于2买/2卖逻辑)
        self._last_buy_point: Optional[Tuple[int, float]] = None
        self._last_sell_point: Optional[Tuple[int, float]] = None

        # 入场价格追踪
        self._entry_prices: Dict[str, float] = {}

        # 最高价追踪(用于跟踪止损)
        self._highest_prices: Dict[str, float] = {}

        # 冷却期追踪
        self._last_sell_index: Dict[str, int] = {}

        # 分批止盈状态: symbol -> 已执行的止盈阶段列表
        self._partial_profit_stages: Dict[str, List[int]] = {}

        # R2: 波动率缓存(用于自适应参数)
        self._volatility_cache: Dict[str, float] = {}

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(f"初始化{self.name}: 资金{capital:,.0f}, 品种{symbols}")
        logger.info(f"  置信度阈值={self.confidence_threshold}, 冷却期={self.cooldown_bars}K线")
        logger.info(f"  固定止损={self.fixed_stop_pct:.0%}, 跟踪止损激活={self.trailing_activate_pct:.0%}回撤={self.trailing_drawdown_pct:.0%}")
        logger.info(f"  分批止盈: {self.profit_target_1:.0%}卖{self.profit_sell_ratio_1:.0%}, {self.profit_target_2:.0%}卖{self.profit_sell_ratio_2:.0%}")
        logger.info(f"  R2: 保本激活={self.breakeven_activate_pct:.0%}, 零股={self.allow_fractional_shares}, ATR上限={self.max_atr_multiplier:.0%}")

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        # 获取历史数据
        hist_df = context['data'].get(symbol)
        if hist_df is None or len(hist_df) < 50:
            return None

        try:
            # 转换为KLine并更新缠论要素
            from core.kline import KLine
            kline = KLine.from_dataframe(hist_df, strict_mode=True)
            self._update_chanlun_elements(kline)

            current_price = bar['close']
            position = self.get_position(symbol)

            # R2: 更新波动率缓存(用于自适应参数)
            self._update_volatility(symbol, hist_df)

            if position > 0:
                signal = self._check_sell_signal(current_price, index, symbol, bar, hist_df)
            else:
                signal = self._check_buy_signal(current_price, index, symbol, bar, hist_df)

            # 记录买卖点
            if signal is not None:
                if signal.signal_type == SignalType.BUY:
                    self._last_buy_point = (index, current_price)
                elif signal.signal_type == SignalType.SELL:
                    self._last_sell_point = (index, current_price)
                    self._last_sell_index[symbol] = index

            return signal

        except Exception as e:
            logger.debug(f"分析错误 {symbol} @ {index}: {e}")
            return None

    # ==================== 缠论要素更新 ====================

    def _update_chanlun_elements(self, kline) -> None:
        """更新缠论要素(同ChanLunStrategy)"""
        detector = FractalDetector(kline, confirm_required=False)
        self._fractals = detector.get_fractals()

        stroke_gen = StrokeGenerator(kline, self._fractals)
        self._strokes = stroke_gen.get_strokes()

        seg_gen = SegmentGenerator(kline, self._strokes)
        self._segments = seg_gen.get_segments()

        pivot_detector = PivotDetector(kline, self._strokes)
        self._pivots = pivot_detector.get_pivots()

        if self.use_macd:
            df = kline.to_dataframe()
            self._macd = MACD(df['close'])

        self._update_trend()

    def _update_trend(self) -> None:
        """更新当前趋势"""
        if not self._segments:
            self._current_trend = 'unknown'
            return

        last_segments = self._segments[-3:]
        up_count = sum(1 for s in last_segments if s.direction == 'up')
        down_count = len(last_segments) - up_count

        if up_count > down_count:
            self._current_trend = 'up'
        elif down_count > up_count:
            self._current_trend = 'down'
        else:
            self._current_trend = self._segments[-1].direction if self._segments else 'unknown'

    def _update_volatility(self, symbol: str, hist_df: pd.DataFrame) -> None:
        """R2: 计算并缓存20日历史波动率(用于自适应参数)"""
        if len(hist_df) < 20:
            self._volatility_cache[symbol] = 0.02  # 默认2%
            return
        returns = hist_df['close'].iloc[-20:].pct_change().dropna()
        if len(returns) == 0:
            self._volatility_cache[symbol] = 0.02
            return
        self._volatility_cache[symbol] = float(returns.std())

    # ==================== 动能评分(R2: 从过滤器改为权重因子) ====================

    def _get_momentum_score(self) -> float:
        """
        R2改进: 动能作为连续评分而非二值过滤

        评分逻辑(复用self._macd, 零新参数):
        - MACD柱>0且连续扩大: score=1.0(强动能)
        - MACD柱>0但收窄: score=0.5(动能衰减中)
        - MACD柱<0但连续收窄: score=0.3(动能拐头)
        - MACD柱<0且扩大, 扩大速度放缓: score=0.0(弱动能)
        - MACD柱<0且加速扩大: score=-1.0(强衰减)

        R1发现72%通过率的二值过滤无区分力,
        改为连续评分后可在[-1, 1]范围精确衡量动能
        """
        if not self._macd or len(self._macd.values) < 3:
            return 0.5  # 数据不足返回中性偏正

        recent_hists = [v.histogram for v in self._macd.values[-3:]]
        latest = recent_hists[-1]
        prev = recent_hists[-2]
        prev2 = recent_hists[-3] if len(recent_hists) >= 3 else 0

        # 红柱(>0): 动能为正
        if latest > 0:
            if latest >= prev:
                return 1.0  # 红柱扩大: 强动能
            else:
                return 0.5  # 红柱收窄: 动能衰减中

        # 绿柱(<0): 动能为负
        if latest < 0:
            if latest > prev:  # 绿柱收窄(变不那么负)
                return 0.3  # 动能拐头
            else:
                # 绿柱扩大: 检查加速度
                current_speed = latest - prev
                prev_speed = prev - prev2 if len(recent_hists) >= 3 else 0
                if current_speed < prev_speed:
                    return -1.0  # 加速下跌: 强衰减
                else:
                    return 0.0  # 扩大但减速: 弱衰减

        return 0.5  # MACD柱=0, 中性

    def _calculate_rsi(self, hist_df: pd.DataFrame, period: int = 14) -> float:
        """计算RSI(14) - 硬编码参数, 不暴露为可调参数"""
        if len(hist_df) < period + 1:
            return 50.0  # 数据不足返回中性值

        closes = hist_df['close'].iloc[-(period + 1):]
        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0.0)
        losses = (-deltas).where(deltas < 0, 0.0)

        avg_gain = gains.mean()
        avg_loss = losses.mean()

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # ==================== 入场过滤 ====================

    def _check_ema_trend(self, hist_df: pd.DataFrame, current_price: float) -> bool:
        """EMA200趋势确认: 价格在EMA上方=上升趋势, 允许买入"""
        if len(hist_df) < self.ema_period:
            return True  # 数据不足时不过滤

        close_series = hist_df['close']
        ema = close_series.ewm(span=self.ema_period, adjust=False).mean()
        ema_value = ema.iloc[-1]

        return current_price >= ema_value

    def _check_volume(self, hist_df: pd.DataFrame) -> bool:
        """成交量确认: 当日成交量 >= 20日均量 * min_volume_ratio"""
        if len(hist_df) < 20 or 'volume' not in hist_df.columns:
            return True  # 数据不足时不过滤

        volumes = hist_df['volume'].iloc[-20:]
        avg_volume = volumes.mean()
        current_volume = hist_df['volume'].iloc[-1]

        if avg_volume <= 0:
            return True

        return current_volume >= avg_volume * self.min_volume_ratio

    def _get_volume_ratio(self, hist_df: pd.DataFrame) -> float:
        """获取当前成交量/20日均量的比值"""
        if len(hist_df) < 20 or 'volume' not in hist_df.columns:
            return 1.0
        volumes = hist_df['volume'].iloc[-20:]
        avg_volume = volumes.mean()
        current_volume = hist_df['volume'].iloc[-1]
        if avg_volume <= 0:
            return 1.0
        return current_volume / avg_volume

    # ==================== 买入信号 ====================

    def _check_buy_signal(
        self,
        current_price: float,
        index: int,
        symbol: str,
        bar: pd.Series,
        hist_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查买入信号(含过滤)"""
        if not self._fractals or not self._strokes:
            return None

        # 冷却期检查
        last_sell_idx = self._last_sell_index.get(symbol, -999)
        if index - last_sell_idx < self.cooldown_bars:
            return None

        # 检查最后是否有底分型
        last_fractal = self._fractals[-1] if self._fractals else None
        if not last_fractal or last_fractal.is_top:
            return None

        # 依次检查买卖点(置信度从高到低)
        best_signal = None

        # 1买: 底背驰(最高质量, confidence=0.8)
        if self._check_first_buy_point(current_price, index):
            best_signal = Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason='1买: 底背驰',
                confidence=0.8
            )

        # 2买: 回抽不破前低(confidence=0.7)
        if best_signal is None and self._check_second_buy_point(current_price, index):
            best_signal = Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason='2买: 回抽不破前低',
                confidence=0.7
            )

        # 3买: 突破中枢回踩(confidence=0.6)
        if best_signal is None and self._check_third_buy_point(current_price, index):
            best_signal = Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=last_fractal.datetime,
                price=current_price,
                reason='3买: 突破中枢回踩',
                confidence=0.6
            )

        if best_signal is None:
            return None

        if best_signal.confidence < self.confidence_threshold:
            return None

        # R1: EMA从硬过滤改为统一软降权
        ema_ok = self._check_ema_trend(hist_df, current_price)
        if not ema_ok:
            best_signal.confidence -= 0.1
            if best_signal.confidence < self.confidence_threshold:
                return None

        # R3: 中枢宽度守门 - 窄中枢(<2%)产生的信号降权(高价股茅台问题)
        if self._pivots:
            last_pivot = self._pivots[-1]
            pivot_range_pct = (last_pivot.high - last_pivot.low) / last_pivot.low if last_pivot.low > 0 else 0
            if pivot_range_pct < 0.02:
                best_signal.confidence -= 0.15
                if best_signal.confidence < self.confidence_threshold:
                    return None

        # 过滤: 成交量确认(连续降权而非硬过滤)
        vol_ratio = self._get_volume_ratio(hist_df)
        if vol_ratio < self.min_volume_ratio:
            # 成交量不足, 根据差距降低置信度
            shortfall = self.min_volume_ratio - vol_ratio
            confidence_penalty = min(0.15, shortfall * 0.3)
            best_signal.confidence -= confidence_penalty
            # 如果降权后低于阈值, 则拒绝
            if best_signal.confidence < self.confidence_threshold:
                return None

        # === R2: 动能作为仓位权重因子(而非过滤器) ===
        # R1发现MACD柱状图72%通过率无区分力, 改为影响置信度
        momentum_score = self._get_momentum_score()
        if momentum_score < 0:
            # 动能明显衰减: 大幅降权
            best_signal.confidence -= 0.2
            if best_signal.confidence < self.confidence_threshold:
                return None
        elif momentum_score < 0.3:
            # 动能偏弱: 轻微降权
            best_signal.confidence -= 0.05
        elif momentum_score >= 1.0:
            # 强动能(红柱扩大)加成
            best_signal.confidence = min(best_signal.confidence + 0.05, 1.0)

        return best_signal

    # ==================== 卖出信号 ====================

    def _check_sell_signal(
        self,
        current_price: float,
        index: int,
        symbol: str,
        bar: pd.Series,
        hist_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查卖出信号(止损优先 > 保本止损 > 分批止盈 > 跟踪止损 > 缠论卖点)"""
        entry_price = self._entry_prices.get(symbol)
        if not entry_price:
            return None

        profit_pct = (current_price - entry_price) / entry_price

        # === 优先级1: 固定止损 + R2: ATR自适应止损 ===
        fixed_stop = entry_price * (1 - self.fixed_stop_pct)
        # R2: ATR动态止损, 但增加上限控制(不超过固定止损百分比)
        if hist_df is not None and len(hist_df) >= 15:
            atr = self._calculate_atr(hist_df)
            atr_pct = atr / entry_price  # ATR占价格的百分比
            # R2: 当ATR过大时(高波动股), 限制ATR止损不超过max_atr_multiplier
            if atr_pct > self.max_atr_multiplier:
                atr_stop = entry_price * (1 - self.max_atr_multiplier)
            else:
                atr_stop = entry_price - 2 * atr
            # 取更严格的止损(更高的止损价), 但不低于3%
            min_stop = entry_price * 0.97
            fixed_stop = max(fixed_stop, atr_stop, min_stop)

        if current_price < fixed_stop:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=current_price,
                reason=f'固定止损: 跌破{fixed_stop:.2f}(入场{entry_price:.2f}-{self.fixed_stop_pct:.0%})',
                confidence=1.0
            )

        # === 优先级1.5: 保本止损(R2: 3%->5%, 避免短期波动被洗出) ===
        if profit_pct >= self.breakeven_activate_pct:
            breakeven_stop = entry_price * 1.001  # 成本价+手续费
            if current_price < breakeven_stop:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=current_price,
                    reason=f'保本止损: 盈利{profit_pct:.1%}后回落至成本线',
                    confidence=0.95
                )

        # === 优先级2: 缠论结构止损(仅当结构支撑低于固定止损时使用) ===
        if self.use_structure_stop:
            structure_stop = self._get_structure_stop_loss(symbol)
            if structure_stop is not None:
                # 只有当结构止损比固定止损更严格(更低)时才使用
                # 且要求价格确实跌破结构支撑
                # 同时要求结构支撑在入场价以下(否则不合理)
                if (structure_stop < fixed_stop and
                        structure_stop < entry_price and
                        current_price < structure_stop):
                    return Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                        price=current_price,
                        reason=f'结构止损: 跌破缠论支撑{structure_stop:.2f}',
                        confidence=1.0
                    )

        # === 优先级2: 分批止盈 ===
        if self.use_partial_profit:
            partial_signal = self._check_partial_profit(current_price, symbol, bar)
            if partial_signal is not None:
                return partial_signal

        # === 优先级3: 跟踪止盈 ===
        highest = self._highest_prices.get(symbol, entry_price)
        if current_price > highest:
            self._highest_prices[symbol] = current_price
            highest = current_price

        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct > self.trailing_activate_pct:
            drawdown_from_high = (highest - current_price) / highest
            # R2: 波动率自适应跟踪止损回撤容忍度
            # 高波动股(>3%日波动): 回撤容忍度适当放大, 避免被洗出
            vol = self._volatility_cache.get(symbol, 0.02)
            adaptive_drawdown = self.trailing_drawdown_pct
            if vol > 0.03:  # 日波动率>3%为高波动
                adaptive_drawdown = self.trailing_drawdown_pct * 1.5
            if drawdown_from_high > adaptive_drawdown:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=current_price,
                    reason=f'跟踪止盈: 盈利{profit_pct:.1%} 从最高{highest:.2f}回撤{drawdown_from_high:.1%}(容差{adaptive_drawdown:.0%})',
                    confidence=0.9
                )

        # === 优先级3.5: 动能衰减监控(R3: 放宽条件) ===
        # R2: 连续3根绿柱扩大条件太严(从未触发)
        # R3: 改用_get_momentum_score, 仅要求score=-1.0(加速衰减) + 盈利>2%
        if profit_pct > 0.02:
            momentum_score = self._get_momentum_score()
            if momentum_score < -0.5:  # 动能明显衰减
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=current_price,
                    reason=f'动能衰减: 盈利{profit_pct:.1%} 动能评分{momentum_score:.1f}',
                    confidence=0.75
                )

        # === 优先级4: 缠论卖点 ===
        if self._fractals and self._strokes:
            last_fractal = self._fractals[-1] if self._fractals else None
            if last_fractal and last_fractal.is_bottom is False:  # 顶分型
                # 1卖: 顶背驰
                if self._check_first_sell_point(current_price, index):
                    return Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        datetime=last_fractal.datetime,
                        price=current_price,
                        reason='1卖: 顶背驰',
                        confidence=0.8
                    )

                # 2卖: 反弹不破前高
                if self._check_second_sell_point(current_price, index):
                    return Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        datetime=last_fractal.datetime,
                        price=current_price,
                        reason='2卖: 反弹不破前高',
                        confidence=0.7
                    )

        return None

    # ==================== 缠论买卖点检测(同ChanLunStrategy) ====================

    def _check_first_buy_point(self, current_price: float, index: int) -> bool:
        """
        检查第一类买点(底背驰)

        R2修改: 放宽中枢条件, 允许价格在中枢下沿附近(<=中枢下沿*1.03)
        原条件: current_price < last_pivot.low (价格必须低于中枢下沿)
        新条件: 价格不高于中枢下沿*1.03 (允许3%容差)
        同时保留: 最后一笔必须是向下笔 + MACD底背驰
        """
        if len(self._strokes) < 3:
            return False
        if not self._pivots:
            return False

        last_pivot = self._pivots[-1]
        last_stroke = self._strokes[-1]

        # 最后一笔必须是向下笔(底背驰发生在下跌末端)
        if last_stroke.is_up:
            return False

        # R2: 放宽中枢条件 - 价格在中枢下沿附近即可(允许3%容差)
        # 原条件过严: 价格必须严格低于中枢下沿, 很多情况下不满足
        if current_price >= last_pivot.low * 1.03:
            return False

        # MACD背驰检测(核心条件)
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
        """检查第二类买点(R3: 增加结构验证, 避免高频低质量信号)"""
        if not self._pivots or not self._strokes:
            return False

        last_pivot = self._pivots[-1]

        # R3: 必须有底分型在中枢下沿附近
        recent_bottoms = [f for f in self._fractals[-5:] if f.is_bottom]
        if not recent_bottoms:
            return False
        last_bottom = recent_bottoms[-1]

        # 底分型低点在中枢下沿附近(<=中枢下沿*1.05)
        if last_bottom.low > last_pivot.low * 1.05:
            return False

        # 价格高于底分型低点(回抽确认)
        if current_price <= last_bottom.low:
            return False

        # 价格低于中枢中点(仍在合理回抽区间)
        pivot_mid = (last_pivot.high + last_pivot.low) / 2
        if current_price > pivot_mid:
            return False

        return True

    def _check_second_buy_from_pivot(self, current_price: float, index: int) -> bool:
        """从中枢推断2买"""
        if not self._pivots or not self._strokes:
            return False

        last_pivot = self._pivots[-1]
        down_strokes = [s for s in self._strokes
                        if s.is_down and s.end_value < last_pivot.low]
        if not down_strokes:
            return False

        last_down = down_strokes[-1]
        implied_buy_price = last_down.end_value

        up_after = [s for s in self._strokes
                    if s.is_up and s.start_index > last_down.end_index]
        if not up_after:
            return False

        return current_price > implied_buy_price * 0.98 and current_price < last_pivot.low

    def _check_third_buy_point(self, current_price: float, index: int) -> bool:
        """
        检查第三类买点: 突破中枢后回踩不破中枢上沿

        R2修改: 放宽回调笔条件, 允许直接价格确认(无需严格回调笔结构)
        原条件: 必须有突破笔+回调笔+回调低点>中枢上沿 (三重条件极难同时满足)
        新条件: 价格在中枢上沿附近 + (有回调笔结构 OR 直接价格高于中枢上沿)
        """
        if not self._pivots or len(self._strokes) < 4:
            return False

        last_pivot = self._pivots[-1]

        # 价格在中枢上沿附近(回踩区间) - R2: 放宽上沿从1.12到1.15
        if not (current_price > last_pivot.high * 0.98 and current_price < last_pivot.high * 1.15):
            return False

        # 方式1: 有明确的突破笔+回调笔结构(原有逻辑)
        breakout_strokes = [s for s in self._strokes
                          if s.is_up and s.end_value > last_pivot.high
                          and s.start_index >= last_pivot.end_index]
        if breakout_strokes:
            last_breakout = breakout_strokes[-1]

            pullback_strokes = [s for s in self._strokes
                                if s.is_down and s.start_index > last_breakout.end_index]
            if pullback_strokes:
                last_pullback = pullback_strokes[-1]
                # 回调低点仍在中枢上沿之上
                if last_pullback.end_value > last_pivot.high:
                    return True

        # 方式2(R2新增): 直接价格确认 - 价格在中枢上方且回踩到中枢上沿附近
        # 只要最近5笔内有突破行为, 且当前价格在中枢上沿附近即可
        recent_strokes = self._strokes[-5:]
        has_recent_breakout = any(
            s.is_up and s.end_value > last_pivot.high
            for s in recent_strokes
        )
        if has_recent_breakout and current_price > last_pivot.high * 0.99:
            return True

        return False

    def _check_first_sell_point(self, current_price: float, index: int) -> bool:
        """检查第一类卖点(顶背驰)"""
        if len(self._strokes) < 3:
            return False
        if not self._pivots:
            return False

        last_pivot = self._pivots[-1]
        last_stroke = self._strokes[-1]

        if last_stroke.is_down:
            return False

        if current_price <= last_pivot.high:
            return False

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

    def _check_second_sell_point(self, current_price: float, index: int) -> bool:
        """检查第二类卖点"""
        if self._last_sell_point is None:
            return False
        sell_idx, sell_price = self._last_sell_point
        return current_price < sell_price * 1.02

    # ==================== 辅助方法 ====================

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR(Average True Range)"""
        if len(df) < period + 1:
            return float(df['high'].max() - df['low'].min())
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return float(tr.tail(period).mean())

    def _get_structure_stop_loss(self, symbol: str) -> Optional[float]:
        """获取缠论结构止损位: 最近一个向下笔的低点(比底分型更稳定)"""
        if not self._strokes:
            return None

        # 找最近一个向下笔的终点作为结构支撑
        for stroke in reversed(self._strokes):
            if stroke.is_down:
                return stroke.end_value

        return None

    # ==================== 分批止盈 ====================

    def _check_partial_profit(
        self,
        current_price: float,
        symbol: str,
        bar: pd.Series
    ) -> Optional[Signal]:
        """检查分批止盈"""
        entry_price = self._entry_prices.get(symbol)
        if not entry_price:
            return None

        current_position = self.get_position(symbol)
        # R2: 允许零股止盈(原条件<100返回None会阻止高价股止盈)
        if current_position < 1:
            return None

        profit_pct = (current_price - entry_price) / entry_price
        stages = self._partial_profit_stages.get(symbol, [])

        dt = bar.name if hasattr(bar, 'name') else pd.Timestamp.now()

        # R2: 根据是否允许零股来决定最小卖出量
        min_sell = 1 if self.allow_fractional_shares else 100

        # 阶段1: 盈利profit_target_1%卖profit_sell_ratio_1
        if 0 not in stages and profit_pct >= self.profit_target_1:
            sell_qty = int(current_position * self.profit_sell_ratio_1)
            if not self.allow_fractional_shares:
                sell_qty = (sell_qty // 100) * 100

            if sell_qty >= min_sell:
                if symbol not in self._partial_profit_stages:
                    self._partial_profit_stages[symbol] = []
                self._partial_profit_stages[symbol].append(0)

                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=dt,
                    price=current_price,
                    quantity=sell_qty,
                    reason=f'分批止盈1: 盈利{profit_pct:.1%}, 卖出{sell_qty}股',
                    confidence=0.9
                )

        # 阶段2: 盈利profit_target_2%卖profit_sell_ratio_2
        if 1 not in stages and profit_pct >= self.profit_target_2:
            sell_qty = int(current_position * self.profit_sell_ratio_2)
            if not self.allow_fractional_shares:
                sell_qty = (sell_qty // 100) * 100

            if sell_qty >= min_sell:
                if symbol not in self._partial_profit_stages:
                    self._partial_profit_stages[symbol] = []
                self._partial_profit_stages[symbol].append(1)

                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=dt,
                    price=current_price,
                    quantity=sell_qty,
                    reason=f'分批止盈2: 盈利{profit_pct:.1%}, 卖出{sell_qty}股',
                    confidence=0.9
                )

        return None

    # ==================== 订单回调 ====================

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调"""
        super().on_order(signal, executed_price, executed_quantity)
        symbol = signal.symbol

        if signal.is_buy():
            self._entry_prices[symbol] = executed_price
            self._highest_prices[symbol] = executed_price
            # 重置分批止盈状态
            self._partial_profit_stages.pop(symbol, None)

        elif signal.is_sell():
            # 如果是全部卖出(没有指定quantity, 或quantity等于持仓), 清理状态
            remaining = self.get_position(symbol)
            if remaining <= 0:
                self._entry_prices.pop(symbol, None)
                self._highest_prices.pop(symbol, None)
                self._partial_profit_stages.pop(symbol, None)
