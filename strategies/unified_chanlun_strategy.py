"""
统一缠论策略

整合信号质量评分、多周期融合、过滤器链、统一出场管理的最终策略。

流水线架构：
    数据 → 多周期分析 → 评分 → 过滤 → 信号融合 → 执行

配置驱动，通过 UnifiedStrategyConfig 切换模式：
- single_daily(): 等价现有 IntegratedChanLunStrategy
- multi_tf(): 启用周线+30分钟多周期
- conservative(): 保守模式
- aggressive(): 激进模式
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from backtest.strategy import Strategy, Signal, SignalType
from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.segment import SegmentGenerator, Segment
from core.pivot import PivotDetector, Pivot, PivotLevel
from core.buy_sell_points import BuySellPointDetector, BuySellPoint
from core.multi_tf_analyzer import MultiTimeFrameAnalyzer, TimeFrameAnalysis
from core.signal_resolver import SignalResolver, ResolvedSignal
from data.multi_tf_manager import MultiTimeFrameManager
from indicator.macd import MACD
from indicator.enhanced_divergence import EnhancedDivergenceDetector, DivergenceResult
from strategies.scoring import AdaptiveSignalScorer, ScoringFactors, ScoringConfig, MarketRegimeDetector
from strategies.filters import CompositeFilter, VolumeFilter, RegimeFilter, CooldownFilter, TrendAlignmentFilter
from strategies.unified_config import UnifiedStrategyConfig
from strategies.unified_exit_manager import UnifiedExitManager, ExitSignal


def _calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> float:
    """计算成交量均线"""
    if len(df) < period:
        return float(df['volume'].mean())
    return float(df['volume'].tail(period).mean())


class UnifiedChanLunStrategy(Strategy):
    """
    统一缠论策略

    使用方法：
        # 默认模式
        strategy = UnifiedChanLunStrategy()

        # 多周期模式
        config = UnifiedStrategyConfig.multi_tf()
        strategy = UnifiedChanLunStrategy(config)

        # 保守模式
        config = UnifiedStrategyConfig.conservative()
        strategy = UnifiedChanLunStrategy(config)
    """

    def __init__(self, config: Optional[UnifiedStrategyConfig] = None):
        name = (config.name if config else '统一缠论策略')
        super().__init__(name)

        self.config = config or UnifiedStrategyConfig()

        # 评分器
        self.scorer = AdaptiveSignalScorer()

        # 过滤器链
        self.filter_chain = self._build_filters()

        # 出场管理器
        self.exit_manager = UnifiedExitManager(self.config.exit)

        # 多周期管理器（每个symbol一个）
        self._tf_managers: Dict[str, MultiTimeFrameManager] = {}

        # 多周期分析器缓存
        self._analyzers: Dict[str, MultiTimeFrameAnalyzer] = {}
        self._analyzer_bar_count: Dict[str, int] = {}

        # 日线缠论分析缓存
        self._fractals: Dict[str, List[Fractal]] = {}
        self._strokes: Dict[str, List[Stroke]] = {}
        self._segments: Dict[str, List[Segment]] = {}
        self._pivots: Dict[str, List[Pivot]] = {}
        self._macd: Dict[str, MACD] = {}
        self._detector: Dict[str, BuySellPointDetector] = {}
        self._last_bar_count: Dict[str, int] = {}
        self._kline_len: Dict[str, int] = {}  # 合并后K线长度（用于索引比较）

        # 冷却过滤器引用（需要记录卖出事件）
        self._cooldown_filter: Optional[CooldownFilter] = None

        # 当前bar索引（用于传递给出场管理器）
        self._current_bar_index: Dict[str, int] = {}

        # 评分历史（用于动量/评分变化量）
        self._last_scores: Dict[str, float] = {}

    def _build_filters(self) -> CompositeFilter:
        """构建过滤器链"""
        chain = CompositeFilter()
        fc = self.config.filters

        if fc.use_volume:
            chain.add(VolumeFilter(fc.volume_min_ratio, fc.volume_ma_period))

        if fc.use_regime:
            chain.add(RegimeFilter())

        # Kronos AI 预测确认 (在便宜过滤器之后，冷却之前)
        if fc.use_kronos:
            try:
                from strategies.kronos import KronosConfig, KronosPredictor, KronosFilter
                kronos_cfg = KronosConfig(
                    enabled=True,
                    model_name=fc.kronos_model,
                    pred_len=fc.kronos_pred_len,
                    min_upside_pct=fc.kronos_min_upside,
                    max_downside_pct=fc.kronos_max_downside,
                )
                self._kronos_predictor = KronosPredictor(kronos_cfg)
                if self._kronos_predictor.is_available():
                    chain.add(KronosFilter(
                        predictor=self._kronos_predictor,
                        pred_len=fc.kronos_pred_len,
                        min_upside_pct=fc.kronos_min_upside,
                        max_downside_pct=fc.kronos_max_downside,
                    ))
                    logger.info(f"Kronos 过滤器已启用 (model={fc.kronos_model})")
                else:
                    logger.warning("Kronos 已启用但依赖不可用，跳过")
            except Exception as e:
                logger.warning(f"Kronos 过滤器加载失败: {e}")

        if fc.use_cooldown:
            self._cooldown_filter = CooldownFilter(fc.cooldown_bars)
            chain.add(self._cooldown_filter)

        if fc.use_trend_alignment:
            chain.add(TrendAlignmentFilter(
                ma_period=fc.trend_ma_period,
                require_macd_turn=fc.trend_require_macd_turn,
                strict_mode=fc.trend_strict_mode,
            ))

        return chain

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化策略"""
        super().initialize(capital, symbols)
        logger.info(
            f"初始化{self.name}: 资金{capital:,}元, "
            f"模式={self.config.timeframes.use_weekly and '周线' or ''}"
            f"{'+' if self.config.timeframes.use_min30 else ''}"
            f"{'30分' if self.config.timeframes.use_min30 else ''}"
            f"{'日线' if not self.config.timeframes.use_weekly and not self.config.timeframes.use_min30 else ''}"
        )

    def reset(self) -> None:
        """重置策略"""
        super().reset()
        self._tf_managers.clear()
        self._analyzers.clear()
        self._analyzer_bar_count.clear()
        self._fractals.clear()
        self._strokes.clear()
        self._segments.clear()
        self._pivots.clear()
        self._macd.clear()
        self._detector.clear()
        self._last_bar_count.clear()
        self._kline_len.clear()
        self._last_scores.clear()

    def set_multi_tf_data(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        min30_df: Optional[pd.DataFrame] = None,
    ):
        """
        预加载多周期数据（回测时使用）

        Args:
            symbol: 股票代码
            daily_df: 日线数据
            min30_df: 30分钟数据（可选）
        """
        mgr = MultiTimeFrameManager(symbol, daily_df, min30_df)
        self._tf_managers[symbol] = mgr
        logger.debug(f"已加载{symbol}多周期数据: 日线={len(daily_df)}, 30分={len(min30_df) if min30_df is not None else 0}")

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        hist_df = context.get('data', {}).get(symbol) if isinstance(context.get('data'), dict) else context.get('data', {}).get(symbol)
        if hist_df is None or len(hist_df) < 60:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 记录当前bar索引
        self._current_bar_index[symbol] = index

        # 更新日线缠论分析（有变化时才更新）
        if len(hist_df) != self._last_bar_count.get(symbol, 0):
            self._update_analysis(symbol, hist_df)
            self._last_bar_count[symbol] = len(hist_df)

        # 已有持仓：检查出场
        if current_position > 0:
            return self._check_exit(symbol, current_price, bar, hist_df, index)

        # 无持仓：检查入场
        return self._check_entry(symbol, current_price, bar, hist_df, index, context)

    def _update_analysis(self, symbol: str, df: pd.DataFrame) -> None:
        """更新缠论分析"""
        kline = KLine.from_dataframe(df, strict_mode=False)
        self._kline_len[symbol] = len(kline)

        detector = FractalDetector(kline, confirm_required=False)
        self._fractals[symbol] = detector.get_fractals()

        stroke_gen = StrokeGenerator(kline, self._fractals[symbol])
        self._strokes[symbol] = stroke_gen.get_strokes()

        seg_gen = SegmentGenerator(kline, self._strokes[symbol])
        self._segments[symbol] = seg_gen.get_segments()

        pivot_detector = PivotDetector(kline, self._strokes[symbol])
        self._pivots[symbol] = pivot_detector.get_pivots()

        self._macd[symbol] = MACD(df['close'])

        self._detector[symbol] = BuySellPointDetector(
            fractals=self._fractals[symbol],
            strokes=self._strokes[symbol],
            segments=self._segments[symbol],
            pivots=self._pivots[symbol],
            macd=self._macd[symbol],
            divergence_threshold=self.config.scoring.divergence_threshold,
        )

    def _get_weekly_df(self, symbol: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """获取周线数据"""
        if not self.config.timeframes.use_weekly:
            return None

        # 优先使用预加载的多周期管理器
        mgr = self._tf_managers.get(symbol)
        if mgr and mgr.weekly_df is not None:
            return mgr.weekly_df

        # 回退：从日线resample
        return self._resample_weekly(df)

    def _get_min30_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取30分钟数据"""
        if not self.config.timeframes.use_min30:
            return None

        mgr = self._tf_managers.get(symbol)
        if mgr:
            return mgr.min30_df
        return None

    def _check_entry(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame,
        index: int,
        context: Dict[str, Any],
    ) -> Optional[Signal]:
        """检查入场条件"""
        strokes = self._strokes.get(symbol, [])
        pivots = self._pivots.get(symbol, [])
        fractals = self._fractals.get(symbol, [])
        detector = self._detector.get(symbol)
        macd = self._macd.get(symbol)

        if not strokes or not pivots:
            return None

        # 最后一个分型必须是底分型
        if not fractals or not fractals[-1].is_bottom:
            return None

        # === 方案A：多周期融合模式 ===
        weekly_df = self._get_weekly_df(symbol, df)
        min30_df = self._get_min30_df(symbol)

        if weekly_df is not None and len(weekly_df) >= 30:
            signal = self._multi_tf_entry(
                symbol, price, bar, df, index,
                weekly_df, min30_df, detector, context,
            )
            if signal is not None:
                return signal

        # === 方案B：单日线模式（回退） ===
        return self._single_tf_entry(
            symbol, price, bar, df, index, detector, macd, context,
        )

    def _multi_tf_entry(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame,
        index: int,
        weekly_df: pd.DataFrame,
        min30_df: Optional[pd.DataFrame],
        detector: BuySellPointDetector,
        context: Dict[str, Any],
    ) -> Optional[Signal]:
        """多周期融合入场"""
        try:
            analyzer = MultiTimeFrameAnalyzer(
                weekly_df=weekly_df,
                daily_df=df,
                min30_df=min30_df,
                min_bars_daily=self.config.timeframes.min_bars_daily,
                min_bars_30m=self.config.timeframes.min_bars_30m,
                divergence_threshold=self.config.scoring.divergence_threshold,
            )

            resolver = SignalResolver(
                analyzer=analyzer,
                current_price=price,
                current_position=self.get_position(symbol),
                min_daily_confidence=self.config.scoring.min_daily_confidence,
                confirmed_ratio=self.config.position.confirmed_ratio,
                unconfirmed_ratio=self.config.position.unconfirmed_ratio,
            )

            resolved = resolver.resolve_buy()
            if resolved is None or resolved.action != 'buy':
                return None

            # 过滤器链
            if resolved.daily_signal:
                passed, reason = self.filter_chain.should_enter(
                    resolved.daily_signal, {
                        'df': df,
                        'price': price,
                        'bar_index': index,
                        'symbol': symbol,
                        'regime_info': analyzer.daily.trend_status if analyzer.daily else None,
                    }
                )
                if not passed:
                    logger.debug(f"{symbol} 信号被过滤: {reason}")
                    return None

            # 计算仓位
            position_ratio = resolved.position_ratio
            cash = self.get_cash()
            target_amount = cash * self.config.position.max_position_pct * position_ratio
            quantity = int(target_amount / price / self.config.position.min_unit) * self.config.position.min_unit
            if quantity <= 0:
                return None

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                price=price,
                quantity=quantity,
                reason=resolved.reason,
                confidence=resolved.confidence,
                metadata={
                    'buy_point_type': resolved.entry_type,
                    'stop_loss': resolved.stop_loss,
                    'weekly_bias': resolved.weekly_bias,
                    'min30_confirmed': resolved.min30_confirmed,
                    'position_ratio': position_ratio,
                }
            )

        except Exception as e:
            logger.debug(f"{symbol} 多周期入场检测失败: {e}")
            return None

    def _single_tf_entry(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame,
        index: int,
        detector: BuySellPointDetector,
        macd: MACD,
        context: Dict[str, Any],
    ) -> Optional[Signal]:
        """单日线入场（回退模式，兼容现有逻辑）"""
        # 批量买点检测
        buy_point = None
        if detector is not None:
            detector.detect_all()
            buys = detector._buy_points
            # 使用合并后K线索引，避免与原始DataFrame索引不一致
            current_kline_idx = self._kline_len.get(symbol, len(df)) - 1

            if buys:
                recent_buys = [b for b in buys if abs(b.index - current_kline_idx) <= 5]
                if recent_buys:
                    best = max(recent_buys, key=lambda b: b.confidence)
                    if best.confidence >= self.config.scoring.min_daily_confidence:
                        buy_point = best

        # 简化结构检测回退
        if buy_point is None:
            buy_point = self._detect_structural_buy(symbol, price)

        if buy_point is None:
            return None

        # 增强评分
        score, score_reason = self._score_buy_signal(symbol, buy_point, bar, df, macd)
        if score < self.config.scoring.min_buy_score:
            return None

        # 过滤器链
        passed, reason = self.filter_chain.should_enter(
            buy_point, {
                'df': df,
                'price': price,
                'bar_index': index,
                'symbol': symbol,
            }
        )
        if not passed:
            logger.debug(f"{symbol} 信号被过滤: {reason}")
            return None

        # 仓位计算
        cash = self.get_cash()
        target_amount = cash * self.config.position.max_position_pct
        quantity = int(target_amount / price / self.config.position.min_unit) * self.config.position.min_unit
        if quantity <= 0:
            return None

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            quantity=quantity,
            reason=f'{buy_point.reason} | 评分={score:.2f}({score_reason})',
            confidence=score,
            metadata={
                'buy_point_type': buy_point.point_type,
                'stop_loss': buy_point.stop_loss,
                'divergence_ratio': getattr(buy_point, 'divergence_ratio', 0.0),
                'score': score,
            }
        )

    def _score_buy_signal(
        self,
        symbol: str,
        buy_point: BuySellPoint,
        bar: pd.Series,
        df: pd.DataFrame,
        macd: MACD,
    ) -> tuple:
        """使用自适应评分器评估买点"""
        strokes = self._strokes.get(symbol, [])
        pivots = self._pivots.get(symbol, [])

        # 增强背离检测
        div_result = None
        if strokes and macd:
            direction = 'down' if buy_point.point_type in ('1buy', '2buy', 'quasi2buy') else 'up'
            try:
                ed = EnhancedDivergenceDetector(macd, strokes)
                div_result = ed.detect_trend_divergence(direction)
            except Exception:
                pass

        # 中枢质量
        pivot_quality = 0.0
        related_pivot = buy_point.related_pivot if hasattr(buy_point, 'related_pivot') else None
        if related_pivot:
            pivot_quality = related_pivot.quality_score
        elif pivots:
            pivot_quality = pivots[-1].quality_score

        # 关键笔强度
        stroke_strength = 0.0
        if strokes:
            stroke_strength = strokes[-1].strength_score

        # 量价
        vol_ratio = 1.0
        if 'volume' in bar.index and bar['volume'] > 0:
            vol_ma = _calculate_volume_ma(df)
            if vol_ma > 0:
                vol_ratio = bar['volume'] / vol_ma

        # 市场状态
        regime_info = None
        if self.config.filters.use_regime:
            try:
                regime_detector = MarketRegimeDetector(df)
                regime_info = regime_detector.detect()
            except Exception:
                pass

        # RSI计算
        rsi = 50.0
        if len(df) >= 15:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 and not pd.isna(rsi_series.iloc[-1]) else 50.0

        # 评分变化量（动量）：与上次评分的差值
        score_delta = 0.0
        last_score = self._last_scores.get(symbol)
        # delta在下次调用时才能计算，首次为0

        factors = ScoringFactors(
            divergence_result=div_result,
            divergence_ratio=getattr(buy_point, 'divergence_ratio', 0.0),
            pivot_quality=pivot_quality,
            stroke_strength=stroke_strength,
            strokes_in_pivot=len(related_pivot.strokes) if related_pivot and hasattr(related_pivot, 'strokes') else 0,
            volume_ratio=vol_ratio,
            regime_info=regime_info,
            point_type=buy_point.point_type,
            rsi=rsi,
            score_delta=score_delta,
        )

        score, reason = self.scorer.score_buy_signal(factors)

        # 更新评分历史（下次调用时可计算delta）
        self._last_scores[symbol] = score

        return (score, reason)

    def _detect_structural_buy(
        self, symbol: str, price: float
    ) -> Optional[BuySellPoint]:
        """简化缠论结构买点检测（复用现有逻辑）"""
        strokes = self._strokes.get(symbol, [])
        pivots = self._pivots.get(symbol, [])
        fractals = self._fractals.get(symbol, [])
        macd = self._macd.get(symbol)

        if not strokes or not pivots or not fractals:
            return None

        # 1买
        result = self._check_1buy(strokes, pivots, macd)
        if result:
            return result

        # 2买
        result = self._check_2buy(strokes, pivots)
        if result:
            return result

        # 3买
        result = self._check_3buy(strokes, pivots, price)
        if result:
            return result

        return None

    def _check_1buy(self, strokes, pivots, macd) -> Optional[BuySellPoint]:
        """第一类买点：底背驰"""
        if len(strokes) < 3 or not pivots:
            return None
        down = [s for s in strokes if s.is_down]
        if len(down) < 2:
            return None

        last_down = down[-1]
        prev_down = down[-2]
        last_pivot = pivots[-1]

        if last_down != strokes[-1]:
            return None
        if last_down.end_value >= last_pivot.low:
            return None

        div_ratio = 0.0
        if macd:
            has_div, div_ratio = macd.check_divergence(
                last_down.start_index, last_down.end_index, 'down',
                prev_start=prev_down.start_index, prev_end=prev_down.end_index
            )
            if not has_div:
                return None
        else:
            drop1 = abs(prev_down.price_change_pct)
            drop2 = abs(last_down.price_change_pct)
            if drop2 >= drop1 * 0.7:
                return None
            div_ratio = 1 - drop2 / drop1 if drop1 > 0 else 0

        confidence = 0.6 + min(div_ratio * 0.4, 0.4)
        return BuySellPoint(
            point_type='1buy',
            price=last_down.end_value,
            index=last_down.end_index,
            related_pivot=last_pivot,
            related_strokes=[prev_down, last_down],
            divergence_ratio=div_ratio,
            confidence=confidence,
            stop_loss=last_down.low * 0.99,
            reason=f'1买: 底背驰(强度{div_ratio:.2f})',
        )

    def _check_2buy(self, strokes, pivots, max_lookback: int = 80) -> Optional[BuySellPoint]:
        """
        第二类买点：回调不破前低

        改进：
        1. 优先使用中枢形成之后的笔作为推断1买（更准确）
        2. 回溯范围限制避免使用远古低点
        3. 2买价格必须在中枢附近（不能偏离太远）
        """
        if not pivots or not strokes:
            return None
        last_pivot = pivots[-1]
        last_bar_idx = strokes[-1].end_index

        # 优先：中枢形成之后的向下笔（离开中枢的笔）作为推断1买
        post_pivot_down = [
            s for s in strokes
            if s.is_down
            and s.start_index >= last_pivot.end_index
            and s.end_value < last_pivot.low
        ]

        # 回退：中枢之前的向下笔，但限制在中枢附近
        if not post_pivot_down:
            post_pivot_down = [
                s for s in strokes
                if s.is_down
                and s.end_value < last_pivot.low
                and (last_bar_idx - s.end_index) <= max_lookback
            ]

        if not post_pivot_down:
            return None

        last_down = post_pivot_down[-1]
        implied_1buy = last_down.end_value

        # 1买之后必须有向上笔
        up_after = [s for s in strokes if s.is_up and s.start_index > last_down.end_index]
        if not up_after:
            return None

        # 向上笔之后必须有回调笔
        pullback = [s for s in strokes if s.is_down and s.start_index > up_after[-1].end_index]
        if not pullback:
            return None

        last_pullback = pullback[-1]

        # 回调不破推断的1买低点
        if last_pullback.end_value <= implied_1buy:
            return None

        # 最后一笔必须是回调笔
        if last_pullback != strokes[-1]:
            return None

        # 止损设在1买低点
        stop_loss = implied_1buy * 0.99

        # 2买和1买的距离越近越可靠
        pullback_depth = (last_pullback.end_value - implied_1buy) / implied_1buy
        confidence = 0.55 + min(pullback_depth * 2, 0.25)

        return BuySellPoint(
            point_type='2buy',
            price=last_pullback.end_value,
            index=last_pullback.end_index,
            related_pivot=last_pivot,
            confidence=confidence,
            stop_loss=stop_loss,
            reason=f'2买: 回调不破推断1买{implied_1buy:.2f}(距{last_bar_idx - last_down.end_index}根)',
        )

    def _check_3buy(self, strokes, pivots, price) -> Optional[BuySellPoint]:
        """第三类买点：突破中枢回踩"""
        if not pivots:
            return None
        last_pivot = pivots[-1]

        if not (price > last_pivot.high * 0.99 and price < last_pivot.high * 1.08):
            return None

        breakout = [s for s in strokes if s.is_up and s.end_value > last_pivot.high and s.start_index >= last_pivot.end_index]
        if not breakout:
            return None

        pullback = [s for s in strokes if s.is_down and s.start_index > breakout[-1].end_index]
        if not pullback:
            return None

        last_pb = pullback[-1]
        if last_pb.end_value <= last_pivot.high:
            return None
        if last_pb != strokes[-1]:
            return None

        margin = (last_pb.end_value - last_pivot.high) / last_pivot.high
        return BuySellPoint(
            point_type='3buy',
            price=last_pb.end_value,
            index=last_pb.end_index,
            related_pivot=last_pivot,
            confidence=0.5 + min(margin * 10, 0.3),
            stop_loss=last_pivot.high * 0.99,
            reason=f'3买: 回踩不破中枢上沿{last_pivot.high:.2f}',
        )

    def _check_exit(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        df: pd.DataFrame,
        index: int,
    ) -> Optional[Signal]:
        """检查出场条件（含动态止盈）"""
        current_qty = self.get_position(symbol)
        if current_qty <= 0:
            return None

        # 获取日线卖点
        sell_signals = self._detect_sell_signals(symbol, price)

        # 获取周线方向（如果有）
        weekly_bias = 'neutral'
        weekly_strength = 0.0
        weekly_df = self._get_weekly_df(symbol, df)
        if weekly_df is not None and len(weekly_df) >= 30:
            try:
                from core.trend_track import TrendTrackDetector
                w_kline = KLine.from_dataframe(weekly_df, strict_mode=False)
                w_fractals = FractalDetector(w_kline, confirm_required=False).get_fractals()
                w_strokes = StrokeGenerator(w_kline, w_fractals).get_strokes()
                w_pivots = PivotDetector(w_kline, w_strokes, level=PivotLevel.WEEK).get_pivots()
                if w_pivots and w_strokes:
                    td = TrendTrackDetector(w_strokes, w_pivots)
                    td.detect()
                    status = td.get_trend_status()
                    if hasattr(status, 'value'):
                        val = status.value
                        if 'down' in val:
                            weekly_bias = 'short'
                            weekly_strength = 0.8 if 'strong' in val else 0.6
            except Exception:
                pass

        # === 动态止盈数据 ===

        # 1) ATR值
        atr_value = 0.0
        try:
            from indicator.atr import ATR
            atr_calc = ATR(df['high'], df['low'], df['close'],
                           period=self.config.exit.atr_period)
            atr_result = atr_calc.get_latest()
            if atr_result:
                atr_value = atr_result.atr
        except Exception:
            pass

        # 2) 趋势状态（日线级别）
        trend_status = 'neutral'
        try:
            strokes = self._strokes.get(symbol, [])
            pivots = self._pivots.get(symbol, [])
            if strokes and pivots:
                from core.trend_track import TrendTrackDetector
                td = TrendTrackDetector(strokes, pivots)
                td.detect()
                ts = td.get_trend_status()
                if hasattr(ts, 'value'):
                    trend_status = ts.value
                elif hasattr(ts, 'name'):
                    trend_status = ts.name
        except Exception:
            pass

        # 3) 结构警告（缠论顶分型+背离）
        structure_warning = 'none'
        try:
            fractals = self._fractals.get(symbol, [])
            if fractals and fractals[-1].is_top:
                # 顶分型出现，检查MACD顶背离
                macd = self._macd.get(symbol)
                up_strokes = [s for s in self._strokes.get(symbol, []) if s.is_up]
                if macd and len(up_strokes) >= 2:
                    last_up, prev_up = up_strokes[-1], up_strokes[-2]
                    has_div, _ = macd.check_divergence(
                        last_up.start_index, last_up.end_index, 'up',
                        prev_start=prev_up.start_index, prev_end=prev_up.end_index,
                    )
                    if has_div:
                        structure_warning = 'danger'  # 顶分型+MACD顶背离
                    else:
                        structure_warning = 'caution'  # 顶分型但无背离
        except Exception:
            pass

        exit_signal = self.exit_manager.check_exit(
            symbol=symbol,
            price=price,
            current_qty=current_qty,
            bar_index=index,
            min_unit=self.config.position.min_unit,
            sell_signals=sell_signals,
            weekly_bias=weekly_bias,
            weekly_strength=weekly_strength,
            atr_value=atr_value,
            trend_status=trend_status,
            structure_warning=structure_warning,
        )

        if exit_signal is None:
            return None

        return Signal(
            signal_type=SignalType.SELL,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            quantity=exit_signal.quantity,
            reason=exit_signal.reason,
            confidence=exit_signal.confidence,
        )

    def _detect_sell_signals(
        self, symbol: str, price: float
    ) -> List[BuySellPoint]:
        """检测日线卖点（仅保留近期信号）"""
        fractals = self._fractals.get(symbol, [])
        strokes = self._strokes.get(symbol, [])
        pivots = self._pivots.get(symbol, [])
        macd = self._macd.get(symbol)

        if not fractals or not strokes:
            return []

        sells = []

        # 批量检测器卖点
        detector = self._detector.get(symbol)
        if detector is not None:
            try:
                detector.detect_all()
                sells.extend(detector._sell_points)
            except Exception:
                pass

        # 结构化卖点
        structural_sell = self._detect_structural_sell(symbol, price)
        if structural_sell:
            sells.append(structural_sell)

        # 过滤过期卖点：仅保留距当前K线末尾10根以内的信号
        current_kline_idx = self._kline_len.get(symbol, 0) - 1
        if sells and current_kline_idx > 0:
            sells = [s for s in sells if abs(s.index - current_kline_idx) <= 10]

        return sells

    def _detect_structural_sell(
        self, symbol: str, price: float
    ) -> Optional[BuySellPoint]:
        """结构化卖点检测"""
        fractals = self._fractals.get(symbol, [])
        strokes = self._strokes.get(symbol, [])
        pivots = self._pivots.get(symbol, [])
        macd = self._macd.get(symbol)

        if not fractals or not strokes:
            return None

        if not fractals[-1].is_top:
            return None

        # 1卖：顶背驰
        if len(strokes) >= 3 and pivots:
            up = [s for s in strokes if s.is_up]
            if len(up) >= 2:
                last_up, prev_up = up[-1], up[-2]
                if last_up == strokes[-1] and last_up.end_value > pivots[-1].high:
                    div_ratio = 0.0
                    has_div = False
                    if macd:
                        has_div, div_ratio = macd.check_divergence(
                            last_up.start_index, last_up.end_index, 'up',
                            prev_start=prev_up.start_index, prev_end=prev_up.end_index,
                        )
                    if has_div or not macd:
                        if not macd:
                            r1, r2 = prev_up.price_change_pct, last_up.price_change_pct
                            if r2 >= r1 * 0.7:
                                return None
                            div_ratio = 1 - r2 / r1 if r1 > 0 else 0
                        return BuySellPoint(
                            point_type='1sell',
                            price=last_up.end_value,
                            index=last_up.end_index,
                            related_pivot=pivots[-1],
                            divergence_ratio=div_ratio,
                            confidence=0.6 + min(div_ratio * 0.4, 0.4),
                            reason=f'1卖: 顶背驰(强度{div_ratio:.2f})',
                        )

        # 2卖：反弹不破前高
        if not pivots:
            return None
        up = [s for s in strokes if s.is_up and s.end_value > pivots[-1].high]
        if not up:
            return None

        last_up = up[-1]
        down_after = [s for s in strokes if s.is_down and s.start_index > last_up.end_index]
        if not down_after:
            return None

        bounce = [s for s in strokes if s.is_up and s.start_index > down_after[-1].end_index]
        if not bounce:
            return None

        last_bounce = bounce[-1]
        if last_bounce.end_value >= last_up.end_value:
            return None
        if last_bounce != strokes[-1]:
            return None

        return BuySellPoint(
            point_type='2sell',
            price=last_bounce.end_value,
            index=last_bounce.end_index,
            related_pivot=pivots[-1],
            confidence=0.6,
            reason=f'2卖: 反弹不破推断1卖{last_up.end_value:.2f}',
        )

    def on_order(self, signal: Signal, executed_price: float, executed_quantity: int) -> None:
        """订单成交回调"""
        super().on_order(signal, executed_price, executed_quantity)
        symbol = signal.symbol
        bar_idx = self._current_bar_index.get(symbol, 0)

        if signal.is_buy():
            chan_stop = signal.metadata.get('stop_loss', 0) if signal.metadata else 0
            buy_type = signal.metadata.get('buy_point_type', '') if signal.metadata else ''
            self.exit_manager.on_buy(
                symbol=symbol,
                price=executed_price,
                bar_index=bar_idx,
                chan_stop=chan_stop,
                buy_point_type=buy_type,
            )

        elif signal.is_sell():
            remaining = self.get_position(symbol)
            if remaining <= 0:
                # Full exit - clear position record
                self.exit_manager.on_sell(symbol)
                if self._cooldown_filter:
                    self._cooldown_filter.record_sell(symbol, bar_idx)
            # Partial exit: keep exit_manager record (preserves exit_stage)

    def _resample_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线→周线resample"""
        if len(df) == 0:
            return df
        weekly = df.resample('W').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
        if 'amount' in df.columns:
            weekly['amount'] = df.resample('W').agg({'amount': 'sum'}).dropna()['amount']
        return weekly
