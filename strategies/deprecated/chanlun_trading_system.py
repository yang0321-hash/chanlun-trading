"""
缠论稳定盈利交易系统

基于缠论理论的核心买卖点，结合多级别联立分析、动态仓位管理和严格风控，
构建一套可复制的稳定盈利交易体系。

## 核心理念

1. **多级别联立**：周线定方向，日线找买点，30分确认入场
2. **三类买点优先**：2买 > 3买 > 1买（安全性递减，收益潜力递增）
3. **中枢操作**：在中枢下沿买入，中枢上沿卖出
4. **背驰确认**：MACD背驰确认转折
5. **严格止损**：每笔交易风险不超过2%

## 交易规则

### 入场条件（多级别共振）

**周线级别**：
- 趋势向上或形成底部中枢
- 存在周线笔结构（至少3笔）

**日线级别**：
- 出现2买或3买信号
- MACD金叉或底背离
- 价格在中枢下沿附近或突破前高

**30分钟级别**（确认）：
- 分型确认
- 笔形成向上

### 出场条件

1. **止损**：
   - 1买：跌破前一中枢下沿
   - 2买：跌破1买低点
   - 3买：跌回中枢内

2. **止盈**：
   - 目标位：前高或中枢上沿
   - MACD顶背离减仓50%
   - 日线2卖清仓

3. **移动止损**：
   - 盈利>10%后，最高价回撤5%止损

### 仓位管理

- 2买：80%仓位（高确定性）
- 3买：60%仓位（中等确定性）
- 1买：40%仓位（低确定性，需抄底）
- 单笔最大风险：2%
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field

from backtest.strategy import Strategy, Signal, SignalType
from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator, Stroke
from core.segment import SegmentGenerator
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD
from indicator.volume import VolumeAnalyzer, VolumePattern
from backtest.position_sizing import RiskParitySizer, PositionResult


@dataclass
class MarketState:
    """市场状态"""
    trend: str = 'unknown'  # up, down, range
    trend_strength: float = 0.5  # 0-1
    volatility: float = 0.2  # 年化波动率
    in_range: bool = False  # 是否在震荡区间
    support: float = 0  # 支撑位
    resistance: float = 0  # 压力位


@dataclass
class BuyPoint:
    """买点信息"""
    point_type: str  # '1买', '2买', '3买'
    price: float
    confidence: float  # 0-1
    reason: str
    stop_loss: float
    target: float
    risk_reward: float


@dataclass
class PositionRecord:
    """持仓记录"""
    symbol: str
    entry_price: float
    entry_date: datetime
    quantity: int
    stop_loss: float
    initial_stop: float
    target_price: float
    buy_point_type: str
    highest_price: float = field(default_factory=lambda: 0.0)
    partial_exit_done: bool = False
    trailing_stop_activated: bool = False


class ChanLunTradingSystem(Strategy):
    """
    缠论稳定盈利交易系统

    核心特点：
    1. 多级别联立判断（周线+日线）
    2. 三类买点精确识别
    3. 动态止损止盈
    4. 风险平价仓位管理
    5. 趋势过滤
    """

    def __init__(
        self,
        name: str = '缠论稳定盈利系统',
        # 风险参数
        max_risk_per_trade: float = 0.02,  # 单笔最大风险2%
        max_drawdown_pct: float = 0.15,    # 最大回撤15%后暂停
        # 买入参数
        enable_buy1: bool = True,   # 是否交易1买
        enable_buy2: bool = True,   # 是否交易2买
        enable_buy3: bool = True,   # 是否交易3买
        min_confidence: float = 0.6,  # 最低信号置信度
        # 止损参数
        stop_loss_atr_multiplier: float = 2.0,
        trailing_stop_pct: float = 0.05,  # 移动止损5%
        trailing_activate_pct: float = 0.10,  # 盈利10%后启用移动止损
        # 止盈参数
        partial_exit_ratio: float = 0.5,  # 顶背离减仓50%
        # 中枢参数
        pivot_touch_tolerance: float = 0.03,  # 中枢边缘容差3%
        # 量能参数
        enable_volume_confirm: bool = True,  # 是否启用量能确认
        min_volume_ratio: float = 1.2,       # 最小量比要求
        enable_volume_divergence: bool = True,  # 是否启用量能背离
        # 延长止盈参数
        second_sell_partial_exit: bool = False,  # 2卖只减仓不平仓
        second_sell_exit_ratio: float = 0.5,    # 2卖减仓比例
    ):
        super().__init__(name)
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct
        self.enable_buy1 = enable_buy1
        self.enable_buy2 = enable_buy2
        self.enable_buy3 = enable_buy3
        self.min_confidence = min_confidence
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activate_pct = trailing_activate_pct
        self.partial_exit_ratio = partial_exit_ratio
        self.pivot_touch_tolerance = pivot_touch_tolerance
        self.enable_volume_confirm = enable_volume_confirm
        self.min_volume_ratio = min_volume_ratio
        self.enable_volume_divergence = enable_volume_divergence
        self.second_sell_partial_exit = second_sell_partial_exit
        self.second_sell_exit_ratio = second_sell_exit_ratio

        # 量能分析器
        self._volume_analyzer: Optional[VolumeAnalyzer] = None

        # 周线数据缓存
        self._weekly_kline: Optional[KLine] = None
        self._weekly_fractals: List[Fractal] = []
        self._weekly_strokes: List[Stroke] = []
        self._weekly_pivots: List[Pivot] = []
        self._weekly_macd: Optional[MACD] = None
        self._weekly_buy_points: List[BuyPoint] = []

        # 日线数据缓存
        self._daily_kline: Optional[KLine] = None
        self._daily_fractals: List[Fractal] = []
        self._daily_strokes: List[Stroke] = []
        self._daily_segments: List = []
        self._daily_pivots: List[Pivot] = []
        self._daily_macd: Optional[MACD] = None

        # 市场状态
        self._market_state: MarketState = MarketState()

        # 持仓管理 {symbol: PositionRecord}
        self._positions: Dict[str, PositionRecord] = {}

        # 缓存标识
        self._last_weekly_count: int = 0
        self._last_daily_count: int = 0

        # 仓位管理器
        self._position_sizer: Optional[RiskParitySizer] = None

        # 风控状态
        self._is_paused: bool = False
        self._peak_equity: float = 0

        # ATR缓存
        self._atr: float = 0

    def initialize(self, capital: float, symbols: List[str]) -> None:
        """初始化系统"""
        super().initialize(capital, symbols)
        self._peak_equity = capital
        self._position_sizer = RiskParitySizer(
            initial_capital=capital,
            risk_per_trade=self.max_risk_per_trade,
            atr_multiplier=self.stop_loss_atr_multiplier,
        )
        logger.info(f"初始化{self.name}: 初始资金¥{capital:,.0f}, 品种{symbols}")

    def reset(self) -> None:
        """重置系统"""
        super().reset()
        self._positions.clear()
        self._is_paused = False
        self._peak_equity = self.initial_capital

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """处理K线"""
        daily_df = context['data'].get(symbol)
        if daily_df is None or len(daily_df) < 60:
            return None

        current_price = bar['close']
        current_position = self.get_position(symbol)

        # 更新权益峰值
        equity = self.get_equity(context.get('current_prices', {symbol: current_price}))
        if equity > self._peak_equity:
            self._peak_equity = equity

        # 检查最大回撤，暂停交易
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self.max_drawdown_pct:
            if not self._is_paused:
                logger.warning(f"最大回撤{drawdown:.1%}超过限制，暂停交易")
                self._is_paused = True
            # 已有持仓只允许止损
            if current_position > 0:
                return self._check_stop_loss_only(symbol, current_price, bar)
            return None

        if drawdown < self.max_drawdown_pct * 0.5:
            if self._is_paused:
                logger.info(f"回撤恢复至{drawdown:.1%}，恢复交易")
                self._is_paused = False

        # 生成周线数据
        weekly_df = self._convert_to_weekly(daily_df)

        # 更新分析
        self._update_weekly_analysis(weekly_df)
        self._update_daily_analysis(daily_df)

        # 更新ATR
        self._atr = self._calculate_atr(daily_df)

        # 更新市场状态
        self._update_market_state(daily_df)

        # 已有持仓：检查出场信号
        if current_position > 0:
            return self._check_exit_signals(symbol, current_price, bar, daily_df)

        # 暂停状态不开新仓
        if self._is_paused:
            return None

        # 无持仓：检查买入信号
        return self._check_entry_signals(symbol, current_price, bar, daily_df)

    def _convert_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """将日线数据转换为周线数据"""
        weekly = daily_df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()
        return weekly

    def _update_weekly_analysis(self, df: pd.DataFrame) -> None:
        """更新周线分析"""
        if len(df) == self._last_weekly_count:
            return

        if len(df) < 30:
            return

        self._weekly_kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(self._weekly_kline, confirm_required=False)
        self._weekly_fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(self._weekly_kline, self._weekly_fractals, min_bars=3)
        self._weekly_strokes = stroke_gen.get_strokes()

        # 识别中枢
        pivot_detector = PivotDetector(self._weekly_kline, self._weekly_strokes)
        self._weekly_pivots = pivot_detector.get_pivots()

        # 计算MACD
        self._weekly_macd = MACD(df['close'])

        # 识别买点
        self._weekly_buy_points = self._identify_buy_points(
            self._weekly_strokes,
            self._weekly_pivots,
            self._weekly_macd,
            df['close'].values
        )

        self._last_weekly_count = len(df)

    def _update_daily_analysis(self, df: pd.DataFrame) -> None:
        """更新日线分析"""
        if len(df) == self._last_daily_count:
            return

        self._daily_kline = KLine.from_dataframe(df, strict_mode=False)

        # 识别分型
        detector = FractalDetector(self._daily_kline, confirm_required=False)
        self._daily_fractals = detector.get_fractals()

        # 生成笔
        stroke_gen = StrokeGenerator(self._daily_kline, self._daily_fractals, min_bars=3)
        self._daily_strokes = stroke_gen.get_strokes()

        # 生成线段
        seg_gen = SegmentGenerator(self._daily_kline, self._daily_strokes)
        self._daily_segments = seg_gen.get_segments()

        # 识别中枢
        pivot_detector = PivotDetector(self._daily_kline, self._daily_strokes)
        self._daily_pivots = pivot_detector.get_pivots()

        # 计算MACD
        self._daily_macd = MACD(df['close'])

        # 初始化量能分析器
        if len(df) >= 30:
            self._volume_analyzer = VolumeAnalyzer(
                df['close'].values,
                df['volume'].values
            )

        self._last_daily_count = len(df)

    def _update_market_state(self, df: pd.DataFrame) -> None:
        """更新市场状态"""
        if len(self._daily_strokes) < 5:
            return

        # 判断趋势
        recent_strokes = self._daily_strokes[-10:]
        ups = [s for s in recent_strokes if s.is_up]
        downs = [s for s in recent_strokes if s.is_down]

        if not ups or not downs:
            self._market_state.trend = 'unknown'
            return

        # 波段高点是否递增
        higher_highs = all(ups[i].end_value >= ups[i-1].end_value for i in range(1, len(ups)))
        # 波段低点是否递增
        higher_lows = all(downs[i].end_value >= downs[i-1].end_value for i in range(1, len(downs)))

        if higher_highs and higher_lows:
            self._market_state.trend = 'up'
            self._market_state.trend_strength = 0.8
        elif not higher_highs and not higher_lows:
            self._market_state.trend = 'down'
            self._market_state.trend_strength = 0.8
        else:
            self._market_state.trend = 'range'
            self._market_state.trend_strength = 0.4

        # 计算支撑压力位
        if self._daily_pivots:
            last_pivot = self._daily_pivots[-1]
            self._market_state.support = last_pivot.low
            self._market_state.resistance = last_pivot.high
            self._market_state.in_range = (
                last_pivot.low * 1.05 < df['close'].iloc[-1] < last_pivot.high * 0.95
            )

    def _identify_buy_points(
        self,
        strokes: List[Stroke],
        pivots: List[Pivot],
        macd: Optional[MACD],
        prices: np.ndarray
    ) -> List[BuyPoint]:
        """识别三类买点"""
        buy_points = []

        if len(strokes) < 3:
            return buy_points

        # 获取最近的向下笔
        down_strokes = [s for s in strokes if s.is_down]

        for i, stroke in enumerate(down_strokes[-3:]):  # 检查最近3个向下笔
            if i >= len(strokes) - 1:
                continue

            # 找到该向下笔之后的向上笔
            idx = strokes.index(stroke)
            if idx + 1 >= len(strokes):
                continue

            next_stroke = strokes[idx + 1]
            if not next_stroke.is_up:
                continue

            # 判断买点类型
            buy_type, confidence, reason = self._classify_buy_point(
                stroke, next_stroke, strokes, pivots, macd, prices
            )

            if buy_type:
                buy_points.append(BuyPoint(
                    point_type=buy_type,
                    price=next_stroke.start_value,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=stroke.low * 0.98,
                    target=next_stroke.end_value * 1.1,
                    risk_reward=abs(next_stroke.end_value - stroke.low) / max(abs(next_stroke.start_value - stroke.low), 0.01)
                        if stroke.low > 0 else 2
                ))

        return buy_points

    def _classify_buy_point(
        self,
        down_stroke: Stroke,
        up_stroke: Stroke,
        all_strokes: List[Stroke],
        pivots: List[Pivot],
        macd: Optional[MACD],
        prices: np.ndarray
    ) -> Tuple[str, float, str]:
        """分类买点类型并计算置信度"""
        confidence = 0.0
        reason = ""

        # 检查是否有中枢
        has_pivot = len(pivots) > 0
        last_pivot = pivots[-1] if has_pivot else None

        # 1买：最后中枢下移后的向下笔结束
        is_buy1 = False
        if has_pivot and len(pivots) >= 2:
            # 检查是否形成下移的中枢
            if pivots[-1].high < pivots[-2].high and pivots[-1].low < pivots[-2].low:
                # 最后一个向下笔的低点可能是1买
                is_buy1 = True

        # 2买：1买后的回踩不创新低
        is_buy2 = False
        if has_pivot and len(all_strokes) >= 3:
            # 检查是否有前低
            prev_lows = [s.low for s in all_strokes[-5:] if s.is_down]
            if len(prev_lows) >= 2:
                if down_stroke.low >= prev_lows[-2] * 0.98:  # 不创新低
                    is_buy2 = True

        # 3买：突破中枢后回踩不回中枢
        is_buy3 = False
        if has_pivot and last_pivot:
            # 价格突破中枢上沿
            if up_stroke.start_value > last_pivot.high:
                # 回踩不回中枢
                if down_stroke.low >= last_pivot.low * 0.98:
                    is_buy3 = True

        # 计算置信度
        if is_buy2 and self.enable_buy2:
            base_conf = 0.7
            # MACD金叉加分
            if macd and self._check_macd_golden_cross(macd):
                base_conf += 0.15
            # 周线向上加分
            if self._market_state.trend == 'up':
                base_conf += 0.1
            # 在中枢下沿加分
            if last_pivot and abs(up_stroke.start_value - last_pivot.low) / last_pivot.low < 0.03:
                base_conf += 0.05
            confidence = min(base_conf, 0.95)
            return '2买', confidence, f'2买:回踩不创新低'

        if is_buy3 and self.enable_buy3:
            base_conf = 0.6
            # 突破确认加分
            if up_stroke.start_value > last_pivot.high * 1.02:
                base_conf += 0.1
            # 周线向上加分
            if self._market_state.trend == 'up':
                base_conf += 0.15
            confidence = min(base_conf, 0.9)
            return '3买', confidence, f'3买:突破后回踩确认'

        if is_buy1 and self.enable_buy1:
            base_conf = 0.5
            # MACD底背离加分
            if macd and self._check_macd_divergence(macd, 'bottom'):
                base_conf += 0.2
            confidence = min(base_conf, 0.7)
            return '1买', confidence, f'1买:底背驰反转'

        return '', 0.0, ''

    def _check_macd_golden_cross(self, macd: MACD) -> bool:
        """检查MACD金叉"""
        return macd.check_golden_cross()

    def _check_macd_divergence(self, macd: MACD, direction: str) -> bool:
        """检查MACD背驰"""
        if len(macd) < 20:
            return False
        has_div, _ = macd.check_divergence(
            max(0, len(macd) - 20),
            len(macd) - 1,
            'down' if direction == 'bottom' else 'up'
        )
        return has_div

    def _classify_current_buy_point(
        self,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Tuple[str, float, str]:
        """分类当前买点类型（基于实时笔结构）"""
        if len(self._daily_strokes) < 3:
            return '', 0.0, ''

        last_stroke = self._daily_strokes[-1]
        if not last_stroke.is_up:
            return '', 0.0, ''

        # 获取最近的向下笔
        prev_down_strokes = [s for s in self._daily_strokes[-5:] if s.is_down]
        if not prev_down_strokes:
            return '', 0.0, ''

        last_down = prev_down_strokes[-1]

        # 检查2买：向上且回踩不创新低
        is_buy2 = False
        if len(prev_down_strokes) >= 2:
            prev_low = prev_down_strokes[-2].low
            if last_down.low >= prev_low * 0.98:  # 不创新低
                is_buy2 = True

        # 检查3买：突破中枢后回踩
        is_buy3 = False
        if self._daily_pivots:
            last_pivot = self._daily_pivots[-1]
            if last_stroke.start_value > last_pivot.high:  # 突破
                if last_down.low >= last_pivot.low * 0.98:  # 回踩不回中枢
                    is_buy3 = True

        # 计算置信度
        confidence = 0.0
        reason = ""

        if is_buy2 and self.enable_buy2:
            confidence = 0.70
            reason = "回踩不创新低"

            # MACD确认
            if self._daily_macd and self._check_macd_golden_cross(self._daily_macd):
                confidence += 0.10
                reason += "+MACD金叉"

            # 量能确认
            if self.enable_volume_confirm and self._volume_analyzer:
                vol_confirmed, vol_reason = self._volume_analyzer.check_volume_confirmation(
                    min_ratio=self.min_volume_ratio
                )
                if vol_confirmed:
                    confidence += 0.10
                    reason += f"+{vol_reason}"
                else:
                    confidence -= 0.15  # 量能不足降低置信度
                    reason += f"+{vol_reason}"

            # 检查量能背离
            if self.enable_volume_divergence and self._volume_analyzer:
                divergence = self._volume_analyzer.check_divergence()
                if divergence.has_divergence and divergence.direction == 'bottom':
                    confidence += 0.15
                    reason += f"+量能底背离({divergence.strength:.1%})"

            # 趋势确认
            if self._market_state.trend == 'up':
                confidence += 0.10
            elif self._market_state.trend == 'range':
                confidence += 0.05
            elif self._market_state.trend == 'down':
                confidence -= 0.10  # 下降趋势降低置信度

            # 中枢位置加分
            if self._daily_pivots:
                last_pivot = self._daily_pivots[-1]
                if abs(price - last_pivot.low) / last_pivot.low < 0.05:
                    confidence += 0.05
                    reason += "+中枢下沿"

            return '2买', min(confidence, 0.90), reason

        if is_buy3 and self.enable_buy3:
            confidence = 0.65
            reason = "突破后回踩确认"

            if self._market_state.trend == 'up':
                confidence += 0.15

            if self._daily_macd and self._check_macd_golden_cross(self._daily_macd):
                confidence += 0.10

            # 量能确认 - 突破需要放量
            if self.enable_volume_confirm and self._volume_analyzer:
                vol_confirmed, vol_reason = self._volume_analyzer.check_volume_confirmation(
                    min_ratio=1.3  # 3买要求更高的量比
                )
                if vol_confirmed:
                    confidence += 0.10
                    reason += f"+{vol_reason}"
                else:
                    confidence -= 0.10  # 无放量确认

            return '3买', min(confidence, 0.85), reason

        # 基础向上笔买入（简化版1买/3买）
        if self.enable_buy1 or self.enable_buy3:
            confidence = 0.50
            reason = "向上笔形成"

            # MACD底背离
            if self._daily_macd and self._check_macd_divergence(self._daily_macd, 'bottom'):
                confidence += 0.20
                reason = "向上笔+底背离"

            # 量能底背离
            if self.enable_volume_divergence and self._volume_analyzer:
                divergence = self._volume_analyzer.check_divergence()
                if divergence.has_divergence and divergence.direction == 'bottom':
                    confidence += 0.15
                    reason += f"+量能底背离"

            # 基础量能确认
            if self.enable_volume_confirm and self._volume_analyzer:
                vol_confirmed, vol_reason = self._volume_analyzer.check_volume_confirmation(
                    min_ratio=1.1  # 较低的量比要求
                )
                if vol_confirmed:
                    confidence += 0.05
                    reason += f"+{vol_reason}"

            return '向上笔', min(confidence, 0.75), reason

        return '', 0.0, ''

    def _check_entry_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查买入信号 - 实时检测新形成的买点"""
        # 1. 周线趋势过滤 - 只过滤强下降趋势
        if self._market_state.trend == 'down' and self._market_state.trend_strength > 0.90:
            return None  # 只在极强下降趋势中不做多

        # 2. 检查是否有足够的笔数据
        if len(self._daily_strokes) < 3:
            return None

        # 3. 检查最后一笔是否是向上笔（刚刚形成）
        last_stroke = self._daily_strokes[-1]
        if not last_stroke.is_up:
            return None  # 最后一笔不是向上，没有买点

        # 4. 使用当前笔模式判断买点类型
        buy_type, confidence, reason = self._classify_current_buy_point(price, bar, daily_df)

        if not buy_type:
            return None

        # 5. 检查置信度 - 使用更低的阈值进行初步筛选
        if confidence < self.min_confidence - 0.1:  # 降低阈值，后面还会再检查
            return None

        # 6. 综合评分
        final_score = confidence
        if self._market_state.trend == 'up':
            final_score += 0.05
        elif self._market_state.trend == 'range':
            final_score += 0.02

        # 7. 最终置信度检查
        if final_score < self.min_confidence:
            return None

        # 8. 计算止损位
        if len(self._daily_strokes) >= 2:
            prev_down_strokes = [s for s in self._daily_strokes[-5:] if s.is_down]
            if prev_down_strokes:
                stop_loss = prev_down_strokes[-1].low * 0.98
            else:
                stop_loss = price * 0.95
        else:
            stop_loss = price * 0.95

        signal = Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            reason=f"{buy_type} {reason} (置信度:{final_score:.2f})",
            confidence=min(final_score, 0.95)
        )
        logger.debug(f"买入信号: {symbol} @ {price:.2f} - {signal.reason}")
        return signal

    def _check_weekly_support(self, price: float) -> bool:
        """检查周线是否支持买入"""
        if not self._weekly_strokes or not self._weekly_pivots:
            return True  # 无周线数据时不限制

        # 检查周线是否处于向上阶段
        if self._weekly_strokes[-1].is_up:
            return True

        # 检查价格是否在周线中枢下沿附近
        if self._weekly_pivots:
            last_pivot = self._weekly_pivots[-1]
            if abs(price - last_pivot.low) / last_pivot.low < 0.05:
                return True

        return False

    def _check_exit_signals(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        daily_df: pd.DataFrame
    ) -> Optional[Signal]:
        """检查卖出信号"""
        record = self._positions.get(symbol)
        if not record:
            return None

        # 更新最高价
        if price > record.highest_price:
            record.highest_price = price

        profit_pct = (price - record.entry_price) / record.entry_price

        # 1. 止损检查
        if price <= record.stop_loss:
            return self._create_exit_signal(
                symbol, price, bar, self.get_position(symbol),
                f"止损: 亏损{profit_pct:.2%}"
            )

        # 2. 移动止损
        if profit_pct > self.trailing_activate_pct:
            if not record.trailing_stop_activated:
                record.trailing_stop_activated = True

            if record.trailing_stop_activated:
                trailing_stop = record.highest_price * (1 - self.trailing_stop_pct)
                if price <= trailing_stop:
                    return self._create_exit_signal(
                        symbol, price, bar, self.get_position(symbol),
                        f"移动止损: 盈利{profit_pct:.2%}"
                    )

        # 3. 目标位止盈
        if price >= record.target_price:
            return self._create_exit_signal(
                symbol, price, bar, self.get_position(symbol),
                f"目标位止盈: 盈利{profit_pct:.2%}"
            )

        # 4. 技术信号减仓
        if not record.partial_exit_done and profit_pct > 0.08:
            exit_reason = None

            # 检查MACD顶背离
            if self._check_macd_divergence(self._daily_macd, 'top'):
                exit_reason = f"MACD顶背离"

            # 检查量能顶背离
            if self.enable_volume_divergence and self._volume_analyzer:
                divergence = self._volume_analyzer.check_divergence()
                if divergence.has_divergence and divergence.direction == 'top':
                    exit_reason = f"量能顶背离({divergence.strength:.1%})"

            # 检查派发阶段
            if self.enable_volume_confirm and self._volume_analyzer:
                if self._volume_analyzer.is_distribution_phase():
                    exit_reason = "进入派发阶段"

            # 检查量价背离：价涨量缩
            if self.enable_volume_confirm and self._volume_analyzer:
                price_change = (price - record.highest_price) / record.highest_price
                if price_change > 0:  # 价格创新高
                    pattern = self._volume_analyzer.get_current_pattern()
                    if pattern.is_contracting:
                        exit_reason = f"价涨量缩(量比{pattern.ratio_vs_avg:.1f})"

            if exit_reason:
                record.partial_exit_done = True
                exit_qty = int(self.get_position(symbol) * self.partial_exit_ratio / 100) * 100
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason=f"{exit_reason}减仓{self.partial_exit_ratio:.0%} (盈利{profit_pct:.2%})",
                    confidence=0.8
                )

        # 5. 日线2卖处理
        if self._check_daily_second_sell():
            if self.second_sell_partial_exit and not record.partial_exit_done:
                # 只减仓，不平仓
                record.partial_exit_done = True
                exit_qty = int(self.get_position(symbol) * self.second_sell_exit_ratio / 100) * 100
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
                    price=price,
                    quantity=exit_qty,
                    reason=f"日线2卖减仓{self.second_sell_exit_ratio:.0%} (盈利{profit_pct:.2%})",
                    confidence=0.7
                )
            else:
                # 原始逻辑：清仓
                return self._create_exit_signal(
                    symbol, price, bar, self.get_position(symbol),
                    f"日线2卖: 盈利{profit_pct:.2%}"
                )

        return None

    def _check_stop_loss_only(self, symbol: str, price: float, bar: pd.Series) -> Optional[Signal]:
        """仅检查止损（暂停交易时）"""
        record = self._positions.get(symbol)
        if not record:
            return None

        if price <= record.stop_loss:
            return self._create_exit_signal(
                symbol, price, bar, self.get_position(symbol),
                "止损（风控暂停期）"
            )
        return None

    def _check_daily_second_sell(self) -> bool:
        """检查日线2卖信号"""
        if len(self._daily_strokes) < 3:
            return False

        last = self._daily_strokes[-1]
        second_last = self._daily_strokes[-2]

        # 向上笔后转为向下笔，且反弹不破前高
        if last.is_down and second_last.is_up:
            if last.end_value < second_last.start_value * 0.98:
                return True

        return False

    def _create_exit_signal(
        self,
        symbol: str,
        price: float,
        bar: pd.Series,
        quantity: int,
        reason: str
    ) -> Signal:
        """创建退出信号"""
        return Signal(
            signal_type=SignalType.SELL,
            symbol=symbol,
            datetime=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            price=price,
            quantity=quantity,
            reason=reason,
            confidence=1.0
        )

    def on_order(
        self,
        signal: Signal,
        executed_price: float,
        executed_quantity: int
    ) -> None:
        """订单成交回调"""
        symbol = signal.symbol

        if signal.is_buy():
            # 记录持仓
            stop_loss = executed_price * 0.95  # 默认5%止损
            if self._atr > 0:
                stop_loss = executed_price - self._atr * self.stop_loss_atr_multiplier

            # 根据买点类型设置止损
            if '1买' in signal.reason:
                stop_loss = executed_price * 0.92
            elif '2买' in signal.reason:
                if self._daily_pivots:
                    stop_loss = self._daily_pivots[-1].low * 0.98

            self._positions[symbol] = PositionRecord(
                symbol=symbol,
                entry_price=executed_price,
                entry_date=signal.datetime,
                quantity=executed_quantity,
                stop_loss=stop_loss,
                initial_stop=stop_loss,
                target_price=executed_price * 1.15,  # 默认15%目标
                buy_point_type=signal.reason.split()[0] if signal.reason else '',
                highest_price=executed_price
            )

            self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
            self.cash -= executed_price * executed_quantity

            logger.info(
                f"买入 {symbol} @ {executed_price:.2f} x {executed_quantity} "
                f"| 止损:{stop_loss:.2f} | {signal.reason}"
            )

        elif signal.is_sell():
            qty = signal.quantity if signal.quantity else self.get_position(symbol)
            self.position[symbol] = self.position.get(symbol, 0) - qty
            self.cash += executed_price * qty

            profit = (executed_price - self._positions[symbol].entry_price) * qty
            profit_pct = (executed_price - self._positions[symbol].entry_price) / self._positions[symbol].entry_price

            logger.info(
                f"卖出 {symbol} @ {executed_price:.2f} x {qty} "
                f"| 盈亏:{profit:,.0f}({profit_pct:.2%}) | {signal.reason}"
            )

            # 如果全部卖出，清除记录
            if self.get_position(symbol) == 0:
                del self._positions[symbol]

    def get_position_size(
        self,
        symbol: str,
        price: float,
        confidence: float
    ) -> int:
        """计算仓位大小"""
        if self._position_sizer is None:
            return 100

        result = self._position_sizer.calculate(
            price=price,
            cash=self.cash,
            atr=self._atr
        )

        # 根据置信度调整
        adjusted_shares = int(result.shares * confidence)
        adjusted_shares = (adjusted_shares // 100) * 100

        return max(adjusted_shares, 100)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        if len(df) < period + 1:
            return 0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_list.append(max(tr1, tr2, tr3))

        if not tr_list:
            return 0

        return np.mean(tr_list[-period:])

    def get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_paused': self._is_paused,
            'market_trend': self._market_state.trend,
            'trend_strength': self._market_state.trend_strength,
            'positions': len(self._positions),
            'peak_equity': self._peak_equity,
            'current_equity': self.cash + sum(
                p.quantity * p.entry_price for p in self._positions.values()
            )
        }
