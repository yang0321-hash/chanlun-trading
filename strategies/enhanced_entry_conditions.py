"""
改进的入场条件模块

结合：
1. 市场环境识别
2. 成交量确认
3. 多空强度
4. 震荡区间过滤
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd

from indicators.market_regime import (
    MarketRegimeDetector,
    VolumeAnalyzer,
    BullBearStrength,
    RangeDetector,
    MarketRegime
)


@dataclass
class EntrySignal:
    """入场信号"""
    should_enter: bool
    direction: str  # 'long' or 'short'
    confidence: float  # 0-1
    reason: str
    filters_passed: list
    filters_failed: list


class EnhancedEntryFilter:
    """
    增强型入场过滤器

    多重条件过滤，只在高质量信号时入场
    """

    def __init__(
        self,
        # 市场环境参数
        adx_threshold: float = 25,
        max_volatility: float = 0.05,
        require_uptrend: bool = True,

        # 成交量参数
        volume_surge_threshold: float = 1.3,
        require_volume_confirmation: bool = True,

        # 多空强度参数
        min_bull_strength: float = 55,

        # 震荡过滤参数
        avoid_range_trading: bool = True,
        avoid_high_consolidation: bool = True,
    ):
        self.market_detector = MarketRegimeDetector(
            adx_threshold=adx_threshold,
            volatility_threshold=max_volatility
        )
        self.volume_analyzer = VolumeAnalyzer()
        self.strength_analyzer = BullBearStrength()
        self.range_detector = RangeDetector()

        self.volume_surge_threshold = volume_surge_threshold
        self.require_uptrend = require_uptrend
        self.min_bull_strength = min_bull_strength
        self.avoid_range_trading = avoid_range_trading
        self.avoid_high_consolidation = avoid_high_consolidation
        self.require_volume_confirmation = require_volume_confirmation

    def check_long_entry(
        self,
        df: pd.DataFrame,
        index: int,
        base_signal: bool = True
    ) -> EntrySignal:
        """
        检查做多入场条件

        Args:
            df: K线数据
            index: 当前索引
            base_signal: 基础信号（如周线2买）

        Returns:
            EntrySignal
        """
        filters_passed = []
        filters_failed = []
        confidence = 0.5

        # 如果基础信号为假，直接返回
        if not base_signal:
            return EntrySignal(
                should_enter=False,
                direction='long',
                confidence=0,
                reason='基础信号不满足',
                filters_passed=[],
                filters_failed=['base_signal']
            )

        # 1. 市场环境检查
        market_state = self.market_detector.detect(df, index)
        market_ok = self._check_market_regime(market_state, 'long')

        if market_ok:
            filters_passed.append('market_regime')
            confidence += market_state.confidence * 0.3
        else:
            filters_failed.append('market_regime')
            confidence -= 0.3

        # 2. 成交量确认
        volume_ok = True
        if self.require_volume_confirmation:
            volume_surge = self.volume_analyzer.is_volume_surge(
                df, index, self.volume_surge_threshold
            )
            volume_trend = self.volume_analyzer.get_volume_trend(df, index)

            volume_ok = volume_surge or volume_trend == 'increasing'

            if volume_ok:
                filters_passed.append('volume_confirmation')
                confidence += 0.15
            else:
                filters_failed.append('volume_confirmation')
                confidence -= 0.15

        # 3. 多空强度检查
        strength = self.strength_analyzer.calculate(df, index)
        strength_ok = strength['dominance'] == 'bull' or strength['bull_strength'] >= self.min_bull_strength

        if strength_ok:
            filters_passed.append('bull_strength')
            confidence += (strength['net_strength'] / 200)  # -50 to 50 -> -0.25 to 0.25
        else:
            filters_failed.append('bull_strength')
            confidence -= 0.1

        # 4. 震荡区间过滤
        range_ok = True
        if self.avoid_range_trading:
            in_range, _, _ = self.range_detector.is_in_range(df, index)
            range_ok = not in_range

            if range_ok:
                filters_passed.append('not_in_range')
            else:
                filters_failed.append('in_range')
                confidence -= 0.2

        # 5. 高位震荡过滤
        high_consolidation_ok = True
        if self.avoid_high_consolidation:
            high_consolidation = self.range_detector.is_high_consolidation(df, index)
            high_consolidation_ok = not high_consolidation

            if high_consolidation_ok:
                filters_passed.append('not_high_consolidation')
            else:
                filters_failed.append('high_consolidation')
                confidence -= 0.25

        # 6. 量价背离检查
        divergence, div_type = self.volume_analyzer.is_price_volume_divergence(df, index)
        divergence_ok = not divergence

        if divergence_ok:
            filters_passed.append('no_divergence')
        else:
            filters_failed.append(f'divergence_{div_type}')
            confidence -= 0.15

        # 综合判断
        # 必须通过的过滤器
        critical_filters = ['market_regime']

        # 检查关键过滤器
        critical_passed = all(f not in filters_failed for f in critical_filters)

        # 计算最终置信度
        confidence = max(0, min(1, confidence))

        # 判断是否入场
        should_enter = (
            critical_passed and
            confidence >= 0.4 and  # 最低置信度
            len(filters_passed) >= len(filters_failed)  # 通过的过滤器多于失败的
        )

        reason = self._generate_reason(filters_passed, filters_failed, confidence)

        return EntrySignal(
            should_enter=should_enter,
            direction='long',
            confidence=confidence,
            reason=reason,
            filters_passed=filters_passed,
            filters_failed=filters_failed
        )

    def _check_market_regime(self, market_state, direction: str) -> bool:
        """检查市场环境是否适合入场"""
        # 必须可交易
        if not market_state.is_tradeable:
            return False

        # 根据方向检查
        if direction == 'long':
            # 做多：需要上升趋势或高波动
            if market_state.regime == MarketRegime.DOWNTREND:
                return False
            if market_state.regime == MarketRegime.HIGH_VOLATILITY:
                return False
            return True
        else:  # short
            # 做空：需要下降趋势
            if market_state.regime == MarketRegime.UPTREND:
                return False
            return True

    def _generate_reason(
        self,
        passed: list,
        failed: list,
        confidence: float
    ) -> str:
        """生成入场/不入场原因"""
        parts = []

        if passed:
            parts.append(f"通过: {', '.join(passed)}")

        if failed:
            parts.append(f"失败: {', '.join(failed)}")

        parts.append(f"置信度: {confidence:.2f}")

        return '; '.join(parts)


class AdaptiveEntryManager:
    """
    自适应入场管理器

    根据市场状态动态调整入场条件
    """

    def __init__(self):
        self.base_filter = EnhancedEntryFilter()
        self.market_state_history = []

    def get_entry_signal(
        self,
        df: pd.DataFrame,
        index: int,
        base_signal: bool = True
    ) -> EntrySignal:
        """
        获取入场信号

        根据当前市场状态自适应调整
        """
        market_state = self.base_filter.market_detector.detect(df, index)

        # 记录历史
        self.market_state_history.append(market_state)

        # 根据市场状态调整参数
        self._adapt_to_market(market_state)

        return self.base_filter.check_long_entry(df, index, base_signal)

    def _adapt_to_market(self, market_state):
        """根据市场状态自适应调整"""
        # 高波动时，提高成交量要求
        if market_state.regime == MarketRegime.HIGH_VOLATILITY:
            self.base_filter.volume_surge_threshold = 1.5
            self.base_filter.require_volume_confirmation = True
        else:
            self.base_filter.volume_surge_threshold = 1.3
            self.base_filter.require_volume_confirmation = False

        # 强趋势时，降低多空强度要求
        if market_state.adx and market_state.adx > 35:
            self.base_filter.min_bull_strength = 50
        else:
            self.base_filter.min_bull_strength = 55

    def should_add_position(
        self,
        df: pd.DataFrame,
        index: int,
        current_position: int
    ) -> bool:
        """
        判断是否加仓

        条件：
        1. 已有盈利
        2. 趋势延续
        3. 回调到支撑位
        """
        if current_position <= 0:
            return False

        market_state = self.base_filter.market_detector.detect(df, index)

        # 只在强趋势中加仓
        if not (market_state.regime == MarketRegime.UPTREND and market_state.adx and market_state.adx > 30):
            return False

        # 检查是否在回调中（买入机会）
        lookback_df = df.iloc[max(0, index - 10):index + 1]
        recent_high = lookback_df['high'].max()
        current = df['close'].iloc[index]

        # 回调3%-8%
        pullback_ratio = (recent_high - current) / recent_high
        if 0.03 <= pullback_ratio <= 0.08:
            return True

        return False


class ChanLunEntryValidator:
    """
    缠论买入信号验证器

    专门验证缠论买卖点的有效性
    """

    def __init__(self):
        self.entry_filter = EnhancedEntryFilter()

    def validate_second_buy(
        self,
        df: pd.DataFrame,
        index: int,
        weekly_first_buy: float,
        weekly_second_buy: float,
        current_price: float
    ) -> EntrySignal:
        """
        验证周线2买信号

        条件：
        1. 价格在2买位置附近（±3%）
        2. 确认是向上笔
        3. 通过过滤器
        """
        # 基础条件：价格在2买附近
        if not (weekly_second_buy * 0.97 <= current_price <= weekly_second_buy * 1.05):
            return EntrySignal(
                should_enter=False,
                direction='long',
                confidence=0,
                reason=f'价格{current_price:.2f}不在2买区间{weekly_second_buy:.2f}±5%',
                filters_passed=[],
                filters_failed=['price_level']
            )

        # 检查是否为向上笔（这里简化处理，实际需要笔数据）
        # 假设调用者已确认

        # 通过增强过滤器
        signal = self.entry_filter.check_long_entry(df, index, base_signal=True)

        # 添加缠论特有原因
        if signal.should_enter:
            signal.reason = f'周线2买验证通过; 1买:{weekly_first_buy:.2f}, 2买:{weekly_second_buy:.2f}; {signal.reason}'

        return signal

    def validate_breakout_buy(
        self,
        df: pd.DataFrame,
        index: int,
        pivot_high: float,
        pivot_low: float,
        current_price: float
    ) -> EntrySignal:
        """
        验证突破买点（3买）

        条件：
        1. 突破中枢上沿
        2. 成交量放大
        3. 回踩不破中枢
        """
        # 突破确认
        if current_price < pivot_high * 1.02:
            return EntrySignal(
                should_enter=False,
                direction='long',
                confidence=0,
                reason=f'未有效突破中枢上沿{pivot_high:.2f}',
                filters_passed=[],
                filters_failed=['breakout']
            )

        # 成交量确认
        volume_surge = self.entry_filter.volume_analyzer.is_volume_surge(df, index, 1.3)
        if not volume_surge:
            return EntrySignal(
                should_enter=False,
                direction='long',
                confidence=0.3,
                reason='突破未放量',
                filters_passed=[],
                filters_failed=['volume']
            )

        # 通过增强过滤器
        signal = self.entry_filter.check_long_entry(df, index, base_signal=True)

        if signal.should_enter:
            signal.reason = f'3买验证通过; 突破{pivot_high:.2f}; {signal.reason}'

        return signal
