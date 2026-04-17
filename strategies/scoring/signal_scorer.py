"""
自适应信号评分器

非线性的信号质量评分系统，替代简单线性加法。

核心公式：
    score = divergence_score × structure_score × volume_score × regime_modifier

特性：
- 乘性组合：因子之间有非线性交互效应
- 市场状态自适应：不同状态下权重不同
- 置信度缩放：强确认放大评分，弱确认压低评分
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from loguru import logger

from .regime_detector import MarketRegime, RegimeInfo
from indicator.enhanced_divergence import DivergenceResult, DivergenceType
from core.pivot import Pivot
from core.stroke import Stroke


@dataclass
class ScoringConfig:
    """评分配置"""
    # 各因子权重（用于乘性组合中的指数，1.0=正常，>1.0=更重要）
    divergence_weight: float = 1.0
    structure_weight: float = 0.8
    volume_weight: float = 0.6
    regime_weight: float = 0.5

    # 市场状态下的买点类型权重调整
    # strong_trend: 1买权重高（趋势反转获利最大）
    # sideways: 2买/3买权重高（中枢内操作更安全）
    trend_1buy_bonus: float = 0.15
    sideways_2buy_bonus: float = 0.15
    sideways_3buy_bonus: float = 0.10

    # 最低评分阈值
    min_score: float = 0.5

    # 交互效应奖励：当多个因子同时高分时额外加分
    synergy_bonus: float = 0.1


@dataclass
class ScoringFactors:
    """评分因子集合"""
    # 背离因子
    divergence_result: Optional[DivergenceResult] = None
    divergence_ratio: float = 0.0             # MACD面积比
    pivot_divergence_ratio: float = 0.0       # 中枢进出比

    # 结构因子
    pivot_quality: float = 0.0                # 中枢质量 0-1
    stroke_strength: float = 0.0              # 关键笔强度 0-1
    strokes_in_pivot: int = 0                 # 中枢内笔数
    pivot_range_pct: float = 0.0              # 中枢振幅百分比

    # 量价因子
    volume_adjustment: float = 0.0            # 量价调整 -0.15 ~ +0.15
    volume_ratio: float = 1.0                 # 当前量/均量

    # 市场状态
    regime_info: Optional[RegimeInfo] = None

    # 趋势跟踪
    trend_status: str = 'neutral'
    trend_modifier: float = 0.0

    # 买点类型
    point_type: str = ''

    # RSI因子（从BuySignalScorer借鉴）
    rsi: float = 50.0                  # 当前RSI值，默认中性

    # 动量因子（从AAAStrategy借鉴）
    score_delta: float = 0.0              # 评分变化量（当前vs上次）


class AdaptiveSignalScorer:
    """
    自适应信号评分器

    使用方法：
        scorer = AdaptiveSignalScorer(config)
        factors = ScoringFactors(
            divergence_ratio=0.3,
            pivot_quality=0.7,
            volume_ratio=1.5,
            regime_info=regime_info,
            point_type='2buy',
        )
        score, reason = scorer.score_buy_signal(factors)
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score_buy_signal(
        self, factors: ScoringFactors
    ) -> Tuple[float, str]:
        """
        评分买入信号

        Returns:
            (score, reason_detail)
            score范围: 0.0 - 1.0
        """
        # 1. 基础背离评分
        div_score = self._compute_divergence_score(factors)
        # 2. 结构质量评分
        struct_score = self._compute_structure_score(factors)
        # 3. 量价确认评分
        vol_score = self._compute_volume_score(factors)
        # 4. 市场状态调整
        regime_mod = self._compute_regime_modifier(factors)

        # 5. 乘性组合（各因子独立评分后加权相乘）
        #    避免任何一个因子为0导致整体归零，加一个底值
        base = 0.3  # 底值

        # 背离因子仅对1买有意义（底背驰），2买/3买不应因无背离而受罚
        if div_score > 0 or factors.point_type == '1buy':
            div_part = base + (1 - base) * (div_score ** self.config.divergence_weight)
        else:
            div_part = 1.0  # 非背离类买点，不参与乘性惩罚

        struct_part = base + (1 - base) * (struct_score ** self.config.structure_weight)
        vol_part = base + (1 - base) * (vol_score ** self.config.volume_weight)

        raw_score = div_part * struct_part * vol_part * regime_mod

        # 6. 交互效应奖励：多个因子同时高分
        synergy = 0.0
        high_count = sum([
            div_score > 0.6,
            struct_score > 0.6,
            vol_score > 0.6,
        ])
        if high_count >= 2:
            synergy = self.config.synergy_bonus * high_count
            raw_score += synergy

        # 7. 趋势跟踪修正
        raw_score += factors.trend_modifier

        # 8. 买点类型×市场状态适配
        raw_score += self._point_type_adjustment(factors)

        # 9. RSI因子（借鉴BuySignalScorer）
        rsi_adj = self._compute_rsi_adjustment(factors)
        raw_score += rsi_adj

        # 10. 动量因子（借鉴AAAStrategy：评分上升时加分，下降时扣分）
        if factors.score_delta > 0.10:
            raw_score += 0.08  # 动量向上
        elif factors.score_delta < -0.10:
            raw_score -= 0.05  # 动量向下

        # 钳位
        score = max(0.0, min(1.0, raw_score))

        # 构建原因
        reason_parts = []
        if div_score > 0:
            reason_parts.append(f'背离={div_score:.2f}')
        if struct_score > 0:
            reason_parts.append(f'结构={struct_score:.2f}')
        if vol_score > 0:
            reason_parts.append(f'量价={vol_score:.2f}')
        if synergy > 0:
            reason_parts.append(f'协同+{synergy:.2f}')
        if abs(rsi_adj) > 0.01:
            reason_parts.append(f'RSI={factors.rsi:.0f}')
        if factors.score_delta > 0.10:
            reason_parts.append(f'动量+{factors.score_delta:.2f}')
        elif factors.score_delta < -0.10:
            reason_parts.append(f'动量{factors.score_delta:.2f}')

        reason = ' | '.join(reason_parts) if reason_parts else '综合评分'
        return (score, reason)

    def score_sell_signal(
        self, factors: ScoringFactors
    ) -> Tuple[float, str]:
        """评分卖出信号（简化版，卖出不需要那么多确认）"""
        div_score = self._compute_divergence_score(factors)
        struct_score = self._compute_structure_score(factors)

        # 卖出评分更宽松：背离+结构即可
        base = 0.4
        raw = (base + (1 - base) * div_score) * (base + (1 - base) * struct_score)

        # 市场状态
        if factors.regime_info:
            if factors.regime_info.regime == MarketRegime.STRONG_TREND:
                if factors.regime_info.trend_direction == 'down':
                    raw *= 1.2  # 强下跌中，卖出信号更可信

        score = max(0.0, min(1.0, raw))
        return (score, f'卖出评分: 背离={div_score:.2f}, 结构={struct_score:.2f}')

    def _compute_divergence_score(self, factors: ScoringFactors) -> float:
        """背离评分"""
        score = 0.0

        # 增强背离检测结果（优先）
        if factors.divergence_result and factors.divergence_result.has_divergence:
            base = factors.divergence_result.strength
            # 常规背离最可靠
            if factors.divergence_result.divergence_type == DivergenceType.REGULAR:
                score = base
            elif factors.divergence_result.divergence_type == DivergenceType.MULTI_PERIOD:
                score = base * 1.2  # 多笔背离更强
            elif factors.divergence_result.divergence_type == DivergenceType.HIDDEN:
                score = base * 0.5  # 隐藏背离较弱

            # 动量上下文加分
            if factors.divergence_result.momentum_context == 'decelerating':
                score *= 1.1  # 减速确认背离
        else:
            # 回退到简单背离比率
            if factors.divergence_ratio > 0:
                score = min(factors.divergence_ratio * 1.5, 1.0)

        # 中枢背离加分
        if factors.pivot_divergence_ratio > 0:
            pivot_div_score = min(factors.pivot_divergence_ratio, 1.0)
            score = max(score, score * 0.7 + pivot_div_score * 0.3)

        return min(score, 1.0)

    def _compute_structure_score(self, factors: ScoringFactors) -> float:
        """结构质量评分"""
        score = 0.5  # 默认中等

        # 中枢质量
        if factors.pivot_quality > 0:
            score = factors.pivot_quality

        # 笔强度
        if factors.stroke_strength > 0:
            score = score * 0.6 + factors.stroke_strength * 0.4

        # 中枢笔画数（越多越可靠）
        if factors.strokes_in_pivot > 0:
            stroke_bonus = min(factors.strokes_in_pivot / 7.0, 1.0) * 0.2
            score += stroke_bonus

        return min(score, 1.0)

    def _compute_volume_score(self, factors: ScoringFactors) -> float:
        """量价确认评分"""
        score = 0.5  # 默认中性

        # 量价调整（来自VolumePriceAnalyzer）
        if factors.volume_adjustment != 0:
            # adjustment 范围 -0.15 ~ +0.15，映射到 0.35 ~ 0.65
            score = 0.5 + factors.volume_adjustment * (0.15 / 0.15)

        # 量比
        if factors.volume_ratio > 1.5:
            score = min(score + 0.15, 1.0)  # 明显放量
        elif factors.volume_ratio > 1.2:
            score = min(score + 0.08, 1.0)  # 温和放量
        elif factors.volume_ratio < 0.7:
            score = max(score - 0.1, 0.0)   # 缩量，降低确认

        return min(max(score, 0.0), 1.0)

    def _compute_regime_modifier(self, factors: ScoringFactors) -> float:
        """市场状态修正系数"""
        if not factors.regime_info:
            return 1.0

        regime = factors.regime_info.regime

        if regime == MarketRegime.STRONG_TREND:
            # 强趋势：顺势信号增强，逆势信号减弱
            if factors.regime_info.trend_direction == 'up':
                if factors.point_type in ('2buy', '3buy', 'quasi2buy', 'quasi3buy'):
                    return 1.15  # 顺势买入增强
                elif factors.point_type in ('1buy',):
                    return 0.85  # 1买是逆趋势抄底，风险高
            else:
                return 0.9  # 下跌趋势中买入谨慎

        elif regime == MarketRegime.SIDEWAYS:
            return 1.0  # 震荡市不调整基础分

        elif regime == MarketRegime.VOLATILE:
            return 0.85  # 高波动降低置信度

        elif regime == MarketRegime.MILD_TREND:
            return 1.05  # 温和趋势略增强

        return 1.0

    def _point_type_adjustment(self, factors: ScoringFactors) -> float:
        """买点类型×市场状态适配调整"""
        if not factors.regime_info or not factors.point_type:
            return 0.0

        regime = factors.regime_info.regime
        pt = factors.point_type
        adj = 0.0

        if regime == MarketRegime.STRONG_TREND:
            if pt == '1buy':
                adj = self.config.trend_1buy_bonus
        elif regime == MarketRegime.SIDEWAYS:
            if pt in ('2buy', 'quasi2buy'):
                adj = self.config.sideways_2buy_bonus
            elif pt in ('3buy', 'quasi3buy'):
                adj = self.config.sideways_3buy_bonus

        return adj

    def _compute_rsi_adjustment(self, factors: ScoringFactors) -> float:
        """RSI因子调整（借鉴BuySignalScorer）"""
        rsi = factors.rsi

        if 30 <= rsi <= 50:
            return 0.10   # 超卖反弹区，买入信号更可靠
        elif 50 < rsi <= 70:
            return 0.05   # 强势区
        elif rsi < 30:
            return 0.03   # 深度超卖，需要其他确认
        elif rsi > 80:
            return -0.15  # 超买，风险高
        elif rsi > 70:
            return -0.05  # 偏高，谨慎

        return 0.0
