"""
委员会辩论机制 — Bull/Bear分析后互相反驳

Phase A+ 输入: Bull/Bear初始AgentArgument
Phase A+ 输出: DebateAdjustment (调整分 ±0.10)

纯规则驱动，无LLM调用，零延迟。
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from agents.committee_agents import AgentArgument, CommitteeContext


@dataclass
class DebateAdjustment:
    bull_rebuttal: AgentArgument
    bear_rebuttal: AgentArgument
    score_adjustment: float  # [-0.10, +0.10]
    debate_summary: str


# 辩论知识库 — 多方反驳空方的模板
BULL_REBUTTALS = {
    '高位风险': [
        lambda ctx: ('3买突破中枢，高位是正常突破形态' if ctx.chanlun and ctx.chanlun.buy_type == '3buy' else None),
        lambda ctx: ('价格突破后回踩确认，非高位风险' if ctx.chanlun and ctx.chanlun.price_vs_pivot == 'above' else None),
    ],
    '背离': [
        lambda ctx: ('MACD背离可被后续放量修复' if ctx.chanlun else None),
        lambda ctx: ('周线多头趋势中，日线背离信号权重降低' if ctx.chanlun and ctx.chanlun.weekly_trend == 'bull' else None),
    ],
    '成交量萎缩': [
        lambda ctx: ('缩量回调是健康调整，抛压衰竭' if True else None),
    ],
    '向下笔': [
        lambda ctx: ('向下笔是回调买点，非趋势反转' if ctx.chanlun and ctx.chanlun.weekly_trend == 'bull' else None),
    ],
    '中枢压力': [
        lambda ctx: ('突破中枢上沿后回踩确认，压力变支撑' if ctx.chanlun and ctx.chanlun.price_vs_pivot == 'above' else None),
    ],
}

# 空方反驳多方的模板
BEAR_REBUTTALS = {
    '买点': [
        lambda ctx: ('2买位置可能形成复杂中枢，需进一步确认' if True else None),
        lambda ctx: ('买点强度偏弱，信号可靠性存疑' if ctx.chanlun and ctx.chanlun.buy_strength in ('weak', '') else None),
    ],
    '突破': [
        lambda ctx: ('突破可能失败，需要成交量持续确认' if True else None),
    ],
    '放量': [
        lambda ctx: ('放量可能是主力出货，需观察后续走势' if True else None),
    ],
    '均线': [
        lambda ctx: ('短期均线金叉可能反复，需等MA20确认' if True else None),
    ],
    '中枢': [
        lambda ctx: ('仍在中枢内震荡，方向不明确' if ctx.chanlun and ctx.chanlun.price_vs_pivot == 'inside' else None),
    ],
    '趋势': [
        lambda ctx: ('周线仍为空头趋势，日线反弹可能是技术性反弹' if ctx.chanlun and ctx.chanlun.weekly_trend == 'bear' else None),
    ],
}


class CommitteeDebateRound:
    """委员会辩论 — 复用Bull/Bear初始分析，互相反驳"""

    def __init__(self, max_rounds: int = 1):
        self.max_rounds = max_rounds

    def run(
        self,
        ctx: CommitteeContext,
        bull_initial: AgentArgument,
        bear_initial: AgentArgument,
    ) -> DebateAdjustment:
        bull_rebuttal = self._rebut_bear(ctx, bull_initial, bear_initial)
        bear_rebuttal = self._rebut_bull(ctx, bear_initial, bull_initial)

        adjustment = self._evaluate_arguments(bull_rebuttal, bear_rebuttal, ctx)

        summary = self._build_summary(bull_rebuttal, bear_rebuttal, adjustment)

        return DebateAdjustment(
            bull_rebuttal=bull_rebuttal,
            bear_rebuttal=bear_rebuttal,
            score_adjustment=adjustment,
            debate_summary=summary,
        )

    def _rebut_bear(
        self,
        ctx: CommitteeContext,
        bull_initial: AgentArgument,
        bear_initial: AgentArgument,
    ) -> AgentArgument:
        """多方反驳空方观点"""
        rebuttals = []
        for point in bear_initial.key_points:
            matched = False
            for keyword, templates in BULL_REBUTTALS.items():
                if keyword in point:
                    for tpl in templates:
                        r = tpl(ctx)
                        if r:
                            rebuttals.append(r)
                            matched = True
                            break
                    break
            if not matched:
                rebuttals.append(f'空方观点"{point[:20]}"可被市场消化')

        reasoning = ' | '.join(rebuttals) if rebuttals else '无明显反驳点'
        success_rate = len(rebuttals) / max(len(bear_initial.key_points), 1)

        return AgentArgument(
            agent_name='BullDebate',
            stance='bull',
            reasoning=reasoning,
            confidence=bull_initial.confidence * (0.8 + 0.2 * success_rate),
            key_points=rebuttals[:5],
            data_references={'rebutted_points': len(rebuttals), 'total_points': len(bear_initial.key_points)},
        )

    def _rebut_bull(
        self,
        ctx: CommitteeContext,
        bear_initial: AgentArgument,
        bull_initial: AgentArgument,
    ) -> AgentArgument:
        """空方反驳多方观点"""
        rebuttals = []
        for point in bull_initial.key_points:
            matched = False
            for keyword, templates in BEAR_REBUTTALS.items():
                if keyword in point:
                    for tpl in templates:
                        r = tpl(ctx)
                        if r:
                            rebuttals.append(r)
                            matched = True
                            break
                    break
            if not matched:
                rebuttals.append(f'多方观点"{point[:20]}"存在不确定性')

        reasoning = ' | '.join(rebuttals) if rebuttals else '无明显反驳点'
        success_rate = len(rebuttals) / max(len(bull_initial.key_points), 1)

        return AgentArgument(
            agent_name='BearDebate',
            stance='bear',
            reasoning=reasoning,
            confidence=bear_initial.confidence * (0.8 + 0.2 * success_rate),
            key_points=rebuttals[:5],
            data_references={'rebutted_points': len(rebuttals), 'total_points': len(bull_initial.key_points)},
        )

    def _evaluate_arguments(
        self,
        bull_rebuttal: AgentArgument,
        bear_rebuttal: AgentArgument,
        ctx: CommitteeContext,
    ) -> float:
        """计算辩论调整分 [-0.10, +0.10]"""
        bull_success = bull_rebuttal.data_references.get('rebutted_points', 0)
        bull_total = max(bull_rebuttal.data_references.get('total_points', 1), 1)
        bear_success = bear_rebuttal.data_references.get('rebutted_points', 0)
        bear_total = max(bear_rebuttal.data_references.get('total_points', 1), 1)

        bull_rate = bull_success / bull_total
        bear_rate = bear_success / bear_total

        # 多方反驳成功率高 → 正调整，反之亦然
        diff = bull_rate - bear_rate

        # 乘以初始置信度差
        conf_diff = bull_rebuttal.confidence - bear_rebuttal.confidence
        combined = diff * 0.7 + conf_diff * 0.3

        return max(-0.10, min(0.10, combined * 0.15))

    def _build_summary(
        self,
        bull_rebuttal: AgentArgument,
        bear_rebuttal: AgentArgument,
        adjustment: float,
    ) -> str:
        bull_n = bull_rebuttal.data_references.get('rebutted_points', 0)
        bear_n = bear_rebuttal.data_references.get('rebutted_points', 0)
        direction = '多方略优' if adjustment > 0.02 else ('空方略优' if adjustment < -0.02 else '势均力敌')
        return f'辩论结果({direction}): 多方反驳{bull_n}点, 空方反驳{bear_n}点, 调整{adjustment:+.3f}'
