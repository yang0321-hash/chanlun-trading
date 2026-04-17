"""
投资委员会评分模块 — 权重、评分函数、决策规则
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 权重配置
# ============================================================

DEFAULT_WEIGHTS = {
    'technical_bull': 0.30,
    'technical_bear': 0.20,
    'sentiment': 0.15,
    'sector_rotation': 0.15,
    'scanner_base': 0.10,
    'risk_adjustment': 0.10,
}


# ============================================================
# 决策阈值
# ============================================================

DECISION_THRESHOLDS = {
    'buy_strong': 70,       # composite >= 70 AND risk <= 0.6 → buy
    'buy_cautious': 55,     # composite >= 55 AND risk <= 0.4 → buy
    'reject_score': 30,     # composite < 30 → reject
    'reject_risk': 0.80,    # risk >= 0.8 → reject
    'veto_bear_gap': 0.30,  # bear_conf > bull_conf + 0.3 → reject
}

VETO_RULES = {
    'extreme_risk': 0.75,           # risk_score >= 0.75 → 一票否决
    'max_sector_positions': 3,      # 同行业已有3+持仓 → 否决
}


# ============================================================
# 评分函数
# ============================================================

def normalize(x: float, lo: float, hi: float) -> float:
    """将x从[lo, hi]映射到[-1, +1]，超出范围截断"""
    if hi == lo:
        return 0.0
    clipped = max(lo, min(hi, x))
    return 2.0 * (clipped - lo) / (hi - lo) - 1.0


def calc_composite_score(
    bull_confidence: float,
    bear_confidence: float,
    sentiment_score: float,
    sector_score: float,
    scanner_score: float,
    risk_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    计算综合评分 (0-100)

    公式:
        raw = W_bull * bull + W_sent * max(0,sent) + W_sector * sector
              + W_scanner * (scan/100) - W_bear * bear
        归一化到 [0, 1] 后乘 100，再乘风险调整
    """
    w = weights or DEFAULT_WEIGHTS

    # 正向分 (最大约 0.30 + 0.15 + 0.15 + 0.10 = 0.70)
    positive = (
        w['technical_bull'] * bull_confidence
        + w['sentiment'] * max(0, sentiment_score)
        + w['sector_rotation'] * sector_score
        + w['scanner_base'] * (scanner_score / 100.0)
    )

    # 负向分 (最大约 0.20)
    negative = w['technical_bear'] * bear_confidence

    # 净得分归一化: positive 范围 [0, ~0.70], negative [0, ~0.20]
    max_positive = (w['technical_bull'] + w['sentiment'] + w['sector_rotation']
                    + w['scanner_base'])
    max_negative = w['technical_bear']

    # 归一化到 [0, 1]: (positive - negative) / max_possible
    raw = (positive - negative) / (max_positive - max_negative) if (max_positive - max_negative) > 0 else 0
    raw = max(0, min(1, raw))

    # 风险惩罚
    risk_penalty = w['risk_adjustment'] * risk_score
    composite = raw * (1.0 - risk_penalty) * 100

    return max(0, min(100, composite))


def make_decision(
    composite_score: float,
    bull_confidence: float,
    bear_confidence: float,
    risk_score: float,
    sector_position_count: int = 0,
) -> Tuple[str, float]:
    """
    根据评分做出决策

    Returns: (decision, confidence)
        decision: 'buy', 'hold', 'reject'
        confidence: 0-1
    """
    t = DECISION_THRESHOLDS
    v = VETO_RULES

    # 一票否决
    if risk_score >= v['extreme_risk']:
        return 'reject', 0.9

    if sector_position_count >= v['max_sector_positions']:
        return 'reject', 0.85

    if bear_confidence > bull_confidence + t['veto_bear_gap']:
        return 'reject', 0.7

    # 评分否决
    if composite_score < t['reject_score'] or risk_score >= t['reject_risk']:
        return 'reject', 0.6

    # 买入
    if composite_score >= t['buy_strong'] and risk_score <= 0.6:
        confidence = min(0.95, 0.5 + (composite_score - t['buy_strong']) / 60)
        return 'buy', confidence

    if composite_score >= t['buy_cautious'] and risk_score <= 0.4:
        confidence = min(0.85, 0.4 + (composite_score - t['buy_cautious']) / 80)
        return 'buy', confidence

    # 观望
    return 'hold', 0.5


# ============================================================
# 风险等级
# ============================================================

def classify_risk(risk_score: float) -> str:
    """风险等级分类"""
    if risk_score < 0.25:
        return 'LOW'
    elif risk_score < 0.50:
        return 'MEDIUM'
    elif risk_score < 0.75:
        return 'HIGH'
    else:
        return 'EXTREME'


# ============================================================
# 仓位计算
# ============================================================

def calc_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    max_position_pct: float = 0.15,
    risk_per_trade: float = 0.02,
) -> Tuple[int, float]:
    """
    计算建议买入股数和仓位比例

    基于固定风险比例法: 每笔交易最大亏损 = capital * risk_per_trade
    股数 = 最大亏损额 / (entry - stop)

    Returns: (shares_in_lots, position_pct)
    """
    risk_amount = capital * risk_per_trade
    per_share_risk = entry_price - stop_loss

    if per_share_risk <= 0:
        return 0, 0.0

    shares = int(risk_amount / per_share_risk)
    # 取整到100股
    shares = (shares // 100) * 100

    position_value = shares * entry_price
    position_pct = position_value / capital if capital > 0 else 0

    # 不超过最大仓位限制
    if position_pct > max_position_pct:
        max_value = capital * max_position_pct
        shares = int(max_value / entry_price / 100) * 100
        position_value = shares * entry_price
        position_pct = position_value / capital if capital > 0 else 0

    return shares, position_pct
