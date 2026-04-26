"""缠论交易系统 v2.0 — 核心规则引擎

大盘评分 → 仓位上限 → 买点限制 → 止盈止损 的完整规则链。

使用:
    from strategies.trading_rules import TradingRules
    rules = TradingRules()
    score = rules.calc_market_score(closes, volumes)
    rules.get_max_position(score, sector_tier=1)
    rules.is_buy_allowed('3buy', score)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ==================== 仓位矩阵 ====================

# (market_score_range, sector_tier) → max_position_pct
POSITION_MATRIX = {
    # 强势 (9-12)
    (9, 1): 80, (10, 1): 80, (11, 1): 80, (12, 1): 80,
    (9, 2): 60, (10, 2): 60, (11, 2): 60, (12, 2): 60,
    (9, 3): 0,
    # 偏强 (6-8)
    (6, 1): 60, (7, 1): 60, (8, 1): 60,
    (6, 2): 40, (7, 2): 40, (8, 2): 40,
    (6, 3): 0, (7, 3): 0, (8, 3): 0,
    # 偏弱 (3-5)
    (3, 1): 40, (4, 1): 40, (5, 1): 40,
    (3, 2): 20, (4, 2): 20, (5, 2): 20,
    (3, 3): 0, (4, 3): 0, (5, 3): 0,
    # 弱势 (0-2)
    (0, 1): 20, (1, 1): 20, (2, 1): 20,
    (0, 2): 0, (1, 2): 0, (2, 2): 0,
    (0, 3): 0, (1, 3): 0, (2, 3): 0,
}

SINGLE_STOCK_MAX = {
    'strong': 30,    # 9-12
    'moderate': 25,  # 6-8
    'weak': 20,      # 3-5
    'very_weak': 10, # 0-2
}


# ==================== 动态止盈档位 ====================

TAKE_PROFIT_TIERS = [
    # (gain_low, gain_high, trailing_stop_pct, action)
    (0.00, 0.05, 0.000, 'hold'),       # 保本
    (0.05, 0.10, 0.010, 'hold'),       # +1%止损
    (0.10, 0.15, 0.030, 'sell_1/3'),   # +3%止损, 卖1/3
    (0.15, 0.25, 0.080, 'sell_1/3'),   # +8%止损, 再卖1/3
    (0.25, 9.99, 0.150, 'hold_rest'),  # +15%止损, 持余仓
]


@dataclass
class MarketScoreResult:
    """大盘评分结果"""
    score: int = 0
    factors: Dict[str, int] = None
    state: str = 'unknown'
    max_total_position: int = 0
    single_stock_max: int = 0
    allowed_buy_types: List[str] = None

    def __post_init__(self):
        if self.factors is None:
            self.factors = {}
        if self.allowed_buy_types is None:
            self.allowed_buy_types = []


class TradingRules:
    """缠论交易系统 v2.0 规则引擎"""

    @staticmethod
    def calc_market_score(closes: np.ndarray,
                          volumes: Optional[np.ndarray] = None
                          ) -> MarketScoreResult:
        """计算大盘6因子12分评分

        Args:
            closes: 收盘价序列 (最近至少25日)
            volumes: 成交量/成交额序列 (可选)
        """
        factors = {}
        n = len(closes)
        if n < 25:
            return MarketScoreResult(score=0, factors={}, state='数据不足',
                                     allowed_buy_types=['1buy'])

        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        last = closes[-1]

        # 因子1: 价格 vs MA5
        pct_ma5 = (last - ma5) / ma5 * 100
        if pct_ma5 > 5:
            factors['price_vs_ma5'] = 2
        elif pct_ma5 >= 0:
            factors['price_vs_ma5'] = 1
        else:
            factors['price_vs_ma5'] = 0

        # 因子2: MA5 vs MA10
        pct_ma5_10 = (ma5 - ma10) / ma10 * 100
        if pct_ma5_10 > 1:
            factors['ma5_vs_ma10'] = 2
        elif pct_ma5_10 >= 0:
            factors['ma5_vs_ma10'] = 1
        else:
            factors['ma5_vs_ma10'] = 0

        # 因子3: 价格 vs MA20
        pct_ma20 = (last - ma20) / ma20 * 100
        if pct_ma20 > 5:
            factors['price_vs_ma20'] = 2
        elif pct_ma20 >= 0:
            factors['price_vs_ma20'] = 1
        else:
            factors['price_vs_ma20'] = 0

        # 因子4: MACD DIF 方向+位置
        try:
            ema12 = _ema(closes, 12)
            ema26 = _ema(closes, 26)
            dif = ema12 - ema26
            dea = _ema(dif, 9)
            if dif[-1] > 0 and dif[-1] > dif[-2]:
                factors['macd_dif'] = 2
            elif dif[-1] > 0:
                factors['macd_dif'] = 1
            else:
                factors['macd_dif'] = 0
        except Exception:
            factors['macd_dif'] = 1

        # 因子5: 成交量变化
        if volumes is not None and len(volumes) >= 2:
            vol_chg = (volumes[-1] - volumes[-2]) / max(volumes[-2], 1) * 100
            if vol_chg > 20:
                factors['volume'] = 2
            elif vol_chg > -20:
                factors['volume'] = 1
            else:
                factors['volume'] = 0
        else:
            factors['volume'] = 1  # 无数据时默认中等

        # 因子6: MA5趋势方向 (连续3日)
        ma5_series = np.convolve(closes, np.ones(5) / 5, mode='valid')
        if len(ma5_series) >= 3:
            last3 = ma5_series[-3:]
            if last3[2] > last3[1] > last3[0]:
                factors['ma5_trend'] = 2
            elif last3[2] < last3[1] < last3[0]:
                factors['ma5_trend'] = 0
            else:
                factors['ma5_trend'] = 1
        else:
            factors['ma5_trend'] = 1

        score = sum(factors.values())

        # 状态判定
        if score >= 9:
            state = 'strong'
            allowed = ['1buy', '2buy', '3buy', 'sub1buy']
            max_pos = 80
            single_max = 30
        elif score >= 6:
            state = 'moderate'
            allowed = ['2buy', '3buy', 'sub1buy']
            max_pos = 60
            single_max = 25
        elif score >= 3:
            state = 'weak'
            allowed = ['1buy', 'sub1buy']
            max_pos = 40
            single_max = 20
        else:
            state = 'very_weak'
            allowed = []
            max_pos = 20
            single_max = 10

        return MarketScoreResult(
            score=score,
            factors=factors,
            state=state,
            max_total_position=max_pos,
            single_stock_max=single_max,
            allowed_buy_types=allowed,
        )

    @staticmethod
    def get_max_position(score: int, sector_tier: int) -> int:
        """仓位矩阵: 大盘评分 × 板块层级 → 最大仓位%

        sector_tier: 1=主线 2=辅线 3=重灾区
        """
        if sector_tier == 3:
            return 0  # 重灾区禁止
        return POSITION_MATRIX.get((score, sector_tier), 20)

    @staticmethod
    def is_buy_allowed(buy_type: str, score: int,
                       sector_tier: int = 1) -> Tuple[bool, str]:
        """买点是否允许

        Returns:
            (allowed, reason)
        """
        if sector_tier == 3:
            return False, '重灾区板块禁止开仓'

        if score >= 9:
            return True, '强势市，所有买点可用'
        elif score >= 6:
            if buy_type == '1buy':
                return True, '偏强市可做1买'
            return True, '偏强市可做2买/3买'
        elif score >= 3:
            if buy_type in ('2buy', '3buy'):
                return False, '偏弱市只做1买'
            return True, '偏弱市可做1买(轻仓)'
        else:
            if buy_type == '3buy' and sector_tier == 1:
                return True, '弱势市可做主线3买(轻仓)'
            return False, '弱势市不开新仓'

    @staticmethod
    def is_weak_market(score: int) -> bool:
        return score <= 4

    @staticmethod
    def get_stop_loss_pct(buy_type: str, score: int) -> float:
        """获取止损百分比 (返回负数)"""
        if TradingRules.is_weak_market(score):
            return -0.04  # 弱势市更严格
        if buy_type == '1buy':
            return -0.05
        elif buy_type == '2buy':
            return -0.03 if score >= 6 else -0.02
        elif buy_type == '3buy':
            return -0.03
        return -0.05

    @staticmethod
    def get_trailing_stop(entry_price: float, current_price: float,
                          buy_type: str = '2buy') -> Tuple[float, str]:
        """根据浮盈比例返回跟踪止损价和动作

        Returns:
            (stop_price, action)
        """
        if entry_price <= 0:
            return entry_price, 'hold'

        gain = (current_price - entry_price) / entry_price

        for gain_low, gain_high, trail_pct, action in TAKE_PROFIT_TIERS:
            if gain_low <= gain < gain_high:
                stop_price = entry_price * (1 + trail_pct)
                return stop_price, action

        return entry_price, 'hold'

    @staticmethod
    def get_weak_market_rules(score: int) -> dict:
        """弱势市规则集"""
        is_weak = TradingRules.is_weak_market(score)
        return {
            'is_weak': is_weak,
            'allowed_buy': ['1buy'] if is_weak else ['1buy', '2buy', '3buy'],
            'max_position': 10 if is_weak else 50,
            'stop_pct': 0.04 if is_weak else 0.05,
            'time_stop_days': 3 if is_weak else 60,
            'can_add': not is_weak,
            'reentry_lock_hours': 24 if is_weak else 0,
        }

    @staticmethod
    def check_intraday_risk(market_open_pct: float, market_ma5_broken: bool,
                            sector_open_pct: float = 0
                            ) -> Tuple[str, float]:
        """盘中风险信号检查

        Args:
            market_open_pct: 大盘开盘涨跌幅%
            market_ma5_broken: 大盘是否跌破MA5
            sector_open_pct: 持仓板块开盘涨跌幅%

        Returns:
            (action, reduce_ratio)  e.g. ('reduce', 0.33)
        """
        if market_open_pct < -1.5:
            return 'market_risk', 0.33
        if market_ma5_broken and market_open_pct < -0.5:
            return 'market_weak', 0.50
        if sector_open_pct < -1.5:
            return 'sector_risk', 0.50
        return 'normal', 0.0

    @staticmethod
    def format_score_report(result: MarketScoreResult) -> str:
        """格式化评分报告"""
        state_cn = {
            'strong': '强势', 'moderate': '偏强/震荡',
            'weak': '偏弱', 'very_weak': '弱势/空头',
        }
        lines = [
            f'大盘评分: {result.score}/12 '
            f'({state_cn.get(result.state, result.state)})',
            f'  仓位上限: {result.max_total_position}%  '
            f'单票上限: {result.single_stock_max}%',
            f'  可用买点: {", ".join(result.allowed_buy_types) or "无"}',
        ]
        for name, val in result.factors.items():
            lines.append(f'    {name}: {val}')
        return '\n'.join(lines)


# ==================== 辅助函数 ====================

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均"""
    if len(data) < period:
        return np.full_like(data, data[0] if len(data) > 0 else 0)
    result = np.empty_like(data)
    result[:period] = np.mean(data[:period])
    k = 2 / (period + 1)
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result
