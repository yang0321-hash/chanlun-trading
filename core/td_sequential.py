"""
TD Sequential (DeMark) — 缠论买卖点辅助确认

TD Setup: 连续9根 close < close[4] (Buy) 或 close > close[4] (Sell)
TD Countdown: Setup完成后, close ≤ low[2] (Buy CD) 或 close ≥ high[2] (Sell CD), 数到13

集成方式:
  - 买点附近出TD Buy Setup(8+) → confidence加分
  - 买点附近出TD Buy Countdown(13) → 强加分
  - 不同买点类型的TD适用性不同，按回测数据加权
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class TDPhase(Enum):
    NONE = "none"
    SETUP_BUY = "setup_buy"
    SETUP_SELL = "setup_sell"
    COUNTDOWN_BUY = "cd_buy"
    COUNTDOWN_SELL = "cd_sell"


@dataclass
class TDReading:
    """某根K线的TD读数"""
    index: int
    setup_buy: int = 0
    setup_sell: int = 0
    setup_buy_completed: bool = False
    setup_sell_completed: bool = False
    cd_buy: int = 0
    cd_sell: int = 0
    cd_buy_completed: bool = False
    cd_sell_completed: bool = False
    phase: TDPhase = TDPhase.NONE

    def buy_score(self) -> float:
        """综合买入信号强度 0-1"""
        s = 0.0
        if self.setup_buy >= 8:
            s = max(s, (self.setup_buy - 7) * 0.25)  # 8→0.25, 9→0.50
        if self.cd_buy >= 8:
            s = max(s, 0.3 + (self.cd_buy - 7) * 0.0875)  # 8→0.39, 13→0.74
        if self.cd_buy >= 13:
            s = max(s, 0.85)
        if self.cd_buy_completed:
            s = max(s, 1.0)
        return min(s, 1.0)

    def sell_score(self) -> float:
        """综合卖出信号强度 0-1"""
        s = 0.0
        if self.setup_sell >= 8:
            s = max(s, (self.setup_sell - 7) * 0.25)
        if self.cd_sell >= 8:
            s = max(s, 0.3 + (self.cd_sell - 7) * 0.0875)
        if self.cd_sell >= 13:
            s = max(s, 0.85)
        if self.cd_sell_completed:
            s = max(s, 1.0)
        return min(s, 1.0)


@dataclass
class TDResult:
    """TD Sequential完整分析结果"""
    readings: List[TDReading] = field(default_factory=list)
    last_buy_setup: Optional[TDReading] = None
    last_sell_setup: Optional[TDReading] = None
    last_buy_cd: Optional[TDReading] = None
    last_sell_cd: Optional[TDReading] = None

    def get_latest(self) -> Optional[TDReading]:
        return self.readings[-1] if self.readings else None

    def get_at(self, bar_index: int) -> Optional[TDReading]:
        """获取指定bar的TD读数"""
        for r in reversed(self.readings):
            if r.index <= bar_index:
                return r
        return None

    def get_buy_score_at(self, bar_index: int, lookback: int = 8) -> float:
        """获取指定bar附近最强的TD买入信号"""
        best = 0.0
        for r in reversed(self.readings):
            dist = bar_index - r.index
            if dist > lookback:
                break
            if dist >= 0:
                best = max(best, r.buy_score())
                if best >= 1.0:
                    break
        return best

    def get_sell_score_at(self, bar_index: int, lookback: int = 8) -> float:
        """获取指定bar附近最强的TD卖出信号"""
        best = 0.0
        for r in reversed(self.readings):
            dist = bar_index - r.index
            if dist > lookback:
                break
            if dist >= 0:
                best = max(best, r.sell_score())
                if best >= 1.0:
                    break
        return best


def analyze_td(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    setup_len: int = 9,
    cd_len: int = 13,
    lookback: int = 4,
) -> TDResult:
    """
    计算TD Sequential

    Args:
        highs/lows/closes: K线最高/最低/收盘价序列
        setup_len: Setup连续数 (标准=9)
        cd_len: Countdown目标数 (标准=13)
        lookback: 对比前N根收盘 (标准=4)
    """
    n = len(closes)
    result = TDResult()
    if n < lookback + 1:
        return result

    setup_buy_count = 0
    setup_sell_count = 0
    cd_buy_count = 0
    cd_sell_count = 0
    in_buy_cd = False
    in_sell_cd = False

    for i in range(lookback, n):
        close = closes[i]
        close4 = closes[i - lookback]
        low2 = lows[i - 2] if i >= 2 else close
        high2 = highs[i - 2] if i >= 2 else close

        reading = TDReading(index=i)

        # Setup计数
        is_buy_setup = close < close4
        is_sell_setup = close > close4

        if is_buy_setup and not is_sell_setup:
            setup_buy_count += 1
            if setup_sell_count > 0 and setup_sell_count < setup_len:
                setup_sell_count = 0
                in_sell_cd = False
                cd_sell_count = 0
            reading.setup_buy = min(setup_buy_count, setup_len)
            if setup_buy_count == setup_len:
                reading.setup_buy_completed = True
                in_buy_cd = True
                cd_buy_count = 0
                setup_buy_count = 0  # Reset for recycling
        elif is_sell_setup and not is_buy_setup:
            setup_sell_count += 1
            if setup_buy_count > 0 and setup_buy_count < setup_len:
                setup_buy_count = 0
                in_buy_cd = False
                cd_buy_count = 0
            reading.setup_sell = min(setup_sell_count, setup_len)
            if setup_sell_count == setup_len:
                reading.setup_sell_completed = True
                in_sell_cd = True
                cd_sell_count = 0
                setup_sell_count = 0  # Reset for recycling
        else:
            if setup_buy_count > 0 and setup_buy_count < setup_len:
                setup_buy_count = 0
            if setup_sell_count > 0 and setup_sell_count < setup_len:
                setup_sell_count = 0

        # Countdown计数
        if in_buy_cd and i >= 2:
            if close <= low2:
                cd_buy_count += 1
                reading.cd_buy = cd_buy_count
                reading.phase = TDPhase.COUNTDOWN_BUY
                if cd_buy_count == cd_len:
                    reading.cd_buy_completed = True
                    in_buy_cd = False
                    cd_buy_count = 0
                    result.last_buy_cd = reading

        if in_sell_cd and i >= 2:
            if close >= high2:
                cd_sell_count += 1
                reading.cd_sell = cd_sell_count
                reading.phase = TDPhase.COUNTDOWN_SELL
                if cd_sell_count == cd_len:
                    reading.cd_sell_completed = True
                    in_sell_cd = False
                    cd_sell_count = 0
                    result.last_sell_cd = reading

        if reading.phase == TDPhase.NONE:
            if reading.setup_buy > 0:
                reading.phase = TDPhase.SETUP_BUY
            elif reading.setup_sell > 0:
                reading.phase = TDPhase.SETUP_SELL

        if reading.setup_buy_completed:
            result.last_buy_setup = reading
        if reading.setup_sell_completed:
            result.last_sell_setup = reading

        result.readings.append(reading)

    return result


def td_confirm_buy(
    td_result: TDResult,
    bar_index: int,
    buy_type: str,
) -> Tuple[float, str]:
    """
    用TD结果确认缠论买点

    Args:
        td_result: TD分析结果
        bar_index: 买点在K线序列中的位置
        buy_type: 买点类型 (1buy/2buy/3buy等)

    Returns:
        (confidence_boost, detail_str)
        boost范围: -0.10 ~ +0.15
    """
    reading = td_result.get_at(bar_index)
    if not reading:
        return 0.0, '无TD数据'

    boost = 0.0
    details = []

    # 买点附近的TD信号
    nearby_score = td_result.get_buy_score_at(bar_index, lookback=8)

    # TD Setup加分
    if reading.setup_buy >= 9:
        boost += 0.10
        details.append(f"BuySetup={reading.setup_buy}(完成)")
    elif reading.setup_buy >= 8:
        boost += 0.06
        details.append(f"BuySetup={reading.setup_buy}")
    elif reading.setup_buy >= 7:
        boost += 0.03
        details.append(f"BuySetup={reading.setup_buy}")

    # TD Countdown加分
    if reading.cd_buy >= 13:
        boost += 0.15
        details.append("BuyCD=13(反转)")
    elif reading.cd_buy >= 10:
        boost += 0.10
        details.append(f"BuyCD={reading.cd_buy}")
    elif reading.cd_buy >= 8:
        boost += 0.06
        details.append(f"BuyCD={reading.cd_buy}")

    # 附近信号
    if nearby_score > 0.5:
        boost += 0.03
        details.append(f"近8日TD={nearby_score:.2f}")

    # 反向信号(卖点TD活跃 → 减分)
    sell_score = td_result.get_sell_score_at(bar_index, lookback=8)
    if sell_score > 0.5:
        boost -= 0.05
        details.append(f"SellTD={sell_score:.2f}")

    # 不同买点类型的TD适用性 (回测验证)
    type_mult = {
        '1buy': 0.8, '2buy': 1.3, '3buy': 1.3,
        'quasi2buy': 1.0, 'sub1buy': 0.5,
        'quasi3buy': 0.8, '2b3bbuy': 0.8,
    }.get(buy_type, 1.0)
    boost *= type_mult

    boost = round(max(-0.10, min(0.15, boost)), 3)
    detail = " | ".join(details) if details else "无TD信号"
    return boost, detail


def td_confirm_sell(
    td_result: 'TDResult',
    bar_index: int,
    sell_type: str,
) -> Tuple[float, str]:
    """用TD结果确认缠论卖点 (v7.3: 卖点必须TD确认, 77% vs 47%)

    Args:
        td_result: TD分析结果
        bar_index: 卖点在K线序列中的位置
        sell_type: 卖点类型 (1sell/2sell/3sell)

    Returns:
        (confidence_boost, detail_str)
        boost范围: -0.10 ~ +0.20
    """
    reading = td_result.get_at(bar_index)
    if not reading:
        return 0.0, '无TD数据'

    boost = 0.0
    details = []

    # TD Sell Setup
    if reading.setup_sell >= 9:
        boost += 0.12
        details.append(f"SellSetup={reading.setup_sell}(完成)")
    elif reading.setup_sell >= 8:
        boost += 0.07
        details.append(f"SellSetup={reading.setup_sell}")

    # TD Sell Countdown
    if reading.cd_sell >= 13:
        boost += 0.20
        details.append("SellCD=13(反转)")
    elif reading.cd_sell >= 10:
        boost += 0.12
        details.append(f"SellCD={reading.cd_sell}")
    elif reading.cd_sell >= 8:
        boost += 0.07
        details.append(f"SellCD={reading.cd_sell}")

    # 附近卖点信号
    nearby_score = td_result.get_sell_score_at(bar_index, lookback=8)
    if nearby_score > 0.5:
        boost += 0.05
        details.append(f"近8日SellTD={nearby_score:.2f}")

    # 反向信号(买点TD活跃 → 减分)
    buy_score = td_result.get_buy_score_at(bar_index, lookback=8)
    if buy_score > 0.5:
        boost -= 0.05
        details.append(f"BuyTD={buy_score:.2f}")

    # 2sell/3sell更可靠
    type_mult = {'1sell': 0.8, '2sell': 1.3, '3sell': 1.3}.get(sell_type, 1.0)
    boost *= type_mult

    boost = round(max(-0.10, min(0.20, boost)), 3)
    detail = " | ".join(details) if details else "无TD卖出信号"
    return boost, detail
