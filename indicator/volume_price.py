"""
量价联合分析模块

将成交量分析与价格位置、缠论笔级别结合，提供：
1. 价格位置分析（当前价格在历史中的分位）
2. 缩量→放量转折检测
3. 缠论笔级别的量价背离检测
4. 换手率分析
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class PricePosition:
    """价格位置分析结果"""
    percentile: float = 50.0       # 0~100，当前价在历史中的分位
    is_low: bool = False           # 分位 < 20
    is_high: bool = False          # 分位 > 80
    is_mid_low: bool = False       # 分位 < 40
    ma_bias: float = 0.0           # 相对长期均线的偏离度 (%)


@dataclass
class VolumeTransition:
    """缩量→放量转折检测结果"""
    has_transition: bool = False    # 是否检测到转折
    direction: str = ''            # 'shrink_to_expand' 或 'expand_to_shrink'
    strength: float = 0.0          # 转折强度 0~1
    shrink_avg_ratio: float = 0.0  # 缩量段平均量比
    expand_avg_ratio: float = 0.0  # 放量段平均量比


@dataclass
class StrokeDivergence:
    """笔级别的量价背离"""
    has_divergence: bool = False
    direction: str = ''            # 'bottom' 或 'top'
    strength: float = 0.0          # 背离强度 0~1
    prev_avg_volume: float = 0.0   # 前一笔均量
    curr_avg_volume: float = 0.0   # 当前笔均量
    prev_extreme: float = 0.0      # 前一笔极值价格
    curr_extreme: float = 0.0      # 当前笔极值价格


@dataclass
class TurnoverResult:
    """换手率分析结果"""
    current: float = 0.0           # 当前换手率（或相对换手率）
    avg: float = 0.0               # N日均换手率
    ratio_vs_avg: float = 1.0      # 当前/均值的比值
    trend: str = 'neutral'         # up, down, neutral
    is_low_phase: bool = False     # 是否处于低换手率阶段


class PricePositionAnalyzer:
    """
    价格位置分析器

    判断当前价格在历史区间中的位置（高位/低位/中间），
    用于辅助判断"低位吸筹"等场景。
    """

    def __init__(self, prices: np.ndarray, ma_period: int = 120):
        self.prices = np.array(prices, dtype=float)
        self.ma_period = ma_period
        self.ma = self._calculate_ma()

    def _calculate_ma(self) -> np.ndarray:
        """计算长期均线"""
        ma = np.full_like(self.prices, np.nan)
        for i in range(self.ma_period - 1, len(self.prices)):
            ma[i] = np.mean(self.prices[i - self.ma_period + 1:i + 1])
        return ma

    def get_position(self, index: int = -1, lookback: int = 120) -> PricePosition:
        """
        计算价格位置

        Args:
            index: K线索引，-1为最新
            lookback: 回溯周期

        Returns:
            PricePosition 对象
        """
        if index < 0:
            index = len(self.prices) + index

        if index < 2:
            return PricePosition()

        start = max(0, index - lookback + 1)
        window = self.prices[start:index + 1]
        current = self.prices[index]

        # 分位数：当前价在窗口内的百分位排名
        percentile = np.sum(window < current) / len(window) * 100.0

        # 均线偏离度
        ma_val = self.ma[index] if not np.isnan(self.ma[index]) else np.mean(window)
        ma_bias = (current - ma_val) / ma_val * 100.0 if ma_val != 0 else 0.0

        return PricePosition(
            percentile=percentile,
            is_low=percentile < 20,
            is_high=percentile > 80,
            is_mid_low=percentile < 40,
            ma_bias=ma_bias
        )


class VolumeTransitionDetector:
    """
    缩量→放量转折检测器

    检测成交量从持续缩量转为放量的转折点，
    这通常是主力资金建仓完成、即将拉升的信号。
    """

    def __init__(self, volumes: np.ndarray, ma_period: int = 20):
        self.volumes = np.array(volumes, dtype=float)
        self.ma_period = ma_period
        self.volume_ma = self._calculate_ma()
        self.volume_ratio = self._calculate_ratio()

    def _calculate_ma(self) -> np.ndarray:
        ma = np.full_like(self.volumes, np.nan)
        for i in range(self.ma_period - 1, len(self.volumes)):
            ma[i] = np.mean(self.volumes[i - self.ma_period + 1:i + 1])
        return ma

    def _calculate_ratio(self) -> np.ndarray:
        ratio = np.ones_like(self.volumes)
        valid = ~np.isnan(self.volume_ma)
        ratio[valid] = self.volumes[valid] / self.volume_ma[valid]
        return ratio

    def detect_transition(
        self,
        index: int = -1,
        lookback: int = 20,
        shrink_threshold: float = 0.8,
        expand_threshold: float = 1.2
    ) -> VolumeTransition:
        """
        检测缩量→放量转折

        逻辑：
        - 将 lookback 窗口分为前半段和后半段
        - 前半段平均量比 < shrink_threshold → 缩量
        - 后半段平均量比 > expand_threshold → 放量
        - 转折强度 = 后半段量比 / 前半段量比 - 1

        Args:
            index: K线索引
            lookback: 回溯周期
            shrink_threshold: 缩量阈值（量比）
            expand_threshold: 放量阈值（量比）

        Returns:
            VolumeTransition 对象
        """
        if index < 0:
            index = len(self.volumes) + index

        if index < lookback:
            return VolumeTransition()

        half = lookback // 2
        start = index - lookback + 1

        # 前半段（缩量期）
        shrink_ratios = self.volume_ratio[start:start + half]
        shrink_avg = np.nanmean(shrink_ratios)
        if np.isnan(shrink_avg):
            shrink_avg = 1.0

        # 后半段（放量期）
        expand_ratios = self.volume_ratio[start + half:index + 1]
        expand_avg = np.nanmean(expand_ratios)
        if np.isnan(expand_avg):
            expand_avg = 1.0

        # 判断转折
        has_shrink = shrink_avg < shrink_threshold
        has_expand = expand_avg > expand_threshold
        has_transition = has_shrink and has_expand

        # 计算强度
        strength = 0.0
        if has_transition and shrink_avg > 0:
            strength = min((expand_avg / shrink_avg - 1.0) / 2.0, 1.0)

        return VolumeTransition(
            has_transition=has_transition,
            direction='shrink_to_expand' if has_transition else '',
            strength=strength,
            shrink_avg_ratio=shrink_avg,
            expand_avg_ratio=expand_avg
        )


class StrokeVolumeDivergence:
    """
    缠论笔级别的量价背离检测

    底背离：向下笔创新低，但该笔区间均量 < 前一个向下笔均量
    顶背离：向上笔创新高，但该笔区间均量 < 前一个向上笔均量

    与缠论背驰概念一致：走势创新低/新高，但力度减弱（用量能衡量）。
    """

    def __init__(self, prices: np.ndarray, volumes: np.ndarray):
        self.prices = np.array(prices, dtype=float)
        self.volumes = np.array(volumes, dtype=float)

    def check_divergence(self, strokes: list) -> StrokeDivergence:
        """
        检测笔级别的量价背离

        Args:
            strokes: 缠论笔列表，每个笔需要有：
                     - is_up / is_down 属性
                     - start_index, end_index 属性
                     - high, low 或 start_value, end_value 属性

        Returns:
            StrokeDivergence 对象
        """
        if len(strokes) < 3:
            return StrokeDivergence()

        last_stroke = strokes[-1]

        # 根据最后一笔方向，找同方向的前一笔进行比较
        if getattr(last_stroke, 'is_up', False):
            return self._check_top_divergence(strokes)
        elif getattr(last_stroke, 'is_down', False):
            return self._check_bottom_divergence(strokes)

        return StrokeDivergence()

    def _check_bottom_divergence(self, strokes: list) -> StrokeDivergence:
        """检测底背离：向下笔创新低但量能萎缩"""
        down_strokes = [s for s in strokes if getattr(s, 'is_down', False)]
        if len(down_strokes) < 2:
            return StrokeDivergence()

        curr = down_strokes[-1]
        prev = down_strokes[-2]

        # 获取极值价格
        curr_low = self._get_stroke_low(curr)
        prev_low = self._get_stroke_low(prev)

        if prev_low == 0 or curr_low >= prev_low:
            return StrokeDivergence()  # 未创新低

        # 计算笔区间内的均量
        curr_avg_vol = self._get_stroke_avg_volume(curr)
        prev_avg_vol = self._get_stroke_avg_volume(prev)

        if prev_avg_vol == 0:
            return StrokeDivergence()

        # 量能萎缩 = 底背离
        vol_ratio = curr_avg_vol / prev_avg_vol
        if vol_ratio < 0.9:  # 量能减少10%以上
            strength = min(1.0 - vol_ratio, 1.0)
            return StrokeDivergence(
                has_divergence=True,
                direction='bottom',
                strength=strength,
                prev_avg_volume=prev_avg_vol,
                curr_avg_volume=curr_avg_vol,
                prev_extreme=prev_low,
                curr_extreme=curr_low
            )

        return StrokeDivergence()

    def _check_top_divergence(self, strokes: list) -> StrokeDivergence:
        """检测顶背离：向上笔创新高但量能萎缩"""
        up_strokes = [s for s in strokes if getattr(s, 'is_up', False)]
        if len(up_strokes) < 2:
            return StrokeDivergence()

        curr = up_strokes[-1]
        prev = up_strokes[-2]

        curr_high = self._get_stroke_high(curr)
        prev_high = self._get_stroke_high(prev)

        if prev_high == 0 or curr_high <= prev_high:
            return StrokeDivergence()  # 未创新高

        curr_avg_vol = self._get_stroke_avg_volume(curr)
        prev_avg_vol = self._get_stroke_avg_volume(prev)

        if prev_avg_vol == 0:
            return StrokeDivergence()

        vol_ratio = curr_avg_vol / prev_avg_vol
        if vol_ratio < 0.9:
            strength = min(1.0 - vol_ratio, 1.0)
            return StrokeDivergence(
                has_divergence=True,
                direction='top',
                strength=strength,
                prev_avg_volume=prev_avg_vol,
                curr_avg_volume=curr_avg_vol,
                prev_extreme=prev_high,
                curr_extreme=curr_high
            )

        return StrokeDivergence()

    def _get_stroke_low(self, stroke) -> float:
        """获取笔的最低价"""
        if hasattr(stroke, 'low'):
            return float(stroke.low)
        if hasattr(stroke, 'end_value') and getattr(stroke, 'is_down', False):
            return float(stroke.end_value)
        if hasattr(stroke, 'start_value') and getattr(stroke, 'is_up', False):
            return float(stroke.start_value)
        # 用索引回查价格
        idx = getattr(stroke, 'end_index', getattr(stroke, 'start_index', 0))
        if 0 <= idx < len(self.prices):
            return float(self.prices[idx])
        return 0.0

    def _get_stroke_high(self, stroke) -> float:
        """获取笔的最高价"""
        if hasattr(stroke, 'high'):
            return float(stroke.high)
        if hasattr(stroke, 'end_value') and getattr(stroke, 'is_up', False):
            return float(stroke.end_value)
        if hasattr(stroke, 'start_value') and getattr(stroke, 'is_down', False):
            return float(stroke.start_value)
        idx = getattr(stroke, 'end_index', getattr(stroke, 'start_index', 0))
        if 0 <= idx < len(self.prices):
            return float(self.prices[idx])
        return 0.0

    def _get_stroke_avg_volume(self, stroke) -> float:
        """计算笔区间内的平均成交量"""
        start_idx = getattr(stroke, 'start_index', 0)
        end_idx = getattr(stroke, 'end_index', 0)

        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(self.volumes):
            end_idx = len(self.volumes) - 1
        if start_idx > end_idx:
            return 0.0

        return float(np.mean(self.volumes[start_idx:end_idx + 1]))


class TurnoverAnalyzer:
    """
    换手率分析器

    如果数据中有 turnover 列则使用真实换手率，
    否则使用成交量/成交量均值作为相对换手率的近似。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        turnover_col: str = 'turnover',
        volume_col: str = 'volume',
        ma_period: int = 20
    ):
        self.has_real_turnover = turnover_col in df.columns

        if self.has_real_turnover:
            self.turnover = df[turnover_col].values.astype(float)
        else:
            # 用成交量均值比作为相对换手率
            vol = df[volume_col].values.astype(float)
            self.turnover = np.ones_like(vol)
            if len(vol) >= ma_period:
                vol_ma = np.full_like(vol, np.nan)
                for i in range(ma_period - 1, len(vol)):
                    vol_ma[i] = np.mean(vol[i - ma_period + 1:i + 1])
                valid = ~np.isnan(vol_ma)
                self.turnover[valid] = vol[valid] / vol_ma[valid]

        self.ma_period = ma_period
        self.turnover_ma = self._calculate_ma()

    def _calculate_ma(self) -> np.ndarray:
        ma = np.full_like(self.turnover, np.nan)
        for i in range(self.ma_period - 1, len(self.turnover)):
            ma[i] = np.mean(self.turnover[i - self.ma_period + 1:i + 1])
        return ma

    def analyze(self, index: int = -1, lookback: int = 10) -> TurnoverResult:
        """
        分析换手率

        Args:
            index: K线索引
            lookback: 趋势回溯周期

        Returns:
            TurnoverResult 对象
        """
        if index < 0:
            index = len(self.turnover) + index

        if index < 0 or index >= len(self.turnover):
            return TurnoverResult()

        current = self.turnover[index]
        avg = self.turnover_ma[index] if not np.isnan(self.turnover_ma[index]) else current
        ratio = current / avg if avg > 0 else 1.0

        # 换手率趋势
        trend = 'neutral'
        if index >= lookback:
            recent = self.turnover[index - lookback + 1:index + 1]
            x = np.arange(len(recent))
            slope, _ = np.polyfit(x, recent, 1)
            if slope > 0:
                trend = 'up'
            elif slope < 0:
                trend = 'down'

        # 低换手率阶段判断
        is_low = ratio < 0.6 and trend == 'down'

        return TurnoverResult(
            current=current,
            avg=avg,
            ratio_vs_avg=ratio,
            trend=trend,
            is_low_phase=is_low
        )


class VolumePriceAnalyzer:
    """
    量价联合分析器（主入口）

    整合价格位置、量能转折、笔级背离、换手率分析，
    为缠论策略提供量价确认信号。
    """

    def __init__(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        volume_ma_period: int = 20,
        price_ma_period: int = 120
    ):
        self.price_pos = PricePositionAnalyzer(prices, price_ma_period)
        self.vol_transition = VolumeTransitionDetector(volumes, volume_ma_period)
        self.stroke_div = StrokeVolumeDivergence(prices, volumes)
        self.turnover = TurnoverAnalyzer(df) if df is not None else None

        self.prices = np.array(prices, dtype=float)
        self.volumes = np.array(volumes, dtype=float)

    def get_price_position(self, index: int = -1, lookback: int = 120) -> PricePosition:
        """获取价格位置分析"""
        return self.price_pos.get_position(index, lookback)

    def get_volume_transition(self, index: int = -1, lookback: int = 20) -> VolumeTransition:
        """获取量能转折分析"""
        return self.vol_transition.detect_transition(index, lookback)

    def get_stroke_divergence(self, strokes: list) -> StrokeDivergence:
        """获取笔级量价背离"""
        return self.stroke_div.check_divergence(strokes)

    def get_turnover(self, index: int = -1) -> Optional[TurnoverResult]:
        """获取换手率分析"""
        if self.turnover is None:
            return None
        return self.turnover.analyze(index)

    def check_buy_confirmation(
        self,
        index: int = -1,
        strokes: Optional[list] = None,
        buy_point_type: str = ''
    ) -> Tuple[float, str]:
        """
        买入信号的量价确认评分

        根据量价分析给出置信度调整值和说明。

        Args:
            index: K线索引
            strokes: 缠论笔列表
            buy_point_type: '1buy', '2buy', '3buy'

        Returns:
            (置信度调整值, 说明)
            调整值范围 -0.15 ~ +0.15
        """
        adjustment = 0.0
        reasons = []

        # 1. 价格位置
        position = self.get_price_position(index)
        if position.is_low:
            adjustment += 0.10
            reasons.append(f"价格低位(分位{position.percentile:.0f}%)")
        elif position.is_mid_low:
            adjustment += 0.05
            reasons.append(f"价格中低位(分位{position.percentile:.0f}%)")
        elif position.is_high:
            adjustment -= 0.10
            reasons.append(f"价格高位(分位{position.percentile:.0f}%)")

        # 2. 笔级量价背离
        if strokes is not None:
            div = self.get_stroke_divergence(strokes)
            if div.has_divergence and div.direction == 'bottom':
                adj = 0.05 + div.strength * 0.10
                adjustment += adj
                reasons.append(f"笔级量价底背离(强度{div.strength:.2f})")

        # 3. 缩量→放量转折（对2买和3买特别有意义）
        if buy_point_type in ('2buy', '3buy', ''):
            trans = self.get_volume_transition(index)
            if trans.has_transition:
                adj = 0.05 + trans.strength * 0.10
                adjustment += adj
                reasons.append(f"缩量→放量转折(强度{trans.strength:.2f})")

        # 4. 换手率
        turnover = self.get_turnover(index)
        if turnover is not None:
            if turnover.is_low_phase:
                adjustment += 0.05
                reasons.append("低换手率阶段")
            elif turnover.ratio_vs_avg > 2.0:
                adjustment -= 0.05
                reasons.append("换手率过高")

        # 限制范围
        adjustment = max(-0.15, min(0.15, adjustment))

        reason_str = '; '.join(reasons) if reasons else '无量价确认'
        if not reasons:
            adjustment = min(adjustment, -0.05)

        return adjustment, reason_str

    def check_sell_confirmation(
        self,
        index: int = -1,
        strokes: Optional[list] = None
    ) -> Tuple[float, str]:
        """
        卖出信号的量价确认评分

        Args:
            index: K线索引
            strokes: 缠论笔列表

        Returns:
            (置信度调整值, 说明)
        """
        adjustment = 0.0
        reasons = []

        # 1. 价格高位
        position = self.get_price_position(index)
        if position.is_high:
            adjustment += 0.10
            reasons.append(f"价格高位(分位{position.percentile:.0f}%)")

        # 2. 笔级量价顶背离
        if strokes is not None:
            div = self.get_stroke_divergence(strokes)
            if div.has_divergence and div.direction == 'top':
                adj = 0.05 + div.strength * 0.10
                adjustment += adj
                reasons.append(f"笔级量价顶背离(强度{div.strength:.2f})")

        # 3. 高位放量滞涨
        if index >= 2:
            recent_return = (self.prices[index] - self.prices[index - 2]) / self.prices[index - 2]
            avg_vol = np.mean(self.volumes[max(0, index - 20):index + 1])
            curr_vol = self.volumes[index]
            if abs(recent_return) < 0.02 and curr_vol > avg_vol * 1.5 and position.is_high:
                adjustment += 0.10
                reasons.append("高位放量滞涨")

        adjustment = max(-0.10, min(0.15, adjustment))

        reason_str = '; '.join(reasons) if reasons else '无量价确认'
        if not reasons:
            adjustment = min(adjustment, -0.05)

        return adjustment, reason_str


def analyze_volume_price(df: pd.DataFrame, index: int = -1) -> dict:
    """
    便捷函数：综合量价分析

    Args:
        df: 包含 OHLCV 的 DataFrame（可选 turnover 列）
        index: 分析的K线索引，-1 为最新

    Returns:
        综合分析结果字典
    """
    if 'close' not in df.columns or 'volume' not in df.columns:
        return {}

    prices = df['close'].values
    volumes = df['volume'].values

    analyzer = VolumePriceAnalyzer(prices, volumes, df)

    position = analyzer.get_price_position(index)
    transition = analyzer.get_volume_transition(index)
    turnover = analyzer.get_turnover(index)

    result = {
        'price_position': {
            'percentile': round(position.percentile, 1),
            'is_low': position.is_low,
            'is_high': position.is_high,
            'ma_bias_pct': round(position.ma_bias, 2)
        },
        'volume_transition': {
            'has_transition': transition.has_transition,
            'direction': transition.direction,
            'strength': round(transition.strength, 3)
        },
        'buy_adjustment': 0.0,
        'buy_reason': '',
        'sell_adjustment': 0.0,
        'sell_reason': ''
    }

    buy_adj, buy_reason = analyzer.check_buy_confirmation(index)
    sell_adj, sell_reason = analyzer.check_sell_confirmation(index)

    result['buy_adjustment'] = round(buy_adj, 3)
    result['buy_reason'] = buy_reason
    result['sell_adjustment'] = round(sell_adj, 3)
    result['sell_reason'] = sell_reason

    if turnover is not None:
        result['turnover'] = {
            'current': round(turnover.current, 4),
            'avg': round(turnover.avg, 4),
            'ratio': round(turnover.ratio_vs_avg, 2),
            'trend': turnover.trend,
            'is_low_phase': turnover.is_low_phase
        }

    return result
