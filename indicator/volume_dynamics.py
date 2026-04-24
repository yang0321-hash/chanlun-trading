"""
成交量动态分析模块

提供跨笔量能趋势分析和量价背离检测，补充MACD背驰判断。

核心能力：
1. 跨笔量能趋势：比较连续同向笔的成交量，检测递增/递减趋势
2. 量价背离：价格创新低但量能萎缩 → 底背离；价格创新高但量能萎缩 → 顶背离
3. 量能-MACD共振：两者同时出现背离时信号更强
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class VolumeDivergenceResult:
    """量能背离检测结果"""
    has_divergence: bool = False
    strength: float = 0.0        # 0-1, 越大越强
    vol_trend: str = ''           # 'declining' / 'rising' / 'flat'
    vol_ratio: float = 0.0        # 最近笔量 / 前一笔量
    description: str = ''


class VolumeDynamics:
    """
    成交量动态分析器

    使用方法：
        vd = VolumeDynamics(strokes)
        result = vd.check_volume_divergence(down_strokes, 'down')
        resonance = vd.check_volume_macd_resonance(macd, strokes, 'down')
    """

    def __init__(self, strokes: list):
        self.strokes = strokes

    def _stroke_avg_volume(self, stroke) -> float:
        """计算笔的平均成交量"""
        if not hasattr(stroke, 'bars') or not stroke.bars:
            return 0.0
        vols = [b.volume for b in stroke.bars if hasattr(b, 'volume')]
        return sum(vols) / len(vols) if vols else 0.0

    def _stroke_total_volume(self, stroke) -> float:
        """计算笔的总成交量"""
        if not hasattr(stroke, 'bars') or not stroke.bars:
            return 0.0
        return sum(b.volume for b in stroke.bars if hasattr(b, 'volume'))

    def check_volume_divergence(
        self,
        same_dir_strokes: list,
        direction: str = 'down',
        min_strokes: int = 2,
    ) -> VolumeDivergenceResult:
        """
        检测量能背离

        底背离(1买)：连续向下笔的成交量递减 + 价格创新低
        顶背离(1卖)：连续向上笔的成交量递减 + 价格创新高

        Args:
            same_dir_strokes: 同方向的笔列表（按时间排序）
            direction: 'down' 检测底背离, 'up' 检测顶背离
            min_strokes: 最少需要的笔数

        Returns:
            VolumeDivergenceResult
        """
        if len(same_dir_strokes) < min_strokes:
            return VolumeDivergenceResult(description='笔数不足')

        # 取最近3-5笔分析量能趋势
        recent = same_dir_strokes[-5:] if len(same_dir_strokes) >= 5 else same_dir_strokes[-3:]
        vols = [self._stroke_total_volume(s) for s in recent]
        prices = [s.end_value for s in recent]

        # 过滤零值
        valid_pairs = [(v, p) for v, p in zip(vols, prices) if v > 0]
        if len(valid_pairs) < 2:
            return VolumeDivergenceResult(description='有效量能数据不足')

        vols_valid = [p[0] for p in valid_pairs]
        prices_valid = [p[1] for p in valid_pairs]

        # 量能趋势判断
        declines = sum(1 for i in range(1, len(vols_valid))
                       if vols_valid[i] < vols_valid[i - 1])
        total = len(vols_valid) - 1
        decline_ratio = declines / total if total > 0 else 0

        if decline_ratio >= 0.7:
            vol_trend = 'declining'
        elif decline_ratio <= 0.3:
            vol_trend = 'rising'
        else:
            vol_trend = 'flat'

        # 最近两笔的量比
        vol_ratio = vols_valid[-1] / vols_valid[-2] if vols_valid[-2] > 0 else 1.0

        # 量价背离判断
        has_divergence = False
        strength = 0.0

        if direction == 'down':
            # 底背离：价格创新低 + 量能萎缩
            price_making_lows = sum(1 for i in range(1, len(prices_valid))
                                    if prices_valid[i] < prices_valid[i - 1])
            price_declining = price_making_lows >= total * 0.5

            if vol_trend == 'declining' and price_declining:
                has_divergence = True
                # 量能萎缩幅度 = 1 - 最后一笔量/最大量
                max_vol = max(vols_valid)
                min_vol = min(vols_valid[-2:])
                shrink = 1.0 - (min_vol / max_vol) if max_vol > 0 else 0
                strength = min(shrink * 2, 1.0)  # 0-1

        elif direction == 'up':
            # 顶背离：价格创新高 + 量能萎缩
            price_making_highs = sum(1 for i in range(1, len(prices_valid))
                                     if prices_valid[i] > prices_valid[i - 1])
            price_rising = price_making_highs >= total * 0.5

            if vol_trend == 'declining' and price_rising:
                has_divergence = True
                max_vol = max(vols_valid)
                min_vol = min(vols_valid[-2:])
                shrink = 1.0 - (min_vol / max_vol) if max_vol > 0 else 0
                strength = min(shrink * 2, 1.0)

        desc = ''
        if has_divergence:
            desc = f'量价背离: 量比{vol_ratio:.2f}, 趋势={vol_trend}, 强度={strength:.2f}'

        return VolumeDivergenceResult(
            has_divergence=has_divergence,
            strength=strength,
            vol_trend=vol_trend,
            vol_ratio=vol_ratio,
            description=desc,
        )

    def check_volume_macd_resonance(
        self,
        macd,
        same_dir_strokes: list,
        direction: str = 'down',
    ) -> Tuple[bool, float, str]:
        """
        量能-MACD共振检测

        当MACD面积递减 AND 量能同时萎缩时，背驰信号更可靠。

        Args:
            macd: MACD指标对象
            same_dir_strokes: 同方向笔列表
            direction: 方向

        Returns:
            (是否共振, 共振强度0-1, 描述)
        """
        if not macd or len(same_dir_strokes) < 2:
            return (False, 0.0, '')

        # 检测量能背离
        vol_result = self.check_volume_divergence(same_dir_strokes, direction)

        # 检测MACD面积趋势
        macd_areas = []
        for s in same_dir_strokes[-5:]:
            area = macd.compute_area(s.start_index, s.end_index, direction)
            macd_areas.append(abs(area))

        macd_declining = False
        macd_strength = 0.0
        if len(macd_areas) >= 2:
            declines = sum(1 for i in range(1, len(macd_areas))
                           if macd_areas[i] < macd_areas[i - 1])
            if declines >= len(macd_areas) - 2:
                macd_declining = True
                max_area = max(macd_areas)
                min_area = min(macd_areas[-2:])
                macd_strength = 1.0 - (min_area / max_area) if max_area > 0 else 0

        # 共振判断
        if vol_result.has_divergence and macd_declining:
            resonance = (vol_result.strength + macd_strength) / 2
            desc = (f'量能+MACD双重背驰: 量能{vol_result.strength:.2f}'
                    f'+MACD{macd_strength:.2f}')
            return (True, resonance, desc)

        if vol_result.has_divergence:
            return (True, vol_result.strength * 0.6,
                    f'量能单边背驰(强度{vol_result.strength:.2f})')

        if macd_declining:
            return (True, macd_strength * 0.6,
                    f'MACD单边背驰(强度{macd_strength:.2f})')

        return (False, 0.0, '')

    def get_breakout_volume_score(
        self,
        breakout_stroke,
        pullback_stroke,
    ) -> Tuple[float, str]:
        """
        突破量价节奏评分（3买专用）

        理想节奏：放量突破 → 缩量回踩
        放量突破量比 > 1.3, 缩量回踩量比 < 0.7

        Args:
            breakout_stroke: 突破笔
            pullback_stroke: 回调笔

        Returns:
            (评分修正 -0.1~+0.15, 描述)
        """
        bo_vol = self._stroke_total_volume(breakout_stroke)
        pb_vol = self._stroke_total_volume(pullback_stroke)

        if bo_vol <= 0 or pb_vol <= 0:
            return (0.0, '')

        # 计算前几笔的平均量作为基准
        base_strokes = [s for s in self.strokes
                        if s.end_index < breakout_stroke.start_index]
        if not base_strokes:
            return (0.0, '')

        base_vols = [self._stroke_total_volume(s) for s in base_strokes[-5:]]
        base_vols = [v for v in base_vols if v > 0]
        if not base_vols:
            return (0.0, '')

        avg_base = sum(base_vols) / len(base_vols)
        bo_ratio = bo_vol / avg_base
        pb_ratio = pb_vol / avg_base

        score = 0.0
        parts = []

        # 突破放量
        if bo_ratio > 1.5:
            score += 0.08
            parts.append(f'强放量突破(×{bo_ratio:.1f})')
        elif bo_ratio > 1.2:
            score += 0.04
            parts.append(f'放量突破(×{bo_ratio:.1f})')
        elif bo_ratio < 0.7:
            score -= 0.06
            parts.append(f'缩量突破(×{bo_ratio:.1f})')

        # 回调缩量
        if pb_ratio < 0.6:
            score += 0.07
            parts.append(f'深度缩量回踩(×{pb_ratio:.1f})')
        elif pb_ratio < 0.8:
            score += 0.03
            parts.append(f'缩量回踩(×{pb_ratio:.1f})')
        elif pb_ratio > 1.5:
            score -= 0.05
            parts.append(f'放量回踩(×{pb_ratio:.1f})')

        return (max(-0.1, min(0.15, score)), ', '.join(parts) if parts else '')
