"""
买点识别 Skill - 识别缠论中的三类买点
"""
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from skills.base import BaseSkill, SkillResult
from core.kline import KLine
from core.stroke import Stroke
from core.pivot import Pivot


class BuyPointType(Enum):
    """买点类型"""
    FIRST_BUY = '1买'      # 第一类买点
    SECOND_BUY = '2买'     # 第二类买点
    THIRD_BUY = '3买'      # 第三类买点


@dataclass
class BuyPoint:
    """买点数据"""
    point_type: BuyPointType
    index: int
    price: float
    confidence: float
    reason: str
    pivot_low: Optional[float] = None
    pivot_high: Optional[float] = None
    divergence_detected: bool = False
    stop_loss: float = 0.0
    target: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'point_type': self.point_type.value,
            'index': self.index,
            'price': self.price,
            'confidence': self.confidence,
            'reason': self.reason,
            'pivot_low': self.pivot_low,
            'pivot_high': self.pivot_high,
            'divergence_detected': self.divergence_detected,
            'stop_loss': self.stop_loss,
            'target': self.target,
        }


class BuyPointSkill(BaseSkill[List[BuyPoint]]):
    """
    买点识别 Skill

    能力:
    - 识别第一类买点 (最后中枢下方底背驰)
    - 识别第二类买点 (回抽不破前低)
    - 识别第三类买点 (突破中枢回抽不进入)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_confidence = self.get_config('min_confidence', 0.6)
        self.use_divergence = self.get_config('use_divergence', True)
        self.stop_loss_pct = self.get_config('stop_loss_pct', 0.08)
        self.target_pct = self.get_config('target_pct', 0.15)

    def validate(
        self,
        kline: Optional[KLine] = None,
        strokes: Optional[List[Stroke]] = None,
        pivots: Optional[List[Pivot]] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """验证输入参数"""
        if strokes is None:
            return False, "需要提供 strokes 参数"

        if len(strokes) < 3:
            return False, "笔数据不足，至少需要3笔"

        return True, None

    def execute(
        self,
        kline: Optional[KLine] = None,
        strokes: Optional[List[Stroke]] = None,
        pivots: Optional[List[Pivot]] = None,
        fractals: Optional[List[Any]] = None,
        current_index: int = -1,
        **kwargs
    ) -> SkillResult:
        """
        执行买点识别

        Args:
            kline: K线对象
            strokes: 笔列表
            pivots: 中枢列表
            fractals: 分型列表 (可选)
            current_index: 当前索引

        Returns:
            SkillResult: 识别的买点列表
        """
        if strokes is None:
            return SkillResult(success=False, error="缺少 strokes 参数")

        # 参数验证
        is_valid, error = self.validate(kline=kline, strokes=strokes, pivots=pivots)
        if not is_valid:
            return SkillResult(success=False, error=error)

        try:
            buy_points = []

            # 识别第一类买点
            if pivots:
                first_buys = self._detect_first_buy(kline, strokes, pivots, current_index)
                buy_points.extend(first_buys)

            # 识别第二类买点
            second_buys = self._detect_second_buy(kline, strokes, current_index)
            buy_points.extend(second_buys)

            # 识别第三类买点
            if pivots:
                third_buys = self._detect_third_buy(kline, strokes, pivots, current_index)
                buy_points.extend(third_buys)

            # 按置信度排序
            buy_points.sort(key=lambda bp: bp.confidence, reverse=True)

            # 计算总体置信度
            overall_confidence = self._calculate_overall_confidence(buy_points)

            # 构建元数据
            metadata = {
                'total_buy_points': len(buy_points),
                'first_buy_count': len([bp for bp in buy_points if bp.point_type == BuyPointType.FIRST_BUY]),
                'second_buy_count': len([bp for bp in buy_points if bp.point_type == BuyPointType.SECOND_BUY]),
                'third_buy_count': len([bp for bp in buy_points if bp.point_type == BuyPointType.THIRD_BUY]),
            }

            return SkillResult(
                success=True,
                data=buy_points,
                confidence=overall_confidence,
                metadata=metadata
            )

        except Exception as e:
            return SkillResult(
                success=False,
                error=f"买点识别失败: {str(e)}"
            )

    def _detect_first_buy(
        self,
        kline: Optional[KLine],
        strokes: List[Stroke],
        pivots: List[Pivot],
        current_index: int
    ) -> List[BuyPoint]:
        """识别第一类买点"""
        buy_points = []

        if not pivots:
            return buy_points

        # 获取最后一个中枢
        last_pivot = pivots[-1]

        # 查找最后中枢之后的向下笔
        for stroke in reversed(strokes):
            if stroke.direction == 'down' and stroke.end_index <= current_index:
                # 检查是否在中枢下方
                if stroke.end_value <= last_pivot.low:
                    # 计算置信度
                    confidence = 0.7

                    # 检查背驰（如果有K线数据）
                    if self.use_divergence and kline is not None:
                        has_divergence = self._check_divergence(kline, stroke.end_index, is_bottom=True)
                        if has_divergence:
                            confidence = 0.85

                    if confidence >= self.min_confidence:
                        buy_points.append(BuyPoint(
                            point_type=BuyPointType.FIRST_BUY,
                            index=stroke.end_index,
                            price=stroke.end_value,
                            confidence=confidence,
                            reason=f"1买：最后中枢[{last_pivot.low:.2f}-{last_pivot.high:.2f}]下方底背驰",
                            pivot_low=last_pivot.low,
                            pivot_high=last_pivot.high,
                            divergence_detected=True,
                            stop_loss=stroke.end_value * (1 - self.stop_loss_pct),
                            target=stroke.end_value * (1 + self.target_pct)
                        ))
                break

        return buy_points

    def _detect_second_buy(
        self,
        kline: Optional[KLine],
        strokes: List[Stroke],
        current_index: int
    ) -> List[BuyPoint]:
        """识别第二类买点"""
        buy_points = []

        # 查找向上笔后的向下笔（回调）
        for i in range(len(strokes) - 1):
            up_stroke = strokes[i]
            down_stroke = strokes[i + 1]

            if (up_stroke.direction == 'up' and
                down_stroke.direction == 'down' and
                down_stroke.end_index <= current_index):

                # 检查回调是否跌破前低
                if down_stroke.end_value > up_stroke.start_value:
                    # 回抽不破，形成2买
                    confidence = 0.7

                    # 检查是否有底分型确认
                    if kline is not None and down_stroke.end_index < len(kline):
                        # 简单检查：当前K线比前一根高
                        if down_stroke.end_index > 0:
                            curr = kline.data[down_stroke.end_index]
                            prev = kline.data[down_stroke.end_index - 1]
                            if curr.close > prev.close:
                                confidence += 0.1

                    if confidence >= self.min_confidence:
                        buy_points.append(BuyPoint(
                            point_type=BuyPointType.SECOND_BUY,
                            index=down_stroke.end_index,
                            price=down_stroke.end_value,
                            confidence=min(confidence, 1.0),
                            reason=f"2买：回抽不破前低{up_stroke.start_value:.2f}",
                            divergence_detected=False,
                            stop_loss=down_stroke.end_value * (1 - self.stop_loss_pct),
                            target=down_stroke.end_value * (1 + self.target_pct)
                        ))

        return buy_points

    def _detect_third_buy(
        self,
        kline: Optional[KLine],
        strokes: List[Stroke],
        pivots: List[Pivot],
        current_index: int
    ) -> List[BuyPoint]:
        """识别第三类买点"""
        buy_points = []

        if not pivots:
            return buy_points

        last_pivot = pivots[-1]

        # 查找突破中枢后的回抽
        for i in range(len(strokes) - 1):
            up_stroke = strokes[i]
            down_stroke = strokes[i + 1]

            if (up_stroke.direction == 'up' and
                down_stroke.direction == 'down' and
                down_stroke.end_index <= current_index):

                # 检查是否突破了中枢上沿
                if up_stroke.end_value > last_pivot.high:
                    # 检查回抽是否不进入中枢
                    if down_stroke.end_value > last_pivot.high:
                        buy_points.append(BuyPoint(
                            point_type=BuyPointType.THIRD_BUY,
                            index=down_stroke.end_index,
                            price=down_stroke.end_value,
                            confidence=0.75,
                            reason=f"3买：突破中枢{last_pivot.high:.2f}后回抽不进入",
                            pivot_low=last_pivot.low,
                            pivot_high=last_pivot.high,
                            divergence_detected=False,
                            stop_loss=down_stroke.end_value * (1 - self.stop_loss_pct),
                            target=down_stroke.end_value * (1 + self.target_pct)
                        ))
                        break

        return buy_points

    def _check_divergence(
        self,
        kline: KLine,
        index: int,
        is_bottom: bool
    ) -> bool:
        """检查背驰"""
        if index < 10:
            return False

        try:
            # 简单的背驰检查
            # 比较最近低点和前面的低点
            recent_low = min(kline.data[max(0, index - 5):index + 1].low)

            # 找到前面的低点
            prev_low = min(kline.data[max(0, index - 20):index - 5].low)

            # 底背驰：价格创新低但指标不创新低
            # 这里简化为检查是否有反弹迹象
            if is_bottom and recent_low < prev_low:
                # 检查最近是否有向上力量
                if kline.data[index].close > kline.data[index].open:
                    return True

        except Exception:
            pass

        return False

    def _calculate_overall_confidence(self, buy_points: List[BuyPoint]) -> float:
        """计算总体置信度"""
        if not buy_points:
            return 0.0

        # 返回最高置信度买点的置信度
        return buy_points[0].confidence if buy_points else 0.0


# 便捷函数
def find_buy_points(
    strokes: List[Stroke],
    pivots: Optional[List[Pivot]] = None,
    min_confidence: float = 0.6
) -> List[BuyPoint]:
    """
    查找买点

    Args:
        strokes: 笔列表
        pivots: 中枢列表
        min_confidence: 最小置信度

    Returns:
        买点列表
    """
    skill = BuyPointSkill(config={'min_confidence': min_confidence})
    result = skill.execute(strokes=strokes, pivots=pivots, current_index=999999)

    if result.success:
        return result.data

    return []
