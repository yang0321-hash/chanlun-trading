"""
中枢引力场模型

缠论核心思想：中枢像一个引力场，价格离开越远，回拉力越强。
引力指数 = 中枢宽度 / 离开距离

应用：
1. 判断3买/3卖的真假（引力大=假突破，引力小=真脱离）
2. 判断趋势力度（引力持续下降=趋势强劲）
"""

from dataclasses import dataclass
from typing import Optional

from .pivot import Pivot


@dataclass
class GravityResult:
    """中枢引力检测结果"""
    pivot_index: int          # 中枢编号
    pivot_zg: float           # 中枢ZG
    pivot_zd: float           # 中枢ZD
    pivot_width: float        # 中枢宽度 W=ZG-ZD
    current_price: float      # 当前价格
    distance: float           # 价格离中枢的距离
    gravity_index: float      # 引力指数 = W / distance
    status: str               # "inside" / "near" / "escaping" / "escaped"
    pull_probability: float   # 回拉概率 0~1


class ZhongshuGravity:
    """
    中枢引力场判定

    引力分级：
    - gravity >= 2.0：强引力区，大概率回归
    - gravity 1.0~2.0：中等引力
    - gravity 0.5~1.0：弱引力，正在脱离
    - gravity < 0.5：几乎脱离引力，趋势可能确立
    """

    STRONG_GRAVITY = 2.0
    MEDIUM_GRAVITY = 1.0
    WEAK_GRAVITY = 0.5

    @staticmethod
    def calc_gravity(pivot: Pivot, price: float) -> GravityResult:
        """计算价格相对于中枢的引力指数"""
        zg = pivot.zg
        zd = pivot.zd
        width = zg - zd

        # 价格在中枢内部
        if zd <= price <= zg:
            return GravityResult(
                pivot_index=id(pivot) % 10000,
                pivot_zg=round(zg, 4),
                pivot_zd=round(zd, 4),
                pivot_width=round(width, 4),
                current_price=round(price, 4),
                distance=0.0,
                gravity_index=float('inf'),
                status="inside",
                pull_probability=0.0,
            )

        distance = (price - zg) if price > zg else (zd - price)
        gravity = width / distance if distance > 0 else float('inf')

        if gravity >= ZhongshuGravity.STRONG_GRAVITY:
            status, pull_prob = "near", 0.7
        elif gravity >= ZhongshuGravity.MEDIUM_GRAVITY:
            status, pull_prob = "near", 0.45
        elif gravity >= ZhongshuGravity.WEAK_GRAVITY:
            status, pull_prob = "escaping", 0.25
        else:
            status, pull_prob = "escaped", 0.10

        return GravityResult(
            pivot_index=id(pivot) % 10000,
            pivot_zg=round(zg, 4),
            pivot_zd=round(zd, 4),
            pivot_width=round(width, 4),
            current_price=round(price, 4),
            distance=round(distance, 4),
            gravity_index=round(gravity, 4),
            status=status,
            pull_probability=round(pull_prob, 2),
        )

    @staticmethod
    def validate_3buy(
        pivot: Pivot,
        pullback_price: float,
        current_price: float,
    ) -> dict:
        """
        验证3买的真实性

        Returns: {
            "is_valid_3buy": bool,
            "entered_pivot": bool,
            "gravity": GravityResult,
            "confidence_modifier": float  # -0.15 ~ +0.10
        }
        """
        gravity = ZhongshuGravity.calc_gravity(pivot, current_price)
        entered = pullback_price <= pivot.zd

        result = {
            "is_valid_3buy": False,
            "entered_pivot": entered,
            "gravity": gravity,
            "confidence_modifier": 0.0,
        }

        if entered:
            result["confidence_modifier"] = -0.15
        elif gravity.status == "escaped":
            result["is_valid_3buy"] = True
            result["confidence_modifier"] = 0.10
        elif gravity.status == "escaping":
            result["is_valid_3buy"] = True
        else:
            result["confidence_modifier"] = -0.05

        return result

    @staticmethod
    def validate_3sell(
        pivot: Pivot,
        pullback_price: float,
        current_price: float,
    ) -> dict:
        """验证3卖的真实性（镜像对称）"""
        gravity = ZhongshuGravity.calc_gravity(pivot, current_price)
        entered = pullback_price >= pivot.zg

        result = {
            "is_valid_3sell": False,
            "entered_pivot": entered,
            "gravity": gravity,
            "confidence_modifier": 0.0,
        }

        if entered:
            result["confidence_modifier"] = -0.15
        elif gravity.status == "escaped":
            result["is_valid_3sell"] = True
            result["confidence_modifier"] = 0.10
        elif gravity.status == "escaping":
            result["is_valid_3sell"] = True
        else:
            result["confidence_modifier"] = -0.05

        return result
