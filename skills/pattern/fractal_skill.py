"""
分型识别 Skill - 检测顶分型和底分型
"""
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from skills.base import BaseSkill, SkillResult, register_skill
from core.fractal import FractalDetector, Fractal, FractalType
from core.kline import KLine


@dataclass
class FractalInfo:
    """分型信息"""
    index: int
    fractal_type: str  # "top" or "bottom"
    high: float
    low: float
    confirmed: bool = False
    strength: float = 0.0  # 分型强度 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'fractal_type': self.fractal_type,
            'high': self.high,
            'low': self.low,
            'confirmed': self.confirmed,
            'strength': self.strength,
        }


class FractalSkill(BaseSkill[List[FractalInfo]]):
    """
    分型识别 Skill

    能力:
    - 检测K线序列中的所有分型
    - 过滤指定类型的分型
    - 提供分型确认状态
    - 计算分型强度
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.confirm_required = self.get_config('confirm_required', True)
        self.min_confidence = self.get_config('min_confidence', 0.6)
        self.lookback = self.get_config('lookback', 50)

    def validate(
        self,
        kline: Optional[KLine] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """验证输入参数"""
        if kline is None and data is None:
            return False, "需要提供 kline 或 data 参数"

        if kline is not None and len(kline) < 3:
            return False, "K线数据不足，至少需要3根K线"

        return True, None

    def execute(
        self,
        kline: Optional[KLine] = None,
        data: Optional[Any] = None,
        fractal_type: Optional[str] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        **kwargs
    ) -> SkillResult:
        """
        执行分型识别

        Args:
            kline: K线对象
            data: 原始数据 (DataFrame, 会转换为 KLine)
            fractal_type: 分型类型过滤 ('top', 'bottom', None=全部)
            start_index: 起始索引
            end_index: 结束索引

        Returns:
            SkillResult: 识别的分型列表
        """
        # 准备 KLine 数据
        if kline is None:
            kline = self._data_to_kline(data)
            if kline is None:
                return SkillResult(
                    success=False,
                    error="无法从 data 创建 KLine 对象"
                )

        # 切片数据
        if end_index is None:
            end_index = len(kline)

        if end_index > start_index:
            kline = self._slice_kline(kline, start_index, end_index)

        try:
            # 检测分型
            detector = FractalDetector(kline, confirm_required=self.confirm_required)
            core_fractals = detector.get_fractals()

            # 转换为 FractalInfo
            fractals = []
            for cf in core_fractals:
                # 过滤类型
                if fractal_type is not None:
                    if fractal_type == 'top' and not cf.is_top:
                        continue
                    if fractal_type == 'bottom' and not cf.is_bottom:
                        continue

                # 计算强度
                strength = self._calculate_strength(cf, kline)

                fractals.append(FractalInfo(
                    index=cf.index,
                    fractal_type='top' if cf.is_top else 'bottom',
                    high=cf.high,
                    low=cf.low,
                    confirmed=cf.confirmed,
                    strength=strength
                ))

            # 计算置信度
            confidence = self._calculate_confidence(fractals, kline)

            # 构建元数据
            metadata = {
                'total_count': len(fractals),
                'top_count': len([f for f in fractals if f.fractal_type == 'top']),
                'bottom_count': len([f for f in fractals if f.fractal_type == 'bottom']),
                'confirmed_count': len([f for f in fractals if f.confirmed]),
                'kline_range': (start_index, end_index),
                'avg_strength': sum(f.strength for f in fractals) / len(fractals) if fractals else 0,
            }

            return SkillResult(
                success=True,
                data=fractals,
                confidence=confidence,
                metadata=metadata
            )

        except Exception as e:
            return SkillResult(
                success=False,
                error=f"分型识别失败: {str(e)}"
            )

    def _data_to_kline(self, data: Any) -> Optional[KLine]:
        """将数据转换为 KLine"""
        if data is None:
            return None

        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return KLine.from_dataframe(data)
        except Exception:
            pass

        return None

    def _slice_kline(self, kline: KLine, start: int, end: int) -> KLine:
        """切片K线数据"""
        if start == 0 and end >= len(kline):
            return kline

        # 创建新的 KLine 只包含指定范围
        sliced_data = kline.data[start:end]
        return KLine(sliced_data, strict_mode=kline.strict_mode)

    def _calculate_strength(self, fractal: Fractal, kline: KLine) -> float:
        """
        计算分型强度

        强度基于:
        - 确认状态
        - 分型大小 (high - low)
        - 与周围K线的对比
        """
        strength = 0.5

        # 确认加分
        if fractal.confirmed:
            strength += 0.2

        # 分型大小
        size = fractal.high - fractal.low
        if size > 0:
            # 计算相对大小
            avg_range = (kline.data[max(0, fractal.index - 5):fractal.index + 6].high.max() -
                        kline.data[max(0, fractal.index - 5):fractal.index + 6].low.min())
            if avg_range > 0:
                relative_size = size / avg_range
                strength += min(relative_size * 0.3, 0.3)

        return min(strength, 1.0)

    def _calculate_confidence(self, fractals: List[FractalInfo], kline: KLine) -> float:
        """计算识别置信度"""
        if not fractals:
            return 0.0

        # 基于确认比例计算置信度
        confirmed_ratio = sum(1 for f in fractals if f.confirmed) / len(fractals)

        # 基于平均强度
        avg_strength = sum(f.strength for f in fractals) / len(fractals)

        # 综合置信度
        confidence = (confirmed_ratio + avg_strength) / 2

        return min(confidence, 1.0)

    def get_latest_fractal(
        self,
        kline: KLine,
        fractal_type: str = 'bottom'
    ) -> Optional[FractalInfo]:
        """
        获取最新的指定类型分型

        Args:
            kline: K线对象
            fractal_type: 分型类型 ('top' or 'bottom')

        Returns:
            最新的分型信息，如果没有则返回 None
        """
        result = self.execute(kline=kline, fractal_type=fractal_type, start_index=len(kline) - self.lookback)

        if result.success and result.data:
            return result.data[-1]  # 返回最新的

        return None

    def get_fractal_at_index(
        self,
        kline: KLine,
        index: int,
        lookback: int = 5
    ) -> Optional[FractalInfo]:
        """
        获取指定索引附近的分型

        Args:
            kline: K线对象
            index: 目标索引
            lookback: 向前查找的范围

        Returns:
            找到的分型信息，如果没有则返回 None
        """
        start = max(0, index - lookback)
        end = min(len(kline), index + lookback + 1)

        result = self.execute(kline=kline, start_index=start, end_index=end)

        if result.success and result.data:
            # 找最接近 index 的分型
            closest = min(result.data, key=lambda f: abs(f.index - index))
            if abs(closest.index - index) <= lookback:
                return closest

        return None


# 便捷函数
def detect_fractals(
    kline: KLine,
    confirm_required: bool = True,
    fractal_type: Optional[str] = None
) -> SkillResult:
    """
    便捷函数：检测分型

    Args:
        kline: K线对象
        confirm_required: 是否要求确认
        fractal_type: 分型类型 ('top', 'bottom', None=全部)

    Returns:
        SkillResult
    """
    skill = FractalSkill(config={'confirm_required': confirm_required})
    return skill.execute(kline=kline, fractal_type=fractal_type)


def find_recent_fractals(
    kline: KLine,
    n: int = 5,
    fractal_type: Optional[str] = None
) -> List[FractalInfo]:
    """
    查找最近的 n 个分型

    Args:
        kline: K线对象
        n: 查找数量
        fractal_type: 分型类型

    Returns:
        分型列表
    """
    skill = FractalSkill()
    result = skill.execute(
        kline=kline,
        fractal_type=fractal_type,
        start_index=max(0, len(kline) - 100)
    )

    if result.success and result.data:
        # 返回最新的 n 个
        return result.data[-n:]

    return []
