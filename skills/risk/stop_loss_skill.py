"""
止损 Skill - 计算动态止损价格
"""
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from skills.base import BaseSkill, SkillResult


class StopLossMethod(Enum):
    """止损方法"""
    FIXED = 'fixed'          # 固定百分比
    PIVOT = 'pivot'          # 中枢止损
    FRACTAL = 'fractal'      # 分型止损
    TRAILING = 'trailing'    # 移动止损
    ATR = 'atr'              # ATR止损


@dataclass
class StopLossResult:
    """止损结果"""
    stop_loss_price: float
    stop_loss_pct: float
    reason: str
    method: str
    original_stop: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stop_loss_price': self.stop_loss_price,
            'stop_loss_pct': self.stop_loss_pct,
            'reason': self.reason,
            'method': self.method,
            'original_stop': self.original_stop,
        }


class StopLossSkill(BaseSkill[StopLossResult]):
    """
    止损计算 Skill

    能力:
    - 固定百分比止损
    - 中枢止损
    - 分型止损
    - 移动止损
    - ATR止损
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_stop_pct = self.get_config('default_stop_pct', 0.08)
        self.trailing_stop_pct = self.get_config('trailing_stop_pct', 0.05)
        self.atr_multiplier = self.get_config('atr_multiplier', 2.0)
        self.min_stop_pct = self.get_config('min_stop_pct', 0.03)
        self.max_stop_pct = self.get_config('max_stop_pct', 0.20)

    def validate(
        self,
        entry_price: float = 0,
        current_price: float = 0,
        position_type: str = 'long',
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """验证输入参数"""
        if entry_price <= 0:
            return False, "入场价格必须大于0"
        if current_price <= 0:
            return False, "当前价格必须大于0"
        if position_type not in ['long', 'short']:
            return False, "持仓方向必须是 'long' 或 'short'"
        return True, None

    def execute(
        self,
        entry_price: float,
        current_price: float,
        position_type: str = 'long',
        method: str = 'fixed',
        pivot: Optional[Dict[str, float]] = None,
        fractal: Optional[Dict[str, float]] = None,
        atr: Optional[float] = None,
        highest_since_entry: Optional[float] = None,
        lowest_since_entry: Optional[float] = None,
        **kwargs
    ) -> SkillResult:
        """
        计算止损价格

        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_type: 持仓方向 ('long' or 'short')
            method: 止损方法
            pivot: 中枢数据 {'low': x, 'high': y}
            fractal: 分型数据 {'low': x, 'high': y}
            atr: ATR值
            highest_since_entry: 入场后最高价 (多头移动止损)
            lowest_since_entry: 入场后最低价 (空头移动止损)

        Returns:
            SkillResult: 止损计算结果
        """
        # 参数验证
        is_valid, error = self.validate(
            entry_price=entry_price,
            current_price=current_price,
            position_type=position_type
        )
        if not is_valid:
            return SkillResult(success=False, error=error)

        try:
            if method == 'fixed':
                result = self._fixed_stop_loss(entry_price, position_type)
            elif method == 'pivot':
                result = self._pivot_stop_loss(entry_price, pivot, position_type)
            elif method == 'fractal':
                result = self._fractal_stop_loss(entry_price, fractal, position_type)
            elif method == 'trailing':
                result = self._trailing_stop_loss(
                    entry_price, current_price, position_type,
                    highest_since_entry, lowest_since_entry
                )
            elif method == 'atr':
                result = self._atr_stop_loss(entry_price, atr, position_type)
            else:
                # 降级为固定止损
                result = self._fixed_stop_loss(entry_price, position_type)

            # 确保止损百分比在合理范围内
            result.stop_loss_pct = max(
                self.min_stop_pct,
                min(self.max_stop_pct, result.stop_loss_pct)
            )

            return SkillResult(
                success=True,
                data=result,
                confidence=1.0,
                metadata={
                    'method': method,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'current_price': current_price,
                }
            )

        except Exception as e:
            return SkillResult(
                success=False,
                error=f"止损计算失败: {str(e)}"
            )

    def _fixed_stop_loss(
        self,
        entry_price: float,
        position_type: str
    ) -> StopLossResult:
        """固定百分比止损"""
        if position_type == 'long':
            stop_loss_price = entry_price * (1 - self.default_stop_pct)
            stop_loss_pct = self.default_stop_pct
        else:
            stop_loss_price = entry_price * (1 + self.default_stop_pct)
            stop_loss_pct = self.default_stop_pct

        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            reason=f"固定{self.default_stop_pct*100:.1f}%止损",
            method='fixed',
            original_stop=stop_loss_price
        )

    def _pivot_stop_loss(
        self,
        entry_price: float,
        pivot: Optional[Dict[str, float]],
        position_type: str
    ) -> StopLossResult:
        """中枢止损"""
        if pivot is None or 'low' not in pivot or 'high' not in pivot:
            # 降级为固定止损
            return self._fixed_stop_loss(entry_price, position_type)

        if position_type == 'long':
            # 多头止损在中枢下沿
            stop_loss_price = pivot['low']
        else:
            # 空头止损在中枢上沿
            stop_loss_price = pivot['high']

        stop_loss_pct = abs(stop_loss_price - entry_price) / entry_price

        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            reason=f"中枢止损 [{pivot['low']:.2f}, {pivot['high']:.2f}]",
            method='pivot',
            original_stop=stop_loss_price
        )

    def _fractal_stop_loss(
        self,
        entry_price: float,
        fractal: Optional[Dict[str, float]],
        position_type: str
    ) -> StopLossResult:
        """分型止损"""
        if fractal is None or 'low' not in fractal or 'high' not in fractal:
            # 降级为固定止损
            return self._fixed_stop_loss(entry_price, position_type)

        if position_type == 'long':
            # 多头止损在分型低点
            stop_loss_price = fractal['low']
        else:
            # 空头止损在分型高点
            stop_loss_price = fractal['high']

        stop_loss_pct = abs(stop_loss_price - entry_price) / entry_price

        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            reason=f"分型止损 {fractal.get('type', '')} @ {stop_loss_price:.2f}",
            method='fractal',
            original_stop=stop_loss_price
        )

    def _trailing_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_type: str,
        highest_since_entry: Optional[float],
        lowest_since_entry: Optional[float]
    ) -> StopLossResult:
        """移动止损"""
        if position_type == 'long':
            if highest_since_entry is None:
                highest_since_entry = max(entry_price, current_price)

            # 移动止损价格 = 最高价 * (1 - 移动止损比例)
            stop_loss_price = highest_since_entry * (1 - self.trailing_stop_pct)
            stop_loss_pct = self.trailing_stop_pct

            # 如果当前价格低于止损价，使用当前价格
            if stop_loss_price > current_price:
                stop_loss_price = current_price * 0.99

        else:  # short
            if lowest_since_entry is None:
                lowest_since_entry = min(entry_price, current_price)

            stop_loss_price = lowest_since_entry * (1 + self.trailing_stop_pct)
            stop_loss_pct = self.trailing_stop_pct

            # 如果当前价格高于止损价，使用当前价格
            if stop_loss_price < current_price:
                stop_loss_price = current_price * 1.01

        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            reason=f"移动止损 {self.trailing_stop_pct*100:.1f}%",
            method='trailing',
            original_stop=stop_loss_price
        )

    def _atr_stop_loss(
        self,
        entry_price: float,
        atr: Optional[float],
        position_type: str
    ) -> StopLossResult:
        """ATR止损"""
        if atr is None or atr <= 0:
            return self._fixed_stop_loss(entry_price, position_type)

        if position_type == 'long':
            stop_loss_price = entry_price - (atr * self.atr_multiplier)
            stop_loss_pct = (atr * self.atr_multiplier) / entry_price
        else:
            stop_loss_price = entry_price + (atr * self.atr_multiplier)
            stop_loss_pct = (atr * self.atr_multiplier) / entry_price

        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            reason=f"ATR止损 ({self.atr_multiplier}xATR={atr:.2f})",
            method='atr',
            original_stop=stop_loss_price
        )


# 便捷函数
def calculate_stop_loss(
    entry_price: float,
    current_price: float,
    position_type: str = 'long',
    method: str = 'fixed',
    stop_pct: float = 0.08
) -> StopLossResult:
    """
    便捷函数：计算止损

    Args:
        entry_price: 入场价格
        current_price: 当前价格
        position_type: 持仓方向
        method: 止损方法
        stop_pct: 止损百分比

    Returns:
        StopLossResult
    """
    skill = StopLossSkill(config={'default_stop_pct': stop_pct})
    result = skill.execute(
        entry_price=entry_price,
        current_price=current_price,
        position_type=position_type,
        method=method
    )
    return result.data if result.success else None


def calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    highest_since_entry: Optional[float] = None,
    trailing_pct: float = 0.05
) -> StopLossResult:
    """
    便捷函数：计算移动止损

    Args:
        entry_price: 入场价格
        current_price: 当前价格
        highest_since_entry: 入场后最高价
        trailing_pct: 移动止损百分比

    Returns:
        StopLossResult
    """
    skill = StopLossSkill(config={'trailing_stop_pct': trailing_pct})
    result = skill.execute(
        entry_price=entry_price,
        current_price=current_price,
        position_type='long',
        method='trailing',
        highest_since_entry=highest_since_entry
    )
    return result.data if result.success else None
