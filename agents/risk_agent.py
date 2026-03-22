"""
风险管理智能体
评估交易风险，控制仓位和止损
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
from enum import Enum

from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.state import SignalType


class RiskLevel(Enum):
    """风险等级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    EXTREME = "极高"


@dataclass
class RiskAssessmentResult:
    """风险评估结果"""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.5  # 0-1

    # 仓位建议
    recommended_position_ratio: float = 0.5
    max_position_ratio: float = 0.95
    min_position_ratio: float = 0.1

    # 止损止盈建议
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15

    # 风险提示
    warnings: List[str] = field(default_factory=list)
    risk_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "recommended_position_ratio": self.recommended_position_ratio,
            "max_position_ratio": self.max_position_ratio,
            "min_position_ratio": self.min_position_ratio,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "warnings": self.warnings,
            "risk_factors": self.risk_factors,
        }


class RiskAgent(BaseAgent):
    """
    风险管理智能体
    评估交易风险并给出仓位建议
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("risk_agent", config)

        # 默认风险参数
        self.max_position_pct = self.config.get('max_position_pct', 0.95)
        self.default_stop_loss = self.config.get('default_stop_loss', 0.05)
        self.default_take_profit = self.config.get('default_take_profit', 0.15)
        self.max_daily_loss = self.config.get('max_daily_loss', 0.02)  # 单日最大亏损2%

        # 账户状态
        self.current_capital = self.config.get('initial_capital', 100000)
        self.daily_pnl = 0.0
        self.open_positions: List[Dict] = []

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        风险评估
        """
        result = RiskAssessmentResult()

        # 获取信号信息
        signal_result = input_data.previous_results.get('signal_agent')
        trend_result = input_data.previous_results.get('trend_agent')

        if not signal_result or not signal_result.success:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning="无有效交易信号"
            )

        signal_data = signal_result.data

        # 1. 计算基础风险分数
        risk_score = self._calculate_base_risk(signal_data, trend_result)

        # 2. 检查风险因子
        risk_factors = self._check_risk_factors(input_data, signal_data)

        # 3. 调整风险分数
        adjusted_score = self._adjust_risk_score(risk_score, risk_factors)

        # 4. 确定风险等级
        risk_level = self._determine_risk_level(adjusted_score)

        # 5. 计算仓位建议
        position_ratio = self._calculate_position_ratio(
            adjusted_score, risk_level, signal_data
        )

        # 6. 计算止损止盈
        stop_loss, take_profit = self._calculate_stop_take(
            input_data, signal_data, adjusted_score
        )

        # 7. 生成警告
        warnings = self._generate_warnings(risk_factors, adjusted_score)

        result.risk_score = adjusted_score
        result.risk_level = risk_level
        result.recommended_position_ratio = position_ratio
        result.stop_loss_price = stop_loss
        result.take_profit_price = take_profit
        result.stop_loss_pct = abs(signal_data.get('entry_price', 0) - stop_loss) / signal_data.get('entry_price', 1)
        result.take_profit_pct = abs(take_profit - signal_data.get('entry_price', 0)) / signal_data.get('entry_price', 1)
        result.warnings = warnings
        result.risk_factors = risk_factors

        return AgentOutput(
            agent_name=self.name,
            success=True,
            confidence=1.0 - adjusted_score,  # 风险越低信心越高
            reasoning=self._generate_reasoning(result, signal_data),
            data=result.to_dict()
        )

    def _calculate_base_risk(self, signal_data: Dict, trend_result: Any) -> float:
        """计算基础风险分数"""
        base_score = 0.5

        # 根据信号类型调整
        signal_type = signal_data.get('signal_type', '')
        if '1买' in signal_type or '1卖' in signal_type:
            base_score += 0.15  # 第一类买卖点风险较高
        elif '2买' in signal_type or '2卖' in signal_type:
            base_score += 0.10
        elif '3买' in signal_type or '3卖' in signal_type:
            base_score += 0.05  # 第三类买卖点风险相对较低

        # 根据信号信心度调整
        signal_confidence = signal_data.get('confidence', 0.5)
        base_score -= (signal_confidence - 0.5) * 0.3  # 信心度越高，风险越低

        # 根据趋势强度调整
        if trend_result and trend_result.success:
            trend_strength = trend_result.data.get('strength', 0.5)
            base_score -= (trend_strength - 0.5) * 0.2

        return max(0.0, min(1.0, base_score))

    def _check_risk_factors(self, input_data: AgentInput,
                           signal_data: Dict) -> Dict[str, float]:
        """检查各种风险因子"""
        factors = {}

        # 1. 波动率风险
        if input_data.ohlcv_data is not None:
            df = input_data.ohlcv_data
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0
            factors['volatility'] = min(volatility * 10, 1.0)  # 波动率越大风险越高

        # 2. 单日亏损风险
        daily_loss_ratio = abs(self.daily_pnl) / self.current_capital
        factors['daily_loss'] = min(daily_loss_ratio / self.max_daily_loss, 1.0)

        # 3. 持仓集中度风险
        total_position_value = sum(p.get('value', 0) for p in self.open_positions)
        concentration = total_position_value / self.current_capital
        factors['concentration'] = min(concentration, 1.0)

        # 4. 连续亏损风险
        consecutive_losses = self._get_consecutive_losses()
        factors['consecutive_losses'] = min(consecutive_losses * 0.1, 0.5)

        # 5. 市场风险（基于趋势）
        trend_result = input_data.previous_results.get('trend_agent')
        if trend_result and trend_result.success:
            trend_direction = trend_result.data.get('direction', '')
            if trend_direction == 'unknown':
                factors['market_uncertainty'] = 0.3

        return factors

    def _adjust_risk_score(self, base_score: float,
                          factors: Dict[str, float]) -> float:
        """根据风险因子调整分数"""
        adjusted = base_score

        for factor_name, factor_value in factors.items():
            weight = self._get_factor_weight(factor_name)
            adjusted += factor_value * weight

        return max(0.0, min(1.0, adjusted))

    def _get_factor_weight(self, factor_name: str) -> float:
        """获取风险因子权重"""
        weights = {
            'volatility': 0.2,
            'daily_loss': 0.3,
            'concentration': 0.2,
            'consecutive_losses': 0.15,
            'market_uncertainty': 0.1,
        }
        return weights.get(factor_name, 0.1)

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """确定风险等级"""
        if risk_score >= 0.75:
            return RiskLevel.EXTREME
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.35:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_position_ratio(self, risk_score: float, risk_level: RiskLevel,
                                  signal_data: Dict) -> float:
        """计算建议仓位比例"""
        # 基础仓位根据风险等级
        base_ratios = {
            RiskLevel.LOW: 0.8,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.3,
            RiskLevel.EXTREME: 0.1,
        }

        base_ratio = base_ratios.get(risk_level, 0.5)

        # 根据信号信心度调整
        signal_confidence = signal_data.get('confidence', 0.5)
        adjusted_ratio = base_ratio * (0.5 + signal_confidence)

        # 应用最大限制
        return min(adjusted_ratio, self.max_position_pct)

    def _calculate_stop_take(self, input_data: AgentInput, signal_data: Dict,
                             risk_score: float) -> tuple[float, float]:
        """计算止损止盈价格"""
        entry_price = signal_data.get('entry_price', 0)
        if entry_price <= 0:
            return 0.0, 0.0

        # 根据风险等级调整止损幅度
        stop_multipliers = {
            RiskLevel.LOW: 1.5,  # 低风险可以宽一点
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.7,  # 高风险要紧一点
            RiskLevel.EXTREME: 0.5,
        }

        risk_level = self._determine_risk_level(risk_score)
        stop_multiplier = stop_multipliers.get(risk_level, 1.0)

        # 判断是多头还是空头
        signal_type = signal_data.get('signal_type', '')
        is_long = '买' in signal_type

        # 计算止损
        stop_loss_pct = self.default_stop_loss * stop_multiplier
        if is_long:
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + self.default_take_profit)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - self.default_take_profit)

        return stop_loss, take_profit

    def _generate_warnings(self, risk_factors: Dict[str, float],
                          risk_score: float) -> List[str]:
        """生成风险警告"""
        warnings = []

        for factor_name, factor_value in risk_factors.items():
            if factor_value > 0.7:
                if factor_name == 'volatility':
                    warnings.append("⚠️ 市场波动率较高")
                elif factor_name == 'daily_loss':
                    warnings.append("⚠️ 今日亏损已接近限制")
                elif factor_name == 'concentration':
                    warnings.append("⚠️ 持仓集中度过高")
                elif factor_name == 'consecutive_losses':
                    warnings.append("⚠️ 连续亏损，建议暂停交易")
                elif factor_name == 'market_uncertainty':
                    warnings.append("⚠️ 市场趋势不明")

        if risk_score >= 0.75:
            warnings.append("🔴 当前风险等级极高，强烈建议减仓或观望")
        elif risk_score >= 0.6:
            warnings.append("🟠 当前风险等级较高，建议控制仓位")

        return warnings

    def _generate_reasoning(self, result: RiskAssessmentResult,
                           signal_data: Dict) -> str:
        """生成风险分析理由"""
        parts = []

        parts.append(f"风险等级: {result.risk_level.value}")
        parts.append(f"风险分数: {result.risk_score:.2f}")
        parts.append(f"建议仓位: {result.recommended_position_ratio:.0%}")

        signal_type = signal_data.get('signal_type', '')
        is_long = '买' in signal_type
        direction = "做多" if is_long else "做空"

        parts.append(f"止损{result.stop_loss_price:.2f} ({result.stop_loss_pct:.1%})")
        parts.append(f"止盈{result.take_profit_price:.2f} ({result.take_profit_pct:.1%})")

        return " | ".join(parts)

    def _get_consecutive_losses(self) -> int:
        """获取连续亏损次数"""
        # 这里应该从交易记忆中获取
        # 简化实现
        return 0

    def update_capital(self, new_capital: float):
        """更新资金"""
        self.current_capital = new_capital

    def update_daily_pnl(self, pnl: float):
        """更新当日盈亏"""
        self.daily_pnl += pnl

    def add_position(self, position: Dict):
        """添加持仓"""
        self.open_positions.append(position)

    def remove_position(self, symbol: str):
        """移除持仓"""
        self.open_positions = [p for p in self.open_positions if p.get('symbol') != symbol]


class PositionSizingAgent(BaseAgent):
    """
    仓位管理智能体
    根据 Kelly 公式等计算最优仓位
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("position_sizing_agent", config)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)  # Kelly系数（实际使用比例）
        self.fixed_fraction = self.config.get('fixed_fraction', 0.02)  # 固定比例

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        计算最优仓位
        """
        signal_result = input_data.previous_results.get('signal_agent')
        risk_result = input_data.previous_results.get('risk_agent')

        if not signal_result or not signal_result.success:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                reasoning="无有效信号"
            )

        signal_data = signal_result.data
        risk_data = risk_result.data if risk_result else {}

        # 获取历史胜率和盈亏比
        win_rate = self._get_historical_win_rate(signal_data)
        avg_win = self._get_avg_win(signal_data)
        avg_loss = self._get_avg_loss(signal_data)

        # Kelly 公式计算
        kelly_pct = self._calculate_kelly(win_rate, avg_win, avg_loss)

        # 固定比例法
        fixed_pct = self.fixed_fraction

        # 风险调整后的仓位
        risk_adjusted = risk_data.get('recommended_position_ratio', 0.5) if risk_data else 0.5

        # 综合三种方法
        final_ratio = min(
            kelly_pct * self.kelly_fraction,
            fixed_pct,
            risk_adjusted
        )

        # 计算具体股数
        entry_price = signal_data.get('entry_price', 0)
        capital = self.config.get('initial_capital', 100000)
        amount = capital * final_ratio
        shares = int(amount / entry_price / 100) * 100  # 整手

        return AgentOutput(
            agent_name=self.name,
            success=True,
            confidence=final_ratio,
            reasoning=f"Kelly:{kelly_pct:.2%} | 固定:{fixed_pct:.2%} | 风险调整:{risk_adjusted:.2%}",
            data={
                'kelly_ratio': kelly_pct,
                'fixed_ratio': fixed_pct,
                'risk_adjusted_ratio': risk_adjusted,
                'final_ratio': final_ratio,
                'shares': shares,
                'amount': amount
            }
        )

    def _calculate_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """计算 Kelly 公式"""
        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        return max(0, kelly)

    def _get_historical_win_rate(self, signal_data: Dict) -> float:
        """获取历史胜率"""
        # 这里应该从记忆系统获取
        # 简化实现，返回默认值
        return 0.55

    def _get_avg_win(self, signal_data: Dict) -> float:
        """获取平均盈利"""
        return 0.12

    def _get_avg_loss(self, signal_data: Dict) -> float:
        """获取平均亏损"""
        return -0.06
