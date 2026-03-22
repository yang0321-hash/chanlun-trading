"""
趋势分析智能体
判断当前趋势方向和强度
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.state import TrendDirection


@dataclass
class TrendAnalysisResult:
    """趋势分析结果"""
    direction: TrendDirection = TrendDirection.UNKNOWN
    strength: float = 0.0  # 0-1，趋势强度
    duration: int = 0  # 趋势持续天数

    # 均线分析
    ma_alignment: str = ""  # "多头排列", "空头排列", "交织"
    price_vs_ma: Dict[str, str] = None  # 价格与各均线关系

    # 波动分析
    volatility: float = 0.0
    atr: float = 0.0

    # 支撑阻力
    support_levels: List[float] = None
    resistance_levels: List[float] = None

    def __post_init__(self):
        if self.price_vs_ma is None:
            self.price_vs_ma = {}
        if self.support_levels is None:
            self.support_levels = []
        if self.resistance_levels is None:
            self.resistance_levels = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction.value,
            "strength": self.strength,
            "duration": self.duration,
            "ma_alignment": self.ma_alignment,
            "price_vs_ma": self.price_vs_ma,
            "volatility": self.volatility,
            "atr": self.atr,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
        }


class TrendAgent(BaseAgent):
    """
    趋势分析智能体
    使用多种方法判断趋势
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("trend_agent", config)
        self.ma_periods = self.config.get('ma_periods', [5, 10, 20, 60])
        self.trend_lookback = self.config.get('trend_lookback', 20)

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        分析趋势
        """
        if input_data.ohlcv_data is None or len(input_data.ohlcv_data) < 60:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning="数据不足"
            )

        try:
            df = input_data.ohlcv_data
            current_index = input_data.current_index

            # 计算各项指标
            ma_analysis = self._analyze_ma(df, current_index)
            adx_analysis = self._calculate_adx(df, current_index)
            price_structure = self._analyze_price_structure(df, current_index)
            support_resistance = self._find_support_resistance(df, current_index)

            # 综合判断趋势
            direction, confidence = self._determine_trend(
                ma_analysis, adx_analysis, price_structure
            )

            result = TrendAnalysisResult(
                direction=direction,
                strength=adx_analysis.get('strength', 0.5),
                duration=price_structure.get('duration', 0),
                ma_alignment=ma_analysis.get('alignment', ''),
                price_vs_ma=ma_analysis.get('price_vs_ma', {}),
                volatility=price_structure.get('volatility', 0),
                atr=adx_analysis.get('atr', 0),
                support_levels=support_resistance.get('support', []),
                resistance_levels=support_resistance.get('resistance', [])
            )

            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=confidence,
                reasoning=self._generate_reasoning(result),
                data=result.to_dict()
            )

        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                confidence=0.0,
                reasoning=f"分析出错: {str(e)}"
            )

    def _analyze_ma(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """分析均线"""
        close = df['close']

        ma_values = {}
        price_vs_ma = {}

        current_price = close.iloc[current_index]

        for period in self.ma_periods:
            if len(close) >= period:
                ma = close.rolling(window=period).mean().iloc[current_index]
                ma_values[f'ma{period}'] = ma

                # 判断价格与均线关系
                if current_price > ma:
                    price_vs_ma[f'ma{period}'] = 'above'
                elif current_price < ma:
                    price_vs_ma[f'ma{period}'] = 'below'
                else:
                    price_vs_ma[f'ma{period}'] = 'at'

        # 判断均线排列
        alignment = self._determine_ma_alignment(ma_values)

        return {
            'ma_values': ma_values,
            'price_vs_ma': price_vs_ma,
            'alignment': alignment
        }

    def _determine_ma_alignment(self, ma_values: Dict[str, float]) -> str:
        """判断均线排列"""
        mas = [ma_values.get(f'ma{p}') for p in sorted(self.ma_periods)
               if f'ma{p}' in ma_values]

        if not mas or len(mas) < 2:
            return "数据不足"

        # 检查多头排列 (短 > 中 > 长)
        bullish = all(mas[i] > mas[i+1] for i in range(len(mas)-1))
        if bullish:
            return "多头排列"

        # 检查空头排列 (短 < 中 < 长)
        bearish = all(mas[i] < mas[i+1] for i in range(len(mas)-1))
        if bearish:
            return "空头排列"

        return "交织"

    def _calculate_adx(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """计算ADX（平均趋向指数）"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']

            period = 14

            # 计算+DM和-DM
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            # 计算TR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # 计算平滑的+DI和-DI
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

            # 计算DX和ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()

            current_adx = adx.iloc[current_index] if current_index < len(adx) else 0
            current_plus_di = plus_di.iloc[current_index] if current_index < len(plus_di) else 50
            current_minus_di = minus_di.iloc[current_index] if current_index < len(minus_di) else 50

            # 判断趋势强度
            strength = min(current_adx / 50, 1.0)  # ADX>25为有趋势，50为强趋势

            # 判断趋势方向
            direction = 'up' if current_plus_di > current_minus_di else 'down'

            return {
                'adx': current_adx,
                'plus_di': current_plus_di,
                'minus_di': current_minus_di,
                'strength': strength,
                'direction': direction,
                'atr': atr.iloc[current_index] if current_index < len(atr) else 0
            }

        except Exception as e:
            return {'strength': 0.5, 'atr': 0, 'adx': 0}

    def _analyze_price_structure(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """分析价格结构"""
        lookback = min(self.trend_lookback, current_index)

        prices = df['close'].iloc[current_index - lookback:current_index + 1]

        # 计算趋势方向
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        change_pct = (end_price - start_price) / start_price

        # 判断是上涨还是下跌趋势
        if change_pct > 0.03:  # 3%以上认为上涨
            direction = 'up'
        elif change_pct < -0.03:
            direction = 'down'
        else:
            direction = 'sideways'

        # 计算波动率
        returns = prices.pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0

        return {
            'direction': direction,
            'change_pct': change_pct,
            'volatility': volatility,
            'duration': lookback
        }

    def _find_support_resistance(self, df: pd.DataFrame, current_index: int,
                                  n_levels: int = 3) -> Dict[str, List[float]]:
        """寻找支撑和阻力位"""
        lookback = min(100, current_index)
        data = df.iloc[current_index - lookback:current_index + 1]

        highs = data['high'].values
        lows = data['low'].values

        # 简单的支撑阻力算法：找局部高低点
        from scipy.signal import argrelextrema

        # 找局部高点作为阻力
        resistance_indices = argrelextrema(highs, np.greater, order=5)[0]
        resistance_levels = sorted(highs[resistance_indices], reverse=True)[:n_levels]

        # 找局部低点作为支撑
        support_indices = argrelextrema(lows, np.less, order=5)[0]
        support_levels = sorted(lows[support_indices])[:n_levels]

        return {
            'support': support_levels.tolist(),
            'resistance': resistance_levels.tolist()
        }

    def _determine_trend(self, ma_analysis: Dict, adx_analysis: Dict,
                         price_structure: Dict) -> tuple[TrendDirection, float]:
        """综合判断趋势"""
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # 1. 均线排列
        if ma_analysis['alignment'] == '多头排列':
            bullish_signals += 2
        elif ma_analysis['alignment'] == '空头排列':
            bearish_signals += 2
        total_signals += 2

        # 2. ADX方向
        if adx_analysis['direction'] == 'up':
            bullish_signals += 1
        else:
            bearish_signals += 1
        total_signals += 1

        # 3. 价格结构
        if price_structure['direction'] == 'up':
            bullish_signals += 1
        elif price_structure['direction'] == 'down':
            bearish_signals += 1
        total_signals += 1

        # 确定趋势方向
        if bullish_signals > bearish_signals:
            direction = TrendDirection.UP
            confidence = (bullish_signals / total_signals) * adx_analysis['strength']
        elif bearish_signals > bullish_signals:
            direction = TrendDirection.DOWN
            confidence = (bearish_signals / total_signals) * adx_analysis['strength']
        else:
            direction = TrendDirection.UNKNOWN
            confidence = 0.5

        return direction, max(0.1, min(0.95, confidence))

    def _generate_reasoning(self, result: TrendAnalysisResult) -> str:
        """生成分析理由"""
        parts = []

        # 趋势方向
        direction_map = {
            TrendDirection.UP: "上涨",
            TrendDirection.DOWN: "下跌",
            TrendDirection.UNKNOWN: "震荡"
        }
        parts.append(f"趋势{direction_map[result.direction]}")

        # 趋势强度
        if result.strength > 0.7:
            parts.append("趋势强劲")
        elif result.strength > 0.4:
            parts.append("趋势中等")
        else:
            parts.append("趋势较弱")

        # 均线排列
        if result.ma_alignment:
            parts.append(result.ma_alignment)

        return "，".join(parts)


class TrendConfirmationAgent(BaseAgent):
    """
    趋势确认智能体
    专门用于确认趋势反转
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("trend_confirmation_agent", config)

    def analyze(self, input_data: AgentInput) -> AgentOutput:
        """
        检测趋势反转信号
        """
        if input_data.ohlcv_data is None or len(input_data.ohlcv_data) < 20:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                reasoning="数据不足"
            )

        df = input_data.ohlcv_data
        current_index = input_data.current_index

        # 检测反转形态
        reversals = self._detect_reversals(df, current_index)

        if reversals['confirmed']:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                confidence=reversals['confidence'],
                reasoning=reversals['reasoning'],
                data={'reversal_type': reversals['type']}
            )

        return AgentOutput(
            agent_name=self.name,
            success=True,
            confidence=0.3,
            reasoning="未发现明确反转信号",
            data={'reversal_type': None}
        )

    def _detect_reversals(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """检测反转形态"""
        close = df['close']
        high = df['high']
        low = df['low']

        # 检测金叉/死叉
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(20).mean()

        if len(ma_short) < 2:
            return {'confirmed': False, 'type': None, 'confidence': 0, 'reasoning': ''}

        # 金叉
        if (ma_short.iloc[current_index - 1] <= ma_long.iloc[current_index - 1] and
            ma_short.iloc[current_index] > ma_long.iloc[current_index]):
            return {
                'confirmed': True,
                'type': 'golden_cross',
                'confidence': 0.7,
                'reasoning': '5日线上穿20日线，金叉看多'
            }

        # 死叉
        if (ma_short.iloc[current_index - 1] >= ma_long.iloc[current_index - 1] and
            ma_short.iloc[current_index] < ma_long.iloc[current_index]):
            return {
                'confirmed': True,
                'type': 'death_cross',
                'confidence': 0.7,
                'reasoning': '5日线下穿20日线，死叉看空'
            }

        return {
            'confirmed': False,
            'type': None,
            'confidence': 0,
            'reasoning': '均线未出现交叉'
        }
