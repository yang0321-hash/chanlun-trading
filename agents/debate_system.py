"""
多Agent辩论系统

参考 TauricResearch/TradingAgents 的辩论模式:
- Bull Agent: 看多分析员，寻找买入信号
- Bear Agent: 看空分析员，识别风险
- Manager Agent: 综合裁决，做出最终决策
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import json
from pathlib import Path


class Decision(Enum):
    """决策类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class AgentArgument:
    """Agent论点"""
    agent_name: str
    stance: str  # 'bull' or 'bear'
    reasoning: str
    confidence: float  # 0-1
    key_points: List[str]
    data_references: Dict[str, Any]


@dataclass
class DebateResult:
    """辩论结果"""
    decision: Decision
    confidence: float
    bull_arguments: List[AgentArgument]
    bear_arguments: List[AgentArgument]
    final_reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebateState:
    """辩论状态管理"""

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self.current_round = 0
        self.history: List[Dict] = []
        self.bull_history: List[str] = []
        self.bear_history: List[str] = []

    def add_bull_argument(self, argument: AgentArgument):
        """添加多方论点"""
        self.bull_history.append(argument.reasoning)
        self.history.append({
            'round': self.current_round,
            'agent': 'bull',
            'argument': argument
        })

    def add_bear_argument(self, argument: AgentArgument):
        """添加空方论点"""
        self.bear_history.append(argument.reasoning)
        self.history.append({
            'round': self.current_round,
            'agent': 'bear',
            'argument': argument
        })

    def next_round(self) -> bool:
        """进入下一轮"""
        self.current_round += 1
        return self.current_round < self.max_rounds

    def get_opponent_arguments(self, stance: str) -> List[str]:
        """获取对手论点"""
        if stance == 'bull':
            return self.bear_history
        else:
            return self.bull_history


class BullAgent:
    """
    看多分析员

    职责：寻找买入信号，评估上涨潜力
    """

    def __init__(self, name: str = "BullAgent"):
        self.name = name
        self.memory = AgentMemory()

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        index: int,
        context: Dict[str, Any],
        opponent_arguments: List[str] = None
    ) -> AgentArgument:
        """
        看多分析

        关注因素：
        1. 缠论买入信号（分型、笔、中枢）
        2. 趋势状态
        3. 成交量确认
        4. 反驳空方观点
        """
        key_points = []
        confidence = 0.5
        reasoning_parts = []

        # 1. 分析缠论结构
        chanlun_analysis = self._analyze_chanlun(context)
        if chanlun_analysis['bullish']:
            key_points.append(chanlun_analysis['reason'])
            confidence += chanlun_analysis['confidence']
            reasoning_parts.append(chanlun_analysis['reason'])

        # 2. 分析趋势
        trend_analysis = self._analyze_trend(df, index)
        if trend_analysis['bullish']:
            key_points.append(trend_analysis['reason'])
            confidence += trend_analysis['confidence'] * 0.3
            reasoning_parts.append(trend_analysis['reason'])

        # 3. 分析成交量
        volume_analysis = self._analyze_volume(df, index)
        if volume_analysis['bullish']:
            key_points.append(volume_analysis['reason'])
            confidence += volume_analysis['confidence'] * 0.2
            reasoning_parts.append(volume_analysis['reason'])

        # 4. 检索类似情况的记忆
        similar_situations = self.memory.retrieve(similar_context=context)
        if similar_situations:
            lesson = similar_situations[0].get('lesson', '')
            if '盈利' in lesson or '成功' in lesson:
                key_points.append(f"历史经验: {lesson}")
                confidence += 0.1

        # 5. 反驳空方观点
        if opponent_arguments:
            rebuttal = self._rebut_bear(opponent_arguments, context)
            if rebuttal:
                reasoning_parts.append(f"反驳空方: {rebuttal}")

        # 限制置信度范围
        confidence = max(0.1, min(0.95, confidence))

        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "无明显看多信号"

        return AgentArgument(
            agent_name=self.name,
            stance='bull',
            reasoning=reasoning,
            confidence=confidence,
            key_points=key_points,
            data_references={
                'symbol': symbol,
                'price': df['close'].iloc[index],
                'trend': trend_analysis.get('trend'),
                'volume_trend': volume_analysis.get('trend')
            }
        )

    def _analyze_chanlun(self, context: Dict) -> Dict:
        """分析缠论信号"""
        analysis = {'bullish': False, 'reason': '', 'confidence': 0}

        # 检查是否有2买信号
        if 'weekly_second_buy' in context:
            if context['weekly_second_buy']:
                analysis['bullish'] = True
                analysis['reason'] = "周线2买形成"
                analysis['confidence'] = 0.4

        # 检查分型
        if 'fractals' in context:
            fractals = context['fractals']
            bottom_fractals = [f for f in fractals if hasattr(f, 'is_bottom') and f.is_bottom]
            if bottom_fractals:
                analysis['bullish'] = True
                analysis['reason'] = f"发现{len(bottom_fractals)}个底分型"
                analysis['confidence'] = 0.2

        # 检查笔的方向
        if 'strokes' in context:
            strokes = context['strokes']
            if strokes and strokes[-1].is_up:
                analysis['bullish'] = True
                analysis['reason'] = "当前为向上笔"
                analysis['confidence'] = 0.15

        return analysis

    def _analyze_trend(self, df: pd.DataFrame, index: int) -> Dict:
        """分析趋势"""
        if len(df) < 20:
            return {'bullish': False, 'reason': '数据不足', 'confidence': 0}

        current = df['close'].iloc[index]
        ma5 = df['close'].iloc[max(0, index-4):index+1].mean()
        ma20 = df['close'].iloc[max(0, index-19):index+1].mean()

        bullish = current > ma5 > ma20

        return {
            'bullish': bullish,
            'reason': f"价格{'高于' if bullish else '低于'}均线",
            'confidence': 0.3 if bullish else 0,
            'trend': 'up' if bullish else 'down'
        }

    def _analyze_volume(self, df: pd.DataFrame, index: int) -> Dict:
        """分析成交量"""
        if len(df) < 20:
            return {'bullish': False, 'reason': '', 'confidence': 0}

        current_vol = df['volume'].iloc[index]
        avg_vol = df['volume'].iloc[max(0, index-20):index].mean()

        volume_surge = current_vol > avg_vol * 1.2

        return {
            'bullish': volume_surge,
            'reason': f"成交量{'放大' if volume_surge else '正常'}",
            'confidence': 0.2 if volume_surge else 0,
            'trend': 'increasing' if volume_surge else 'normal'
        }

    def _rebut_bear(self, bear_arguments: List[str], context: Dict) -> str:
        """反驳空方观点"""
        rebuttals = []

        for arg in bear_arguments:
            if '震荡' in arg or '盘整' in arg:
                # 震荡末期往往会有方向选择
                if 'price_range' in context:
                    range_ratio = context['price_range']
                    if range_ratio < 0.1:
                        rebuttals.append("震荡收窄，即将突破")
            elif 'MACD' in arg and '背离' in arg:
                rebuttals.append("MACD背离可被后续走势修复")
            elif '成交' in arg and '萎缩' in arg:
                rebuttals.append("缩量回调是健康调整")

        return " | ".join(rebuttals)


class BearAgent:
    """
    看空分析员

    职责：识别风险，评估下跌可能性
    """

    def __init__(self, name: str = "BearAgent"):
        self.name = name
        self.memory = AgentMemory()

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        index: int,
        context: Dict[str, Any],
        opponent_arguments: List[str] = None
    ) -> AgentArgument:
        """
        看空分析

        关注因素：
        1. 风险信号（顶分型、向下笔）
        2. 背离信号
        3. 成交量异常
        4. 反驳多方观点
        """
        key_points = []
        confidence = 0.5
        reasoning_parts = []

        # 1. 分析风险信号
        risk_analysis = self._analyze_risks(context)
        if risk_analysis['bearish']:
            key_points.append(risk_analysis['reason'])
            confidence += risk_analysis['confidence']
            reasoning_parts.append(risk_analysis['reason'])

        # 2. 分析背离
        divergence_analysis = self._analyze_divergence(context)
        if divergence_analysis['bearish']:
            key_points.append(divergence_analysis['reason'])
            confidence += divergence_analysis['confidence'] * 0.3
            reasoning_parts.append(divergence_analysis['reason'])

        # 3. 分析位置风险
        position_analysis = self._analyze_position(df, index)
        if position_analysis['risky']:
            key_points.append(position_analysis['reason'])
            confidence += position_analysis['confidence'] * 0.2
            reasoning_parts.append(position_analysis['reason'])

        # 4. 检索记忆
        similar_situations = self.memory.retrieve(similar_context=context)
        if similar_situations:
            lesson = similar_situations[0].get('lesson', '')
            if '亏损' in lesson or '风险' in lesson:
                key_points.append(f"历史教训: {lesson}")
                confidence += 0.1

        # 5. 反驳多方观点
        if opponent_arguments:
            rebuttal = self._rebut_bull(opponent_arguments, context)
            if rebuttal:
                reasoning_parts.append(f"反驳多方: {rebuttal}")

        # 限制置信度范围
        confidence = max(0.1, min(0.95, confidence))

        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "无明显风险信号"

        return AgentArgument(
            agent_name=self.name,
            stance='bear',
            reasoning=reasoning,
            confidence=confidence,
            key_points=key_points,
            data_references={
                'symbol': symbol,
                'price': df['close'].iloc[index],
                'risks': risk_analysis.get('risks', [])
            }
        )

    def _analyze_risks(self, context: Dict) -> Dict:
        """分析风险信号"""
        analysis = {'bearish': False, 'reason': '', 'confidence': 0, 'risks': []}

        # 检查顶分型
        if 'fractals' in context:
            fractals = context['fractals']
            top_fractals = [f for f in fractals if hasattr(f, 'is_top') and f.is_top]
            if top_fractals:
                analysis['bearish'] = True
                analysis['reason'] = f"发现{len(top_fractals)}个顶分型"
                analysis['confidence'] = 0.25
                analysis['risks'].append('top_fractal')

        # 检查向下笔
        if 'strokes' in context:
            strokes = context['strokes']
            if strokes and not strokes[-1].is_up:
                analysis['bearish'] = True
                analysis['reason'] = "当前为向下笔"
                analysis['confidence'] = 0.2
                analysis['risks'].append('down_stroke')

        # 检查中枢位置
        if 'pivot' in context and context['pivot']:
            pivot = context['pivot']
            current_price = context.get('current_price', 0)
            if pivot.high < current_price * 1.05:
                analysis['bearish'] = True
                analysis['reason'] = "价格接近中枢上沿压力"
                analysis['confidence'] = 0.2
                analysis['risks'].append('pivot_resistance')

        return analysis

    def _analyze_divergence(self, context: Dict) -> Dict:
        """分析背离"""
        analysis = {'bearish': False, 'reason': '', 'confidence': 0}

        if 'macd_divergence' in context:
            if context['macd_divergence']:
                analysis['bearish'] = True
                analysis['reason'] = "MACD顶背离"
                analysis['confidence'] = 0.35

        return analysis

    def _analyze_position(self, df: pd.DataFrame, index: int) -> Dict:
        """分析位置风险"""
        if len(df) < 60:
            return {'risky': False, 'reason': '', 'confidence': 0}

        current = df['close'].iloc[index]
        high_60 = df['high'].iloc[max(0, index-60):index+1].max()
        low_60 = df['low'].iloc[max(0, index-60):index+1].min()

        # 计算相对位置
        position = (current - low_60) / (high_60 - low_60) if high_60 > low_60 else 0.5

        # 高位风险
        risky = position > 0.8

        return {
            'risky': risky,
            'reason': f"价格处于{position:.1%}分位" + ("，高位风险" if risky else "") ,
            'confidence': 0.25 if risky else 0
        }

    def _rebut_bull(self, bull_arguments: List[str], context: Dict) -> str:
        """反驳多方观点"""
        rebuttals = []

        for arg in bull_arguments:
            if '2买' in arg:
                rebuttals.append("2买位置可能形成复杂中枢")
            elif '突破' in arg:
                rebuttals.append("突破可能失败，需要确认")
            elif '放量' in arg:
                rebuttals.append("放量可能是诱多")

        return " | ".join(rebuttals)


class ManagerAgent:
    """
    管理员Agent

    职责：综合多方空方论点，做出最终决策
    """

    def __init__(self, name: str = "ManagerAgent"):
        self.name = name

    def evaluate(
        self,
        bull_arguments: List[AgentArgument],
        bear_arguments: List[AgentArgument],
        context: Dict[str, Any]
    ) -> DebateResult:
        """
        综合评估辩论结果

        评估标准：
        1. 双方置信度差值
        2. 论点数量和质量
        3. 市场环境权重
        4. 风险收益比
        """
        # 计算平均置信度
        bull_conf = sum(a.confidence for a in bull_arguments) / len(bull_arguments) if bull_arguments else 0.3
        bear_conf = sum(a.confidence for a in bear_arguments) / len(bear_arguments) if bear_arguments else 0.3

        # 计算论点强度
        bull_score = self._calculate_score(bull_arguments)
        bear_score = self._calculate_score(bear_arguments)

        # 市场环境调整
        market_adjustment = self._get_market_adjustment(context)

        # 综合评分
        final_bull_score = bull_score + market_adjustment
        final_bear_score = bear_score - market_adjustment

        # 决策
        diff = final_bull_score - final_bear_score

        if diff > 0.3:
            decision = Decision.BUY
            confidence = min(0.9, 0.5 + diff / 2)
        elif diff < -0.3:
            decision = Decision.SELL
            confidence = min(0.9, 0.5 + abs(diff) / 2)
        else:
            decision = Decision.HOLD
            confidence = 0.5

        # 生成决策理由
        reasoning = self._generate_reasoning(
            bull_arguments, bear_arguments,
            final_bull_score, final_bear_score,
            decision
        )

        return DebateResult(
            decision=decision,
            confidence=confidence,
            bull_arguments=bull_arguments,
            bear_arguments=bear_arguments,
            final_reasoning=reasoning,
            metadata={
                'bull_score': final_bull_score,
                'bear_score': final_bear_score,
                'score_diff': diff
            }
        )

    def _calculate_score(self, arguments: List[AgentArgument]) -> float:
        """计算论点得分"""
        if not arguments:
            return 0.3

        score = 0
        for arg in arguments:
            # 基础置信度
            score += arg.confidence

            # 关键点数量
            score += len(arg.key_points) * 0.05

        return score / len(arguments) if arguments else 0.3

    def _get_market_adjustment(self, context: Dict) -> float:
        """获取市场环境调整系数"""
        adjustment = 0

        # 市场环境偏好
        if 'market_regime' in context:
            regime = context['market_regime']
            if regime == 'uptrend':
                adjustment += 0.2
            elif regime == 'downtrend':
                adjustment -= 0.2

        # 波动率调整
        if 'volatility' in context:
            vol = context['volatility']
            if vol > 0.05:  # 高波动
                adjustment -= 0.1

        return adjustment

    def _generate_reasoning(
        self,
        bull_args: List[AgentArgument],
        bear_args: List[AgentArgument],
        bull_score: float,
        bear_score: float,
        decision: Decision
    ) -> str:
        """生成决策理由"""
        parts = []

        # 多方观点
        if bull_args:
            bull_points = []
            for arg in bull_args:
                bull_points.extend(arg.key_points[:2])
            parts.append(f"多头观点: {', '.join(bull_points[:3])}")

        # 空方观点
        if bear_args:
            bear_points = []
            for arg in bear_args:
                bear_points.extend(arg.key_points[:2])
            parts.append(f"空头观点: {', '.join(bear_points[:3])}")

        # 评分
        parts.append(f"多空得分: {bull_score:.2f} vs {bear_score:.2f}")

        # 决策
        decision_map = {
            Decision.BUY: "买入",
            Decision.SELL: "卖出",
            Decision.HOLD: "观望"
        }
        parts.append(f"最终决策: {decision_map[decision]}")

        return " | ".join(parts)


class AgentMemory:
    """Agent记忆系统"""

    def __init__(self, memory_file: str = ".chanlun/agent_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory: List[Dict] = []
        self._load_memory()

    def _load_memory(self):
        """加载记忆"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
            except:
                self.memory = []

    def _save_memory(self):
        """保存记忆"""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def save(self, situation: Dict, outcome: str, lesson: str):
        """保存记忆"""
        memory_entry = {
            'situation': situation,
            'outcome': outcome,
            'lesson': lesson,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.memory.append(memory_entry)
        self._save_memory()

    def retrieve(self, similar_context: Dict, n_matches: int = 2) -> List[Dict]:
        """检索相似情况的记忆"""
        # 简化实现：返回最近的记忆
        # 实际可以使用相似度匹配
        if not self.memory:
            return []

        return self.memory[-n_matches:]


class ChanLunDebateSystem:
    """
    缠论多Agent辩论系统

    整合Bull Agent、Bear Agent和Manager Agent
    """

    def __init__(self, max_rounds: int = 2):
        self.bull_agent = BullAgent()
        self.bear_agent = BearAgent()
        self.manager_agent = ManagerAgent()
        self.max_rounds = max_rounds
        self.memory = AgentMemory()

    def debate(
        self,
        symbol: str,
        df: pd.DataFrame,
        index: int,
        context: Dict[str, Any]
    ) -> DebateResult:
        """
        运行辩论

        Args:
            symbol: 股票代码
            df: K线数据
            index: 当前索引
            context: 上下文信息（缠论信号等）

        Returns:
            DebateResult: 辩论结果
        """
        state = DebateState(max_rounds=self.max_rounds)

        bull_arguments = []
        bear_arguments = []

        # 第一轮：初始论点
        bull_arg = self.bull_agent.analyze(symbol, df, index, context)
        bear_arg = self.bear_agent.analyze(symbol, df, index, context)

        state.add_bull_argument(bull_arg)
        state.add_bear_argument(bear_arg)

        bull_arguments.append(bull_arg)
        bear_arguments.append(bear_arg)

        # 后续轮次：互相反驳
        while state.next_round():
            bull_arg = self.bull_agent.analyze(
                symbol, df, index, context,
                opponent_arguments=state.get_opponent_arguments('bull')
            )
            bear_arg = self.bear_agent.analyze(
                symbol, df, index, context,
                opponent_arguments=state.get_opponent_arguments('bear')
            )

            state.add_bull_argument(bull_arg)
            state.add_bear_argument(bear_arg)

            bull_arguments.append(bull_arg)
            bear_arguments.append(bear_arg)

        # Manager 综合评估
        result = self.manager_agent.evaluate(bull_arguments, bear_arguments, context)

        # 保存辩论记录
        self._save_debate(symbol, context, result)

        return result

    def _save_debate(self, symbol: str, context: Dict, result: DebateResult):
        """保存辩论记录"""
        debate_record = {
            'symbol': symbol,
            'decision': result.decision.value,
            'confidence': result.confidence,
            'bull_points': [a.key_points for a in result.bull_arguments],
            'bear_points': [a.key_points for a in result.bear_arguments],
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # 可以保存到文件用于后续分析
        pass

    def reflect(
        self,
        symbol: str,
        decision: Decision,
        actual_outcome: str,
        profit_loss: float
    ):
        """
        反思学习

        根据实际结果更新记忆
        """
        outcome_type = "盈利" if profit_loss > 0 else "亏损"
        lesson = f"{decision.value}决策后{outcome_type}{abs(profit_loss):.2%}"

        # 保存到记忆
        situation = {
            'symbol': symbol,
            'decision': decision.value,
            'outcome': actual_outcome
        }

        self.memory.save(situation, actual_outcome, lesson)
