"""
缠论分析 Agent - 使用 Skills 完成缠论分析
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

from agents.base_agent import BaseAgent, AgentConfig, AgentState, register_agent
from skills.base import SkillResult
from skills.pattern.fractal_skill import FractalSkill
from skills.signal.buy_point_skill import BuyPointSkill, BuyPointType
from skills.risk.stop_loss_skill import StopLossSkill, StopLossMethod
from core.kline import KLine
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD


@register_agent('chanlun')
class ChanLunAgent(BaseAgent):
    """
    缠论分析 Agent

    职责:
    1. 协调多个 Skills 完成缠论分析
    2. 生成买卖信号
    3. 计算止损价格
    4. 评估信号质量
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name='缠论分析Agent',
                description='基于缠论理论的交易信号分析',
                skills=['fractal', 'buy_point', 'stop_loss'],
                skill_configs={
                    'fractal': {'confirm_required': True},
                    'buy_point': {'min_confidence': 0.6},
                    'stop_loss': {'default_stop_pct': 0.08},
                },
                agent_config={
                    'lookback': 100,
                    'min_data_points': 50,
                }
            )
        super().__init__(config)

        # 设置默认值（如果配置中没有）
        self.lookback = getattr(self, 'lookback', 100)
        self.min_data_points = getattr(self, 'min_data_points', 50)

        # 缓存
        self._kline_cache: Dict[str, KLine] = {}
        self._stroke_cache: Dict[str, List] = {}
        self._pivot_cache: Dict[str, List] = {}
        self._macd_cache: Dict[str, Any] = {}

    def _register_skills(self) -> None:
        """注册 Skills"""
        # 分型识别 Skill
        fractal_config = self.config.get_skill_config('fractal')
        self.register_skill('fractal', FractalSkill(config=fractal_config))

        # 买点识别 Skill
        buy_point_config = self.config.get_skill_config('buy_point')
        self.register_skill('buy_point', BuyPointSkill(config=buy_point_config))

        # 止损 Skill
        stop_loss_config = self.config.get_skill_config('stop_loss')
        self.register_skill('stop_loss', StopLossSkill(config=stop_loss_config))

    def analyze(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整的缠论分析流程

        Args:
            symbol: 股票代码
            context: 上下文数据，应包含 'data' 键存储 DataFrame

        Returns:
            分析结果字典
        """
        self.log(f"开始分析 {symbol}")

        # 1. 准备数据
        kline = self._prepare_kline(symbol, context)
        if kline is None:
            return self._error_result(symbol, "K线数据准备失败")

        # 2. 识别基础结构
        strokes = self._detect_strokes(kline)
        pivots = self._detect_pivots(kline, strokes)

        # 3. 计算指标
        macd = self._calculate_macd(kline)

        # 4. 识别分型
        current_index = len(kline) - 1
        fractal_result = self.execute_skill(
            'fractal',
            kline=kline,
            start_index=max(0, current_index - self.lookback)
        )

        # 5. 识别买点
        buy_point_result = self.execute_skill(
            'buy_point',
            kline=kline,
            strokes=strokes,
            pivots=pivots,
            current_index=current_index
        )

        # 6. 生成信号
        signal = self._generate_signal(
            symbol, kline, buy_point_result, current_index
        )

        # 7. 构建分析结果
        result = {
            'symbol': symbol,
            'success': True,
            'signal': signal.get('action', 'hold'),
            'signal_type': signal.get('type'),
            'signal_confidence': signal.get('confidence', 0),
            'fractals': fractal_result.data if fractal_result else [],
            'strokes_count': len(strokes),
            'pivots_count': len(pivots),
            'buy_points': buy_point_result.data if buy_point_result else [],
            'buy_confidence': buy_point_result.confidence if buy_point_result else 0,
            'current_price': kline.data[current_index].close,
            'analysis_time': datetime.now().isoformat(),
        }

        # 8. 更新状态
        self.state.update_analysis(symbol, 'last_result', result)
        self.state.update_analysis(symbol, 'last_signal', signal)

        self.log(f"{symbol} 分析完成: 信号={result['signal']}, "
               f"买点={len(result['buy_points'])}个")

        return result

    def generate_trade_plan(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        capital: float = 100000,
        position: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        生成交易计划

        Args:
            symbol: 股票代码
            analysis: 分析结果
            capital: 可用资金
            position: 当前持仓

        Returns:
            交易计划或None
        """
        if not analysis.get('success'):
            return None

        signal = analysis['signal']
        current_price = analysis['current_price']
        buy_points = analysis['buy_points']

        # 已有持仓：检查卖出
        if position and position > 0:
            if signal == 'sell':
                entry_price = self.state.get_analysis(symbol, 'entry_price', current_price)
                stop_loss_result = self.execute_skill(
                    'stop_loss',
                    entry_price=entry_price,
                    current_price=current_price,
                    position_type='long',
                    method='trailing',
                    highest_since_entry=analysis.get('highest_price', current_price)
                )

                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': position,
                    'price': current_price,
                    'reason': analysis.get('sell_reason', '信号卖出'),
                    'stop_loss': stop_loss_result.data.stop_loss_price if stop_loss_result else None,
                }

        # 无持仓：检查买入
        elif signal == 'buy' and buy_points:
            best_buy = buy_points[0]
            if best_buy.confidence >= 0.6:
                # 计算止损
                stop_loss_method = 'pivot' if best_buy.pivot_low else 'fixed'
                pivot = None
                if best_buy.pivot_low:
                    pivot = {'low': best_buy.pivot_low, 'high': best_buy.pivot_high or best_buy.pivot_low * 1.1}

                stop_loss_result = self.execute_skill(
                    'stop_loss',
                    entry_price=current_price,
                    current_price=current_price,
                    method=stop_loss_method,
                    position_type='long',
                    pivot=pivot
                )

                # 计算仓位 (使用固定比例)
                position_ratio = 0.3  # 30% 仓位
                amount = capital * position_ratio
                shares = int(amount / current_price / 100) * 100  # 整手

                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': shares,
                    'price': current_price,
                    'reason': best_buy.reason,
                    'confidence': best_buy.confidence,
                    'stop_loss': stop_loss_result.data.stop_loss_price if stop_loss_result else None,
                    'target': best_buy.target,
                }

        return None

    def _prepare_kline(self, symbol: str, context: Dict[str, Any]) -> Optional[KLine]:
        """准备K线数据"""
        if symbol in self._kline_cache:
            return self._kline_cache[symbol]

        # 从 context 获取数据
        data = context.get('data', {}).get(symbol)
        if data is None:
            data = context.get('data')

        if data is None or len(data) < self.min_data_points:
            return None

        kline = KLine.from_dataframe(data)
        self._kline_cache[symbol] = kline
        return kline

    def _detect_strokes(self, kline: KLine) -> List:
        """检测笔"""
        if kline not in self._stroke_cache:
            generator = StrokeGenerator(kline)
            self._stroke_cache[kline] = generator.get_strokes()
        return self._stroke_cache[kline]

    def _detect_pivots(self, kline: KLine, strokes: List) -> List:
        """检测中枢"""
        if kline not in self._pivot_cache:
            detector = PivotDetector(kline, strokes=strokes)
            self._pivot_cache[kline] = detector.get_pivots()
        return self._pivot_cache[kline]

    def _calculate_macd(self, kline: KLine) -> Optional[MACD]:
        """计算MACD"""
        if kline not in self._macd_cache:
            df = kline.to_dataframe()
            if len(df) >= 26:
                self._macd_cache[kline] = MACD(df)
        return self._macd_cache.get(kline)

    def _generate_signal(
        self,
        symbol: str,
        kline: KLine,
        buy_point_result: Optional[SkillResult],
        current_index: int
    ) -> Dict[str, Any]:
        """生成交易信号"""
        if buy_point_result and buy_point_result.success and buy_point_result.data:
            buy_point = buy_point_result.data[0]
            if buy_point.confidence >= 0.6:
                return {
                    'action': 'buy',
                    'type': buy_point.point_type.value,
                    'confidence': buy_point.confidence,
                    'reason': buy_point.reason,
                    'price': buy_point.price,
                }

        # 检查是否有卖出信号（基于之前买入的点位）
        last_signal = self.state.get_analysis(symbol, 'last_signal')
        if last_signal and last_signal.get('action') == 'buy':
            # 简单的止盈逻辑：涨幅超过15%
            entry_price = last_signal.get('price', 0)
            current_price = kline.data[current_index].close
            if entry_price > 0 and current_price > entry_price * 1.15:
                return {
                    'action': 'sell',
                    'type': '止盈',
                    'confidence': 0.8,
                    'reason': f"涨幅超过15% ({(current_price/entry_price - 1)*100:.1f}%)",
                    'price': current_price,
                }

        return {
            'action': 'hold',
            'type': None,
            'confidence': 0.5,
            'reason': '无明确信号',
        }

    def _error_result(self, symbol: str, message: str) -> Dict[str, Any]:
        """返回错误结果"""
        return {
            'symbol': symbol,
            'success': False,
            'error': message,
            'signal': 'hold',
            'buy_points': [],
            'analysis_time': datetime.now().isoformat(),
        }

    def get_state_summary(self) -> str:
        """获取状态摘要"""
        lines = [
            f"Agent: {self.name}",
            f"分析股票数: {len(self.state.symbols)}",
            f"生成信号数: {len(self.state.signals_generated)}",
        ]

        for symbol in self.state.symbols:
            last_result = self.state.get_analysis(symbol, 'last_result')
            if last_result:
                lines.append(f"  {symbol}: {last_result.get('signal', 'N/A')}")

        return "\n".join(lines)


class MultiLevelAgent(BaseAgent):
    """
    多级别分析 Agent
    同时分析周线和日线
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name='多级别分析Agent',
                description='周线日线联合分析',
                agent_config={'lookback': 100}
            )
        super().__init__(config)

        # 子 Agent
        self.weekly_agent = ChanLunAgent(AgentConfig(
            name='周线Agent',
            description='周线级别分析'
        ))
        self.daily_agent = ChanLunAgent(AgentConfig(
            name='日线Agent',
            description='日线级别分析'
        ))

    def _register_skills(self) -> None:
        """多级别 Agent 不直接注册 Skills"""
        pass

    def analyze(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多级别分析

        Args:
            symbol: 股票代码
            context: 上下文数据

        Returns:
            分析结果
        """
        self.log(f"开始多级别分析 {symbol}")

        # 准备周线和日线数据
        data = context.get('data')
        if data is None or len(data) < 200:
            return {
                'symbol': symbol,
                'success': False,
                'error': '数据不足，需要至少200根日线数据',
            }

        # 转换为周线
        weekly_data = self._to_weekly(data)

        # 分别分析
        weekly_result = self.weekly_agent.analyze(symbol, {'data': {symbol: weekly_data}})
        daily_result = self.daily_agent.analyze(symbol, {'data': {symbol: data}})

        # 综合判断
        combined = self._combine_results(weekly_result, daily_result)

        combined['symbol'] = symbol
        combined['weekly_signal'] = weekly_result.get('signal', 'hold')
        combined['daily_signal'] = daily_result.get('signal', 'hold')
        combined['analysis_time'] = datetime.now().isoformat()

        self.log(f"{symbol} 周线={weekly_result.get('signal')}, 日线={daily_result.get('signal')}, "
               f"综合={combined.get('signal')}")

        return combined

    def _to_weekly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """将日线数据转换为周线"""
        df = daily_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        weekly.reset_index(inplace=True)
        return weekly

    def _combine_results(
        self,
        weekly_result: Dict[str, Any],
        daily_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """综合周线和日线结果"""
        weekly_signal = weekly_result.get('signal', 'hold')
        daily_signal = daily_result.get('signal', 'hold')
        weekly_conf = weekly_result.get('signal_confidence', 0)
        daily_conf = daily_result.get('signal_confidence', 0)

        # 共振逻辑
        if weekly_signal == 'buy' and daily_signal == 'buy':
            return {
                'success': True,
                'signal': 'buy',
                'signal_type': '周日线共振买',
                'confidence': (weekly_conf + daily_conf) / 2,
                'reason': f"周线{weekly_result.get('signal_type')} + 日线{daily_result.get('signal_type')}",
            }
        elif weekly_signal == 'sell' and daily_signal == 'sell':
            return {
                'success': True,
                'signal': 'sell',
                'signal_type': '周日线共振卖',
                'confidence': (weekly_conf + daily_conf) / 2,
                'reason': f"周线{weekly_result.get('signal_type')} + 日线{daily_result.get('signal_type')}",
            }
        elif weekly_signal == 'buy' and daily_signal == 'hold':
            return {
                'success': True,
                'signal': 'buy',
                'signal_type': '周线买日线观望',
                'confidence': weekly_conf * 0.8,
                'reason': f"周线{weekly_result.get('signal_type')}，日线等待确认",
            }
        else:
            return {
                'success': True,
                'signal': 'hold',
                'signal_type': '观望',
                'confidence': 0.5,
                'reason': '无明确共振信号',
            }
