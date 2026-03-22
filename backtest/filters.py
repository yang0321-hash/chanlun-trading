"""
进出场过滤器模块

提供多种技术指标过滤器：
1. VolumeFilter - 成交量确认
2. RSIFilter - RSI超买超卖
3. BollingerFilter - 布林带突破
4. MultiStageExit - 多级止盈
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class FilterResult:
    """过滤器结果"""
    passed: bool          # 是否通过
    reason: str           # 原因说明
    confidence: float     # 置信度 0-1


class BaseFilter:
    """过滤器基类"""

    def __init__(self, name: str):
        self.name = name
        self.pass_count = 0
        self.total_count = 0

    def check(
        self,
        bar: pd.Series,
        historical: pd.DataFrame,
        signal_type: str = 'buy'
    ) -> FilterResult:
        """检查是否通过"""
        raise NotImplementedError

    def reset_stats(self):
        """重置统计"""
        self.pass_count = 0
        self.total_count = 0

    @property
    def pass_rate(self) -> float:
        """通过率"""
        if self.total_count == 0:
            return 0
        return self.pass_count / self.total_count


class VolumeFilter(BaseFilter):
    """
    成交量过滤器

    买入条件：
    - 放量突破：成交量 > MA(20) * 1.5
    - 缩量回调：成交量 < MA(20) * 0.8

    卖出条件：
    - 放量滞涨：成交量高但价格不涨
    """

    def __init__(
        self,
        ma_period: int = 20,
        breakout_multiplier: float = 1.5,  # 放量倍数
        pullback_ratio: float = 0.8,       # 缩量比例
    ):
        super().__init__("成交量过滤器")
        self.ma_period = ma_period
        self.breakout_multiplier = breakout_multiplier
        self.pullback_ratio = pullback_ratio

    def check(
        self,
        bar: pd.Series,
        historical: pd.DataFrame,
        signal_type: str = 'buy'
    ) -> FilterResult:
        """检查成交量条件"""
        self.total_count += 1

        if len(historical) < self.ma_period:
            self.pass_count += 1
            return FilterResult(True, "数据不足，跳过", 0.5)

        # 计算成交量均线
        vol_ma = historical['volume'].tail(self.ma_period).mean()
        current_vol = bar['volume']

        # 检查买入
        if signal_type == 'buy':
            # 检查是否放量
            if current_vol > vol_ma * self.breakout_multiplier:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"放量买入 量={current_vol:.0f} > MA*{self.breakout_multiplier}",
                    0.8
                )
            # 检查是否缩量回调
            elif current_vol < vol_ma * self.pullback_ratio:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"缩量回调 量={current_vol:.0f} < MA*{self.pullback_ratio}",
                    0.6
                )
            else:
                return FilterResult(
                    False,
                    f"成交量不达标 量={current_vol:.0f} MA={vol_ma:.0f}",
                    0
                )

        # 检查卖出
        else:
            # 检查是否放量滞涨
            if current_vol > vol_ma * self.breakout_multiplier:
                # 计算价格涨幅
                if len(historical) >= 5:
                    price_change = (bar['close'] - historical['close'].iloc[-5]) / historical['close'].iloc[-5]
                    if price_change < 0.02:  # 涨幅小于2%
                        self.pass_count += 1
                        return FilterResult(
                            True,
                            f"放量滞涨 涨幅={price_change:.2%}",
                            0.9
                        )

            # 默认通过卖出
            self.pass_count += 1
            return FilterResult(True, "卖出无需量能过滤", 0.5)


class RSIFilter(BaseFilter):
    """
    RSI过滤器

    买入条件：
    - RSI < 30 (超卖)
    - RSI从低位回升

    卖出条件：
    - RSI > 70 (超买)
    - RSI顶背离
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        allow_range: tuple = (30, 70),  # 允许买入的RSI范围
    ):
        super().__init__("RSI过滤器")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.allow_range = allow_range

    def calculate_rsi(self, df: pd.DataFrame) -> float:
        """计算RSI"""
        if len(df) < self.period + 1:
            return 50

        closes = df['close'].values
        deltas = np.diff(closes)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def check(
        self,
        bar: pd.Series,
        historical: pd.DataFrame,
        signal_type: str = 'buy'
    ) -> FilterResult:
        """检查RSI条件"""
        self.total_count += 1

        if len(historical) < self.period + 1:
            self.pass_count += 1
            return FilterResult(True, "数据不足，跳过", 0.5)

        rsi = self.calculate_rsi(historical)

        # 检查买入
        if signal_type == 'buy':
            # RSI在允许范围内
            if self.allow_range[0] <= rsi <= self.allow_range[1]:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"RSI正常 rsi={rsi:.1f}",
                    0.7
                )
            # 超卖区域，允许买入
            elif rsi < self.oversold:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"RSI超卖 rsi={rsi:.1f}",
                    0.9
                )
            # 超买，不买入
            else:
                return FilterResult(
                    False,
                    f"RSI超买 rsi={rsi:.1f} > {self.overbought}",
                    0
                )

        # 检查卖出
        else:
            # RSI超买，强烈卖出信号
            if rsi > self.overbought:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"RSI超买 rsi={rsi:.1f}",
                    0.9
                )
            # RSI正常，卖出信号较弱
            else:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"RSI正常 rsi={rsi:.1f}",
                    0.5
                )


class BollingerFilter(BaseFilter):
    """
    布林带过滤器

    买入条件：
    - 价格触及下轨反弹
    - 价格突破中轨

    卖出条件：
    - 价格触及上轨
    - 价格跌破中轨
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ):
        super().__init__("布林带过滤器")
        self.period = period
        self.std_dev = std_dev

    def calculate_bands(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """计算布林带"""
        if len(df) < self.period:
            return 0, 0, 0

        closes = df['close'].tail(self.period)
        middle = closes.mean()
        std = closes.std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        return upper, middle, lower

    def check(
        self,
        bar: pd.Series,
        historical: pd.DataFrame,
        signal_type: str = 'buy'
    ) -> FilterResult:
        """检查布林带条件"""
        self.total_count += 1

        if len(historical) < self.period:
            self.pass_count += 1
            return FilterResult(True, "数据不足，跳过", 0.5)

        upper, middle, lower = self.calculate_bands(historical)
        price = bar['close']

        # 检查买入
        if signal_type == 'buy':
            # 价格在中轨下方接近下轨 - 买入信号
            if price <= lower * 1.02:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"触及下轨 price={price:.2f} lower={lower:.2f}",
                    0.9
                )
            # 价格突破中轨 - 买入确认
            elif price > middle * 0.98:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"突破中轨 price={price:.2f} middle={middle:.2f}",
                    0.7
                )
            # 价格在中轨和下轨之间 - 观望
            else:
                return FilterResult(
                    False,
                    f"价格在中下轨之间 price={price:.2f}",
                    0
                )

        # 检查卖出
        else:
            # 价格触及上轨 - 卖出信号
            if price >= upper * 0.98:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"触及上轨 price={price:.2f} upper={upper:.2f}",
                    0.9
                )
            # 价格跌破中轨 - 卖出确认
            elif price < middle * 0.98:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"跌破中轨 price={price:.2f} middle={middle:.2f}",
                    0.7
                )
            # 价格在中轨上方 - 持有
            else:
                return FilterResult(
                    False,
                    f"持有 price={price:.2f}",
                    0
                )


class MACDDivergenceFilter(BaseFilter):
    """
    MACD背驰过滤器
    """

    def __init__(self):
        super().__init__("MACD背驰过滤器")

    def check(
        self,
        bar: pd.Series,
        historical: pd.DataFrame,
        signal_type: str = 'buy'
    ) -> FilterResult:
        """检查MACD背驰"""
        self.total_count += 1

        if len(historical) < 26:
            self.pass_count += 1
            return FilterResult(True, "数据不足，跳过", 0.5)

        # 计算MACD
        macd_line, signal_line, histogram = self._calculate_macd(historical)

        if macd_line is None:
            self.pass_count += 1
            return FilterResult(True, "MACD计算失败", 0.5)

        # 检查买入 - 底背驰
        if signal_type == 'buy':
            # 检查最近是否有金叉
            if len(histogram) >= 2:
                if histogram[-2] < 0 and histogram[-1] > histogram[-2]:
                    # 金叉且在负值区域
                    self.pass_count += 1
                    return FilterResult(
                        True,
                        f"MACD金叉 hist={histogram[-1]:.4f}",
                        0.8
                    )

            # 检查是否在负值区域开始回升
            if histogram[-1] < 0 and histogram[-1] > histogram[-2] if len(histogram) >= 2 else False:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"MACD负值区域回升",
                    0.6
                )

            return FilterResult(False, "无MACD买入信号", 0)

        # 检查卖出 - 顶背驰
        else:
            # 检查最近是否有死叉
            if len(histogram) >= 2:
                if histogram[-2] > 0 and histogram[-1] < histogram[-2]:
                    # 死叉且在正值区域
                    self.pass_count += 1
                    return FilterResult(
                        True,
                        f"MACD死叉 hist={histogram[-1]:.4f}",
                        0.8
                    )

            return FilterResult(False, "无MACD卖出信号", 0)

    def _calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple:
        """计算MACD"""
        if len(df) < slow:
            return None, None, None

        closes = df['close']

        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.values


class TrendFilter(BaseFilter):
    """
    趋势过滤器

    使用均线判断趋势方向
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 60,
    ):
        super().__init__("趋势过滤器")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def check(
        self,
        bar: pd.Series,
        historical: pd.DataFrame,
        signal_type: str = 'buy'
    ) -> FilterResult:
        """检查趋势条件"""
        self.total_count += 1

        if len(historical) < self.slow_period:
            self.pass_count += 1
            return FilterResult(True, "数据不足，跳过", 0.5)

        # 计算均线
        fast_ma = historical['close'].tail(self.fast_period).mean()
        slow_ma = historical['close'].tail(self.slow_period).mean()
        price = bar['close']

        # 检查买入
        if signal_type == 'buy':
            # 上升趋势：快线在慢线上方
            if fast_ma > slow_ma:
                # 价格在快线附近或上方
                if price >= fast_ma * 0.98:
                    self.pass_count += 1
                    return FilterResult(
                        True,
                        f"上升趋势 fast={fast_ma:.2f} > slow={slow_ma:.2f}",
                        0.8
                    )
                # 价格在快线下方但接近
                elif price >= fast_ma * 0.95:
                    self.pass_count += 1
                    return FilterResult(
                        True,
                        f"回调买入位 price={price:.2f} fast={fast_ma:.2f}",
                        0.6
                    )
                else:
                    return FilterResult(
                        False,
                        f"价格偏离均线过多",
                        0
                    )
            else:
                return FilterResult(
                    False,
                    f"下降趋势 fast={fast_ma:.2f} < slow={slow_ma:.2f}",
                    0
                )

        # 检查卖出
        else:
            # 下降趋势：快线在慢线下方
            if fast_ma < slow_ma:
                self.pass_count += 1
                return FilterResult(
                    True,
                    f"下降趋势 fast={fast_ma:.2f} < slow={slow_ma:.2f}",
                    0.8
                )
            else:
                return FilterResult(
                    False,
                    f"仍处上升趋势",
                    0
                )


class MultiStageExit:
    """
    多级止盈模块

    分批止盈策略：
    - 第一级：盈利10%卖出30%
    - 第二级：盈利20%卖出30%
    - 第三级：盈利30%卖出剩余40%
    - 或使用ATR移动止盈
    """

    def __init__(
        self,
        stages: List[Tuple[float, float]] = None,
        use_trailing: bool = True,
        trailing_atr_multiple: float = 3.0,
    ):
        """
        Args:
            stages: [(利润比例, 卖出比例), ...]
                   例如 [(0.10, 0.30), (0.20, 0.30), (0.30, 0.40)]
        """
        self.stages = stages or [
            (0.10, 0.30),  # 盈利10%卖出30%
            (0.20, 0.30),  # 盈利20%卖出30%
            (0.30, 0.40),  # 盈利30%卖出40%
        ]
        self.use_trailing = use_trailing
        self.trailing_atr_multiple = trailing_atr_multiple

        # 持仓记录 {symbol: {'entry_price': float, 'highest': float, 'exited_stages': int}}
        self.positions = {}

    def add_position(self, symbol: str, entry_price: float, atr: float = 0):
        """添加持仓"""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'highest': entry_price,
            'atr': atr,
            'exited_stages': 0,
        }

    def remove_position(self, symbol: str):
        """移除持仓"""
        if symbol in self.positions:
            del self.positions[symbol]

    def update_price(self, symbol: str, current_price: float) -> Optional[float]:
        """
        更新价格并检查是否需要止盈

        Returns:
            卖出比例 (0-1)，None表示不卖出
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        # 更新最高价
        if current_price > pos['highest']:
            pos['highest'] = current_price

        entry_price = pos['entry_price']
        profit_pct = (current_price - entry_price) / entry_price

        # 检查多级止盈
        for i, (stage_profit, stage_ratio) in enumerate(self.stages):
            if i >= pos['exited_stages'] and profit_pct >= stage_profit:
                pos['exited_stages'] = i + 1
                logger.info(
                    f"[{symbol}] 第{i+1}级止盈: "
                    f"盈利{profit_pct:.2%} >= {stage_profit:.2%}, "
                    f"卖出{stage_ratio:.0%}"
                )
                return stage_ratio

        # 检查移动止盈
        if self.use_trailing and pos['atr'] > 0:
            trailing_stop = pos['highest'] - pos['atr'] * self.trailing_atr_multiple
            if current_price < trailing_stop and profit_pct > 0.05:
                # 从最高点回撤超过ATR倍数，且盈利至少5%
                logger.info(
                    f"[{symbol}] 移动止盈: "
                    f"价格{current_price:.2f} < 止损{trailing_stop:.2f}, "
                    f"盈利{profit_pct:.2%}"
                )
                return 1.0  # 全部卖出

        return None

    def get_position_info(self, symbol: str) -> dict:
        """获取持仓信息"""
        return self.positions.get(symbol, {})


class FilterChain:
    """
    过滤器链

    将多个过滤器串联使用
    """

    def __init__(self, filters: List[BaseFilter], mode: str = 'all'):
        """
        Args:
            filters: 过滤器列表
            mode: 'all' 全部通过, 'any' 任一通过, 'majority' 多数通过
        """
        self.filters = filters
        self.mode = mode

    def check(
        self,
        bar: pd.Series,
        historical: pd.DataFrame,
        signal_type: str = 'buy'
    ) -> FilterResult:
        """检查所有过滤器"""
        results = []
        for f in self.filters:
            result = f.check(bar, historical, signal_type)
            results.append(result)

        if self.mode == 'all':
            passed = all(r.passed for r in results)
            reason = ' | '.join([r.reason for r in results if r.passed])
            confidence = min([r.confidence for r in results if r.passed], default=0)

        elif self.mode == 'any':
            passed = any(r.passed for r in results)
            reason = ' | '.join([r.reason for r in results if r.passed])
            confidence = max([r.confidence for r in results if r.passed], default=0)

        elif self.mode == 'majority':
            passed_count = sum(1 for r in results if r.passed)
            passed = passed_count > len(results) / 2
            reason = f"通过{passed_count}/{len(results)}: " + ' | '.join([r.reason for r in results if r.passed])
            confidence = passed_count / len(results)

        else:
            passed = True
            reason = ""
            confidence = 0.5

        return FilterResult(passed, reason, confidence)

    def get_stats(self) -> dict:
        """获取各过滤器统计"""
        return {
            f.name: {
                'pass_count': f.pass_count,
                'total_count': f.total_count,
                'pass_rate': f.pass_rate,
            }
            for f in self.filters
        }
