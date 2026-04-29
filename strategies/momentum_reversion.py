"""动量+均值回归策略引擎

纯价格因子，无缠论依赖，天然无 look-ahead bias。

入场逻辑:
  1. 动量确认: 中期趋势向上 (MA_short > MA_long, N日涨幅>0)
  2. 均值回归触发: 短期超跌回调 (价格跌破均线, RSI超卖)
  3. 综合评分 >= 阈值

出场逻辑:
  - 止盈: 价格冲高+RSI超买
  - 止损: ATR动态止损, 最大止损P%
  - 时间止损: 持仓超限

支持两套参数:
  - short_term: MA10/MA20, 持仓5-20天, 快进快出
  - mid_term: MA20/MA60, 持仓20-60天, 波段趋势
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI 计算"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR 计算"""
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)),
                                            np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values


# 预设参数组
PRESETS = {
    'short': {
        'ma_short': 10,
        'ma_long': 20,
        'lookback_return': 10,
        'pullback_pct': 0.03,       # 跌破均线3%
        'deep_pullback_pct': 0.05,  # 深度回调5%
        'rsi_oversold': 35,
        'rsi_extreme': 25,
        'rsi_overbought': 65,
        'rsi_extreme_high': 70,
        'take_profit_1': 0.05,      # 止盈5%减半
        'take_profit_2': 0.10,      # 止盈10%清仓
        'max_stop_pct': 0.05,       # 最大止损5%
        'atr_stop_mult': 1.5,
        'trailing_stop_pct': 0.05,  # 移动止损5%
        'time_stop_bars': 12,       # 12天时间止损
        'min_score': 4,             # 提高入场门槛
        'max_positions': 5,
        'position_weight': 0.20,    # 单只20%
    },
    'mid': {
        'ma_short': 20,
        'ma_long': 60,
        'lookback_return': 20,
        'pullback_pct': 0.03,
        'deep_pullback_pct': 0.05,
        'rsi_oversold': 35,
        'rsi_extreme': 25,
        'rsi_overbought': 65,
        'rsi_extreme_high': 70,
        'take_profit_1': 0.08,      # 止盈8%减半
        'take_profit_2': 0.15,      # 止盈15%清仓
        'max_stop_pct': 0.08,
        'atr_stop_mult': 2.0,
        'trailing_stop_pct': 0.05,  # 移动止损5%
        'time_stop_bars': 40,
        'min_score': 3,
        'max_positions': 5,
        'position_weight': 0.20,
    },
}


class MomentumReversionEngine:
    """动量+均值回归策略引擎"""

    def __init__(self, preset: str = 'mid', params: dict = None):
        if params:
            self.p = params
        else:
            self.p = PRESETS.get(preset, PRESETS['mid'])

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """生成交易信号

        Args:
            data_map: {code: DataFrame(DatetimeIndex, open/high/low/close/volume)}

        Returns:
            {code: Series(position_weight, index=df.index)}
            0 = 空仓, 0.1 = 轻仓, 0.2 = 标准仓
        """
        signals = {}
        for code, df in data_map.items():
            signals[code] = self._generate_single(df)
        return signals

    def _generate_single(self, df: pd.DataFrame) -> pd.Series:
        p = self.p
        n = len(df)
        signals = pd.Series(0.0, index=df.index)

        if n < max(p['ma_long'], 60) + 10:
            return signals

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        vol = df['volume'].values.astype(float) if 'volume' in df.columns else np.ones(n)

        # 指标
        ma_short = pd.Series(close).rolling(p['ma_short']).mean().values
        ma_long = pd.Series(close).rolling(p['ma_long']).mean().values
        rsi = _rsi(close, 14)
        atr_val = _atr(high, low, close, 14)
        vol_ma = pd.Series(vol).rolling(20).mean().values

        # MACD (简化)
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values

        start = p['ma_long'] + 5

        # 持仓状态
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        stop_loss = 0.0
        highest = 0.0
        half_sold = False

        for i in range(start, n):
            price = close[i]

            if in_pos:
                bars_held = i - entry_idx
                profit = (price - entry_price) / entry_price

                if price > highest:
                    highest = price

                # 1. 止损
                if price <= stop_loss:
                    signals.iloc[i] = 0.0
                    in_pos = False
                    continue

                # 2. 止盈1: 减半仓
                if not half_sold and profit > p['take_profit_1'] and rsi[i] > p['rsi_overbought']:
                    signals.iloc[i] = 0.05  # 减半标记
                    half_sold = True
                    continue

                # 3. 止盈2: 清仓
                if profit > p['take_profit_2'] and rsi[i] > p['rsi_extreme_high']:
                    signals.iloc[i] = 0.0
                    in_pos = False
                    continue

                # 4. 移动止损 (盈利>3%后, 从最高点回撤)
                trail_pct = p.get('trailing_stop_pct', 0.03)
                if profit > 0.03 and price < highest * (1 - trail_pct):
                    signals.iloc[i] = 0.0
                    in_pos = False
                    continue

                # 5. 时间止损
                if bars_held >= p['time_stop_bars'] and profit < 0.01:
                    signals.iloc[i] = 0.0
                    in_pos = False
                    continue

                # 持仓中 — 小正值(0.1)表示满仓持有, 区别于入场分数(3~6)
                signals.iloc[i] = 0.1 if not half_sold else 0.05

            else:
                # === 入场评估 ===

                # 动量确认: 趋势向上
                trend_up = ma_short[i] > ma_long[i]
                positive_return = close[i] > close[i - p['lookback_return']]

                if not (trend_up and positive_return):
                    continue

                # 均值回归: 短期超跌
                pullback = price < ma_short[i] * (1 - p['pullback_pct'])
                if not pullback:
                    continue

                # 流动性
                if vol_ma[i] > 0 and vol[i] < vol_ma[i] * 0.5:
                    continue

                # 综合评分
                score = 0

                # 回调深度
                if price < ma_short[i] * (1 - p['deep_pullback_pct']):
                    score += 2
                elif price < ma_short[i] * (1 - p['pullback_pct']):
                    score += 1

                # RSI
                if rsi[i] < p['rsi_extreme']:
                    score += 2
                elif rsi[i] < p['rsi_oversold']:
                    score += 1

                # 短期均线确认
                ma5 = pd.Series(close[max(0,i-5):i+1]).mean()
                if ma5 < ma_short[i]:
                    score += 1

                # MACD多头
                if dif[i] > dea[i]:
                    score += 1

                if score < p['min_score']:
                    continue

                # 入场 — signal 编码: score(3~6) 用于回测排序
                entry_price = price
                entry_idx = i
                atr_stop = atr_val[i] * p['atr_stop_mult'] if atr_val[i] > 0 else price * 0.04
                stop_loss = max(price - atr_stop, price * (1 - p['max_stop_pct']))
                highest = price
                half_sold = False
                in_pos = True
                signals.iloc[i] = float(score)  # 入场信号 = 质量分数

        return signals
