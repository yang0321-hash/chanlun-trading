"""V11d 30分钟参数独立优化

30分钟 bar 密度是日线 8 倍，指标周期需要按比例缩放。
百分比型参数（trailing、stop loss）不变，bar 计数型参数 ×8。

关键参数:
- bi_min_gap: 需要独立调参（最关键）
- MACD: EMA 12/26 → 96/208
- MA20 → MA160, ATR 14 → 112
- trailing/stop: 百分比参数不变

用法: python optimize_30min.py
"""

import sys
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parent
MIN30_DIR = PROJECT_ROOT / "chanlun_system" / "artifacts"

# 复用 daily backtest 的向量回测框架
sys.path.insert(0, str(RUN_DIR))
from backtest_30min import (
    load_min30_data, run_30min_backtest, log_trades_30min
)


class SignalEngine30min:
    """30分钟专用信号引擎

    基于 SignalEngine 但所有指标周期按 ×8 缩放。
    bi_min_gap 参数需要独立优化。
    """

    def __init__(self, bi_min_gap=15, trailing_start=0.015, trailing_tight=0.02,
                 trailing_medium=0.05, trailing_wide=0.07, weekly_threshold=0.97,
                 time_stop_days=60, min_hold_days=7):
        # ===== 不变的参数（百分比型）=====
        self.risk_per_trade = 0.03
        self.max_positions = 5
        self.max_drawdown_pct = 0.15
        self.max_stop_pct = 0.15
        self.min_position = 0.10
        self.max_position = 0.30
        self.base_position = 0.15

        # Trailing (百分比，不变)
        self.trailing_start = trailing_start
        self.trailing_tight = trailing_tight
        self.trailing_medium = trailing_medium
        self.trailing_wide = trailing_wide
        self.trailing_tier1 = 0.06
        self.trailing_tier2 = 0.15

        # 百分比型（不变）
        self.profit_add_threshold = 0.05
        self.profit_add_ratio = 0.50
        self.reduce_start = 0.04
        self.reduce_position_pct = 0.50
        self.big_loss_threshold = -0.03

        # ===== ×8 缩放参数 =====
        self.atr_stop_mult = 2.0
        self.atr_period = 14 * 8          # 112
        self.vol_lookback = 120 * 8       # 960

        # Bar 计数型参数（按天数 ×8）
        self.cooldown_bars = 1 * 8         # 8 bars
        self.big_loss_cooldown = 30 * 8    # 240 bars
        self.time_stop_bars = time_stop_days * 8
        self.min_hold_before_sell = min_hold_days * 8

        # ===== 独立调参 =====
        self.bi_min_gap = bi_min_gap
        self.bi_confirm_delay = 2          # 30分钟确认稍慢
        self.weekly_threshold = weekly_threshold

        # MACD 参数（×8 缩放）
        self.ema_fast = 12 * 8            # 96
        self.ema_slow = 26 * 8            # 208
        self.ema_signal = 9 * 8           # 72

        # MA/Volume MA（×8 缩放）
        self.ma_period = 20 * 8           # 160
        self.vol_ma_period = 20 * 8       # 160

        # 动量（×8 缩放）
        self.momentum_period = 60 * 8     # 480
        self.momentum_factor_enabled = True
        self.vol_regime_enabled = True

        # 背驰窗口（×8 缩放）
        self.divergence_window = 120 * 8  # 960

        # 最小数据量（×8 缩放）
        self.min_bars = 120 * 8           # 960

        # 结构止损回看（×8 缩放）
        self.struct_lookback = 30 * 8     # 240

        # 连续亏损参数（×8 缩放）
        self.consec_loss_bars = 5 * 8     # 40 bars 内算连续
        self.consec_loss_pause = 2 * 8    # 暂停 16 bars

        # 组合级状态
        self._active_positions = 0
        self._last_loss_codes = {}
        self._portfolio_peak = 1_000_000
        self._portfolio_equity = 1_000_000
        self._trading_halted = False
        self._consecutive_losses = 0
        self._last_loss_bar = -999
        self._loss_pause_until = -1

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        self._active_positions = 0
        self._last_loss_codes = {}
        self._portfolio_peak = 1_000_000
        self._portfolio_equity = 1_000_000
        self._trading_halted = False
        self._consecutive_losses = 0
        self._last_loss_bar = -999
        self._loss_pause_until = -1

        monthly_momentum_rank = {}
        if self.momentum_factor_enabled:
            monthly_momentum_rank = self._build_momentum_ranking(data_map)

        signals = {}
        for code in sorted(data_map.keys()):
            df = data_map[code]
            signals[code] = self._generate_single(code, df, monthly_momentum_rank)
        return signals

    def _build_momentum_ranking(self, data_map):
        monthly_scores = {}
        for code, df in data_map.items():
            if len(df) < self.min_bars:
                continue
            close = df['close']
            ret = close.pct_change(self.momentum_period)
            try:
                monthly_first = df.resample('MS').first()
            except Exception:
                continue
            for dt in monthly_first.index:
                if dt not in ret.index:
                    valid = ret.index[ret.index <= dt]
                    if len(valid) == 0:
                        continue
                    dt = valid[-1]
                if pd.isna(ret.loc[dt]):
                    continue
                month_key = dt.strftime('%Y-%m')
                if month_key not in monthly_scores:
                    monthly_scores[month_key] = {}
                monthly_scores[month_key][code] = ret.loc[dt]

        monthly_rank = {}
        for month, scores in monthly_scores.items():
            if len(scores) < 3:
                continue
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            n = len(sorted_items)
            ranks = {}
            for rank_idx, (code, _) in enumerate(sorted_items):
                ranks[code] = (rank_idx + 1) / n
            monthly_rank[month] = ranks
        return monthly_rank

    def _generate_single(self, code, df, monthly_momentum_rank=None):
        n = len(df)
        signals = pd.Series(0.0, index=df.index)
        if n < self.min_bars:
            return signals

        close = df['close']
        high = df['high']
        low = df['low']

        # 指标（全部 ×8 缩放）
        bi_buy, bi_sell = self._detect_bi_deterministic(df)

        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.ema_signal, adjust=False).mean()
        macd_hist = 2 * (dif - dea)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        atr_pct = pd.Series(0.5, index=df.index)
        if self.vol_regime_enabled:
            atr_pct = atr.rolling(self.vol_lookback, min_periods=60).rank(pct=True)

        weekly_not_down = self._compute_weekly_not_down(df)
        ma = close.rolling(self.ma_period).mean()
        vol_ma = df['volume'].rolling(self.vol_ma_period).mean()

        # 逐 bar 模拟
        position = 0.0
        entry_idx = -1
        entry_price = 0.0
        stop_loss = 0.0
        highest = 0.0
        has_added = False
        original_position = 0.0
        trailing_activated = False
        reduced = False

        macd_buy_points = []
        macd_sell_points = []

        for i in range(self.min_bars, n):
            price = close.iloc[i]

            # 背驰检测
            has_buy_divergence = False
            has_sell_divergence = False

            if bi_buy.iloc[i]:
                curr = {'idx': i, 'price': low.iloc[i], 'dif': dif.iloc[i], 'hist': macd_hist.iloc[i]}
                for k in range(len(macd_buy_points) - 1, -1, -1):
                    prev = macd_buy_points[k]
                    if curr['idx'] - prev['idx'] > self.divergence_window:
                        break
                    if curr['price'] >= prev['price']:
                        continue
                    if curr['dif'] > prev['dif'] or curr['hist'] > prev['hist']:
                        has_buy_divergence = True
                        break
                macd_buy_points.append(curr)

            if bi_sell.iloc[i]:
                curr = {'idx': i, 'price': high.iloc[i], 'dif': dif.iloc[i], 'hist': macd_hist.iloc[i]}
                for k in range(len(macd_sell_points) - 1, -1, -1):
                    prev = macd_sell_points[k]
                    if curr['idx'] - prev['idx'] > self.divergence_window:
                        break
                    if curr['price'] <= prev['price']:
                        continue
                    if curr['dif'] < prev['dif'] or curr['hist'] < prev['hist']:
                        has_sell_divergence = True
                        break
                macd_sell_points.append(curr)

            # 组合回撤
            if position > 0:
                day_pnl = position * (price - close.iloc[i-1]) / close.iloc[i-1] if i > 0 and close.iloc[i-1] > 0 else 0
                self._portfolio_equity += day_pnl * self._portfolio_equity
                if self._portfolio_equity > self._portfolio_peak:
                    self._portfolio_peak = self._portfolio_equity
                drawdown = (self._portfolio_equity - self._portfolio_peak) / self._portfolio_peak
                self._trading_halted = drawdown < -self.max_drawdown_pct

            if position > 0:
                if price > highest:
                    highest = price
                bars_held = i - entry_idx
                profit_pct = (price - entry_price) / entry_price if entry_price > 0 else 0

                # 硬止损
                if profit_pct < -self.max_stop_pct:
                    hard_stop_price = entry_price * (1 - self.max_stop_pct)
                    exit_price = max(hard_stop_price, low.iloc[i])
                    capped_pct = (exit_price - entry_price) / entry_price
                    signals.iloc[i] = 0.0
                    self._last_loss_codes[code] = (i, capped_pct)
                    self._update_loss_streak(capped_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 结构止损
                if price <= stop_loss:
                    struct_exit = max(stop_loss, low.iloc[i])
                    struct_pct = (struct_exit - entry_price) / entry_price
                    signals.iloc[i] = 0.0
                    if struct_pct < 0:
                        self._last_loss_codes[code] = (i, struct_pct)
                    self._update_loss_streak(struct_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 移动止损
                if profit_pct > self.trailing_start:
                    trailing_activated = True
                if trailing_activated:
                    max_profit = (highest - entry_price) / entry_price if entry_price > 0 else 0
                    if max_profit >= self.trailing_tier2:
                        trailing_dist = self.trailing_wide
                    elif max_profit >= self.trailing_tier1:
                        trailing_dist = self.trailing_medium
                    else:
                        trailing_dist = self.trailing_tight
                    trailing_stop = highest * (1 - trailing_dist)
                    if price <= trailing_stop:
                        signals.iloc[i] = 0.0
                        if profit_pct < 0:
                            self._last_loss_codes[code] = (i, profit_pct)
                        self._update_loss_streak(profit_pct, i)
                        self._active_positions = max(0, self._active_positions - 1)
                        position = 0.0; original_position = 0.0; has_added = False
                        trailing_activated = False; reduced = False
                        continue

                # 顶背驰卖出
                if (bars_held >= self.min_hold_before_sell
                        and has_sell_divergence
                        and profit_pct > 0.05):
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 2卖出局
                if (bars_held >= self.min_hold_before_sell
                        and bi_sell.iloc[i]
                        and profit_pct > 0.05):
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 时间止损
                if bars_held >= self.time_stop_bars:
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 减仓
                if (not reduced and profit_pct > self.reduce_start
                        and bi_sell.iloc[i] and bars_held >= 24):
                    reduced_pos = position * self.reduce_position_pct
                    reduced_pos = max(self.min_position * 0.5, reduced_pos)
                    signals.iloc[i] = reduced_pos
                    position = reduced_pos
                    reduced = True
                    continue

                # 加仓
                if (not has_added and profit_pct > self.profit_add_threshold
                        and bi_buy.iloc[i] and profit_pct < 0.25
                        and self._active_positions <= self.max_positions):
                    add_size = original_position * self.profit_add_ratio
                    new_position = min(position + add_size, self.max_position)
                    actual_add = new_position - position
                    if actual_add > 0.001:
                        entry_price = (entry_price * position + price * actual_add) / new_position
                        stop_loss = self._compute_stop_loss(price, atr, i, low, df)
                        stop_loss = max(stop_loss, entry_price * (1 - self.max_stop_pct))
                        signals.iloc[i] = new_position
                        position = new_position
                        has_added = True
                        continue

                signals.iloc[i] = position

            else:
                # 空仓 - 寻找入场
                if self._trading_halted:
                    continue
                if i <= self._loss_pause_until:
                    continue
                if self._active_positions >= self.max_positions:
                    continue

                if code in self._last_loss_codes:
                    loss_idx, loss_pct = self._last_loss_codes[code]
                    cooldown = self.big_loss_cooldown if loss_pct < self.big_loss_threshold else self.cooldown_bars
                    if i - loss_idx < cooldown:
                        continue

                if not weekly_not_down.iloc[i]:
                    continue
                if not bi_buy.iloc[i]:
                    continue

                # MA 过滤
                if not pd.isna(ma.iloc[i]) and price < ma.iloc[i]:
                    continue

                # 涨跌停过滤（30分钟级别用前一根收盘价）
                if i > 0 and close.iloc[i-1] > 0:
                    pct_change = (price - close.iloc[i-1]) / close.iloc[i-1]
                    if pct_change >= 0.095:
                        continue

                # MACD 因子
                macd_factor = 0.0
                macd_confirm = (
                    dif.iloc[i] > dif.iloc[i-1]
                    or (macd_hist.iloc[i] > macd_hist.iloc[i-1] and macd_hist.iloc[i] <= 0)
                    or macd_hist.iloc[i] > 0
                )
                if macd_confirm:
                    macd_factor = 0.02
                elif macd_hist.iloc[i] < macd_hist.iloc[i-1] and macd_hist.iloc[i] < 0:
                    macd_factor = -0.02
                if has_buy_divergence:
                    macd_factor = 0.05

                # 量能因子
                vol_factor = 0.0
                if not pd.isna(vol_ma.iloc[i]) and vol_ma.iloc[i] > 0:
                    vol_ratio = df['volume'].iloc[i] / vol_ma.iloc[i]
                    if vol_ratio >= 2.0:
                        vol_factor = 0.04
                    elif vol_ratio >= 1.5:
                        vol_factor = 0.02
                    elif vol_ratio < 0.5:
                        vol_factor = -0.02

                # 弱信号过滤
                macd_weak = macd_hist.iloc[i] < macd_hist.iloc[i-1] and macd_hist.iloc[i] < 0
                vol_very_weak = (not pd.isna(vol_ma.iloc[i]) and vol_ma.iloc[i] > 0
                                 and df['volume'].iloc[i] / vol_ma.iloc[i] < 0.5)
                ma_very_weak = (not pd.isna(ma.iloc[i]) and price < ma.iloc[i] * 0.93)
                if macd_weak and vol_very_weak and ma_very_weak and not has_buy_divergence:
                    continue

                # 止损位
                stop = self._compute_stop_loss(price, atr, i, low, df)
                stop_distance = price - stop
                if stop_distance <= 0:
                    continue
                stop_pct = stop_distance / price
                if stop_pct > self.max_stop_pct:
                    stop = price * (1 - self.max_stop_pct)
                    stop_distance = price - stop
                    stop_pct = self.max_stop_pct

                # 仓位
                risk_pct = self.risk_per_trade / stop_pct
                risk_pct = min(risk_pct, self.max_position)
                base_pos = min(self.base_position, risk_pct)
                final_pos = base_pos + macd_factor + vol_factor
                if has_buy_divergence:
                    final_pos = max(final_pos, 0.18)
                final_pos = max(self.min_position, min(final_pos, self.max_position))

                actual_risk = final_pos * stop_pct
                if actual_risk > self.risk_per_trade:
                    final_pos = self.risk_per_trade / stop_pct
                    final_pos = max(self.min_position, min(final_pos, self.max_position))

                # 动量因子
                if (self.momentum_factor_enabled and monthly_momentum_rank
                        and not has_buy_divergence):
                    month_key = df.index[i].strftime('%Y-%m')
                    if month_key in monthly_momentum_rank:
                        mom_rank = monthly_momentum_rank[month_key].get(code, 0.5)
                        if mom_rank > 0.7:
                            final_pos += 0.01
                        elif mom_rank < 0.3:
                            final_pos -= 0.01

                # 波动率自适应
                if self.vol_regime_enabled and not pd.isna(atr_pct.iloc[i]):
                    pct_val = atr_pct.iloc[i]
                    if pct_val > 0.70:
                        final_pos *= 0.85
                        stop_pct *= 1.15
                    elif pct_val < 0.30:
                        final_pos *= 1.05
                        stop_pct *= 0.95
                    final_pos = max(self.min_position * 0.5, min(final_pos, self.max_position))
                    stop_pct = min(stop_pct, self.max_stop_pct)

                signals.iloc[i] = final_pos
                position = final_pos
                original_position = final_pos
                has_added = False
                reduced = False
                entry_idx = i
                entry_price = price
                stop_loss = stop
                highest = price
                self._active_positions += 1

        return signals

    def _compute_stop_loss(self, price, atr, i, low, df):
        atr_stop = 0.0
        if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0:
            atr_stop = price - self.atr_stop_mult * atr.iloc[i]
        lookback = min(self.struct_lookback, i - 1)
        struct_stop = df['low'].iloc[i-lookback:i].min()
        return max(atr_stop, struct_stop)

    def _detect_bi_deterministic(self, df):
        n = len(df)
        buy = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)
        if n < 5:
            return buy, sell

        high_arr = df['high'].values.astype(float)
        low_arr = df['low'].values.astype(float)

        merged = [{'high': high_arr[0], 'low': low_arr[0], 'idx': 0}]
        direction = 0

        for i in range(1, n):
            prev = merged[-1]
            if len(merged) >= 2:
                prev2 = merged[-2]
                if prev['high'] > prev2['high'] and prev['low'] > prev2['low']:
                    direction = 1
                elif prev['high'] < prev2['high'] and prev['low'] < prev2['low']:
                    direction = -1

            pcc = prev['high'] >= high_arr[i] and prev['low'] <= low_arr[i]
            ccp = high_arr[i] >= prev['high'] and low_arr[i] <= prev['low']

            if pcc or ccp:
                if direction == 1:
                    prev['high'] = max(prev['high'], high_arr[i])
                    prev['low'] = max(prev['low'], low_arr[i])
                elif direction == -1:
                    prev['high'] = min(prev['high'], high_arr[i])
                    prev['low'] = min(prev['low'], low_arr[i])
                else:
                    if ccp:
                        prev['high'] = high_arr[i]
                        prev['low'] = low_arr[i]
            else:
                merged.append({'high': high_arr[i], 'low': low_arr[i], 'idx': i})

        fractals = []
        for j in range(1, len(merged) - 1):
            if merged[j]['high'] > merged[j-1]['high'] and merged[j]['high'] > merged[j+1]['high']:
                fractals.append({'type': 'top', 'midx': j, 'idx': merged[j]['idx'], 'val': merged[j]['high']})
            elif merged[j]['low'] < merged[j-1]['low'] and merged[j]['low'] < merged[j+1]['low']:
                fractals.append({'type': 'bottom', 'midx': j, 'idx': merged[j]['idx'], 'val': merged[j]['low']})

        if not fractals:
            return buy, sell

        filtered = [fractals[0]]
        for f in fractals[1:]:
            if f['type'] == filtered[-1]['type']:
                if f['type'] == 'top' and f['val'] > filtered[-1]['val']:
                    filtered[-1] = f
                elif f['type'] == 'bottom' and f['val'] < filtered[-1]['val']:
                    filtered[-1] = f
            else:
                if f['midx'] - filtered[-1]['midx'] >= self.bi_min_gap:
                    filtered.append(f)
                else:
                    if f['type'] == 'top' and f['val'] > filtered[-1]['val']:
                        filtered[-1] = f
                    elif f['type'] == 'bottom' and f['val'] < filtered[-1]['val']:
                        filtered[-1] = f

        for j in range(1, len(filtered)):
            prev = filtered[j-1]
            curr = filtered[j]
            signal_idx = curr['idx'] + self.bi_confirm_delay
            if signal_idx >= n:
                continue
            if prev['type'] == 'top' and curr['type'] == 'bottom':
                buy.iloc[signal_idx] = True
            elif prev['type'] == 'bottom' and curr['type'] == 'top':
                sell.iloc[signal_idx] = True

        return buy, sell

    def _compute_weekly_not_down(self, df):
        n = len(df)
        trends = pd.Series(True, index=df.index)
        try:
            weekly = df.resample('W').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
            if len(weekly) < 10:
                return trends
            ma5 = weekly['close'].rolling(5).mean()
            ma10 = weekly['close'].rolling(10).mean()
            for i in range(n):
                dt = df.index[i]
                prev_weekly = weekly.loc[:dt]
                if len(prev_weekly) < 10:
                    continue
                w_idx = len(prev_weekly) - 1
                if pd.isna(ma10.iloc[w_idx]) or pd.isna(ma5.iloc[w_idx]):
                    continue
                if ma5.iloc[w_idx] < ma10.iloc[w_idx] * self.weekly_threshold:
                    trends.iloc[i] = False
        except Exception:
            pass
        return trends

    def _update_loss_streak(self, profit_pct, bar_idx):
        if profit_pct < 0:
            if bar_idx - self._last_loss_bar <= self.consec_loss_bars:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 1
            self._last_loss_bar = bar_idx
            if self._consecutive_losses >= 3:
                self._loss_pause_until = bar_idx + self.consec_loss_pause
        else:
            self._consecutive_losses = 0


@dataclass
class SweepResult:
    label: str
    sharpe: float
    annual_return: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trade_count: int
    final_value: float
    params: dict


def main():
    print("=" * 80)
    print("V11d 30-Minute Parameter Optimization")
    print("=" * 80)

    # Load data
    print("\nLoading 30min data...")
    real_30min_codes = []
    for f in sorted(MIN30_DIR.glob("min30_*.csv")):
        df = pd.read_csv(f)
        dates = pd.to_datetime(df['datetime']).dt.date.unique()
        if len(df) / max(1, len(dates)) >= 4:
            real_30min_codes.append(f.stem.replace("min30_", ""))

    print(f"  {len(real_30min_codes)} stocks with real 30min data: {real_30min_codes}")
    data_map = load_min30_data(real_30min_codes)
    print(f"  Loaded {len(data_map)} stocks")

    results = []

    # ===== Phase 1: bi_min_gap sweep =====
    print("\n--- Phase 1: bi_min_gap sweep (most critical) ---")
    for gap in [5, 8, 10, 12, 15, 18, 20, 25, 30]:
        engine = SignalEngine30min(bi_min_gap=gap)
        metrics = run_30min_backtest(data_map, engine)
        r = SweepResult(
            label=f"gap={gap}",
            sharpe=metrics.get("sharpe", 0),
            annual_return=metrics.get("annual_return", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            calmar=metrics.get("calmar", 0),
            win_rate=metrics.get("win_rate", 0),
            trade_count=metrics.get("trade_count", 0),
            final_value=metrics.get("final_value", 0),
            params={"bi_min_gap": gap},
        )
        results.append(r)
        print(f"  gap={gap:2d}: S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | C={r.calmar:.2f} | T={r.trade_count}")

    # ===== Phase 2: trailing params (using best gap) =====
    best_gap = max(results, key=lambda r: r.sharpe).params["bi_min_gap"]
    print(f"\n--- Phase 2: trailing sweep (best gap={best_gap}) ---")

    trailing_configs = [
        ("v11d", 0.015, 0.02, 0.05, 0.07),
        ("v11c", 0.02, 0.025, 0.06, 0.07),
        ("v10", 0.03, 0.03, 0.05, 0.07),
        ("tighter", 0.01, 0.015, 0.04, 0.06),
        ("looser", 0.02, 0.03, 0.07, 0.10),
        ("ultra_tight", 0.008, 0.012, 0.03, 0.05),
    ]

    for name, ts, tt, tm, tw in trailing_configs:
        engine = SignalEngine30min(bi_min_gap=best_gap,
                                   trailing_start=ts, trailing_tight=tt,
                                   trailing_medium=tm, trailing_wide=tw)
        metrics = run_30min_backtest(data_map, engine)
        r = SweepResult(
            label=f"gap{best_gap}_{name}",
            sharpe=metrics.get("sharpe", 0),
            annual_return=metrics.get("annual_return", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            calmar=metrics.get("calmar", 0),
            win_rate=metrics.get("win_rate", 0),
            trade_count=metrics.get("trade_count", 0),
            final_value=metrics.get("final_value", 0),
            params={"bi_min_gap": best_gap, "trailing": name},
        )
        results.append(r)
        print(f"  {name:12s}: S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | C={r.calmar:.2f} | T={r.trade_count}")

    # ===== Phase 3: time_stop + min_hold =====
    print(f"\n--- Phase 3: time_stop + min_hold sweep ---")
    best_trailing = max(
        [r for r in results if "trailing" in r.params],
        key=lambda r: r.sharpe
    )
    best_tname = best_trailing.params["trailing"]
    tconf = next(c for c in trailing_configs if c[0] == best_tname)

    for ts_days, mh_days in [(30, 5), (40, 5), (60, 7), (60, 10), (80, 7), (90, 10)]:
        engine = SignalEngine30min(
            bi_min_gap=best_gap,
            trailing_start=tconf[1], trailing_tight=tconf[2],
            trailing_medium=tconf[3], trailing_wide=tconf[4],
            time_stop_days=ts_days, min_hold_days=mh_days
        )
        metrics = run_30min_backtest(data_map, engine)
        r = SweepResult(
            label=f"ts{ts_days}_mh{mh_days}",
            sharpe=metrics.get("sharpe", 0),
            annual_return=metrics.get("annual_return", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            calmar=metrics.get("calmar", 0),
            win_rate=metrics.get("win_rate", 0),
            trade_count=metrics.get("trade_count", 0),
            final_value=metrics.get("final_value", 0),
            params={"time_stop": ts_days, "min_hold": mh_days},
        )
        results.append(r)
        print(f"  ts={ts_days:2d}d mh={mh_days:2d}d: S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | C={r.calmar:.2f} | T={r.trade_count}")

    # ===== Final ranking =====
    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)

    print(f"\n\n{'='*80}")
    print("FINAL RANKING BY SHARPE")
    print(f"{'='*80}")

    for i, r in enumerate(sorted_results[:15]):
        print(f"  #{i+1:2d} {r.label:25s} | S={r.sharpe:.3f} | A={r.annual_return*100:.1f}% | "
              f"DD={r.max_drawdown*100:.1f}% | C={r.calmar:.2f} | WR={r.win_rate*100:.1f}% | "
              f"T={r.trade_count} | F={r.final_value/1e6:.2f}M")

    # Save best config
    if sorted_results:
        best = sorted_results[0]
        best_config = {
            "engine": "30min",
            "params": best.params,
            "metrics": {
                "sharpe": best.sharpe,
                "annual_return": best.annual_return,
                "max_drawdown": best.max_drawdown,
                "calmar": best.calmar,
                "win_rate": best.win_rate,
                "trade_count": best.trade_count,
                "final_value": best.final_value,
            }
        }
        out = RUN_DIR / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "best_30min_config.json", "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to {out / 'best_30min_config.json'}")


if __name__ == "__main__":
    main()
