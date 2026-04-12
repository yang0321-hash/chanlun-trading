"""
缠论区间套信号引擎 - 日线+30分钟双级别共振

核心逻辑：
1. 日线级别：用czsc计算笔，找2买/3买方向（向下笔结束=买入候选）
2. 30分钟级别：在日线买入候选日附近，找30分钟级别的向下笔结束+MACD金叉确认
3. 两个级别共振才发出买入信号
4. 卖出同理：日线2卖 + 30分钟确认

与v4基线的区别：
- v4只用日线级别信号
- 本引擎增加30分钟级别确认，提高精确度
- 日线方向过滤：只在日线级别有买入候选时，才去30分钟找入场点
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path


class SignalEngine:
    """缠论区间套信号引擎 - 日线+30分钟双级别"""

    def __init__(self):
        # 风控参数（与v4一致）
        self.risk_per_trade = 0.03
        self.max_positions = 5
        self.max_drawdown_pct = 0.15
        self.cooldown_bars = 3
        self.time_stop_bars = 45
        self.min_hold_before_sell = 7
        self.max_stop_pct = 0.25

        # 仓位参数
        self.min_position = 0.10
        self.max_position = 0.25
        self.base_position = 0.12

        # 移动止损
        self.trailing_start = 0.08
        self.trailing_distance = 0.04

        # 盈利加仓
        self.profit_add_threshold = 0.05
        self.profit_add_ratio = 0.50

        # 区间套参数
        self.lookback_days = 5        # 日线买入候选前后的搜索窗口（天）
        self.min30_macd_confirm = True # 30分钟MACD金叉确认

        # 30分钟数据路径
        self.min30_dir = Path("/workspace/chanlun_system/artifacts")
        self._min30_cache: Dict[str, pd.DataFrame] = {}

    def _load_min30(self, code: str) -> Optional[pd.DataFrame]:
        """加载30分钟数据"""
        if code in self._min30_cache:
            return self._min30_cache[code]

        path = self.min30_dir / f"min30_{code}.csv"
        if not path.exists():
            return None

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # 确保列标准化
            if 'vol' in df.columns and 'volume' not in df.columns:
                df = df.rename(columns={'vol': 'volume'})
            self._min30_cache[code] = df
            return df
        except Exception:
            return None

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        生成交易信号

        Args:
            data_map: code -> DataFrame (日线, columns: open, high, low, close, volume)

        Returns:
            code -> signal Series, value range [-1.0, 1.0]
        """
        signals = {}
        for code, df in data_map.items():
            signals[code] = self._generate_single(code, df)
        return signals

    def _generate_single(self, code: str, df: pd.DataFrame) -> pd.Series:
        """单标的信号生成 - 区间套版本"""
        n = len(df)
        signals = pd.Series(0.0, index=df.index)

        if n < 120:
            return signals

        close = df['close']
        high = df['high']
        low = df['low']

        # ===== 预计算日线级别指标 =====
        # 1. 缠论笔信号（日线）
        daily_bi_buy, daily_bi_sell = self._compute_bi_signals(df, freq='D')

        # 2. MACD（日线）
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)

        # 3. 周线趋势
        weekly_not_down = self._compute_weekly_not_down(df)

        # 4. 日线趋势辅助
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()

        # 5. 量能
        vol_ma5 = df['volume'].rolling(5).mean()
        vol_ma20 = df['volume'].rolling(20).mean()

        # 6. MACD底背驰检测
        macd_divergence = self._compute_macd_divergence(df, daily_bi_buy, dif, macd_hist)

        # ===== 预计算30分钟级别信号 =====
        min30_confirmed = self._compute_min30_confirmation(code, df)

        # ===== 逐K线模拟 =====
        position = 0.0
        entry_idx = -1
        entry_price = 0.0
        stop_loss = 0.0
        highest = 0.0
        active_positions = 0
        last_loss_idx = -999
        peak_equity = 1_000_000.0
        equity = 1_000_000.0
        original_position = 0.0
        has_added = False

        for i in range(120, n):
            price = close.iloc[i]

            # 更新权益
            if position > 0:
                pnl = (price - entry_price) / entry_price if entry_price > 0 else 0
                equity = 1_000_000 * (1 - position) + 1_000_000 * position * (1 + pnl)
            else:
                equity = 1_000_000 * (1 - position)
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

            trading_halted = dd >= self.max_drawdown_pct

            if position > 0:
                # === 持仓中 ===
                if price > highest:
                    highest = price

                bars_held = i - entry_idx
                profit_pct = (price - entry_price) / entry_price if entry_price > 0 else 0

                # 1. 结构止损
                if price <= stop_loss:
                    signals.iloc[i] = 0.0
                    position = 0.0
                    original_position = 0.0
                    has_added = False
                    active_positions = max(0, active_positions - 1)
                    last_loss_idx = i
                    continue

                # 2. 移动止损
                if profit_pct > self.trailing_start:
                    trailing_stop = highest * (1 - self.trailing_distance)
                    if price <= trailing_stop:
                        signals.iloc[i] = 0.0
                        position = 0.0
                        original_position = 0.0
                        has_added = False
                        active_positions = max(0, active_positions - 1)
                        continue

                # 3. 2卖出局（日线2卖 + 30分钟确认）
                if bars_held >= self.min_hold_before_sell and daily_bi_sell.iloc[i]:
                    # 卖出时也要求30分钟确认（如果有数据）
                    if min30_confirmed is not None:
                        sell_confirmed = min30_confirmed['sell'].iloc[i] if i < len(min30_confirmed) else True
                        if sell_confirmed or bars_held >= self.time_stop_bars:
                            signals.iloc[i] = 0.0
                            position = 0.0
                            original_position = 0.0
                            has_added = False
                            active_positions = max(0, active_positions - 1)
                            continue
                    else:
                        signals.iloc[i] = 0.0
                        position = 0.0
                        original_position = 0.0
                        has_added = False
                        active_positions = max(0, active_positions - 1)
                        continue

                # 4. 时间止损
                if bars_held >= self.time_stop_bars:
                    signals.iloc[i] = 0.0
                    position = 0.0
                    original_position = 0.0
                    has_added = False
                    active_positions = max(0, active_positions - 1)
                    continue

                # 5. 盈利加仓
                if (not has_added
                        and profit_pct > self.profit_add_threshold
                        and daily_bi_buy.iloc[i]
                        and profit_pct < 0.25
                        and active_positions <= self.max_positions):
                    add_size = original_position * self.profit_add_ratio
                    new_position = min(position + add_size, self.max_position)
                    actual_add = new_position - position
                    if actual_add > 0.001:
                        entry_price = (entry_price * position + price * actual_add) / new_position
                        signals.iloc[i] = new_position
                        position = new_position
                        has_added = True
                        stop_loss = max(stop_loss, entry_price * 0.98)
                        continue

                # 继续持仓
                signals.iloc[i] = position

            else:
                # === 空仓 ===
                if trading_halted:
                    continue
                if active_positions >= self.max_positions:
                    continue
                if i - last_loss_idx < self.cooldown_bars:
                    continue

                # 周线过滤
                if not weekly_not_down.iloc[i]:
                    continue

                # 日线买入候选：向下笔结束
                if not daily_bi_buy.iloc[i]:
                    continue

                # ===== 区间套核心：30分钟级别确认 =====
                min30_buy_ok = False
                min30_strength = 1.0  # 30分钟确认强度因子

                if min30_confirmed is not None and i < len(min30_confirmed):
                    min30_buy_ok = bool(min30_confirmed['buy'].iloc[i])
                    min30_strength = float(min30_confirmed['strength'].iloc[i]) if min30_buy_ok else 0.0
                else:
                    # 无30分钟数据时，降级为纯日线（但给个折扣因子）
                    min30_buy_ok = True
                    min30_strength = 0.8  # 降级惩罚

                if not min30_buy_ok:
                    continue

                # ---- 信号强度评估 ----

                # MACD因子（日线）
                macd_factor = 1.0
                macd_confirm = (
                    dif.iloc[i] > dif.iloc[i-1]
                    or (macd_hist.iloc[i] > macd_hist.iloc[i-1]
                        and macd_hist.iloc[i] <= 0)
                    or macd_hist.iloc[i] > 0
                )
                if macd_confirm:
                    macd_factor = 1.15
                elif macd_hist.iloc[i] < macd_hist.iloc[i-1] and macd_hist.iloc[i] < 0:
                    macd_factor = 0.85

                # MACD底背驰
                if macd_divergence.iloc[i]:
                    macd_factor = 1.5

                # 量能因子
                vol_factor = 1.0
                if not pd.isna(vol_ma20.iloc[i]) and vol_ma20.iloc[i] > 0:
                    vol_ratio = df['volume'].iloc[i] / vol_ma20.iloc[i]
                    if vol_ratio >= 2.0:
                        vol_factor = 1.25
                    elif vol_ratio >= 1.5:
                        vol_factor = 1.15
                    elif vol_ratio >= 1.0:
                        vol_factor = 1.0
                    elif vol_ratio >= 0.5:
                        vol_factor = 0.9
                    else:
                        vol_factor = 0.75

                # 30分钟确认强度因子
                # 双级别共振时给额外加成
                if min30_buy_ok and min30_strength > 1.0:
                    min30_factor = min(min30_strength, 1.3)  # 最多30%加成
                elif min30_buy_ok:
                    min30_factor = 1.0
                else:
                    min30_factor = 0.0

                # 止损位
                lookback = min(30, i - 1)
                recent_low = df['low'].iloc[i-lookback:i].min()
                stop = recent_low
                stop_distance = price - stop

                if stop_distance <= 0:
                    continue

                stop_pct = stop_distance / price
                if stop_pct > self.max_stop_pct:
                    stop = price * (1 - self.max_stop_pct)
                    stop_distance = price - stop
                    stop_pct = self.max_stop_pct

                # 仓位计算
                risk_amount = 1_000_000 * self.risk_per_trade
                risk_based_pct = min(risk_amount / stop_distance / 1_000_000, self.max_position)

                base_pos = max(self.base_position, risk_based_pct)
                final_pos = base_pos * macd_factor * vol_factor * min30_factor
                final_pos = max(self.min_position, min(final_pos, self.max_position))

                # 价格在均线压力位时降低仓位
                if not pd.isna(ma20.iloc[i]) and price < ma20.iloc[i] * 0.93:
                    if not macd_confirm and not macd_divergence.iloc[i]:
                        final_pos = self.min_position

                signals.iloc[i] = final_pos
                position = final_pos
                original_position = final_pos
                has_added = False
                entry_idx = i
                entry_price = price
                stop_loss = stop
                highest = price
                active_positions += 1

        return signals

    def _compute_min30_confirmation(self, code: str, daily_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        预计算30分钟级别的确认信号

        对每个交易日，检查该日附近是否有30分钟级别的向下笔结束+MACD金叉

        Returns:
            DataFrame with columns: buy (bool), sell (bool), strength (float)
            indexed same as daily_df
        """
        min30_df = self._load_min30(code)
        if min30_df is None or len(min30_df) < 100:
            return None

        n = len(daily_df)
        result = pd.DataFrame({
            'buy': pd.Series(False, index=daily_df.index),
            'sell': pd.Series(False, index=daily_df.index),
            'strength': pd.Series(1.0, index=daily_df.index),
        })

        # 计算30分钟级别的缠论笔信号
        min30_bi_buy, min30_bi_sell = self._compute_bi_signals(min30_df, freq='F30')

        # 计算30分钟MACD
        min30_close = min30_df['close']
        m_ema12 = min30_close.ewm(span=12, adjust=False).mean()
        m_ema26 = min30_close.ewm(span=26, adjust=False).mean()
        m_dif = m_ema12 - m_ema26
        m_dea = m_dif.ewm(span=9, adjust=False).mean()
        m_hist = 2 * (m_dif - m_dea)

        # 对每个交易日，检查该日的30分钟K线是否有确认信号
        for i in range(n):
            dt = daily_df.index[i]
            date_str = pd.Timestamp(dt).strftime('%Y-%m-%d')

            # 搜索窗口：当前日 ± lookback_days
            # 但30分钟数据只按天切分
            try:
                # 找到该日期范围内的30分钟K线
                mask = min30_df.index.strftime('%Y-%m-%d') == date_str
                day_min30 = min30_df[mask]

                if len(day_min30) == 0:
                    continue

                # 在该日的30分钟K线中检查买入确认
                buy_found = False
                sell_found = False
                buy_strength = 1.0

                for idx in day_min30.index:
                    if idx not in min30_bi_buy.index:
                        continue

                    if min30_bi_buy.loc[idx]:
                        # 30分钟向下笔结束
                        # 检查MACD金叉确认
                        macd_idx = min30_df.index.get_loc(idx)
                        if macd_idx > 0:
                            # MACD金叉：DIF上穿DEA 或 绿柱缩短转红
                            hist_now = m_hist.iloc[macd_idx]
                            hist_prev = m_hist.iloc[macd_idx - 1] if macd_idx > 0 else 0
                            dif_now = m_dif.iloc[macd_idx]
                            dif_prev = m_dif.iloc[macd_idx - 1] if macd_idx > 0 else 0

                            # 金叉条件（任一满足）
                            golden_cross = (
                                (dif_now > m_dea.iloc[macd_idx] and dif_prev <= m_dea.iloc[macd_idx - 1])  # DIF上穿DEA
                                or (hist_now > 0 and hist_prev <= 0)  # 柱子转红
                                or (hist_now > hist_prev and hist_now <= 0)  # 绿柱缩短
                                or dif_now > dif_prev  # DIF上翘
                            )

                            if golden_cross or not self.min30_macd_confirm:
                                buy_found = True
                                # 计算确认强度
                                strength = 1.0
                                if dif_now > m_dea.iloc[macd_idx]:
                                    strength += 0.1  # DIF在DEA上方
                                if hist_now > 0:
                                    strength += 0.1  # 红柱
                                if hist_now > hist_prev:
                                    strength += 0.05  # 柱子放大
                                buy_strength = max(buy_strength, strength)

                    if min30_bi_sell.loc[idx]:
                        # 30分钟向上笔结束 = 卖出确认
                        sell_found = True

                if buy_found:
                    result.iloc[i, result.columns.get_loc('buy')] = True
                    result.iloc[i, result.columns.get_loc('strength')] = buy_strength

                if sell_found:
                    result.iloc[i, result.columns.get_loc('sell')] = True

            except Exception:
                continue

        return result

    def _compute_bi_signals(self, df: pd.DataFrame, freq: str = 'D') -> tuple:
        """
        使用czsc计算笔信号

        Args:
            df: OHLCV DataFrame
            freq: 频率 ('D' or 'F30')

        Returns:
            (buy_signals, sell_signals): 两个布尔Series
        """
        n = len(df)
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)

        try:
            from czsc import CZSC, RawBar, Freq

            freq_map = {'D': Freq.D, 'F30': Freq.F30}
            czsc_freq = freq_map.get(freq, Freq.D)

            bars = []
            for i in range(n):
                dt_val = df.index[i]
                if not isinstance(dt_val, pd.Timestamp):
                    dt_val = pd.Timestamp(dt_val)

                vol = float(df['volume'].iloc[i]) if 'volume' in df.columns else 0
                amt = float(df['close'].iloc[i]) * vol if vol > 0 else 0

                bar = RawBar(
                    symbol='A',
                    id=i,
                    dt=dt_val,
                    freq=czsc_freq,
                    open=float(df['open'].iloc[i]),
                    close=float(df['close'].iloc[i]),
                    high=float(df['high'].iloc[i]),
                    low=float(df['low'].iloc[i]),
                    vol=vol,
                    amount=amt,
                )
                bars.append(bar)

            c = CZSC(bars)

            for bi in c.bi_list:
                if not bi.raw_bars:
                    continue
                end_idx = bi.raw_bars[-1].id
                if end_idx is None or end_idx >= n:
                    continue

                direction = str(bi.direction)
                if '下' in direction:
                    buy_signals.iloc[end_idx] = True
                elif '上' in direction:
                    sell_signals.iloc[end_idx] = True

        except ImportError:
            buy_signals, sell_signals = self._fallback_fractal(df)

        return buy_signals, sell_signals

    def _fallback_fractal(self, df: pd.DataFrame) -> tuple:
        """回退分型检测"""
        n = len(df)
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)

        high = df['high'].values
        low = df['low'].values

        for i in range(1, n - 1):
            if low[i] < low[i-1] and low[i] < low[i+1]:
                buy_signals.iloc[i] = True
            if high[i] > high[i-1] and high[i] > high[i+1]:
                sell_signals.iloc[i] = True

        return buy_signals, sell_signals

    def _compute_weekly_not_down(self, df: pd.DataFrame) -> pd.Series:
        """周线趋势过滤（宽松版）"""
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
                if ma5.iloc[w_idx] < ma10.iloc[w_idx] * 0.95:
                    trends.iloc[i] = False
        except Exception:
            pass

        return trends

    def _compute_macd_divergence(self, df: pd.DataFrame, bi_buy: pd.Series,
                                  dif: pd.Series, macd_hist: pd.Series) -> pd.Series:
        """MACD底背驰检测"""
        n = len(df)
        divergence = pd.Series(False, index=df.index)

        buy_points = []
        for i in range(n):
            if bi_buy.iloc[i]:
                buy_points.append({
                    'idx': i,
                    'price': df['low'].iloc[i],
                    'dif': dif.iloc[i],
                    'hist': macd_hist.iloc[i],
                })

        if len(buy_points) < 2:
            return divergence

        for j in range(1, len(buy_points)):
            curr = buy_points[j]
            curr_idx = curr['idx']

            for k in range(j - 1, max(-1, j - 10), -1):
                prev = buy_points[k]
                if curr_idx - prev['idx'] > 120:
                    break
                if curr['price'] >= prev['price']:
                    continue
                if curr['dif'] > prev['dif'] or curr['hist'] > prev['hist']:
                    divergence.iloc[curr_idx] = True
                    break

        return divergence
