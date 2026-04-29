"""
30分钟T+0策略 v3a+v14 - 回测信号引擎

v3a核心逻辑（回测验证 Sharpe 9.27）：
1. 2买入场：向下笔结束 + MACD确认
2. 止损：入场后最低点回撤≤12%
3. 背驰止盈半仓：价格创新高+MACD柱缩短 → 平半仓
4. 动态止盈：盈利>5%后，从最高点回撤3% → 清仓

v14新增：
1. 线段检测：3笔重叠+中枢破坏
2. 线段级别MACD面积背驰（与笔级别合并取并集）
3. 2卖检测：1卖后下跌→反弹不过前高

数据格式要求：
- data_map: Dict[str, pd.DataFrame]
- DataFrame columns: open, high, low, close, volume
- DataFrame index: datetime
- signal Series: value 0~1 表示仓位比例，0=空仓
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Set


class SignalEngine:
    def __init__(self, daily_filters: dict = None, daily_data: dict = None):
        # 风控参数
        self.stop_loss_pct = 0.12           # 最大止损12%
        self.trailing_start_pct = 0.05      # 动态止盈启动阈值5%
        self.trailing_distance = 0.03       # 动态止盈回撤3%
        self.divergence_profit_pct = 0.03   # 背驰止盈最低盈利3%
        self.cooldown_bars = 3              # 卖出冷却3根K线
        self.min_hold_bars = 6              # 最少持仓6根K线(3小时)
        self.max_hold_bars = 80             # 最长持仓80根K线(≈5天)
        self.bi_confirm_delay = 1           # 笔确认延迟

        # 仓位
        self.entry_position = 0.5           # 入场半仓

        # 每日过滤器
        self.daily_filters = daily_filters or {}
        self.daily_data = daily_data or {}  # {trade_date: {code: {turnover_rate, ...}}}

    def _check_daily_filter(self, code: str, dt) -> bool:
        """检查每日过滤器，dt 可以是 datetime/timestamp"""
        if not self.daily_filters:
            return True
        trade_date = pd.Timestamp(dt).strftime('%Y%m%d')
        day_data = self.daily_data.get(trade_date, {})
        stock_data = day_data.get(code, {})
        if not stock_data:
            return True  # 无数据不过滤

        # 换手率
        tr_min = self.daily_filters.get('turnover_rate_min', 0)
        if tr_min > 0 and stock_data.get('turnover_rate', 0) < tr_min:
            return False

        # 资金流
        mf_min = self.daily_filters.get('net_mf_min', -999999)
        if stock_data.get('net_mf_amount', 0) < mf_min:
            return False

        # PE
        pe_max = self.daily_filters.get('pe_max', 0)
        if pe_max > 0:
            pe = stock_data.get('pe_ttm', 0)
            if pe > 0 and pe > pe_max:
                return False

        return True

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        signals = {}
        for code, df in data_map.items():
            signals[code] = self._generate_single(code, df)
        return signals

    def _generate_single(self, code: str, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index)

        if n < 120:
            return signals

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        hist = 2 * (dif - dea)
        macd_hist_series = pd.Series(hist, index=df.index)

        # 笔信号 + strokes
        bi_buy, bi_sell, strokes = self._compute_bi_with_strokes(df)

        # ===== v14: 线段检测 + 线段背驰 + 2卖 =====
        segments = self._detect_segments(strokes)
        seg_buy_div, seg_sell_div = self._compute_segment_divergence(
            segments, strokes, macd_hist_series, n)
        sell_2sell_set = self._detect_2sell(strokes, seg_sell_div, n)

        # 笔级别面积背驰
        bi_buy_div, bi_sell_div = self._compute_area_divergence(
            strokes, macd_hist_series, n)

        # 合并: 笔级别 + 线段级别
        all_buy_div = bi_buy_div | seg_buy_div
        all_sell_div = bi_sell_div | seg_sell_div

        # 逐K线模拟
        position = 0.0
        entry_idx = -1
        entry_price = 0.0
        stop_loss = 0.0
        highest_since_entry = 0.0
        last_sell_idx = -999
        has_diverged = False

        for i in range(120, n):
            price = close[i]

            if position > 0:
                # === 持仓中 ===
                bars_held = i - entry_idx
                profit_pct = (price - entry_price) / entry_price if entry_price > 0 else 0

                if price > highest_since_entry:
                    highest_since_entry = price

                # 1. 结构止损
                if price <= stop_loss:
                    signals.iloc[i] = 0.0
                    position = 0.0
                    last_sell_idx = i
                    has_diverged = False
                    continue

                # 2. 背驰止盈半仓
                if (not has_diverged
                    and profit_pct >= self.divergence_profit_pct
                    and bars_held >= self.min_hold_bars):
                    recent_high = np.max(high[max(0, i-5):i+1])
                    macd_shrink = (i >= 2 and hist[i] > 0
                                   and hist[i] < hist[i-1]
                                   and hist[i-1] < hist[i-2])
                    if recent_high >= highest_since_entry * 0.995 and macd_shrink:
                        if bi_sell.iloc[i]:
                            signals.iloc[i] = position * 0.5
                            has_diverged = True
                            continue

                # 2.5. v14: 2卖出局（反弹不过1卖高点 = 确认顶部）
                is_2sell = i in sell_2sell_set
                if (bars_held >= self.min_hold_bars
                        and is_2sell
                        and profit_pct > 0.03):
                    signals.iloc[i] = 0.0
                    position = 0.0
                    last_sell_idx = i
                    has_diverged = False
                    continue

                # 3. 动态止盈
                if profit_pct > self.trailing_start_pct:
                    trailing_stop = highest_since_entry * (1 - self.trailing_distance)
                    if price <= trailing_stop:
                        signals.iloc[i] = 0.0
                        position = 0.0
                        last_sell_idx = i
                        has_diverged = False
                        continue

                # 4. 时间止损
                if bars_held >= self.max_hold_bars:
                    signals.iloc[i] = 0.0
                    position = 0.0
                    last_sell_idx = i
                    has_diverged = False
                    continue

                signals.iloc[i] = position

            else:
                # === 空仓 ===
                if i - last_sell_idx < self.cooldown_bars:
                    continue

                # 2买入场 + v14线段底背驰增强
                has_buy_signal = bi_buy.iloc[i]
                has_seg_buy_div = i in all_buy_div

                if not has_buy_signal and not has_seg_buy_div:
                    continue

                macd_confirm = (
                    dif[i] > dea[i]
                    or (hist[i] > hist[i-1] and hist[i] <= 0)
                    or (dif[i] > dif[i-1])
                )
                if not macd_confirm:
                    continue

                # 每日级过滤器（换手率/资金流/PE）
                if not self._check_daily_filter(code, df.index[i]):
                    continue

                # 计算止损位
                lookback = min(30, i - 1)
                recent_low = np.min(low[i-lookback:i])
                stop_distance = price - recent_low

                if stop_distance <= 0:
                    continue

                stop_pct = stop_distance / price
                if stop_pct > self.stop_loss_pct:
                    stop_loss = price * (1 - self.stop_loss_pct)
                else:
                    stop_loss = recent_low

                signals.iloc[i] = self.entry_position
                position = self.entry_position
                entry_idx = i
                entry_price = price
                highest_since_entry = price
                has_diverged = False

        return signals

    # ===== v14: 从czsc bi_list构建strokes =====

    def _compute_bi_with_strokes(self, df: pd.DataFrame):
        """用czsc计算笔信号 + 构建strokes列表

        Returns:
            buy_signals, sell_signals: pd.Series
            strokes: List[Dict] — 笔列表, 每笔有start_idx/end_idx/start_val/end_val/high/low
        """
        n = len(df)
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)
        strokes = []

        try:
            from czsc import CZSC, RawBar, Freq
            bars = []
            for i in range(n):
                vol = float(df['volume'].iloc[i]) if 'volume' in df.columns else 0
                amt = float(df['close'].iloc[i]) * vol if vol > 0 else 0
                bars.append(RawBar(
                    symbol='A', id=i,
                    dt=pd.Timestamp(df.index[i]),
                    freq=Freq.F30,
                    open=float(df['open'].iloc[i]),
                    close=float(df['close'].iloc[i]),
                    high=float(df['high'].iloc[i]),
                    low=float(df['low'].iloc[i]),
                    vol=vol, amount=amt,
                ))
            c = CZSC(bars)
            for bi in c.bi_list:
                if not bi.raw_bars or len(bi.raw_bars) < 2:
                    continue
                start_idx = bi.raw_bars[0].id
                end_idx = bi.raw_bars[-1].id
                if start_idx is None or end_idx is None:
                    continue
                if end_idx >= n:
                    end_idx = n - 1

                direction = str(bi.direction)
                is_down = '下' in direction

                # 笔的高低
                bi_high = float(df['high'].iloc[start_idx:end_idx+1].max())
                bi_low = float(df['low'].iloc[start_idx:end_idx+1].min())

                stroke = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_val': float(df['close'].iloc[start_idx]),
                    'end_val': float(df['close'].iloc[end_idx]),
                    'high': bi_high,
                    'low': bi_low,
                    'start_type': 'top' if is_down else 'bottom',
                    'end_type': 'bottom' if is_down else 'top',
                }
                strokes.append(stroke)

                if is_down:
                    buy_signals.iloc[end_idx] = True
                else:
                    sell_signals.iloc[end_idx] = True
        except ImportError:
            h = df['high'].values
            l = df['low'].values
            for i in range(1, n - 1):
                if l[i] < l[i-1] and l[i] < l[i+1]:
                    buy_signals.iloc[i] = True
                if h[i] > h[i-1] and h[i] > h[i+1]:
                    sell_signals.iloc[i] = True

        return buy_signals, sell_signals, strokes

    # ===== v14: 线段检测 =====

    def _detect_segments(self, strokes: List[Dict]) -> List[Dict]:
        """v14: 线段检测 — 中枢ZG/ZD突破 + 极值突破"""
        if len(strokes) < 3:
            return []

        segments = []
        seg_start = 0

        for j in range(1, len(strokes)):
            seg_strokes = strokes[seg_start:j + 1]
            if len(seg_strokes) < 3:
                continue

            first = seg_strokes[0]
            seg_dir = 'up' if first['start_type'] == 'bottom' else 'down'

            last3 = seg_strokes[-3:]
            zg = max(s['high'] for s in last3)
            zd = min(s['low'] for s in last3)

            current = strokes[j]
            break_seg = False

            if seg_dir == 'up':
                if current['end_type'] == 'bottom':
                    all_down_ends = [s['end_val'] for s in seg_strokes
                                     if s['end_type'] == 'bottom']
                    if len(all_down_ends) >= 2 and current['end_val'] < all_down_ends[-2]:
                        break_seg = True
                    if current['end_val'] < zd:
                        break_seg = True
            else:
                if current['end_type'] == 'top':
                    all_up_ends = [s['end_val'] for s in seg_strokes
                                   if s['end_type'] == 'top']
                    if len(all_up_ends) >= 2 and current['end_val'] > all_up_ends[-2]:
                        break_seg = True
                    if current['end_val'] > zg:
                        break_seg = True

            if break_seg:
                end_stroke_idx = j - 1
                if end_stroke_idx - seg_start >= 2:
                    end_segs = strokes[seg_start:end_stroke_idx + 1]
                    segments.append({
                        'direction': seg_dir,
                        'start_idx': end_segs[0]['start_idx'],
                        'end_idx': end_segs[-1]['end_idx'],
                        'start_val': end_segs[0]['start_val'],
                        'end_val': end_segs[-1]['end_val'],
                        'high': max(s['high'] for s in end_segs),
                        'low': min(s['low'] for s in end_segs),
                        'stroke_start': seg_start,
                        'stroke_end': end_stroke_idx,
                    })
                seg_start = j

        if len(strokes) - seg_start >= 3:
            end_segs = strokes[seg_start:]
            last_dir = 'up' if end_segs[0]['start_type'] == 'bottom' else 'down'
            segments.append({
                'direction': last_dir,
                'start_idx': end_segs[0]['start_idx'],
                'end_idx': end_segs[-1]['end_idx'],
                'start_val': end_segs[0]['start_val'],
                'end_val': end_segs[-1]['end_val'],
                'high': max(s['high'] for s in end_segs),
                'low': min(s['low'] for s in end_segs),
                'stroke_start': seg_start,
                'stroke_end': len(strokes) - 1,
            })

        return segments

    # ===== v14: 线段级别MACD面积背驰 =====

    def _compute_segment_divergence(self, segments: List[Dict],
                                     strokes: List[Dict],
                                     macd_hist: pd.Series, n: int):
        buy_div = set()
        sell_div = set()

        up_segs = [s for s in segments if s['direction'] == 'up']
        down_segs = [s for s in segments if s['direction'] == 'down']

        for k in range(1, len(down_segs)):
            prev, curr = down_segs[k-1], down_segs[k]
            prev_area = abs(sum(macd_hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            curr_area = abs(sum(macd_hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
            if curr['low'] < prev['low'] and curr_area < prev_area:
                sig = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= sig < n:
                    buy_div.add(sig)

        for k in range(1, len(up_segs)):
            prev, curr = up_segs[k-1], up_segs[k]
            prev_area = abs(sum(macd_hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            curr_area = abs(sum(macd_hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
            if curr['high'] > prev['high'] and curr_area < prev_area:
                sig = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= sig < n:
                    sell_div.add(sig)

        return buy_div, sell_div

    # ===== v14: 笔级别面积背驰 =====

    def _compute_area_divergence(self, strokes: List[Dict],
                                  macd_hist: pd.Series, n: int):
        buy_div = set()
        sell_div = set()

        down_strokes = [s for s in strokes if s['start_type'] == 'top']
        up_strokes = [s for s in strokes if s['start_type'] == 'bottom']

        for k in range(1, len(down_strokes)):
            prev, curr = down_strokes[k-1], down_strokes[k]
            prev_area = abs(sum(macd_hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            curr_area = abs(sum(macd_hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
            if curr['low'] < prev['low'] and curr_area < prev_area:
                sig = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= sig < n:
                    buy_div.add(sig)

        for k in range(1, len(up_strokes)):
            prev, curr = up_strokes[k-1], up_strokes[k]
            prev_area = abs(sum(macd_hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            curr_area = abs(sum(macd_hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
            if curr['high'] > prev['high'] and curr_area < prev_area:
                sig = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= sig < n:
                    sell_div.add(sig)

        return buy_div, sell_div

    # ===== v14: 2卖检测 =====

    def _detect_2sell(self, strokes: List[Dict], sell_divergence: Set[int], n: int) -> Set[int]:
        sell_2 = set()
        up_strokes = [s for s in strokes if s['start_type'] == 'bottom']
        down_strokes = [s for s in strokes if s['start_type'] == 'top']

        for us in up_strokes:
            sig_1sell = us['end_idx'] + self.bi_confirm_delay
            if sig_1sell not in sell_divergence:
                continue

            high_1sell = us['end_val']
            drop = None
            for ds in down_strokes:
                if ds['start_idx'] > us['end_idx']:
                    drop = ds
                    break
            if drop is None:
                continue

            bounce = None
            for us2 in up_strokes:
                if us2['start_idx'] > drop['end_idx']:
                    bounce = us2
                    break
            if bounce is None:
                continue

            if bounce['end_val'] <= high_1sell:
                sig_2sell = bounce['end_idx'] + self.bi_confirm_delay
                if 0 <= sig_2sell < n and sig_2sell not in sell_divergence:
                    sell_2.add(sig_2sell)

        return sell_2
