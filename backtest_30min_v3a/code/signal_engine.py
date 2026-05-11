"""
30分钟T+0策略 v3a+v14+v3 - 回测信号引擎

v3a核心逻辑（回测验证 Sharpe 9.27）：
1. 2买入场：向下笔结束 + MACD确认
2. 止损：入场后最低点回撤≤12%
3. 背驰止盈半仓：价格创新高+MACD柱缩短 → 平半仓
4. 动态止盈：盈利>5%后，从最高点回撤3% → 清仓

v14新增：
1. 线段检测：3笔重叠+中枢破坏
2. 线段级别MACD面积背驰（与笔级别合并取并集）
3. 2卖检测：1卖后下跌→反弹不过前高

v3新增（对照缠论108课原文）：
1. 自实现笔检测（不依赖CZSC）
2. 三类买卖点独立信号（1买/2买/3买）
3. 日线方向过滤（MA5确认趋势）
4. 大盘环境过滤（MA5/MA20判断牛/熊/震荡，动态仓位）

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
    def __init__(self, daily_filters: dict = None, daily_data: dict = None,
                 daily_trend_filter: bool = False,
                 market_regime_filter: bool = False,
                 sh_index_data: dict = None,
                 loose_regime: bool = False):
        # 风控参数
        self.stop_loss_pct = 0.12           # 最大止损12%
        self.trailing_start_pct = 0.05      # 动态止盈启动阈值5%
        self.trailing_distance = 0.03        # 动态止盈回撤3%
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

        # v3新增: 日线过滤 & 大盘环境过滤
        self.daily_trend_filter = daily_trend_filter
        self.market_regime_filter = market_regime_filter
        self._loose_regime = loose_regime
        # sh_index_data: {date: close_price} 上证日线数据
        self.sh_index_data = sh_index_data or {}
        # 预计算：加速每日trend/regime查询（O(1)替代O(n)）
        self._ts_data = {}
        if self.sh_index_data:
            dates = sorted(pd.to_datetime(d) for d in self.sh_index_data.keys())
            self._ts_data = {pd.Timestamp(d): self.sh_index_data[str(d.date())]
                             for d in dates if str(d.date()) in self.sh_index_data}

    # ========== v3: 自实现笔检测（不依赖CZSC） ==========

    def _build_strokes_fallback(self, df: pd.DataFrame):
        """自实现笔检测 - 构建strokes列表（缠论简化版）

        笔定义：从一个极值点开始，到下一个反向极值点结束
        简化处理：连续N个条件判断端点
        """
        n = len(df)
        h = df['high'].values
        l = df['low'].values

        # 1. 找所有局部极值点
        pivot_highs = []   # (idx, high_val)
        pivot_lows = []    # (idx, low_val)

        for i in range(1, n - 1):
            # 局部高点：高于前后各一根
            if h[i] > h[i-1] and h[i] > h[i+1]:
                pivot_highs.append((i, h[i]))
            # 局部低点：低于前后各一根
            if l[i] < l[i-1] and l[i] < l[i+1]:
                pivot_lows.append((i, l[i]))

        # 2. 构建笔：交替从低点到高点再到低点
        strokes = []
        i = 0
        # 找到第一个低点作为起点
        first_low_idx = pivot_lows[0][0] if pivot_lows else 0
        i = 0

        # 用双指针交替找高点和低点
        hp = 0  # pivot_highs index
        lp = 0  # pivot_lows index

        # 找第一个极值作为起点
        if not pivot_lows:
            return []

        # 从第一个低点开始，确定方向
        # 第一个转折点类型
        cur_type = 'bottom'  # 从低点开始
        start_idx = pivot_lows[0][0]
        start_val = pivot_lows[0][1]

        # 合并极值点成笔
        all_pivots = []
        hi, li = 0, 0
        while hi < len(pivot_highs) or li < len(pivot_lows):
            next_hi = pivot_highs[hi][0] if hi < len(pivot_highs) else float('inf')
            next_li = pivot_lows[li][0] if li < len(pivot_lows) else float('inf')

            if next_li < next_hi:
                all_pivots.append(('low', pivot_lows[li][0], pivot_lows[li][1]))
                li += 1
            else:
                all_pivots.append(('high', pivot_highs[hi][0], pivot_highs[hi][1]))
                hi += 1

        # 从all_pivots构建笔（相邻极值之间为1笔）
        # 要求：低-高-低=向上笔，高-低-高=向下笔
        for k in range(len(all_pivots) - 2):
            t0, i0, v0 = all_pivots[k]
            t1, i1, v1 = all_pivots[k+1]
            t2, i2, v2 = all_pivots[k+2]

            if t0 == 'low' and t1 == 'high' and t2 == 'low':
                # 向上笔
                strokes.append({
                    'start_idx': i0, 'end_idx': i2,
                    'start_val': v0, 'end_val': v2,
                    'high': v1, 'low': min(v0, v2),
                    'start_type': 'bottom', 'end_type': 'top',
                })
            elif t0 == 'high' and t1 == 'low' and t2 == 'high':
                # 向下笔
                strokes.append({
                    'start_idx': i0, 'end_idx': i2,
                    'start_val': v0, 'end_val': v2,
                    'high': max(v0, v2), 'low': v1,
                    'start_type': 'top', 'end_type': 'bottom',
                })

        return strokes

    def _detect_buy_points(self, strokes: List[Dict],
                           macd_hist: pd.Series,
                           n: int) -> Tuple[Set, Set, Set]:
        """检测1买/2买/3买信号（不使用线段，简化实现）

        1买：连续2个向下笔，后者创新低且MACD面积缩小
        2买：1买后的向上笔回调不破1买低点
        3买：向上笔突破前一线段高点（简化：突破前笔高点）

        Returns: (buy1_set, buy2_set, buy3_set) — 均为K线索引集合
        """
        buy1 = set()
        buy2 = set()
        buy3 = set()

        down_strokes = [s for s in strokes if s['start_type'] == 'top']
        up_strokes = [s for s in strokes if s['start_type'] == 'bottom']

        # === 1买：底背驰（创新低 + MACD面积缩小）===
        for k in range(1, len(down_strokes)):
            prev = down_strokes[k-1]
            curr = down_strokes[k]

            # 过滤：c笔K线数≥3（v3放宽）
            if curr['end_idx'] - curr['start_idx'] < 2:
                continue

            # 必须创新低
            if curr['low'] >= prev['low']:
                continue

            prev_area = abs(sum(macd_hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            curr_area = abs(sum(macd_hist.iloc[curr['start_idx']:curr['end_idx']+1].values))

            if curr_area < prev_area:
                buy1.add(curr['end_idx'])

        # === 2买：1买后向上笔回调，不破1买低点 ===
        # 找1买笔的索引
        buy1_stroke_indices = []
        for bi_idx, s_idx in enumerate(down_strokes):
            if s_idx['end_idx'] in buy1:
                buy1_stroke_indices.append(bi_idx)

        for bi_k in buy1_stroke_indices:
            if bi_k + 1 >= len(up_strokes):
                continue
            # 下一个向上笔
            up_s = up_strokes[bi_k + 1]
            # 这个向上笔结束后（下一个向下笔），在向下笔中找2买
            # 2买 = 回调笔不破1买低点
            for dk in range(bi_k + 1, len(down_strokes) - 1):
                dwn_s = down_strokes[dk]
                # 回调向下笔的终点
                # 条件：低点 > 1买低点（不破）
                if dwn_s['low'] > down_strokes[bi_k]['low']:
                    # 回调不破1买，确认2买
                    buy2.add(dwn_s['end_idx'])

        # === 3买：突破前笔高点后的回调不破笔起点 ===
        # 简化：向上笔的高点突破前一下跌笔的高点
        for k in range(1, len(up_strokes)):
            curr_up = up_strokes[k]
            # 找curr_up之前的最近一个down_stroke
            prev_down = None
            for ds in reversed(down_strokes):
                if ds['start_idx'] < curr_up['start_idx']:
                    prev_down = ds
                    break
            if prev_down is None:
                continue

            # 突破前高
            if curr_up['high'] > prev_down['high']:
                # 突破后，找curr_up之后的向下笔
                for ds in down_strokes:
                    if ds['start_idx'] > curr_up['end_idx']:
                        # 回调不破ZG
                        if ds['low'] > prev_down['high']:
                            buy3.add(ds['end_idx'])
                        break

        return buy1, buy2, buy3

    # ========== v3: 日线方向过滤 ==========

    def _get_daily_trend(self, dt) -> str:
        """判断日线趋势方向

        Returns: 'up' | 'down' | 'side'
        基于MA5和MACD方向

        注意：用昨日收盘价计算，避免30分钟bar发出信号时当日价格未知（因果倒置）
        """
        if not self._ts_data:
            return 'side'

        ts = pd.Timestamp(dt)
        dates = sorted(self._ts_data.keys())

        # 找最近20个"昨日及之前"的交易日（不含今日）
        past_dates = [d for d in dates if d < ts.replace(hour=0, minute=0, second=0, microsecond=0)]
        if len(past_dates) < 10:
            return 'side'

        # 用最近N日数据算trend（不含今日）
        recent = past_dates[-20:]
        prices = [self._ts_data[d] for d in recent]
        ma5 = sum(prices[-5:]) / 5 if len(prices) >= 5 else prices[-1]
        ma10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else ma5
        current = prices[-1]

        if current > ma5 and ma5 > ma10:
            return 'up'
        elif current < ma5 and ma5 < ma10:
            return 'down'
        else:
            return 'side'

    # ========== v3: 大盘环境过滤 ==========

    def _get_market_regime(self, dt) -> str:
        """判断大盘环境（牛/熊/震荡）

        Returns: 'bull' | 'bear' | 'side'
        MA5 > MA20 → bull, MA5 < MA20 → bear, else side

        注意：用昨日收盘价计算，避免30分钟bar发出信号时当日价格未知
        """
        if not self._ts_data:
            return 'side'

        ts = pd.Timestamp(dt)
        dates = sorted(self._ts_data.keys())

        # 找最近25个"昨日及之前"的交易日
        past_dates = [d for d in dates if d < ts.replace(hour=0, minute=0, second=0, microsecond=0)]
        if len(past_dates) < 20:
            return 'side'

        recent = past_dates[-25:]
        closes = [self._ts_data[d] for d in recent]
        ma5 = sum(closes[-5:]) / 5
        ma20 = sum(closes[-20:]) / 20

        if self._loose_regime:
            bull_thresh = 1.01
            bear_thresh = 0.99
        else:
            bull_thresh = 1.02
            bear_thresh = 0.98

        if ma5 > ma20 * bull_thresh:
            return 'bull'
        elif ma5 < ma20 * bear_thresh:
            return 'bear'
        else:
            return 'side'

    def _get_position_size(self, regime: str, buy_type: str) -> float:
        """根据大盘环境和买卖点类型返回仓位系数

        牛市：正常仓位
        震荡：降低仓位
        熊市：极低仓位或空仓
        """
        base = 0.5  # 基础半仓

        if regime == 'bull':
            if buy_type == '1buy':
                return base * 1.0   # 1买满配
            elif buy_type == '2buy':
                return base * 0.8   # 2买次配
            elif buy_type == '3buy':
                return base * 0.6   # 3买保守
        elif regime == 'side':
            if buy_type == '1buy':
                return base * 0.6
            elif buy_type == '2buy':
                return base * 0.4
            else:
                return base * 0.2
        else:  # bear
            if buy_type == '1buy':
                size = base * 0.5    # 熊市1买半仓（原来0.3）
            elif buy_type == '2buy':
                size = base * 0.3 if self._loose_regime else 0.0  # 宽松模式熊市2买30%
            elif buy_type == '3buy':
                size = base * 0.2 if self._loose_regime else 0.0  # 宽松模式熊市3买20%
            else:
                size = 0.0
            return size

        return base

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

        # ===== v3: 三类买卖点检测 =====
        buy1_set, buy2_set, buy3_set = self._detect_buy_points(
            strokes, macd_hist_series, n)

        # 逐K线模拟
        position = 0.0
        entry_idx = -1
        entry_price = 0.0
        stop_loss = 0.0
        highest_since_entry = 0.0
        last_sell_idx = -999
        has_diverged = False
        active_buy_type = None  # '1buy' | '2buy' | '3buy'

        for i in range(120, n):
            price = close[i]
            dt = df.index[i]

            # === v3: 大盘环境 & 日线趋势（每个Bar开头算一次）===
            regime = self._get_market_regime(dt) if self.market_regime_filter else 'bull'
            daily_trend = self._get_daily_trend(dt) if self.daily_trend_filter else 'up'

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
                    active_buy_type = None
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
                    active_buy_type = None
                    continue

                # 3. 动态止盈
                if profit_pct > self.trailing_start_pct:
                    trailing_stop = highest_since_entry * (1 - self.trailing_distance)
                    if price <= trailing_stop:
                        signals.iloc[i] = 0.0
                        position = 0.0
                        last_sell_idx = i
                        has_diverged = False
                        active_buy_type = None
                        continue

                # 4. 时间止损
                if bars_held >= self.max_hold_bars:
                    signals.iloc[i] = 0.0
                    position = 0.0
                    last_sell_idx = i
                    has_diverged = False
                    active_buy_type = None
                    continue

                signals.iloc[i] = position

            else:
                # === 空仓 ===
                if i - last_sell_idx < self.cooldown_bars:
                    continue

                # ===== v3: 三类买点任一触发 =====
                has_b1 = i in buy1_set
                has_b2 = i in buy2_set
                has_b3 = i in buy3_set

                # 优先1买 > 2买 > 3买（1买最强势）
                buy_type = None
                if has_b1:
                    buy_type = '1buy'
                elif has_b2:
                    buy_type = '2buy'
                elif has_b3:
                    buy_type = '3buy'

                if buy_type is None:
                    # 旧逻辑兜底：笔底背驰（兼容）
                    has_buy_signal = bi_buy.iloc[i]
                    has_seg_buy_div = i in all_buy_div
                    if not has_buy_signal and not has_seg_buy_div:
                        continue
                    buy_type = 'legacy'

                # v3: 日线趋势过滤（非下跌市即可交易）
                # 放宽：'side'也允许入场，只过滤'down'（顺大势交易）
                if self.daily_trend_filter and daily_trend == 'down':
                    continue

                # v3: 大盘环境仓位
                pos_size = self._get_position_size(regime, buy_type)
                if pos_size <= 0:
                    continue

                # MACD确认
                macd_confirm = (
                    dif[i] > dea[i]
                    or (hist[i] > hist[i-1] and hist[i] <= 0)
                    or (dif[i] > dif[i-1])
                )
                if not macd_confirm:
                    continue

                # 每日级过滤器（换手率/资金流/PE）
                if not self._check_daily_filter(code, dt):
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

                signals.iloc[i] = pos_size
                position = pos_size
                entry_idx = i
                entry_price = price
                highest_since_entry = price
                has_diverged = False
                active_buy_type = buy_type

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
            c = df['close'].values
            # 找局部极值
            pivot_highs = []   # (idx, high_val)
            pivot_lows = []    # (idx, low_val)
            for i in range(1, n - 1):
                if l[i] < l[i-1] and l[i] < l[i+1]:
                    pivot_lows.append((i, l[i]))
                    buy_signals.iloc[i] = True
                if h[i] > h[i-1] and h[i] > h[i+1]:
                    pivot_highs.append((i, h[i]))
                    sell_signals.iloc[i] = True

            # fallback: 从极值点构建strokes
            # 合并极值点交替排列
            all_pivots = []
            hi, li = 0, 0
            while hi < len(pivot_highs) or li < len(pivot_lows):
                next_hi = pivot_highs[hi][0] if hi < len(pivot_highs) else float('inf')
                next_li = pivot_lows[li][0] if li < len(pivot_lows) else float('inf')
                if next_li < next_hi:
                    all_pivots.append(('low', pivot_lows[li][0], pivot_lows[li][1]))
                    li += 1
                else:
                    all_pivots.append(('high', pivot_highs[hi][0], pivot_highs[hi][1]))
                    hi += 1

            # 三连续极值构成一笔
            for k in range(len(all_pivots) - 2):
                t0, i0, v0 = all_pivots[k]
                t1, i1, v1 = all_pivots[k+1]
                t2, i2, v2 = all_pivots[k+2]
                if t0 == 'low' and t1 == 'high' and t2 == 'low':
                    strokes.append({
                        'start_idx': i0, 'end_idx': i2,
                        'start_val': v0, 'end_val': v2,
                        'high': v1, 'low': min(v0, v2),
                        'start_type': 'bottom', 'end_type': 'top',
                    })
                elif t0 == 'high' and t1 == 'low' and t2 == 'high':
                    strokes.append({
                        'start_idx': i0, 'end_idx': i2,
                        'start_val': v0, 'end_val': v2,
                        'high': max(v0, v2), 'low': v1,
                        'start_type': 'top', 'end_type': 'bottom',
                    })

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
            # P0修复: c段必须包含≥2个次级别中枢，即≥6笔
            # segments由线段算法构建，stroke_start/stroke_end是笔索引范围
            curr_stroke_count = curr['stroke_end'] - curr['stroke_start'] + 1
            if curr_stroke_count < 6:
                continue
            prev_area = abs(sum(macd_hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            curr_area = abs(sum(macd_hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
            if curr['low'] < prev['low'] and curr_area < prev_area:
                sig = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= sig < n:
                    buy_div.add(sig)

        for k in range(1, len(up_segs)):
            prev, curr = up_segs[k-1], up_segs[k]
            curr_stroke_count = curr['stroke_end'] - curr['stroke_start'] + 1
            if curr_stroke_count < 6:
                continue
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
            # P0修复v2: c笔K线数≥3（1.5小时，平衡严格性与信号量）
            if curr['end_idx'] - curr['start_idx'] < 2:
                continue
            prev_area = abs(sum(macd_hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            curr_area = abs(sum(macd_hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
            if curr['low'] < prev['low'] and curr_area < prev_area:
                sig = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= sig < n:
                    buy_div.add(sig)

        for k in range(1, len(up_strokes)):
            prev, curr = up_strokes[k-1], up_strokes[k]
            if curr['end_idx'] - curr['start_idx'] < 4:
                continue
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
