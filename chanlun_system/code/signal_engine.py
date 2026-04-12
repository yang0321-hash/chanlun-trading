"""
缠论交易信号引擎 v9

v8→v9 优化清单:
1. [T1] 分阶段移动止损: 盈利3-6%回撤3%/6-15%回撤5%/>15%回撤7%
   - 让利润奔跑,避免小震洗出大趋势
   - 盈利3%即启动(比v8的6%更早保本)
2. [T1] 背驰信号加成0.18 + 三重弱势过滤
3. [P0] 阻断大亏:
   - max_stop_pct 25%→15% (防单笔-19%灾难)
   - 大亏(>3%)后30天冷却 (防反复踩坑)
   - 连续3笔亏损暂停2天 (防情绪化交易)
4. [P1] 中期退出优化:
   - 顶背驰门槛: 盈利3%→5% (避免小盈利被震出)
   - 2卖条件: 盈利>5%才允许 (避免2卖洗出中趋势)

继承v8核心:
- 自研确定性笔检测(消除CZSC前视)
- MACD背驰增量计算(底背驰+顶背驰)
- ATR动态止损 + 减仓50% + max_dd暂停
- 涨跌停过滤
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class SignalEngine:
    """缠论买卖点信号引擎 v9"""

    def __init__(self):
        # 风控参数
        self.risk_per_trade = 0.03       # 单笔风险预算(占组合权益)
        self.max_positions = 5           # 组合最大持仓数
        self.max_drawdown_pct = 0.15     # 最大回撤暂停线
        self.cooldown_bars = 1           # 亏损后冷却期(小亏)
        self.big_loss_cooldown = 30     # 大亏(>3%)后冷却期(防反复踩坑)
        self.big_loss_threshold = -0.03 # 大亏定义: 亏损>3%
        self.time_stop_bars = 60         # 时间止损
        self.min_hold_before_sell = 7    # 最短持仓后才允许2卖出局
        self.max_stop_pct = 0.15         # 最大止损距离(收紧防单笔灾难,曾出现-19%)

        # 仓位参数
        self.min_position = 0.10        # C级轻仓最低10%
        self.max_position = 0.30
        self.base_position = 0.15

        # 移动止损 [v9] 分阶段: 盈利越多,回撤容忍越大,让利润奔跑
        self.trailing_start = 0.03       # 盈利3%即启动移动止损(比v8的6%更早保本)
        self.trailing_tight = 0.03       # 盈利3-6%: 回撤3%平仓(保本区)
        self.trailing_medium = 0.05      # 盈利6-15%: 回撤5%平仓(正常区)
        self.trailing_wide = 0.07        # 盈利>15%: 回撤7%平仓(让大趋势跑完,比8%更保守)
        self.trailing_tier1 = 0.06       # 分界线1
        self.trailing_tier2 = 0.15       # 分界线2

        # 盈利加仓
        self.profit_add_threshold = 0.05
        self.profit_add_ratio = 0.50

        # 减仓参数 [v8新增]
        self.reduce_start = 0.04         # 盈利4%后开始考虑减仓
        self.reduce_position_pct = 0.50  # 减仓到原仓位的50%

        # ATR止损参数 [v8新增]
        self.atr_stop_mult = 2.0         # 止损=entry_price - mult*ATR
        self.atr_period = 14             # ATR计算周期

        # 笔参数(自研确定性检测)
        self.bi_min_gap = 4             # 笔的最小包含处理后K线数(含端点)
        self.bi_confirm_delay = 1       # 分型确认延迟(1根原始K线)

        # 组合级共享状态(由generate()初始化)
        self._active_positions = 0
        self._last_loss_codes = {}       # code -> (last_loss_bar_idx, loss_pct)
        self._portfolio_peak = 0.0       # 组合权益峰值(用于回撤计算)
        self._portfolio_equity = 0.0     # 组合权益当前值
        self._trading_halted = False     # 回撤暂停标志
        self._consecutive_losses = 0     # 连续亏损计数
        self._last_loss_bar = -999       # 上次亏损的bar索引
        self._loss_pause_until = -1      # 连续亏损暂停截止bar

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """组合层信号生成，初始化共享状态"""
        self._active_positions = 0
        self._last_loss_codes = {}
        self._portfolio_peak = 1_000_000  # 初始资金
        self._portfolio_equity = 1_000_000
        self._trading_halted = False
        self._consecutive_losses = 0
        self._last_loss_bar = -999
        self._loss_pause_until = -1

        signals = {}
        for code in sorted(data_map.keys()):
            df = data_map[code]
            signals[code] = self._generate_single(code, df)

        return signals

    def _generate_single(self, code: str, df: pd.DataFrame) -> pd.Series:
        """单只股票信号生成

        信号语义:
        - 0.0: 空仓/平仓
        - 正值(0.05~0.30): 目标仓位权重
        - 减仓时输出减仓后的权重(如原0.15减半为0.075)
        """
        n = len(df)
        signals = pd.Series(0.0, index=df.index)

        if n < 120:
            return signals

        close = df['close']
        high = df['high']
        low = df['low']

        # ===== 预计算指标 =====
        bi_buy, bi_sell = self._detect_bi_deterministic(df)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)

        # ATR [v8新增]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        weekly_not_down = self._compute_weekly_not_down(df)
        ma20 = close.rolling(20).mean()
        vol_ma20 = df['volume'].rolling(20).mean()

        # ===== 逐K线模拟 =====
        position = 0.0
        entry_idx = -1
        entry_price = 0.0
        stop_loss = 0.0
        highest = 0.0
        has_added = False
        original_position = 0.0
        trailing_activated = False
        reduced = False  # [v8] 是否已减仓

        # 增量MACD背驰: 底背驰(买) + 顶背驰(卖) [v8扩展]
        macd_buy_points: List[Dict] = []
        macd_sell_points: List[Dict] = []

        for i in range(120, n):
            price = close.iloc[i]

            # --- 增量收集分型点(用于背驰检测) ---
            has_buy_divergence = False
            has_sell_divergence = False

            if bi_buy.iloc[i]:
                curr_point = {
                    'idx': i,
                    'price': low.iloc[i],
                    'dif': dif.iloc[i],
                    'hist': macd_hist.iloc[i],
                }
                # 底背驰: 价格更低 + DIF或hist更高
                for k in range(len(macd_buy_points) - 1, -1, -1):
                    prev = macd_buy_points[k]
                    if curr_point['idx'] - prev['idx'] > 120:
                        break
                    if curr_point['price'] >= prev['price']:
                        continue
                    if curr_point['dif'] > prev['dif'] or curr_point['hist'] > prev['hist']:
                        has_buy_divergence = True
                        break
                macd_buy_points.append(curr_point)

            if bi_sell.iloc[i]:
                curr_sell = {
                    'idx': i,
                    'price': high.iloc[i],
                    'dif': dif.iloc[i],
                    'hist': macd_hist.iloc[i],
                }
                # 顶背驰: 价格更高 + DIF或hist更低 [v8新增]
                for k in range(len(macd_sell_points) - 1, -1, -1):
                    prev = macd_sell_points[k]
                    if curr_sell['idx'] - prev['idx'] > 120:
                        break
                    if curr_sell['price'] <= prev['price']:
                        continue
                    if curr_sell['dif'] < prev['dif'] or curr_sell['hist'] < prev['hist']:
                        has_sell_divergence = True
                        break
                macd_sell_points.append(curr_sell)

            # [v8] 组合回撤跟踪
            if position > 0:
                day_pnl = position * (price - close.iloc[i-1]) / close.iloc[i-1] if i > 0 and close.iloc[i-1] > 0 else 0
                self._portfolio_equity += day_pnl * self._portfolio_equity
                if self._portfolio_equity > self._portfolio_peak:
                    self._portfolio_peak = self._portfolio_equity
                drawdown = (self._portfolio_equity - self._portfolio_peak) / self._portfolio_peak
                self._trading_halted = drawdown < -self.max_drawdown_pct

            if position > 0:
                # --- 持仓中 ---
                if price > highest:
                    highest = price

                bars_held = i - entry_idx
                profit_pct = (price - entry_price) / entry_price if entry_price > 0 else 0

                # 0. 硬止损 [v10 P0] 亏损超过max_stop_pct立刻平仓
                # 使用止损价而非收盘价计算亏损，防止跳空越过止损线
                if profit_pct < -self.max_stop_pct:
                    # 模拟止损单：即使跳空，也按止损价成交
                    hard_stop_price = entry_price * (1 - self.max_stop_pct)
                    # 如果开盘已越过止损，用开盘价（更接近实际可成交价）
                    exit_price = max(hard_stop_price, low.iloc[i])
                    capped_profit_pct = (exit_price - entry_price) / entry_price
                    signals.iloc[i] = 0.0
                    self._last_loss_codes[code] = (i, capped_profit_pct)
                    self._update_loss_streak(capped_profit_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 1. 结构止损
                if price <= stop_loss:
                    # 同理：用止损价而非收盘价，防止跳空放大亏损
                    struct_exit = max(stop_loss, low.iloc[i])
                    struct_profit_pct = (struct_exit - entry_price) / entry_price
                    signals.iloc[i] = 0.0
                    if struct_profit_pct < 0:
                        self._last_loss_codes[code] = (i, struct_profit_pct)
                    self._update_loss_streak(struct_profit_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 2. 分阶段移动止损 [v9]
                # 盈利越多,回撤容忍越大,避免小震洗出大趋势
                if profit_pct > self.trailing_start:
                    trailing_activated = True
                if trailing_activated:
                    # 根据最高盈利区间选择回撤容忍度
                    max_profit = (highest - entry_price) / entry_price if entry_price > 0 else 0
                    if max_profit >= self.trailing_tier2:
                        # 盈利>15%: 宽止损8%,让大趋势跑完
                        trailing_dist = self.trailing_wide
                    elif max_profit >= self.trailing_tier1:
                        # 盈利6-15%: 中止损5%
                        trailing_dist = self.trailing_medium
                    else:
                        # 盈利3-6%: 紧止损3%,快速保本
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

                # 3. 顶背驰卖出 [v8新增]
                # 1卖信号: 顶背驰 + 盈利 > 5% [v9 P1.2: 从3%提高到5%]
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

                # 4. 2卖出局 [v9 P1.1: 盈利>5%才允许2卖出局,避免小盈利被2卖洗出]
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

                # 5. 时间止损
                if bars_held >= self.time_stop_bars:
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._active_positions = max(0, self._active_positions - 1)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False
                    continue

                # 6. 减仓机制 [v8新增]
                # 条件: 盈利>reduce_start + bi_sell + 未减仓过
                # 效果: 锁定一半利润，保留底仓等移动止损
                if (not reduced
                        and profit_pct > self.reduce_start
                        and bi_sell.iloc[i]
                        and bars_held >= 3):
                    reduced_pos = position * self.reduce_position_pct
                    reduced_pos = max(self.min_position * 0.5, reduced_pos)
                    signals.iloc[i] = reduced_pos
                    position = reduced_pos
                    reduced = True
                    continue

                # 7. 盈利加仓
                if (not has_added
                        and profit_pct > self.profit_add_threshold
                        and bi_buy.iloc[i]
                        and profit_pct < 0.25
                        and self._active_positions <= self.max_positions):
                    add_size = original_position * self.profit_add_ratio
                    new_position = min(position + add_size, self.max_position)
                    actual_add = new_position - position
                    if actual_add > 0.001:
                        entry_price = (entry_price * position + price * actual_add) / new_position
                        # [v10] 加仓后重算止损，确保止损与新均价匹配
                        stop_loss = self._compute_stop_loss(price, atr, i, low, df)
                        stop_loss = max(stop_loss, entry_price * (1 - self.max_stop_pct))
                        signals.iloc[i] = new_position
                        position = new_position
                        has_added = True
                        continue

                # 继续持仓
                signals.iloc[i] = position

            else:
                # --- 空仓中，寻找入场 ---

                # [v8] 组合回撤暂停
                if self._trading_halted:
                    continue

                # [v9 P0.3] 连续亏损暂停
                if i <= self._loss_pause_until:
                    continue

                if self._active_positions >= self.max_positions:
                    continue

                if code in self._last_loss_codes:
                    loss_idx, loss_pct = self._last_loss_codes[code]
                    # 大亏(>3%)后30天冷却, 小亏1天冷却
                    cooldown = self.big_loss_cooldown if loss_pct < self.big_loss_threshold else self.cooldown_bars
                    if i - loss_idx < cooldown:
                        continue

                if not weekly_not_down.iloc[i]:
                    continue

                if not bi_buy.iloc[i]:
                    continue

                # [v8] 涨跌停过滤: 涨停封板买不进，跳过
                if i > 0 and close.iloc[i-1] > 0:
                    pct_change = (price - close.iloc[i-1]) / close.iloc[i-1]
                    if pct_change >= 0.095:  # 涨停(主板10%/科创板20%统一用9.5%阈值)
                        continue

                # MACD因子 [v9保留v8加法模型]
                macd_factor = 0.0
                macd_confirm = (
                    dif.iloc[i] > dif.iloc[i-1]
                    or (macd_hist.iloc[i] > macd_hist.iloc[i-1]
                        and macd_hist.iloc[i] <= 0)
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
                if not pd.isna(vol_ma20.iloc[i]) and vol_ma20.iloc[i] > 0:
                    vol_ratio = df['volume'].iloc[i] / vol_ma20.iloc[i]
                    if vol_ratio >= 2.0:
                        vol_factor = 0.04
                    elif vol_ratio >= 1.5:
                        vol_factor = 0.02
                    elif vol_ratio < 0.5:
                        vol_factor = -0.02

                # 弱信号过滤: MACD弱+量弱+MA20弱势 → 跳过
                macd_weak = macd_hist.iloc[i] < macd_hist.iloc[i-1] and macd_hist.iloc[i] < 0
                vol_very_weak = (not pd.isna(vol_ma20.iloc[i]) and vol_ma20.iloc[i] > 0
                                 and df['volume'].iloc[i] / vol_ma20.iloc[i] < 0.5)
                ma_very_weak = (not pd.isna(ma20.iloc[i]) and price < ma20.iloc[i] * 0.93)
                if macd_weak and vol_very_weak and ma_very_weak and not has_buy_divergence:
                    continue  # 三重弱势无背驰,跳过

                # 止损位 [v8] ATR动态止损
                stop = self._compute_stop_loss(price, atr, i, low, df)
                stop_distance = price - stop

                if stop_distance <= 0:
                    continue

                stop_pct = stop_distance / price
                if stop_pct > self.max_stop_pct:
                    stop = price * (1 - self.max_stop_pct)
                    stop_distance = price - stop
                    stop_pct = self.max_stop_pct

                # 仓位计算 [v9] 加法模型(v8逻辑) + 背驰信号加成
                risk_based_pct = self.risk_per_trade / stop_pct
                risk_based_pct = min(risk_based_pct, self.max_position)
                base_pos = min(self.base_position, risk_based_pct)

                # 加法: base_pos + macd_factor + vol_factor
                final_pos = base_pos + macd_factor + vol_factor
                # 背驰信号额外加成: 从0.15提升到0.18(不超0.20)
                if has_buy_divergence:
                    final_pos = max(final_pos, 0.18)
                final_pos = max(self.min_position, min(final_pos, self.max_position))

                # 硬性单笔风险上限
                actual_risk = final_pos * stop_pct
                if actual_risk > self.risk_per_trade:
                    final_pos = self.risk_per_trade / stop_pct
                    final_pos = max(self.min_position, min(final_pos, self.max_position))

                # MA20弱势过滤(非背驰信号降为最小仓位)
                if not pd.isna(ma20.iloc[i]) and price < ma20.iloc[i] * 0.93:
                    if not macd_confirm and not has_buy_divergence:
                        final_pos = self.min_position

                # 输出买入信号
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

    def _compute_stop_loss(self, price: float, atr: pd.Series, i: int,
                           low: pd.Series, df: pd.DataFrame) -> float:
        """计算止损位 [v8] ATR动态止损

        策略: 取以下两者中更高的(更保守):
        1. ATR止损: entry_price - mult * ATR
        2. 结构止损: 30天最低价

        ATR止损优点: 波动大时止损宽，波动小时止损紧
        结构止损优点: 尊重市场结构，不易被震出
        """
        # ATR止损
        atr_stop = 0.0
        if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0:
            atr_stop = price - self.atr_stop_mult * atr.iloc[i]

        # 结构止损(30天最低价)
        lookback = min(30, i - 1)
        struct_stop = df['low'].iloc[i-lookback:i].min()

        # 取更高者(更保守)
        stop = max(atr_stop, struct_stop)

        return stop

    def _detect_bi_deterministic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """确定性笔检测(替代CZSC，完全无前视偏差)

        算法:
        1. 包含关系处理: 合并高低相含的相邻K线(确定性，不随数据量变化)
        2. 分型检测: 处理后K线序列的顶底分型(确定性)
        3. 顶底交替: 同类型取极值，不同类型检查间距>=min_gap
        4. 确认延迟: 分型需等待bi_confirm_delay根K线后才触发信号

        与CZSC对比:
        - CZSC: 全量构建，所有笔端点回溯修正(49个端点全部随数据量变化)
        - 自研: 逐根处理，不回溯修正(已验证确定性)

        信号密度: 自研28买27卖 vs CZSC 25买25卖(接近)
        """
        n = len(df)
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)

        if n < 5:
            return buy_signals, sell_signals

        high_arr = df['high'].values.astype(float)
        low_arr = df['low'].values.astype(float)

        # Step 1: 包含关系处理
        merged: List[Dict] = [{'high': high_arr[0], 'low': low_arr[0], 'idx': 0}]
        direction = 0

        for i in range(1, n):
            prev = merged[-1]
            if len(merged) >= 2:
                prev2 = merged[-2]
                if prev['high'] > prev2['high'] and prev['low'] > prev2['low']:
                    direction = 1
                elif prev['high'] < prev2['high'] and prev['low'] < prev2['low']:
                    direction = -1

            prev_contains_curr = prev['high'] >= high_arr[i] and prev['low'] <= low_arr[i]
            curr_contains_prev = high_arr[i] >= prev['high'] and low_arr[i] <= prev['low']

            if prev_contains_curr or curr_contains_prev:
                if direction == 1:
                    prev['high'] = max(prev['high'], high_arr[i])
                    prev['low'] = max(prev['low'], low_arr[i])
                elif direction == -1:
                    prev['high'] = min(prev['high'], high_arr[i])
                    prev['low'] = min(prev['low'], low_arr[i])
                else:
                    if curr_contains_prev:
                        prev['high'] = high_arr[i]
                        prev['low'] = low_arr[i]
            else:
                merged.append({'high': high_arr[i], 'low': low_arr[i], 'idx': i})

        # Step 2: 分型检测(elif: 一个j只能是一种分型)
        fractals: List[Dict] = []
        for j in range(1, len(merged) - 1):
            if merged[j]['high'] > merged[j-1]['high'] and merged[j]['high'] > merged[j+1]['high']:
                fractals.append({'type': 'top', 'midx': j, 'idx': merged[j]['idx'], 'val': merged[j]['high']})
            elif merged[j]['low'] < merged[j-1]['low'] and merged[j]['low'] < merged[j+1]['low']:
                fractals.append({'type': 'bottom', 'midx': j, 'idx': merged[j]['idx'], 'val': merged[j]['low']})

        if not fractals:
            return buy_signals, sell_signals

        # Step 3: 顶底交替 + 间距检查
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

        # Step 4: 生成信号(带确认延迟)
        for j in range(1, len(filtered)):
            prev = filtered[j-1]
            curr = filtered[j]

            signal_idx = curr['idx'] + self.bi_confirm_delay
            if signal_idx >= n:
                continue

            if prev['type'] == 'top' and curr['type'] == 'bottom':
                buy_signals.iloc[signal_idx] = True
            elif prev['type'] == 'bottom' and curr['type'] == 'top':
                sell_signals.iloc[signal_idx] = True

        return buy_signals, sell_signals

    def _compute_weekly_not_down(self, df):
        """周线非下跌过滤"""
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

    def _update_loss_streak(self, profit_pct: float, bar_idx: int):
        """更新组合级连续亏损追踪 [v9 P0.3]"""
        if profit_pct < 0:
            # 与上次亏损同一天或相邻天 → 连续亏损+1
            if bar_idx - self._last_loss_bar <= 5:  # 5天内算连续
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 1
            self._last_loss_bar = bar_idx
            # 连续3笔亏损 → 暂停2天
            if self._consecutive_losses >= 3:
                self._loss_pause_until = bar_idx + 2
        else:
            # 盈利退出 → 重置
            self._consecutive_losses = 0
