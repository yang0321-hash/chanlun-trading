"""
缠论交易信号引擎 v15-final (中枢延伸/扩张/升级, 2026-04-23)

v14.1→v15-final 新增(按v2.0附录A规范):
1. [P0] 中枢延伸检测: N=4~8笔在中枢区间内震荡, ZG/ZD不变
   - extended=True (N=6~8, 走势积蓄期, 2买降权)
2. [P0] 中枢扩张检测(条件A/B):
   - 条件A(上扩): stroke.high > ZG_orig 且 next.low ∈ [ZD_orig, ZG_orig]
   - 条件B(下扩): stroke.low < ZD_orig 且 next.high ∈ [ZD_orig, ZG_orig]
   - expanded=True, expanded_mode='A'/'B', zg_expanded/zd_expanded
3. [P0] 中枢升级检测: N≥9触发升级
   - upgraded=True (需上级别周期验证方能确认)
4. 信号层影响:
   - 扩展中枢(N=6~8): 2买/3买降权(方向不明)
   - 扩张中枢: 3买目标位用zg_expanded
   - 升级中枢: 3买失效需重新确认

继承v14核心:
v13→v14 新增:
1. [P0] 线段检测: 3笔重叠+极值突破, 笔级别背驰升级为线段级别
2. [P0] 线段级别MACD面积背驰: 两段同向线段面积缩小=大级别背驰
3. [P0] 2卖检测: 1卖后下跌→反弹不过前高=确认顶部离场
4. 背驰合并: 笔级别+线段级别取并集, 不遗漏

继承v13核心:
1. [P0] 背驰检测: 从DIF/hist时点值改为MACD面积(缠论标准)
2. [P0] 新增2买检测: 1买后反弹再回调不破前低(最实用买点)
3. [P1] 修复3买or True bug: 弱3买不再无条件覆盖标准3买
4. [P1] 笔检测返回strokes: 为背驰面积计算和中枢检测提供结构数据

继承v12核心:
1. [T0] 动态月度轮换池: 从200只CSI500候选中每月评分选top50
   - 评分因子: ATR波动率(0.35) + 动量(0.30) + 量能活跃度(0.20) + 趋势强度(0.15)
   - 不在池中的股票不允许新开仓, 已持仓可继续管理(止损/止盈正常执行)
   - 目标: 过滤死票, 提高信号密度和资金效率

继承v11.1核心:
- 3买独立入场 + 中高端tier+1%空间
- 分阶段移动止损 + 背驰信号加成
- 自研确定性笔检测(消除CZSC前视)

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
        self.max_stop_pct = 0.10         # 最大止损距离(回测验证: 10%最优Sharpe 1.33)

        # 仓位参数
        self.min_position = 0.10        # C级轻仓最低10%
        self.max_position = 0.20        # 单票上限(回测验证: 20%回撤11.8%/Sharpe 1.98)
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
        self.bi_min_gap = 3             # 笔的最小包含处理后K线数(含端点)
        self.bi_confirm_delay = 1       # 分型确认延迟(1根原始K线)

        # ===== v10 新增: 动量仓位因子 =====
        self.momentum_factor_enabled = True
        self.momentum_period = 60        # 动量计算周期

        # ===== v10 新增: 波动率自适应 =====
        self.vol_regime_enabled = True
        self.vol_lookback = 120          # ATR百分位计算窗口

        # 组合级共享状态(由generate()初始化)
        self._active_positions = 0
        self._active_industries = {}  # {industry: count} [v12.2]
        self._last_loss_codes = {}       # code -> (last_loss_bar_idx, loss_pct)
        self._portfolio_peak = 0.0       # 组合权益峰值(用于回撤计算)
        self._portfolio_equity = 0.0     # 组合权益当前值
        self._trading_halted = False     # 回撤暂停标志
        self._consecutive_losses = 0     # 连续亏损计数
        self._last_loss_bar = -999
        self._loss_pause_until = -1

        # ===== v11: 3买信号 =====
        self.third_buy_enabled = True    # 启用3买检测
        self.third_buy_boost = 0.18      # 3买仓位保底(与背驰加成相同)
        # 3买专项移动止损: 只在盈利>8%时给额外1%空间
        # 低端(3-8%)和普通完全一致 — 3买的小利润交易不需要特殊对待
        # 中高端多1% — 3买趋势确认后让利润多跑一点
        self.third_buy_tier1 = 0.03       # 盈利3%启动(和普通一致)
        self.third_buy_tight = 0.03       # 3-8%: 回撤3%(和普通完全一致)
        self.third_buy_medium = 0.06     # 8-15%: 回撤6%(vs普通5%, +1%)
        self.third_buy_wide = 0.08       # >15%: 回撤8%(vs普通7%, +1%)

        # ===== v12: 动态月度轮换池 =====
        self.dynamic_pool_enabled = True  # 启用动态轮换
        self.pool_size = 50              # 每月选出的股票数
        self.pool_score_lookback = 60    # 评分回看天数
        # 评分权重: ATR波动率 + 动量 + 量能活跃度 + 趋势强度
        self.pool_w_atr = 0.35
        self.pool_w_momentum = 0.30
        self.pool_w_volume = 0.20
        self.pool_w_trend = 0.15

        # ===== v12.2: 板块去重 =====
        self.sector_dedup_enabled = True
        self.sector_max_holdings = 2  # 同行业最多持仓数
        self._load_industry_map()

    def _load_industry_map(self):
        """加载股票→行业映射"""
        self._industry_map = {}
        try:
            import os, json
            map_path = os.path.join(os.path.dirname(__file__), '..', 'industry_map.json')
            if os.path.exists(map_path):
                with open(map_path) as f:
                    self._industry_map = json.load(f)
        except Exception:
            pass

    def generate(self, data_map: Dict[str, pd.DataFrame], live_mode: bool = False) -> Dict[str, pd.Series]:
        """组合层信号生成，初始化共享状态

        live_mode=True: 只算当月动态池top50(秒级), 用于实盘扫描
        live_mode=False: 算所有月份进入过池的股票(分钟级), 用于回测
        """
        # 标准化: 确保datetime列设为index
        normalized_map = {}
        for code, df in data_map.items():
            if 'datetime' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('datetime')
            elif 'trade_date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('trade_date')
            normalized_map[code] = df
        data_map = normalized_map

        self._active_positions = 0
        self._active_industries = {}  # {industry: count} [v12.2]
        self._last_loss_codes = {}
        self._portfolio_peak = 1_000_000  # 初始资金
        self._portfolio_equity = 1_000_000
        self._trading_halted = False
        self._consecutive_losses = 0
        self._last_loss_bar = -999
        self._loss_pause_until = -1
        self.pivot_buy_pending = {}  # {code: {'date': str, 'price': float}} [v14.1] 日线中枢背驰1买待确认

        # ===== v10: 预计算月度动量排名(用于仓位调节) =====
        monthly_momentum_rank = {}
        if self.momentum_factor_enabled:
            monthly_momentum_rank = self._get_momentum_ranking(data_map)

        # ===== v12: 预计算动态月度轮换池 =====
        dynamic_pool = {}
        if self.dynamic_pool_enabled and len(data_map) > self.pool_size:
            dynamic_pool = self._get_dynamic_pool(data_map)

        # live模式: 只保留当月池
        if live_mode and dynamic_pool:
            latest_month = sorted(dynamic_pool.keys())[-1]
            dynamic_pool = {latest_month: dynamic_pool[latest_month]}
            print(f"  [live模式] 仅计算{latest_month}池({len(dynamic_pool[latest_month])}只)", flush=True)

        signals = {}
        for code in sorted(data_map.keys()):
            df = data_map[code]
            # 优化: 不在动态池中的股票直接返回全0，跳过昂贵的缠论计算
            if dynamic_pool:
                in_pool = False
                for month_set in dynamic_pool.values():
                    if code in month_set:
                        in_pool = True
                        break
                if not in_pool:
                    signals[code] = pd.Series(0.0, index=df.index)
                    continue
            signals[code] = self._generate_single(code, df, monthly_momentum_rank, dynamic_pool)

        return signals

    def _get_momentum_ranking(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """动量排名: 缓存版, 每天只算一次"""
        import os, pickle
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'live_signals')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'momentum_rank_cache.pkl')
        sample_code = next(iter(data_map))
        last_date = str(data_map[sample_code].index[-1].date())

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('data_date') == last_date:
                    print(f"  [缓存] 动量排名命中 ({last_date})", flush=True)
                    return cached['rank']
            except Exception:
                pass

        print(f"  [计算] 动量排名全量 ({len(data_map)}只)...", flush=True)
        rank = self._build_momentum_ranking(data_map)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'rank': rank, 'data_date': last_date}, f)
            print(f"  [缓存] 动量排名已缓存", flush=True)
        except Exception:
            pass

        return rank

    def _build_momentum_ranking(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """构建月度动量排名 [v10] (并发优化版)

        返回每月每只股票的动量分位数(0~1), 1=最强

        Returns: {month_key: {code: momentum_rank}}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _momentum_one(code, df):
            if len(df) < 120:
                return {}
            close = df['close']
            ret = close.pct_change(self.momentum_period)
            try:
                monthly_first = df.resample('MS').first()
            except Exception:
                return {}
            scores = {}
            for dt in monthly_first.index:
                if dt not in ret.index:
                    valid = ret.index[ret.index <= dt]
                    if len(valid) == 0:
                        continue
                    dt = valid[-1]
                if pd.isna(ret.loc[dt]):
                    continue
                month_key = dt.strftime('%Y-%m')
                scores[month_key] = ret.loc[dt]
            return scores

        monthly_scores = {}
        codes = list(data_map.keys())
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_momentum_one, code, data_map[code]): code for code in codes}
            for f in as_completed(futures):
                scores = f.result()
                for month, score in scores.items():
                    if month not in monthly_scores:
                        monthly_scores[month] = {}
                    monthly_scores[month][futures[f]] = score

        monthly_rank = {}
        for month, scores in monthly_scores.items():
            if len(scores) < 5:
                continue
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            n = len(sorted_items)
            ranks = {}
            for rank_idx, (code, _) in enumerate(sorted_items):
                ranks[code] = (rank_idx + 1) / n  # 1/n ~ 1.0
            monthly_rank[month] = ranks
        return monthly_rank

    def _score_single_for_pool(self, code: str, df: pd.DataFrame) -> Dict[str, float]:
        """计算单只股票的月度评分(用于动态池)"""
        lb = self.pool_score_lookback
        if len(df) < lb + 30:
            return {}

        close = df['close']
        high = df['high']
        low = df['low']
        vol = df['volume']

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # MA60
        ma60 = close.rolling(60).mean()
        vol_ma60 = vol.rolling(60).mean()

        # 动量
        momentum = close.pct_change(lb)

        # 遍历每个月末
        try:
            monthly_dates = df.resample('MS').first().index
        except Exception:
            return {}

        scores = {}
        for dt in monthly_dates:
            valid = df.index[df.index >= dt]
            if len(valid) == 0:
                continue
            bar_idx = df.index.get_loc(valid[0])
            if bar_idx < lb + 30:
                continue

            prev_idx = bar_idx - 1

            # 1. ATR波动率百分位
            atr_window = atr.iloc[max(0, prev_idx-120):prev_idx+1]
            atr_val = atr.iloc[prev_idx]
            if pd.isna(atr_val) or atr_val <= 0 or len(atr_window) < 30:
                atr_score = 0.5
            else:
                atr_score = (atr_window < atr_val).sum() / len(atr_window)

            # 2. 动量
            mom_val = momentum.iloc[prev_idx]
            if pd.isna(mom_val):
                momentum_score = 0.5
            else:
                momentum_score = np.clip((mom_val + 0.3) / 0.6, 0, 1)

            # 3. 量能活跃度
            vol_ratio = vol.iloc[prev_idx] / vol_ma60.iloc[prev_idx] if not pd.isna(vol_ma60.iloc[prev_idx]) and vol_ma60.iloc[prev_idx] > 0 else 1.0
            volume_score = np.clip(vol_ratio / 2.0, 0, 1)

            # 4. 趋势强度
            ma60_val = ma60.iloc[prev_idx]
            price_val = close.iloc[prev_idx]
            if pd.isna(ma60_val) or ma60_val <= 0:
                trend_score = 0.5
            else:
                trend_score = np.clip((price_val / ma60_val - 0.9) / 0.2, 0, 1)

            total = (self.pool_w_atr * atr_score
                     + self.pool_w_momentum * momentum_score
                     + self.pool_w_volume * volume_score
                     + self.pool_w_trend * trend_score)

            month_key = dt.strftime('%Y-%m')
            scores[month_key] = total

        return scores

    def _get_dynamic_pool(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, set]:
        """动态池: 优先读缓存(每月初刷新), 避免每天重算5287只"""
        import os, pickle
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'live_signals')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'dynamic_pool_cache.pkl')
        # 用数据的最后日期作为key(避免不同数据源日期不一致)
        sample_code = next(iter(data_map))
        last_date = str(data_map[sample_code].index[-1].date())
        current_month = last_date[:7]  # YYYY-MM

        # 尝试读缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('data_month') == current_month and cached.get('data_date') == last_date:
                    print(f"  [缓存] 动态池命中 ({last_date})", flush=True)
                    return cached['pool']
            except Exception:
                pass

        # 缓存失效或不存在, 重新计算
        print(f"  [计算] 动态池全量计算 ({len(data_map)}只)...", flush=True)
        pool = self._build_dynamic_pool(data_map)

        # 写缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'pool': pool, 'data_month': current_month, 'data_date': last_date}, f)
            print(f"  [缓存] 动态池已缓存", flush=True)
        except Exception:
            pass

        return pool

    def _build_dynamic_pool(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, set]:
        """构建动态月度轮换池 [v12] (并发优化版)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1: 并发计算每只股票评分
        monthly_scores = {}
        codes = list(data_map.keys())

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self._score_single_for_pool, code, data_map[code]): code for code in codes}
            for f in as_completed(futures):
                scores = f.result()
                for month, score in scores.items():
                    if month not in monthly_scores:
                        monthly_scores[month] = {}
                    monthly_scores[month][futures[f]] = score

        # Step 2: 每月选top pool_size
        dynamic_pool = {}
        for month, scores in monthly_scores.items():
            if len(scores) <= self.pool_size:
                dynamic_pool[month] = set(scores.keys())
            else:
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selected = set(code for code, _ in sorted_items[:self.pool_size])
                dynamic_pool[month] = selected

        return dynamic_pool

    def _generate_single(self, code: str, df: pd.DataFrame,
                         monthly_momentum_rank: Dict[str, Dict[str, float]] = None,
                         dynamic_pool: Dict[str, set] = None) -> pd.Series:
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
        bi_buy, bi_sell, filtered_fractals, strokes = self._detect_bi_deterministic(df)

        # MACD (v13: 提前计算, 面积背驰需要)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)

        # ===== v11: 3买上下文检测 =====
        third_buy = pd.Series(False, index=df.index)
        if self.third_buy_enabled:
            third_buy = self._detect_3buy_context(filtered_fractals, df)

        # ===== v13→v15: 面积背驰(区分趋势/盘整) + 2买预计算 =====
        buy_div_set, sell_div_set = self._compute_area_divergence(strokes, macd_hist, n)
        buy_2buy_set = self._detect_2buy(strokes, buy_div_set, n)

        # ===== v14: 线段检测 + 线段背驰 + 2卖 =====
        segments = self._detect_segments(strokes)
        seg_buy_div, seg_sell_div = self._compute_segment_divergence(segments, strokes, macd_hist, n)
        sell_2sell_set = self._detect_2sell(strokes, sell_div_set, n)

        # 合并: 笔级别 + 线段级别背驰取并集
        buy_divergence_set = buy_div_set | seg_buy_div
        sell_divergence_set = sell_div_set | seg_sell_div

        # ===== v14.1: 中枢识别 + 趋势背驰 =====
        pivots = self._detect_pivots(strokes)
        pivot_buy_div, pivot_sell_div = self._detect_pivot_divergence(
            pivots, strokes, macd_hist, n
        )

        # ATR [v8新增]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        # ===== v10: ATR滚动百分位(无前视偏差) =====
        atr_pct = pd.Series(0.5, index=df.index)
        if self.vol_regime_enabled:
            atr_pct = atr.rolling(self.vol_lookback, min_periods=60).rank(pct=True)

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
        is_3buy_trade = False  # [v11.1] 当前持仓是否为3买入场

        for i in range(120, n):
            price = close.iloc[i]

            # 增量MACD背驰: 已由v13面积法预计算, 此处仅查询
            has_buy_divergence = i in buy_divergence_set
            has_sell_divergence = i in sell_divergence_set
            is_2buy = i in buy_2buy_set
            has_pivot_buy_div = i in pivot_buy_div
            has_pivot_sell_div = i in pivot_sell_div

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
                    self._close_position(code)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False; is_3buy_trade = False
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
                    self._close_position(code)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False; is_3buy_trade = False
                    continue

                # 2. 分阶段移动止损 [v9] + [v11.1 3买更宽tier]
                # 盈利越多,回撤容忍越大,避免小震洗出大趋势
                # 3买是趋势确认信号, 给更宽的止损让利润多跑
                if is_3buy_trade:
                    active_start = self.third_buy_tier1
                else:
                    active_start = self.trailing_start
                if profit_pct > active_start:
                    trailing_activated = True
                if trailing_activated:
                    max_profit = (highest - entry_price) / entry_price if entry_price > 0 else 0
                    if is_3buy_trade:
                        # 3买专项tier: 3-8%→4%, 8-15%→6%, >15%→8%
                        if max_profit >= 0.15:
                            trailing_dist = self.third_buy_wide
                        elif max_profit >= 0.08:
                            trailing_dist = self.third_buy_medium
                        else:
                            trailing_dist = self.third_buy_tight
                    else:
                        # 普通tier: 3-6%→3%, 6-15%→5%, >15%→7%
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
                        self._close_position(code)
                        position = 0.0; original_position = 0.0; has_added = False
                        trailing_activated = False; reduced = False; is_3buy_trade = False
                        continue

                # 3. 顶背驰卖出 [v8新增]
                # 1卖信号: 顶背驰 + 盈利 > 5% [v9 P1.2: 从3%提高到5%]
                if (bars_held >= self.min_hold_before_sell
                        and (has_sell_divergence or has_pivot_sell_div)
                        and profit_pct > 0.05):
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._close_position(code)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False; is_3buy_trade = False
                    continue

                # 3.5. 2卖出局 [v14: 缠论标准2卖, 反弹不过前高]
                # 1卖后下跌→反弹不过1卖高点 = 确认顶部, 立即离场
                is_2sell = i in sell_2sell_set
                if (bars_held >= self.min_hold_before_sell
                        and is_2sell
                        and profit_pct > 0.03):
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._close_position(code)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False; is_3buy_trade = False
                    continue

                # 4. bi_sell+盈利卖出(旧2卖替代, 保留作为兜底)
                if (bars_held >= self.min_hold_before_sell
                        and bi_sell.iloc[i]
                        and profit_pct > 0.05):
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._close_position(code)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False; is_3buy_trade = False
                    continue

                # 5. 时间止损
                if bars_held >= self.time_stop_bars:
                    signals.iloc[i] = 0.0
                    if profit_pct < 0:
                        self._last_loss_codes[code] = (i, profit_pct)
                    self._update_loss_streak(profit_pct, i)
                    self._close_position(code)
                    position = 0.0; original_position = 0.0; has_added = False
                    trailing_activated = False; reduced = False; is_3buy_trade = False
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

                # ===== v12.2: 板块去重 =====
                if self.sector_dedup_enabled:
                    industry = self._industry_map.get(code, '')
                    if industry and self._active_industries.get(industry, 0) >= self.sector_max_holdings:
                        continue

                if code in self._last_loss_codes:
                    loss_idx, loss_pct = self._last_loss_codes[code]
                    # 大亏(>3%)后30天冷却, 小亏1天冷却
                    cooldown = self.big_loss_cooldown if loss_pct < self.big_loss_threshold else self.cooldown_bars
                    if i - loss_idx < cooldown:
                        continue

                if not weekly_not_down.iloc[i]:
                    continue

                if not bi_buy.iloc[i] and not third_buy.iloc[i] and not is_2buy and not has_pivot_buy_div:
                    continue

                # ===== v12: 动态池过滤 =====
                # 不在当月池中的股票不允许新开仓(已持仓的止损/止盈不受影响)
                if dynamic_pool:
                    month_key = df.index[i].strftime('%Y-%m')
                    if month_key in dynamic_pool and code not in dynamic_pool[month_key]:
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
                # [v14.1] 中枢背驰信号同等加成
                if has_pivot_buy_div:
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
                # [v11] 3买有中枢结构确认, 跳过弱信号过滤
                is_3buy = third_buy.iloc[i]
                macd_weak = macd_hist.iloc[i] < macd_hist.iloc[i-1] and macd_hist.iloc[i] < 0
                vol_very_weak = (not pd.isna(vol_ma20.iloc[i]) and vol_ma20.iloc[i] > 0
                                 and df['volume'].iloc[i] / vol_ma20.iloc[i] < 0.5)
                ma_very_weak = (not pd.isna(ma20.iloc[i]) and price < ma20.iloc[i] * 0.93)
                if macd_weak and vol_very_weak and ma_very_weak and not has_buy_divergence and not is_3buy and not is_2buy and not has_pivot_buy_div:
                    continue  # 三重弱势无背驰无3买无2买无中枢背驰,跳过

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
                # [v13] 2买信号同等加成: 回踩不破前低确认底部
                if is_2buy:
                    final_pos = max(final_pos, 0.18)
                # [v14.1] 中枢背驰1买信号同等加成
                if has_pivot_buy_div:
                    final_pos = max(final_pos, 0.18)
                    # 记录到pending，供30分钟区间套使用
                    self.pivot_buy_pending[code] = {
                        'date': str(df.index[i].date()),
                        'price': float(price),
                        'stop_loss': float(stop),
                        '2buy_confirmed': False,
                    }
                # [v14.1] 2买确认区间套: 日线1买+2买组合后才允许30分入场
                # 必须在1买后30个交易日内确认，否则视为不同波段
                if is_2buy and code in self.pivot_buy_pending:
                    pending_entry = self.pivot_buy_pending[code]
                    if not pending_entry.get('2buy_confirmed'):
                        buy1_date = pd.Timestamp(pending_entry['date'])
                        buy2_date = df.index[i]
                        days_gap = len(df[buy1_date:buy2_date])  # 交易天数
                        if 0 < days_gap <= 30:
                            pending_entry['2buy_confirmed'] = True
                            pending_entry['2buy_date'] = str(buy2_date.date())
                            pending_entry['2buy_price'] = float(price)
                            pending_entry['stop_loss'] = float(stop)
                        else:
                            # 超时清除pending(不同波段，1买失效)
                            del self.pivot_buy_pending[code]
                # [v11] 3买信号同等加成: 中枢突破回踩确认
                if is_3buy:
                    final_pos = max(final_pos, self.third_buy_boost)
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

                # ===== v10: 动量仓位因子(加法,与MACD/量能一致) =====
                # 动量排名0~1, 1=最强。高动量加分，低动量微扣
                if (self.momentum_factor_enabled
                        and monthly_momentum_rank
                        and not has_buy_divergence):
                    month_key = df.index[i].strftime('%Y-%m')
                    if month_key in monthly_momentum_rank:
                        mom_rank = monthly_momentum_rank[month_key].get(code, 0.5)
                        # rank>0.7: +0.01, rank<0.3: -0.01, 中间: 0
                        if mom_rank > 0.7:
                            final_pos += 0.01
                        elif mom_rank < 0.3:
                            final_pos -= 0.01

                # ===== v10: 波动率自适应 =====
                # 高波动(>70%): 仓位×0.85, 止损×1.15
                # 低波动(<30%): 仓位×1.05, 止损×0.95
                if self.vol_regime_enabled and not pd.isna(atr_pct.iloc[i]):
                    pct_val = atr_pct.iloc[i]
                    if pct_val > 0.70:
                        final_pos *= 0.85
                        stop_pct *= 1.15
                    elif pct_val < 0.30:
                        final_pos *= 1.05
                        stop_pct *= 0.95
                    # 重新检查约束
                    final_pos = max(self.min_position * 0.5, min(final_pos, self.max_position))
                    stop_pct = min(stop_pct, self.max_stop_pct)

                # 输出买入信号
                signals.iloc[i] = final_pos
                position = final_pos
                original_position = final_pos
                has_added = False
                reduced = False
                is_3buy_trade = is_3buy  # [v11.1] 标记3买交易
                entry_idx = i
                entry_price = price
                stop_loss = stop
                highest = price
                self._active_positions += 1
                # v12.2: 更新行业计数
                industry = self._industry_map.get(code, '')
                if industry:
                    self._active_industries[industry] = self._active_industries.get(industry, 0) + 1

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

    def _detect_bi_deterministic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[Dict], List[Dict]]:
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
            return buy_signals, sell_signals, [], []

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
            return buy_signals, sell_signals, [], []

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
                # 不同类型+间距不足: 跳过，保留当前filtered[-1]
                # 注意: 不能用val比较(top=high, bottom=low, 跨类型比较无意义)

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

        # Step 5: 构建笔列表 [v13] 用于背驰面积计算和中枢检测
        # [P1-4] 笔的high/low用原始K线极值(而非两端分型价格), 更精确计算中枢ZG/ZD
        strokes = []
        for j in range(len(filtered) - 1):
            f1, f2 = filtered[j], filtered[j + 1]
            start_idx = f1['idx']
            end_idx = f2['idx']
            # 笔内原始K线极值
            stroke_high = high_arr[start_idx:end_idx + 1].max() if end_idx >= start_idx else max(f1['val'], f2['val'])
            stroke_low = low_arr[start_idx:end_idx + 1].min() if end_idx >= start_idx else min(f1['val'], f2['val'])
            strokes.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_type': f1['type'],
                'end_type': f2['type'],
                'start_val': f1['val'],
                'end_val': f2['val'],
                'high': stroke_high,
                'low': stroke_low,
            })

        return buy_signals, sell_signals, filtered, strokes

    def _compute_area_divergence(self, strokes: list, macd_hist, n: int):
        """v13: MACD面积背驰检测(缠论标准)

        Returns:
            buy_divergence: set of signal_idx — 底背驰(1买)位置
            sell_divergence: set of signal_idx — 顶背驰(1卖)位置
        """
        buy_divergence = set()
        sell_divergence = set()

        # 分类下跌笔和上涨笔
        down_strokes = [s for s in strokes
                        if s['start_type'] == 'top' and s['end_type'] == 'bottom']
        up_strokes = [s for s in strokes
                      if s['start_type'] == 'bottom' and s['end_type'] == 'top']

        # 底背驰: 连续下跌笔, 价格新低但MACD面积缩小
        # 缠论标准: 面积取下跌笔中macd_hist<0(绿柱)的绝对值之和
        for k in range(1, len(down_strokes)):
            prev = down_strokes[k - 1]
            curr = down_strokes[k]

            prev_hist = macd_hist.iloc[prev['start_idx']:prev['end_idx'] + 1].values
            curr_hist = macd_hist.iloc[curr['start_idx']:curr['end_idx'] + 1].values
            prev_area = sum(abs(x) for x in prev_hist if x < 0)
            curr_area = sum(abs(x) for x in curr_hist if x < 0)

            if curr['end_val'] < prev['end_val'] and curr_area < prev_area:
                signal_idx = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_idx < n:
                    buy_divergence.add(signal_idx)

        # 顶背驰: 连续上涨笔, 价格新高但MACD面积缩小
        # 缠论标准: 面积取上涨笔中macd_hist>0(红柱)的绝对值之和
        for k in range(1, len(up_strokes)):
            prev = up_strokes[k - 1]
            curr = up_strokes[k]

            prev_hist = macd_hist.iloc[prev['start_idx']:prev['end_idx'] + 1].values
            curr_hist = macd_hist.iloc[curr['start_idx']:curr['end_idx'] + 1].values
            prev_area = sum(abs(x) for x in prev_hist if x > 0)
            curr_area = sum(abs(x) for x in curr_hist if x > 0)

            if curr['end_val'] > prev['end_val'] and curr_area < prev_area:
                signal_idx = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_idx < n:
                    sell_divergence.add(signal_idx)

        return buy_divergence, sell_divergence



    def _detect_2buy(self, strokes: List[Dict], buy_divergence: set, n: int) -> set:
        """v13: 2买检测

        缠论2买 = 1买(底背驰)后反弹, 再回调不破1买低点
        2买是最实用的买点: 既确认了底部, 价格又不差

        Returns:
            set of signal_idx — 2买位置
        """
        buy_2 = set()

        down_strokes = [s for s in strokes
                        if s['start_type'] == 'top' and s['end_type'] == 'bottom']
        up_strokes = [s for s in strokes
                      if s['start_type'] == 'bottom' and s['end_type'] == 'top']

        for ds in down_strokes:
            signal_1buy = ds['end_idx'] + self.bi_confirm_delay
            if signal_1buy not in buy_divergence:
                continue  # 不是1买, 跳过

            low_1buy = ds['end_val']  # 1买的低点

            # 找1买后的下一笔上涨(反弹)
            bounce = None
            for us in up_strokes:
                if us['start_idx'] > ds['end_idx']:
                    bounce = us
                    break
            if bounce is None:
                continue

            # 找反弹后的下一笔下跌(回调)
            pullback = None
            for ds2 in down_strokes:
                if ds2['start_idx'] > bounce['end_idx']:
                    pullback = ds2
                    break
            if pullback is None:
                continue

            # 回调不破1买低点 = 2买 (严格不破: 回调低点 > 1买低点)
            if pullback['end_val'] > low_1buy:
                signal_2buy = pullback['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_2buy < n and signal_2buy not in buy_divergence:
                    buy_2.add(signal_2buy)

        return buy_2

    def _detect_segments(self, strokes: List[Dict]) -> List[Dict]:
        """v14: 线段检测

        缠论线段破坏条件(两种任一):
        A. 笔破坏: 反向笔突破线段极值点(上涨线段的最新上涨笔高点<前高,或下跌线段最新下跌笔低点>前低)
        B. 中枢破坏: 反向笔突破线段中枢的ZG/ZD

        算法: 滑动窗口, 逐笔判断是否需要结束当前线段

        Returns:
            segments: [{
                'direction': 'up'|'down',
                'start_idx': int, 'end_idx': int,
                'start_val': float, 'end_val': float,
                'high': float, 'low': float,
                'stroke_start': int, 'stroke_end': int,
            }]
        """
        if len(strokes) < 3:
            return []

        segments = []
        seg_start = 0  # 当前线段起始笔索引

        for j in range(1, len(strokes)):
            seg_strokes = strokes[seg_start:j + 1]
            if len(seg_strokes) < 3:
                continue

            # 线段方向: 第一笔
            first = seg_strokes[0]
            seg_dir = 'up' if first['start_type'] == 'bottom' else 'down'

            # 线段中所有笔的高低
            seg_high = max(s['high'] for s in seg_strokes)
            seg_low = min(s['low'] for s in seg_strokes)

            # 线段中枢: 最近3笔的ZG/ZD (缠论标准: ZG=各笔高点最低值, ZD=各笔低点最高值)
            last3 = seg_strokes[-3:]
            zg = min(s['high'] for s in last3)
            zd = max(s['low'] for s in last3)

            current = strokes[j]
            break_seg = False

            if seg_dir == 'up':
                # 上涨线段破坏条件:
                # A: 最新下跌笔低点 < 线段起始低点(跌破起点)
                # B: 最新下跌笔低点 < 前一个下跌笔的低点(创新低)
                # C: 最新下跌笔的end_val < 中枢下沿ZD (笔破坏)
                down_strokes_in_seg = [s for s in seg_strokes[:-1]
                                       if s['end_type'] == 'bottom']
                if current['end_type'] == 'bottom':
                    if down_strokes_in_seg:
                        prev_lowest = min(s['end_val'] for s in down_strokes_in_seg)
                        # 跌破前一个下跌笔低点 (包含前一根)
                        all_down_ends = [s['end_val'] for s in seg_strokes
                                         if s['end_type'] == 'bottom']
                        if len(all_down_ends) >= 2 and current['end_val'] < all_down_ends[-2]:
                            break_seg = True
                    # 跌破中枢下沿
                    if current['end_val'] < zd:
                        break_seg = True
            else:
                # 下跌线段破坏条件:
                up_strokes_in_seg = [s for s in seg_strokes[:-1]
                                     if s['end_type'] == 'top']
                if current['end_type'] == 'top':
                    # 涨过前一个上涨笔高点
                    all_up_ends = [s['end_val'] for s in seg_strokes
                                   if s['end_type'] == 'top']
                    if len(all_up_ends) >= 2 and current['end_val'] > all_up_ends[-2]:
                        break_seg = True
                    # 涨过中枢上沿
                    if current['end_val'] > zg:
                        break_seg = True

            if break_seg:
                # 线段结束
                end_stroke_idx = j - 1  # 不含破坏笔
                if end_stroke_idx - seg_start >= 2:  # 至少3笔
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
                seg_start = j  # 破坏笔开始新线段

        # 处理最后的线段
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

    def _compute_segment_divergence(self, segments: List[Dict],
                                     strokes: List[Dict],
                                     macd_hist: pd.Series, n: int):
        """v14: 线段级别MACD面积背驰

        线段级别背驰比笔级别更可靠:
        - 上涨线段: 价格新高但MACD面积缩小 → 顶背驰(1卖)
        - 下跌线段: 价格新低但MACD面积缩小 → 底背驰(1买)

        Returns:
            buy_divergence: set of signal_idx
            sell_divergence: set of signal_idx
        """
        buy_divergence = set()
        sell_divergence = set()

        up_segs = [s for s in segments if s['direction'] == 'up']
        down_segs = [s for s in segments if s['direction'] == 'down']

        # 底背驰: 连续下跌线段, 价格新低 + 面积缩小
        # 缠论标准: 面积取下跌线段中macd_hist<0(绿柱)的绝对值之和
        for k in range(1, len(down_segs)):
            prev = down_segs[k - 1]
            curr = down_segs[k]

            prev_hist = macd_hist.iloc[prev['start_idx']:prev['end_idx'] + 1].values
            curr_hist = macd_hist.iloc[curr['start_idx']:curr['end_idx'] + 1].values
            prev_area = sum(abs(x) for x in prev_hist if x < 0)
            curr_area = sum(abs(x) for x in curr_hist if x < 0)

            if curr['low'] < prev['low'] and curr_area < prev_area:
                signal_idx = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_idx < n:
                    buy_divergence.add(signal_idx)

        # 顶背驰: 连续上涨线段, 价格新高 + 面积缩小
        # 缠论标准: 面积取上涨线段中macd_hist>0(红柱)的绝对值之和
        for k in range(1, len(up_segs)):
            prev = up_segs[k - 1]
            curr = up_segs[k]

            prev_hist = macd_hist.iloc[prev['start_idx']:prev['end_idx'] + 1].values
            curr_hist = macd_hist.iloc[curr['start_idx']:curr['end_idx'] + 1].values
            prev_area = sum(abs(x) for x in prev_hist if x > 0)
            curr_area = sum(abs(x) for x in curr_hist if x > 0)

            if curr['high'] > prev['high'] and curr_area < prev_area:
                signal_idx = curr['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_idx < n:
                    sell_divergence.add(signal_idx)

        return buy_divergence, sell_divergence

    def _detect_2sell(self, strokes: List[Dict], sell_divergence: set, n: int) -> set:
        """v14: 2卖检测

        缠论2卖 = 1卖(顶背驰)后下跌, 再反弹不过1卖高点
        2卖是最实用的卖点: 确认顶部后离场, 避免A浪杀跌

        Returns:
            set of signal_idx — 2卖位置
        """
        sell_2 = set()

        up_strokes = [s for s in strokes
                      if s['start_type'] == 'bottom' and s['end_type'] == 'top']
        down_strokes = [s for s in strokes
                        if s['start_type'] == 'top' and s['end_type'] == 'bottom']

        for us in up_strokes:
            signal_1sell = us['end_idx'] + self.bi_confirm_delay
            if signal_1sell not in sell_divergence:
                continue  # 不是1卖, 跳过

            high_1sell = us['end_val']  # 1卖的高点

            # 找1卖后的下一笔下跌(回调)
            drop = None
            for ds in down_strokes:
                if ds['start_idx'] > us['end_idx']:
                    drop = ds
                    break
            if drop is None:
                continue

            # 找回调后的下一笔上涨(反弹)
            bounce = None
            for us2 in up_strokes:
                if us2['start_idx'] > drop['end_idx']:
                    bounce = us2
                    break
            if bounce is None:
                continue

            # 反弹不过1卖高点 = 2卖 (严格不过: 反弹高点 < 1卖高点)
            if bounce['end_val'] < high_1sell:
                signal_2sell = bounce['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_2sell < n and signal_2sell not in sell_divergence:
                    sell_2.add(signal_2sell)

        return sell_2

    def _detect_pivots(self, strokes: List[Dict]) -> List[Dict]:
        """v15-final: 中枢识别 + 延伸/扩张/升级 (严格按附录A)

        【标准中枢】N=3笔重叠
        【延伸 Extension】N=4~8: 笔在中枢区间内反复震荡，ZG/ZD不变
            → extended=True (N=6~8), 走势积蓄期
        【扩张 Expansion】ZG或ZD被突破后回撤，区间扩大但级别不变
            → expanded=True, zg_expanded/zd_expanded更新
            → 条件A(上扩): stroke.high > ZG_orig 且 next.low ∈ [ZD_orig, ZG_orig]
            → 条件B(下扩): stroke.low < ZD_orig 且 next.high ∈ [ZD_orig, ZG_orig]
        【升级 Upgrade】N≥9: 中枢级别提升(30分→日线等)
            → upgraded=True, upgrade_level

        Returns:
            [{
                'zg': float,           # 中枢上沿(原始)
                'zd': float,           # 中枢下沿(原始)
                'zg_expanded': float,  # 扩张后ZG(仅expanded=True时有效)
                'zd_expanded': float,  # 扩张后ZD(仅expanded=True时有效)
                'stroke_start': int,
                'stroke_end': int,
                'direction': 'up'|'down',
                'start_idx': int,
                'end_idx': int,
                'total_strokes': int,    # 笔总数(含形成段+延伸段)
                'extended': bool,         # N=6~8延伸(extended zone)
                'expanded': bool,         # 扩张(区间已扩大)
                'expanded_mode': None|'A'|'B'|'C',  # 扩张类型
                'upgraded': bool,          # N≥9触发升级
            }]
        """
        if len(strokes) < 3:
            return []

        pivots = []
        i = 0

        while i <= len(strokes) - 3:
            s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]

            # === Step 1: 形成标准中枢 (缠论原文优化) ===
            # ZG = min(下降笔的high) — 下降笔是 end_type='top'
            # ZD = max(上升笔的low)  — 上升笔是 end_type='bottom'
            down_strokes = [s for s in [s1, s2, s3] if s['end_type'] == 'top']
            up_strokes   = [s for s in [s1, s2, s3] if s['end_type'] == 'bottom']

            if not down_strokes or not up_strokes:
                i += 1
                continue  # 需要至少1笔上升+1笔下降才能形成中枢

            zg_raw = min(s['high'] for s in down_strokes)
            zd_raw = max(s['low']  for s in up_strokes)

            # 重叠检查 (缠论skill方法): ZG > ZD 表示有效重叠
            if zg_raw > zd_raw:
                direction = 'down' if s1['end_type'] == 'bottom' else 'up'

                # 中枢内笔序列(从形成段开始)
                pivot_strokes = [s1, s2, s3]
                pivot_end = i + 2
                total_strokes = 3  # 包含形成段

                # === Step 2: 延伸/扩张/升级检测 ===
                expanded_flag = False
                expanded_mode = None   # 'A'/'B'/'C'
                extended_flag = False
                upgraded_flag = False
                zg_cur = zg_raw       # 当前ZG(可能因扩张而变化)
                zd_cur = zd_raw       # 当前ZD
                zg_expanded = zg_raw  # 扩张后ZG(仅expanded时有效)
                zd_expanded = zd_raw  # 扩张后ZD
                expansion_triggered_stroke = None  # 触发扩张的那笔

                j = i + 3

                while j < len(strokes):
                    sj = strokes[j]

                    # ===== 判断该笔是否属于本中枢 =====
                    # A. 完全在中枢内 → 延伸
                    if sj['high'] <= zg_cur and sj['low'] >= zd_cur:
                        pivot_strokes.append(sj)
                        pivot_end = j
                        total_strokes += 1

                        # 扩展标记: N=6~8 (total_strokes=6~9 means extension=3~6)
                        if 6 <= total_strokes <= 9:
                            extended_flag = True

                        j += 1
                        continue

                    # B. 扩张触发检测 (条件A/B/C)
                    # 条件A(上扩): sj的高点>ZG_orig 但 下一笔回到[ZD_orig, ZG_orig]内
                    cond_A = (sj['high'] > zg_raw and
                              j + 1 < len(strokes) and
                              zd_raw <= strokes[j + 1]['low'] <= zg_raw)

                    # 条件B(下扩): sj的低点<ZD_orig 但 下一笔回到[ZD_orig, ZG_orig]内
                    cond_B = (sj['low'] < zd_raw and
                              j + 1 < len(strokes) and
                              zd_raw <= strokes[j + 1]['high'] <= zg_raw)

                    if cond_A or cond_B:
                        # 扩张触发! 将该笔的极端点加入中枢
                        if cond_A:
                            zg_expanded = max(zg_raw, sj['high'])
                            zd_expanded = zd_raw
                            expanded_mode = 'A'
                        else:  # cond_B
                            zg_expanded = zg_raw
                            zd_expanded = min(zd_raw, sj['low'])
                            expanded_mode = 'B'

                        expanded_flag = True
                        expansion_triggered_stroke = sj

                        # 扩张触发笔加入中枢
                        pivot_strokes.append(sj)
                        pivot_end = j
                        total_strokes += 1
                        zg_cur = zg_expanded
                        zd_cur = zd_expanded

                        # 继续收集扩张后的延伸笔(用扩大后的区间判断)
                        k = j + 1
                        while k < len(strokes):
                            sk = strokes[k]
                            if sk['high'] <= zg_cur and sk['low'] >= zd_cur:
                                pivot_strokes.append(sk)
                                pivot_end = k
                                total_strokes += 1
                                k += 1
                            else:
                                break
                        j = k  # 推进到脱离点

                        # N≥9 → 升级
                        if total_strokes >= 9:
                            upgraded_flag = True

                        break  # 扩张笔处理完，退出延伸循环

                    # C. 既不在中枢内，也不触发扩张 → 中枢结束
                    else:
                        break

                # ===== 构建中枢记录 =====
                pivot_rec = {
                    'zg': zg_raw,
                    'zd': zd_raw,
                    'zg_expanded': zg_expanded,
                    'zd_expanded': zd_expanded,
                    'stroke_start': i,
                    'stroke_end': pivot_end,
                    'direction': direction,
                    'start_idx': s1['start_idx'],
                    'end_idx': strokes[pivot_end]['end_idx'],
                    'total_strokes': total_strokes,
                    'extended': extended_flag,            # N=6~8延伸
                    'expanded': expanded_flag,            # 扩张状态
                    'expanded_mode': expanded_mode,      # 'A'/'B'/'C'/None
                    'upgraded': upgraded_flag,           # N≥9升级
                }
                pivots.append(pivot_rec)

                # 从中枢结束的下一笔继续
                i = pivot_end + 1
            else:
                i += 1

        return pivots

    def _detect_pivot_divergence(self, pivots: List[Dict], strokes: List[Dict],
                                  macd_hist, n: int) -> Tuple[set, set]:
        """v14.1: 趋势背驰检测(缠论标准1买/1卖)

        趋势背驰定义:
        - 下跌趋势: 2个以上下跌中枢，最后一个中枢之后离开段MACD面积 < 进入段面积
        - 上涨趋势: 2个以上上涨中枢，最后一个中枢之后离开段MACD面积 < 进入段面积

        注意: 背驰信号仅标记位置，入场还需分型确认(bi_buy)配合

        Returns:
            buy_divergence: set of signal_idx — 趋势背驰1买位置
            sell_divergence: set of signal_idx — 趋势背驰1卖位置
        """
        buy_divergence = set()
        sell_divergence = set()

        if len(pivots) < 2:
            return buy_divergence, sell_divergence

        # ===== 下跌趋势背驰(1买) =====
        # 找连续的下跌中枢(方向=down)
        down_pivot_groups = []
        group = []
        for p in pivots:
            if p['direction'] == 'down':
                group.append(p)
            else:
                if len(group) >= 2:
                    down_pivot_groups.append(group)
                group = []
        if len(group) >= 2:
            down_pivot_groups.append(group)

        for grp in down_pivot_groups:
            if len(grp) < 2:
                continue

            last_pivot = grp[-1]

            # 进入段: 中枢前的下跌笔
            enter_stroke = strokes[last_pivot['stroke_start'] - 1] \
                if last_pivot['stroke_start'] > 0 else None
            # 离开段: 中枢后的下跌笔
            leave_stroke = strokes[last_pivot['stroke_end'] + 1] \
                if last_pivot['stroke_end'] + 1 < len(strokes) else None

            if enter_stroke is None or leave_stroke is None:
                continue
            if leave_stroke['end_type'] != 'bottom':
                continue  # 离开段必须是下跌笔(到底)

            # 价格创新低(离开段低点 < 进入段低点)
            if leave_stroke['end_val'] >= enter_stroke['end_val']:
                continue  # 没创新低，不是趋势背驰

            # MACD面积比较 (缠论标准: 下跌笔取绿柱面积)
            enter_hist = macd_hist.iloc[enter_stroke['start_idx']:enter_stroke['end_idx'] + 1].values
            leave_hist = macd_hist.iloc[leave_stroke['start_idx']:leave_stroke['end_idx'] + 1].values
            enter_area = sum(abs(x) for x in enter_hist if x < 0)
            leave_area = sum(abs(x) for x in leave_hist if x < 0)

            # 离开段面积 < 进入段面积 = 背驰
            if leave_area < enter_area:
                signal_idx = leave_stroke['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_idx < n:
                    buy_divergence.add(signal_idx)

        # ===== 上涨趋势背驰(1卖) =====
        up_pivot_groups = []
        group = []
        for p in pivots:
            if p['direction'] == 'up':
                group.append(p)
            else:
                if len(group) >= 2:
                    up_pivot_groups.append(group)
                group = []
        if len(group) >= 2:
            up_pivot_groups.append(group)

        for grp in up_pivot_groups:
            if len(grp) < 2:
                continue

            last_pivot = grp[-1]

            enter_stroke = strokes[last_pivot['stroke_start'] - 1] \
                if last_pivot['stroke_start'] > 0 else None
            leave_stroke = strokes[last_pivot['stroke_end'] + 1] \
                if last_pivot['stroke_end'] + 1 < len(strokes) else None

            if enter_stroke is None or leave_stroke is None:
                continue
            if leave_stroke['end_type'] != 'top':
                continue  # 离开段必须是上涨笔(到顶)

            # 价格创新高(离开段高点 > 进入段高点)
            if leave_stroke['end_val'] <= enter_stroke['end_val']:
                continue

            # MACD面积比较 (缠论标准: 上涨笔取红柱面积)
            enter_hist = macd_hist.iloc[enter_stroke['start_idx']:enter_stroke['end_idx'] + 1].values
            leave_hist = macd_hist.iloc[leave_stroke['start_idx']:leave_stroke['end_idx'] + 1].values
            enter_area = sum(abs(x) for x in enter_hist if x > 0)
            leave_area = sum(abs(x) for x in leave_hist if x > 0)

            if leave_area < enter_area:
                signal_idx = leave_stroke['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_idx < n:
                    sell_divergence.add(signal_idx)

        return buy_divergence, sell_divergence

    def _detect_3buy_context(self, filtered_fractals: List[Dict], df: pd.DataFrame,
                              strokes_from_bi: List[Dict] = None) -> pd.Series:
        """v15: 检测3买信号上下文

        3买条件(缠论标准定义):
        1. 中枢形成: >=3笔重叠区间 [ZD, ZG]
        2. 向上突破: 笔的高点 > ZG
        3. 回踩不破: 下一笔回调的低点 >= ZG (扩张中枢用zg_expanded)
        4. 回踩底分型 = 3买点

        v15升级:
        - 调用统一的_detect_pivots获取中枢(含扩张/扩展状态)
        - 扩张中枢的回踩验证使用zg_expanded而非zg
        - 扩展中枢的3买降权处理

        Returns: 与bi_buy同索引的布尔Series, True表示该bi_buy同时也是3买
        """
        n = len(df)
        third_buy = pd.Series(False, index=df.index)

        if len(filtered_fractals) < 5:
            return third_buy

        # 构建笔序列 (供_detect_pivots使用)
        strokes = []
        for j in range(len(filtered_fractals) - 1):
            f1, f2 = filtered_fractals[j], filtered_fractals[j + 1]
            strokes.append({
                'start_idx': f1['idx'],
                'high': max(f1['val'], f2['val']),
                'low': min(f1['val'], f2['val']),
                'end_idx': f2['idx'],
                'end_type': f2['type'],
            })

        if len(strokes) < 3:
            return third_buy

        # v15: 使用统一的_detect_pivots获取中枢(含扩张/扩展字段)
        pivots = self._detect_pivots(strokes)

        if not pivots:
            return third_buy

        # Step 2: 检测3买 — 中枢后向上突破 + 回踩不破ZG(扩张时用zg_expanded)
        for pivot in pivots:
            # v15: 使用扩张后的ZG
            zg = pivot['zg_expanded'] if pivot['expanded'] else pivot['zg']
            zd = pivot['zd_expanded'] if pivot['expanded'] else pivot['zd']
            pivot_expanded = pivot['expanded']
            pivot_extended = pivot['extended']

            next_s = pivot['stroke_end'] + 1

            if next_s >= len(strokes):
                continue

            # Phase 1: 寻找向上突破ZG的笔
            broke_out_idx = -1
            for k in range(next_s, len(strokes)):
                sk = strokes[k]
                if sk['high'] > zg:
                    broke_out_idx = k
                    break
                elif sk['low'] < zd:
                    # 向下破中枢下沿, 此中枢3买失效
                    break

            if broke_out_idx < 0:
                continue  # 未突破, 跳过

            # Phase 2: 突破后寻找回踩
            # 标准3买: 回踩低点 >= ZG (扩张中枢用zg_expanded)
            # 弱3买:   回踩低点 >= ZD (仅当无标准3买时)
            found_standard_3buy = False
            weak_3buy_signal = None
            for k in range(broke_out_idx + 1, len(strokes)):
                sk = strokes[k]
                if sk['end_type'] == 'bottom' and sk['low'] >= zg:
                    # v15: 扩张中枢的3买有效, 扩展中枢3买降权(不做为独立信号增强)
                    signal_idx = sk['end_idx'] + self.bi_confirm_delay
                    if 0 <= signal_idx < n:
                        third_buy.iloc[signal_idx] = True
                    found_standard_3buy = True
                    break  # 找到标准3买, 完成
                elif sk['end_type'] == 'bottom' and sk['low'] >= zd:
                    # 弱3买候选: 破ZG但未破ZD
                    if weak_3buy_signal is None:
                        weak_3buy_signal = sk
                elif sk['low'] < zd:
                    # 跌破中枢下沿, 3买彻底失效
                    break

            # 无标准3买但有弱3买, 使用弱3买
            if not found_standard_3buy and weak_3buy_signal is not None:
                signal_idx = weak_3buy_signal['end_idx'] + self.bi_confirm_delay
                if 0 <= signal_idx < n and not third_buy.iloc[signal_idx]:
                    third_buy.iloc[signal_idx] = True

        return third_buy

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

    def _close_position(self, code: str):
        """平仓辅助: 更新持仓数和行业计数 [v12.2]"""
        self._active_positions = max(0, self._active_positions - 1)
        industry = self._industry_map.get(code, '')
        if industry and industry in self._active_industries:
            self._active_industries[industry] = max(0, self._active_industries[industry] - 1)

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
