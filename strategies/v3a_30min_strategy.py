"""v3a 30分钟缠论策略 — T+0 / 趋势双模式

已验证: Sharpe 4.11 (回测)
核心逻辑 (与回测引擎一致):
  1. 日线确认非下跌趋势 (MA20 > MA60)
  2. 30分钟向下笔结束 + MACD确认 → 入场
  3. 突破前一个30分钟中枢ZG时加仓
  4. MACD背驰/笔级别面积背驰 → 止盈
  5. 追踪止损 + 时间止损

信号源: core/ StrokeGenerator (等效于czsc bi)
入场: 向下笔结束(非BuySellPointDetector)

双模式:
  - T+0: 日内清仓, 不隔夜
  - trend: 持仓多日, 追踪止损保护

用法:
  from strategies.v3a_30min_strategy import V3a30MinStrategy, V3aConfig
  from data.hybrid_source import HybridSource

  config = V3aConfig(mode='trend')
  strategy = V3a30MinStrategy(config, HybridSource())

  # 入场扫描
  signal = strategy.scan_entry('002600')
  if signal:
      print(f'{signal.signal_type} @ {signal.price}, stop={signal.stop_loss}')

  # ZG突破加仓
  add_signal = strategy.check_zg_breakout('002600', position)

  # 出场检查
  exit_reason = strategy.check_exit('002600', position, price)
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Tuple

import pandas as pd

# 清除代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD

# czsc桥接 — 优先使用czsc bi (原始回测引擎)
try:
    from core.czsc_bridge import get_czsc_bi
    HAS_CZSC = True
except ImportError:
    HAS_CZSC = False


@dataclass
class V3aConfig:
    """v3a策略配置"""
    mode: str = 'trend'             # 't0' 或 'trend'
    stop_loss_pct: float = 0.12     # 最大止损12%
    trailing_start: float = 0.05    # 盈利5%启动追踪止损
    trailing_dist: float = 0.03     # 追踪止损距离3%
    min_hold_bars: int = 6          # 最少持有6根K线(3小时)
    max_hold_bars: int = 80         # 最多持有80根K线(~5天)
    cooldown_bars: int = 3          # 卖出后冷却3根K线
    entry_pct: float = 0.5          # 初始仓位50%
    add_pct: float = 0.3            # 加仓30%
    macd_confirm: bool = True       # MACD确认要求
    daily_ma_short: int = 20        # 日线短期均线
    daily_ma_long: int = 60         # 日线长期均线
    recent_bars: int = 30           # 信号回看K线数
    min_confidence: float = 0.5     # 最低置信度
    stroke_min_bars: int = 3        # 笔最小K线数 (网格搜索最优: 3>4>5)
    enable_5min_filter: bool = True  # 启用5分钟级别过滤
    min_5min_bars: int = 100        # 5分钟最少K线数
    stroke_5min_min_bars: int = 3   # 5分钟笔最小K线数


@dataclass
class V3aSignal:
    """v3a策略信号"""
    code: str
    signal_type: str       # '2buy', 'quasi2buy', 'zg_breakout', '2sell', 'quasi2sell'
    price: float
    stop_loss: float
    chan_stop: float = 0.0
    confidence: float = 0.0
    reason: str = ''
    pivot_zg: float = 0.0  # 关联中枢ZG
    pivot_zd: float = 0.0  # 关联中枢ZD


class V3a30MinStrategy:
    """v3a 30分钟缠论策略"""

    def __init__(self, config: V3aConfig = None, hybrid_source=None):
        self.config = config or V3aConfig()
        self.hs = hybrid_source

        # 日线趋势缓存: {code: (date_str, is_bullish)}
        self._daily_cache: Dict[str, Tuple[str, bool]] = {}

        # ChanLun分析缓存: {code: (timestamp, analysis_result)}
        self._analysis_cache: Dict[str, Tuple[float, dict]] = {}
        self._cache_ttl = 60  # 缓存60秒

    def check_daily_trend(self, code: str) -> Optional[bool]:
        """检查日线趋势: MA20 > MA60 = 非下跌

        Returns:
            True: 非下跌趋势(可交易)
            False: 下跌趋势(不入场)
            None: 数据不足
        """
        if not self.hs:
            return None

        # 缓存检查
        today = datetime.now().strftime('%Y-%m-%d')
        if code in self._daily_cache:
            cache_date, cache_val = self._daily_cache[code]
            if cache_date == today:
                return cache_val

        try:
            df = self.hs.get_kline(code, period='daily')
            if df is None or len(df) < self.config.daily_ma_long:
                return None

            close = df['close']
            ma_short = close.rolling(self.config.daily_ma_short).mean().iloc[-1]
            ma_long = close.rolling(self.config.daily_ma_long).mean().iloc[-1]

            is_bullish = bool(ma_short > ma_long)
            self._daily_cache[code] = (today, is_bullish)
            return is_bullish

        except Exception:
            return None

    def _run_chanlun_pipeline(self, code: str) -> Optional[dict]:
        """运行ChanLun分析管道 — 与回测引擎一致

        优先使用czsc bi (通过czsc_bridge), 回退到core/ stroke
        不使用BuySellPointDetector

        Returns:
            {
                'kline': KLine,
                'pivots': [...],
                'macd': MACD,
                'bi_buy_indices': [...],   # 向下笔结束位置
                'bi_sell_indices': [...],  # 向上笔结束位置
                'current_price': float,
                'klen': int,
                'close_series': pd.Series,
            }
        """
        if not self.hs:
            return None

        # 缓存检查
        now = datetime.now().timestamp()
        if code in self._analysis_cache:
            cache_time, cache_result = self._analysis_cache[code]
            if now - cache_time < self._cache_ttl:
                return cache_result

        try:
            df = self.hs.get_kline(code, period='30min')
            if df is None or len(df) < 120:
                return None

            close_s = pd.Series(df['close'].values)
            low_s = pd.Series(df['low'].values)
            macd = MACD(close_s)

            # 运行core/ ChanLun管道 (用于中枢计算, 以及czsc不可用时的bi)
            kline = KLine.from_dataframe(df, strict_mode=False)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 6:
                return None

            strokes = StrokeGenerator(kline, fractals, min_bars=self.config.stroke_min_bars).get_strokes()
            if len(strokes) < 4:
                return None

            pivots = PivotDetector(kline, strokes).get_pivots()

            # bi信号: 优先czsc bi, 回退到core/ stroke端点
            bi_buy_indices = []
            bi_sell_indices = []

            if HAS_CZSC:
                bi_list = get_czsc_bi(df)
                if bi_list:
                    for bi in bi_list:
                        if bi['is_down']:
                            bi_buy_indices.append(bi['end_idx'])
                        else:
                            bi_sell_indices.append(bi['end_idx'])

            if not bi_buy_indices and not bi_sell_indices:
                for s in strokes:
                    if s.end_value < s.start_value:
                        bi_buy_indices.append(s.end_index)
                    else:
                        bi_sell_indices.append(s.end_index)

            vol_s = pd.Series(df['volume'].values) if 'volume' in df.columns else None

            result = {
                'kline': kline,
                'pivots': pivots,
                'macd': macd,
                'strokes': strokes,
                'bi_buy_indices': bi_buy_indices,
                'bi_sell_indices': bi_sell_indices,
                'current_price': float(close_s.iloc[-1]),
                'klen': len(close_s),
                'close_series': close_s,
                'low_series': low_s,
                'volume_series': vol_s,
            }

            self._analysis_cache[code] = (now, result)
            return result

        except Exception:
            return None

    def _check_macd_confirm(self, macd: MACD, klen: int) -> bool:
        """MACD确认: 至少满足2项, 或仅绿柱缩短(最强单独条件)

        3个条件:
          1. DIF > DEA (金叉)
          2. 绿柱缩短 (HIST<0且递增, 最佳提前量)
          3. DIF递增 (动能向上)
        """
        latest = macd.get_latest()
        if not latest:
            return False

        score = 0
        has_green_shrink = False

        # 条件1: DIF > DEA
        if latest.macd > latest.signal:
            score += 1

        # 条件2: 绿柱缩短
        hist_series = macd.get_histogram_series()
        if len(hist_series) >= 2:
            hist_now = float(hist_series.iloc[-1])
            hist_prev = float(hist_series.iloc[-2])
            if hist_now <= 0 and hist_now > hist_prev:
                score += 1
                has_green_shrink = True

        # 条件3: DIF递增
        dif_series = macd.get_dif_series()
        if len(dif_series) >= 2:
            if float(dif_series.iloc[-1]) > float(dif_series.iloc[-2]):
                score += 1

        # 至少2项, 或仅绿柱缩短
        return score >= 2 or (score == 1 and has_green_shrink)

    def _check_5min_filter(self, code: str,
                           cutoff_time=None) -> Optional[dict]:
        """5分钟级别过滤 — 检测超涨 (短期上涨趋势是否已经耗尽)

        核心逻辑: 5分钟走势类型判断
          判断5分钟是否已经走了"多中枢上涨趋势", 若趋势已耗尽,
          在30分钟出现买点时不应入场, 因为短期可能面临深度回调.

        超涨 = 上涨趋势成熟 + 动能衰竭:
          1. 从最近显著低点至今涨幅 > 10%
          2. 上涨过程中形成2+个中枢 (趋势成熟)
          3. 最后一个主要上冲笔的MACD面积 < 峰值上冲笔的40% (动能衰竭)
          4. 当前价格接近上涨趋势高点 (不是已经回调后的入场)

        600869案例:
          5min从13.94涨到15.79, 二中枢上涨完成
          最后主要上冲(14.86→15.83)MACD面积=0.81, 峰值=2.83
          ratio=28.6% → 超涨否决

        Args:
            code: 股票代码
            cutoff_time: 截止时间 (回测用, 只看此时间之前的5min数据)
                         None = 用全量数据 (实盘)

        Returns:
            {'pass': bool, 'has_sell': bool, 'reason': str,
             'stop_5min': float, 'has_buy': bool}
            None: 数据不足或未启用
        """
        if not self.config.enable_5min_filter or not self.hs:
            return None

        try:
            df = self.hs.get_kline(code, period='5min')
            if df is None or len(df) < self.config.min_5min_bars:
                return None

            if cutoff_time is not None:
                df = df[df.index <= cutoff_time]
                if len(df) < self.config.min_5min_bars:
                    return None

            kline = KLine.from_dataframe(df, strict_mode=False)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 6:
                return None

            strokes = StrokeGenerator(
                kline, fractals, min_bars=self.config.stroke_5min_min_bars
            ).get_strokes()
            if len(strokes) < 6:
                return None

            close_s = pd.Series(df['close'].values)
            macd = MACD(close_s)
            current_price = float(close_s.iloc[-1])

            # === 1. 找最近显著低点 (最近300笔范围内的最低点) ===
            recent_n = min(300, len(strokes))
            recent_strokes = strokes[-recent_n:]
            low_val = float('inf')
            low_idx = 0
            for s in recent_strokes:
                s_low = min(s.start_value, s.end_value)
                if s_low < low_val:
                    low_val = s_low
                    low_idx = min(s.start_index, s.end_index)

            # 从低点至今的总涨幅
            rise_pct = (current_price - low_val) / low_val if low_val > 0 else 0

            # === 2. 中枢数量 (趋势成熟度) ===
            pivots_5m = PivotDetector(kline, strokes).get_pivots()
            # 统计低点之后形成的中枢数
            num_pivots = 0
            if pivots_5m:
                for p in pivots_5m:
                    # 中枢的笔在低点之后形成
                    if hasattr(p, 'start_index') and p.start_index >= low_idx:
                        num_pivots += 1
                    elif not hasattr(p, 'start_index'):
                        num_pivots += 1
                if not hasattr(pivots_5m[0], 'start_index'):
                    num_pivots = len(pivots_5m[-5:])

            # === 3. 动能衰竭: 最后主要上冲 vs 峰值上冲 ===
            up_strokes = [s for s in strokes if s.end_value > s.start_value]
            # 主要上冲笔: 涨幅 > 3%
            major_ups = []
            for s in up_strokes:
                pct = (s.end_value - s.start_value) / s.start_value
                if pct > 0.03 and s.start_index >= low_idx:
                    area = macd.compute_area(s.start_index, s.end_index, 'up')
                    major_ups.append((s, area))

            momentum_ratio = 1.0
            peak_area = 0.0
            last_major_area = 0.0
            if len(major_ups) >= 2:
                peak_area = max(a for _, a in major_ups)
                last_major_area = major_ups[-1][1]
                if peak_area > 0:
                    momentum_ratio = last_major_area / peak_area

            # === 4. 价格位置: 是否接近上涨趋势高点 ===
            high_val = max(s.end_value if s.end_value > s.start_value
                          else s.start_value for s in recent_strokes)
            near_high = current_price >= high_val * 0.97

            # === 5. 综合判断: 超涨检测 ===
            overextended = False
            reason = ''

            # 标准超涨: 多中枢上涨 + 动能衰竭 + 近高点
            if (rise_pct > 0.10 and num_pivots >= 2
                    and momentum_ratio < 0.40 and near_high):
                overextended = True
                reason = (f'超涨: {num_pivots}中枢+{rise_pct:.0%}涨幅'
                          f'+动能{momentum_ratio:.0%}+近高点')

            # 极端超涨: 即使中枢少, 极端动能衰竭也算
            elif (rise_pct > 0.15 and momentum_ratio < 0.20
                    and near_high and len(major_ups) >= 3):
                overextended = True
                reason = (f'超涨: 极端动能衰竭({momentum_ratio:.0%})'
                          f'+{rise_pct:.0%}涨幅')

            # 5分钟最后向下笔低点作为参考止损
            stop_5min = 0.0
            for s in reversed(strokes):
                if s.end_value < s.start_value:
                    stop_5min = float(df['low'].iloc[s.end_index])
                    break

            # 看涨信号: 最近有向上突破 + 动能充沛
            has_bullish = (momentum_ratio > 0.8 and not overextended
                           and not (strokes[-1].end_value < strokes[-1].start_value))

            return {
                'pass': not overextended,
                'has_sell': overextended,
                'reason': reason,
                'sell_types': [reason] if overextended else [],
                'has_buy': has_bullish,
                'stop_5min': stop_5min,
                'debug': {
                    'rise_pct': rise_pct,
                    'num_pivots': num_pivots,
                    'momentum_ratio': momentum_ratio,
                    'peak_area': peak_area,
                    'last_major_area': last_major_area,
                    'near_high': near_high,
                    'current_price': current_price,
                    'high_val': high_val,
                    'low_val': low_val,
                },
            }

        except Exception:
            return None

    def scan_entry(self, code: str) -> Optional[V3aSignal]:
        """入场扫描 — 与回测引擎一致

        逻辑:
          1. 日线非下跌趋势 (MA20 > MA60)
          2. 30分钟向下笔结束 (stroke_buy)
          3. MACD确认 (DIF>DEA 或 HIST递增 或 DIF递增)
          4. 止损: 近30根K线最低价, 上限12%

        Returns:
            V3aSignal 如果有入场信号, 否则 None
        """
        # 1. 日线趋势过滤
        daily_ok = self.check_daily_trend(code)
        if daily_ok is False:  # 明确下跌, 跳过; None=数据不足则放行
            return None

        # 2. 运行ChanLun管道
        analysis = self._run_chanlun_pipeline(code)
        if not analysis:
            return None

        bi_buy_indices = analysis['bi_buy_indices']
        if not bi_buy_indices:
            return None

        current_price = analysis['current_price']
        klen = analysis['klen']
        macd = analysis['macd']
        pivots = analysis['pivots']
        cfg = self.config

        # 3. 找最近的向下笔结束
        recent_buy_idx = None
        for idx in reversed(bi_buy_indices):
            if idx >= klen - cfg.recent_bars and idx < klen:
                recent_buy_idx = idx
                break

        if recent_buy_idx is None:
            return None

        # 3.5 2买结构确认: 当前笔低点 > 前一个笔低点 (higher low)
        if len(bi_buy_indices) >= 2:
            prev_buy_idx = None
            for idx in reversed(bi_buy_indices):
                if idx < recent_buy_idx:
                    prev_buy_idx = idx
                    break
            if prev_buy_idx is not None:
                low_s = analysis['low_series']
                current_low = float(low_s.iloc[recent_buy_idx])
                prev_low = float(low_s.iloc[prev_buy_idx])
                if current_low <= prev_low:
                    return None  # 创新低, 不是2买结构

        # 4. MACD确认 (收紧: ≥2项 或 绿柱缩短)
        if cfg.macd_confirm and not self._check_macd_confirm(macd, klen):
            return None

        # 4.5 5分钟级别过滤
        filter_5min = self._check_5min_filter(code)
        if filter_5min is not None and filter_5min['has_sell']:
            return None

        # 4.6 回调缩量过滤
        vol_s = analysis.get('volume_series')
        if vol_s is not None and recent_buy_idx >= 5:
            pullback_vol = float(vol_s.iloc[max(0, recent_buy_idx - 10):recent_buy_idx + 1].mean())
            pre_vol_start = max(0, recent_buy_idx - 30)
            pre_vol_end = max(0, recent_buy_idx - 10)
            if pre_vol_end > pre_vol_start:
                pre_vol = float(vol_s.iloc[pre_vol_start:pre_vol_end].mean())
                if pre_vol > 0 and pullback_vol > pre_vol * 0.85:
                    return None  # 回调未缩量, 抛压未竭

        # 4.7 结构支撑过滤: 入场位置需靠近支撑
        close_s = analysis['close_series']
        low_s = analysis['low_series']

        # 计算止损先确定pivot
        lookback = min(30, recent_buy_idx)
        recent_low = float(low_s.iloc[recent_buy_idx - lookback:recent_buy_idx + 1].min())
        stop_loss = max(recent_low, current_price * (1 - cfg.stop_loss_pct))

        pivot_zg = 0.0
        pivot_zd = 0.0
        if pivots:
            for p in reversed(pivots):
                if p.zg > 0:
                    pivot_zg = p.zg
                    pivot_zd = p.zd
                    break

        near_support = False
        if pivot_zd > 0 and current_price <= pivot_zd * 1.03:
            near_support = True
        if not near_support:
            strokes_list = analysis.get('strokes')
            if strokes_list:
                for s in reversed(strokes_list):
                    if s.end_value > s.start_value:
                        if current_price <= s.start_value * 1.03:
                            near_support = True
                        break
        if not near_support:
            return None  # 悬空入场, 无结构支撑

        # 5. 止损修正: 类2买不低于中枢ZD, 参考分钟止损
        if pivot_zd > 0:
            stop_loss = max(stop_loss, pivot_zd)
        if filter_5min and filter_5min['stop_5min'] > 0:
            stop_loss = max(stop_loss, filter_5min['stop_5min'])

        # === 动态置信度计算 ===
        conf = 0.5  # 基础分

        # 1. MACD状态 (+0.20) — 奖励提前量(绿柱缩短), 不奖励追涨(红柱扩大)
        latest_macd = macd.get_latest()
        if latest_macd:
            hist = latest_macd.histogram
            hist_series = macd.get_histogram_series()
            dif_val = latest_macd.macd  # DIF

            # DIF上穿零轴 = 趋势由空转多, 强确认 (+0.10)
            dif_series = macd.get_dif_series()
            if len(dif_series) >= 2:
                dif_now = float(dif_series.iloc[-1])
                dif_prev = float(dif_series.iloc[-2])
                if dif_prev <= 0 and dif_now > 0:
                    conf += 0.10  # DIF刚上零轴

            # 绿柱缩短 = 下行动能衰竭, 最佳提前入场信号 (+0.10)
            if hist < 0 and len(hist_series) >= 2:
                if float(hist_series.iloc[-1]) > float(hist_series.iloc[-2]):
                    conf += 0.10  # 绿柱缩短
            elif hist >= 0:
                # 已转红柱(金叉已发生), 信号确认但偏追涨 (+0.05)
                conf += 0.05

        # 2. 量价配合 (+0.15)
        vol_s = analysis.get('volume_series')
        if vol_s is not None and len(vol_s) >= 10:
            vol_ma5 = float(vol_s.iloc[-5:].mean())
            vol_ma20 = float(vol_s.iloc[-20:].mean()) if len(vol_s) >= 20 else vol_ma5
            if vol_ma20 > 0 and vol_ma5 > vol_ma20 * 1.3:
                conf += 0.10  # 放量
            elif vol_ma20 > 0 and vol_ma5 > vol_ma20 * 1.1:
                conf += 0.05  # 温和放量
            # 下跌缩量 = 健康回调
            if recent_buy_idx >= 5:
                pullback_vol = float(vol_s.iloc[recent_buy_idx-5:recent_buy_idx+1].mean())
                pre_vol = float(vol_s.iloc[max(0,recent_buy_idx-20):recent_buy_idx-5].mean()) if recent_buy_idx >= 20 else pullback_vol
                if pre_vol > 0 and pullback_vol < pre_vol * 0.7:
                    conf += 0.05  # 回调缩量

        # 3. 中枢位置 (+0.15)
        if pivot_zg > 0 and pivot_zd > 0:
            mid_pivot = (pivot_zg + pivot_zd) / 2
            if current_price >= pivot_zg:
                conf += 0.10  # 中枢上方 = 强
            elif current_price >= mid_pivot:
                conf += 0.05  # 中枢中上部
            # 中枢宽度适中 (不太宽不太窄)
            pivot_range_pct = (pivot_zg - pivot_zd) / pivot_zd
            if 0.02 < pivot_range_pct < 0.10:
                conf += 0.05  # 标准中枢

        # 4. 止损空间合理 (+0.05)
        stop_pct = (current_price - stop_loss) / current_price
        if 0.02 <= stop_pct <= 0.08:
            conf += 0.05  # 止损空间合理(2-8%)

        # 5. 5分钟级别确认 (+0.10)
        if filter_5min and filter_5min['has_buy']:
            conf += 0.10

        conf = min(0.95, max(0.3, conf))

        # === 信号类型细分 ===
        signal_type = 'bi_buy'
        if pivot_zg > 0 and pivot_zd > 0:
            if current_price >= pivot_zg:
                signal_type = '3buy'  # 中枢上方回踩 = 3买特征
            elif current_price >= pivot_zd:
                signal_type = '2buy'  # 中枢内 = 2买特征
            else:
                signal_type = '1buy'  # 中枢下方 = 1买特征

        return V3aSignal(
            code=code,
            signal_type=signal_type,
            price=current_price,
            stop_loss=float(stop_loss),
            chan_stop=float(recent_low),
            confidence=conf,
            reason=f'{signal_type} idx={recent_buy_idx} conf={conf:.2f}',
            pivot_zg=float(pivot_zg),
            pivot_zd=float(pivot_zd),
        )

    def check_zg_breakout(self, code: str, current_position: dict = None) -> Optional[V3aSignal]:
        """检查ZG突破加仓

        条件: 当前价格突破最近30分钟中枢的ZG

        Args:
            code: 股票代码
            current_position: 当前持仓信息 (需要已有持仓才加仓)
        """
        if not current_position:
            return None

        analysis = self._run_chanlun_pipeline(code)
        if not analysis:
            return None

        pivots = analysis['pivots']
        current_price = analysis['current_price']

        if not pivots:
            return None

        # 找最近已确认的中枢
        last_pivot = None
        for p in reversed(pivots):
            if p.confirmed and p.zg > 0:
                last_pivot = p
                break

        if not last_pivot:
            return None

        # 当前价格突破ZG
        if current_price <= last_pivot.zg:
            return None

        # 确认突破幅度 > 0.5%
        breakout_pct = (current_price - last_pivot.zg) / last_pivot.zg
        if breakout_pct < 0.005:
            return None

        stop_loss = last_pivot.zd  # 加仓止损在中枢下沿

        # ZG突破置信度: 基于突破幅度
        conf = min(0.85, 0.6 + breakout_pct * 2)  # 突破0.5%=0.61, 突破5%=0.70, 突破12.5%=0.85

        return V3aSignal(
            code=code,
            signal_type='zg_breakout',
            price=current_price,
            stop_loss=float(stop_loss),
            chan_stop=float(last_pivot.zd),
            confidence=conf,
            reason=f'突破中枢ZG={last_pivot.zg:.2f} (+{breakout_pct*100:.1f}%)',
            pivot_zg=float(last_pivot.zg),
            pivot_zd=float(last_pivot.zd),
        )

    def check_exit(self, code: str, entry_price: float,
                   highest_price: float, bars_held: int = 0,
                   current_stop: float = 0.0,
                   last_trail_bi: int = -1) -> Optional[str]:
        """出场检查 — 缠论结构出场

        优先级:
          1. 固定止损 (-12%)
          2. 结构追踪止损 (向上笔起点跟随)
          3. 时间止损 (>120根K线)
          4. T+0强制清仓 (14:50)

        Args:
            code: 股票代码
            entry_price: 入场价
            highest_price: 持仓期间最高价
            bars_held: 已持有K线数
            current_stop: 当前止损价
            last_trail_bi: 上次追踪的向上笔结束索引

        Returns:
            出场原因字符串, None表示继续持有
        """
        cfg = self.config

        analysis = self._run_chanlun_pipeline(code)
        if not analysis:
            return None

        current_price = analysis['current_price']

        # 1. 固定止损
        if current_price <= entry_price * (1 - cfg.stop_loss_pct):
            return f'固定止损 -{cfg.stop_loss_pct*100:.0f}%'

        # 2. 结构追踪止损 (由调用方根据向上笔起点维护 current_stop)
        if current_stop > 0 and current_price <= current_stop:
            return f'结构止损 (跌破{current_stop:.2f})'

        # 3. 时间止损
        if bars_held >= 120:
            return f'时间止损 (持仓{bars_held}根K线)'

        # 4. T+0模式: 强制清仓
        if cfg.mode == 't0':
            now = datetime.now()
            if now.hour == 14 and now.minute >= 50:
                return 'T+0强制清仓 (14:50)'

        return None

    def get_structural_stop(self, code: str, last_trail_bi: int = -1) -> Tuple[float, int]:
        """获取最新的结构追踪止损位

        检查是否有新的向上笔完成, 返回更新后的止损价和笔索引。
        调用方应: new_stop = max(old_stop, returned_stop)

        Args:
            code: 股票代码
            last_trail_bi: 上次追踪的向上笔结束索引

        Returns:
            (structural_stop, new_last_trail_bi)
            structural_stop = 0 表示无新向上笔
        """
        analysis = self._run_chanlun_pipeline(code)
        if not analysis:
            return 0.0, last_trail_bi

        strokes_list = analysis.get('strokes')
        if not strokes_list:
            return 0.0, last_trail_bi

        best_stop = 0.0
        best_bi = last_trail_bi
        for s in strokes_list:
            if s.end_value > s.start_value:  # 向上笔
                if s.end_index > last_trail_bi:
                    struct_stop = s.start_value
                    if struct_stop > best_stop:
                        best_stop = struct_stop
                        best_bi = s.end_index

        return best_stop, best_bi

    def check_macd_divergence_exit(self, code: str, entry_idx: int) -> bool:
        """检查MACD面积背驰 — 是否应该减仓

        比较持仓期间最近两个向上笔的MACD面积,
        如果后者 < 前者的50%, 说明动能衰竭。

        Args:
            code: 股票代码
            entry_idx: 入场时的K线索引

        Returns:
            True: 应该减仓
        """
        analysis = self._run_chanlun_pipeline(code)
        if not analysis:
            return False

        strokes_list = analysis.get('strokes')
        macd = analysis['macd']
        if not strokes_list:
            return False

        held_up_bis = []
        for s in strokes_list:
            if s.end_value > s.start_value and s.end_index > entry_idx:
                area = macd.compute_area(s.start_index, s.end_index, 'up')
                if area > 0:
                    held_up_bis.append(area)

        if len(held_up_bis) >= 2:
            return held_up_bis[-1] < held_up_bis[-2] * 0.5

        return False

    def get_exit_config_dict(self) -> dict:
        """返回适配 UnifiedExitManager 的配置字典"""
        return {
            'fixed_stop_pct': self.config.stop_loss_pct,
            'trailing_activation': self.config.trailing_start,
            'trailing_offset': self.config.trailing_dist,
            'use_atr_trailing': False,
            'time_stop_bars': self.config.max_hold_bars,
        }

    def clear_cache(self):
        """清空缓存 (每日开始时调用)"""
        self._daily_cache.clear()
        self._analysis_cache.clear()


# 便捷函数
def create_strategy(mode: str = 'trend', hybrid_source=None) -> V3a30MinStrategy:
    """创建v3a策略实例"""
    config = V3aConfig(mode=mode)
    return V3a30MinStrategy(config, hybrid_source)
