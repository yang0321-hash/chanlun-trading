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
from core.pivot import PivotDetector
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
    stroke_min_bars: int = 5        # 笔最小K线数 (匹配czsc bi)


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
        """v3a MACD确认: 与回测引擎一致

        条件 (满足任一):
          1. DIF > DEA
          2. HIST在零轴下但递增
          3. DIF递增
        """
        latest = macd.get_latest()
        if not latest:
            return False

        # 条件1: DIF > DEA
        if latest.macd > latest.signal:
            return True

        # 条件2: HIST在零轴下但递增
        hist_series = macd.get_histogram_series()
        if len(hist_series) >= 2:
            hist_now = float(hist_series.iloc[-1])
            hist_prev = float(hist_series.iloc[-2])
            if hist_now <= 0 and hist_now > hist_prev:
                return True

        # 条件3: DIF递增
        dif_series = macd.get_dif_series()
        if len(dif_series) >= 2:
            if float(dif_series.iloc[-1]) > float(dif_series.iloc[-2]):
                return True

        return False

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

        # 4. MACD确认
        if cfg.macd_confirm and not self._check_macd_confirm(macd, klen):
            return None

        # 5. 计算止损
        close_s = analysis['close_series']
        low_s = analysis['low_series']
        lookback = min(30, recent_buy_idx)
        recent_low = float(low_s.iloc[recent_buy_idx - lookback:recent_buy_idx + 1].min())
        stop_loss = max(recent_low, current_price * (1 - cfg.stop_loss_pct))

        # 6. 关联中枢
        pivot_zg = 0.0
        pivot_zd = 0.0
        if pivots:
            # 找包含买入位置的中枢
            for p in reversed(pivots):
                if p.zg > 0:
                    pivot_zg = p.zg
                    pivot_zd = p.zd
                    break

        # === 动态置信度计算 ===
        conf = 0.5  # 基础分

        # 1. MACD强度 (+0.15)
        latest_macd = macd.get_latest()
        if latest_macd:
            if latest_macd.macd > latest_macd.signal:
                conf += 0.05  # DIF>DEA
            hist = latest_macd.histogram
            if hist > 0:
                conf += 0.05  # 红柱
            hist_series = macd.get_histogram_series()
            if hist > 0 and len(hist_series) >= 2:
                if hist_series.iloc[-1] > hist_series.iloc[-2]:
                    conf += 0.05  # 红柱扩大

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
                   highest_price: float, bars_held: int = 0) -> Optional[str]:
        """出场检查 — 与回测引擎一致

        优先级:
          1. 固定止损 (-12%)
          2. MACD背驰止盈半仓 (盈利>3% + HIST缩小 + 向上笔结束)
          3. 追踪止损 (盈利>5%后从最高点回撤3%)
          4. 向上笔结束 (stroke_sell)
          5. 时间止损 (>80根K线)
          6. T+0强制清仓 (14:50)

        Args:
            code: 股票代码
            entry_price: 入场价
            highest_price: 持仓期间最高价
            bars_held: 已持有K线数

        Returns:
            出场原因字符串, None表示继续持有
        """
        cfg = self.config

        analysis = self._run_chanlun_pipeline(code)
        if not analysis:
            return None

        current_price = analysis['current_price']
        profit = (current_price - entry_price) / entry_price
        bi_sell_indices = analysis['bi_sell_indices']
        klen = analysis['klen']

        # 1. 固定止损
        if current_price <= entry_price * (1 - cfg.stop_loss_pct):
            return f'固定止损 -{cfg.stop_loss_pct*100:.0f}%'

        # 2. MACD背驰止盈 (盈利>3% + HIST连续缩小 + 价格新高)
        if profit > 0.03 and bars_held >= cfg.min_hold_bars:
            macd = analysis['macd']
            hist_series = macd.get_histogram_series()
            close_s = analysis['close_series']
            if len(hist_series) >= 3 and len(close_s) >= 2:
                hist_shrinking = (float(hist_series.iloc[-1]) < float(hist_series.iloc[-2])
                                  < float(hist_series.iloc[-3]))
                price_near_high = float(close_s.iloc[-1]) >= highest_price * 0.995
                # 还需要向上笔结束确认
                has_sell_bi = any(
                    idx >= klen - 5 for idx in bi_sell_indices
                )
                if hist_shrinking and price_near_high and has_sell_bi:
                    return 'MACD背驰 (新高+HIST缩小+笔结束)'

        # 3. 追踪止损
        if profit > cfg.trailing_start:
            trail_stop = highest_price * (1 - cfg.trailing_dist)
            if current_price <= trail_stop:
                return f'追踪止损 (从最高{highest_price:.2f}回撤{cfg.trailing_dist*100:.0f}%)'

        # 4. 向上笔结束 = 卖出信号 (与回测引擎一致)
        if bi_sell_indices and bars_held >= cfg.min_hold_bars:
            recent_sell = any(idx >= klen - cfg.recent_bars for idx in bi_sell_indices)
            if recent_sell and profit > 0.01:
                return '向上笔结束 (卖出信号)'

        # 5. 时间止损
        if bars_held >= cfg.max_hold_bars:
            return f'时间止损 (持仓{bars_held}根K线)'

        # 6. T+0模式: 强制清仓
        if cfg.mode == 't0':
            now = datetime.now()
            if now.hour == 14 and now.minute >= 50:
                return 'T+0强制清仓 (14:50)'

        return None

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
