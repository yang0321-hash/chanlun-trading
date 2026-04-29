"""日线信号的30min缠论确认

统一入场逻辑:
  所有日线买点 → 30min检测中枢背驰1买 → 等2买确认 → 入场
  止损 = 30min 1买低点 (破了说明底是假的)

状态:
  - 30min出现2买 → 确认入场, 止损=1买低点
  - 30min出现1买但无2买 → 等2买 (选股池观察)
  - 30min无买点 → 等回调 (选股池观察)
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict

import pandas as pd

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD


@dataclass
class ConfirmResult:
    passed: bool = False
    confidence: float = 0.0
    stop_loss: float = 0.0
    reason: str = ''
    details: Dict = field(default_factory=dict)


class Daily30minConfirmer:
    def __init__(self, hybrid_source, min_30min_bars: int = 80):
        self.hs = hybrid_source
        self.min_30min_bars = min_30min_bars
        self._analysis_cache: Dict[str, tuple] = {}
        self._cache_ttl = 120

    def confirm_daily_signal(self, code: str, daily_context: dict) -> Optional[ConfirmResult]:
        """日线买点 → 30min等2买入场, 止损=1买低点

        核心逻辑:
          1. 30min检测买卖点 (1买/2买/3买)
          2. 最新买点是2买 → 入场, 找前面1买低点作止损
          3. 最新买点是1买 → 等2买形成
          4. 无买点 → 等回调

        Args:
            code: 股票代码 (sh600519)
            daily_context: 日线上下文
        """
        analysis = self._run_30min_pipeline(code)
        daily_stop = daily_context.get('stop_price', 0)

        if not analysis:
            return ConfirmResult(
                passed=True, confidence=0.40, stop_loss=0.0,
                reason='30min数据不足，默认放行',
                details={'no_data': True},
            )

        buy_points = analysis.get('buy_points', [])
        current_price = analysis['current_price']

        if not buy_points:
            return ConfirmResult(
                passed=False, confidence=0.0, stop_loss=daily_stop,
                reason='30min无买点, 等回调',
                details={'waiting': True, 'stage': 'no_buy_point'},
            )

        last_buy = buy_points[-1]
        buy_type = last_buy.point_type
        kline_len = len(analysis['close_series'])
        distance = kline_len - last_buy.index

        # === 2买(含强2买/类2买): 确认入场 ===
        if buy_type in ('2buy', '2buy_strong', 'class2buy', '2b3bbuy'):
            # 找1买低点作止损
            stop_1buy = self._find_1buy_stop(buy_points, last_buy.index)
            final_stop = stop_1buy if stop_1buy > 0 else daily_stop
            if daily_stop > 0:
                final_stop = max(final_stop, daily_stop)

            if distance > 20:
                return ConfirmResult(
                    passed=False, confidence=0.0, stop_loss=final_stop,
                    reason=f'30min 2买距今{distance}根, 可能已过时, 等1买→2买',
                    details={'waiting': True, 'stage': 'stale_2buy', 'distance': distance},
                )

            conf = 0.50
            conf += last_buy.confidence * 0.20
            if buy_type in ('2buy_strong', '2b3bbuy'):
                conf += 0.15
            if self._has_recent_bottom_fractal(analysis, window=5):
                conf += 0.10
            conf = min(0.95, max(0.45, conf))

            stop_info = f', 止损@1买低{stop_1buy:.2f}' if stop_1buy > 0 else ''
            return ConfirmResult(
                passed=True, confidence=conf, stop_loss=float(final_stop),
                reason=f'30min {buy_type}确认入场{stop_info}',
                details={
                    'buy_type': buy_type,
                    'buy_confidence': round(last_buy.confidence, 2),
                    'stop_1buy': round(stop_1buy, 2) if stop_1buy > 0 else 0,
                    'distance': distance,
                },
            )

        # === 1买: 等2买 ===
        if buy_type in ('1buy',):
            stop_1buy = last_buy.stop_loss if last_buy.stop_loss > 0 else 0
            return ConfirmResult(
                passed=False, confidence=0.0,
                stop_loss=max(stop_1buy, daily_stop) if stop_1buy > 0 else daily_stop,
                reason=f'30min 1买已出(距今{distance}根), 等2买确认入场',
                details={
                    'waiting': True, 'stage': 'wait_2buy',
                    'stop_1buy': round(stop_1buy, 2),
                    'distance': distance,
                },
            )

        # === 3买: 也等 → 30min 3买说明已经在反弹了, 等下一个1买→2买更安全 ===
        if buy_type in ('3buy', '3buy_strong'):
            return ConfirmResult(
                passed=False, confidence=0.0, stop_loss=daily_stop,
                reason=f'30min {buy_type}(已在反弹), 等1买→2买更安全',
                details={'waiting': True, 'stage': f'wait_1buy_after_{buy_type}'},
            )

        # 其他: 等
        return ConfirmResult(
            passed=False, confidence=0.0, stop_loss=daily_stop,
            reason=f'30min {buy_type}, 等明确的1买→2买',
            details={'waiting': True, 'stage': 'wait'},
        )

    def check_top_fractal_5stroke(self, code: str, daily_context: dict) -> Optional[ConfirmResult]:
        """日线顶分型也走统一逻辑: 等30min 1买→2买"""
        return self.confirm_daily_signal(code, daily_context)

    def _find_1buy_stop(self, buy_points: list, current_2buy_idx: int) -> float:
        """找到2买之前最近的1买, 返回其stop_loss(=1买低点)"""
        for bp in reversed(buy_points):
            if bp.index >= current_2buy_idx:
                continue
            if bp.point_type in ('1buy',):
                return bp.stop_loss if bp.stop_loss > 0 else 0
        # fallback: 找2买前最近的向下笔末端
        return 0

    def _has_recent_bottom_fractal(self, analysis: dict, window: int = 5) -> bool:
        from core.fractal import FractalType
        fractals = analysis.get('fractals', [])
        kline_len = len(analysis['close_series'])
        if not fractals:
            return False
        for f in reversed(fractals[-10:]):
            if f.type == FractalType.BOTTOM and (kline_len - f.index) <= window:
                return True
        return False

    def _run_30min_pipeline(self, code: str) -> Optional[dict]:
        now = datetime.now().timestamp()
        if code in self._analysis_cache:
            cache_time, cache_result = self._analysis_cache[code]
            if now - cache_time < self._cache_ttl:
                return cache_result

        try:
            df = self.hs.get_kline(code, period='30min')
            if df is None or len(df) < self.min_30min_bars:
                return None

            close_s = pd.Series(df['close'].values)
            macd = MACD(close_s)

            kline = KLine.from_dataframe(df, strict_mode=False)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 4:
                return None

            strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
            if len(strokes) < 3:
                return None

            pivots = PivotDetector(kline, strokes).get_pivots()

            det = BuySellPointDetector(fractals, strokes, [], pivots, macd=macd)
            buys, _ = det.detect_all()

            seen = {}
            for b in buys:
                if b.index not in seen or b.confidence > seen[b.index].confidence:
                    seen[b.index] = b
            buy_points = sorted(seen.values(), key=lambda x: x.index)

            result = {
                'strokes': strokes,
                'pivots': pivots,
                'fractals': fractals,
                'macd': macd,
                'buy_points': buy_points,
                'current_price': float(close_s.iloc[-1]),
                'klen': len(close_s),
                'close_series': close_s,
            }
            self._analysis_cache[code] = (now, result)
            return result
        except Exception:
            return None
