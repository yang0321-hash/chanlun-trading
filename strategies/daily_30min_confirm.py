"""日线信号感知的30min确认

替代v3a作为日线级别买卖点的30min二次确认。
v3a是独立的30min交易策略，条件与日线3买信号不匹配。
本模块知道日线信号为什么触发（中枢ZG/ZD），验证30min是否支持入场。

5个检查:
  1. 30min回调状态 — 最近笔是向下的
  2. 接近日线支撑 — 价格 >= 日线pivot_zd * 0.97
  3. MACD支持 — DIF>DEA / 绿柱缩短 / DIF递增 (任1)
  4. 未超涨 — 复用5min过滤器
  5. 结构质量 — 回调未破日线支撑 / 底分型形成

通过条件: 1 + 2 + (3或5) + 4
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
        """确认日线信号在30min级别是否支持入场

        Args:
            code: 股票代码 (sh600519)
            daily_context: {
                'signal_type': '3buy',
                'pivot_zg': float,  # 日线中枢ZG
                'pivot_zd': float,  # 日线中枢ZD
                'pivot_gg': float,  # 日线中枢GG (optional)
                'entry_price': float,
                'stop_price': float,
            }
        """
        analysis = self._run_30min_pipeline(code)
        if not analysis:
            return ConfirmResult(
                passed=True, confidence=0.40, stop_loss=0.0,
                reason='30min数据不足，默认放行',
                details={'no_data': True},
            )

        daily_pivot_zg = daily_context.get('pivot_zg', 0)
        daily_pivot_zd = daily_context.get('pivot_zd', 0)
        daily_pivot_gg = daily_context.get('pivot_gg', daily_pivot_zg)
        signal_type = daily_context.get('signal_type', '2buy')
        daily_stop = daily_context.get('stop_price', 0)
        current_price = analysis['current_price']

        checks = {}
        stop_loss = daily_stop

        # Check 1: 30min回调状态
        checks['pullback'] = self._check_pullback_state(analysis)
        if not checks['pullback']['pass']:
            return ConfirmResult(
                passed=False, confidence=0.0, stop_loss=stop_loss,
                reason=f"回调状态: {checks['pullback']['reason']}",
                details=checks,
            )

        # Check 2: 接近日线支撑
        checks['support'] = self._check_daily_support(
            current_price, daily_pivot_zg, daily_pivot_zd, signal_type,
        )
        if not checks['support']['pass']:
            return ConfirmResult(
                passed=False, confidence=0.0, stop_loss=stop_loss,
                reason=f"远离日线支撑: {checks['support']['reason']}",
                details=checks,
            )

        # Check 3: MACD支持 (scored)
        checks['macd'] = self._check_macd(analysis)

        # Check 4: 5min超涨过滤
        checks['overextended'] = self._check_5min_overextended(code)

        # Check 5: 结构质量 (scored)
        checks['structure'] = self._check_structure_quality(
            analysis, daily_pivot_zd, daily_pivot_zg,
        )

        # 5min超涨否决
        if checks['overextended'].get('has_sell'):
            return ConfirmResult(
                passed=False, confidence=0.0, stop_loss=stop_loss,
                reason=f"5min超涨: {checks['overextended'].get('reason', '')}",
                details=checks,
            )

        # 综合判断: MACD或结构质量至少一个正面
        macd_ok = checks['macd']['score'] >= 0.3
        struct_ok = checks['structure']['score'] >= 0.3
        if not macd_ok and not struct_ok:
            return ConfirmResult(
                passed=False, confidence=0.0, stop_loss=stop_loss,
                reason='MACD和结构均不支持',
                details=checks,
            )

        # 计算置信度
        conf = 0.40
        conf += checks['macd']['score'] * 0.20
        conf += checks['structure']['score'] * 0.20
        if checks['support'].get('strong'):
            conf += 0.10
        vol_info = checks.get('volume', {})
        if vol_info.get('contraction'):
            conf += 0.05
        conf = min(0.95, max(0.35, conf))

        # 止损: 取日线止损和30min结构止损中较高的
        struct_stop = checks['structure'].get('stop_price', 0)
        if struct_stop > 0:
            stop_loss = max(stop_loss, struct_stop)

        reason_parts = []
        if macd_ok:
            reason_parts.append('MACD支持')
        if struct_ok:
            reason_parts.append('结构确认')
        if checks['support'].get('strong'):
            reason_parts.append('强支撑')
        reason = ' + '.join(reason_parts) if reason_parts else '基础确认'

        return ConfirmResult(
            passed=True, confidence=conf, stop_loss=float(stop_loss),
            reason=reason, details=checks,
        )

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
            low_s = pd.Series(df['low'].values)
            vol_s = pd.Series(df['volume'].values) if 'volume' in df.columns else None
            macd = MACD(close_s)

            kline = KLine.from_dataframe(df, strict_mode=False)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 4:
                return None

            strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
            if len(strokes) < 3:
                return None

            pivots = PivotDetector(kline, strokes).get_pivots()

            result = {
                'kline': kline,
                'strokes': strokes,
                'pivots': pivots,
                'fractals': fractals,
                'macd': macd,
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

    def _check_pullback_state(self, analysis: dict) -> dict:
        """Check 1: 最近笔是否在回调中"""
        strokes = analysis['strokes']
        if not strokes:
            return {'pass': False, 'reason': '无笔数据'}

        last = strokes[-1]
        is_down = last.end_value < last.start_value

        if is_down:
            return {'pass': True, 'reason': '当前向下笔(回调中)'}

        if len(strokes) >= 2:
            prev = strokes[-2]
            prev_down = prev.end_value < prev.start_value
            if prev_down:
                return {'pass': True, 'reason': '前一笔向下(回调刚结束)'}

        return {'pass': False, 'reason': '连续向上，无回调'}

    def _check_daily_support(self, current_price: float,
                             daily_zg: float, daily_zd: float,
                             signal_type: str) -> dict:
        """Check 2: 价格是否在日线支撑位附近"""
        if daily_zd <= 0 and daily_zg <= 0:
            return {'pass': True, 'reason': '无日线中枢数据，放行', 'strong': False}

        if signal_type == '3buy' and daily_zg > 0:
            if current_price >= daily_zg * 0.98:
                return {'pass': True, 'strong': True,
                        'reason': f'价格{current_price:.2f}在中枢ZG{daily_zg:.2f}之上'}
            if current_price >= daily_zd * 0.97:
                return {'pass': True, 'strong': False,
                        'reason': f'价格{current_price:.2f}在中枢ZD{daily_zd:.2f}附近'}
            return {'pass': False, 'strong': False,
                    'reason': f'价格{current_price:.2f}低于中枢ZD{daily_zd:.2f}*0.97'}

        if daily_zd > 0:
            if current_price >= daily_zd * 0.95:
                return {'pass': True, 'strong': current_price >= daily_zd,
                        'reason': '价格在中枢ZD附近'}

        return {'pass': True, 'reason': '常规信号放行', 'strong': False}

    def _check_macd(self, analysis: dict) -> dict:
        """Check 3: MACD动能支持 (scored 0-1)"""
        macd = analysis['macd']
        latest = macd.get_latest()
        if not latest:
            return {'score': 0.0, 'reason': '无MACD数据'}

        score = 0.0
        reasons = []

        if latest.macd > latest.signal:
            score += 0.4
            reasons.append('DIF>DEA')

        hist_series = macd.get_histogram_series()
        if len(hist_series) >= 2:
            hist_now = float(hist_series.iloc[-1])
            hist_prev = float(hist_series.iloc[-2])
            if hist_now <= 0 and hist_now > hist_prev:
                score += 0.3
                reasons.append('绿柱缩短')

        dif_series = macd.get_dif_series()
        if len(dif_series) >= 2:
            if float(dif_series.iloc[-1]) > float(dif_series.iloc[-2]):
                score += 0.3
                reasons.append('DIF递增')

        return {'score': min(1.0, score), 'reason': '+'.join(reasons) or '无确认'}

    def _check_5min_overextended(self, code: str) -> dict:
        """Check 4: 5min超涨检测"""
        try:
            from strategies.v3a_30min_strategy import V3a30MinStrategy, V3aConfig
            cfg = V3aConfig(enable_5min_filter=True)
            v3a = V3a30MinStrategy(cfg, self.hs)
            result = v3a._check_5min_filter(code)
            if result is None:
                return {'has_sell': False, 'reason': '5min数据不足'}
            return {
                'has_sell': result.get('has_sell', False),
                'reason': result.get('reason', ''),
                'stop_5min': result.get('stop_5min', 0),
            }
        except Exception:
            return {'has_sell': False, 'reason': '5min检查失败'}

    def _check_structure_quality(self, analysis: dict,
                                  daily_zd: float, daily_zg: float) -> dict:
        """Check 5: 30min结构质量 (scored 0-1)"""
        strokes = analysis['strokes']
        fractals = analysis['fractals']
        score = 0.0
        stop_price = 0.0

        recent_n = min(15, len(strokes))
        recent_strokes = strokes[-recent_n:]
        down_strokes = [s for s in recent_strokes if s.end_value < s.start_value]

        if down_strokes:
            score += 0.3
            last_down = down_strokes[-1]
            last_down_low = min(last_down.start_value, last_down.end_value)
            if daily_zd > 0 and last_down_low >= daily_zd * 0.98:
                score += 0.3
            if daily_zg > 0 and last_down_low >= daily_zg * 0.98:
                score += 0.2
            stop_price = last_down_low

        if fractals:
            from core.fractal import FractalType
            bottom_fractals = [f for f in fractals[-10:] if f.type == FractalType.BOTTOM]
            if bottom_fractals and analysis['klen'] - bottom_fractals[-1].index <= 5:
                score += 0.2

        vol_s = analysis.get('volume_series')
        if vol_s is not None and len(vol_s) >= 20 and down_strokes:
            last_down = down_strokes[-1]
            pb_start = max(0, last_down.start_index - 2)
            pb_end = min(len(vol_s) - 1, last_down.end_index + 2)
            if pb_end > pb_start:
                pb_vol = float(vol_s.iloc[pb_start:pb_end + 1].mean())
                up_start = max(0, pb_start - 15)
                up_end = pb_start
                if up_end > up_start:
                    up_vol = float(vol_s.iloc[up_start:up_end].mean())
                    if up_vol > 0 and pb_vol < up_vol * 0.85:
                        score += 0.1

        return {'score': min(1.0, score), 'stop_price': stop_price}
