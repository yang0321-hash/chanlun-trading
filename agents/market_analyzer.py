"""
大盘多周期缠论分析器

在委员会评估前，对创业板指和上证指数做多级别缠论分析，
输出市场状态(regime/phase/risk_premium)注入到 CommitteeContext。

参考: 多级别联立 — 日线定位 → 30min找结构 → 综合判断
"""

import os
import re
import json
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from core.kline import KLine
from core.fractal import detect_fractals
from core.stroke import generate_strokes
from core.pivot import detect_pivots, PivotLevel
from indicator.macd import MACD


@dataclass
class IndexAnalysis:
    """单指数分析结果"""
    name: str
    code: str
    close: float
    ma5: float = 0
    ma10: float = 0
    ma20: float = 0
    ret_5d: float = 0
    ret_20d: float = 0
    high_20d: float = 0
    low_20d: float = 0
    position_pct: float = 0  # 当前价在20日高低之间的分位
    ma_trend: str = 'mixed'  # bullish/bearish/mixed
    ma5_trend_score: int = 1  # 0=连续3日下降 1=拐头/持平 2=连续3日上升
    # 缠论
    daily_pivot_zg: float = 0
    daily_pivot_zd: float = 0
    daily_strokes_count: int = 0
    pivot_position: str = 'inside'  # above/inside/below
    # 30min
    min30_strokes_count: int = 0
    min30_pivot_zg: float = 0
    min30_pivot_zd: float = 0
    min30_macd_hist: float = 0
    min30_golden_cross: bool = False
    min30_death_cross: bool = False
    # 综合判断
    phase: str = 'unknown'  # 上涨趋势/高位震荡/下跌趋势/低位反弹/收敛变盘
    v2_score: int = 6       # TradingRules 12分制评分


@dataclass
class MarketContext:
    """大盘多周期分析结果 — 注入到 CommitteeContext"""
    regime: str = 'normal'  # strong/normal/weak/danger
    index_phase: str = 'unknown'  # 上涨趋势/高位震荡/下跌趋势/低位反弹/收敛变盘
    risk_premium: float = 0.0  # -0.3 ~ +0.3 (正值=额外惩罚)
    position_adjust: float = 1.0  # 0.5~1.5 (>1放大, <1收缩)
    key_levels: Dict[str, Dict] = field(default_factory=dict)
    stroke_summary: str = ''
    warnings: List[str] = field(default_factory=list)
    indices: Dict[str, IndexAnalysis] = field(default_factory=dict)


class MarketAnalyzer:
    """大盘多周期缠论分析 — 委员会前置"""

    INDICES = {
        'cyb': {'code': 'sz399006', 'name': '创业板指'},
        'sh': {'code': 'sh000001', 'name': '上证指数'},
    }

    def analyze(self) -> MarketContext:
        """主入口: 分析大盘并返回 MarketContext"""
        analyses = {}
        for key, info in self.INDICES.items():
            try:
                idx = self._analyze_index(info['code'], info['name'])
                analyses[key] = idx
            except Exception as e:
                print(f'[MarketAnalyzer] {info["name"]} analysis failed: {e}')

        if not analyses:
            return MarketContext()

        regime = self._classify_regime(analyses)
        phase = self._classify_phase(analyses)
        risk_premium = self._calc_risk_premium(regime, phase, analyses)
        position_adjust = self._calc_position_adjust(regime, phase)
        warnings = self._generate_warnings(analyses, regime, phase)
        key_levels = self._extract_key_levels(analyses)
        summary = self._build_stroke_summary(analyses)

        ctx = MarketContext(
            regime=regime,
            index_phase=phase,
            risk_premium=risk_premium,
            position_adjust=position_adjust,
            key_levels=key_levels,
            stroke_summary=summary,
            warnings=warnings,
            indices=analyses,
        )
        return ctx

    def _fetch_sina_kline(self, symbol: str, scale: int = 240, count: int = 120) -> Optional[pd.DataFrame]:
        """从Sina获取K线数据"""
        session = requests.Session()
        session.trust_env = False
        url = (f'https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/'
               f'CN_MarketDataService.getKLineData?symbol={symbol}'
               f'&scale={scale}&ma=no&datalen={count}')
        resp = session.get(url, timeout=15)
        match = re.search(r'callback\((.*)\)', resp.text)
        if not match:
            return None
        klines = json.loads(match.group(1))
        if not klines:
            return None
        df = pd.DataFrame(klines)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df['date'] = pd.to_datetime(df['day'])
        df.set_index('date', inplace=True)
        return df

    def _analyze_index(self, code: str, name: str) -> IndexAnalysis:
        """单指数多周期分析"""
        idx = IndexAnalysis(name=name, code=code, close=0)

        # === 日线 ===
        df = self._fetch_sina_kline(code, scale=240, count=120)
        if df is None or len(df) < 60:
            return idx

        close = df['close']
        idx.close = float(close.iloc[-1])
        idx.ma5 = float(close.rolling(5).mean().iloc[-1])
        idx.ma10 = float(close.rolling(10).mean().iloc[-1])
        idx.ma20 = float(close.rolling(20).mean().iloc[-1])
        idx.ret_5d = float(close.iloc[-1] / close.iloc[-5] - 1)
        idx.ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1)
        idx.high_20d = float(df['high'].iloc[-20:].max())
        idx.low_20d = float(df['low'].iloc[-20:].min())

        h_range = idx.high_20d - idx.low_20d
        idx.position_pct = (idx.close - idx.low_20d) / h_range if h_range > 0 else 0.5

        # MA趋势
        if idx.ma5 > idx.ma10 > idx.ma20:
            idx.ma_trend = 'bullish'
        elif idx.ma5 < idx.ma10 < idx.ma20:
            idx.ma_trend = 'bearish'
        else:
            idx.ma_trend = 'mixed'

        # MA5趋势方向 (连续3日变化)
        ma5_series = close.rolling(5).mean()
        if len(ma5_series) >= 3:
            last3 = ma5_series.iloc[-3:].values
            diffs = [last3[i+1] - last3[i] for i in range(2)]
            if all(d > 0 for d in diffs):
                idx.ma5_trend_score = 2
            elif all(d < 0 for d in diffs):
                idx.ma5_trend_score = 0
            else:
                idx.ma5_trend_score = 1

        # 日线缠论
        try:
            kline = KLine.from_dataframe(df)
            fractals = detect_fractals(kline)
            strokes = generate_strokes(kline, fractals)
            pivots = detect_pivots(kline, strokes, level=PivotLevel.DAY)
            idx.daily_strokes_count = len(strokes)

            if pivots:
                p = pivots[-1]
                idx.daily_pivot_zg = p.zg
                idx.daily_pivot_zd = p.zd
                if idx.close > p.zg:
                    idx.pivot_position = 'above'
                elif idx.close < p.zd:
                    idx.pivot_position = 'below'
                else:
                    idx.pivot_position = 'inside'
        except Exception:
            pass

        # === 30min ===
        try:
            df30 = self._fetch_sina_kline(code, scale=30, count=80)
            if df30 is not None and len(df30) >= 20:
                kline30 = KLine.from_dataframe(df30)
                fr30 = detect_fractals(kline30)
                st30 = generate_strokes(kline30, fr30)
                pv30 = detect_pivots(kline30, st30, level=PivotLevel.DAY)
                macd30 = MACD(pd.Series([k.close for k in kline30]))

                idx.min30_strokes_count = len(st30)
                if pv30:
                    idx.min30_pivot_zg = pv30[-1].zg
                    idx.min30_pivot_zd = pv30[-1].zd

                lm30 = macd30.get_latest()
                if lm30:
                    idx.min30_macd_hist = lm30.histogram
                idx.min30_golden_cross = macd30.check_golden_cross()
                idx.min30_death_cross = macd30.check_death_cross()
        except Exception:
            pass

        # 综合判断phase
        idx.phase = self._classify_index_phase(idx)

        # v2.0评分 (12分制，供规则引擎使用)
        try:
            from strategies.trading_rules import TradingRules
            result = TradingRules.calc_market_score(close.values)
            idx.v2_score = result.score
        except Exception:
            pass

        return idx

    def _classify_index_phase(self, idx: IndexAnalysis) -> str:
        """单指数阶段判断 (6因子，满分+8/-8)"""
        score = 0

        # 因子1: MA趋势排列 (+2/-2)
        if idx.ma_trend == 'bullish':
            score += 2
        elif idx.ma_trend == 'bearish':
            score -= 2

        # 因子2: MA5趋势方向 (+2/0)
        #   2=连续3日上升, 1=拐头/持平, 0=连续3日下降
        if idx.ma5_trend_score == 2:
            score += 2
        elif idx.ma5_trend_score == 0:
            score -= 0  # 下降不额外惩罚，但不得分

        # 因子3: 涨幅贡献 (+2/-2)
        if idx.ret_20d > 0.08:
            score += 2
        elif idx.ret_20d > 0.03:
            score += 1
        elif idx.ret_20d < -0.08:
            score -= 2
        elif idx.ret_20d < -0.03:
            score -= 1

        # 因子4: 位置分位 (+1/-1)
        if idx.position_pct > 0.8:
            score += 1
        elif idx.position_pct < 0.2:
            score -= 1

        # 因子5: 中枢位置 (+1/-1)
        if idx.pivot_position == 'above':
            score += 1
        elif idx.pivot_position == 'below':
            score -= 1

        if score >= 4:
            return '上涨趋势'
        elif score >= 2:
            if idx.position_pct > 0.7:
                return '高位震荡'
            return '上涨趋势'
        elif score >= 0:
            return '收敛变盘'
        elif score >= -2:
            if idx.position_pct < 0.3:
                return '低位反弹'
            return '下跌趋势'
        else:
            return '下跌趋势'

    def _classify_regime(self, analyses: Dict[str, IndexAnalysis]) -> str:
        """综合多指数判断市场regime"""
        scores = []
        for idx in analyses.values():
            if idx.phase == '上涨趋势':
                scores.append(3)
            elif idx.phase == '高位震荡':
                scores.append(2)
            elif idx.phase == '收敛变盘':
                scores.append(1)
            elif idx.phase == '低位反弹':
                scores.append(0)
            elif idx.phase == '下跌趋势':
                scores.append(-1)
            else:
                scores.append(1)

        avg = sum(scores) / len(scores) if scores else 1

        if avg >= 2.5:
            return 'strong'
        elif avg >= 1.0:
            return 'normal'
        elif avg >= 0:
            return 'normal'
        else:
            return 'weak'

    def _classify_phase(self, analyses: Dict[str, IndexAnalysis]) -> str:
        """综合多指数判断phase"""
        phases = [idx.phase for idx in analyses.values()]
        # 取最谨慎的判断
        priority = {'下跌趋势': 0, '低位反弹': 1, '收敛变盘': 2, '高位震荡': 3, '上涨趋势': 4}
        # 如果任一指数是高位震荡，整体标记为高位震荡
        if '高位震荡' in phases:
            return '高位震荡'
        if '下跌趋势' in phases:
            return '下跌趋势'
        # 取优先级最高的
        return max(phases, key=lambda p: priority.get(p, 2)) if phases else 'unknown'

    def _calc_risk_premium(self, regime: str, phase: str, analyses: Dict) -> float:
        """风险溢价: 正值=需额外惩罚, 负值=可放宽"""
        premium = 0.0

        # phase贡献
        phase_premiums = {
            '上涨趋势': -0.15,
            '高位震荡': 0.20,
            '下跌趋势': 0.25,
            '低位反弹': -0.05,
            '收敛变盘': 0.05,
            'unknown': 0.0,
        }
        premium += phase_premiums.get(phase, 0.0)

        # regime贡献
        if regime == 'strong':
            premium -= 0.05
        elif regime == 'weak':
            premium += 0.10

        # 高位分位额外惩罚
        for idx in analyses.values():
            if idx.position_pct > 0.85:
                premium += 0.05
            if idx.pivot_position == 'above' and idx.close > idx.daily_pivot_zg * 1.1:
                premium += 0.05  # 远离中枢上方

        return max(-0.30, min(0.30, premium))

    def _calc_position_adjust(self, regime: str, phase: str) -> float:
        """仓位调整系数"""
        adjusts = {
            ('strong', '上涨趋势'): 1.3,
            ('strong', '高位震荡'): 0.8,
            ('strong', '下跌趋势'): 0.5,
            ('normal', '上涨趋势'): 1.1,
            ('normal', '高位震荡'): 0.8,
            ('normal', '收敛变盘'): 0.9,
            ('normal', '下跌趋势'): 0.6,
            ('normal', '低位反弹'): 1.0,
            ('weak', '上涨趋势'): 0.9,
            ('weak', '高位震荡'): 0.5,
            ('weak', '下跌趋势'): 0.4,
            ('weak', '低位反弹'): 0.7,
        }
        return adjusts.get((regime, phase), 1.0)

    def _generate_warnings(self, analyses: Dict, regime: str, phase: str) -> List[str]:
        """生成风险警告"""
        warnings = []

        for key, idx in analyses.items():
            if idx.position_pct > 0.85:
                warnings.append(f'{idx.name}处于20日高位区域(分位{idx.position_pct:.0%})')
            if idx.pivot_position == 'above' and idx.daily_pivot_zg > 0:
                above_pct = (idx.close - idx.daily_pivot_zg) / idx.daily_pivot_zg
                if above_pct > 0.1:
                    warnings.append(f'{idx.name}远离中枢上方{above_pct:.1%}，追高风险大')
            if idx.min30_death_cross:
                warnings.append(f'{idx.name}30min死叉')
            if idx.ret_5d < -0.05:
                warnings.append(f'{idx.name}近5日跌{idx.ret_5d:.1%}')

        if regime == 'weak':
            warnings.append('大盘弱势环境，建议收缩仓位')
        if phase == '高位震荡':
            warnings.append('大盘高位震荡，追高风险大于收益')

        return warnings

    def _extract_key_levels(self, analyses: Dict) -> Dict[str, Dict]:
        """提取各指数关键位"""
        levels = {}
        for key, idx in analyses.items():
            levels[idx.name] = {
                'close': idx.close,
                'support': idx.daily_pivot_zg if idx.pivot_position == 'above' else idx.daily_pivot_zd,
                'resistance': idx.high_20d,
                'pivot_zg': idx.daily_pivot_zg,
                'pivot_zd': idx.daily_pivot_zd,
                'position_pct': round(idx.position_pct, 2),
            }
        return levels

    def _build_stroke_summary(self, analyses: Dict) -> str:
        """构建笔结构文字摘要"""
        parts = []
        for key, idx in analyses.items():
            parts.append(
                f"{idx.name}: {idx.phase}, MA{'多头' if idx.ma_trend == 'bullish' else '空头' if idx.ma_trend == 'bearish' else '混合'}, "
                f"中枢{'上方' if idx.pivot_position == 'above' else '内' if idx.pivot_position == 'inside' else '下方'}, "
                f"5日{idx.ret_5d:+.1%}"
            )
        return ' | '.join(parts)
