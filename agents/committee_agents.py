"""
投资委员会 — 6个Agent实现

Agent列表:
  1. BullAnalyst    — 看多分析师（缠论买信号+趋势+量价确认+自适应评分）
  2. BearAnalyst    — 看空分析师（风险信号+背离+高位警告+过滤器链）
  3. SentimentAnalyzer — 市场情绪分析（纯价格/量计算，无API）
  4. SectorRotation — 行业轮动评估（动量+成长性+资金流）
  5. RiskManager    — 风控经理（ATR止损+仓位+集中度）
  6. FundManager    — 基金经理（组合决策+约束+最终评分）
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from agents.debate_system import AgentArgument
from agents.scoring import (
    normalize, classify_risk, calc_position_size,
    DEFAULT_WEIGHTS, DECISION_THRESHOLDS, VETO_RULES,
)

# 可选集成: AdaptiveSignalScorer 和 Filter 链
try:
    from strategies.scoring import AdaptiveSignalScorer, ScoringFactors, ScoringConfig
    from strategies.scoring.regime_detector import MarketRegimeDetector
    SCORING_AVAILABLE = True
except ImportError:
    SCORING_AVAILABLE = False

try:
    from strategies.filters import (
        VolumeFilter, RegimeFilter, CooldownFilter,
        TrendAlignmentFilter, CompositeFilter,
    )
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False


# ============================================================
# 共享数据结构
# ============================================================

@dataclass
class ChanlunInfo:
    """缠论结构信息（由扫描器或实时计算提供）"""
    pivot_zg: float = 0.0       # 中枢上沿
    pivot_zd: float = 0.0       # 中枢下沿
    pivot_gg: float = 0.0       # 中枢波动高点
    pivot_dd: float = 0.0       # 中枢波动低点
    buy_type: str = ''           # 最近买点类型: '1buy', '2buy', '3buy', 'quasi2buy', 'quasi3buy'
    buy_price: float = 0.0      # 买点价格
    buy_date: str = ''           # 买点日期
    strokes_count: int = 0       # 笔数
    price_vs_pivot: str = ''     # 'above'(中枢上方), 'inside'(中枢内), 'below'(中枢下方)
    divergence_detected: bool = False  # 是否有背驰信号
    stop_by_structure: float = 0.0     # 缠论结构止损位


@dataclass
class CommitteeContext:
    """委员会共享上下文"""
    symbol: str
    name: str
    sector: str
    df_daily: pd.DataFrame
    df_30min: Optional[pd.DataFrame] = None
    entry_price: float = 0.0
    stop_price: float = 0.0
    scanner_score: float = 0.0
    risk_reward: float = 0.0
    sector_momentum: Dict[str, float] = field(default_factory=dict)
    sector_map: Dict[str, str] = field(default_factory=dict)
    portfolio_state: Dict[str, Any] = field(default_factory=dict)
    chanlun: Optional[ChanlunInfo] = None  # 缠论结构信息


@dataclass
class RiskAssessment:
    """风控评估结果"""
    risk_score: float = 0.0           # 0-1
    risk_level: str = 'MEDIUM'        # LOW/MEDIUM/HIGH/EXTREME
    position_pct: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    max_shares: int = 0
    risk_factors: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    sector_concentration: float = 0.0  # 同行业仓位占比


# ============================================================
# 成长性行业列表 (与 scan_enhanced_v3.py 保持一致)
# ============================================================

GROWTH_SECTORS = {
    '专用设备', '能源金属', '航天军工', '电池', '半导体',
    '电气设备', '电力设备', '军工电子', '航空装备',
    '光伏设备', '风电设备', '锂电', '新能源',
    '汽车零部件', '消费电子', '自动化设备',
}


# ============================================================
# 缠论结构分析辅助
# ============================================================

def analyze_chanlun_structure(df: pd.DataFrame, candidate: Dict = None) -> Optional[ChanlunInfo]:
    """
    从日线数据计算缠论结构信息，包含标准买卖点和类买卖点检测

    Returns: ChanlunInfo or None (数据不足时)
    """
    if len(df) < 60:
        return None

    try:
        from core.kline import KLine
        from core.fractal import detect_fractals
        from core.stroke import generate_strokes
        from core.pivot import detect_pivots, PivotLevel
        from core.buy_sell_points import BuySellPointDetector
        from indicator.macd import MACD

        # K线处理
        kline = KLine.from_dataframe(df)

        # 分型检测
        fractals = detect_fractals(kline)

        # 笔检测
        strokes = generate_strokes(kline, fractals)

        if len(strokes) < 5:
            return None

        # 中枢检测
        pivots = detect_pivots(kline, strokes, level=PivotLevel.DAY)

        if not pivots:
            return None

        # MACD
        macd = MACD(df['close'])

        # === 买卖点检测（包含1买/2买/3买/类2买/类3买） ===
        buy_points = []
        sell_points = []
        try:
            detector = BuySellPointDetector(
                fractals=fractals,
                strokes=strokes,
                segments=[],  # 日线级别不需要线段
                pivots=pivots,
                macd=macd,
            )
            buy_points, sell_points = detector.detect_all()
        except Exception:
            pass

        # 取最近的中枢
        last_pivot = pivots[-1]
        last_close = float(df['close'].iloc[-1])

        # 判断价格相对中枢位置
        if last_close > last_pivot.zg:
            price_pos = 'above'
        elif last_close < last_pivot.zd:
            price_pos = 'below'
        else:
            price_pos = 'inside'

        # 结构止损: 中枢下沿(ZD) 或 最近向下笔低点
        stop_by_structure = last_pivot.zd
        if len(strokes) >= 2:
            for s in reversed(strokes):
                if s.is_down:
                    if s.end_value < stop_by_structure:
                        stop_by_structure = s.end_value
                    break

        # === 找最近的买点 ===
        # 优先级: 1买 > 2买 > 3买 > 类2买 > 类3买
        buy_type_priority = {'1buy': 5, '2buy': 4, '3buy': 3, 'quasi2buy': 2, 'quasi3buy': 1}
        best_buy = None
        best_priority = 0

        # 先从BuySellPointDetector结果中找（包含类2买/类3买）
        # 放宽到最近120个交易日
        for bp in buy_points:
            if bp.index < len(df) - 120:
                continue
            p = buy_type_priority.get(bp.point_type, 0)
            if p > best_priority or (p == best_priority and best_buy and bp.index > best_buy.index):
                best_priority = p
                best_buy = bp

        # 再看扫描器提供的2买信息（可能有更近期的信号）
        scan_buy_type = ''
        scan_buy_price = 0.0
        scan_buy_date = ''
        if candidate:
            scan_buy_date = candidate.get('2buy_date', '')
            if scan_buy_date:
                scan_buy_type = '2buy'
                scan_buy_price = candidate.get('entry_price', 0)

        # 确定最终使用的买点：扫描器优先（它的引擎更贴近实战）
        buy_type = scan_buy_type
        buy_price = scan_buy_price
        buy_date = scan_buy_date
        if not buy_type and best_buy:
            buy_type = best_buy.point_type
            buy_price = best_buy.price
            buy_date = str(df.index[best_buy.index].date()) if best_buy.index < len(df) else ''

        # 如果检测器给了买点的止损位，优先使用
        if best_buy and best_buy.stop_loss > 0:
            stop_by_structure = max(stop_by_structure, best_buy.stop_loss)

        # MACD背驰检测（从买点中获取，更精确）
        divergence = False
        if best_buy and best_buy.divergence_ratio > 0:
            divergence = True
        elif len(strokes) >= 4:
            # 手动计算MACD用于背驰检测
            hist_series = pd.Series([v.histogram for v in macd.values],
                                     index=range(len(macd.values)))
            down_strokes = [s for s in strokes if s.is_down]
            if len(down_strokes) >= 2:
                s1, s2 = down_strokes[-2], down_strokes[-1]
                if s1.end_index < len(hist_series) and s2.end_index < len(hist_series):
                    area1 = abs(hist_series.iloc[s1.start_index:s1.end_index+1].sum())
                    area2 = abs(hist_series.iloc[s2.start_index:s2.end_index+1].sum())
                    if area1 > 0 and area2 < area1 * 0.7 and s2.end_value < s1.end_value:
                        divergence = True

        info = ChanlunInfo(
            pivot_zg=last_pivot.zg,
            pivot_zd=last_pivot.zd,
            pivot_gg=last_pivot.gg,
            pivot_dd=last_pivot.dd,
            buy_type=buy_type,
            buy_price=buy_price,
            buy_date=buy_date,
            strokes_count=len(strokes),
            price_vs_pivot=price_pos,
            divergence_detected=divergence,
            stop_by_structure=stop_by_structure,
        )
        return info

    except Exception:
        return None


# ============================================================
# 1. BullAnalyst — 看多分析师
# ============================================================

class BullAnalyst:
    """增强版看多分析师 — 集成缠论结构分析 + 自适应评分器"""

    def __init__(self):
        self.scorer = None
        if SCORING_AVAILABLE:
            self.scorer = AdaptiveSignalScorer()

    def analyze(self, ctx: CommitteeContext) -> AgentArgument:
        df = ctx.df_daily
        if len(df) < 20:
            return self._no_signal(ctx, '数据不足')

        key_points = []
        confidence = 0.3
        reasoning_parts = []
        last_close = float(df['close'].iloc[-1])

        # === 缠论结构分析（核心优先） ===
        cl = ctx.chanlun
        if cl:
            # 1. 买点类型评估
            if cl.buy_type == '1buy':
                key_points.append('1买信号(趋势底背驰)')
                confidence += 0.35
                reasoning_parts.append('1买底背驰')
            elif cl.buy_type == '2buy':
                key_points.append('2买信号(回调不破前低)')
                confidence += 0.30
                reasoning_parts.append('2买确认')
            elif cl.buy_type == '3buy':
                key_points.append('3买信号(突破回踩不进中枢)')
                confidence += 0.25
                reasoning_parts.append('3买突破确认')
            elif cl.buy_type in ('quasi2buy', 'quasi3buy'):
                key_points.append(f'类{cl.buy_type}信号')
                confidence += 0.15
                reasoning_parts.append(f'类{cl.buy_type}')

            # 2. 价格相对中枢位置
            if cl.price_vs_pivot == 'below':
                key_points.append('价格在中枢下方(低位机会)')
                confidence += 0.15
                reasoning_parts.append('中枢下方低位')
            elif cl.price_vs_pivot == 'inside':
                key_points.append('价格在中枢内部(盘整)')
                confidence += 0.05
                reasoning_parts.append('中枢内部盘整')
            elif cl.price_vs_pivot == 'above':
                # 中枢上方要看是不是刚突破（3买）还是已经远离
                if cl.pivot_zg > 0:
                    pct_above = (last_close - cl.pivot_zg) / cl.pivot_zg
                    if pct_above < 0.05:
                        key_points.append('刚突破中枢上沿')
                        confidence += 0.10
                        reasoning_parts.append('突破中枢上沿')
                    elif pct_above > 0.15:
                        key_points.append(f'远离中枢({pct_above:.0%}),追高风险')
                        confidence -= 0.10
                        reasoning_parts.append(f'远离中枢{pct_above:.0%}')

            # 3. 背驰检测
            if cl.divergence_detected:
                key_points.append('MACD底背驰确认')
                confidence += 0.10
                reasoning_parts.append('底背驰')

            # 4. 当前价vs买点价的偏离（关键！）
            if cl.buy_price > 0 and last_close > 0:
                pct_drift = (last_close - cl.buy_price) / cl.buy_price
                if pct_drift > 0.20:
                    # 远离买点20%以上，大幅降低置信度
                    key_points.append(f'偏离买点{pct_drift:.0%}(追高风险大)')
                    confidence -= 0.25
                    reasoning_parts.append(f'偏离买点{pct_drift:.0%}，非理想入场')
                elif pct_drift > 0.10:
                    key_points.append(f'偏离买点{pct_drift:.0%}')
                    confidence -= 0.10
                    reasoning_parts.append(f'偏离买点{pct_drift:.0%}')
                elif abs(pct_drift) <= 0.03:
                    key_points.append('价格贴近买点(理想入场区)')
                    confidence += 0.10
                    reasoning_parts.append('贴近买点')

            # 5. 结构止损位评估
            if cl.stop_by_structure > 0 and last_close > 0:
                stop_pct = (last_close - cl.stop_by_structure) / last_close
                if stop_pct < 0.03:
                    key_points.append(f'结构止损很近({stop_pct:.1%})')
                    confidence -= 0.05
                elif stop_pct > 0.10:
                    key_points.append(f'结构止损空间充裕({stop_pct:.1%})')
                    confidence += 0.05

        # --- 基础技术分析 (补充缠论) ---
        trend = self._analyze_trend(df)
        if trend['bullish']:
            key_points.append(trend['reason'])
            confidence += trend['bonus']
            reasoning_parts.append(trend['reason'])

        macd = self._analyze_macd(df)
        if macd['bullish']:
            key_points.append(macd['reason'])
            confidence += macd['bonus']
            reasoning_parts.append(macd['reason'])

        vol = self._analyze_volume(df)
        if vol['bullish']:
            key_points.append(vol['reason'])
            confidence += vol['bonus']
            reasoning_parts.append(vol['reason'])

        if ctx.risk_reward > 3:
            key_points.append(f'风险收益比优异: {ctx.risk_reward:.1f}')
            confidence += 0.1
        elif ctx.risk_reward > 1.5:
            confidence += 0.05

        if ctx.entry_price > 0:
            pct_from_entry = (last_close - ctx.entry_price) / ctx.entry_price
            if abs(pct_from_entry) < 0.03:
                key_points.append('价格接近入场点')
                confidence += 0.1

        if ctx.scanner_score >= 80:
            confidence += 0.05

        # --- 自适应评分器增强 ---
        if self.scorer and len(df) >= 60:
            advanced_score = self._advanced_score(df, ctx)
            if advanced_score > 0:
                key_points.append(f'高级评分: {advanced_score:.0%}')
                confidence = confidence * 0.7 + advanced_score * 0.3

        confidence = max(0.1, min(0.95, confidence))
        reasoning = ' | '.join(reasoning_parts) if reasoning_parts else '无明显看多信号'

        return AgentArgument(
            agent_name='BullAnalyst',
            stance='bull',
            reasoning=reasoning,
            confidence=confidence,
            key_points=key_points,
            data_references={'symbol': ctx.symbol, 'price': last_close},
        )

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        if len(df) < 60:
            current = df['close'].iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            bullish = current > ma20
            return {
                'bullish': bullish,
                'reason': f"价格{'高于' if bullish else '低于'}MA20",
                'bonus': 0.15 if bullish else 0,
            }
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]
        if ma5 > ma20 > ma60:
            return {'bullish': True, 'reason': '多头排列(MA5>MA20>MA60)', 'bonus': 0.25}
        elif ma5 > ma20:
            return {'bullish': True, 'reason': '短期多头(MA5>MA20)', 'bonus': 0.15}
        return {'bullish': False, 'reason': '均线无多头排列', 'bonus': 0}

    def _analyze_macd(self, df: pd.DataFrame) -> Dict:
        if len(df) < 35:
            return {'bullish': False, 'reason': '', 'bonus': 0}
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd = (dif - dea) * 2
        bullish_parts = []
        bonus = 0
        if macd.iloc[-1] > 0:
            bullish_parts.append('MACD红柱')
            bonus += 0.05
        if dif.iloc[-1] > dea.iloc[-1]:
            bullish_parts.append('DIF>DEA')
            bonus += 0.05
        if len(dif) >= 3 and dif.iloc[-1] > dif.iloc[-3]:
            bullish_parts.append('DIF上升')
            bonus += 0.05
        if bullish_parts:
            return {'bullish': True, 'reason': '+'.join(bullish_parts), 'bonus': bonus}
        return {'bullish': False, 'reason': '', 'bonus': 0}

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {'bullish': False, 'reason': '', 'bonus': 0}
        vol = df['volume'].values.astype(float)
        vol_ma5 = np.mean(vol[-5:])
        vol_ma20 = np.mean(vol[-20:])
        price_up = df['close'].iloc[-1] > df['close'].iloc[-5]
        vol_expand = vol_ma20 > 0 and vol_ma5 > vol_ma20 * 1.2
        if vol_expand and price_up:
            return {'bullish': True, 'reason': '放量上涨', 'bonus': 0.15}
        elif price_up:
            return {'bullish': True, 'reason': '价格上行', 'bonus': 0.08}
        return {'bullish': False, 'reason': '', 'bonus': 0}

    def _advanced_score(self, df: pd.DataFrame, ctx: CommitteeContext) -> float:
        """使用自适应评分器计算高级评分"""
        try:
            # 计算量价确认
            vol_adj = 0.0
            vol_ratio = 1.0
            if len(df) >= 20:
                vol = df['volume'].values.astype(float)
                vol_ma5 = np.mean(vol[-5:])
                vol_ma20 = np.mean(vol[-20:])
                if vol_ma20 > 0:
                    vol_ratio = vol_ma5 / vol_ma20
                    if vol_ratio > 1.2 and df['close'].iloc[-1] > df['close'].iloc[-5]:
                        vol_adj = 0.10
                    elif vol_ratio < 0.7:
                        vol_adj = -0.10

            # 市场状态检测（lazy init，因为需要df）
            regime_info = None
            if SCORING_AVAILABLE:
                try:
                    detector = MarketRegimeDetector(df)
                    regime_info = detector.detect(df)
                except Exception:
                    pass

            # 构建评分因子
            factors = ScoringFactors(
                volume_adjustment=vol_adj,
                volume_ratio=vol_ratio,
                regime_info=regime_info,
                point_type='2buy',
                trend_status='up' if df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1] else 'down',
            )

            score, _ = self.scorer.score_buy_signal(factors)
            return float(score)
        except Exception:
            return 0.0

    def _no_signal(self, ctx: CommitteeContext, reason: str) -> AgentArgument:
        return AgentArgument(
            agent_name='BullAnalyst', stance='bull',
            reasoning=reason, confidence=0.1,
            key_points=[], data_references={'symbol': ctx.symbol},
        )


# ============================================================
# 2. BearAnalyst — 看空分析师
# ============================================================

class BearAnalyst:
    """增强版看空分析师 — 集成缠论结构风险"""

    def analyze(self, ctx: CommitteeContext) -> AgentArgument:
        df = ctx.df_daily
        if len(df) < 20:
            return self._no_signal(ctx)

        key_points = []
        confidence = 0.2  # 基础值（默认不太看空）
        reasoning_parts = []
        last_close = float(df['close'].iloc[-1])

        # === 缠论结构风险 ===
        cl = ctx.chanlun
        if cl:
            # 1. 远离中枢（追高风险）
            if cl.price_vs_pivot == 'above' and cl.pivot_zg > 0:
                pct_above = (last_close - cl.pivot_zg) / cl.pivot_zg
                if pct_above > 0.15:
                    key_points.append(f'远离中枢上方{pct_above:.0%}(严重追高)')
                    confidence += 0.30
                    reasoning_parts.append(f'远离中枢{pct_above:.0%}')
                elif pct_above > 0.08:
                    key_points.append(f'中枢上方{pct_above:.0%}(偏高)')
                    confidence += 0.15
                    reasoning_parts.append(f'中枢上方{pct_above:.0%}')

            # 2. 无明确买点信号
            if not cl.buy_type:
                key_points.append('无缠论买点信号')
                confidence += 0.10
                reasoning_parts.append('无买点')

            # 3. 偏离买点过远
            if cl.buy_price > 0 and last_close > 0:
                pct_drift = (last_close - cl.buy_price) / cl.buy_price
                if pct_drift > 0.25:
                    key_points.append(f'偏离买点{pct_drift:.0%}(严重追高)')
                    confidence += 0.25
                    reasoning_parts.append(f'偏离买点{pct_drift:.0%}')

            # 4. 3买尚未形成
            if cl.price_vs_pivot == 'above' and not cl.buy_type:
                key_points.append('3买尚未确认(在中枢上方但无回踩)')
                confidence += 0.15
                reasoning_parts.append('3买未确认')

            # 5. 止损空间过小
            if cl.stop_by_structure > 0 and last_close > 0:
                stop_pct = (last_close - cl.stop_by_structure) / last_close
                if stop_pct < 0.02:
                    key_points.append(f'结构止损极窄({stop_pct:.1%})')
                    confidence += 0.15
                    reasoning_parts.append('止损空间极小')

        # --- 基础风险分析 ---
        # 高位风险
        position = self._check_position(df)
        if position['risky']:
            key_points.append(position['reason'])
            confidence += position['bonus']
            reasoning_parts.append(position['reason'])

        # MACD顶背离迹象
        divergence = self._check_divergence(df)
        if divergence['bearish']:
            key_points.append(divergence['reason'])
            confidence += divergence['bonus']
            reasoning_parts.append(divergence['reason'])

        # 均线空头
        trend = self._check_trend(df)
        if trend['bearish']:
            key_points.append(trend['reason'])
            confidence += trend['bonus']
            reasoning_parts.append(trend['reason'])

        # 缩量警告
        vol = self._check_volume(df)
        if vol['bearish']:
            key_points.append(vol['reason'])
            confidence += vol['bonus']
            reasoning_parts.append(vol['reason'])

        # R/R比差
        if ctx.risk_reward < 0.5 and ctx.risk_reward >= 0:
            key_points.append(f'风险收益比差: {ctx.risk_reward:.1f}')
            confidence += 0.1

        # 入场价远高于当前价
        if ctx.entry_price > 0 and ctx.entry_price > last_close * 1.05:
            key_points.append('入场价远高于现价')
            confidence += 0.1

        confidence = max(0.1, min(0.95, confidence))
        reasoning = ' | '.join(reasoning_parts) if reasoning_parts else '无明显风险信号'

        return AgentArgument(
            agent_name='BearAnalyst',
            stance='bear',
            reasoning=reasoning,
            confidence=confidence,
            key_points=key_points,
            data_references={'symbol': ctx.symbol, 'price': last_close},
        )

    def _check_position(self, df: pd.DataFrame) -> Dict:
        if len(df) < 60:
            return {'risky': False, 'reason': '', 'bonus': 0}
        high_60 = df['high'].iloc[-60:].max()
        low_60 = df['low'].iloc[-60:].min()
        current = df['close'].iloc[-1]
        pct = (current - low_60) / (high_60 - low_60) if high_60 > low_60 else 0.5
        if pct > 0.85:
            return {'risky': True, 'reason': f'高位风险({pct:.0%}分位)', 'bonus': 0.25}
        elif pct > 0.70:
            return {'risky': True, 'reason': f'偏高位置({pct:.0%}分位)', 'bonus': 0.10}
        return {'risky': False, 'reason': '', 'bonus': 0}

    def _check_divergence(self, df: pd.DataFrame) -> Dict:
        if len(df) < 35:
            return {'bearish': False, 'reason': '', 'bonus': 0}
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd = (dif - dea) * 2
        # 简单背离检测: 价格创新高但MACD没有
        if len(df) >= 20:
            recent_high = df['high'].iloc[-10:].max()
            prev_high = df['high'].iloc[-20:-10].max()
            recent_macd_max = macd.iloc[-10:].max()
            prev_macd_max = macd.iloc[-20:-10].max()
            if recent_high > prev_high and recent_macd_max < prev_macd_max * 0.8:
                return {'bearish': True, 'reason': 'MACD顶背离迹象', 'bonus': 0.20}
        return {'bearish': False, 'reason': '', 'bonus': 0}

    def _check_trend(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {'bearish': False, 'reason': '', 'bonus': 0}
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        current = df['close'].iloc[-1]
        if current < ma5 < ma20:
            return {'bearish': True, 'reason': '均线空头排列', 'bonus': 0.20}
        elif current < ma5:
            return {'bearish': True, 'reason': '跌破MA5', 'bonus': 0.10}
        return {'bearish': False, 'reason': '', 'bonus': 0}

    def _check_volume(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {'bearish': False, 'reason': '', 'bonus': 0}
        vol = df['volume'].values.astype(float)
        vol_ma5 = np.mean(vol[-5:])
        vol_ma20 = np.mean(vol[-20:])
        price_up = df['close'].iloc[-1] > df['close'].iloc[-5]
        if vol_ma20 > 0 and vol_ma5 < vol_ma20 * 0.6 and price_up:
            return {'bearish': True, 'reason': '缩量上涨(动力不足)', 'bonus': 0.10}
        if vol_ma20 > 0 and vol_ma5 < vol_ma20 * 0.5 and not price_up:
            return {'bearish': True, 'reason': '缩量下跌', 'bonus': 0.15}
        return {'bearish': False, 'reason': '', 'bonus': 0}

    def _no_signal(self, ctx: CommitteeContext) -> AgentArgument:
        return AgentArgument(
            agent_name='BearAnalyst', stance='bear',
            reasoning='数据不足', confidence=0.2,
            key_points=[], data_references={'symbol': ctx.symbol},
        )


# ============================================================
# 3. SentimentAnalyzer — 市场情绪分析 (纯本地计算)
# ============================================================

class SentimentAnalyzer:
    """
    市场情绪分析 — 纯价格/量计算，无外部API

    评分维度:
      - 短期动量 (5d, 10d, 20d)
      - 量能信号
      - 价格位置
      - MACD柱状图趋势
      - 筹码收集/派发迹象
    """

    def analyze(self, ctx: CommitteeContext) -> AgentArgument:
        df = ctx.df_daily
        if len(df) < 30:
            return AgentArgument(
                agent_name='SentimentAnalyzer', stance='neutral',
                reasoning='数据不足', confidence=0.0,
                key_points=[], data_references={'symbol': ctx.symbol},
            )

        scores = {}
        key_points = []

        # 1. 短期动量 (0.25)
        ret_5d = self._calc_return(df, 5)
        scores['momentum_5d'] = normalize(ret_5d, -0.05, 0.05) * 0.25

        # 2. 中期动量 (0.20)
        ret_10d = self._calc_return(df, 10)
        scores['momentum_10d'] = normalize(ret_10d, -0.10, 0.10) * 0.20

        # 3. 量能信号 (0.15)
        vol_signal = self._volume_signal(df)
        scores['volume'] = vol_signal['score'] * 0.15
        if vol_signal['note']:
            key_points.append(vol_signal['note'])

        # 4. 价格位置 (0.15)
        pos_signal = self._price_position(df)
        scores['position'] = pos_signal['score'] * 0.15
        if pos_signal['note']:
            key_points.append(pos_signal['note'])

        # 5. MACD柱状图趋势 (0.15)
        macd_signal = self._macd_trend(df)
        scores['macd'] = macd_signal['score'] * 0.15
        if macd_signal['note']:
            key_points.append(macd_signal['note'])

        # 6. 筹码收集迹象 (0.10)
        accum_signal = self._accumulation(df)
        scores['accumulation'] = accum_signal['score'] * 0.10
        if accum_signal['note']:
            key_points.append(accum_signal['note'])

        # 综合: range [-1, +1]
        sentiment_raw = sum(scores.values())
        sentiment_raw = max(-1.0, min(1.0, sentiment_raw))

        # 动量描述
        if ret_5d > 0.03:
            key_points.append(f'5日涨幅{ret_5d:+.1%}')
        elif ret_5d < -0.03:
            key_points.append(f'5日跌幅{ret_5d:+.1%}')

        confidence = abs(sentiment_raw)
        reasoning = f"情绪评分: {sentiment_raw:+.2f} ({'偏多' if sentiment_raw > 0.2 else '偏空' if sentiment_raw < -0.2 else '中性'})"

        return AgentArgument(
            agent_name='SentimentAnalyzer',
            stance='bull' if sentiment_raw > 0 else 'bear',
            reasoning=reasoning,
            confidence=confidence,
            key_points=key_points,
            data_references={
                'symbol': ctx.symbol,
                'sentiment_raw': float(sentiment_raw),
                'sub_scores': {k: float(v) for k, v in scores.items()},
            },
        )

    def _calc_return(self, df: pd.DataFrame, days: int) -> float:
        if len(df) < days + 1:
            return 0.0
        return (df['close'].iloc[-1] / df['close'].iloc[-days-1] - 1)

    def _volume_signal(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {'score': 0, 'note': ''}
        vol = df['volume'].values.astype(float)
        vol_ma5 = np.mean(vol[-5:])
        vol_ma20 = np.mean(vol[-20:])
        if vol_ma20 == 0:
            return {'score': 0, 'note': ''}
        ratio = vol_ma5 / vol_ma20
        if ratio > 1.5:
            return {'score': 0.8, 'note': '明显放量'}
        elif ratio > 1.2:
            return {'score': 0.4, 'note': '温和放量'}
        elif ratio < 0.6:
            return {'score': -0.3, 'note': '缩量'}
        return {'score': 0.0, 'note': ''}

    def _price_position(self, df: pd.DataFrame) -> Dict:
        if len(df) < 60:
            return {'score': 0, 'note': ''}
        high_60 = df['high'].iloc[-60:].max()
        low_60 = df['low'].iloc[-60:].min()
        current = df['close'].iloc[-1]
        pct = (current - low_60) / (high_60 - low_60) if high_60 > low_60 else 0.5
        if pct < 0.2:
            return {'score': 0.6, 'note': '低位区间(可能超跌反弹)'}
        elif pct < 0.4:
            return {'score': 0.3, 'note': '中低位'}
        elif pct > 0.85:
            return {'score': -0.4, 'note': '高位区间(注意回调)'}
        return {'score': 0.0, 'note': ''}

    def _macd_trend(self, df: pd.DataFrame) -> Dict:
        if len(df) < 35:
            return {'score': 0, 'note': ''}
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        hist = (dif - dea) * 2
        # 柱状图趋势: 最近3根是否递增
        if len(hist) >= 3:
            if hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3] and hist.iloc[-1] > 0:
                return {'score': 0.6, 'note': 'MACD红柱扩大'}
            elif hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3] and hist.iloc[-1] < 0:
                return {'score': -0.5, 'note': 'MACD绿柱扩大'}
            elif hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
                return {'score': 0.4, 'note': 'MACD金叉'}
        return {'score': 0.0, 'note': ''}

    def _accumulation(self, df: pd.DataFrame) -> Dict:
        """检测筹码收集/派发迹象"""
        if len(df) < 15:
            return {'score': 0, 'note': ''}
        close = df['close'].values
        vol = df['volume'].values.astype(float)
        # 价格窄幅波动 + 缩量 = 收集
        recent_range = (close[-10:].max() - close[-10:].min()) / close[-10:].mean()
        recent_vol_avg = np.mean(vol[-5:])
        prev_vol_avg = np.mean(vol[-15:-5]) if len(vol) >= 15 else recent_vol_avg
        if recent_range < 0.05 and recent_vol_avg < prev_vol_avg * 0.8 and prev_vol_avg > 0:
            return {'score': 0.5, 'note': '缩量横盘(收集迹象)'}
        # 价格上涨 + 放量后缩量 = 可能派发
        if close[-1] > close[-10] and recent_vol_avg < prev_vol_avg * 0.6 and prev_vol_avg > 0:
            return {'score': -0.3, 'note': '高位缩量(派发风险)'}
        return {'score': 0.0, 'note': ''}


# ============================================================
# 4. SectorRotation — 行业轮动评估
# ============================================================

class SectorRotation:
    """
    行业轮动评估

    评分维度:
      - 行业相对强度 (vs 市场中位数)
      - 成长性行业加成
      - 行业趋势 (改善/稳定/衰退)
      - 行业内上涨股票比例
    """

    def __init__(self, sector_map: Dict[str, str] = None,
                 sector_momentum: Dict[str, float] = None):
        self.sector_map = sector_map or {}
        self.sector_momentum = sector_momentum or {}

    def analyze(self, ctx: CommitteeContext) -> AgentArgument:
        sector = ctx.sector
        key_points = []
        scores = {}

        # 1. 行业相对强度 (0.40)
        sector_ret = self.sector_momentum.get(sector, 0)
        all_rets = list(self.sector_momentum.values()) if self.sector_momentum else [0]
        market_median = float(np.median(all_rets)) if all_rets else 0
        if market_median != 0:
            relative_strength = sector_ret / market_median
        else:
            relative_strength = 1.0
        rs_score = normalize(relative_strength, 0.5, 2.0)  # 0.5x ~ 2.0x
        rs_score = max(0, rs_score)  # 不给负分
        scores['relative_strength'] = rs_score * 0.40
        if relative_strength > 1.3:
            key_points.append(f'行业动量强势({sector_ret:+.1f}%)')

        # 2. 成长行业加成 (0.25)
        growth_bonus = 1.0 if sector in GROWTH_SECTORS else 0.0
        scores['growth'] = growth_bonus * 0.25
        if sector in GROWTH_SECTORS:
            key_points.append(f'成长性行业({sector})')

        # 3. 行业趋势 (0.20)
        trend_score = self._sector_trend(sector)
        scores['trend'] = trend_score * 0.20

        # 4. 行业内上涨比例 (0.15) — 基于sector_momentum
        rising_ratio = self._sector_rising_ratio(sector)
        scores['rising_ratio'] = rising_ratio * 0.15

        sector_score = sum(scores.values())
        sector_score = max(0.0, min(1.0, sector_score))

        reasoning = f'行业评分: {sector_score:.2f} (动量{sector_ret:+.1f}%, RS={relative_strength:.1f}x)'

        return AgentArgument(
            agent_name='SectorRotation',
            stance='bull' if sector_score > 0.5 else 'neutral',
            reasoning=reasoning,
            confidence=sector_score,
            key_points=key_points,
            data_references={
                'symbol': ctx.symbol,
                'sector': sector,
                'sector_return': float(sector_ret),
                'relative_strength': float(relative_strength),
                'sub_scores': scores,
            },
        )

    def _sector_trend(self, sector: str) -> float:
        """行业趋势: 改善=1.0, 稳定=0.5, 衰退=0.0"""
        ret = self.sector_momentum.get(sector, 0)
        if ret > 3:
            return 1.0
        elif ret > 0:
            return 0.7
        elif ret > -2:
            return 0.3
        else:
            return 0.0

    def _sector_rising_ratio(self, sector: str) -> float:
        """行业内上涨股票比例(近似用momentum符号)"""
        ret = self.sector_momentum.get(sector, 0)
        if ret > 2:
            return 0.8
        elif ret > 0:
            return 0.6
        elif ret > -2:
            return 0.4
        else:
            return 0.2


# ============================================================
# 5. RiskManager — 风控经理
# ============================================================

class RiskManager:
    """
    风控评估

    评估维度:
      - 波动率 (ATR-based)
      - 回撤风险
      - 行业集中度
      - 组合仓位限制
      - 单票仓位限制
    """

    def evaluate(
        self,
        ctx: CommitteeContext,
        bull_arg: AgentArgument,
        bear_arg: AgentArgument,
        sentiment_arg: AgentArgument,
        sector_arg: AgentArgument,
    ) -> RiskAssessment:
        df = ctx.df_daily
        portfolio = ctx.portfolio_state
        risk_factors = {}
        warnings = []

        # 1. 波动率风险
        vol_risk = self._volatility_risk(df)
        risk_factors['volatility'] = vol_risk

        # 2. 最大回撤风险
        dd_risk = self._drawdown_risk(df)
        risk_factors['drawdown'] = dd_risk

        # 3. 行业集中度
        sector_conc = self._sector_concentration(ctx)
        risk_factors['sector_concentration'] = sector_conc

        # 4. 组合仓位
        position_risk = self._position_risk(portfolio)
        risk_factors['position_usage'] = position_risk

        # 5. 技术面风险 (从bear_arg获取)
        risk_factors['bear_signal'] = bear_arg.confidence

        # 综合风险评分
        risk_score = (
            vol_risk * 0.25
            + dd_risk * 0.20
            + sector_conc * 0.20
            + position_risk * 0.15
            + bear_arg.confidence * 0.20
        )
        risk_score = max(0.0, min(1.0, risk_score))

        # 计算止损/止盈
        stop_loss = self._calc_stop_loss(df, ctx)
        take_profit = self._calc_take_profit(df, ctx, stop_loss)

        # 计算仓位
        capital = portfolio.get('capital', 1000000)
        max_positions = portfolio.get('max_positions', 10)
        shares, position_pct = calc_position_size(
            capital, ctx.entry_price, stop_loss,
            max_position_pct=1.0 / max_positions,
        )

        # 警告
        if risk_score > 0.6:
            warnings.append(f'风险较高({risk_score:.0%})')
        if sector_conc > 0.3:
            warnings.append(f'行业集中度高({ctx.sector}: {sector_conc:.0%})')

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=classify_risk(risk_score),
            position_pct=position_pct,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            max_shares=shares,
            risk_factors=risk_factors,
            warnings=warnings,
            sector_concentration=sector_conc,
        )

    def _volatility_risk(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.5
        returns = df['close'].pct_change().dropna()
        if len(returns) < 10:
            return 0.5
        vol = returns.iloc[-20:].std()
        # 年化波动率映射: 20%以下=低风险, 40%以上=高风险
        annual_vol = vol * np.sqrt(252)
        return min(1.0, max(0.0, (annual_vol - 0.15) / 0.35))

    def _drawdown_risk(self, df: pd.DataFrame) -> float:
        if len(df) < 30:
            return 0.3
        high = df['high'].iloc[-30:]
        close = df['close'].iloc[-1]
        max_high = high.max()
        if max_high <= 0:
            return 0.3
        drawdown = (max_high - close) / max_high
        return min(1.0, drawdown / 0.20)  # 回撤10% = 0.5, 20% = 1.0

    def _sector_concentration(self, ctx: CommitteeContext) -> float:
        positions = ctx.portfolio_state.get('positions', [])
        if not positions:
            return 0.0
        sector_count = sum(1 for p in positions if p.get('sector') == ctx.sector)
        return sector_count / max(len(positions), 1)

    def _position_risk(self, portfolio: Dict) -> float:
        positions = portfolio.get('positions', [])
        max_pos = portfolio.get('max_positions', 10)
        return len(positions) / max_pos if max_pos > 0 else 0

    def _calc_stop_loss(self, df: pd.DataFrame, ctx: CommitteeContext) -> float:
        """动态止损: 优先使用缠论结构止损, 结合ATR"""
        stops = []

        # 1. 扫描器止损
        if ctx.stop_price > 0:
            stops.append(ctx.stop_price)

        # 2. 缠论结构止损（最优先 — 中枢下沿或笔低点）
        if ctx.chanlun and ctx.chanlun.stop_by_structure > 0:
            stops.append(ctx.chanlun.stop_by_structure)

        # 3. ATR止损
        if len(df) >= 15:
            high = df['high'].iloc[-14:]
            low = df['low'].iloc[-14:]
            close = df['close'].iloc[-15:-1]
            tr = pd.concat([
                high - low,
                (high - close.values).abs(),
                (low - close.values).abs(),
            ], axis=1).max(axis=1)
            atr = tr.mean()
            atr_stop = ctx.entry_price - 2 * atr
            stops.append(atr_stop)

        if not stops:
            return ctx.entry_price * 0.95

        # 取所有止损中最高者（最保守）
        best_stop = max(stops)

        # 确保止损不超过入场价
        if best_stop >= ctx.entry_price:
            return ctx.entry_price * 0.97

        return best_stop

    def _calc_take_profit(self, df: pd.DataFrame, ctx: CommitteeContext,
                          stop_loss: float) -> float:
        """止盈: 基于R/R比"""
        risk = ctx.entry_price - stop_loss
        if risk <= 0:
            return ctx.entry_price * 1.15
        # 目标R/R = 2.5
        return ctx.entry_price + risk * 2.5


# ============================================================
# 6. FundManager — 基金经理
# ============================================================

class FundManager:
    """
    基金经理 — 最终决策

    职责:
      - 综合所有Agent意见
      - 应用组合约束
      - 计算最终评分和决策
      - 确定仓位大小
    """

    def decide(
        self,
        ctx: CommitteeContext,
        bull_arg: AgentArgument,
        bear_arg: AgentArgument,
        sentiment_arg: AgentArgument,
        sector_arg: AgentArgument,
        risk: RiskAssessment,
        weights: Dict[str, float] = None,
    ) -> Dict:
        """做出最终投资决策"""
        from agents.scoring import calc_composite_score, make_decision

        # 获取情绪评分
        sentiment_raw = sentiment_arg.data_references.get('sentiment_raw', 0)

        # 计算综合评分
        composite = calc_composite_score(
            bull_confidence=bull_arg.confidence,
            bear_confidence=bear_arg.confidence,
            sentiment_score=sentiment_raw,
            sector_score=sector_arg.confidence,
            scanner_score=ctx.scanner_score,
            risk_score=risk.risk_score,
            weights=weights,
        )

        # 统计同行业持仓
        positions = ctx.portfolio_state.get('positions', [])
        sector_count = sum(1 for p in positions if p.get('sector') == ctx.sector)

        # 决策
        decision, confidence = make_decision(
            composite, bull_arg.confidence, bear_arg.confidence,
            risk.risk_score, sector_count,
        )

        # 过滤器链最终把关（仅对buy决策）
        if decision == 'buy' and FILTERS_AVAILABLE:
            filter_result = self._run_filter_chain(ctx, risk)
            if not filter_result['passed']:
                decision = 'hold'
                confidence = 0.4
                risk.warnings.append(f"过滤器否决: {filter_result['reason']}")

        # 关键因素
        key_factors = []
        if bull_arg.key_points:
            key_factors.extend(bull_arg.key_points[:2])
        if sector_arg.key_points:
            key_factors.extend(sector_arg.key_points[:1])
        if sentiment_arg.key_points:
            key_factors.extend(sentiment_arg.key_points[:1])
        key_factors = key_factors[:5]

        # 生成摘要
        decision_cn = {'buy': '买入', 'hold': '观望', 'reject': '否决'}
        summary = (
            f"[{decision_cn.get(decision, decision)}] {ctx.symbol} {ctx.name} "
            f"| 综合{composite:.0f}分 | 多{bull_arg.confidence:.0%} vs 空{bear_arg.confidence:.0%} "
            f"| 情绪{sentiment_raw:+.2f} | 行业{sector_arg.confidence:.0%} "
            f"| 风险{risk.risk_level}"
        )

        return {
            'symbol': ctx.symbol,
            'name': ctx.name,
            'sector': ctx.sector,
            'decision': decision,
            'confidence': float(confidence),
            'composite_score': float(composite),
            'position_pct': float(risk.position_pct) if decision == 'buy' else 0,
            'shares': risk.max_shares if decision == 'buy' else 0,
            'entry_price': float(ctx.entry_price),
            'stop_loss': float(risk.stop_loss_price),
            'take_profit': float(risk.take_profit_price),
            'key_factors': key_factors,
            'bull_confidence': float(bull_arg.confidence),
            'bear_confidence': float(bear_arg.confidence),
            'sentiment_score': float(sentiment_raw),
            'sector_score': float(sector_arg.confidence),
            'risk_score': float(risk.risk_score),
            'risk_level': risk.risk_level,
            'debate_summary': summary,
            'warnings': risk.warnings,
        }

    def _run_filter_chain(self, ctx: CommitteeContext, risk: RiskAssessment) -> Dict:
        """运行过滤器链作为最终把关"""
        try:
            filters = [
                VolumeFilter(min_ratio=0.6),
                TrendAlignmentFilter(require_macd=False),
                RegimeFilter(),
                CooldownFilter(cooldown_bars=10),
            ]
            chain = CompositeFilter(filters)
            # 构造简单的signal和context给过滤器
            signal = {'type': 'buy', 'point_type': '2buy'}
            filter_ctx = {
                'df': ctx.df_daily,
                'sector': ctx.sector,
                'risk_score': risk.risk_score,
            }
            passed, reason = chain.should_enter(signal, filter_ctx)
            return {'passed': passed, 'reason': reason}
        except Exception as e:
            # 过滤器失败时不阻止
            return {'passed': True, 'reason': f'过滤器异常: {e}'}
