"""
缠论实时交易决策分析

用法:
    python trade_analysis.py sz002600
    python trade_analysis.py sh600519 --tf daily,weekly
    python trade_analysis.py 000001.SZ --tf 30min,daily
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

# 项目根目录 (.claude/skills/trade/scripts/ → 5 levels up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class TimeframeResult:
    """单周期分析结果"""
    period: str
    price: float
    price_date: str
    fractals_count: int
    strokes_count: int
    segments_count: int
    pivots: List[dict]
    latest_pivot: Optional[dict]
    buy_points: List[dict]
    sell_points: List[dict]
    latest_signal: Optional[dict]
    macd_status: dict
    trend: str  # 'up', 'down', 'sideways'


@dataclass
class TradeDecision:
    """综合交易决策"""
    action: str          # BUY, SELL, HOLD, EXIT
    signal_type: str     # 1buy, 2buy, 3buy, 1sell, 2sell, 3sell, none
    confidence: int      # 0-100
    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    risk_reward: float
    position_shares: int
    position_value: float
    max_loss: float
    reason: str


# ── 工具函数 ──────────────────────────────────────────────

def normalize_code(code: str) -> str:
    """统一股票代码为6位数字

    支持: sz002600, sh600519, 000001.SZ, 600519, 002600
    返回: 6位数字字符串
    """
    code = code.strip().upper()
    # 去除前缀
    for prefix in ('SH', 'SZ', 'BJ'):
        code = code.replace(prefix, '')
    # 去除后缀
    for suffix in ('.SH', '.SZ', '.BJ', '.XSHG', '.XSHE'):
        code = code.replace(suffix, '')
    return code.zfill(6)


def infer_market(code6: str) -> str:
    """根据6位代码推断市场

    6xx/5xx → sh, 0xx/3xx → sz, 8xx/4xx → bj
    """
    first = code6[0]
    if first in ('6', '5'):
        return 'sh'
    elif first in ('0', '3'):
        return 'sz'
    elif first in ('8', '4'):
        return 'bj'
    return 'sz'


def code_display(code: str) -> str:
    """显示用代码: sz002600"""
    c = normalize_code(code)
    return f"{infer_market(c)}{c}"


def fetch_stock_name(code6: str) -> str:
    """尝试获取股票名称"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        match = df[df['代码'] == code6]
        if not match.empty:
            return match.iloc[0]['名称']
    except Exception:
        pass
    return code6


# ── 数据获取 ──────────────────────────────────────────────

def fetch_data(code6: str, period: str) -> Optional[pd.DataFrame]:
    """获取单周期 OHLCV 数据，优先在线，回退本地缓存"""
    # 1. 尝试本地缓存 (chanlun_system/artifacts/)
    for suffix in ('.SZ', '.SH', '.BJ'):
        p = PROJECT_ROOT / 'chanlun_system' / 'artifacts' / f'ohlcv_{code6}{suffix}.csv'
        if p.exists():
            return _load_local(p)
    # .claude/temp/ 缓存
    for ext in ('.csv', '.day.json'):
        p = PROJECT_ROOT / '.claude' / 'temp' / f'{code6}{ext}'
        if p.exists():
            return _load_local(p)

    # 2. 尝试 AKShare 在线
    try:
        from data.akshare_source import AKShareSource
        source = AKShareSource()

        end_date = datetime.now()
        if period == '30min':
            start_date = end_date - timedelta(days=90)
        elif period == 'daily':
            start_date = end_date - timedelta(days=730)
        else:  # weekly
            start_date = end_date - timedelta(days=1825)

        df = source.get_kline(
            symbol=code6,
            start_date=start_date,
            end_date=end_date,
            period=period,
            adjust='qfq',
        )
        if df is not None and len(df) > 30:
            return df
    except Exception as e:
        print(f"  [WARN] AKShare 获取 {period} 数据失败: {e}")

    # 3. 尝试 mootdx 在线
    try:
        df = _fetch_mootdx(code6, period)
        if df is not None and len(df) > 30:
            return df
    except Exception as e:
        print(f"  [WARN] mootdx 获取 {period} 数据失败: {e}")

    return None


def _fetch_mootdx(code6: str, period: str) -> Optional[pd.DataFrame]:
    """通过 mootdx 获取数据（直连通达信服务器，绕过代理）"""
    from mootdx.quotes import Quotes
    import time

    client = Quotes.factory(market='std')

    # frequency: 4=日线, 5=周线, 2=30分钟
    freq_map = {'30min': 2, 'daily': 4, 'weekly': 5}
    freq = freq_map.get(period, 4)

    all_dfs = []
    for batch in range(5):
        offset = (batch + 1) * 800  # offset=0 returns empty, start from 800
        df = client.bars(symbol=code6, frequency=freq, offset=offset)
        if df is None or len(df) == 0:
            break
        all_dfs.append(df)
        if len(df) < 800:
            break
        time.sleep(0.3)

    if not all_dfs:
        return None

    result = pd.concat(all_dfs)
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()

    # 标准化列名: mootdx 可能返回 'vol' 或 'volume'，统一为 'volume'
    if 'vol' in result.columns and 'volume' not in result.columns:
        result = result.rename(columns={'vol': 'volume'})

    # 去重列（mootdx 可能同时有 vol 和 volume）
    result = result.loc[:, ~result.columns.duplicated()]

    # 确保有 datetime 列: mootdx 索引是 datetime，可能也有 datetime 列
    if 'datetime' in result.columns:
        result = result.drop(columns=['datetime'])
    result.index.name = 'datetime'
    result = result.reset_index()
    result['datetime'] = pd.to_datetime(result['datetime'])

    return result[['datetime', 'open', 'high', 'low', 'close', 'volume']]


def _market_suffix(prefix: str) -> str:
    return {'sh': 'SH', 'sz': 'SZ', 'bj': 'BJ'}.get(prefix, 'SZ')


def _load_local(path: Path) -> Optional[pd.DataFrame]:
    """加载本地数据文件"""
    try:
        if path.suffix == '.csv':
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif path.suffix == '.json':
            import json
            with open(path) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            return None

        # 确保列名标准化
        col_map = {'日期': 'datetime', '开': 'open', '高': 'high', '低': 'low',
                    '收': 'close', '量': 'volume', '成交额': 'amount'}
        df.rename(columns=col_map, inplace=True)

        if len(df) > 30:
            return df
    except Exception:
        pass
    return None


# ── 单周期缠论分析 ──────────────────────────────────────

def analyze_timeframe(df: pd.DataFrame, period: str) -> TimeframeResult:
    """对单个周期运行完整缠论 pipeline"""
    from core.kline import KLine
    from core.fractal import detect_fractals
    from core.stroke import generate_strokes
    from core.segment import generate_segments
    from core.pivot import detect_pivots, PivotLevel
    from core.buy_sell_points import BuySellPointDetector
    from indicator.macd import MACD

    # K线处理
    kline = KLine.from_dataframe(df)

    # 分型
    fractals = detect_fractals(kline)

    # 笔
    strokes = generate_strokes(kline, fractals)

    # 线段
    segments = generate_segments(kline, strokes)

    # 中枢
    level_map = {'30min': PivotLevel.MIN_30, 'daily': PivotLevel.DAY, 'weekly': PivotLevel.WEEK}
    pivot_level = level_map.get(period, PivotLevel.DAY)
    pivots = detect_pivots(kline, strokes, pivot_level)

    # MACD
    close = df['close']
    macd = MACD(close)

    # 买卖点
    detector = BuySellPointDetector(
        fractals=fractals,
        strokes=strokes,
        segments=segments,
        pivots=pivots,
        macd=macd,
    )
    buy_points, sell_points = detector.detect_all()

    # 当前价格
    price = float(df['close'].iloc[-1])
    price_date = str(df['datetime'].iloc[-1].date()) if 'datetime' in df.columns else str(df.index[-1].date())

    # MACD 状态
    macd_status = _get_macd_status(macd)

    # 趋势判断
    trend = _judge_trend(strokes, pivots, price)

    # 中枢信息
    pivot_dicts = [_pivot_to_dict(p) for p in pivots[-5:]]
    latest_pivot = pivot_dicts[-1] if pivot_dicts else None

    # 买卖点信息
    bp_dicts = [_point_to_dict(p) for p in buy_points[-3:]]
    sp_dicts = [_point_to_dict(p) for p in sell_points[-3:]]

    # 最新信号
    latest_signal = _get_latest_signal(buy_points, sell_points, len(df))

    return TimeframeResult(
        period=period,
        price=price,
        price_date=price_date,
        fractals_count=len(fractals),
        strokes_count=len(strokes),
        segments_count=len(segments),
        pivots=pivot_dicts,
        latest_pivot=latest_pivot,
        buy_points=bp_dicts,
        sell_points=sp_dicts,
        latest_signal=latest_signal,
        macd_status=macd_status,
        trend=trend,
    )


def _get_macd_status(macd) -> dict:
    """提取 MACD 当前状态"""
    if not macd.values:
        return {'dif': 0, 'dea': 0, 'hist': 0, 'signal': 'N/A', 'cross': 'none'}

    v = macd.values[-1]
    vp = macd.values[-2] if len(macd.values) > 1 else v

    # 金叉/死叉
    cross = 'none'
    if vp.macd <= vp.signal and v.macd > v.signal:
        cross = 'golden'
    elif vp.macd >= vp.signal and v.macd < v.signal:
        cross = 'death'

    # 信号描述
    if v.histogram > 0 and v.histogram > vp.histogram:
        signal = '多头增强'
    elif v.histogram > 0 and v.histogram < vp.histogram:
        signal = '多头减弱'
    elif v.histogram < 0 and v.histogram < vp.histogram:
        signal = '空头增强'
    elif v.histogram < 0 and v.histogram > vp.histogram:
        signal = '空头减弱'
    else:
        signal = '平衡'

    return {
        'dif': round(v.macd, 4),
        'dea': round(v.signal, 4),
        'hist': round(v.histogram, 4),
        'signal': signal,
        'cross': cross,
    }


def _judge_trend(strokes, pivots, price) -> str:
    """判断趋势方向"""
    if not strokes:
        return 'sideways'
    last = strokes[-1]
    if not pivots:
        return 'up' if last.is_up else 'down'
    last_pivot = pivots[-1]
    if price > last_pivot.high:
        return 'up'
    elif price < last_pivot.low:
        return 'down'
    return 'sideways'


def _pivot_to_dict(p) -> dict:
    """中枢转字典"""
    return {
        'high': round(p.high, 2),
        'low': round(p.low, 2),
        'zg': round(getattr(p, 'zg', p.high), 2),
        'zd': round(getattr(p, 'zd', p.low), 2),
        'strokes_count': getattr(p, 'strokes_count', len(p.strokes)),
        'quality': round(getattr(p, 'quality_score', 0.5), 2),
    }


def _point_to_dict(p) -> dict:
    """买卖点转字典"""
    return {
        'type': p.point_type,
        'price': round(p.price, 2),
        'index': p.index,
        'confidence': round(p.confidence, 2),
        'stop_loss': round(p.stop_loss, 2) if p.stop_loss > 0 else 0,
        'reason': p.reason[:60] if p.reason else '',
        'divergence': round(p.divergence_ratio, 3),
    }


def _get_latest_signal(buy_points, sell_points, total_bars: int, window: int = 60) -> Optional[dict]:
    """获取最近的买卖点信号"""
    recent = max(1, total_bars - window)

    latest_buy = None
    latest_sell = None
    for bp in reversed(buy_points):
        if bp.index >= recent:
            latest_buy = _point_to_dict(bp)
            break
    for sp in reversed(sell_points):
        if sp.index >= recent:
            latest_sell = _point_to_dict(sp)
            break

    # 返回最近的一个
    if latest_buy and latest_sell:
        return latest_buy if latest_buy['index'] > latest_sell['index'] else latest_sell
    return latest_buy or latest_sell


# ── 多周期综合决策 ──────────────────────────────────────

def make_decision(results: Dict[str, TimeframeResult], capital: float = 100000) -> TradeDecision:
    """综合多周期分析，生成交易决策"""
    daily = results.get('daily')
    weekly = results.get('weekly')

    if not daily:
        return _no_signal_decision(capital)

    price = daily.price

    # 1. 周线战略方向
    weekly_bias = 'neutral'
    if weekly:
        weekly_bias = weekly.trend

    # 2. 日线信号
    daily_signal = daily.latest_signal

    # 3. 无买卖点时，基于趋势和结构给 HOLD 建议
    if not daily_signal:
        return _make_trend_hold_decision(daily, weekly, capital)

    signal_type = daily_signal['type']
    is_buy = 'buy' in signal_type
    signal_price = daily_signal['price']
    signal_confidence = daily_signal.get('confidence', 0.5)

    # 4. 多周期一致性加成
    confidence = int(signal_confidence * 70)
    if weekly and weekly_bias == 'up' and is_buy:
        confidence += 15
    elif weekly and weekly_bias == 'down' and not is_buy:
        confidence += 15
    elif weekly and weekly_bias == 'down' and is_buy:
        confidence -= 20  # 逆势降低

    # MACD 确认
    macd = daily.macd_status
    if macd['cross'] == 'golden' and is_buy:
        confidence += 10
    elif macd['cross'] == 'death' and not is_buy:
        confidence += 10

    confidence = max(10, min(100, confidence))

    # 5. 止损计算
    stop_loss = daily_signal.get('stop_loss', 0)
    if stop_loss <= 0:
        stop_loss = price * 0.92  # 默认8%止损

    # 使用最新中枢辅助止损
    if daily.latest_pivot and is_buy:
        pivot_low = daily.latest_pivot['low']
        stop_loss = min(stop_loss, pivot_low) if stop_loss > 0 else pivot_low

    # 6. 目标价
    if daily.latest_pivot:
        target1 = daily.latest_pivot['high'] * 1.05
        target2 = daily.latest_pivot['high'] * 1.15
    else:
        target1 = price * 1.10
        target2 = price * 1.20

    # 7. 盈亏比
    risk = price - stop_loss if stop_loss > 0 else price * 0.08
    reward = target1 - price
    rr = round(reward / risk, 1) if risk > 0 else 0

    # 8. 仓位
    risk_per_trade = capital * 0.02
    if risk > 0:
        shares = int(risk_per_trade / risk / 100) * 100  # 整百
        shares = max(100, min(shares, int(capital * 0.3 / price / 100) * 100))
    else:
        shares = 0
    position_value = shares * price
    max_loss = shares * risk

    action = 'BUY' if is_buy else 'SELL'
    reason = daily_signal.get('reason', signal_type)
    if weekly and weekly_bias != 'neutral':
        reason += f' | 周线{weekly_bias}'

    return TradeDecision(
        action=action,
        signal_type=signal_type,
        confidence=confidence,
        entry_price=round(price, 2),
        stop_loss=round(stop_loss, 2),
        target1=round(target1, 2),
        target2=round(target2, 2),
        risk_reward=rr,
        position_shares=shares,
        position_value=round(position_value, 2),
        max_loss=round(max_loss, 2),
        reason=reason,
    )


def _make_trend_hold_decision(
    daily: TimeframeResult,
    weekly: Optional['TimeframeResult'],
    capital: float,
) -> TradeDecision:
    """无买卖点时，基于趋势和结构给出 HOLD 建议（含观望/关注/警惕三级）"""
    price = daily.price
    trend = daily.trend
    macd = daily.macd_status
    pivot = daily.latest_pivot

    # 价格与中枢的距离分析
    pivot_distance = ''
    if pivot:
        mid = (pivot['high'] + pivot['low']) / 2
        range_size = pivot['high'] - pivot['low']
        if range_size > 0:
            dist_pct = (price - mid) / range_size * 100
            if dist_pct > 150:
                pivot_distance = '远超中枢上沿'
            elif dist_pct > 100:
                pivot_distance = '突破中枢上方'
            elif dist_pct > 50:
                pivot_distance = '中枢偏上'
            elif dist_pct > -50:
                pivot_distance = '中枢内部'
            elif dist_pct > -100:
                pivot_distance = '中枢偏下'
            else:
                pivot_distance = '跌破中枢下方'

    # 构建原因
    parts = []
    if trend == 'up':
        parts.append('趋势偏多')
    elif trend == 'down':
        parts.append('趋势偏空')
    else:
        parts.append('趋势震荡')

    if pivot_distance:
        parts.append(pivot_distance)

    if macd['hist'] > 0 and macd['signal'] in ('多头增强',):
        parts.append('MACD多头增强')
    elif macd['hist'] > 0 and macd['signal'] in ('多头减弱',):
        parts.append('MACD多头减弱')
    elif macd['hist'] < 0 and macd['signal'] in ('空头增强',):
        parts.append('MACD空头增强')
    elif macd['hist'] < 0 and macd['signal'] in ('空头减弱',):
        parts.append('MACD空头减弱')

    if weekly:
        parts.append(f'周线{weekly.trend}')

    reason = ' | '.join(parts)

    # 建议: up趋势+MACD多头 → 关注; down趋势+MACD空头 → 警惕; 其他 → 观望
    if trend == 'up' and macd['hist'] > 0:
        suggestion = '关注回调买点'
        confidence = 50
    elif trend == 'down' and macd['hist'] < 0:
        suggestion = '警惕继续下跌'
        confidence = 45
    elif trend == 'up' and macd['hist'] < 0:
        suggestion = '多头动能减弱，注意风险'
        confidence = 40
    elif trend == 'down' and macd['hist'] > 0:
        suggestion = '空头动能减弱，关注反弹'
        confidence = 40
    else:
        suggestion = '等待结构形成'
        confidence = 30

    return TradeDecision(
        action='HOLD', signal_type='none', confidence=confidence,
        entry_price=round(price, 2), stop_loss=0, target1=0, target2=0,
        risk_reward=0, position_shares=0, position_value=0, max_loss=0,
        reason=f'{suggestion} — {reason}',
    )


def _no_signal_decision(capital: float) -> TradeDecision:
    return TradeDecision(
        action='HOLD', signal_type='none', confidence=0,
        entry_price=0, stop_loss=0, target1=0, target2=0,
        risk_reward=0, position_shares=0, position_value=0, max_loss=0,
        reason='无法获取数据或数据不足',
    )


# ── 报告生成 ──────────────────────────────────────────────

TREND_CN = {'up': '偏多', 'down': '偏空', 'sideways': '震荡'}
PERIOD_CN = {'30min': '30分钟', 'daily': '日线', 'weekly': '周线'}
ACTION_CN = {'BUY': '买入', 'SELL': '卖出', 'HOLD': '观望', 'EXIT': '清仓'}
SIGNAL_CN = {
    '1buy': '1买', '2buy': '2买', '3buy': '3买',
    '1sell': '1卖', '2sell': '2卖', '3sell': '3卖',
    'none': '无信号',
}


def generate_report(
    code: str,
    name: str,
    results: Dict[str, TimeframeResult],
    decision: TradeDecision,
    capital: float,
) -> str:
    """生成格式化报告"""
    lines = []
    w = lines.append

    daily = results.get('daily')

    w('=' * 60)
    w(f'缠论交易决策: {code_display(code)} ({name})')
    if daily:
        w(f'分析日期: {daily.price_date}')
    w('=' * 60)

    # 当前价格 & 操作建议
    if daily:
        w(f'【当前价格】{daily.price:.2f}')
    w(f'【操作建议】{ACTION_CN.get(decision.action, decision.action)} ({SIGNAL_CN.get(decision.signal_type, decision.signal_type)})')
    w(f'【置信度】  {decision.confidence}/100')

    # 周线
    weekly = results.get('weekly')
    if weekly:
        w('')
        w(f'--- 周线 (战略方向) ---')
        w(f'  趋势: {TREND_CN.get(weekly.trend, weekly.trend)}')
        if weekly.latest_pivot:
            p = weekly.latest_pivot
            w(f'  最新中枢: [{p["low"]}, {p["high"]}] ({p["strokes_count"]}笔)')
        w(f'  笔/分型: {weekly.strokes_count}笔 / {weekly.fractals_count}分型')

    # 日线
    if daily:
        w('')
        w('--- 日线 (操作信号) ---')
        if daily.latest_signal:
            s = daily.latest_signal
            w(f'  买卖点: {SIGNAL_CN.get(s["type"], s["type"])} @ {s["price"]:.2f}')
            if s['reason']:
                w(f'  原因: {s["reason"]}')
            if s['divergence'] > 0:
                w(f'  背驰强度: {s["divergence"]:.3f}')
        else:
            w('  近期无买卖点信号')

        if daily.latest_pivot:
            p = daily.latest_pivot
            w(f'  当前中枢: [{p["low"]}, {p["high"]}] ({p["strokes_count"]}笔)')

        m = daily.macd_status
        cross_cn = {'golden': '金叉', 'death': '死叉', 'none': ''}
        cross_str = cross_cn.get(m['cross'], '')
        w(f'  MACD: DIF={m["dif"]} DEA={m["dea"]} 柱={m["hist"]:+.4f} {m["signal"]}')
        if cross_str:
            w(f'  MACD信号: {cross_str}')

        w(f'  趋势: {TREND_CN.get(daily.trend, daily.trend)} | 笔: {daily.strokes_count} | 中枢: {len(daily.pivots)}')

    # 30分钟
    m30 = results.get('30min')
    if m30:
        w('')
        w('--- 30分钟 (入场确认) ---')
        if m30.latest_signal:
            s = m30.latest_signal
            w(f'  信号: {SIGNAL_CN.get(s["type"], s["type"])} @ {s["price"]:.2f}')
        else:
            w('  近期无信号')
        w(f'  趋势: {TREND_CN.get(m30.trend, m30.trend)} | 笔: {m30.strokes_count}')

    # 风控
    if decision.stop_loss > 0:
        w('')
        w('--- 风控 ---')
        w(f'  建议入场价: {decision.entry_price:.2f}')
        w(f'  建议止损: {decision.stop_loss:.2f} (风险 {(1 - decision.stop_loss/decision.entry_price)*100:.1f}%)')
        w(f'  目标1: {decision.target1:.2f} | 目标2: {decision.target2:.2f}')
        w(f'  盈亏比: {decision.risk_reward}')

    # 仓位
    if decision.position_shares > 0:
        w('')
        w(f'--- 仓位 (资金 {capital:,.0f}) ---')
        w(f'  建议仓位: {decision.position_shares}股 ({decision.position_value:,.0f}元, 占 {decision.position_value/capital*100:.1f}%)')
        w(f'  最大可承受亏损: {decision.max_loss:,.0f}元 ({decision.max_loss/capital*100:.1f}%)')

    # 原因
    if decision.reason:
        w('')
        w(f'--- 决策原因 ---')
        w(f'  {decision.reason}')

    # 摘要行
    w('')
    w('=' * 60)
    if decision.action in ('BUY', 'SELL'):
        sl_str = f'SL={decision.stop_loss:.2f}' if decision.stop_loss > 0 else 'SL=N/A'
        t1_str = f'T1={decision.target1:.2f}' if decision.target1 > 0 else 'T1=N/A'
        rr_str = f'R/R={decision.risk_reward}' if decision.risk_reward > 0 else ''
        w(f'决策: {decision.action} {code_display(code)} @ {decision.entry_price:.2f} | {sl_str} | {t1_str} | {rr_str}')
    else:
        w(f'决策: {decision.action} {code_display(code)} — {decision.reason}')
    w('=' * 60)

    return '\n'.join(lines)


# ── 主入口 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='缠论交易决策分析')
    parser.add_argument('code', help='股票代码 (如 sz002600, sh600519, 000001.SZ)，逗号分隔支持批量')
    parser.add_argument('--tf', default='daily,weekly', help='时间框架 (默认 daily,weekly)')
    parser.add_argument('--capital', type=float, default=100000, help='模拟资金 (默认 100000)')
    parser.add_argument('--batch', action='store_true', help='批量模式，输出简要汇总表')
    args = parser.parse_args()

    # 支持批量: "sz002600,sh600519,300936"
    codes = [c.strip() for c in args.code.split(',') if c.strip()]
    timeframes = [tf.strip() for tf in args.tf.split(',')]

    if args.batch and len(codes) > 1:
        _batch_scan(codes, timeframes, args.capital)
        return

    for code_raw in codes:
        code6 = normalize_code(code_raw)
        print(f'分析 {code_display(code6)} ...')

        # 获取股票名称
        name = fetch_stock_name(code6)

        # 多周期分析
        results = {}
        for period in timeframes:
            print(f'  获取 {PERIOD_CN.get(period, period)} 数据...')
            df = fetch_data(code6, period)
            if df is None:
                print(f'  [WARN] {period} 数据不可用，跳过')
                continue
            print(f'  分析 {PERIOD_CN.get(period, period)} ({len(df)}根K线)...')
            try:
                results[period] = analyze_timeframe(df, period)
            except Exception as e:
                print(f'  [WARN] {period} 分析失败: {e}')

        if not results:
            print('错误: 无法获取任何数据')
            continue

        # 综合决策
        decision = make_decision(results, args.capital)

        # 输出报告
        report = generate_report(code_raw, name, results, decision, args.capital)
        print()
        print(report)


def _batch_scan(codes: List[str], timeframes: List[str], capital: float):
    """批量扫描多只股票，输出汇总表"""
    rows = []
    for code_raw in codes:
        code6 = normalize_code(code_raw)
        display = code_display(code6)
        name = fetch_stock_name(code6)

        results = {}
        for period in timeframes:
            df = fetch_data(code6, period)
            if df is None:
                continue
            try:
                results[period] = analyze_timeframe(df, period)
            except Exception:
                pass

        if not results:
            rows.append(f'{display} ({name}) | 无数据')
            continue

        decision = make_decision(results, capital)
        daily = results.get('daily')

        action_cn = ACTION_CN.get(decision.action, decision.action)
        signal_cn = SIGNAL_CN.get(decision.signal_type, decision.signal_type)
        price_str = f'{daily.price:.2f}' if daily else 'N/A'
        trend_cn = TREND_CN.get(daily.trend, 'N/A') if daily else 'N/A'

        # 简要行
        row = (f'{display} ({name}) | {price_str} | '
               f'{action_cn}({signal_cn}) | 趋势{trend_cn} | '
               f'置信{decision.confidence} | {decision.reason[:40]}')
        rows.append(row)

    print()
    print('=' * 80)
    print(f'批量扫描 {len(codes)} 只股票 | {", ".join(timeframes)}')
    print('=' * 80)
    for row in rows:
        print(row)
    print('=' * 80)


if __name__ == '__main__':
    main()
