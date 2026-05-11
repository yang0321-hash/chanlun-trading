#!/usr/bin/env python3
"""增强版实盘扫描 V5 — v7.3策略: 日线CC15主信号 + MA250环境 + 30min入场确认

基于V3演进, 核心变化:
  - 删除周线趋势过滤 → 替换为MA250三级环境过滤
  - 30min完整缠论分析 → 替换为轻量入场确认(底分型+阳线+量≥5日均)
  - 新增quasi2buy检测
  - ATR止损按买点类型区分(1B×0.75, 2B×1.5, 3B×3.0)
"""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

import json, time
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.hybrid_source import HybridSource

from backtest_cc15_mtf import (
    find_daily_1buy_2buy,
    fetch_sina_30min,
)
from lib.entry_confirm_30min import confirm_entry_30min
from small_to_large import detect_small_to_large, apply_score_adjustment

from core.trend_type import classify_trend_type, TrendType
from indicator.vol_price_divergence import detect_volume_price_divergence, DivergenceType
from indicator.market_environment import MarketEnvironment

try:
    from backtest.ml_signal_scorer import predict_signal as _ml_predict
    _ML_AVAILABLE = True
except Exception:
    _ml_predict = None
    _ML_AVAILABLE = False


# 成长性行业优先列表（基于回测验证的高胜率行业）
GROWTH_SECTORS = {
    '专用设备', '能源金属', '航天军工', '电池', '半导体',
    '电气设备', '电力设备', '军工电子', '航空装备',
    '光伏设备', '风电设备', '锂电', '新能源',
    '汽车零部件', '消费电子', '自动化设备',
}

# 行业评分加成
SECTOR_BONUS = {
    'growth': 15,    # 成长性行业加15分
}


# ==================== 主线赛道过滤 ====================

from dataclasses import dataclass

@dataclass
class MainThemeConfig:
    """主线赛道过滤配置"""
    enabled: bool = True
    main_theme_phases: tuple = ('启动', '加速')
    active_phases: tuple = ('高潮', '震荡')
    fading_phases: tuple = ('退潮',)
    max_active_stocks: int = 100
    max_discovery_stocks: int = 50
    fallback_to_all: bool = True


def _load_main_theme_config() -> MainThemeConfig:
    path = 'chanlun_system/main_theme_config.json'
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            valid = {k: v for k, v in data.items()
                     if k in MainThemeConfig.__dataclass_fields__}
            return MainThemeConfig(**valid)
        except Exception:
            pass
    return MainThemeConfig()


def classify_sector_tier(sector_name: str, hot_sector_list: list,
                         sector_mom: dict, config: MainThemeConfig,
                         pool_main_sectors: list = None,
                         pool_disaster_sectors: list = None) -> int:
    """板块三级分类: 1=主线 2=活跃 3=非活跃"""
    # 重灾区一票否决
    if pool_disaster_sectors and sector_name in pool_disaster_sectors:
        return 3

    # HotSectorAnalyzer阶段判定
    sector_info = None
    for hs in hot_sector_list:
        if hs.name == sector_name:
            sector_info = hs
            break

    if sector_info is None:
        # 不在HotSector列表中，检查板块扫描器是否标记为主线
        if pool_main_sectors and sector_name in pool_main_sectors:
            return 1
        mom = sector_mom.get(sector_name, 0)
        return 2 if mom > 0 else 3

    if sector_info.phase in config.main_theme_phases:
        return 1
    elif sector_info.phase in config.active_phases:
        return 2
    elif sector_info.phase in config.fading_phases:
        return 3
    else:
        if pool_main_sectors and sector_name in pool_main_sectors:
            return 1
        mom = sector_mom.get(sector_name, 0)
        return 2 if mom > 3 else 3


def compute_sector_tiers(daily_map: dict, sector_map: dict,
                         hot_sector_list: list, sector_mom: dict,
                         config: MainThemeConfig,
                         pool_main_sectors: list = None,
                         pool_disaster_sectors: list = None) -> dict:
    """计算所有股票的tier评分（纯评分，不过滤）

    Returns:
        {code: tier} 映射，tier 1/2/3 用于后续评分加成
    """
    tier_map = {}
    tier_counts = {1: 0, 2: 0, 3: 0}
    growth_promoted = 0

    for code in daily_map:
        sector = sector_map.get(code, '')
        if not sector or sector == '未知':
            tier_map[code] = 3
            tier_counts[3] += 1
            continue
        tier = classify_sector_tier(sector, hot_sector_list, sector_mom, config,
                                    pool_main_sectors, pool_disaster_sectors)
        if tier > 2 and sector in GROWTH_SECTORS:
            tier = 2
            growth_promoted += 1
        tier_map[code] = tier
        tier_counts[tier] += 1

    t1, t2, t3 = tier_counts[1], tier_counts[2], tier_counts[3]
    print(f'   [赛道评分] Tier1(主线)={t1} Tier2(活跃)={t2} Tier3(其他)={t3} '
          f'成长保护={growth_promoted}')

    main_names = [hs.name for hs in hot_sector_list
                  if hs.phase in config.main_theme_phases]
    if main_names:
        print(f'   [主线赛道] {", ".join(main_names)}')

    return tier_map


def load_sector_map():
    """加载行业映射（优先使用完整版，回退到旧版）"""
    # 优先加载完整版
    full_path = 'chanlun_system/full_sector_map.json'
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('stock_to_sector', {})

    # 回退到旧版
    path = 'chanlun_system/thshy_sector_map.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('stock_to_sector', {})
    return {}


def calc_sector_momentum(daily_map, sector_map, lookback=5):
    """计算各行业近N日涨幅"""
    sector_returns = {}
    for code, df in daily_map.items():
        if len(df) < lookback + 1:
            continue
        sector = sector_map.get(code)
        # [修复] sector_map为空时(KShare失败)，不过滤股票
        # 用股票代码本身代替行业做聚类，保证全市场都能参与动量计算
        if not sector:
            sector = f'_stock_{code}'
        ret = (df['close'].iloc[-1] / df['close'].iloc[-lookback-1] - 1) * 100
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append(ret)

    # 各行业中位数涨幅
    sector_median = {}
    for sector, rets in sector_returns.items():
        if len(rets) >= 3:  # 至少3只股票才有统计意义
            sector_median[sector] = np.median(rets)

    return sector_median


def score_technical(df, entry_price, stop_price, rr_ratio):
    """技术面评分 (0-100)"""
    score = 0

    # 1. MACD方向 (+20)
    if len(df) >= 35:
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd = (dif - dea) * 2
        if macd.iloc[-1] > 0:
            score += 10
        if dif.iloc[-1] > dif.iloc[-3]:
            score += 10  # DIF上升

    # 2. 量价配合 (+20)
    if len(df) >= 10:
        vol_ma5 = df['volume'].iloc[-5:].mean()
        vol_ma20 = df['volume'].iloc[-20:].mean() if len(df) >= 20 else vol_ma5
        if vol_ma20 > 0 and vol_ma5 > vol_ma20 * 1.2:
            score += 10  # 放量
        if df['close'].iloc[-1] > df['close'].iloc[-5]:
            score += 10  # 价涨

    # 3. R/R比 (+20)
    if rr_ratio > 5:
        score += 20
    elif rr_ratio > 3:
        score += 15
    elif rr_ratio > 1.5:
        score += 10
    elif rr_ratio > 0.5:
        score += 5

    # 4. 均线排列 (+20)
    if len(df) >= 60:
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]
        if ma5 > ma20 > ma60:
            score += 20  # 多头排列
        elif ma5 > ma20:
            score += 10

    # 5. 距离入场价位置 (+20)
    last_close = df['close'].iloc[-1]
    pct_from_entry = (last_close - entry_price) / entry_price * 100
    if -3 <= pct_from_entry <= 3:
        score += 20  # 接近入场价
    elif -5 <= pct_from_entry <= 5:
        score += 10

    return score


def _detect_weekly_trend(code, hs, daily_df=None):
    """检测周线趋势方向: bull/bear/range

    周线定方向: 缠论周线分析(笔/中枢/买卖点) + MA/MACD辅助

    Args:
        code: 股票代码
        hs: HybridSource实例
        daily_df: 日线DataFrame (如果提供则从日线resample，避免重新读TDX)

    Returns: (trend, score, weekly_rise_pct)
        weekly_rise_pct: 从周线最低点至今的涨幅百分比 (如 25.3 = 25.3%)
    """
    try:
        if daily_df is not None and len(daily_df) >= 120:
            df_w = daily_df.resample('W').agg({
                'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
        else:
            df_w = hs.get_kline(code, period='weekly')
        if len(df_w) < 30:
            return 'range', 0.0, 0.0

        close = df_w['close']
        low = df_w['low']
        last = close.iloc[-1]

        # === 周线涨幅: 从近期最低点(20周)到现在的涨幅 ===
        recent_low = low.iloc[-20:].min() if len(df_w) >= 20 else low.min()
        weekly_rise_pct = (last / recent_low - 1) * 100 if recent_low > 0 else 0.0

        # === 缠论周线分析 (核心) ===
        chanlun_score = _weekly_chanlun_score(df_w)

        # === MA/MACD辅助评分 ===
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1] if len(df_w) >= 20 else ma10

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_val = (dif - dea).iloc[-1] * 2

        tech_score = 0.0
        if last > ma5 > ma10:
            tech_score += 0.15
        if last > ma20:
            tech_score += 0.10
        if macd_val > 0:
            tech_score += 0.10
        if close.iloc[-1] > close.iloc[-5]:
            tech_score += 0.10

        # 综合评分: 缠论60% + 传统40%
        score = chanlun_score * 0.6 + tech_score * 0.4 / 0.45  # 归一化tech到0-1

        # 读取周线底分型标记
        weekly_bottom_fractal = getattr(_weekly_chanlun_score, 'weekly_bottom_fractal', False)
        weekly_target_gg = getattr(_weekly_chanlun_score, 'weekly_target_gg', None)
        weekly_bf_low = getattr(_weekly_chanlun_score, 'weekly_bf_low', None)

        # 计算盈亏比
        weekly_risk_reward = None
        if weekly_bf_low and weekly_target_gg and last > 0:
            risk = last - weekly_bf_low
            reward = weekly_target_gg - last
            if risk > 0:
                weekly_risk_reward = round(reward / risk, 1)
        _detect_weekly_trend.weekly_risk_reward = weekly_risk_reward

        if score >= 0.5:
            return 'bull', score, weekly_rise_pct, weekly_bottom_fractal
        elif score <= 0.15:
            return 'bear', score, weekly_rise_pct, weekly_bottom_fractal
        else:
            return 'range', score, weekly_rise_pct, weekly_bottom_fractal
    except Exception:
        return 'range', 0.3, 0.0, False


def _weekly_chanlun_score(df_w):
    """周线缠论打分: 笔方向 + 中枢位置 + 买卖点 + 背驰"""
    try:
        from core.kline import KLine
        from core.fractal import detect_fractals
        from core.stroke import generate_strokes
        from core.pivot import detect_pivots, PivotLevel
        from core.buy_sell_points import BuySellPointDetector
        from indicator.macd import MACD

        kline = KLine.from_dataframe(df_w, strict_mode=False)
        fractals = detect_fractals(kline)
        if len(fractals) < 4:
            return 0.3
        strokes = generate_strokes(kline, fractals, min_bars=3)
        if len(strokes) < 3:
            return 0.3

        pivots = detect_pivots(kline, strokes, level=PivotLevel.WEEK)
        last_close = float(df_w['close'].iloc[-1])

        score = 0.0

        # 1. 笔方向 (权重最高)
        if strokes:
            if not strokes[-1].is_down:
                score += 0.30  # 向上笔
            else:
                score -= 0.10  # 向下笔

        # 2. 中枢位置
        if pivots:
            last_p = pivots[-1]
            if last_close > last_p.zg:
                score += 0.25  # 中枢上方
            elif last_close > last_p.zd:
                score += 0.10  # 中枢内偏上
            else:
                score -= 0.10  # 中枢下方

        # 3. 周线买卖点
        try:
            closes_w = [k.close for k in kline]
            highs_w = [k.high for k in kline]
            lows_w = [k.low for k in kline]
            macd = MACD(pd.Series(closes_w))
            det = BuySellPointDetector(
                fractals, strokes, [], pivots, macd=macd,
                closes=closes_w, highs=highs_w, lows=lows_w,
            )
            buys, sells = det.detect_all()
            for bp in reversed(buys):
                if bp.index >= len(df_w) - 60:
                    if bp.point_type in ('2buy', '3buy'):
                        score += 0.20
                    elif bp.point_type == '1buy':
                        score += 0.10
                    if bp.divergence_ratio > 0:
                        score += 0.10
                    break
            # 最近有卖点 → 减分
            for sp in reversed(sells):
                if sp.index >= len(df_w) - 20:
                    score -= 0.10
                    break
        except Exception:
            pass

        # 5. 周线笔阶段 (回测验证: late>mid>early)
        if len(strokes) >= 2:
            last_s = strokes[-1]
            prev_s = strokes[-2]
            if prev_s.is_down and not last_s.is_down:
                up_gain = (last_s.end_value - last_s.start_value) / last_s.start_value * 100 if last_s.start_value > 0 else 0
                if up_gain >= 25:
                    score += 0.15  # late: 趋势确立, 加分
                elif up_gain >= 10:
                    score += 0.05  # mid
                else:
                    score -= 0.05  # early: 未确认, 减分

        # 6. 周线下跌笔末端底分型检测
        # 下跌笔中，底分型是笔的终点，顶分型之后才有底分型
        # 查找最近的底分型（不一定是最后一个分型）
        weekly_bottom_fractal = False
        if len(strokes) >= 1 and strokes[-1].is_down:
            total_bars = len(kline.processed_data)
            for frac in reversed(fractals):
                if frac.type.value == 'bottom':
                    dist = total_bars - frac.index
                    if dist <= 8:  # 最近8根周线(约2个月)内有底分型
                        weekly_bottom_fractal = True
                        score += 0.10
                    break  # 只看最近一个底分型

        # 将标记附加到函数属性，供调用方读取
        _weekly_chanlun_score.weekly_bottom_fractal = weekly_bottom_fractal

        # 周线中枢GG作为目标价, 底分型低点作为止损参考
        weekly_target_gg = None
        weekly_bf_low = None
        if pivots:
            weekly_target_gg = pivots[-1].gg
        if weekly_bottom_fractal:
            for frac in reversed(fractals):
                if frac.type.value == 'bottom':
                    total_bars2 = len(kline.processed_data)
                    if total_bars2 - frac.index <= 8:
                        weekly_bf_low = frac.low
                    break
        _weekly_chanlun_score.weekly_target_gg = weekly_target_gg
        _weekly_chanlun_score.weekly_bf_low = weekly_bf_low

        return max(0, min(1, score))
    except Exception:
        return 0.3


def _fetch_realtime_quotes_today(codes: list) -> dict:
    """通过腾讯实时行情接口获取当前价格（秒级）

    Returns: {code: {price, prev_close, chg_pct, volume, name}}
    """
    import urllib.request, re

    result = {}
    if not codes:
        return result

    # 腾讯实时行情：每批≤15个（过多会超时）
    BATCH = 15
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://finance.qq.com/'
    }
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i+BATCH]
        # 腾讯格式: sh000001,sz300936,bj...
        qstr = ','.join(batch)
        url = 'https://qt.gtimg.cn/q=' + qstr
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read().decode('gbk', errors='replace')
            for line in raw.strip().split('\n'):
                m = re.match(r'v_(?:sh|sz|bj)(\w+)=\"(\d+)~(.*)"', line)
                if m:
                    code = m.group(1)
                    parts = m.group(3).split('~')
                    if len(parts) < 5:
                        continue
                    try:
                        price = float(parts[3])
                        prev = float(parts[4])
                        result[code] = {
                            'price': price,
                            'prev_close': prev,
                            'chg_pct': (price - prev) / prev * 100 if prev else 0,
                            'volume': float(parts[5]) if parts[5] else 0,
                            'name': parts[1],
                        }
                    except (ValueError, IndexError):
                        continue
        except Exception:
            continue
    return result


# ==================== v7.3: MA250三级环境过滤 ====================

def _apply_ma250_filter(all_signals: list, market_env) -> list:
    """v7.3 MA250 3级环境过滤: 按买点类型决定环境权重"""
    if market_env is None:
        return all_signals

    ms = market_env.get_market_state()
    env_state = str(ms.state) if hasattr(ms, 'state') else 'BULL'

    pos_coef_map = {'BULL': 1.0, 'NEUTRAL': 0.6, 'BEAR': 0.3}
    pos_coef = pos_coef_map.get(env_state, 0.6)

    filtered = []
    env_counts = defaultdict(int)
    for sig in all_signals:
        sig_type = sig.get('signal_type', '')
        # sub1buy→1buy, quasi2buy→2buy for weight lookup
        lookup = sig_type
        if sig_type == 'sub1buy':
            lookup = '1buy'
        elif sig_type in ('quasi2buy', '2b3bbuy'):
            lookup = '2buy'

        env_weight = market_env.get_signal_weight(lookup) if hasattr(market_env, 'get_signal_weight') else 1.0
        if env_weight <= 0:
            env_counts[sig_type + '_blocked'] += 1
            continue

        sig['env_weight'] = env_weight
        sig['pos_coef'] = pos_coef
        sig['eff_regime'] = env_state
        filtered.append(sig)
        env_counts[sig_type] += 1

    regime_icon = {'BULL': '[UP]', 'NEUTRAL': '[--]', 'BEAR': '[DN]'}.get(env_state, '[??]')
    blocked = sum(v for k, v in env_counts.items() if '_blocked' in k)
    print(f'   [MA250过滤] regime={env_state}{regime_icon} 仓位={pos_coef:.0%} '
          f'过滤={blocked} 剩余={len(filtered)}')
    return filtered


def _detect_market_regime():
    """检测当前市场环境: BULL/NEUTRAL/BEAR

    基于上证指数MA250+斜率+20日动量三级判定。
    返回 MarketEnvironment 实例。
    """
    env = MarketEnvironment()
    ms = env.get_market_state()
    # 兼容旧接口: 映射到 strong/normal/weak
    _map = {'BULL': 'strong', 'NEUTRAL': 'normal', 'BEAR': 'weak'}
    env._compat_regime = _map.get(ms.state, 'normal')
    return env


def _classify_2buy_strength(df, pair, engine):
    """2买三档强度分类

    - strong: 2买3买重叠 (回踩不进中枢, 几乎不回踩) = 最强
    - medium: 类2买 (中枢内回踩) = 中等
    - weak:   中枢下2买 = 较弱
    """
    try:
        buy_idx = pair.get('2buy_idx', pair.get('sig_idx', 0))
        buy_price = pair.get('2buy_price', pair.get('entry_price', 0))
        first_buy_low = pair.get('1buy_low', buy_price * 0.95)
        n = len(df)

        # 需要中枢数据来判断
        bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)
        if len(strokes) < 6:
            return 'medium'

        # 找中枢
        pivots = []
        for i in range(len(strokes) - 2):
            s1, s2, s3 = strokes[i], strokes[i+1], strokes[i+2]
            highs = [max(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
            lows = [min(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
            zg = min(highs)
            zd = max(lows)
            if zg > zd:
                pivots.append({'zg': zg, 'zd': zd})

        if not pivots:
            return 'medium'

        # 找2买位置附近最近的中枢
        last_close = df['close'].iloc[-1] if buy_idx >= n else df['close'].iloc[min(buy_idx, n-1)]
        relevant_pivot = None
        for pv in pivots:
            if pv['zd'] <= buy_price <= pv['zg'] * 1.05:
                relevant_pivot = pv
                break

        if not relevant_pivot:
            # 2买价格不在任何中枢附近
            if buy_price < min(pv['zd'] for pv in pivots):
                return 'weak'   # 中枢下
            return 'medium'

        # 2买3买重叠检查: 回踩不进中枢 = 2买同时满足3买条件
        zg = relevant_pivot['zg']
        zd = relevant_pivot['zd']

        if first_buy_low >= zd:
            # 1买低点都不在中枢以下 = 回踩极浅 = 2买3买重叠
            return 'strong'
        elif buy_price >= zg:
            # 2买在中枢上方 = 2买3买重叠
            return 'strong'
        elif buy_price >= zd:
            # 2买在中枢内 = 类2买
            return 'medium'
        else:
            # 2买在中枢下 = 较弱
            return 'weak'
    except Exception:
        return 'medium'


def _find_quasi2buy_standalone(engine, code, df):
    """找类2买(quasi2buy)独立信号 — 回踩至中枢中部以上, 温和缩量

    类2买定义: 中枢内回踩到中枢中部以上位置企稳, 成交量温和萎缩。
    不同于标准2买(需要先有1买), quasi2buy是中枢震荡中的低风险买点。

    胜率~55% (v73), 置信度0.65, ATR止损×0.5
    """
    n = len(df)
    if n < 120:
        return []

    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']

    bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)
    if len(strokes) < 6:
        return []

    # 识别中枢
    pivots = []
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i+1], strokes[i+2]
        h = [max(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
        l = [min(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
        zg = min(h)
        zd = max(l)
        if zg > zd:
            # 扩展检测
            end_i = i + 2
            for j in range(i + 3, len(strokes)):
                sj_h = max(strokes[j]['start_val'], strokes[j]['end_val'])
                sj_l = min(strokes[j]['start_val'], strokes[j]['end_val'])
                if min(sj_h, zg) > max(sj_l, zd):
                    end_i = j
                else:
                    break
            pivots.append({
                'zg': zg, 'zd': zd,
                'mid': (zg + zd) / 2,
                'start_idx': min(*[strokes[k]['start_idx'] for k in range(i, end_i + 1)]),
                'end_idx': max(*[strokes[k]['end_idx'] for k in range(i, end_i + 1)]),
                'bi_count': end_i - i + 1,
            })
            i = end_i + 1
        else:
            i += 1

    if not pivots:
        return []

    results = []
    for pv in pivots:
        pv_end = pv['end_idx']
        mid = pv['mid']
        zg = pv['zg']
        zd = pv['zd']

        # 在中枢结束后找回踩到中枢中部以上的位置
        for j in range(pv_end + 1, min(pv_end + 30, n - 3)):
            pullback_low = low.iloc[j]
            # 回踩到中枢中部以上 (高于中枢50%位置)
            if pullback_low >= mid and pullback_low <= zg:
                # 量确认: 回踩缩量 (低于20日均量的80%)
                if j >= 20:
                    vol_ma20 = volume.iloc[j-20:j].mean()
                    vol_j = volume.iloc[j]
                    vol_shrink = vol_j < vol_ma20 * 0.80 if vol_ma20 > 0 else False
                else:
                    vol_shrink = False

                # 底分型确认 (当前K线低点 < 前后K线低点)
                has_bf = (j >= 1 and j < n - 1 and
                          low.iloc[j] < low.iloc[j-1] and
                          low.iloc[j] < low.iloc[j+1])

                if vol_shrink and has_bf:
                    results.append({
                        'signal_type': 'quasi2buy',
                        'entry_price': close.iloc[j],
                        'stop_price': zd,
                        'sig_idx': j,
                        'buy_strength': 'quasi2buy',
                        'pivot_zg': zg,
                        'pivot_zd': zd,
                        'confidence': 0.65,
                        'vol_shrink': True,
                    })
                    break  # 每个中枢只取第一个

    return results


# ============================================================
# 强三买7条件辅助检测函数
# ============================================================

def _check_support_structure(df, pullback_idx, pullback_low, pivot_zg):
    """条件2: 回踩位置是否有支撑结构

    支撑来源:
    - 前高/中枢ZG转化为支撑 (回踩在前ZG上方)
    - MA20/MA60在回踩位置附近 (距离<3%)
    """
    try:
        n = len(df)
        if pullback_idx < 30 or pullback_idx >= n:
            return False

        close = df['close']
        low = df['low']

        # 1. 前高支撑: 回踩位置之前的20日内高点作为支撑
        lookback = max(0, pullback_idx - 30)
        prev_highs = df['high'].iloc[lookback:pullback_idx]
        if len(prev_highs) > 0:
            recent_high = prev_highs.max()
            # 回踩低点在前高附近(下方3%以内) = 前高支撑
            if pullback_low >= recent_high * 0.97:
                return True

        # 2. MA支撑: MA20/MA60在回踩位置附近
        if pullback_idx >= 60:
            ma20 = close.iloc[pullback_idx - 20:pullback_idx].mean()
            ma60 = close.iloc[pullback_idx - 60:pullback_idx].mean()
            for ma in [ma20, ma60]:
                if ma > 0 and abs(pullback_low - ma) / ma < 0.03:
                    return True

        # 3. 中枢上沿支撑: 回踩在ZG附近
        if pivot_zg > 0 and abs(pullback_low - pivot_zg) / pivot_zg < 0.02:
            return True

        return False
    except Exception:
        return False


def _check_pullback_divergence(df, breakout_idx, pullback_idx):
    """条件3: 回撤时有背驰结构

    回撤段MACD柱面积递减 = 抛压衰竭 = 回撤背驰
    """
    try:
        if pullback_idx - breakout_idx < 5:
            return False

        close = df['close'].iloc[breakout_idx:pullback_idx + 1]
        if len(close) < 8:
            return False

        # 计算MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = 2 * (dif - dea)

        # 回撤段的负柱面积 (取绝对值)
        neg_hist = hist[hist < 0]
        if len(neg_hist) < 4:
            return False

        # 将负柱面积分为前后两半，比较面积
        mid = len(neg_hist) // 2
        first_half = abs(neg_hist.iloc[:mid].sum())
        second_half = abs(neg_hist.iloc[mid:].sum())

        # 后半段面积 < 前半段的80% = 衰竭
        if first_half > 0 and second_half < first_half * 0.8:
            return True

        return False
    except Exception:
        return False


def _check_top_micro_pivot(df, breakout_high_idx, breakout_high, pivot_zg, n_bars=5):
    """条件4: 出中枢那笔顶部区域有小级别中枢

    在突破高点前后n_bars根K线内，检查是否存在窄幅横盘区间:
    - 至少3根K线的高低点在突破高点的2%范围内
    - 形成小级别中枢 = 上方有引力结构
    """
    try:
        n = len(df)
        if breakout_high <= 0:
            return False

        # 在突破高点附近找窄幅横盘区间
        start = max(0, breakout_high_idx - n_bars)
        end = min(n, breakout_high_idx + n_bars + 1)

        if end - start < 5:
            return False

        segment = df.iloc[start:end]

        # 计算每根K线距离突破高点的偏差
        threshold = breakout_high * 0.02  # 2%范围内
        near_top_count = 0
        for _, bar in segment.iterrows():
            # K线整体在突破高点附近 (高点在高点的2%以内)
            if bar['high'] >= breakout_high - threshold and bar['low'] >= breakout_high - threshold * 2:
                near_top_count += 1

        # 至少3根K线在顶部区域横盘 = 小级别中枢
        if near_top_count >= 3:
            return True

        # 备选: 检测是否有局部双顶/三重顶 (高点接近)
        highs = segment['high'].values
        local_maxes = []
        for i in range(1, len(highs) - 1):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i+1]:
                local_maxes.append(highs[i])

        if len(local_maxes) >= 2:
            max_range = max(local_maxes) - min(local_maxes)
            if max_range < breakout_high * 0.015:  # 多个高点差距<1.5%
                return True

        return False
    except Exception:
        return False


def _check_breakout_strength(df, pivot_end_idx, breakout_idx):
    """条件5: 出中枢的那笔不背驰

    突破笔MACD面积 > 前一笔面积 = 力度增强 = 不背驰
    Returns: (bool, strength_ratio)
    """
    try:
        if breakout_idx - pivot_end_idx < 5:
            return False, 0.0

        close = df['close']
        n = len(close)

        # 计算MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = 2 * (dif - dea)

        # 突破段: 中枢结束 → 突破高点
        breakout_area = abs(hist.iloc[pivot_end_idx:breakout_idx + 1].sum())

        # 前一笔参考: 中枢内最近一个上涨段的MACD面积
        # 取中枢内正柱面积作为参考
        pivot_area = abs(hist.iloc[max(0, pivot_end_idx - 20):pivot_end_idx + 1]
                         [hist.iloc[max(0, pivot_end_idx - 20):pivot_end_idx + 1] > 0].sum())

        if pivot_area <= 0:
            # 无参考数据，用突破段自身是否为正来判断
            return breakout_area > 0, 1.0

        ratio = breakout_area / pivot_area
        # 突破面积 > 参考 = 力度增强 = 不背驰
        return ratio > 1.0, ratio

    except Exception:
        return False, 0.0


def _check_volume_pattern(df, pivot_end_idx, breakout_idx, pullback_idx):
    """条件7: 出中枢放量，回踩缩量

    突破段均量 > 回撤段均量 × 1.3 = 放量突破 + 缩量回踩
    Returns: (bool, vol_ratio)
    """
    try:
        if 'volume' not in df.columns:
            return False, 0.0

        vol = df['volume'].astype(float)
        n = len(vol)

        if pivot_end_idx >= n or breakout_idx >= n or pullback_idx >= n:
            return False, 0.0

        # 突破段均量: 中枢结束 → 突破高点
        breakout_vol = vol.iloc[pivot_end_idx:breakout_idx + 1].mean()

        # 回撤段均量: 突破高点 → 回踩低点
        pullback_vol = vol.iloc[breakout_idx:pullback_idx + 1].mean()

        if breakout_vol <= 0:
            return False, 0.0

        ratio = breakout_vol / pullback_vol if pullback_vol > 0 else 999

        # 突破量 > 回撤量 × 1.3 = 放量突破缩量回踩
        return ratio > 1.3, ratio

    except Exception:
        return False, 0.0


def _find_3buy_standalone(engine, code, df):
    """找日线3买信号 (突破中枢后回踩不进中枢)

    3买定义: 价格突破最近一个中枢ZG后回调, 低点不低于ZD

    3买三档强度:
      - 强三买: 回踩低点 > GG (中枢波动高点)
      - 标准三买: ZG < 回踩低点 <= GG
      - 弱三买: ZD < 回踩低点 <= ZG

    黄金分割加分: 回撤未跌破0.618 = 多头强
    """
    n = len(df)
    if n < 120:
        return []

    close = df['close']
    high = df['high']
    low = df['low']

    bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)
    if len(strokes) < 6:
        return []

    # 从笔中识别中枢: 3笔重叠区间 + 扩展检测
    pivots = []  # list of {zg, zd, start_idx, end_idx, bi_count, is_expanded}
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i+1], strokes[i+2]
        # 取3笔的价格区间
        highs = []
        lows = []
        for s in [s1, s2, s3]:
            highs.append(max(s['start_val'], s['end_val']))
            lows.append(min(s['start_val'], s['end_val']))
        zg = min(highs)  # 中枢上沿 = 重叠区间的最小高点
        zd = max(lows)   # 中枢下沿 = 重叠区间的最大低点
        if zg > zd:  # 有效中枢
            # 扩展检测: 后续笔是否仍与[ZD,ZG]重叠
            end_i = i + 2
            for j in range(i + 3, len(strokes)):
                sj_high = max(strokes[j]['start_val'], strokes[j]['end_val'])
                sj_low = min(strokes[j]['start_val'], strokes[j]['end_val'])
                if min(sj_high, zg) > max(sj_low, zd):  # 仍有重叠
                    end_i = j
                else:
                    break
            bi_count = end_i - i + 1
            is_expanded = bi_count >= 6
            end_idx = max(*[strokes[k]['end_idx'] for k in range(i, end_i + 1)])
            start_idx = min(*[strokes[k]['start_idx'] for k in range(i, end_i + 1)])
            pivots.append({
                'zg': zg, 'zd': zd,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'bi_count': bi_count,
                'is_expanded': is_expanded,
            })
            i = end_i + 1  # 跳过已处理的中枢笔
        else:
            i += 1

    if not pivots:
        return []

    # 中枢合并: 相邻中枢重叠时合并 (防止假3买)
    merged = [pivots[0]]
    for pv in pivots[1:]:
        prev = merged[-1]
        # 重叠条件: ZD1 < ZG2 and ZD2 < ZG1
        if prev['zd'] < pv['zg'] and pv['zd'] < prev['zg']:
            # 合并: 取更宽的区间
            prev['zg'] = max(prev['zg'], pv['zg'])
            prev['zd'] = min(prev['zd'], pv['zd'])
            prev['end_idx'] = max(prev['end_idx'], pv['end_idx'])
            prev['start_idx'] = min(prev['start_idx'], pv['start_idx'])
            prev['bi_count'] += pv['bi_count']
            prev['is_expanded'] = prev['bi_count'] >= 6
        else:
            merged.append(pv)
    pivots = merged

    results = []
    # 对每个中枢, 找突破后回踩不进中枢的位置
    for pv in pivots:
        zg = pv['zg']
        zd = pv['zd']
        pv_end = pv['end_idx']
        pv_start = pv['start_idx']

        # === 条件1修正: 正确计算GG/DD ===
        # GG = 中枢所有笔的最高点 (波动高点)
        # DD = 中枢所有笔的最低点 (波动低点)
        pivot_strokes = [s for s in strokes
                         if s['end_idx'] >= pv_start and s['start_idx'] <= pv_end]
        if not pivot_strokes:
            pivot_strokes = strokes[max(0, pivots.index(pv)):min(len(strokes), pivots.index(pv)+5)]

        gg = zg  # 默认值
        dd = zd
        if pivot_strokes:
            gg = max(max(s.get('high', max(s['start_val'], s['end_val'])),
                         max(s['start_val'], s['end_val'])) for s in pivot_strokes)
            dd = min(min(s.get('low', min(s['start_val'], s['end_val'])),
                         min(s['start_val'], s['end_val'])) for s in pivot_strokes)

        # 从中枢结束后开始找突破
        for j in range(pv_end + 1, n - 5):
            if high.iloc[j] > zg:  # 突破中枢上沿
                # 记录突破高点 (用于0.618计算)
                breakout_high = high.iloc[j]
                breakout_idx = j
                # 找突破后的回调低点
                for k in range(j + 1, min(j + 20, n)):
                    pullback_low = low.iloc[k]
                    if pullback_low > zd:  # 回踩不进中枢 = 3买
                        entry_price = close.iloc[k]
                        stop_price = zd  # 止损=中枢下沿

                        # === 黄金分割0.618检查 (条件6) ===
                        golden_pass = False
                        if breakout_high > zg:
                            retrace = (breakout_high - pullback_low) / (breakout_high - zg)
                            if retrace <= 0.618:
                                golden_pass = True

                        # === 强三买7条件检测 ===
                        # 条件1: 回撤不触碰下方中枢最高点(GG)
                        above_gg = pullback_low > gg

                        # 条件2: 有支撑结构
                        has_support = _check_support_structure(
                            df, k, pullback_low, zg)

                        # 条件3: 回撤时有背驰结构
                        pullback_div = _check_pullback_divergence(
                            df, breakout_idx, k)

                        # 条件4: 突破顶部有小级别中枢
                        top_micro = _check_top_micro_pivot(
                            df, breakout_idx, breakout_high, zg)

                        # 条件5: 出中枢那笔不背驰
                        breakout_strong, breakout_ratio = _check_breakout_strength(
                            df, pv_end, breakout_idx)

                        # 条件7: 出中枢放量，回踩缩量
                        vol_ok, vol_ratio = _check_volume_pattern(
                            df, pv_end, breakout_idx, k)

                        # 条件8: 实体突破 (收盘价>ZG，非影线)
                        solid_breakout = close.iloc[j] > zg

                        # 条件9: 中枢扩张 (≥6笔=震荡充分，突破更可靠)
                        zs_expanded = pv.get('is_expanded', False)

                        three_buy_checks = {
                            'above_gg': bool(above_gg),         # 条件1
                            'support': bool(has_support),       # 条件2
                            'pullback_div': bool(pullback_div), # 条件3
                            'top_micro': bool(top_micro),       # 条件4
                            'breakout_strong': bool(breakout_strong),  # 条件5
                            'golden_pass': bool(golden_pass),   # 条件6
                            'vol_pattern': bool(vol_ok),        # 条件7
                            'solid_breakout': bool(solid_breakout),  # 条件8
                            'zs_expanded': bool(zs_expanded),   # 条件9
                        }

                        # === 基于回测验证的加权评分 + 新增三要素 ===
                        # 原最优: top_micro=+2, golden_pass=-2, support=-1, 其余=0
                        # 新增: solid_breakout=+1, zs_expanded=+1
                        WEIGHTS = {
                            'top_micro': 2,          # 最强正贡献 — 顶部小中枢
                            'golden_pass': -2,       # 惩罚 — 回调太浅=追高
                            'support': -1,           # 惩罚 — 支撑检测太宽松,有支撑反而弱
                            'above_gg': 0,           # 无效
                            'pullback_div': 0,       # 无效
                            'breakout_strong': 0,    # 无效
                            'vol_pattern': 0,        # 无效
                            'solid_breakout': 1,     # 实体突破 — 突破有效性加分
                            'zs_expanded': 1,        # 扩张中枢 — 震荡充分后突破更可靠
                        }
                        weighted_score = sum(
                            WEIGHTS[k] * (1 if v else 0)
                            for k, v in three_buy_checks.items()
                        )
                        passed_count = sum(1 for v in three_buy_checks.values() if v)

                        # 加权评分分类 (基于网格搜索最优阈值)
                        if weighted_score >= 2:
                            strength = 'strong'      # 顶部小中枢+非追高: 57.7%胜率,+4.08%
                        elif weighted_score >= 1:
                            strength = 'standard'    # 有关键条件但不够强
                        else:
                            strength = 'normal'

                        results.append({
                            'signal_type': '3buy',
                            'entry_price': entry_price,
                            'stop_price': stop_price,
                            'sig_idx': k,
                            'buy_strength': strength,
                            'golden_ratio_pass': golden_pass,
                            'pivot_zg': zg,
                            'pivot_zd': zd,
                            'pivot_gg': gg,
                            'breakout_high': breakout_high,
                            'pullback_low': pullback_low,
                            'zs_bi_count': pv.get('bi_count', 3),
                            'zs_expanded': zs_expanded,
                            'solid_breakout': solid_breakout,
                            'three_buy_checks': three_buy_checks,
                            'three_buy_passed': passed_count,
                            'three_buy_weighted': weighted_score,
                            'breakout_ratio': round(breakout_ratio, 2),
                            'vol_ratio': round(vol_ratio, 2),
                            'confidence': 0.85 if strength == 'strong' else (0.75 if strength == 'standard' else 0.60),
                        })
                        break  # 每个中枢只取第一个3买
                break  # 每个中枢只看一次突破

    # === 中枢法3买: P2.ZD > P1.ZG (第二中枢形成在第一中枢上方) ===
    if len(pivots) >= 2:
        for pi in range(len(pivots) - 1):
            p1 = pivots[pi]
            p2 = pivots[pi + 1]
            if p2['zd'] <= p1['zg']:
                continue
            if p2['end_idx'] <= p1['start_idx']:
                continue

            # 3买价格 = P2内最后一笔向下笔的终点（回调低点）
            entry_idx = p2['end_idx']
            entry_price = close.iloc[entry_idx] if entry_idx < n else close.iloc[-1]
            stop_price = p2['zd']  # 止损=P2下沿

            # gap = P2.ZD - P1.ZG 越大越强
            gap = (p2['zd'] - p1['zg']) / p1['zg'] if p1['zg'] > 0 else 0

            # === 质量检查 (与标准法相同条件) ===
            zg = p1['zg']
            zd = p1['zd']
            gg = p1.get('gg', zg)
            pv_end = p1['end_idx']
            breakout_idx = p2['start_idx']

            # 条件8: 实体突破
            solid_breakout = entry_idx < n and close.iloc[entry_idx] > zg

            # 条件9: 中枢扩张
            zs_expanded = p2.get('is_expanded', False)

            # 条件4: 顶部小中枢 (P2宽度 < P1宽度的50%)
            p1_width = gg - zd
            p2_width = (p2.get('gg', p2['zg']) - p2['zd'])
            top_micro = p1_width > 0 and p2_width < p1_width * 0.5

            # 条件6: 黄金分割
            golden_pass = False
            if p1_width > 0 and entry_idx > pv_end:
                run_up_h = df['high'].iloc[pv_end:entry_idx + 1].max()
                pullback = run_up_h - entry_price
                run_up = run_up_h - zd
                if run_up > 0:
                    golden_pass = pullback < run_up * 0.382

            # 条件7: 出中枢放量，回踩缩量
            vol_ok, vol_ratio = _check_volume_pattern(
                df, pv_end, breakout_idx, entry_idx)

            three_buy_checks = {
                'above_gg': False,
                'support': False,
                'pullback_div': False,
                'top_micro': bool(top_micro),
                'breakout_strong': False,
                'golden_pass': bool(golden_pass),
                'vol_pattern': bool(vol_ok),
                'solid_breakout': bool(solid_breakout),
                'zs_expanded': bool(zs_expanded),
            }

            # 加权评分 (同标准法权重)
            WEIGHTS = {
                'top_micro': 2,
                'golden_pass': -2,
                'support': -1,
                'above_gg': 0,
                'pullback_div': 0,
                'breakout_strong': 0,
                'vol_pattern': 0,
                'solid_breakout': 1,
                'zs_expanded': 1,
            }
            weighted_score = sum(
                WEIGHTS[k] * (1 if v else 0)
                for k, v in three_buy_checks.items()
            )
            passed_count = sum(1 for v in three_buy_checks.values() if v)

            if gap > 0.10:
                strength = 'strong'
            elif gap > 0.03:
                strength = 'standard'
            else:
                strength = 'normal'

            results.append({
                'signal_type': '3buy',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'sig_idx': entry_idx,
                'buy_strength': strength,
                'golden_ratio_pass': golden_pass,
                'pivot_zg': p1['zg'],
                'pivot_zd': p1['zd'],
                'pivot_gg': p1.get('gg', p1['zg']),
                'breakout_high': p2.get('gg', p2['zg']),
                'pullback_low': p2['zd'],
                'zs_bi_count': p2.get('bi_count', 3),
                'zs_expanded': zs_expanded,
                'solid_breakout': solid_breakout,
                'three_buy_checks': three_buy_checks,
                'three_buy_passed': passed_count,
                'three_buy_weighted': weighted_score,
                'breakout_ratio': 0,
                'vol_ratio': round(vol_ratio, 2),
                'pivot_method': True,
                'confidence': 0.85 if strength == 'strong' else (0.75 if strength == 'standard' else 0.60),
            })

    # 走势类型分类 — 附加到每个3买信号
    if pivots:
        trend_result = classify_trend_type(pivots)
        for r in results:
            r['trend_type'] = trend_result.current_type.value
            r['trend_strength'] = round(trend_result.trend_strength, 3)

    return results


def _find_sub1buy_standalone(engine, code, df):
    """找日线盘整背驰1买 (sub1B)

    盘整背驰: 中枢震荡中，离开中枢的向下笔出现MACD面积背驰，
    但不要求之前有更高中枢（无下跌趋势）。

    检测逻辑:
    1. 识别中枢
    2. 找无下跌趋势的中枢（盘整中枢）
    3. 离开中枢的向下笔MACD面积 < 进入中枢的向下笔MACD面积
    """
    n = len(df)
    if n < 120:
        return []

    close = df['close']
    low = df['low']
    high = df['high']

    bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)
    if len(strokes) < 6:
        return []

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    # 检测中枢 (复用3买的逻辑)
    pivots = []
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i+1], strokes[i+2]
        highs = [max(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
        lows = [min(s['start_val'], s['end_val']) for s in [s1, s2, s3]]
        zg = min(highs)
        zd = max(lows)
        if zg > zd:
            end_i = i + 2
            for j in range(i + 3, len(strokes)):
                sj_high = max(strokes[j]['start_val'], strokes[j]['end_val'])
                sj_low = min(strokes[j]['start_val'], strokes[j]['end_val'])
                if min(sj_high, zg) > max(sj_low, zd):
                    end_i = j
                else:
                    break
            end_idx = max(*[strokes[k]['end_idx'] for k in range(i, end_i + 1)])
            start_idx = min(*[strokes[k]['start_idx'] for k in range(i, end_i + 1)])
            pivots.append({
                'zg': zg, 'zd': zd,
                'start_idx': start_idx, 'end_idx': end_idx,
                'stroke_start': i, 'stroke_end': end_i,
                'bi_count': end_i - i + 1,
            })
            i = end_i + 1
        else:
            i += 1

    if not pivots:
        return []

    # 中枢合并: 相邻中枢重叠时合并
    merged = [pivots[0]]
    for pv in pivots[1:]:
        prev = merged[-1]
        if prev['zd'] < pv['zg'] and pv['zd'] < prev['zg']:
            prev['zg'] = max(prev['zg'], pv['zg'])
            prev['zd'] = min(prev['zd'], pv['zd'])
            prev['end_idx'] = max(prev['end_idx'], pv['end_idx'])
            prev['start_idx'] = min(prev['start_idx'], pv['start_idx'])
            prev['bi_count'] += pv['bi_count']
        else:
            merged.append(pv)
    pivots = merged

    results = []
    for pi, pv in enumerate(pivots):
        # 跳过有下跌趋势的中枢（标准1buy处理）
        has_downtrend = any(
            prev['zd'] > pv['zg'] for prev in pivots[:pi]
        )
        if has_downtrend:
            continue

        # 找离开中枢的向下笔
        for si in range(pv['stroke_end'] + 1, len(strokes)):
            s = strokes[si]
            if s['start_type'] != 'top' or s['end_type'] != 'bottom':
                continue
            s_low = min(s['start_val'], s['end_val'])
            if s_low >= pv['zd']:
                continue  # 未离开中枢

            # MACD面积背驰检测
            leave_start = max(0, s['start_idx'] - 1)
            leave_end = min(n - 1, s['end_idx'] + 1)
            leave_area = abs(hist.iloc[leave_start:leave_end+1].clip(upper=0).sum())

            # 找进入中枢的向下笔MACD面积
            enter_area = 0
            for ei in range(pv['stroke_start'], pv['stroke_end'] + 1):
                es = strokes[ei]
                if es['start_type'] == 'top' and es['end_type'] == 'bottom':
                    e_start = max(0, es['start_idx'] - 1)
                    e_end = min(n - 1, es['end_idx'] + 1)
                    e_area = abs(hist.iloc[e_start:e_end+1].clip(upper=0).sum())
                    enter_area = max(enter_area, e_area)

            if enter_area > 0 and leave_area < enter_area * 0.8:
                # 背驰确认：离开面积 < 进入面积 × 0.8
                entry_idx = s['end_idx']
                if entry_idx >= n:
                    continue
                entry_price = close.iloc[entry_idx]
                stop_price = s_low * 0.99
                ratio = leave_area / enter_area

                results.append({
                    'signal_type': 'sub1buy',
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'sig_idx': entry_idx,
                    'pivot_zg': pv['zg'],
                    'pivot_zd': pv['zd'],
                    'divergence_ratio': round(ratio, 3),
                    'confidence': min(0.75, 0.40 + (1.0 - ratio) * 0.35),
                })

            break  # 每个中枢只看第一笔离开

    return results


def _mp_worker(args):
    """多进程Worker: 每个进程独立创建SignalEngine, 分析单只股票"""
    code, sdata, mkt_regime, cutoff = args
    import pandas as pd
    df = pd.DataFrame({
        'open': sdata['open'], 'high': sdata['high'],
        'low': sdata['low'], 'close': sdata['close'],
        'volume': sdata['volume'],
    }, index=sdata['index'])

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / 'chanlun_unified'))
    from signal_engine_cc15 import SignalEngine
    from backtest_cc15_mtf import find_daily_1buy_2buy
    engine = SignalEngine()
    engine.dynamic_pool_enabled = False
    engine.momentum_factor_enabled = False
    engine.vol_regime_enabled = False

    # TD Sequential分析 (一次性，用于所有信号)
    # TD Sequential分析 (一次性，用于所有信号)
    td_result = None
    try:
        sys.path.insert(0, '.')
        from core.td_sequential import analyze_td, td_confirm_buy
        td_result = analyze_td(
            df['high'].tolist(), df['low'].tolist(), df['close'].tolist())
    except Exception:
        pass

    # 计算日线走势类型（用于2买过滤）
    _daily_trend_type = ''
    try:
        from core.pivot import detect_pivots
        _, pivots_tmp = detect_pivots(engine.strokes)
        if pivots_tmp:
            _tr = classify_trend_type(pivots_tmp)
            _daily_trend_type = _tr.current_type.value
    except Exception:
        pass

    signals = []
    try:
        pairs = find_daily_1buy_2buy(engine, code, df)
        for p in pairs:
            if p['2buy_idx'] >= len(df):
                continue
            # 2买过滤: 排除日线下跌趋势（与3买一致）
            if _daily_trend_type == 'down':
                continue
            # 2买过滤: 波动率<3%的横盘股2买胜率42% vs >3%的51%
            idx_2b = p['2buy_idx']
            if idx_2b >= 20:
                _atr20 = 0
                for _j in range(idx_2b - 19, idx_2b + 1):
                    _h, _l, _pc = float(df['high'].iloc[_j]), float(df['low'].iloc[_j]), float(df['close'].iloc[_j - 1]) if _j > 0 else _l
                    _atr20 += max(_h - _l, abs(_h - _pc), abs(_l - _pc))
                _atr20 /= 20
                _volatility = _atr20 / float(df['close'].iloc[idx_2b]) if float(df['close'].iloc[idx_2b]) > 0 else 0
                if _volatility < 0.03:
                    continue
            sig_date = df.index[p['2buy_idx']]
            if sig_date >= pd.Timestamp(cutoff):
                p['code'] = code
                p['signal_type'] = '2buy'
                p['entry_price'] = p['2buy_price']
                p['stop_price'] = p['1buy_low']
                p['sig_idx'] = p['2buy_idx']
                p['sig_date'] = sig_date
                p['buy_strength'] = _classify_2buy_strength(df, p, engine)
                p['golden_ratio_pass'] = False
                # 2buy置信度: 基于回踩深度和量价
                _2b_conf = 0.5
                if p['buy_strength'] == 'strong':
                    _2b_conf = 0.85
                elif p['buy_strength'] == 'medium':
                    _2b_conf = 0.75
                else:
                    _2b_conf = 0.60
                p['confidence'] = p.get('confidence', _2b_conf)
                signals.append(p)

        if mkt_regime != 'strong':
            for p in pairs:
                idx = p.get('1buy_idx', -1)
                if idx < 0 or idx >= len(df):
                    continue
                sig_date = df.index[idx]
                if sig_date >= pd.Timestamp(cutoff):
                    signals.append({
                        'code': code, 'signal_type': '1buy',
                        'entry_price': p.get('1buy_price', df['close'].iloc[idx]),
                        'stop_price': p.get('1buy_low', df['low'].iloc[idx]),
                        'sig_idx': idx, 'sig_date': sig_date,
                        'confidence': p.get('confidence', 0.5),
                    })

        threes = _find_3buy_standalone(engine, code, df)
        for s in threes:
            if s.get('trend_type') == 'down':
                continue
            if s['sig_idx'] < len(df):
                sig_date = df.index[s['sig_idx']]
                if sig_date >= pd.Timestamp(cutoff):
                    s['code'] = code
                    s['sig_date'] = sig_date
                    signals.append(s)

        # 盘整背驰1买 (sub1B) — 所有行情均可检测
        sub1s = _find_sub1buy_standalone(engine, code, df)
        for s in sub1s:
            if s['sig_idx'] < len(df):
                sig_date = df.index[s['sig_idx']]
                if sig_date >= pd.Timestamp(cutoff):
                    s['code'] = code
                    s['sig_date'] = sig_date
                    signals.append(s)

        # 类2买 (quasi2buy) — 中枢内回踩中部以上+缩量
        quasi2s = _find_quasi2buy_standalone(engine, code, df)
        for s in quasi2s:
            if s['sig_idx'] < len(df):
                sig_date = df.index[s['sig_idx']]
                if sig_date >= pd.Timestamp(cutoff):
                    s['code'] = code
                    s['sig_date'] = sig_date
                    signals.append(s)
    except Exception:
        pass

    # TD Sequential确认 — 给每个信号打TD分
    if td_result is not None:
        for s in signals:
            try:
                sig_idx = s.get('sig_idx', -1)
                sig_type = s.get('signal_type', '')
                if sig_idx >= 0:
                    boost, detail = td_confirm_buy(td_result, sig_idx, sig_type)
                    s['td_boost'] = boost
                    s['td_detail'] = detail
                    # TD卖出风险: 买入信号附近出现卖出倒计时
                    sell_risk = td_result.get_sell_score_at(sig_idx, lookback=5)
                    s['td_sell_risk'] = round(sell_risk, 3)
            except Exception:
                pass

    # ATR分段止损 (v73: 按买点类型使用不同ATR倍数)
    _atr_mult_map = {
        '1buy': 0.75, '2buy': 1.50, '3buy': 0.75,
        'sub1buy': 0.50, 'quasi2buy': 0.50, 'quasi3buy': 0.75,
        '2b3bbuy': 1.50, 'consolidationB': 0.50,
    }
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    for s in signals:
        try:
            sig_idx = s.get('sig_idx', -1)
            if sig_idx < 14 or sig_idx >= len(df):
                continue
            # True Range ATR(14)
            atr_sum = 0.0
            for j in range(sig_idx - 13, sig_idx + 1):
                tr = max(highs[j] - lows[j],
                         abs(highs[j] - closes[j - 1]),
                         abs(lows[j] - closes[j - 1]))
                atr_sum += tr
            atr14 = atr_sum / 14
            sig_type = s.get('signal_type', '2buy')
            mult = _atr_mult_map.get(sig_type, 0.75)
            entry = s.get('entry_price', closes[sig_idx])
            s['atr_stop_price'] = round(entry - atr14 * mult, 3)
            s['atr_multiplier'] = mult
        except Exception:
            pass

    return signals


def scan_enhanced(pool='tdx_all', lookback_days=30, min_price=3.0, max_price=200.0,
                  top_n=10):
    """增强版扫描"""
    hs = HybridSource()
    t0 = time.time()

    print(f'=== MTF V3 增强版扫描 ===')
    print(f'股票池: {pool}  回看: {lookback_days}天  价格: {min_price}-{max_price}')
    print()

    # 1. 获取股票代码
    print('[1] 获取股票池...')
    from chanlun_unified.stock_pool import StockPoolManager
    spm = StockPoolManager()
    codes = spm.get_pool(pool)
    pure_codes = [c.split('.')[0] for c in codes]
    print(f'   {len(pure_codes)} 只')

    # 2. 加载行业映射
    print('[2] 加载行业数据...')
    sector_map = load_sector_map()
    print(f'   行业映射: {len(sector_map)} 只')

    # 3. 加载日线数据 (向量化解析 + pickle缓存)
    print('[3] 加载日线数据...')
    t_load_start = time.time()
    daily_map = hs.load_all_daily(pure_codes, min_price=min_price,
                                   max_price=max_price, min_bars=200)
    t_load = time.time() - t_load_start
    print(f'   日线数据: {len(daily_map)} 只 ({t_load:.1f}s)')

    # 4. 计算行业动量
    print('[4] 计算行业动量...')
    sector_mom = calc_sector_momentum(daily_map, sector_map)
    # 只保留前50%的行业
    if sector_mom:
        median_mom = np.median(list(sector_mom.values()))
        hot_sectors = {s: r for s, r in sector_mom.items() if r >= median_mom}
    else:
        hot_sectors = set()
        median_mom = 0
    print(f'   行业数: {len(sector_mom)}  强势行业(>中位数{median_mom:.1f}%): {len(hot_sectors)}')

    # 4.5 热点板块识别 (本地TDX)
    print('[4.5] 热点板块识别...')
    try:
        from data.hot_sector_analyzer import HotSectorAnalyzer
        hsa = HotSectorAnalyzer()
        hot_sector_list = hsa.identify_hot_sectors(top_n=10)
        hsa.save_results(hot_sector_list)
        hot_sector_names = {s.name for s in hot_sector_list[:5]}
        hot_sector_stock_map = {}
        for s in hot_sector_list[:5]:
            for code in s.all_codes:
                # 统一用纯数字key (000001, 600519)
                num = code[2:] if code[:2] in ('sh', 'sz') else code
                hot_sector_stock_map[num] = s.name
        print(f'   TOP5热点: {", ".join(s.name + "(" + s.phase + ")" for s in hot_sector_list[:5])}')
    except Exception as e:
        print(f'   热点板块识别跳过: {e}')
        hot_sector_names = set()
        hot_sector_stock_map = {}
        hot_sector_list = []

    # 4.55 读取今日板块候选池扫描结果 (补充主线来源)
    sector_pool_data = {}
    sector_pool_path = f'signals/sector_pool_{datetime.now().strftime("%Y-%m-%d")}.json'
    if os.path.exists(sector_pool_path):
        try:
            with open(sector_pool_path, 'r', encoding='utf-8') as f:
                sector_pool_data = json.load(f)
            pool_main = sector_pool_data.get('main_sectors', [])
            if pool_main:
                print(f'   板块扫描器主线: {", ".join(pool_main[:5])}')
        except Exception:
            pass

    # 4.6 主线赛道评分 (纯评分，不过滤)
    mt_config = _load_main_theme_config()
    print('[4.6] 主线赛道评分...')
    _pool_main = sector_pool_data.get('main_sectors', [])
    _pool_disaster = list(sector_pool_data.get('disaster_sectors', {}).keys())
    tier_map = compute_sector_tiers(
        daily_map, sector_map, hot_sector_list, sector_mom, mt_config,
        pool_main_sectors=_pool_main,
        pool_disaster_sectors=_pool_disaster)

    # 4.7 快速粗筛: 排除明显无买点的股票 (加速CC15)
    print('[4.7] 快速粗筛...')
    prefiltered_map = {}
    for code, df in daily_map.items():
        n = len(df)
        if n < 120:
            continue
        close = df['close']
        low = df['low']
        high = df['high']
        # 条件1: 最近20日不创新低 (排除持续下跌)
        if close.iloc[-1] <= low.iloc[-20:].min():
            continue
        # 条件2: 最近5日至少1天收阳
        if all(close.iloc[-i] <= df['open'].iloc[-i] for i in range(1, 6)):
            continue
        # 条件3: 价格在60日均线附近或上方 (排除远低于均线的)
        ma60 = close.rolling(60).mean().iloc[-1]
        if pd.notna(ma60) and close.iloc[-1] < ma60 * 0.85:
            continue
        # 条件4: MA20在MA60附近或上方 (中期趋势非空头)
        if n >= 60:
            ma20 = close.rolling(20).mean().iloc[-1]
            if pd.notna(ma60) and pd.notna(ma20) and ma20 < ma60 * 0.92:
                continue
        # 条件5: MACD至少DIF不深度死叉 (DIF > -历史波动率的20%)
        if n >= 35:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            if dif.iloc[-1] < dif.iloc[-60:].min() * 0.5:
                continue
        prefiltered_map[code] = df
    print(f'   粗筛后: {len(prefiltered_map)}/{len(daily_map)} 只')

    # 5. 缠论分析 + 找所有买点(1买/2买/3买)
    # 直接创建引擎, 跳过CC15的动态池/动量排名/组合管理 (扫描不需要回测管道)
    print('[5] 缠论分析 + 识别买点 (多进程)...')
    t_scan_start = time.time()

    # === 规则5: 检测市场环境 ===
    market_env = _detect_market_regime()
    market_regime = market_env._compat_regime
    print(f'   {market_env.get_summary()}')

    cutoff_ts = datetime.now() - timedelta(days=lookback_days)

    # === 增量缓存: 用数据指纹跳过未变动的股票 ===
    import pickle as _pickle, hashlib as _hashlib
    _sig_cache_dir = os.path.join('.claude', 'cache')
    os.makedirs(_sig_cache_dir, exist_ok=True)
    _sig_cache_file = os.path.join(_sig_cache_dir, 'signal_cache.pkl')
    _sig_cache = {}
    if os.path.exists(_sig_cache_file):
        try:
            with open(_sig_cache_file, 'rb') as _f:
                _sig_cache = _pickle.load(_f)
        except Exception:
            pass
    _cache_date = _sig_cache.get('_date', '')
    _today_str = datetime.now().strftime('%Y-%m-%d')

    def _data_fp(df):
        return _hashlib.md5(df['close'].iloc[-5:].values.tobytes()).hexdigest()[:12]

    codes_to_analyze = []
    cached_signals = []
    for code, df in prefiltered_map.items():
        fp = _data_fp(df)
        cached = _sig_cache.get(code)
        if _cache_date == _today_str and cached and cached.get('fp') == fp:
            cached_signals.extend(cached['signals'])
        else:
            codes_to_analyze.append(code)

    _cache_hits = len(prefiltered_map) - len(codes_to_analyze)
    if _cache_hits > 0:
        print(f'   信号缓存: {_cache_hits}/{len(prefiltered_map)} 命中, '
              f'{len(codes_to_analyze)} 只需分析')

    # 准备轻量数据用于多进程传输 (仅未缓存的部分)
    stock_data = {}
    for code in codes_to_analyze:
        df = prefiltered_map[code]
        stock_data[code] = {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values,
            'index': df.index,
        }

    # 分批提交到进程池
    new_signals = []
    if codes_to_analyze:
        from multiprocessing import Pool
        n_workers = min(os.cpu_count() or 4, 8)
        task_args = [(code, stock_data[code], market_regime, cutoff_ts)
                     for code in codes_to_analyze]

        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_mp_worker, task_args, chunksize=50):
                new_signals.extend(result)

        # 更新缓存 (从new_signals中提取code)
        sig_by_code = defaultdict(list)
        for sig in new_signals:
            sig_by_code[sig.get('code', '')].append(sig)
        for code, sigs in sig_by_code.items():
            if code in prefiltered_map:
                _sig_cache[code] = {
                    'fp': _data_fp(prefiltered_map[code]),
                    'signals': sigs,
                }

        # 更新缓存
        _sig_cache['_date'] = _today_str
        try:
            with open(_sig_cache_file, 'wb') as _f:
                _pickle.dump(_sig_cache, _f, protocol=_pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    all_signals = cached_signals + new_signals

    # === v7.3: MA250三级环境过滤 (替代E-version) ===
    if all_signals:
        all_signals = _apply_ma250_filter(all_signals, market_env)

    # 按类型统计
    type_counts = {}
    for s in all_signals:
        t = s.get('signal_type', '?')
        type_counts[t] = type_counts.get(t, 0) + 1
    t_scan = time.time() - t_scan_start
    print(f'   最近{lookback_days}天信号: {dict(type_counts)} 共{len(all_signals)}个 '
          f'({t_scan:.1f}s, 缓存{_cache_hits}+分析{len(codes_to_analyze)})')

    if not all_signals:
        print('无近期买点信号')
        return []

    # 6. 30min确认 + 评分
    print('[6] 30分钟确认 + 综合评分...')
    t_30min_start = time.time()
    results = []
    scanned = set()

    # 5.5. TD Sequential确认 (主进程, 避免multiprocessing import问题)
    t_td_start = time.time()
    td_codes_done = set()
    for item in all_signals:
        code = item['code']
        if code in td_codes_done:
            continue
        td_codes_done.add(code)
        df = daily_map.get(code)
        if df is None:
            df = prefiltered_map.get(code)
        if df is None or len(df) < 30:
            continue
        try:
            from core.td_sequential import analyze_td, td_confirm_buy
            td_result = analyze_td(
                df['high'].tolist(), df['low'].tolist(), df['close'].tolist())
            # 给该股票的所有信号打TD分
            for s in all_signals:
                if s['code'] == code:
                    sig_idx = s.get('sig_idx', -1)
                    if sig_idx >= 0:
                        boost, detail = td_confirm_buy(td_result, sig_idx, s.get('signal_type', ''))
                        s['td_boost'] = boost
                        s['td_detail'] = detail
        except Exception:
            pass
    td_hit = sum(1 for s in all_signals if s.get('td_boost', 0) != 0)
    print(f'   [5.5] TD Sequential: {td_hit}/{len(all_signals)} signals affected ({time.time()-t_td_start:.1f}s)')

    # 6a. v7.3: 无周线过滤, MA250已在Step 5.5完成
    print('   [6a] MA250环境已过滤, 跳过周线...')
    weekly_passed = all_signals  # all_signals已通过MA250过滤
    print(f'   候选信号: {len(weekly_passed)}个')

    if not weekly_passed:
        print('   无股票通过周线过滤')
        return []

    # 6b. 获取30min数据 (TDX本地5min→30min, Sina fallback)
    print(f'   [6b] 获取30min数据 ({len(weekly_passed)}只)...')

    data_30m = {}
    tdx_30m_start = time.time()

    # Phase 1: TDX本地5min→30min (极快)
    codes_need_sina = []
    for item in weekly_passed:
        code = item['code']
        try:
            df5 = hs._read_tdx_5min(code)
            if len(df5) >= 100:
                df30 = df5.resample('30min').agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                if len(df30) >= 100:
                    data_30m[code] = df30
                    continue
        except Exception:
            pass
        codes_need_sina.append(code)

    tdx_count = len(data_30m)
    print(f'     TDX本地: {tdx_count}/{len(weekly_passed)} ({time.time()-tdx_30m_start:.1f}s)', flush=True)

    # Phase 2: Sina HTTP fallback (仅本地缺失的)
    if codes_need_sina:
        print(f'     Sina fallback: {len(codes_need_sina)}只...', flush=True)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_30min_one(code):
            try:
                df = fetch_sina_30min(code)
                return (code, df)
            except Exception:
                return (code, pd.DataFrame())

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(_fetch_30min_one, code): code
                       for code in codes_need_sina}
            for f in as_completed(futures):
                code, df = f.result()
                if len(df) >= 100:
                    data_30m[code] = df

    valid_30m = sum(1 for df in data_30m.values() if len(df) >= 100)
    print(f'   30min获取完成: {valid_30m}/{len(weekly_passed)} 有效 (TDX={tdx_count}, Sina={valid_30m-tdx_count})')

    # 6c. 批量获取实时报价
    print('   [6c] 批量获取实时报价...')
    quote_codes = []
    for item in weekly_passed:
        code = item['code']
        pure = code
        if not pure.endswith('.SZ') and not pure.endswith('.SH'):
            pure = code + ('.SH' if code[0] in ('6', '9') else '.SZ')
        quote_codes.append(pure)

    quote_map = {}
    try:
        batch_size = 200
        for i in range(0, len(quote_codes), batch_size):
            batch = quote_codes[i:i+batch_size]
            q = hs.get_realtime_quote(batch)
            if len(q) > 0:
                for _, row in q.iterrows():
                    c = row.get('code', '')
                    # 去掉.SH/.SZ后缀匹配
                    pure = c.replace('.SH', '').replace('.SZ', '')
                    quote_map[pure] = row
    except Exception:
        pass
    print(f'   实时报价: {len(quote_map)}/{len(weekly_passed)} 获取成功')

    # 6d. 30min分析 + 评分 (纯计算, 无网络)
    # 6d. v7.3: 30min入场确认 + 评分
    print('   [6d] 30min入场确认 + 评分...')
    for item in weekly_passed:
        code = item['code']
        df_30 = data_30m.get(code, pd.DataFrame())

        # 获取报价
        pure = code
        if not pure.endswith('.SZ') and not pure.endswith('.SH'):
            pure = code + ('.SH' if code[0] in ('6', '9') else '.SZ')
        q_row = quote_map.get(code, None)
        if q_row is not None:
            name = q_row.get('name', code)
            price = float(q_row.get('price', 0))
            pct = float(q_row.get('pct_chg', 0))
        else:
            name, price, pct = code, 0, 0

        # 行业信息
        sector = sector_map.get(code, '未知')
        sector_ret = sector_mom.get(sector, 0)

        # === 过滤条件 ===
        # 排除ST
        if 'ST' in name or '*ST' in name:
            continue

        # 行业动量过滤: 弱势行业跳过 (如果有行业数据的话)
        if hot_sectors and sector not in hot_sectors and sector != '未知':
            if sector_ret < median_mom - 2:
                continue

        # v7.3: 30min入场确认 (底分型+阳线+量≥5日均)
        entry_confirmed = False
        entry_price_30m = 0
        vol_ratio_30m = 0
        entry_detail = ''
        if len(df_30) >= 100:
            sig_date = item.get('sig_date', item.get('2buy_date', None))
            if sig_date is not None:
                result_30m = confirm_entry_30min(df_30, sig_date)
                if result_30m.confirmed:
                    entry_confirmed = True
                    entry_price_30m = result_30m.entry_price
                    vol_ratio_30m = result_30m.vol_ratio
                    entry_detail = result_30m.detail

        # 风险收益比
        entry_price = item.get('entry_price', item.get('2buy_price', price))
        stop_price = item.get('stop_price', item.get('1buy_low', entry_price * 0.95))
        signal_type = item.get('signal_type', '2buy')
        df = daily_map[code]
        last_close = df['close'].iloc[-1]
        risk = entry_price - stop_price
        recent_high = df['high'].iloc[-20:].max()
        reward = recent_high - entry_price
        rr_ratio = reward / risk if risk > 0 else 0

        # === 综合评分 ===
        tech_score = score_technical(df, entry_price, stop_price, rr_ratio)

        # 行业加分: 成长性 + 主线赛道层级
        sector_score = 0
        if sector in GROWTH_SECTORS:
            sector_score += SECTOR_BONUS['growth']
        tier = tier_map.get(code, 3)
        if tier == 1:
            sector_score += 20
        elif tier == 2:
            sector_score += 10

        is_hot_sector = tier == 1

        # === 买点强度加分 ===
        strength_bonus = 0

        # 量价背离加分
        vpd = detect_volume_price_divergence(
            df['close'].tolist(), df['volume'].tolist()
        )
        if vpd:
            if vpd.divergence_type == DivergenceType.BULLISH_ACCUMULATION:
                strength_bonus += 3  # 价跌量增 → 资金进场
            elif vpd.divergence_type == DivergenceType.BEARISH_DISTRIBUTION:
                strength_bonus -= 5  # 价涨量缩 → 资金出逃
        buy_strength = item.get('buy_strength', '')
        golden_pass = item.get('golden_ratio_pass', False)

        # 三档强度加分 (2买和3买分开处理)
        if signal_type == '2buy':
            # 2买: 网格搜索最优 — medium(中枢内回踩)最佳, strong(不进中枢)反而追高
            if buy_strength == 'strong':
                strength_bonus += 20  # 2买3买重叠: 缠论最强形态
            elif buy_strength == 'medium':
                strength_bonus += 12  # 类2买(中枢内): 67.7%胜率, +4.87%
            else:
                strength_bonus += 5   # 普通中枢下2买
        elif signal_type == 'sub1buy':
            # sub1B: 盘整背驰，胜率与1buy持平，略低于2/3买
            div_ratio = item.get('divergence_ratio', 1.0)
            strength_bonus += 5  # 基础分
            if div_ratio < 0.5:
                strength_bonus += 8  # 强背驰
            elif div_ratio < 0.8:
                strength_bonus += 3
        elif signal_type == 'quasi2buy':
            # quasi2B: 中枢内回踩中部+缩量, 胜率~55%
            strength_bonus += 8  # 基础分
            if item.get('vol_shrink'):
                strength_bonus += 5  # 缩量确认
        else:
            # 3买和其他: 保持原逻辑
            if buy_strength == 'strong':
                strength_bonus += 10
            elif buy_strength == 'standard':
                strength_bonus += 5
            elif buy_strength == 'weak':
                strength_bonus += 0

        # === 强三买加权评分 (5184组网格搜索最优) ===
        if signal_type == '3buy':
            three_checks = item.get('three_buy_checks', {})
            weighted = item.get('three_buy_weighted', 0)
            if weighted >= 2:
                strength_bonus += 15  # 顶部小中枢+非追高: 57.7%胜率
            elif weighted >= 1:
                strength_bonus += 6
            elif weighted < 0:
                strength_bonus -= 5   # 追高惩罚
            elif weighted == 0:
                # 无任何正面条件 → 大幅降分（减少低质3买噪音）
                strength_bonus -= 15

        # 黄金分割0.618加分 (仅3买)
        if golden_pass:
            strength_bonus += 8   # 3买回撤未破0.618

        # === v7.3: 入场确认加分 (替代周线加分) ===
        entry_bonus = 15 if entry_confirmed else 0

        # === 走势类型加分 ===
        trend_type_val = item.get('trend_type', '')
        trend_str_val = item.get('trend_strength', 0)
        trend_bonus = 0
        if trend_type_val == 'up':
            trend_bonus = 8
        elif trend_type_val == 'down':
            trend_bonus = -20
        elif trend_type_val == 'consolidation':
            trend_bonus = -8

        # === TD Sequential加分 ===
        td_boost = item.get('td_boost', 0)
        td_bonus = int(td_boost * 100) if td_boost else 0
        # TD卖出风险惩罚
        td_sell_risk = item.get('td_sell_risk', 0)
        if td_sell_risk >= 0.5:
            td_bonus -= int(td_sell_risk * 15)  # 强卖出信号: -15分

        total_score = tech_score + sector_score + int(strength_bonus * 0.7) + entry_bonus + trend_bonus + td_bonus

        # === 大盘环境权重 ===
        env_weight = market_env.get_signal_weight(signal_type)
        if env_weight <= 0:
            continue  # 当前环境禁用此买点类型
        total_score = int(total_score * env_weight)

        # === ML信号打分 ===
        ml_score = 0.0
        ml_label = ''
        try:
            if not _ML_AVAILABLE:
                raise ImportError('ml_signal_scorer not available')
            sig_dict = {
                'signal_type': signal_type,
                'confidence': item.get('confidence', 0.5),
                'pivot_info': '',
                'stop_price': stop_price,
            }
            # 取该股票的日线数据（daily_map → prefiltered_map → old_cache fallback）
            ml_daily = daily_map.get(code)
            if ml_daily is None:
                ml_daily = prefiltered_map.get(code)
            if ml_daily is None:
                # Fallback: 从scanner_new_fw_cache_120.pkl加载
                _oc_key = ('SH' if code[0] in ('6', '9') else 'SZ') + code + ('.SH' if code[0] in ('6', '9') else '.SZ')
                if not hasattr(scan_enhanced, '_old_cache'):
                    try:
                        import pickle as _pkl
                        with open('scanner_new_fw_cache_120.pkl', 'rb') as _f:
                            scan_enhanced._old_cache = _pkl.load(_f)
                    except Exception:
                        scan_enhanced._old_cache = {}
                ml_daily = scan_enhanced._old_cache.get(_oc_key)
            if ml_daily is not None and len(ml_daily) >= 30:
                ml_pred = _ml_predict(sig_dict, ml_daily, '', market_env)
                p_strong = ml_pred.get('p_bigwin', 0.33)
                p_weak = ml_pred.get('p_bigloss', 0.33)
                ml_score = round(p_strong - p_weak, 3)
                if ml_score >= 0.3:
                    ml_label = 'strong'
                elif ml_score < -0.1:
                    ml_label = 'weak'
                else:
                    ml_label = 'neutral'
        except Exception as _ml_err:
            if not hasattr(scan_enhanced, '_ml_err_logged'):
                scan_enhanced._ml_err_logged = True
                print(f'   [ML] 首次异常: {_ml_err.__class__.__name__}: {_ml_err}')
            pass

        # === 小转大检测 (v7 from v73) ===
        small_to_large = None
        small_to_large_note = ''
        try:
            if len(df_30) >= 100:
                small_to_large = detect_small_to_large(df_30, df_daily)
                if small_to_large and small_to_large.get('detected'):
                    # 买点减分/卖点加分
                    total_score = int(apply_score_adjustment(
                        total_score / 100.0, signal_type, small_to_large
                    ) * 100)
                    small_to_large_note = (
                        f"小转大! {small_to_large['direction']} | "
                        f"30m{small_to_large['sell_30m_type']} conf={small_to_large['sell_30m_conf']} | "
                        f"笔衰减{small_to_large['decay_pct']}% | "
                        f"中枢{small_to_large['pivot_count_30m']}个"
                    )
                    print(f'   ⚡ 小转大预警: {code} {signal_type} → score {total_score}')
        except Exception as _stl_err:
            if not hasattr(scan_enhanced, '_stl_err_logged'):
                scan_enhanced._stl_err_logged = True
                print(f'   [STL] 首次异常: {_stl_err.__class__.__name__}: {_stl_err}')

        results.append({
            'code': code,
            'name': name,
            'sector': sector,
            'sector_ret': round(sector_ret, 2),
            'sector_tier': tier,
            'hot_sector': is_hot_sector,
            'price': price,
            'pct_chg': pct,
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'risk_reward': round(rr_ratio, 1),
            '2buy_date': str(item.get('sig_date', item.get('2buy_date', ''))),
            'pivot_info': '',
            'last_close': last_close,
            'tech_score': tech_score,
            'sector_score': sector_score,
            'strength_bonus': strength_bonus,
            'total_score': total_score,
            'buy_strength': buy_strength,
            'golden_ratio_pass': golden_pass,
            'entry_confirmed': entry_confirmed,
            'entry_detail': entry_detail,
            'vol_ratio_30m': vol_ratio_30m,
            'weekly_trend': item.get('eff_regime', ''),
            'weekly_score': 0,
            'weekly_rise_pct': 0,
            'weekly_bottom_fractal': False,
            'weekly_risk_reward': None,
            'weekly_bf_low': None,
            'weekly_target_gg': None,
            'three_buy_checks': item.get('three_buy_checks', {}),
            'three_buy_passed': item.get('three_buy_passed', 0),
            'three_buy_weighted': item.get('three_buy_weighted', 0),
            'trend_type': trend_type_val,
            'trend_strength': trend_str_val,
            'pos_coef': item.get('pos_coef', 1.0),
            'eff_regime': item.get('eff_regime', ''),
            'ml_score': ml_score,
            'ml_label': ml_label,
            'td_boost': item.get('td_boost', 0),
            'td_detail': item.get('td_detail', ''),
            'td_sell_risk': item.get('td_sell_risk', 0),
            'atr_stop_price': item.get('atr_stop_price', 0),
            'atr_multiplier': item.get('atr_multiplier', 0),
            'confidence': item.get('confidence', 0.5),
            'small_to_large_note': small_to_large_note,
        })

    # 7. 排序 + 最低分过滤
    t_30min = time.time() - t_30min_start
    elapsed = time.time() - t0
    MIN_SCORE = 80  # 低于80分胜率56%、平均+1.8% (404信号回测验证)
    before_filter = len(results)

    # ML信号调整: 用连续ml_score映射到乘数 (比3分类更精细)
    # IC/IR验证: ml_score在多天全0(降级), 跳过0值避免误调整
    for r in results:
        ms = r.get('ml_score', 0)
        if ms == 0:
            continue  # ML降级时跳过，不调整
        if ms >= 0.5:
            r['total_score'] = int(r['total_score'] * 1.20)   # 高置信: +20%
        elif ms >= 0.3:
            r['total_score'] = int(r['total_score'] * 1.10)   # 中高: +10%
        elif ms >= 0.1:
            r['total_score'] = int(r['total_score'] * 1.02)   # 微正: +2%
        elif ms < -0.1:
            r['total_score'] = int(r['total_score'] * 0.75)   # 负面: -25%
        elif ms < 0.0:
            r['total_score'] = int(r['total_score'] * 0.90)   # 微负: -10%

    # 风控惩罚: 高风险信号降分 (回测验证)
    risk_filtered = 0
    for r in results:
        penalty = 0
        # 周线非上升趋势: 33%大亏率
        if r.get('trend_type') in ('consolidation', 'down'):
            penalty += 20
        # 3买质量差(weighted<=-1): 22.9%大亏率
        if r.get('signal_type') == '3buy' and int(r.get('three_buy_weighted', 0)) <= -1:
            penalty += 15
        # 盈亏比极低(R/R<0.5): 20.7%大亏率
        if float(r.get('risk_reward', 0)) < 0.5:
            penalty += 10
        if penalty > 0:
            r['total_score'] -= penalty
            r['risk_penalty'] = penalty
            if r['total_score'] < MIN_SCORE:
                risk_filtered += 1

    results = [r for r in results if r['total_score'] >= MIN_SCORE]

    # P0-2: 置信度过滤 (CONF≥0.6, 回测验证阈值)
    _conf_before = len(results)
    results = [r for r in results if r.get('confidence', 0.6) >= 0.6]
    _conf_filtered = _conf_before - len(results)
    if _conf_filtered > 0:
        print(f'   置信度过滤: {_conf_filtered}只 CONF<0.6 已移除')

    results.sort(key=lambda x: x['total_score'], reverse=True)

    print(f'\n{"="*90}')
    print(f'扫描完成 ({elapsed:.0f}s) — {len(results)} 只候选股 (过滤{before_filter - len(results)}只: {before_filter - len(results) - risk_filtered}低分 + {risk_filtered}风控), 显示Top {top_n}')
    print(f'  Step5缠论: {t_scan:.0f}s | Step6确认+评分: {t_30min:.0f}s')
    ml_strong = sum(1 for r in results if r.get('ml_label') == 'strong')
    ml_weak = sum(1 for r in results if r.get('ml_label') == 'weak')
    ml_neutral = sum(1 for r in results if r.get('ml_label') == 'neutral')
    print(f'  ML信号: 强={ml_strong} 中性={ml_neutral} 弱={ml_weak}')
    print(f'{"="*90}')

    print(f'\n{"排名":<4} {"层级":<6} {"代码":<8} {"名称":<8} {"类型":<5} {"强度":<6} {"行业":<8} {"行业涨幅":>7} '
          f'{"现价":>8} {"入场价":>8} {"R/R":>5} {"评分":>4} {"ML":>4} {"周线涨幅":>7} {"周线":<10} {"信号日":<12}')
    print('-' * 120)

    strength_cn = {'strong': '强', 'standard': '标准', 'weak': '弱', '': ''}
    tier_labels = {1: '[主线]', 2: '[活跃]', 3: '[发现]'}
    trend_labels_map = {'up': '[up]', 'down': '[dn]', 'consolidation': '[==]', '': ''}
    for i, r in enumerate(results[:top_n]):
        st = strength_cn.get(r.get('buy_strength', ''), '')
        if r.get('golden_ratio_pass'):
            st += '★'  # 黄金分割加分标记
        tl = tier_labels.get(r.get('sector_tier', 2), '')
        t_lbl = trend_labels_map.get(r.get('trend_type', ''), '')
        print(f'{i+1:<4} {tl:<6} {r["code"]:<8} {r["name"]:<8} {r.get("signal_type","2buy"):<5} '
              f'{st:<6} '
              f'{r["sector"]:<8} '
              f'{r["sector_ret"]:>+6.1f}% {r["price"]:>8.2f} {r["entry_price"]:>8.2f} '
              f'{r["risk_reward"]:>5.1f} {r["total_score"]:>4} {r.get("ml_label","")[:4]:>4} {r.get("weekly_rise_pct",0):>6.1f}% {r["weekly_trend"]:<10} {t_lbl} {r["2buy_date"]:<12}'
              f' 仓位:{r.get("pos_coef",1.0):.0%} {r.get("eff_regime","")}')

    # 周线底分型候选（盈亏比排序）
    wbf_results = [r for r in results if r.get('weekly_bottom_fractal') and r.get('weekly_risk_reward')]
    if wbf_results:
        wbf_results.sort(key=lambda x: -(x.get('weekly_risk_reward') or 0))
        print(f'\n  周线底分型候选（盈亏比排序）: {len(wbf_results)}只')
        print(f'  {"代码":<10} {"名称":<8} {"类型":<5} {"现价":>8} {"止损":>8} {"目标GG":>8} {"盈亏比":>6}')
        print(f'  {"-"*55}')
        for r in wbf_results[:15]:
            bf_low = r.get('weekly_bf_low', 0) or 0
            target_gg = r.get('weekly_target_gg', 0) or 0
            rr = r.get('weekly_risk_reward', 0)
            print(f'  {r["code"]:<10} {r["name"]:<8} {r.get("signal_type","2buy"):<5} '
                  f'{r["price"]:>8.2f} {bf_low:>8.2f} {target_gg:>8.2f} {rr:>6.1f}')

    # 保存
    output_file = f'signals/scan_enhanced_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    os.makedirs('signals', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'pool': pool,
            'lookback_days': lookback_days,
            'total_candidates': len(results),
            'top_n': results[:top_n],
            'all_signals': results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果已保存: {output_file}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', default='tdx_all')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--min-price', type=float, default=3.0)
    parser.add_argument('--max-price', type=float, default=200.0)
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--committee', action='store_true', help='运行投资委员会评估')
    args = parser.parse_args()

    results = scan_enhanced(pool=args.pool, lookback_days=args.days,
                  min_price=args.min_price, max_price=args.max_price,
                  top_n=args.top)

    # 投资委员会评估
    if args.committee and results:
        print(f'\n[投资委员会] 评估Top {len(results)}候选股...')
        from agents.investment_committee import InvestmentCommittee

        # 加载持仓
        positions = {'positions': [], 'capital': 1000000}
        pos_path = 'signals/positions.json'
        if os.path.exists(pos_path):
            with open(pos_path, 'r', encoding='utf-8') as f:
                positions = json.load(f)

        # 计算行业动量（从扫描结果中提取）
        sector_mom = {}
        for r in results[:50]:
            sector_mom[r.get('sector', '')] = r.get('sector_ret', 0)

        committee = InvestmentCommittee(
            hs=HybridSource(), sector_map=load_sector_map(),
            portfolio_state=positions, sector_momentum=sector_mom,
        )
        committee_results = committee.evaluate_batch(results)
        committee.print_report(committee_results)
        committee.save_results(committee_results)
