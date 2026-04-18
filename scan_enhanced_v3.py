#!/usr/bin/env python3
"""增强版实盘扫描 V3 — 行业动量 + 技术评分 + 基本面过滤

从324只候选股筛选出Top 10最佳入场机会:
  1. 行业动量过滤: 只选近5日涨幅前50%的行业
  2. 基本面过滤: 排除ST、低价、高价股
  3. 技术评分: MACD方向 + 量价配合 + R/R比
  4. 综合排名输出
"""
import sys, os
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

import json, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.hybrid_source import HybridSource

from backtest_cc15_mtf import (
    detect_fractals_30min, detect_strokes_30min,
    detect_pivot_30min,
    run_daily_cc15, find_daily_1buy_2buy,
    fetch_sina_30min,
)


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
    'hot': 10,       # 行业动量强势加10分
    'normal': 0,     # 普通行业
}


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
        if not sector:
            continue
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


def _detect_weekly_trend(code, hs):
    """检测周线趋势方向: bull/bear/range

    周线定方向: 周线多头才做多，周线空头不做

    Returns: (trend, score, weekly_rise_pct)
        weekly_rise_pct: 从周线最低点至今的涨幅百分比 (如 25.3 = 25.3%)
    """
    try:
        df_w = hs.get_kline(code, period='weekly')
        if len(df_w) < 30:
            return 'range', 0.0, 0.0

        close = df_w['close']
        low = df_w['low']
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1] if len(df_w) >= 20 else ma10
        last = close.iloc[-1]

        # === 周线涨幅: 从近期最低点(20周)到现在的涨幅 ===
        recent_low = low.iloc[-20:].min() if len(df_w) >= 20 else low.min()
        weekly_rise_pct = (last / recent_low - 1) * 100 if recent_low > 0 else 0.0

        # 周线MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_val = (dif - dea).iloc[-1] * 2

        score = 0.0
        if last > ma5 > ma10:
            score += 0.3
        if last > ma20:
            score += 0.2
        if macd_val > 0:
            score += 0.2
        # 周线趋势: 近5周是否上涨
        if close.iloc[-1] > close.iloc[-5]:
            score += 0.2
        # 周线MA斜率
        if len(df_w) >= 10:
            ma5_slope = (ma5 - close.rolling(5).mean().iloc[-5]) / close.rolling(5).mean().iloc[-5]
            if ma5_slope > 0.02:
                score += 0.1
            elif ma5_slope < -0.02:
                score -= 0.1

        if score >= 0.5:
            return 'bull', score, weekly_rise_pct
        elif score <= 0.1:
            return 'bear', score, weekly_rise_pct
        else:
            return 'range', score, weekly_rise_pct
    except Exception:
        return 'range', 0.3, 0.0


def _detect_market_regime(index_code='000001'):
    """检测当前市场环境: strong/normal/weak

    强势行情不做1买抄底
    """
    try:
        hs_tmp = HybridSource()
        df = hs_tmp.get_kline(index_code, period='daily')
        if len(df) < 60:
            return 'normal'

        close = df['close']
        # MA趋势
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]

        # 近20日涨幅
        ret_20 = (close.iloc[-1] / close.iloc[-20] - 1)

        # 近5日涨幅
        ret_5 = (close.iloc[-1] / close.iloc[-5] - 1)

        bull_count = 0
        if ma5 > ma20 > ma60:
            bull_count += 2
        if ma5 > ma20:
            bull_count += 1
        if ret_20 > 0.05:
            bull_count += 2
        elif ret_20 > 0.02:
            bull_count += 1
        if ret_5 > 0.03:
            bull_count += 1

        if bull_count >= 5:
            return 'strong'
        elif bull_count >= 2:
            return 'normal'
        else:
            return 'weak'
    except Exception:
        return 'normal'


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


    # === 规则5: 强势行情不做1买抄底 ===
    if market_regime == 'strong':
        return []  # 强势行情中1买不可靠, 直接跳过

    n = len(df)
    if n < 120:
        return []

    close = df['close']
    low = df['low']

    bi_buy, bi_sell, filtered_fractals, strokes = engine._detect_bi_deterministic(df)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    buy_div_set, _, _, _ = engine._compute_area_divergence(strokes, hist, n)

    results = []
    for idx in buy_div_set:
        if 0 <= idx < n:
            entry_price = close.iloc[idx]
            stop_price = low.iloc[idx] * 0.98  # 1买止损=最低点下2%
            results.append({
                'signal_type': '1buy',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'sig_idx': idx,
                'buy_strength': 'normal',  # 1买不做三档分类
            })

    return results


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

    # 从笔中识别中枢: 3笔重叠区间
    pivots = []  # list of {zg, zd, start_idx, end_idx}
    for i in range(len(strokes) - 2):
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
            end_idx = max(s1['end_idx'], s2['end_idx'], s3['end_idx'])
            pivots.append({
                'zg': zg, 'zd': zd,
                'start_idx': min(s1['start_idx'], s2['start_idx'], s3['start_idx']),
                'end_idx': end_idx,
            })

    if not pivots:
        return []

    results = []
    # 对每个中枢, 找突破后回踩不进中枢的位置
    for pv in pivots:
        zg = pv['zg']
        zd = pv['zd']
        pv_end = pv['end_idx']

        # 计算GG (中枢波动高点): 取组成中枢的所有笔的最高点
        gg = zg  # 默认GG=ZG
        dd = zd  # 默认DD=ZD

        # 从中枢结束后开始找突破
        for j in range(pv_end + 1, n - 5):
            if high.iloc[j] > zg:  # 突破中枢上沿
                # 记录突破高点 (用于0.618计算)
                breakout_high = high.iloc[j]
                # 找突破后的回调低点
                for k in range(j + 1, min(j + 20, n)):
                    pullback_low = low.iloc[k]
                    if pullback_low > zd:  # 回踩不进中枢 = 3买
                        entry_price = close.iloc[k]
                        stop_price = zd  # 止损=中枢下沿

                        # === 3买三档强度分类 ===
                        if pullback_low > gg:
                            strength = 'strong'     # 强三买: 回踩>GG
                        elif pullback_low > zg:
                            strength = 'standard'   # 标准三买: ZG~GG
                        else:
                            strength = 'weak'       # 弱三买: ZD~ZG

                        # === 黄金分割0.618检查 ===
                        # 突破段: 中枢结束 → 突破高点
                        # 回撤: 突破高点 → 回踩低点
                        # 回撤比例 = (breakout_high - pullback_low) / (breakout_high - zg)
                        golden_pass = False
                        if breakout_high > zg:
                            retrace = (breakout_high - pullback_low) / (breakout_high - zg)
                            if retrace <= 0.618:  # 未跌破0.618 = 多头强
                                golden_pass = True

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
                        })
                        break  # 每个中枢只取第一个3买
                break  # 每个中枢只看一次突破

    return results


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

    # 3. 加载日线数据
    print('[3] 加载日线数据...')
    daily_map = {}
    batch_size = 100
    for i in range(0, len(pure_codes), batch_size):
        batch = pure_codes[i:i+batch_size]
        for code in batch:
            try:
                df = hs.get_kline(code, period='daily')
                if len(df) >= 200:
                    last_close = df['close'].iloc[-1]
                    if min_price <= last_close <= max_price:
                        daily_map[code] = df
            except Exception:
                pass
        print(f'   [{min(i+batch_size, len(pure_codes))}/{len(pure_codes)}] '
              f'日线={len(daily_map)}', end='\r')
    print(f'   日线数据: {len(daily_map)} 只')

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

    # 5. CC15引擎 + 找所有买点(1买/2买/3买)
    print('[5] 运行CC15引擎 + 识别买点...')
    engine, daily_signals = run_daily_cc15(daily_map)

    # === 规则5: 检测市场环境 ===
    market_regime = _detect_market_regime()
    print(f'   市场环境: {market_regime} (强势行情不做1买)')

    cutoff = datetime.now() - timedelta(days=lookback_days)
    all_signals = []  # 统一收集所有买点信号

    for code in daily_map:
        df = daily_map[code]

        # --- 2买信号 ---
        pairs = find_daily_1buy_2buy(engine, code, df)
        for p in pairs:
            if p['2buy_idx'] >= len(df):
                continue
            sig_date = df.index[p['2buy_idx']]
            if sig_date >= pd.Timestamp(cutoff):
                p['code'] = code
                p['signal_type'] = '2buy'
                p['entry_price'] = p['2buy_price']
                p['stop_price'] = p['1buy_low']
                p['sig_idx'] = p['2buy_idx']
                p['sig_date'] = sig_date

                # === 2买三档强度分类 ===
                # 1. 找中枢位置(需要从CC15引擎获取)
                p['buy_strength'] = _classify_2buy_strength(df, p, engine)
                p['golden_ratio_pass'] = False  # 2买不用黄金分割

                all_signals.append(p)

        # --- 1买信号 (底背驰) ---
        ones = _find_1buy_standalone(engine, code, df, market_regime=market_regime)
        for s in ones:
            if s['sig_idx'] < len(df):
                sig_date = df.index[s['sig_idx']]
                if sig_date >= pd.Timestamp(cutoff):
                    s['code'] = code
                    s['sig_date'] = sig_date
                    all_signals.append(s)

        # --- 3买信号 (突破中枢回踩不进) ---
        threes = _find_3buy_standalone(engine, code, df)
        for s in threes:
            if s['sig_idx'] < len(df):
                sig_date = df.index[s['sig_idx']]
                if sig_date >= pd.Timestamp(cutoff):
                    s['code'] = code
                    s['sig_date'] = sig_date
                    all_signals.append(s)

    # 按类型统计
    type_counts = {}
    for s in all_signals:
        t = s.get('signal_type', '?')
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f'   最近{lookback_days}天信号: {dict(type_counts)} 共{len(all_signals)}个')

    if not all_signals:
        print('无近期买点信号')
        return []

    # 6. 30min确认 + 评分
    print('[6] 30分钟确认 + 综合评分...')
    results = []
    scanned = set()

    for item in all_signals:
        code = item['code']
        if code in scanned:
            continue
        scanned.add(code)

        # === 规则4: 周线定方向过滤 ===
        weekly_trend, weekly_score, weekly_rise_pct = _detect_weekly_trend(code, hs)
        item['weekly_rise_pct'] = round(weekly_rise_pct, 1)

        # === 30分钟策略: 周线涨幅硬过滤 ===
        # 周线从低点涨幅不足20% → 跳过 (回测验证: 20%阈值最优)
        if weekly_rise_pct < 20.0:
            continue

        if weekly_trend == 'bear':
            # 周线空头不做多 → 降级但不过滤(保留记录)
            item['weekly_veto'] = True
        else:
            item['weekly_veto'] = False

        df_30 = fetch_sina_30min(code)
        if len(df_30) < 100:
            continue

        fractals_30 = detect_fractals_30min(df_30)
        if not fractals_30:
            continue

        strokes_30 = detect_strokes_30min(df_30)
        pivots_30 = detect_pivot_30min(strokes_30)

        # 获取最新价格
        try:
            pure = code
            if not pure.endswith('.SZ') and not pure.endswith('.SH'):
                if pure.startswith(('6', '9')):
                    pure = pure + '.SH'
                else:
                    pure = pure + '.SZ'
            q = hs.get_realtime_quote([pure])
            if len(q) > 0:
                name = q.iloc[0].get('name', code)
                price = float(q.iloc[0].get('price', 0))
                pct = float(q.iloc[0].get('pct_chg', 0))
            else:
                name, price, pct = code, 0, 0
        except Exception:
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
            # 允许未知行业通过, 但已知弱势行业过滤
            if sector_ret < median_mom - 2:  # 给2%的容差
                continue

        # 30min中枢状态
        pivot_info = ''
        latest_pivot = pivots_30[-1] if pivots_30 else None
        if latest_pivot:
            pivot_info = f'ZG={latest_pivot["ZG"]:.2f} ZD={latest_pivot["ZD"]:.2f}'

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

        # 行业加分: 成长性行业 + 动量强势
        sector_score = 0
        if sector in GROWTH_SECTORS:
            sector_score += SECTOR_BONUS['growth']
        if sector in hot_sectors:
            sector_score += SECTOR_BONUS['hot']

        # === 买点强度加分 ===
        strength_bonus = 0
        buy_strength = item.get('buy_strength', '')
        golden_pass = item.get('golden_ratio_pass', False)

        # 三档强度加分
        if buy_strength == 'strong':
            strength_bonus += 10  # 强2买(2买3买重叠) / 强3买(回踩>GG)
        elif buy_strength == 'standard':
            strength_bonus += 5   # 标准三买(ZG~GG) / 类2买
        elif buy_strength == 'weak':
            strength_bonus += 0   # 弱三买(ZD~ZG) / 中枢下2买

        # 黄金分割0.618加分 (仅3买)
        if golden_pass:
            strength_bonus += 8   # 3买回撤未破0.618

        # === 周线方向过滤 ===
        weekly_penalty = 0
        weekly_trend_str = '未知'
        if item.get('weekly_veto'):
            weekly_penalty = -15  # 周线空头大幅扣分
            weekly_trend_str = '空头(扣分)'
        elif weekly_trend == 'bull':
            weekly_trend_str = '多头'
            strength_bonus += 5  # 周线多头加分
        else:
            weekly_trend_str = '盘整'

        total_score = tech_score + sector_score + strength_bonus + weekly_penalty

        results.append({
            'code': code,
            'name': name,
            'sector': sector,
            'sector_ret': round(sector_ret, 2),
            'price': price,
            'pct_chg': pct,
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'risk_reward': round(rr_ratio, 1),
            '2buy_date': str(item.get('sig_date', item.get('2buy_date', ''))),
            'pivot_info': pivot_info,
            'last_close': last_close,
            'tech_score': tech_score,
            'sector_score': sector_score,
            'strength_bonus': strength_bonus,
            'total_score': total_score,
            'buy_strength': buy_strength,
            'golden_ratio_pass': golden_pass,
            'weekly_trend': weekly_trend_str,
            'weekly_score': round(weekly_score, 2),
            'weekly_rise_pct': item.get('weekly_rise_pct', 0),
        })

        time.sleep(0.1)

    # 7. 排序输出
    elapsed = time.time() - t0
    results.sort(key=lambda x: x['total_score'], reverse=True)

    print(f'\n{"="*90}')
    print(f'扫描完成 ({elapsed:.0f}s) — {len(results)} 只候选股, 显示Top {top_n}')
    print(f'{"="*90}')

    print(f'\n{"排名":<4} {"代码":<8} {"名称":<8} {"类型":<5} {"强度":<6} {"行业":<8} {"行业涨幅":>7} '
          f'{"现价":>8} {"入场价":>8} {"R/R":>5} {"评分":>4} {"周线涨幅":>7} {"周线":<10} {"信号日":<12}')
    print('-' * 120)

    strength_cn = {'strong': '强', 'standard': '标准', 'weak': '弱', '': ''}
    for i, r in enumerate(results[:top_n]):
        st = strength_cn.get(r.get('buy_strength', ''), '')
        if r.get('golden_ratio_pass'):
            st += '★'  # 黄金分割加分标记
        print(f'{i+1:<4} {r["code"]:<8} {r["name"]:<8} {r.get("signal_type","2buy"):<5} '
              f'{st:<6} '
              f'{r["sector"]:<8} '
              f'{r["sector_ret"]:>+6.1f}% {r["price"]:>8.2f} {r["entry_price"]:>8.2f} '
              f'{r["risk_reward"]:>5.1f} {r["total_score"]:>4} {r.get("weekly_rise_pct",0):>6.1f}% {r["weekly_trend"]:<10} {r["2buy_date"]:<12}')

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
