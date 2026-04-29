#!/usr/bin/env python3
"""买点预警池生成器

每天收盘后运行，从全市场筛选"接近买点"的股票，生成预警池供盘中监控使用。

三种预警类型：
1. 潜在3买：价格在中枢ZD上方1-3%范围内，接近中枢支撑
2. 潜在2买：近期已出1买（底背驰），等待回踩不破前低
3. 潜在1买：MACD面积递减，价格接近前低，背驰形成中

输出: signals/buy_alert_pool.json
"""
import sys, os
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import json, time
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from data.hybrid_source import HybridSource
from backtest_cc15_mtf import run_daily_cc15, _build_pivots_from_strokes


def _check_approaching_3buy(strokes, pivots, close, low, high, n, threshold_pct=3.0):
    """检查是否接近3买：价格在中枢ZD上方threshold_pct%以内"""
    if not pivots:
        return None
    last_pivot = pivots[-1]
    zg = last_pivot['zg']
    zd = last_pivot['zd']
    pv_end = last_pivot['end_idx']

    # 价格必须在中枢上方（已突破或正在突破）
    last_close = close.iloc[-1]
    if last_close < zd:
        return None

    # 检查是否已经突破ZG（如果已突破且回踩，就是真正的3买，不在预警范围）
    recent_high = high.iloc[max(0, n-20):].max()
    if recent_high > zg and last_close > zg:
        return None  # 已经是完成态3买，不需要预警

    # 接近ZD支撑 = 潜在3买触发区域
    dist_pct = (last_close - zd) / zd * 100
    if dist_pct <= threshold_pct:
        return {
            'alert_type': 'approaching_3buy',
            'distance_pct': round(dist_pct, 2),
            'pivot_zg': round(zg, 2),
            'pivot_zd': round(zd, 2),
            'target_zone': f'{round(zd, 2)}-{round(zg, 2)}',
            'reason': f'价格{last_close:.2f}距中枢ZD({zd:.2f})仅{dist_pct:.1f}%',
        }
    return None


def _check_approaching_2buy(strokes, buy_div_set, close, low, n):
    """检查是否接近2买：已出1买，价格回踩中，接近1买低点但不破"""
    if not buy_div_set:
        return None

    last_1buy = max(buy_div_set)
    if last_1buy >= n:
        return None

    buy_low = low.iloc[last_1buy]
    buy_price = close.iloc[last_1buy]
    last_close = close.iloc[-1]

    # 1买后至少5根K线，最多30根（太远不算）
    bars_since = n - 1 - last_1buy
    if bars_since < 5 or bars_since > 30:
        return None

    # 价格在1买低点上方0-5%范围内
    dist_pct = (last_close - buy_low) / buy_low * 100
    if 0 < dist_pct <= 5.0:
        return {
            'alert_type': 'approaching_2buy',
            'distance_pct': round(dist_pct, 2),
            'buy_low': round(buy_low, 2),
            'buy_price': round(buy_price, 2),
            'bars_since_1buy': bars_since,
            'reason': f'1买后{bars_since}根K线,价格距1买低点({buy_low:.2f})仅{dist_pct:.1f}%',
        }
    return None


def _check_approaching_1buy(strokes, hist, close, low, n):
    """检查是否接近1买：MACD面积递减，价格接近前低"""
    down_strokes = [s for s in strokes
                    if s['start_type'] == 'top' and s['end_type'] == 'bottom']
    if len(down_strokes) < 2:
        return None

    curr = down_strokes[-1]
    prev = down_strokes[-2]

    curr_area = abs(sum(hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
    prev_area = abs(sum(hist.iloc[prev['start_idx']:prev['end_idx']+1].values))

    # 面积递减（背驰形成中）
    if prev_area == 0 or curr_area / prev_area >= 0.8:
        return None

    # 价格接近或低于前低
    curr_low = curr['end_val']
    prev_low = prev['end_val']
    if curr_low > prev_low * 1.02:
        return None  # 价格还没到前低附近

    last_close = close.iloc[-1]
    dist_pct = (last_close - curr_low) / curr_low * 100

    return {
        'alert_type': 'approaching_1buy',
        'area_ratio': round(curr_area / prev_area, 2),
        'curr_low': round(curr_low, 2),
        'prev_low': round(prev_low, 2),
        'distance_pct': round(dist_pct, 2),
        'reason': f'MACD面积比={curr_area/prev_area:.2f}(递减), 价格{last_close:.2f}接近前低{curr_low:.2f}',
    }


def generate_alert_pool():
    """生成买点预警池"""
    t0 = time.time()
    hs = HybridSource()

    # 加载日线扫描候选（如果有）
    scan_candidates = set()
    scan_files = sorted([f for f in os.listdir('signals') if f.startswith('scan_enhanced_')])
    if scan_files:
        with open(f'signals/{scan_files[-1]}', 'r', encoding='utf-8') as f:
            scan_data = json.load(f)
        scan_candidates = {r['code'] for r in scan_data.get('top_n', [])}

    # 获取全市场股票列表
    tdx_path = hs.tdx_path
    all_codes = []
    for market in ['sh', 'sz']:
        lday = os.path.join(tdx_path, market, 'lday')
        if not os.path.exists(lday):
            continue
        for f in os.listdir(lday):
            if f.endswith('.day'):
                code = f.replace('.day', '')
                if code.startswith('sh6') or code.startswith('sz0') or code.startswith('sz3'):
                    all_codes.append(code)

    print(f'=== 买点预警池生成器 ===')
    print(f'全市场: {len(all_codes)} 只, 日线候选: {len(scan_candidates)} 只')
    print()

    # 阶段1: 扫描候选优先（这些已经通过日线筛选）
    # 阶段2: 全市场快速筛选（只看最后20根K线）
    priority_codes = list(scan_candidates)
    remaining_codes = [c for c in all_codes if c not in scan_candidates]

    alerts = []
    stats = {'3buy': 0, '2buy': 0, '1buy': 0, 'error': 0}

    def scan_code(code):
        """扫描单只股票"""
        df = hs.get_kline(code, period='daily')
        if df is None or len(df) < 120:
            return None

        n = len(df)
        close = df['close']
        high = df['high']
        low = df['low']

        # 使用CC15引擎的Swing算法检测笔
        sys.path.insert(0, 'chanlun_unified')
        from signal_engine_cc15 import SignalEngine
        engine = SignalEngine()
        _, _, _, strokes = engine._detect_bi_deterministic(df)
        if len(strokes) < 6:
            return None

        # 构建中枢
        pivots = _build_pivots_from_strokes(strokes)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = 2 * (dif - dea)

        # 底背驰
        buy_div_set = set()
        down_strokes = [s for s in strokes if s['start_type'] == 'top' and s['end_type'] == 'bottom']
        for k in range(1, len(down_strokes)):
            curr = down_strokes[k]
            curr_area = abs(sum(hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
            for lb in range(1, min(4, k+1)):
                prev = down_strokes[k - lb]
                prev_area = abs(sum(hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
                if curr['end_val'] < prev['end_val'] and curr_area < prev_area:
                    buy_div_set.add(curr['end_idx'])
                    break

        # 三种预警检查
        results = []
        r = _check_approaching_3buy(strokes, pivots, close, low, high, n)
        if r:
            r['code'] = code
            results.append(r)
        r = _check_approaching_2buy(strokes, buy_div_set, close, low, n)
        if r:
            r['code'] = code
            results.append(r)
        r = _check_approaching_1buy(strokes, hist, close, low, n)
        if r:
            r['code'] = code
            results.append(r)

        return results

    # 扫描候选股（完整分析）
    print(f'[1/2] 扫描 {len(priority_codes)} 只日线候选...')
    for i, code in enumerate(priority_codes):
        try:
            rs = scan_code(code)
            if rs:
                for r in rs:
                    alerts.append(r)
                    stats[r['alert_type'].split('_')[-1]] += 1
        except Exception:
            stats['error'] += 1

    # 全市场快速筛选（只看近期有量的活跃股）
    print(f'[2/2] 扫描 {len(remaining_codes)} 只剩余股票...')
    scanned = 0
    for code in remaining_codes:
        try:
            df = hs.get_kline(code, period='daily')
            if df is None or len(df) < 120:
                continue
            close = df['close']
            volume = df['volume']

            # 快速过滤：最近5日有放量或涨跌活跃
            recent_vol = volume.iloc[-5:].mean()
            avg_vol = volume.iloc[-20:].mean()
            if recent_vol < avg_vol * 0.8:
                continue  # 成交量萎缩，不活跃

            rs = scan_code(code)
            if rs:
                for r in rs:
                    alerts.append(r)
                    stats[r['alert_type'].split('_')[-1]] += 1
            scanned += 1
            if scanned % 200 == 0:
                print(f'   已扫描 {scanned} 只, 预警 {len(alerts)} 个')
        except Exception:
            stats['error'] += 1

    # 获取股票名称
    for alert in alerts:
        try:
            pure = alert['code']
            if not pure.endswith(('.SZ', '.SH')):
                pure = pure + ('.SH' if pure.startswith('6') else '.SZ')
            q = hs.get_realtime_quote([pure])
            alert['name'] = q.iloc[0].get('name', alert['code']) if len(q) > 0 else alert['code']
            alert['last_price'] = float(q.iloc[0].get('price', 0)) if len(q) > 0 else 0
        except:
            alert['name'] = alert['code']
            alert['last_price'] = 0

    # 按预警紧迫度排序，选出盘中监控top候选
    def _urgency(a):
        t = a['alert_type']
        if t == 'approaching_3buy':
            return a['distance_pct']  # 越小越紧迫
        elif t == 'approaching_2buy':
            return a['distance_pct'] + 3  # 次优先
        else:  # approaching_1buy
            return (1 - a.get('area_ratio', 0.5)) * 10 + 6  # MACD面积比越小越紧迫

    sorted_alerts = sorted(alerts, key=_urgency)
    top_candidates = sorted_alerts[:100]

    # 按预警类型分组
    pool = {
        'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_alerts': len(alerts),
        'stats': stats,
        'scan_candidates': list(scan_candidates),
        'top_candidates': top_candidates,  # 盘中扫描用（最紧迫的100只）
        'alerts': sorted_alerts,
    }

    # 保存
    os.makedirs('signals', exist_ok=True)
    output_path = 'signals/buy_alert_pool.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f'\n=== 预警池生成完成 ({elapsed:.0f}s) ===')
    print(f'总预警: {len(alerts)} 只')
    print(f'  潜在3买: {stats["3buy"]}')
    print(f'  潜在2买: {stats["2buy"]}')
    print(f'  潜在1买: {stats["1buy"]}')
    print(f'  错误: {stats["error"]}')
    print(f'\n已保存: {output_path}')

    # 打印Top预警
    by_type = {}
    for a in alerts:
        t = a['alert_type']
        by_type.setdefault(t, []).append(a)

    print(f'\n--- 潜在3买 (最接近中枢支撑) ---')
    for a in sorted(by_type.get('approaching_3buy', []), key=lambda x: x.get('distance_pct', 99))[:10]:
        print(f'  {a["code"]} {a["name"]} | price={a.get("last_price",0):.2f} | {a["reason"]}')

    print(f'\n--- 潜在2买 (等回踩确认) ---')
    for a in sorted(by_type.get('approaching_2buy', []), key=lambda x: x.get('distance_pct', 99))[:10]:
        print(f'  {a["code"]} {a["name"]} | price={a.get("last_price",0):.2f} | {a["reason"]}')

    print(f'\n--- 潜在1买 (MACD背驰中) ---')
    for a in sorted(by_type.get('approaching_1buy', []), key=lambda x: x.get('area_ratio', 1))[:10]:
        print(f'  {a["code"]} {a["name"]} | price={a.get("last_price",0):.2f} | {a["reason"]}')

    return pool


if __name__ == '__main__':
    generate_alert_pool()
