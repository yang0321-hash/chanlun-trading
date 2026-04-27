#!/usr/bin/env python3
"""观察池回踩监控 — 定时检查BUY候选股回踩确认

使用: python signals/watch_observer.py
"""
import sys, os, json, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import requests
import numpy as np
import pandas as pd
from datetime import datetime

POOL_FILE = 'signals/observation_pool.json'
NOTIFY_FILE = 'signals/observation_alerts.json'


def get_realtime_prices(codes):
    """Sina批量获取实时报价"""
    session = requests.Session()
    session.trust_env = False
    sina_codes = []
    for c in codes:
        if c.startswith('sz'):
            sina_codes.append(c)
        elif c.startswith('sh'):
            sina_codes.append(c)
        else:
            prefix = 'sz' if c.startswith(('0', '3')) else 'sh'
            sina_codes.append(f'{prefix}{c}')

    url = f'https://hq.sinajs.cn/list={",".join(sina_codes)}'
    resp = session.get(url, headers={'Referer': 'https://finance.sina.com.cn'}, timeout=10)
    resp.encoding = 'gbk'

    prices = {}
    for line in resp.text.strip().split('\n'):
        if '=' not in line:
            continue
        parts = line.split('="')
        if len(parts) < 2:
            continue
        raw = parts[0]
        code = raw.replace('var hq_str_', '')
        data = parts[1].rstrip('";')
        fields = data.split(',')
        if len(fields) < 6:
            continue
        name = fields[0]
        price = float(fields[3]) if float(fields[3]) > 0 else float(fields[2])
        yesterday = float(fields[2])
        high = float(fields[4])
        low = float(fields[5])
        change = (price - yesterday) / yesterday * 100 if yesterday > 0 else 0
        prices[code] = {
            'name': name, 'price': price, 'yesterday': yesterday,
            'high': high, 'low': low, 'change_pct': change,
        }
    return prices


def check_pullback(stock, price_info):
    """检查是否回踩到入场区"""
    p = price_info['price']
    entry_zone = stock['entry_zone']
    # Parse entry zone "70-72"
    try:
        lo, hi = entry_zone.split('(')[0].strip().split('-')
        lo, hi = float(lo), float(hi)
    except:
        # Fallback: 3% below watch price
        lo = stock['watch_price'] * 0.97
        hi = stock['watch_price'] * 1.00

    if lo <= p <= hi:
        return True, f'价格{p:.2f}在入场区[{lo:.1f}-{hi:.1f}]'
    if p < lo:
        return True, f'价格{p:.2f}已低于入场区下限{lo:.1f}'
    return False, f'价格{p:.2f}未回踩,高于入场区{hi:.1f}'


def check_chanlun_signal(code, price_info):
    """快速缠论买点确认"""
    try:
        from data.hybrid_source import HybridSource
        from core.kline import KLine
        from core.fractal import FractalDetector
        from core.stroke import StrokeGenerator
        from core.pivot import PivotDetector
        from indicator.macd import MACD
        from core.buy_sell_points import BuySellPointDetector

        hs = HybridSource()
        df = hs.get_kline(code, period='daily')
        if df is None or len(df) < 60:
            return None, '数据不足'

        close_s = pd.Series(df['close'].values)
        macd = MACD(close_s)
        kline = KLine.from_dataframe(df, strict_mode=False)
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        if len(fractals) < 4:
            return None, '分型不足'
        strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
        if len(strokes) < 3:
            return None, '笔不足'
        pivots = PivotDetector(kline, strokes).get_pivots()
        det = BuySellPointDetector(fractals, strokes, [], pivots, macd=macd)
        buys, _ = det.detect_all()

        # Check for recent buy signals (last 5 bars)
        n = len(df)
        recent_buys = [b for b in buys if b.index >= n - 5]
        if recent_buys:
            best = max(recent_buys, key=lambda b: b.confidence)
            types = [b.point_type for b in recent_buys]
            return best, f'最近买点: {",".join(set(types))} conf={best.confidence:.2f}'
        return None, '无近期买点'
    except Exception as e:
        return None, f'分析异常: {e}'


def main():
    if not os.path.exists(POOL_FILE):
        print('观察池为空')
        return

    with open(POOL_FILE, encoding='utf-8') as f:
        pool = json.load(f)

    stocks = [s for s in pool['stocks'] if s['status'] == 'watching']
    if not stocks:
        print('无观察中的股票')
        return

    now = datetime.now().strftime('%H:%M')
    print(f'=== 观察池监控 {now} ===')

    codes = [s['code'] for s in stocks]
    prices = get_realtime_prices(codes)

    alerts = []
    for s in stocks:
        code = s['code']
        if code not in prices:
            print(f'  {s["name"]}: 无法获取报价')
            continue

        pi = prices[code]
        pullback, msg = check_pullback(s, pi)

        status_mark = ' ★' if pullback else ''
        print(f'  {s["name"]} {pi["price"]:.2f} ({pi["change_pct"]:+.2f}%) {msg}{status_mark}')

        if pullback:
            # Run ChanLun confirmation
            sig, sig_msg = check_chanlun_signal(code, pi)
            print(f'    缠论确认: {sig_msg}')

            if sig:
                alert = {
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'code': code,
                    'name': s['name'],
                    'price': pi['price'],
                    'stop_loss': s['stop_loss'],
                    'score': s['score'],
                    'signal_type': sig.point_type,
                    'confidence': sig.confidence,
                    'message': f'{s["name"]} 回踩确认! {sig.point_type} conf={sig.confidence:.2f} 价格={pi["price"]:.2f} 止损={s["stop_loss"]:.2f}',
                }
                alerts.append(alert)
                print(f'    *** 回踩确认! {sig.point_type} conf={sig.confidence:.2f} ***')

    # Save alerts
    if alerts:
        all_alerts = []
        if os.path.exists(NOTIFY_FILE):
            with open(NOTIFY_FILE, encoding='utf-8') as f:
                all_alerts = json.load(f)
        all_alerts.extend(alerts)
        with open(NOTIFY_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_alerts, f, ensure_ascii=False, indent=2)
        print(f'\n*** {len(alerts)}个回踩确认信号! 查看 signals/observation_alerts.json ***')
    else:
        print('\n暂无回踩确认')

    return alerts


if __name__ == '__main__':
    main()
