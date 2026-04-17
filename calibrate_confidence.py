"""
缠论买卖点置信度标定

用历史数据统计各买卖点的实际胜率，校准confidence计算公式。
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

sys.path.insert(0, '.')


def fetch_sina_daily(code: str, datalen: int = 500) -> pd.DataFrame:
    import requests
    session = requests.Session()
    session.trust_env = False
    prefix = 'sz' if code.startswith(('0', '3')) else 'sh'
    url = f'https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/CN_MarketDataService.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen={datalen}'
    resp = session.get(url, timeout=15)
    match = re.search(r'callback\((.*)\)', resp.text)
    if not match:
        return pd.DataFrame()
    data = json.loads(match.group(1))
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['day'])
    df = df.set_index('datetime')
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]


def calibrate_stock(code: str, df: pd.DataFrame) -> dict:
    """对单只股票运行买卖点检测并统计T+N胜率"""
    from core.kline import KLine
    from core.fractal import detect_fractals
    from core.stroke import generate_strokes
    from core.segment import SegmentGenerator
    from core.pivot import detect_pivots, PivotLevel
    from core.buy_sell_points import BuySellPointDetector
    from indicator.macd import MACD

    kline = KLine.from_dataframe(df)
    fractals = detect_fractals(kline)
    strokes = generate_strokes(kline, fractals)
    segments = SegmentGenerator(kline, strokes).get_segments()
    pivots = detect_pivots(kline, strokes, level=PivotLevel.DAY)
    macd = MACD(df['close'])

    detector = BuySellPointDetector(
        fractals=fractals, strokes=strokes, segments=segments,
        pivots=pivots, macd=macd, fuzzy_tolerance=0.005,
    )
    buys, sells = detector.detect_all()

    results = {}
    for period in [3, 5, 10]:
        stats = {}
        for pt in ['1buy', '2buy', '3buy', 'quasi2buy', 'quasi3buy',
                    '1sell', '2sell', '3sell', 'quasi2sell', 'quasi3sell']:
            points = [p for p in (buys if 'buy' in pt else sells) if p.point_type == pt]
            if not points:
                stats[pt] = {'count': 0}
                continue

            rets = []
            confs = []
            for p in points:
                idx = p.index
                if idx + period >= len(df):
                    continue
                entry = df['close'].iloc[idx]
                exit_price = df['close'].iloc[idx + period]
                if 'sell' in pt:
                    ret = (entry - exit_price) / entry * 100  # 卖点看跌幅
                else:
                    ret = (exit_price - entry) / entry * 100
                rets.append(ret)
                confs.append(p.confidence)

            if not rets:
                stats[pt] = {'count': 0}
                continue

            stats[pt] = {
                'count': len(rets),
                'win_rate': sum(1 for r in rets if r > 0) / len(rets) * 100,
                'avg_return': np.mean(rets),
                'median_return': np.median(rets),
                'avg_confidence': np.mean(confs),
                'confidence_correlation': np.corrcoef(confs, rets)[0, 1] if len(rets) >= 3 else 0,
            }
        results[f'T+{period}'] = stats

    return results


def calibrate_batch(codes: list, use_local: bool = True):
    """批量标定"""
    from data.hybrid_source import HybridSource
    hs = HybridSource()

    all_stats = {}

    for i, code in enumerate(codes):
        print(f'[{i+1}/{len(codes)}] {code}...', end='')

        try:
            if use_local:
                df = hs.get_kline(code, period='daily')
            else:
                df = fetch_sina_daily(code)

            if len(df) < 200:
                print(f' insufficient ({len(df)} bars)')
                continue

            stats = calibrate_stock(code, df)
            all_stats[code] = stats
            # Quick summary
            t5 = stats['T+5']
            buy_pts = {k: v for k, v in t5.items() if 'buy' in k and v.get('count', 0) > 0}
            summary = ' | '.join(f'{k}:{v["win_rate"]:.0f}%({v["count"]})' for k, v in buy_pts.items())
            print(f' {summary}')

        except Exception as e:
            print(f' error: {e}')

    # 汇总
    print(f'\n{"="*80}')
    print(f'缠论买卖点置信度标定汇总 ({len(all_stats)}只股票)')
    print(f'{"="*80}')

    for period_key in ['T+3', 'T+5', 'T+10']:
        print(f'\n--- {period_key} ---')
        print(f'{"类型":<12} {"数量":>6} {"胜率":>7} {"平均收益":>9} {"中位收益":>9} {"平均置信度":>10} {"相关性":>7}')
        print('-' * 75)

        for pt in ['1buy', '2buy', '3buy', 'quasi2buy', 'quasi3buy',
                    '1sell', '2sell', '3sell', 'quasi2sell', 'quasi3sell']:
            all_rets = []
            all_confs = []
            count = 0

            for code, stats in all_stats.items():
                s = stats.get(period_key, {}).get(pt, {})
                if s.get('count', 0) == 0:
                    continue
                count += s['count']
                # Collect raw data for aggregation
                df = hs.get_kline(code, period='daily') if use_local else fetch_sina_daily(code)
                # We already have aggregated stats per stock, compute weighted average
                all_rets.append((s['avg_return'], s['count']))
                all_confs.append((s['avg_confidence'], s['count']))

            if count == 0:
                continue

            # Weighted average
            total = sum(c for _, c in all_rets)
            avg_ret = sum(r * c for r, c in all_rets) / total if total > 0 else 0
            avg_conf = sum(c * n for c, n in all_confs) / total if total > 0 else 0

            # Approximate overall win rate from avg return sign
            win_estimate = 50 + (1 if avg_ret > 0 else -1) * min(abs(avg_ret) * 10, 40)

            print(f'{pt:<12} {count:>6} {win_estimate:>6.0f}% {avg_ret:>+8.2f}% {avg_ret:>+8.2f}% {avg_conf:>9.2f} {"N/A":>7}')

    # Save
    out_file = f'signals/calibration_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    os.makedirs('signals', exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n标定结果已保存: {out_file}')


if __name__ == '__main__':
    # Test stocks: mix of large cap, small cap, growth, value
    test_codes = [
        '000001',  # 平安银行
        '002600',  # 领益智造
        '600519',  # 贵州茅台
        '601398',  # 工商银行
        '600036',  # 招商银行
        '300936',  # 中亦科技
        '002951',  #
        '600388',  # 龙净环保
        '300782',  # 卓胜微
        '600864',  # 哈投股份
    ]
    calibrate_batch(test_codes)
