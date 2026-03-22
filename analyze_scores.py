"""
分析得分构成，找出优化方向
"""

import os
import struct
import pandas as pd
from collections import Counter


def is_valid_stock(code: str) -> bool:
    if code.startswith('sh6') and not code.startswith('sh688'): return True
    if code.startswith('sh688'): return True
    if code.startswith('sz0'): return True
    if code.startswith('sz3'): return True
    return False


def get_sector(code: str) -> str:
    if code.startswith('sh688'): return '科创板'
    if code.startswith('sz300'): return '创业板'
    if code.startswith('sh60'): return '沪主板'
    if code.startswith('sz0'): return '深主板'
    return '其他'


def read_day_file(filepath):
    data = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(32)
            if len(chunk) < 32: break
            v = struct.unpack('IIIIIfII', chunk)
            try:
                date = pd.to_datetime(str(v[0]), format='%Y%m%d')
                open_p, high, low, close = v[1]/100, v[2]/100, v[3]/100, v[4]/100
                if 0 < close < 200:
                    data.append({'date': date, 'open': open_p, 'high': high, 'low': low, 'close': close})
            except: pass
    if data:
        return pd.DataFrame(data).set_index('date')
    return pd.DataFrame()


# 导入之前的评分逻辑
import sys
sys.path.insert(0, 'D:/新建文件夹/claude')
from optimized_screener import WeightedScreener


def analyze_score_details(screener, start_idx=-30, min_score=8):
    """分析得分构成"""
    all_dates = set()
    for df in screener.stock_data.values():
        all_dates.update(df.index)
    all_dates = sorted(list(all_dates))[start_idx:]

    # 收集所有得分详情
    all_scores = []

    for i, date in enumerate(all_dates[:-3]):
        market_stats = screener._calc_market_stats(date)
        sector_stats = screener._calc_sector_stats(date, market_stats['stock_change'])

        for code in screener.stock_data.keys():
            result = screener.score_stock(code, date, market_stats, sector_stats)
            if result['total_score'] >= min_score:
                df = screener.stock_data[code]

                buy_date = all_dates[all_dates.index(date) + 1]
                sell_date = all_dates[all_dates.index(date) + 2]

                if buy_date not in df.index or sell_date not in df.index:
                    continue

                buy_price = df.loc[buy_date, 'open']
                sell_price = df.loc[sell_date, 'close']
                profit = (sell_price - buy_price) / buy_price * 100

                score_info = {
                    'date': date,
                    'code': code,
                    'score': result['total_score'],
                    'details': result['details'],
                    'profit': profit,
                    'sector': result['sector']
                }
                all_scores.append(score_info)

    # 统计各得分项的出现频率和盈利情况
    print("\n" + "="*60)
    print("得分项分析")
    print("="*60)

    # 统计每个得分项
    item_stats = {}
    for s in all_scores:
        for item, points in s['details'].items():
            if item not in item_stats:
                item_stats[item] = {'count': 0, 'profits': []}
            item_stats[item]['count'] += 1
            item_stats[item]['profits'].append(s['profit'])

    print("\n得分项统计:")
    print("-" * 50)
    for item, stats in sorted(item_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        avg_profit = sum(stats['profits']) / len(stats['profits'])
        win_rate = sum(1 for p in stats['profits'] if p > 0) / len(stats['profits']) * 100
        print(f"{item:12s}: {stats['count']:3d}次, 胜率{win_rate:5.1f}%, 平均收益{avg_profit:6.2f}%")

    # 按分数分析
    print("\n按总分分析:")
    print("-" * 50)
    for score in sorted(set(s['score'] for s in all_scores)):
        subset = [s for s in all_scores if s['score'] == score]
        avg_profit = sum(s['profit'] for s in subset) / len(subset)
        win_rate = sum(1 for s in subset if s['profit'] > 0) / len(subset) * 100

        # 统计这个分数的常见得分组合
        detail_combos = Counter(tuple(sorted(s['details'].items())) for s in subset)
        most_common = detail_combos.most_common(1)

        print(f"\n{score}分: {len(subset)}笔, 胜率{win_rate:.1f}%, 平均{avg_profit:.2f}%")
        if most_common:
            combo, count = most_common[0]
            print(f"  最常见组合({count}笔): {dict(combo)}")

    # 找出亏损的高分股票
    print("\n高分亏损案例:")
    print("-" * 50)
    high_loss = [s for s in all_scores if s['score'] >= 8 and s['profit'] < -5]
    for s in high_loss[:10]:
        print(f"{s['code']} {s['date'].strftime('%Y-%m-%d')}: {s['score']}分, 亏损{s['profit']:.2f}%, 得分项={s['details']}")

    # 找出盈利的高分股票
    print("\n高分盈利案例:")
    print("-" * 50)
    high_profit = [s for s in all_scores if s['score'] >= 8 and s['profit'] > 5]
    for s in high_profit[:10]:
        print(f"{s['code']} {s['date'].strftime('%Y-%m-%d')}: {s['score']}分, 盈利{s['profit']:.2f}%, 得分项={s['details']}")


def main():
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    screener = WeightedScreener(tdx_path)
    screener.load_data(max_stocks=2000)

    analyze_score_details(screener, start_idx=-30, min_score=6)


if __name__ == "__main__":
    main()
