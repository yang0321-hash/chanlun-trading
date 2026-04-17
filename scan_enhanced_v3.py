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

    # 5. CC15引擎 + 找2买
    print('[5] 运行CC15引擎 + 识别2买...')
    engine, daily_signals = run_daily_cc15(daily_map)

    cutoff = datetime.now() - timedelta(days=lookback_days)
    recent_2buys = []

    for code in daily_map:
        pairs = find_daily_1buy_2buy(engine, code, daily_map[code])
        for p in pairs:
            p['code'] = code
            df = daily_map[code]
            if p['2buy_idx'] >= len(df):
                continue
            sig_date = df.index[p['2buy_idx']]
            if sig_date >= pd.Timestamp(cutoff):
                recent_2buys.append(p)

    print(f'   最近{lookback_days}天2买信号: {len(recent_2buys)} 个')

    if not recent_2buys:
        print('无近期2买信号')
        return []

    # 6. 30min确认 + 评分
    print('[6] 30分钟确认 + 综合评分...')
    results = []
    scanned = set()

    for item in recent_2buys:
        code = item['code']
        if code in scanned:
            continue
        scanned.add(code)

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
        entry_price = item['2buy_price']
        stop_price = item['1buy_low']
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

        total_score = tech_score + sector_score

        results.append({
            'code': code,
            'name': name,
            'sector': sector,
            'sector_ret': round(sector_ret, 2),
            'price': price,
            'pct_chg': pct,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'risk_reward': round(rr_ratio, 1),
            '2buy_date': str(df.index[item['2buy_idx']].date()),
            'pivot_info': pivot_info,
            'last_close': last_close,
            'tech_score': tech_score,
            'sector_score': sector_score,
            'total_score': total_score,
        })

        time.sleep(0.1)

    # 7. 排序输出
    elapsed = time.time() - t0
    results.sort(key=lambda x: x['total_score'], reverse=True)

    print(f'\n{"="*90}')
    print(f'扫描完成 ({elapsed:.0f}s) — {len(results)} 只候选股, 显示Top {top_n}')
    print(f'{"="*90}')

    print(f'\n{"排名":<4} {"代码":<8} {"名称":<8} {"行业":<8} {"行业涨幅":>7} '
          f'{"现价":>8} {"入场价":>8} {"R/R":>5} {"评分":>4} {"2买日":<12}')
    print('-' * 95)

    for i, r in enumerate(results[:top_n]):
        print(f'{i+1:<4} {r["code"]:<8} {r["name"]:<8} {r["sector"]:<8} '
              f'{r["sector_ret"]:>+6.1f}% {r["price"]:>8.2f} {r["entry_price"]:>8.2f} '
              f'{r["risk_reward"]:>5.1f} {r["total_score"]:>4} {r["2buy_date"]:<12}')

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
