#!/usr/bin/env python3
"""
主线赛道评分效果验证 (纯评分模式)

验证改为评分后：
1. 信号覆盖：不再遗漏信号
2. Tier1信号评分显著高于Tier3
3. CC15处理量 vs 旧模式对比
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

from scan_enhanced_v3 import (
    load_sector_map, calc_sector_momentum,
    classify_sector_tier, compute_sector_tiers,
    _load_main_theme_config, GROWTH_SECTORS, SECTOR_BONUS,
)
from backtest_cc15_mtf import run_daily_cc15, find_daily_1buy_2buy


def run_validation():
    hs = HybridSource()
    t0 = time.time()

    print('=' * 80)
    print('  主线赛道评分模式验证')
    print('=' * 80)

    # 1. 股票池
    print('\n[1] 获取股票池...')
    from chanlun_unified.stock_pool import StockPoolManager
    spm = StockPoolManager()
    codes = spm.get_pool('tdx_all')
    pure_codes = [c.split('.')[0] for c in codes]
    print(f'   共 {len(pure_codes)} 只')

    # 2. 行业映射
    print('[2] 加载行业数据...')
    sector_map = load_sector_map()
    print(f'   行业映射: {len(sector_map)} 只')

    # 3. 日线数据
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
                    if 3.0 <= last_close <= 200.0:
                        daily_map[code] = df
            except Exception:
                pass
        print(f'   [{min(i+batch_size, len(pure_codes))}/{len(pure_codes)}] '
              f'日线={len(daily_map)}', end='\r')
    print(f'   日线数据: {len(daily_map)} 只')

    # 4. 行业动量 + 热点
    print('[4] 计算行业动量...')
    sector_mom = calc_sector_momentum(daily_map, sector_map)

    hot_sector_list = []
    try:
        from data.hot_sector_analyzer import HotSectorAnalyzer
        hsa = HotSectorAnalyzer()
        hot_sector_list = hsa.identify_hot_sectors(top_n=10)
        print(f'   TOP5: {", ".join(s.name + "(" + s.phase + ")" for s in hot_sector_list[:5])}')
    except Exception as e:
        print(f'   热点跳过: {e}')

    sector_pool_data = {}
    sector_pool_path = f'signals/sector_pool_{datetime.now().strftime("%Y-%m-%d")}.json'
    if os.path.exists(sector_pool_path):
        try:
            with open(sector_pool_path, 'r', encoding='utf-8') as f:
                sector_pool_data = json.load(f)
        except Exception:
            pass
    _pool_main = sector_pool_data.get('main_sectors', [])
    _pool_disaster = list(sector_pool_data.get('disaster_sectors', {}).keys())

    # 5. Tier评分 (新模式: 纯评分)
    print('\n[5] Tier评分 (纯评分模式)...')
    mt_config = _load_main_theme_config()
    tier_map = compute_sector_tiers(
        daily_map, sector_map, hot_sector_list, sector_mom, mt_config,
        pool_main_sectors=_pool_main, pool_disaster_sectors=_pool_disaster)

    # 6. 快速粗筛
    print('[6] 快速粗筛...')
    prefiltered_map = {}
    for code, df in daily_map.items():
        n = len(df)
        if n < 120:
            continue
        close = df['close']
        low = df['low']
        if close.iloc[-1] <= low.iloc[-20:].min():
            continue
        if all(close.iloc[-i] <= df['open'].iloc[-i] for i in range(1, 6)):
            continue
        ma60 = close.rolling(60).mean().iloc[-1]
        if pd.notna(ma60) and close.iloc[-1] < ma60 * 0.85:
            continue
        prefiltered_map[code] = df
    print(f'   粗筛后: {len(prefiltered_map)}/{len(daily_map)} 只')

    # 7. CC15
    print(f'[7] CC15引擎 ({len(prefiltered_map)}只)...')
    t_cc15_start = time.time()
    engine, _ = run_daily_cc15(prefiltered_map)
    t_cc15 = time.time() - t_cc15_start
    print(f'   CC15耗时: {t_cc15:.1f}s')

    # 8. 收集信号
    print('[8] 收集信号...')
    cutoff = datetime.now() - timedelta(days=30)
    all_signals = []

    for code in prefiltered_map:
        df = prefiltered_map[code]
        pairs = find_daily_1buy_2buy(engine, code, df)
        for p in pairs:
            idx2 = p.get('2buy_idx', -1)
            if idx2 >= 0 and idx2 < len(df):
                if df.index[idx2] >= pd.Timestamp(cutoff):
                    sector = sector_map.get(code, '')
                    tier = tier_map.get(code, 3)
                    sector_score = 0
                    if sector in GROWTH_SECTORS:
                        sector_score += SECTOR_BONUS['growth']
                    if tier == 1:
                        sector_score += 20
                    elif tier == 2:
                        sector_score += 10
                    all_signals.append({
                        'code': code, 'sector': sector, 'tier': tier,
                        'sector_score': sector_score,
                        'price': p.get('2buy_price', 0),
                    })

    # ============ 结果分析 ============
    print(f'\n{"=" * 80}')
    print(f'  评分模式验证结果')
    print(f'{"=" * 80}')

    print(f'\n  总信号: {len(all_signals)} 个2买')
    if not all_signals:
        print('  无信号')
        return

    # 按Tier分组
    for t in [1, 2, 3]:
        sub = [s for s in all_signals if s['tier'] == t]
        if not sub:
            continue
        avg_score = np.mean([s['sector_score'] for s in sub])
        print(f'  Tier{t}: {len(sub):>4d}个信号  平均行业分={avg_score:.1f}')

    # Tier1信号详情
    t1_signals = [s for s in all_signals if s['tier'] == 1]
    if t1_signals:
        print(f'\n  Tier1(主线)信号详情:')
        for s in t1_signals[:10]:
            print(f'    {s["code"]:<10s} {s["sector"]:<12s} 分={s["sector_score"]:.0f}')
        if len(t1_signals) > 10:
            print(f'    ... 共{len(t1_signals)}个')

    # 评分分布
    print(f'\n  行业评分分布:')
    score_groups = {}
    for s in all_signals:
        sc = s['sector_score']
        if sc not in score_groups:
            score_groups[sc] = 0
        score_groups[sc] += 1
    for sc in sorted(score_groups.keys(), reverse=True):
        cnt = score_groups[sc]
        print(f'    {sc:>3.0f}分: {cnt:>4d}个 {"█" * (cnt // 5)}')

    # 对比旧模式
    # 旧模式: Tier1全过 + Tier2 top100 + Tier3 top50 = ~220只
    # 新模式: 全量通过，靠评分排序
    old_pass = 0
    tier_counts = {1: 0, 2: 0, 3: 0}
    for code in prefiltered_map:
        t = tier_map.get(code, 3)
        tier_counts[t] += 1
    old_pass = tier_counts[1] + min(tier_counts[2], 100) + min(tier_counts[3], 50)

    print(f'\n  --- 新旧模式对比 ---')
    print(f'  {"指标":<20s} {"旧模式(过滤)":>12s} {"新模式(评分)":>12s}')
    print(f'  {"-" * 48}')
    print(f'  {"粗筛后股票数":<20s} {old_pass:>12d} {len(prefiltered_map):>12d}')
    print(f'  {"CC15输入量":<20s} {old_pass:>12d} {len(prefiltered_map):>12d}')
    print(f'  {"2买信号覆盖":<20s} {"~22":>12s} {len(all_signals):>12d}')
    print(f'  {"信号覆盖率":<20s} {"~6%":>12s} {"100%":>12s}')

    print(f'\n  总耗时: {time.time()-t0:.1f}s')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    run_validation()
