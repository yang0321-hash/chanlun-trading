#!/usr/bin/env python3
"""
信号质量回测验证

验证扫描器各类型信号的历史表现:
- 1买/2买/3买各类型的胜率、平均收益、P/L比
- buy_strength/sector_tier/评分等维度对信号质量的区分能力
- lookback=90天, 用TDX日线数据计算后续收益
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.hybrid_source import HybridSource
from scan_enhanced_v3 import (
    _mp_worker, load_sector_map, compute_sector_tiers,
    _load_main_theme_config, GROWTH_SECTORS, SECTOR_BONUS,
    score_technical,
)
from chanlun_unified.stock_pool import StockPoolManager
from multiprocessing import Pool


LOOKBACK_DAYS = 90
MIN_FUTURE_BARS = 5


def compute_forward_returns(df, sig_idx, entry_price, stop_price, max_days=20):
    future_start = sig_idx + 1
    future_end = min(sig_idx + 1 + max_days, len(df))
    if future_start >= len(df):
        return None
    future = df.iloc[future_start:future_end]
    if len(future) < MIN_FUTURE_BARS:
        return None

    ret = {}
    for days, label in [(5, '5d'), (10, '10d'), (20, '20d')]:
        if len(future) >= days:
            ret[label] = (future.iloc[days - 1]['close'] - entry_price) / entry_price * 100
        else:
            ret[label] = None

    if len(future) >= 5:
        lows = future['low'].values
        highs = future['high'].values
        ret['max_dd'] = float(np.min((lows - entry_price) / entry_price * 100))
        ret['max_gain'] = float(np.max((highs - entry_price) / entry_price * 100))

    ret['sl_hit'] = bool(np.any(future['low'].values <= stop_price))
    if ret['sl_hit']:
        for i in range(len(future)):
            if future.iloc[i]['low'] <= stop_price:
                ret['sl_hit_day'] = i + 1
                break
    else:
        ret['sl_hit_day'] = -1

    if ret['sl_hit']:
        ret['composite'] = (stop_price - entry_price) / entry_price * 100
    elif ret.get('20d') is not None:
        ret['composite'] = ret['20d']
    elif ret.get('10d') is not None:
        ret['composite'] = ret['10d']
    else:
        ret['composite'] = ret.get('5d', 0)

    return ret


def _print_group(signals, field, labels=None):
    groups = {}
    for s in signals:
        g = s.get(field, '?')
        if g is None or (isinstance(g, float) and np.isnan(g)):
            g = '?'
        groups.setdefault(g, []).append(s)

    if not groups:
        return

    lbl = (lambda x: labels.get(x, str(x))) if labels else str

    print(f'\n  [{field}]')
    print(f'  {"分组":<16s} {"数量":>6s} {"胜率5d":>8s} {"胜率20d":>8s} '
          f'{"平均20d":>8s} {"中位20d":>8s} {"P/L比":>6s} {"止损率":>7s}')
    print(f'  {"-" * 80}')

    for g in sorted(groups.keys(), key=lambda x: len(groups[x]), reverse=True):
        sub = groups[g]
        n = len(sub)
        r5 = [s['fwd']['5d'] for s in sub if s['fwd'].get('5d') is not None]
        r20 = [s['fwd']['20d'] for s in sub if s['fwd'].get('20d') is not None]
        rc = [s['fwd']['composite'] for s in sub]
        sl = sum(1 for s in sub if s['fwd'].get('sl_hit'))

        wr5 = sum(1 for r in r5 if r > 0) / len(r5) * 100 if r5 else 0
        wr20 = sum(1 for r in r20 if r > 0) / len(r20) * 100 if r20 else 0
        a20 = np.mean(r20) if r20 else 0
        m20 = np.median(r20) if r20 else 0

        w = [r for r in rc if r > 0]
        l = [r for r in rc if r <= 0]
        pl = abs(np.mean(w) / np.mean(l)) if l and np.mean(l) != 0 else (999 if w else 0)
        slr = sl / n * 100

        print(f'  {lbl(g):<16s} {n:>6d} {wr5:>7.1f}% {wr20:>7.1f}% '
              f'{a20:>+7.2f}% {m20:>+7.2f}% {pl:>5.2f} {slr:>6.1f}%')


def main():
    hs = HybridSource()
    t0 = time.time()

    print('=' * 90)
    print(f'  信号质量回测验证 (lookback={LOOKBACK_DAYS}天)')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 90)

    # 1. 股票池 + 日线
    print('\n[1] 加载数据...')
    spm = StockPoolManager()
    codes = spm.get_pool('tdx_all')
    pure_codes = [c.split('.')[0] for c in codes]
    print(f'   股票池: {len(pure_codes)} 只')

    daily_map = hs.load_all_daily(pure_codes, min_price=3.0, max_price=200.0, min_bars=200)
    print(f'   日线数据: {len(daily_map)} 只 ({time.time() - t0:.1f}s)')

    # 2. 行业Tier
    print('[2] 行业Tier...')
    sector_map = load_sector_map()
    mt_config = _load_main_theme_config()
    tier_map = compute_sector_tiers(daily_map, sector_map, [], {}, mt_config)
    tc = {}
    for code in daily_map:
        t = tier_map.get(code, 3)
        tc[t] = tc.get(t, 0) + 1
    print(f'   Tier1={tc.get(1, 0)} Tier2={tc.get(2, 0)} Tier3={tc.get(3, 0)}')

    # 3. 粗筛
    print('[3] 快速粗筛...')
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
        if n >= 60:
            ma20 = close.rolling(20).mean().iloc[-1]
            if pd.notna(ma60) and pd.notna(ma20) and ma20 < ma60 * 0.92:
                continue
        if n >= 35:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            if dif.iloc[-1] < dif.iloc[-60:].min() * 0.5:
                continue
        prefiltered_map[code] = df
    print(f'   粗筛后: {len(prefiltered_map)}/{len(daily_map)} 只')

    # 4. 缠论分析 (多进程)
    print(f'[4] 缠论分析 (多进程, {LOOKBACK_DAYS}天回看)...')
    t_scan_start = time.time()

    cutoff_ts = datetime.now() - timedelta(days=LOOKBACK_DAYS)

    stock_data = {}
    for code, df in prefiltered_map.items():
        stock_data[code] = {
            'open': df['open'].values, 'high': df['high'].values,
            'low': df['low'].values, 'close': df['close'].values,
            'volume': df['volume'].values, 'index': df.index,
        }

    task_args = [(code, stock_data[code], 'weak', cutoff_ts)
                 for code in stock_data]

    all_signals = []
    n_workers = min(os.cpu_count() or 4, 8)
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_mp_worker, task_args, chunksize=50):
            all_signals.extend(result)

    t_scan = time.time() - t_scan_start
    type_counts = {}
    for s in all_signals:
        t = s.get('signal_type', '?')
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f'   信号: {dict(type_counts)} 共{len(all_signals)}个 ({t_scan:.1f}s)')

    # 5. 计算后续收益
    print('[5] 计算后续收益...')
    validated = []
    no_future = 0
    for sig in all_signals:
        code = sig['code']
        df = daily_map.get(code)
        if df is None:
            continue

        sig_idx = sig.get('sig_idx', -1)
        entry_price = sig.get('entry_price', 0)
        stop_price = sig.get('stop_price', entry_price * 0.95)

        if sig_idx < 0 or entry_price <= 0:
            continue

        fwd = compute_forward_returns(df, sig_idx, entry_price, stop_price)
        if fwd is None:
            no_future += 1
            continue

        sector = sector_map.get(code, '未知')
        tier = tier_map.get(code, 3)
        st = sig.get('signal_type', '')
        bs = sig.get('buy_strength', '')

        # 技术评分
        risk = entry_price - stop_price
        if risk > 0:
            recent_high = df['high'].iloc[-20:].max()
            reward = recent_high - entry_price
            rr = reward / risk
        else:
            rr = 0
        tech_score = score_technical(df, entry_price, stop_price, rr)

        # 行业评分
        sector_score = 0
        if sector in GROWTH_SECTORS:
            sector_score += SECTOR_BONUS['growth']
        if tier == 1:
            sector_score += 20
        elif tier == 2:
            sector_score += 10

        # 强度加分
        strength_bonus = 0
        if st == '2buy':
            if bs == 'medium':
                strength_bonus += 12
            elif bs == 'strong':
                strength_bonus += 3
            else:
                strength_bonus += 5
        else:
            if bs == 'strong':
                strength_bonus += 10
            elif bs == 'standard':
                strength_bonus += 5

        # 3买加权
        if st == '3buy':
            weighted = sig.get('three_buy_weighted', 0)
            if weighted >= 2:
                strength_bonus += 15
            elif weighted >= 1:
                strength_bonus += 6
            elif weighted < 0:
                strength_bonus -= 5

        total_score = tech_score + sector_score + strength_bonus

        validated.append({
            'code': code, 'signal_type': st, 'buy_strength': bs,
            'entry_price': entry_price, 'stop_price': stop_price,
            'sig_date': str(sig.get('sig_date', '')),
            'sector': sector, 'sector_tier': tier,
            'tech_score': tech_score, 'sector_score': sector_score,
            'strength_bonus': strength_bonus, 'total_score': total_score,
            'confidence': sig.get('confidence', 0.5),
            'three_buy_passed': sig.get('three_buy_passed', 0),
            'three_buy_weighted': sig.get('three_buy_weighted', 0),
            'trend_type': sig.get('trend_type', ''),
            'fwd': fwd,
        })

    print(f'   有效信号: {len(validated)} (无后续数据: {no_future})')

    if not validated:
        print('   无有效信号')
        return

    # ============ 统计分析 ============
    elapsed = time.time() - t0
    print(f'\n{"=" * 90}')
    print(f'  信号质量验证结果')
    print(f'  {LOOKBACK_DAYS}天 | {len(prefiltered_map)}只股票 | {len(validated)}个有效信号 | {elapsed:.0f}s')
    print(f'{"=" * 90}')

    # 整体统计
    print(f'\n  [整体统计]')
    rets_5d = [s['fwd']['5d'] for s in validated if s['fwd'].get('5d') is not None]
    rets_10d = [s['fwd']['10d'] for s in validated if s['fwd'].get('10d') is not None]
    rets_20d = [s['fwd']['20d'] for s in validated if s['fwd'].get('20d') is not None]
    rets_comp = [s['fwd']['composite'] for s in validated]

    for label, rets in [('5d', rets_5d), ('10d', rets_10d), ('20d', rets_20d)]:
        if not rets:
            continue
        wr = sum(1 for r in rets if r > 0) / len(rets) * 100
        avg = np.mean(rets)
        med = np.median(rets)
        print(f'  {label:>4s}持有: 胜率={wr:.1f}% 平均={avg:+.2f}% 中位={med:+.2f}% (n={len(rets)})')

    wr_c = sum(1 for r in rets_comp if r > 0) / len(rets_comp) * 100
    avg_c = np.mean(rets_comp)
    avg_dd = np.mean([s['fwd']['max_dd'] for s in validated])
    avg_gain = np.mean([s['fwd']['max_gain'] for s in validated])
    sl_rate = sum(1 for s in validated if s['fwd'].get('sl_hit')) / len(validated) * 100
    w = [r for r in rets_comp if r > 0]
    l = [r for r in rets_comp if r <= 0]
    pl = abs(np.mean(w) / np.mean(l)) if l and np.mean(l) != 0 else (999 if w else 0)
    print(f'\n  综合: 胜率={wr_c:.1f}% 平均={avg_c:+.2f}% P/L={pl:.2f}')
    print(f'  平均最大回撤={avg_dd:.2f}% 平均最大涨幅={avg_gain:.2f}% 止损率={sl_rate:.1f}%')

    # 按维度分组
    _print_group(validated, 'signal_type',
                 {'1buy': '1买', '2buy': '2买', '3buy': '3买'})

    if any(s['signal_type'] == '2buy' for s in validated):
        _print_group([s for s in validated if s['signal_type'] == '2buy'],
                     'buy_strength', {'strong': '强2买', 'medium': '类2买', 'weak': '弱2买', '': '未分类'})

    if any(s['signal_type'] == '3buy' for s in validated):
        _print_group([s for s in validated if s['signal_type'] == '3buy'],
                     'three_buy_passed', {0: '0项', 1: '1项', 2: '2项', 3: '3项', 4: '4项', 5: '5项'})

    _print_group(validated, 'sector_tier',
                 {1: 'Tier1主线', 2: 'Tier2活跃', 3: 'Tier3其他'})

    _print_group(validated, 'trend_type',
                 {'up': '上涨', 'down': '下跌', 'consolidation': '盘整', '': '未知'})

    # 评分四分位
    scores = [s['total_score'] for s in validated]
    p75, p50, p25 = np.percentile(scores, [75, 50, 25])
    for s in validated:
        sc = s['total_score']
        if sc >= p75:
            s['_sq'] = f'Q1(≥{p75:.0f})'
        elif sc >= p50:
            s['_sq'] = f'Q2({p50:.0f}-{p75:.0f})'
        elif sc >= p25:
            s['_sq'] = f'Q3({p25:.0f}-{p50:.0f})'
        else:
            s['_sq'] = f'Q4(<{p25:.0f})'
    _print_group(validated, '_sq')

    # Top/Bottom 10
    print(f'\n  [Top 10 最高分信号]')
    print(f'  {"代码":<8s} {"类型":<5s} {"强度":<8s} {"评分":>4s} '
          f'{"5d%":>7s} {"10d%":>7s} {"20d%":>7s} {"综合%":>7s}')
    print(f'  {"-" * 65}')
    for s in sorted(validated, key=lambda x: x['total_score'], reverse=True)[:10]:
        f = s['fwd']
        r5 = f'{f["5d"]:>+6.1f}%' if f.get('5d') is not None else '   N/A'
        r10 = f'{f["10d"]:>+6.1f}%' if f.get('10d') is not None else '   N/A'
        r20 = f'{f["20d"]:>+6.1f}%' if f.get('20d') is not None else '   N/A'
        print(f'  {s["code"]:<8s} {s["signal_type"]:<5s} {s["buy_strength"]:<8s} '
              f'{s["total_score"]:>4d} {r5:>7s} {r10:>7s} {r20:>7s} {f["composite"]:>+6.1f}%')

    print(f'\n  [Bottom 10 最低分信号]')
    for s in sorted(validated, key=lambda x: x['total_score'])[:10]:
        f = s['fwd']
        r5 = f'{f["5d"]:>+6.1f}%' if f.get('5d') is not None else '   N/A'
        r10 = f'{f["10d"]:>+6.1f}%' if f.get('10d') is not None else '   N/A'
        r20 = f'{f["20d"]:>+6.1f}%' if f.get('20d') is not None else '   N/A'
        print(f'  {s["code"]:<8s} {s["signal_type"]:<5s} {s["buy_strength"]:<8s} '
              f'{s["total_score"]:>4d} {r5:>7s} {r10:>7s} {r20:>7s} {f["composite"]:>+6.1f}%')

    print(f'\n  总耗时: {time.time() - t0:.0f}s')
    print(f'{"=" * 90}')

    # Save
    out = {
        'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'lookback_days': LOOKBACK_DAYS,
        'total_signals': len(validated),
        'signals': [{
            'code': s['code'], 'signal_type': s['signal_type'],
            'buy_strength': s['buy_strength'],
            'entry_price': s['entry_price'], 'stop_price': s['stop_price'],
            'sig_date': s['sig_date'], 'total_score': s['total_score'],
            'sector_tier': s['sector_tier'], 'trend_type': s['trend_type'],
            'fwd_5d': s['fwd'].get('5d'), 'fwd_10d': s['fwd'].get('10d'),
            'fwd_20d': s['fwd'].get('20d'), 'fwd_composite': s['fwd']['composite'],
            'fwd_max_dd': s['fwd'].get('max_dd'), 'fwd_max_gain': s['fwd'].get('max_gain'),
            'fwd_sl_hit': s['fwd'].get('sl_hit'),
        } for s in validated],
    }
    os.makedirs('signals', exist_ok=True)
    out_file = f'signals/signal_quality_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f'  结果已保存: {out_file}')


if __name__ == '__main__':
    main()
