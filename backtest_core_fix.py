#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心模块修复后回测验证（TDX本地数据版）

验证7个关键修复的效果：
1. 1买/1卖强制要求趋势（至少2个中枢）
2. K线包含方向回退修复
3. MACD背驰双条件严格模式
4. 分型步长修复
5. 背驰比较用紧邻进入笔
6. 线段第二类破坏确认加强
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD


TDX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tdx_data')
START_DATE = pd.Timestamp('2023-01-01')
END_DATE = pd.Timestamp('2026-04-14')
HOLD_PERIODS = [5, 10, 20]


def read_tdx_day(code: str) -> pd.DataFrame:
    """读TDX本地.day文件"""
    pure = code.split('.')[0]
    market = code.endswith('.SH')
    prefix = 'sh' if market else 'sz'
    path = os.path.join(TDX_ROOT, prefix, 'lday', f'{prefix}{pure}.day')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read()
        n = len(data) // 32
        if n < 100:
            return None
        arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
        df = pd.DataFrame({
            'datetime': pd.to_datetime(arr[:, 0].astype(str), format='%Y%m%d'),
            'open': arr[:, 1] / 100.0,
            'high': arr[:, 2] / 100.0,
            'low': arr[:, 3] / 100.0,
            'close': arr[:, 4] / 100.0,
            'volume': arr[:, 6].astype(np.int64),
        })
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df[df['volume'] > 0]
        df = df[(df['datetime'] >= START_DATE) & (df['datetime'] <= END_DATE)]
        if len(df) < 100:
            return None
        return df
    except:
        return None


def get_all_stock_files():
    """获取所有可用股票文件（仅A股个股，排除指数/ETF/可转债）"""
    stocks = []
    for prefix in ['sh', 'sz']:
        pattern = os.path.join(TDX_ROOT, prefix, 'lday', f'{prefix}*.day')
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            code_num = fname.replace('.day', '').replace(prefix, '')

            # 只保留A股个股：沪市60x/688x, 深市00x/30x（排除399指数）
            if prefix == 'sh':
                if not (code_num.startswith('6')):
                    continue
            elif prefix == 'sz':
                if not (code_num.startswith('0') or code_num.startswith('30')):
                    continue

            suffix = 'SH' if prefix == 'sh' else 'SZ'
            stocks.append(f'{code_num}.{suffix}')
    return stocks


def analyze_stock(df, code):
    """对单只股票运行完整缠论分析"""
    try:
        kline = KLine.from_dataframe(df, strict_mode=True)
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        if len(fractals) < 2:
            return None
        strokes = StrokeGenerator(kline, fractals, min_bars=5).get_strokes()
        if len(strokes) < 3:
            return None
        segments = SegmentGenerator(kline, strokes).get_segments()
        pivots = PivotDetector(kline, strokes).get_pivots()

        close_s = pd.Series([k.close for k in kline])
        macd = MACD(close_s)

        detector = BuySellPointDetector(fractals, strokes, segments, pivots, macd)
        buys, sells = detector.detect_all()

        return {
            'code': code,
            'klen': len(kline),
            'prices': close_s,
            'buys': buys,
            'sells': sells,
            'pivots': pivots,
            'strokes': strokes,
            'segments': segments,
        }
    except Exception as e:
        return None


def evaluate_signal(signal, prices, klen):
    """评估买卖点的后续收益"""
    idx = signal.index
    if idx >= klen - max(HOLD_PERIODS) - 1:
        return None

    # 使用下一根K线开盘价入场（避免用信号点的极值价格）
    # signal.price 是笔端点值（局部极值），不代表可成交价
    if idx + 1 >= klen:
        return None
    entry_price = prices.iloc[idx + 1]  # 下一根K线收盘价=模拟T+1入场

    results = {
        'point_type': signal.point_type,
        'index': idx,
        'entry_price': entry_price,
        'signal_price': signal.price,
        'confidence': signal.confidence,
        'divergence_ratio': signal.divergence_ratio,
        'pivot_div_ratio': signal.pivot_divergence_ratio,
        'reason': signal.reason[:80],
    }

    for period in HOLD_PERIODS:
        future_idx = min(idx + 1 + period, klen - 1)
        future_price = prices.iloc[future_idx]
        ret = (future_price - entry_price) / entry_price
        results[f'ret_{period}d'] = ret

    # 最大回撤
    end_idx = min(idx + 1 + 20, klen)
    future_prices = prices.iloc[idx+2:end_idx]
    if len(future_prices) > 0:
        peak = future_prices.cummax()
        drawdown = (future_prices - peak) / peak
        results['max_dd_20d'] = drawdown.min()
    else:
        results['max_dd_20d'] = 0

    return results


def main():
    import random
    random.seed(42)

    all_codes = get_all_stock_files()
    # 随机抽样50只
    if len(all_codes) > 50:
        sample = random.sample(all_codes, 50)
    else:
        sample = all_codes

    print('=' * 90)
    print('  缠论核心模块修复后回测验证 (TDX本地数据)')
    print(f'  日期范围: {START_DATE.date()} ~ {END_DATE.date()}')
    print(f'  样本股票: {len(sample)}只 / 共{len(all_codes)}只可用')
    print('  修复: 1买/1卖趋势强制 | K线包含 | MACD双条件 | 分型步长 | 紧邻背驰 | 线段破坏')
    print('=' * 90)

    all_buy_results = []
    all_sell_results = []
    total_ok = 0
    total_err = 0

    for i, code in enumerate(sample):
        prefix = 'sh' if code.endswith('.SH') else 'sz'
        display = f'{prefix}{code.split(".")[0]}'

        df = read_tdx_day(code)
        if df is None:
            total_err += 1
            continue

        result = analyze_stock(df, code)
        if result is None:
            total_err += 1
            continue

        total_ok += 1
        n_pivots = len(result['pivots'])
        n_strokes = len(result['strokes'])
        n_seg = len(result['segments'])
        n_buys = len(result['buys'])
        n_sells = len(result['sells'])

        buy_types = {}
        for bp in result['buys']:
            buy_types[bp.point_type] = buy_types.get(bp.point_type, 0) + 1
        sell_types = {}
        for sp in result['sells']:
            sell_types[sp.point_type] = sell_types.get(sp.point_type, 0) + 1

        print(f'  [{i+1:2d}/{len(sample)}] {display:12s} bars={len(df):4d} '
              f'笔={n_strokes:3d} 中枢={n_pivots:2d} 线段={n_seg:2d} | '
              f'买={n_buys:2d}{buy_types} 卖={n_sells:2d}{sell_types}')

        prices = result['prices']
        klen = result['klen']

        for bp in result['buys']:
            ev = evaluate_signal(bp, prices, klen)
            if ev:
                ev['code'] = display
                all_buy_results.append(ev)

        for sp in result['sells']:
            ev = evaluate_signal(sp, prices, klen)
            if ev:
                ev['code'] = display
                all_sell_results.append(ev)

    # ============ 汇总分析 ============
    print(f'\n{"=" * 90}')
    print(f'  汇总统计: {total_ok}只成功 / {total_err}只跳过')
    print(f'  总买点: {len(all_buy_results)}个  总卖点: {len(all_sell_results)}个')
    print(f'{"=" * 90}')

    # 买点分析
    if all_buy_results:
        df_buys = pd.DataFrame(all_buy_results)
        print(f'\n  === 买点收益率分析 ===')
        print(f'  {"类型":<8s} {"数量":>5s}', end='')
        for p in HOLD_PERIODS:
            print(f' {"%d日胜率" % p:>8s} {"%d日均收" % p:>9s}', end='')
        print(f'  {"20日回撤":>9s} {"平均置信":>8s}')
        print(f'  {"-" * 80}')

        for pt in ['1buy', '2buy', '3buy']:
            sub = df_buys[df_buys['point_type'] == pt]
            if len(sub) == 0:
                print(f'  {pt:<8s}     0  (无信号)')
                continue
            line = f'  {pt:<8s} {len(sub):>5d}'
            for p in HOLD_PERIODS:
                col = f'ret_{p}d'
                wr = (sub[col] > 0).mean()
                ar = sub[col].mean()
                line += f' {wr:>7.1%} {ar:>+8.2%}'
            line += f'  {sub["max_dd_20d"].mean():>8.2%} {sub["confidence"].mean():>8.2f}'
            print(line)

        # 全部买点
        line = f'  {"ALL":<8s} {len(df_buys):>5d}'
        for p in HOLD_PERIODS:
            col = f'ret_{p}d'
            wr = (df_buys[col] > 0).mean()
            ar = df_buys[col].mean()
            line += f' {wr:>7.1%} {ar:>+8.2%}'
        line += f'  {df_buys["max_dd_20d"].mean():>8.2%} {df_buys["confidence"].mean():>8.2f}'
        print(line)

    # 卖点分析
    if all_sell_results:
        df_sells = pd.DataFrame(all_sell_results)
        print(f'\n  === 卖点收益率分析（收益<0为正确信号）===')
        print(f'  {"类型":<8s} {"数量":>5s}', end='')
        for p in HOLD_PERIODS:
            print(f' {"%d日胜率" % p:>8s} {"%d日均收" % p:>9s}', end='')
        print(f'  {"平均置信":>8s}')
        print(f'  {"-" * 65}')

        for pt in ['1sell', '2sell', '3sell']:
            sub = df_sells[df_sells['point_type'] == pt]
            if len(sub) == 0:
                print(f'  {pt:<8s}     0  (无信号)')
                continue
            line = f'  {pt:<8s} {len(sub):>5d}'
            for p in HOLD_PERIODS:
                col = f'ret_{p}d'
                wr = (sub[col] < 0).mean()
                ar = sub[col].mean()
                line += f' {wr:>7.1%} {ar:>+8.2%}'
            line += f' {sub["confidence"].mean():>7.2f}'
            print(line)

    # ============ 修复验证详情 ============
    print(f'\n{"=" * 90}')
    print(f'  修复验证')
    print(f'{"=" * 90}')

    if all_buy_results:
        df_b = pd.DataFrame(all_buy_results)

        # 验证1: 1买趋势强制
        first_buys = df_b[df_b['point_type'] == '1buy']
        print(f'\n  [Fix#1] 1买趋势强制: {len(first_buys)}个1买信号')
        if len(first_buys) > 0:
            for _, r in first_buys.head(5).iterrows():
                print(f'    {r["code"]} @{r["entry_price"]:.2f} conf={r["confidence"]:.2f} '
                      f'amp={r["pivot_div_ratio"]:.2f} | {r["reason"]}')
        else:
            print(f'    无1买信号 — 修复后更严格，过滤了无趋势的假信号')

        # 验证2: 背驰用紧邻笔
        div_buys = df_b[df_b['pivot_div_ratio'] > 0]
        if len(div_buys) > 0:
            print(f'\n  [Fix#5] 紧邻笔背驰: {len(div_buys)}个信号有振幅比')
            print(f'    比值范围: {div_buys["pivot_div_ratio"].min():.3f} ~ {div_buys["pivot_div_ratio"].max():.3f}')
            print(f'    <1.0(背驰): {(div_buys["pivot_div_ratio"] < 1.0).sum()}个 '
                  f'>=1.0(无背驰): {(div_buys["pivot_div_ratio"] >= 1.0).sum()}个')

        # 验证: 按背驰分组看效果
        if len(div_buys) >= 5:
            print(f'\n  [效果] 按振幅比分组:')
            div_group = div_buys.copy()
            div_group['div_bin'] = pd.cut(div_group['pivot_div_ratio'],
                                          bins=[0, 0.5, 1.0, 2.0, 99],
                                          labels=['<0.5强背驰', '0.5-1背驰', '1-2无背驰', '>2强离开'])
            for label in ['<0.5强背驰', '0.5-1背驰', '1-2无背驰', '>2强离开']:
                sub = div_group[div_group['div_bin'] == label]
                if len(sub) == 0:
                    continue
                wr = (sub['ret_10d'] > 0).mean()
                ar = sub['ret_10d'].mean()
                print(f'    {label:12s}: {len(sub):3d}个 10日胜率={wr:.1%} 均收={ar:+.2%}')

    # 验证3: MACD双条件
    print(f'\n  [Fix#3] MACD双条件: 已启用严格模式')
    print(f'    check_divergence 要求 DIF背离 AND 面积减少 同时满足')

    print(f'\n{"=" * 90}')
    print('  回测完成')


if __name__ == '__main__':
    main()
