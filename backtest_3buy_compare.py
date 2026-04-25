#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中枢法 vs 回调法 3买对比回测

对指定股票分别用两种3买检测方法跑回测，对比：
- 信号数量、位置
- 胜率、平均收益、最大回撤
- 信号重叠/差异
"""
import sys, os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD
from core.buy_sell_points import BuySellPointDetector


def run_chanlun(code: str, days: int = 600):
    """运行缠论分析，返回三种买卖点列表"""
    src = HybridSource()
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = src.get_kline(code, start_date=start, period='daily')
    if df is None or len(df) < 60:
        print(f'{code}: 数据不足 ({len(df) if df is not None else 0}行)')
        return None

    kline = KLine.from_dataframe(df, strict_mode=True)
    fractals = FractalDetector(kline, confirm_required=False).get_fractals()
    strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
    if len(strokes) < 3:
        print(f'{code}: 笔不足')
        return None
    segments = SegmentGenerator(kline, strokes).get_segments()
    pivots = PivotDetector(kline, strokes).get_pivots()
    close_s = pd.Series([k.close for k in kline])
    macd = MACD(close_s)

    # 方法1：仅回调法
    det1 = BuySellPointDetector(fractals, strokes, segments, pivots, macd)
    det1._buy_points = []
    det1._sell_points = []
    det1._detect_third_buy()
    buys_pullback = [b for b in det1._buy_points if '3buy' == b.point_type]

    # 方法2：仅中枢法
    det2 = BuySellPointDetector(fractals, strokes, segments, pivots, macd)
    det2._buy_points = []
    det2._sell_points = []
    det2._detect_third_buy_pivot_based()
    buys_pivot = [b for b in det2._buy_points if '3buy' == b.point_type]

    # 价格序列（用于模拟回测）
    prices = np.array([k.close for k in kline])
    dates = [k.datetime for k in kline]

    return {
        'code': code,
        'prices': prices,
        'dates': dates,
        'kline_len': len(kline),
        'pivots': pivots,
        'buys_pullback': buys_pullback,
        'buys_pivot': buys_pivot,
    }


def simulate_trades(buys, prices, dates, hold_days=20, stop_loss_pct=0.05):
    """简单模拟：买入后持有N天或止损"""
    trades = []
    for b in buys:
        entry_idx = b.index
        entry_price = b.price
        if entry_idx >= len(prices) - 1:
            continue

        # 止损价
        sl_price = entry_price * (1 - stop_loss_pct)

        # 持有期内检查
        exit_idx = min(entry_idx + hold_days, len(prices) - 1)
        actual_exit = exit_idx
        exit_price = prices[exit_idx]
        exit_reason = 'timeout'

        for j in range(entry_idx + 1, exit_idx + 1):
            if prices[j] <= sl_price:
                actual_exit = j
                exit_price = sl_price
                exit_reason = 'stop_loss'
                break

        pnl = (exit_price - entry_price) / entry_price
        trades.append({
            'entry_date': dates[entry_idx].strftime('%Y-%m-%d') if hasattr(dates[entry_idx], 'strftime') else str(dates[entry_idx]),
            'exit_date': dates[actual_exit].strftime('%Y-%m-%d') if hasattr(dates[actual_exit], 'strftime') else str(dates[actual_exit]),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'confidence': b.confidence,
            'hold_days': actual_exit - entry_idx,
            'reason': b.reason[:80],
        })
    return trades


def print_summary(name, trades):
    if not trades:
        print(f'\n{name}: 无交易')
        return
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_pnl = np.mean(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    max_loss = min(pnls) if pnls else 0
    max_win = max(pnls) if pnls else 0

    # 累计收益
    cum = 1.0
    max_dd = 0.0
    peak = 1.0
    for p in pnls:
        cum *= (1 + p)
        if cum > peak:
            peak = cum
        dd = (peak - cum) / peak
        if dd > max_dd:
            max_dd = dd

    print(f'\n{"="*60}')
    print(f'{name}: {len(trades)}笔交易')
    print(f'{"="*60}')
    print(f'  胜率: {win_rate:.1%} ({len(wins)}/{len(pnls)})')
    print(f'  平均收益: {avg_pnl:.2%}')
    print(f'  最大单笔盈利: {max_win:.2%}')
    print(f'  最大单笔亏损: {max_loss:.2%}')
    print(f'  累计收益: {(cum-1):.2%}')
    print(f'  最大回撤: {max_dd:.2%}')
    print(f'  盈亏比: {np.mean(wins)/abs(np.mean(losses)) if losses else 0:.2f}')

    for t in trades:
        tag = 'W' if t['pnl'] > 0 else 'L'
        print(f'  [{tag}] {t["entry_date"]} @{t["entry_price"]:.2f} → '
              f'{t["exit_date"]} @{t["exit_price"]:.2f} '
              f'pnl={t["pnl"]:+.2%} conf={t["confidence"]:.2f} '
              f'({t["exit_reason"]}, {t["hold_days"]}d)')


def run_batch(n=100, seed=42):
    """批量跑随机股票，汇总统计"""
    import random
    from chanlun_unified.stock_pool import StockPoolManager

    sp = StockPoolManager()
    all_codes = sp.get_tdx_all_codes()
    random.seed(seed)
    sampled = random.sample(all_codes, min(n, len(all_codes)))

    # 转换格式：000001.SZ → sz000001
    def to_hybrid(code):
        sym, ex = code.split('.')
        prefix = 'sh' if ex == 'SH' else 'sz'
        return prefix + sym

    total_pb_trades = []
    total_pv_trades = []
    stats = {'ok': 0, 'no_data': 0, 'no_strokes': 0, 'no_pivots': 0}

    for i, code in enumerate(sampled):
        hybrid_code = to_hybrid(code)
        if i % 20 == 0:
            print(f'  [{i}/{len(sampled)}] {hybrid_code}...', flush=True)

        result = run_chanlun(hybrid_code, days=600)
        if result is None:
            stats['no_data'] += 1
            continue
        if len(result['pivots']) < 2:
            stats['no_pivots'] += 1
            continue

        stats['ok'] += 1
        trades_pb = simulate_trades(result['buys_pullback'], result['prices'], result['dates'])
        trades_pv = simulate_trades(result['buys_pivot'], result['prices'], result['dates'])
        total_pb_trades.extend(trades_pb)
        total_pv_trades.extend(trades_pv)

    # 汇总
    print(f'\n{"="*70}')
    print(f'批量回测完成: {stats}')
    print(f'{"="*70}')
    print_summary('回调法3买 (全市场)', total_pb_trades)
    print_summary('中枢法3买 (全市场)', total_pv_trades)

    # 对比表
    if total_pb_trades and total_pv_trades:
        pb_pnls = [t['pnl'] for t in total_pb_trades]
        pv_pnls = [t['pnl'] for t in total_pv_trades]
        print(f'\n{"="*70}')
        print(f'对比汇总')
        print(f'{"="*70}')
        print(f'{"指标":<16} {"回调法":>10} {"中枢法":>10} {"差异":>10}')
        print(f'{"-"*50}')
        print(f'{"交易数":<16} {len(pb_pnls):>10} {len(pv_pnls):>10} {len(pv_pnls)-len(pb_pnls):>+10}')
        pb_wr = sum(1 for p in pb_pnls if p > 0) / len(pb_pnls) if pb_pnls else 0
        pv_wr = sum(1 for p in pv_pnls if p > 0) / len(pv_pnls) if pv_pnls else 0
        print(f'{"胜率":<16} {pb_wr:>10.1%} {pv_wr:>10.1%} {pv_wr-pb_wr:>+10.1%}')
        print(f'{"平均收益":<16} {np.mean(pb_pnls):>10.2%} {np.mean(pv_pnls):>10.2%} {np.mean(pv_pnls)-np.mean(pb_pnls):>+10.2%}')
        pb_pos = [p for p in pb_pnls if p > 0]
        pv_pos = [p for p in pv_pnls if p > 0]
        pb_neg = [p for p in pb_pnls if p <= 0]
        pv_neg = [p for p in pv_pnls if p <= 0]
        print(f'{"平均盈利":<16} {np.mean(pb_pos):>10.2%} {np.mean(pv_pos):>10.2%}' if pb_pos and pv_pos else '')
        print(f'{"平均亏损":<16} {np.mean(pb_neg):>10.2%} {np.mean(pv_neg):>10.2%}' if pb_neg and pv_neg else '')
        if pb_neg and pv_neg:
            avg_w = np.mean(pb_pos) if pb_pos else 0
            avg_l = abs(np.mean(pb_neg))
            pb_pf = avg_w / avg_l if avg_l > 0 else 0
            avg_w2 = np.mean(pv_pos) if pv_pos else 0
            avg_l2 = abs(np.mean(pv_neg))
            pv_pf = avg_w2 / avg_l2 if avg_l2 > 0 else 0
            print(f'{"盈亏比":<16} {pb_pf:>10.2f} {pv_pf:>10.2f} {pv_pf-pb_pf:>+10.2f}')

        # 持仓天数
        pb_days = [t['hold_days'] for t in total_pb_trades]
        pv_days = [t['hold_days'] for t in total_pv_trades]
        print(f'{"平均持仓天数":<16} {np.mean(pb_days):>10.1f} {np.mean(pv_days):>10.1f}')

        # 止损率
        pb_sl = sum(1 for t in total_pb_trades if t['exit_reason'] == 'stop_loss') / len(total_pb_trades)
        pv_sl = sum(1 for t in total_pv_trades if t['exit_reason'] == 'stop_loss') / len(total_pv_trades)
        print(f'{"止损率":<16} {pb_sl:>10.1%} {pv_sl:>10.1%} {pv_sl-pb_sl:>+10.1%}')

    # 保存结果
    import json
    result_path = 'backtest_3buy_compare_result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pullback_trades': total_pb_trades,
            'pivot_trades': total_pv_trades,
            'stats': stats,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果已保存到 {result_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=0, help='批量回测股票数量')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.batch > 0:
        run_batch(n=args.batch, seed=args.seed)
    else:
        # 默认跑5只
        codes = ['sh601857', 'sh600519', 'sz002600', 'sz300750', 'sh601318']
        names = ['中国石油', '贵州茅台', '领益智造', '宁德时代', '中国平安']

        for code, name in zip(codes, names):
            print(f'\n{"#"*70}')
            print(f'# {code} ({name})')
            print(f'{"#"*70}')

            result = run_chanlun(code, days=600)
            if result is None:
                continue

            print(f'\n中枢: {len(result["pivots"])}')
            for i, p in enumerate(result['pivots']):
                sc = len(p.strokes) if p.strokes else '?'
                print(f'  P#{i}: ZG={p.zg:.2f} ZD={p.zd:.2f} strokes={sc}')

            trades_pb = simulate_trades(result['buys_pullback'], result['prices'], result['dates'])
            print_summary(f'{code} 回调法3买', trades_pb)

            trades_pv = simulate_trades(result['buys_pivot'], result['prices'], result['dates'])
            print_summary(f'{code} 中枢法3买', trades_pv)

            pb_idx = {b.index for b in result['buys_pullback']}
            pv_idx = {b.index for b in result['buys_pivot']}
            overlap = pb_idx & pv_idx
            only_pb = pb_idx - pv_idx
            only_pv = pv_idx - pb_idx
            print(f'\n  信号对比: 回调法={len(pb_idx)} 中枢法={len(pv_idx)} '
                  f'重叠={len(overlap)} 仅回调={len(only_pb)} 仅中枢={len(only_pv)}')


if __name__ == '__main__':
    main()
