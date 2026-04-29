#!/usr/bin/env python3
"""对比回测: quasi3buy置信度 + 3买止损参数

方案A (规范文档): quasi3buy=0.35, 止损=ZG*0.99
方案B (优化版):   quasi3buy=0.50, 止损=ZG*0.98

评估指标: 信号数、胜率(T+5/T+10)、平均收益、盈亏比
"""
import sys, os, json, glob
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import pandas as pd
import numpy as np
from datetime import datetime


def analyze_buy_points(df, code, quasi3_base, stop_mult):
    """用指定参数分析买卖点"""
    from core.kline import KLine
    from core.fractal import FractalDetector
    from core.stroke import StrokeGenerator
    from core.segment import SegmentGenerator
    from core.pivot import PivotDetector
    from core.buy_sell_points import BuySellPointDetector
    from indicator.macd import MACD

    try:
        kline = KLine.from_dataframe(df)
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
        if len(strokes) < 5:
            return []
        segments = SegmentGenerator(kline, strokes).get_segments()
        pivots = PivotDetector(kline, strokes).get_pivots()
        if not pivots:
            return []
        close_s = pd.Series([k.close for k in kline])
        macd = MACD(close_s)

        # 临时修改参数
        import core.buy_sell_points as bsp_module

        # 保存原始值
        orig_quasi3_code = None

        det = BuySellPointDetector(fractals, strokes, segments, pivots, macd)
        buys, sells = det.detect_all()

        prices = [k.close for k in kline]
        n = len(prices)

        results = []
        for bp in buys:
            if bp.point_type not in ('2buy', '3buy', 'quasi3buy'):
                continue

            idx = bp.index
            if idx < 0 or idx >= n:
                continue

            entry = prices[idx]
            stop = bp.stop_loss

            # 应用自定义止损
            if bp.point_type == '3buy' and bp.related_pivot:
                zg = bp.related_pivot.zg
                stop = zg * stop_mult

            # 应用自定义quasi3buy置信度
            conf = bp.confidence
            if bp.point_type == 'quasi3buy':
                # 重新计算: 用自定义base替换
                margin = (bp.price - stop) / stop if stop > 0 else 0
                # 保留其他加分，只改base差值
                delta = quasi3_base - 0.35  # 原base是0.35
                conf = max(0.1, min(1.0, bp.confidence + delta))

            # T+N收益
            t5 = t10 = t20 = None
            max_gain5 = max_loss5 = 0
            max_gain10 = max_loss10 = 0
            for t in range(1, 21):
                if idx + t >= n:
                    break
                pct = (prices[idx + t] - entry) / entry
                if t <= 5:
                    max_gain5 = max(max_gain5, pct)
                    max_loss5 = min(max_loss5, pct)
                if t <= 10:
                    max_gain10 = max(max_gain10, pct)
                    max_loss10 = min(max_loss10, pct)
                if t == 5:
                    t5 = pct
                if t == 10:
                    t10 = pct
                if t == 20:
                    t20 = pct

            # 检查是否触发止损
            stopped_out = False
            for t in range(1, min(21, n - idx)):
                if prices[idx + t] <= stop:
                    stopped_out = True
                    break

            results.append({
                'code': code,
                'type': bp.point_type,
                'confidence': conf,
                'entry': entry,
                'stop': stop,
                't5': t5,
                't10': t10,
                't20': t20,
                'max_gain5': max_gain5,
                'max_loss5': max_loss5,
                'max_gain10': max_gain10,
                'max_loss10': max_loss10,
                'stopped': stopped_out,
                'reason': bp.reason[:80] if bp.reason else '',
            })
        return results
    except Exception:
        return []


def run_comparison(data_dir='test_output'):
    """运行对比回测"""
    # 只用本地已解析的个股数据 (排除指数和北交所)
    import re
    all_files = glob.glob(f'{data_dir}/*.day.json')
    files = [f for f in all_files if re.match(r'(sh6|sz0|sz3)', os.path.basename(f))]
    print(f'找到 {len(files)} 只个股数据')
    # 随机抽50只加速测试
    if len(files) > 50:
        import random
        random.seed(42)
        files = random.sample(files, 50)
        print(f'随机抽取 {len(files)} 只进行对比')

    print(f'加载 {len(files)} 只股票')

    # 方案A: 规范文档参数
    print('\n=== 方案A: quasi3buy=0.35, 止损=ZG*0.99 ===')
    results_a = []
    for i, f in enumerate(files):
        code = os.path.basename(f).replace('.day.json', '')
        try:
            df = pd.read_json(f)
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'datetime'}).set_index('datetime')
            if len(df) >= 120:
                r = analyze_buy_points(df, code, quasi3_base=0.35, stop_mult=0.99)
                results_a.extend(r)
                if r:
                    print(f'  [{i+1}/{len(files)}] {code}: {len(r)}个买点')
        except Exception as e:
            print(f'  [{i+1}/{len(files)}] {code}: ERROR {e}')

    # 方案B: 优化参数
    print(f'\n=== 方案B: quasi3buy=0.50, 止损=ZG*0.98 ===')
    results_b = []
    for i, f in enumerate(files):
        code = os.path.basename(f).replace('.day.json', '')
        try:
            df = pd.read_json(f)
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'datetime'}).set_index('datetime')
            if len(df) >= 120:
                r = analyze_buy_points(df, code, quasi3_base=0.50, stop_mult=0.98)
                results_b.extend(r)
                if r:
                    print(f'  [{i+1}/{len(files)}] {code}: {len(r)}个买点')
        except Exception as e:
            print(f'  [{i+1}/{len(files)}] {code}: ERROR {e}')

    # 统计对比
    print(f'\n{"="*70}')
    print(f'{"指标":<25} {"方案A(0.35/0.99)":>20} {"方案B(0.50/0.98)":>20}')
    print(f'{"="*70}')

    for label, results in [('方案A', results_a), ('方案B', results_b)]:
        total = len(results)
        by_type = {}
        for r in results:
            t = r['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)

        if label == '方案A':
            continue  # 先收集，下面打印

    # 打印总体对比
    for metric_name, metric_fn in [
        ('总信号数', lambda rs: str(len(rs))),
        ('3买信号数', lambda rs: str(sum(1 for r in rs if r['type'] == '3buy'))),
        ('quasi3买信号数', lambda rs: str(sum(1 for r in rs if r['type'] == 'quasi3buy'))),
        ('2买信号数', lambda rs: str(sum(1 for r in rs if r['type'] == '2buy'))),
        ('止损触发率', lambda rs: f'{sum(1 for r in rs if r["stopped"])/max(len(rs),1)*100:.1f}%'),
        ('T+5胜率', lambda rs: f'{sum(1 for r in rs if r["t5"] and r["t5"]>0)/max(sum(1 for r in rs if r["t5"] is not None),1)*100:.1f}%'),
        ('T+10胜率', lambda rs: f'{sum(1 for r in rs if r["t10"] and r["t10"]>0)/max(sum(1 for r in rs if r["t10"] is not None),1)*100:.1f}%'),
        ('T+5平均收益', lambda rs: f'{np.mean([r["t5"] for r in rs if r["t5"] is not None])*100:.2f}%' if any(r["t5"] for r in rs) else 'N/A'),
        ('T+10平均收益', lambda rs: f'{np.mean([r["t10"] for r in rs if r["t10"] is not None])*100:.2f}%' if any(r["t10"] for r in rs) else 'N/A'),
        ('T+5最大均盈利', lambda rs: f'{np.mean([r["max_gain5"] for r in rs if r["max_gain5"]])*100:.2f}%' if rs else 'N/A'),
        ('T+5最大均亏损', lambda rs: f'{np.mean([r["max_loss5"] for r in rs if r["max_loss5"]])*100:.2f}%' if rs else 'N/A'),
    ]:
        va = metric_fn(results_a) if results_a else 'N/A'
        vb = metric_fn(results_b) if results_b else 'N/A'
        print(f'{metric_name:<25} {va:>20} {vb:>20}')

    # 按买点类型细拆
    for ptype in ['2buy', '3buy', 'quasi3buy']:
        ra = [r for r in results_a if r['type'] == ptype]
        rb = [r for r in results_b if r['type'] == ptype]
        if not ra and not rb:
            continue

        print(f'\n--- {ptype} 细分 ---')
        for metric_name, metric_fn in [
            ('信号数', lambda rs: str(len(rs))),
            ('止损触发率', lambda rs: f'{sum(1 for r in rs if r["stopped"])/max(len(rs),1)*100:.1f}%'),
            ('T+5胜率', lambda rs: f'{sum(1 for r in rs if r["t5"] and r["t5"]>0)/max(sum(1 for r in rs if r["t5"] is not None),1)*100:.1f}%'),
            ('T+10胜率', lambda rs: f'{sum(1 for r in rs if r["t10"] and r["t10"]>0)/max(sum(1 for r in rs if r["t10"] is not None),1)*100:.1f}%'),
            ('T+5平均收益', lambda rs: f'{np.mean([r["t5"] for r in rs if r["t5"] is not None])*100:.2f}%' if any(r["t5"] is not None for r in rs) else 'N/A'),
            ('T+10平均收益', lambda rs: f'{np.mean([r["t10"] for r in rs if r["t10"] is not None])*100:.2f}%' if any(r["t10"] is not None for r in rs) else 'N/A'),
        ]:
            va = metric_fn(ra) if ra else 'N/A'
            vb = metric_fn(rb) if rb else 'N/A'
            print(f'  {metric_name:<23} {va:>20} {vb:>20}')

    print(f'\n{"="*70}')

    # 结论
    def calc_score(results):
        if not results:
            return 0
        wins = sum(1 for r in results if r.get('t5') and r['t5'] > 0)
        total = sum(1 for r in results if r.get('t5') is not None)
        wr = wins / max(total, 1)
        avg_ret = np.mean([r['t5'] for r in results if r.get('t5') is not None]) if any(r.get('t5') is not None for r in results) else 0
        so_rate = sum(1 for r in results if r['stopped']) / max(len(results), 1)
        return wr * 100 + avg_ret * 500 - so_rate * 50

    sa = calc_score(results_a)
    sb = calc_score(results_b)
    winner = 'A' if sa > sb else 'B'
    print(f'\n综合评分: A={sa:.1f} B={sb:.1f} → 方案{winner}更优')

    return results_a, results_b


if __name__ == '__main__':
    run_comparison()
