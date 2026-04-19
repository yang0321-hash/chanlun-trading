#!/usr/bin/env python3
"""CC1买卖点引擎优化前后对比回测

对比:
- 旧引擎: 2买固定0.6, 3买无分类/无黄金分割
- 新引擎: 2买加权评分+重叠检测, 3买三强度分类+黄金分割+量价

评估指标: 信号数量分布、置信度分布、T+N胜率
"""
import sys, os, json, glob
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import pandas as pd
import numpy as np
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD


def load_stock(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    return df


def analyze_buy_points(df, code):
    """用新引擎分析买卖点，模拟旧引擎和新引擎的评分"""
    try:
        kline = KLine.from_dataframe(df)
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
        if len(strokes) < 5:
            return None
        segments = SegmentGenerator(kline, strokes).get_segments()
        pivots = PivotDetector(kline, strokes).get_pivots()
        if not pivots:
            return None
        close_s = pd.Series([k.close for k in kline])
        macd = MACD(close_s)

        det = BuySellPointDetector(fractals, strokes, segments, pivots, macd)
        buys, sells = det.detect_all()

        prices = [k.close for k in kline]
        dates = [k.datetime for k in kline]

        results = []
        for bp in buys:
            if bp.point_type not in ('1buy', '2buy', '3buy'):
                continue

            idx = bp.index
            # T+5 和 T+10 收益
            ret_5 = ret_10 = None
            if idx + 5 < len(prices):
                ret_5 = (prices[idx + 5] - bp.price) / bp.price * 100
            if idx + 10 < len(prices):
                ret_10 = (prices[idx + 10] - bp.price) / bp.price * 100

            # 新引擎特征
            has_overlap = '重叠' in bp.reason
            has_fib = '黄金分割' in bp.reason
            has_vol = '缩量' in bp.reason
            strength = 'strong' if '强三买' in bp.reason else ('weak' if '弱三买' in bp.reason else 'standard')
            new_conf = bp.confidence

            # 模拟旧引擎: 2买固定0.6, 3买无分类
            if bp.point_type == '2buy':
                old_conf = 0.6
            elif bp.point_type == '3buy':
                old_conf = 0.55  # 旧基础值
            else:
                old_conf = new_conf

            results.append({
                'code': code,
                'type': bp.point_type,
                'price': bp.price,
                'idx': idx,
                'new_conf': new_conf,
                'old_conf': old_conf,
                'ret_5': ret_5,
                'ret_10': ret_10,
                'overlap': has_overlap,
                'fib': has_fib,
                'vol': has_vol,
                'strength': strength if bp.point_type == '3buy' else '',
            })
        return results
    except Exception as e:
        return None


def main():
    # 选50只代表性股票
    files = sorted(glob.glob('test_output/sh6*.day.json') + glob.glob('test_output/sz0*.day.json'))
    # 过滤掉指数
    files = [f for f in files if not f.endswith('sh000001.day.json')]
    np.random.seed(42)
    if len(files) > 50:
        files = np.random.choice(files, 50, replace=False).tolist()

    print(f'=== CC1买卖点引擎优化对比回测 ===')
    print(f'样本: {len(files)} 只股票\n')

    all_results = []
    for i, filepath in enumerate(files):
        code = os.path.basename(filepath).replace('.day.json', '')
        try:
            df = load_stock(filepath)
            if len(df) < 100:
                continue
            r = analyze_buy_points(df, code)
            if r:
                all_results.extend(r)
        except Exception:
            continue
        if (i + 1) % 10 == 0:
            print(f'  已处理 {i+1}/{len(files)} 只...')

    if not all_results:
        print('无有效信号')
        return

    df = pd.DataFrame(all_results)
    print(f'\n总计信号: {len(df)} 条')

    # === 按买点类型对比 ===
    print('\n' + '='*70)
    print('按买点类型对比 (新引擎 vs 旧引擎置信度)')
    print('='*70)

    for pt in ['1buy', '2buy', '3buy']:
        sub = df[df['type'] == pt]
        if len(sub) == 0:
            continue

        ret5_valid = sub[sub['ret_5'].notna()]
        ret10_valid = sub[sub['ret_10'].notna()]

        print(f'\n--- {pt} ({len(sub)} 条) ---')
        print(f'  置信度: 旧={sub["old_conf"].mean():.2f} → 新={sub["new_conf"].mean():.2f} (范围 {sub["new_conf"].min():.2f}~{sub["new_conf"].max():.2f})')
        if len(ret5_valid) > 0:
            win5 = (ret5_valid['ret_5'] > 0).mean()
            avg5 = ret5_valid['ret_5'].mean()
            print(f'  T+5: 胜率={win5:.1%} 平均收益={avg5:+.2f}% (n={len(ret5_valid)})')
        if len(ret10_valid) > 0:
            win10 = (ret10_valid['ret_10'] > 0).mean()
            avg10 = ret10_valid['ret_10'].mean()
            print(f'  T+10: 胜率={win10:.1%} 平均收益={avg10:+.2f}% (n={len(ret10_valid)})')

    # === 新引擎特征分析 ===
    print('\n' + '='*70)
    print('新引擎特征效果分析')
    print('='*70)

    # 高置信度 vs 低置信度
    high_conf = df[df['new_conf'] >= 0.75]
    low_conf = df[df['new_conf'] < 0.55]

    for label, sub in [('高置信度(≥0.75)', high_conf), ('低置信度(<0.55)', low_conf)]:
        rv5 = sub[sub['ret_5'].notna()]
        if len(rv5) > 0:
            print(f'  {label}: T+5胜率={((rv5["ret_5"]>0).mean()):.1%} 收益={rv5["ret_5"].mean():+.2f}% (n={len(rv5)})')

    # 2买特征
    b2 = df[df['type'] == '2buy']
    if len(b2) > 0:
        print(f'\n  2买特征:')
        for feat, label in [('overlap', '2买3买重叠'), ('vol', '缩量确认')]:
            with_feat = b2[b2[feat] == True]
            without = b2[b2[feat] == False]
            rv_w = with_feat[with_feat['ret_5'].notna()]
            rv_wo = without[without['ret_5'].notna()]
            if len(rv_w) > 0 and len(rv_wo) > 0:
                print(f'    {label}: 有→T+5胜率={((rv_w["ret_5"]>0).mean()):.1%} 收益={rv_w["ret_5"].mean():+.2f}% (n={len(rv_w)}) | 无→{((rv_wo["ret_5"]>0).mean()):.1%} {rv_wo["ret_5"].mean():+.2f}% (n={len(rv_wo)})')

    # 3买特征
    b3 = df[df['type'] == '3buy']
    if len(b3) > 0:
        print(f'\n  3买特征:')
        # 强度分类
        for s, label in [('strong', '强三买'), ('standard', '标准三买'), ('weak', '弱三买')]:
            ss = b3[b3['strength'] == s]
            rv = ss[ss['ret_5'].notna()]
            if len(rv) > 0:
                print(f'    {label}: T+5胜率={((rv["ret_5"]>0).mean()):.1%} 收益={rv["ret_5"].mean():+.2f}% (n={len(rv)})')

        # 黄金分割
        for feat, label in [('fib', '黄金分割'), ('vol', '缩量回踩')]:
            with_feat = b3[b3[feat] == True]
            without = b3[b3[feat] == False]
            rv_w = with_feat[with_feat['ret_5'].notna()]
            rv_wo = without[without['ret_5'].notna()]
            if len(rv_w) > 0 and len(rv_wo) > 0:
                print(f'    {label}: 有→T+5胜率={((rv_w["ret_5"]>0).mean()):.1%} 收益={rv_w["ret_5"].mean():+.2f}% (n={len(rv_w)}) | 无→{((rv_wo["ret_5"]>0).mean()):.1%} {rv_wo["ret_5"].mean():+.2f}% (n={len(rv_wo)})')

    # === 置信度分位数胜率 ===
    print('\n' + '='*70)
    print('置信度分位数 vs T+5胜率')
    print('='*70)
    valid = df[df['ret_5'].notna()]
    if len(valid) > 10:
        for lo, hi, label in [(0, 0.5, '<0.50'), (0.5, 0.65, '0.50~0.65'), (0.65, 0.80, '0.65~0.80'), (0.80, 1.01, '≥0.80')]:
            sub = valid[(valid['new_conf'] >= lo) & (valid['new_conf'] < hi)]
            if len(sub) > 0:
                wr = (sub['ret_5'] > 0).mean()
                avg = sub['ret_5'].mean()
                print(f'  {label:>10}: 胜率={wr:.1%} 平均收益={avg:+.2f}% (n={len(sub)})')

    # 保存结果
    output_path = 'signals/cc1_engine_compare.json'
    summary = {
        'total_signals': len(df),
        'by_type': {},
    }
    for pt in ['1buy', '2buy', '3buy']:
        sub = df[df['type'] == pt]
        rv = sub[sub['ret_5'].notna()]
        if len(rv) > 0:
            summary['by_type'][pt] = {
                'count': int(len(sub)),
                'old_conf_avg': float(sub['old_conf'].mean()),
                'new_conf_avg': float(sub['new_conf'].mean()),
                't5_win_rate': float((rv['ret_5'] > 0).mean()),
                't5_avg_ret': float(rv['ret_5'].mean()),
            }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {output_path}')


if __name__ == '__main__':
    main()
