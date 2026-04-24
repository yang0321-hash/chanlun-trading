#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论系统 Walk-Forward 验证

方法：全段分析 + 按信号时间戳切分（而非按窗口分别分析）
1. 对每只股票用全部历史运行完整缠论分析，收集所有买卖点
2. 按信号发生时间戳分配到不同时间段
3. 比较各时间段信号质量 → 检验稳定性
4. Walk-Forward：训练期信号 vs 测试期信号的质量衰减

数据：TDX本地日线（纯A股，排除ETF/可转债/指数）
"""
import sys, os, random, glob, time
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
END_DATE = pd.Timestamp('2026-04-14')
HOLD_PERIODS = [5, 10, 20]


# ======================== 数据读取 ========================

def read_tdx_day(code: str) -> pd.DataFrame:
    """读TDX日线 .day 文件（全部历史）"""
    pure = code.split('.')[0]
    is_sh = code.endswith('.SH')
    prefix = 'sh' if is_sh else 'sz'
    path = os.path.join(TDX_ROOT, prefix, 'lday', f'{prefix}{pure}.day')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read()
        n = len(data) // 32
        if n < 200:
            return None
        import struct
        import numpy as np
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
        if len(df) < 200:
            return None
        return df
    except:
        return None


# ======================== 股票筛选 ========================

def is_pure_a_share(code: str) -> bool:
    """判断是否为纯A股"""
    num = code.split('.')[0]
    mkt = code.split('.')[1]
    if num.startswith(('127', '123', '113', '110')):
        return False
    if num.startswith(('159', '161', '160', '510', '511', '512', '501', '502')):
        return False
    if mkt == 'SH' and (num.startswith('000') or num.startswith('9')):
        return False
    if mkt == 'SZ' and num.startswith('399'):
        return False
    if num.startswith('8') or num.startswith('4'):
        return False
    if mkt == 'SH' and num[:3] in ('600', '601', '603', '605'):
        return True
    if mkt == 'SZ' and num[:3] in ('000', '002', '300'):
        return True
    return False


def get_pure_a_shares() -> list:
    """获取所有纯A股代码"""
    stocks = []
    for prefix in ['sh', 'sz']:
        pattern = os.path.join(TDX_ROOT, prefix, 'lday', f'{prefix}*.day')
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            code_num = fname.replace('.day', '').replace(prefix, '')
            suffix = 'SH' if prefix == 'sh' else 'SZ'
            code = f'{code_num}.{suffix}'
            if is_pure_a_share(code):
                stocks.append(code)
    return sorted(stocks)


# ======================== 分析 ========================

def analyze_full(df, code):
    """对一只股票做完整缠论分析，返回所有买卖点及其时间戳"""
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

        # 为每个信号附加时间戳
        buy_results = []
        for bp in buys:
            if bp.index >= len(df):
                continue
            ts = df['datetime'].iloc[bp.index]
            buy_results.append({
                'timestamp': ts,
                'point_type': bp.point_type,
                'index': bp.index,
                'price': bp.price,
                'confidence': bp.confidence,
                'reason': bp.reason[:120],
            })

        sell_results = []
        for sp in sells:
            if sp.index >= len(df):
                continue
            ts = df['datetime'].iloc[sp.index]
            sell_results.append({
                'timestamp': ts,
                'point_type': sp.point_type,
                'index': sp.index,
                'price': sp.price,
                'confidence': sp.confidence,
                'reason': sp.reason[:120],
            })

        return {
            'buys': buy_results,
            'sells': sell_results,
            'prices': close_s,
            'klen': len(kline),
            'df': df,
        }
    except:
        return None


def evaluate_at(signal_info, prices, klen, df):
    """评估单个信号的后续收益"""
    idx = signal_info['index']
    if idx >= klen - max(HOLD_PERIODS) - 1:
        return None
    # 用下一根K线收盘价入场（模拟T+1），不用笔端点极值
    if idx + 1 >= klen:
        return None
    entry_price = prices.iloc[idx + 1]
    result = dict(signal_info)
    result['entry_price'] = entry_price
    for p in HOLD_PERIODS:
        fi = min(idx + 1 + p, klen - 1)
        result[f'ret_{p}d'] = (prices.iloc[fi] - entry_price) / entry_price
    end_idx = min(idx + 1 + 20, klen)
    fp = prices.iloc[idx+2:end_idx]
    if len(fp) > 0:
        result['max_dd'] = ((fp - fp.cummax()) / fp.cummax()).min()
    else:
        result['max_dd'] = 0
    return result


# ======================== 统计工具 ========================

def calc_metrics(results):
    """计算一组信号的统计指标"""
    if not results:
        return {'count': 0, 'wr_5d': None, 'wr_10d': None, 'wr_20d': None,
                'ret_5d': None, 'ret_10d': None, 'ret_20d': None, 'dd': None}
    df = pd.DataFrame(results)
    return {
        'count': len(df),
        'wr_5d': (df['ret_5d'] > 0).mean(),
        'wr_10d': (df['ret_10d'] > 0).mean(),
        'wr_20d': (df['ret_20d'] > 0).mean(),
        'ret_5d': df['ret_5d'].mean(),
        'ret_10d': df['ret_10d'].mean(),
        'ret_20d': df['ret_20d'].mean(),
        'dd': df['max_dd'].mean(),
    }


def fmt_pct(v):
    """格式化百分比，None显示为N/A"""
    if v is None:
        return '  N/A '
    return f'{v:6.1%}'


def fmt_ret(v):
    if v is None:
        return '   N/A  '
    return f'{v:+7.2%}'


# ======================== 主流程 ========================

def main():
    random.seed(42)
    t0 = time.time()

    # 获取纯A股
    all_codes = get_pure_a_shares()
    print(f'pure A-shares: {len(all_codes)}')

    sample_size = min(200, len(all_codes))
    sample = random.sample(all_codes, sample_size)

    print('=' * 100)
    print('  ChanLun Walk-Forward Validation')
    print(f'  Sample: {sample_size} pure A-shares | Eval range: 2023-01 ~ 2026-04')
    print(f'  Method: Full-period analysis, split signals by timestamp')
    print('=' * 100)

    # ==================== Step 1: 全量分析 ====================
    print(f'\n  Step 1: Running full analysis on {sample_size} stocks...')

    all_evaluated = []  # 所有评估后的买点
    all_sell_evaluated = []  # 所有评估后的卖点
    stock_ok = 0

    for i, code in enumerate(sample):
        prefix = 'sh' if code.endswith('.SH') else 'sz'
        display = f'{prefix}{code.split(".")[0]}'

        df = read_tdx_day(code)
        if df is None:
            continue

        result = analyze_full(df, code)
        if result is None:
            continue

        stock_ok += 1
        prices = result['prices']
        klen = result['klen']

        for bp in result['buys']:
            # 只评估2023年以后的信号
            if bp['timestamp'] < pd.Timestamp('2023-01-01'):
                continue
            # 不评估最后20天的信号（无法计算20日收益）
            if bp['timestamp'] > pd.Timestamp('2026-03-15'):
                continue
            ev = evaluate_at(bp, prices, klen, df)
            if ev:
                ev['code'] = display
                all_evaluated.append(ev)

        for sp in result['sells']:
            if sp['timestamp'] < pd.Timestamp('2023-01-01'):
                continue
            if sp['timestamp'] > pd.Timestamp('2026-03-15'):
                continue
            ev = evaluate_at(sp, prices, klen, df)
            if ev:
                ev['code'] = display
                all_sell_evaluated.append(ev)

        if (i + 1) % 50 == 0:
            print(f'    [{i+1}/{sample_size}] {stock_ok} OK, '
                  f'{len(all_evaluated)} buys, {len(all_sell_evaluated)} sells')

    print(f'  Done: {stock_ok} stocks, {len(all_evaluated)} buys, {len(all_sell_evaluated)} sells')

    if not all_evaluated:
        print('  ERROR: No signals generated!')
        return

    df_all = pd.DataFrame(all_evaluated)

    # ==================== Step 2: 按时间段分割 ====================
    # 用信号时间戳分配到各时间段
    periods = [
        ('2023-H1', '2023-01-01', '2023-06-30'),
        ('2023-H2', '2023-07-01', '2023-12-31'),
        ('2024-H1', '2024-01-01', '2024-06-30'),
        ('2024-H2', '2024-07-01', '2024-12-31'),
        ('2025-H1', '2025-01-01', '2025-06-30'),
        ('2025-H2', '2025-07-01', '2025-12-31'),
        ('2026-Q1', '2026-01-01', '2026-04-14'),
    ]

    # 分配信号到各时段
    period_signals = {}
    for label, start, end in periods:
        mask = (df_all['timestamp'] >= pd.Timestamp(start)) & \
               (df_all['timestamp'] <= pd.Timestamp(end))
        period_signals[label] = df_all[mask].to_dict('records')

    # ==================== Step 3: 固定周期分析 ====================
    print(f'\n{"=" * 100}')
    print(f'  Part A: Buy Signal Quality by Period (Daily)')
    print(f'{"=" * 100}')
    print(f'  {"Period":10s} | {"Count":>5s} | {"5d WR":>6s} | {"10d WR":>6s} | {"20d WR":>6s} | '
          f'{"5d Ret":>8s} | {"10d Ret":>8s} | {"20d Ret":>8s} | {"20d DD":>7s}')
    print(f'  {"-" * 95}')

    period_metrics = []
    for label, _, _ in periods:
        m = calc_metrics(period_signals[label])
        m['label'] = label
        period_metrics.append(m)
        print(f'  {label:10s} | {m["count"]:5d} | {fmt_pct(m["wr_5d"])} | {fmt_pct(m["wr_10d"])} | '
              f'{fmt_pct(m["wr_20d"])} | {fmt_ret(m["ret_5d"])} | {fmt_ret(m["ret_10d"])} | '
              f'{fmt_ret(m["ret_20d"])} | {fmt_pct(m["dd"])}')

    # 全段汇总
    m_all = calc_metrics(df_all.to_dict('records'))
    print(f'  {"ALL":10s} | {m_all["count"]:5d} | {fmt_pct(m_all["wr_5d"])} | {fmt_pct(m_all["wr_10d"])} | '
          f'{fmt_pct(m_all["wr_20d"])} | {fmt_ret(m_all["ret_5d"])} | {fmt_ret(m_all["ret_10d"])} | '
          f'{fmt_ret(m_all["ret_20d"])} | {fmt_pct(m_all["dd"])}')

    # 稳定性
    wrs_10d = [m['wr_10d'] for m in period_metrics if m['wr_10d'] is not None]
    rets_10d = [m['ret_10d'] for m in period_metrics if m['ret_10d'] is not None]
    print(f'\n  Stability:')
    print(f'    10d WR: mean={np.mean(wrs_10d):.1%}  std={np.std(wrs_10d):.1%}  '
          f'range=[{min(wrs_10d):.1%}, {max(wrs_10d):.1%}]')
    print(f'    10d Ret: mean={np.mean(rets_10d):+.2%}  std={np.std(rets_10d):+.2%}  '
          f'range=[{min(rets_10d):+.2%}, {max(rets_10d):+.2%}]')

    # ==================== Step 4: 按买点类型 x 时间段 ====================
    print(f'\n{"=" * 100}')
    print(f'  Part B: Signal Quality by Type x Period')
    print(f'{"=" * 100}')

    for pt in ['1buy', '2buy', '3buy']:
        sub = df_all[df_all['point_type'] == pt]
        if len(sub) == 0:
            continue
        print(f'\n  {pt} (total {len(sub)}):')
        print(f'    {"Period":10s} | {"Count":>5s} | {"10d WR":>6s} | {"10d Ret":>8s} | {"20d DD":>7s}')
        print(f'    {"-" * 45}')
        for label, _, _ in periods:
            mask = sub['timestamp'].between(
                pd.Timestamp(periods[[l[0] for l in periods].index(label)][1]),
                pd.Timestamp(periods[[l[0] for l in periods].index(label)][2])
            ) if False else None
            # Simpler approach
            ps = period_signals[label]
            ps_pt = [s for s in ps if s['point_type'] == pt]
            m = calc_metrics(ps_pt)
            if m['count'] > 0:
                print(f'    {label:10s} | {m["count"]:5d} | {fmt_pct(m["wr_10d"])} | '
                      f'{fmt_ret(m["ret_10d"])} | {fmt_pct(m["dd"])}')

        # 全段
        wr = (sub['ret_10d'] > 0).mean()
        ar = sub['ret_10d'].mean()
        print(f'    {"ALL":10s} | {len(sub):5d} | {wr:5.1%} | {ar:+7.2%} |')

    # ==================== Step 5: Walk-Forward (训练 vs 测试) ====================
    print(f'\n{"=" * 100}')
    print(f'  Part C: Walk-Forward (Train vs Test)')
    print(f'{"=" * 100}')

    wf_windows = [
        # (label, train_period_label, test_period_label)
        ('WF-1', ['2023-H1', '2023-H2'], ['2024-H1']),
        ('WF-2', ['2023-H2', '2024-H1'], ['2024-H2']),
        ('WF-3', ['2024-H1', '2024-H2'], ['2025-H1']),
        ('WF-4', ['2024-H2', '2025-H1'], ['2025-H2']),
        ('WF-5', ['2025-H1', '2025-H2'], ['2026-Q1']),
    ]

    print(f'  {"Window":6s} | {"Train":20s} | {"Test":10s} | '
          f'{"Trn#":>4s} | {"Tst#":>4s} | '
          f'{"Trn WR":>6s} | {"Tst WR":>6s} | {"Decay":>7s} | '
          f'{"Trn Ret":>8s} | {"Tst Ret":>8s}')
    print(f'  {"-" * 100}')

    wf_results = []
    for wf_label, train_labels, test_label in wf_windows:
        train_signals = []
        for tl in train_labels:
            train_signals.extend(period_signals.get(tl, []))
        test_signals = period_signals.get(test_label[0] if isinstance(test_label, list) else test_label, [])

        tm = calc_metrics(train_signals)
        em = calc_metrics(test_signals)

        decay = None
        if tm['wr_10d'] is not None and em['wr_10d'] is not None and tm['wr_10d'] > 0:
            decay = (tm['wr_10d'] - em['wr_10d']) / tm['wr_10d'] * 100

        train_str = '+'.join(train_labels)
        test_str = test_label if isinstance(test_label, str) else test_label[0]

        decay_str = f'{decay:+6.1f}%' if decay is not None else '   N/A '
        print(f'  {wf_label:6s} | {train_str:20s} | {test_str:10s} | '
              f'{tm["count"]:4d} | {em["count"]:4d} | '
              f'{fmt_pct(tm["wr_10d"])} | {fmt_pct(em["wr_10d"])} | {decay_str} | '
              f'{fmt_ret(tm["ret_10d"])} | {fmt_ret(em["ret_10d"])}')

        wf_results.append({
            'label': wf_label,
            'train_wr': tm['wr_10d'],
            'test_wr': em['wr_10d'],
            'train_ret': tm['ret_10d'],
            'test_ret': em['ret_10d'],
            'train_count': tm['count'],
            'test_count': em['count'],
            'decay': decay,
        })

    # WF 汇总
    valid_decays = [w['decay'] for w in wf_results if w['decay'] is not None]
    train_wrs = [w['train_wr'] for w in wf_results if w['train_wr'] is not None]
    test_wrs = [w['test_wr'] for w in wf_results if w['test_wr'] is not None]

    print(f'\n  Walk-Forward Summary:')
    if valid_decays:
        avg_decay = np.mean(valid_decays)
        print(f'    Avg decay: {avg_decay:+.1f}%')
        print(f'    Train avg WR: {np.mean(train_wrs):.1%}')
        print(f'    Test avg WR: {np.mean(test_wrs):.1%}')
        if avg_decay < 0:
            print(f'    Verdict: EXCELLENT (test > train)')
        elif avg_decay < 15:
            print(f'    Verdict: OK (decay < 15%)')
        elif avg_decay < 30:
            print(f'    Verdict: CAUTION (mild overfit, 15-30%)')
        else:
            print(f'    Verdict: WARNING (overfit risk, >30%)')

    # ==================== Part B2: 置信度分组 ====================
    print(f'\n{"=" * 100}')
    print(f'  Part B2: Signal Quality by Confidence Level')
    print(f'{"=" * 100}')

    conf_bins = [(0, 0.4, 'low(<0.4)'), (0.4, 0.6, 'mid(0.4-0.6)'),
                 (0.6, 0.8, 'high(0.6-0.8)'), (0.8, 1.01, 'vhigh(≥0.8)')]
    print(f'  {"Confidence":15s} | {"Count":>5s} | {"10d WR":>6s} | {"10d Ret":>8s} | '
          f'{"20d DD":>7s} | {"Avg Conf":>8s}')
    print(f'  {"-" * 60}')

    for lo, hi, label in conf_bins:
        sub = df_all[(df_all['confidence'] >= lo) & (df_all['confidence'] < hi)]
        if len(sub) == 0:
            print(f'  {label:15s} |     0 |   N/A |    N/A  |    N/A |     N/A')
            continue
        wr = (sub['ret_10d'] > 0).mean()
        ar = sub['ret_10d'].mean()
        dd = sub['max_dd'].mean()
        ac = sub['confidence'].mean()
        print(f'  {label:15s} | {len(sub):5d} | {wr:5.1%} | {ar:+7.2%} | '
              f'{dd:6.2%} | {ac:8.2f}')

    # 置信度-胜率相关性
    if len(df_all) > 10:
        corr = df_all['confidence'].corr(df_all['ret_10d'].apply(lambda x: 1 if x > 0 else 0))
        print(f'\n  Confidence-WR correlation: {corr:.3f} (positive = good)')

    # ==================== Part E: 因子效果分析 ====================
    print(f'\n{"=" * 100}')
    print(f'  Part E: Factor Analysis (data-driven)')
    print(f'{"=" * 100}')

    # === E1: 逐因子WR对比 ===
    factor_checks = {
        '量能背驰': lambda r: '量能' in r,
        'MACD确认': lambda r: 'MACD确认' in r,
        '强引力': lambda r: '强引力' in r,
        '中引力': lambda r: '中引力' in r,
        '黄金分割': lambda r: '黄金分割' in r,
        '放量突破': lambda r: '放量突破' in r,
        '缩量回踩': lambda r: '缩量回踩' in r,
        '放量回踩': lambda r: '放量回踩' in r,
        '强三买': lambda r: '强三买' in r,
        '弱三买': lambda r: '弱三买' in r,
        '次级别中枢': lambda r: '次级别中枢' in r,
        '延伸': lambda r: '延伸' in r and '中枢' not in r[:5],
        '扩张': lambda r: '扩张' in r,
        '收敛': lambda r: '收敛' in r,
        'MACD金叉': lambda r: 'MACD金叉' in r,
        '2买3买重叠': lambda r: '2买3买重叠' in r,
        '缩量回调': lambda r: '缩量回调' in r,
        '放量回调': lambda r: '放量回调' in r,
        '实体突破': lambda r: '实体突破' in r,
        '强上升确认': lambda r: '强上升趋势确认' in r,
    }

    print(f'  {"Factor":12s} | {"With":>5s} | {"WR":>6s} | {"Ret":>8s} | '
          f'{"W/O":>5s} | {"WR":>6s} | {"Ret":>8s} | {"dWR":>6s}')
    print(f'  {"-" * 80}')

    factor_impact = {}
    for fname, check_fn in factor_checks.items():
        has = df_all['reason'].apply(check_fn) if 'reason' in df_all.columns else pd.Series([False]*len(df_all))
        w = df_all[has]
        wo = df_all[~has]
        if len(w) < 3:
            continue
        wr_w = (w['ret_10d'] > 0).mean()
        ret_w = w['ret_10d'].mean()
        wr_wo = (wo['ret_10d'] > 0).mean()
        ret_wo = wo['ret_10d'].mean()
        dwr = wr_w - wr_wo
        factor_impact[fname] = (dwr, len(w), wr_w)
        print(f'  {fname:12s} | {len(w):5d} | {wr_w:5.1%} | {ret_w:+7.2%} | '
              f'{len(wo):5d} | {wr_wo:5.1%} | {ret_wo:+7.2%} | {dwr:+5.1%}')

    # 排序显示最有影响力的因子
    sorted_factors = sorted(factor_impact.items(), key=lambda x: x[1][0], reverse=True)
    print(f'\n  Top factors by WR improvement:')
    for fname, (dwr, cnt, wr) in sorted_factors[:8]:
        print(f'    {fname:12s}: {dwr:+5.1%} ({cnt} signals, {wr:.1%} WR)')

    # === E2: 振幅比分组 ===
    print(f'\n  --- Amplitude Ratio Bins ---')
    if 'pivot_divergence_ratio' in df_all.columns:
        div = df_all[df_all['pivot_divergence_ratio'] > 0]
        if len(div) > 10:
            div_bins = [(0, 0.3, '<0.3强'), (0.3, 0.6, '0.3-0.6'),
                        (0.6, 1.0, '0.6-1.0背驰'), (1.0, 2.0, '1.0-2.0无'),
                        (2.0, 99, '>2.0强离开')]
            print(f'  {"Amp Range":12s} | {"Count":>5s} | {"10d WR":>6s} | {"10d Ret":>8s}')
            print(f'  {"-" * 40}')
            for lo, hi, label in div_bins:
                sub = div[(div['pivot_divergence_ratio'] >= lo) & (div['pivot_divergence_ratio'] < hi)]
                if len(sub) < 3:
                    continue
                wr = (sub['ret_10d'] > 0).mean()
                ar = sub['ret_10d'].mean()
                print(f'  {label:12s} | {len(sub):5d} | {wr:5.1%} | {ar:+7.2%}')

    # === E3: 按买点类型x置信度细分 ===
    print(f'\n  --- Confidence by Type ---')
    print(f'  {"Type":8s} | {"n":>5s} | {"Low WR":>7s} | {"Mid WR":>7s} | {"High WR":>7s} | {"VHigh WR":>8s}')
    print(f'  {"-" * 55}')
    for pt in ['1buy', '2buy', '3buy']:
        sub = df_all[df_all['point_type'] == pt]
        if len(sub) < 10:
            continue
        parts = []
        for lo, hi, _ in conf_bins:
            s = sub[(sub['confidence'] >= lo) & (sub['confidence'] < hi)]
            if len(s) >= 3:
                parts.append(f'{(s["ret_10d"] > 0).mean():6.1%}')
            else:
                parts.append(f'  N/A ')
        print(f'  {pt:8s} | {len(sub):5d} | {" | ".join(parts)}')

    # ==================== Step 6: 卖点验证 ====================
    print(f'\n{"=" * 100}')
    print(f'  Part D: Sell Signal Quality by Period')
    print(f'{"=" * 100}')

    if all_sell_evaluated:
        df_sells = pd.DataFrame(all_sell_evaluated)
        print(f'  Total sell signals: {len(df_sells)}')
        print(f'  {"Period":10s} | {"Count":>5s} | {"10d WR":>6s} | {"10d Ret":>8s} | '
              f'{"Types":20s}')
        print(f'  {"-" * 60}')
        for label, start_s, end_s in periods:
            sub = df_sells[
                (df_sells['timestamp'] >= pd.Timestamp(start_s)) &
                (df_sells['timestamp'] <= pd.Timestamp(end_s))
            ]
            if len(sub) > 0:
                # 卖点正确 = 后续下跌
                wr = (sub['ret_10d'] < 0).mean()
                ar = sub['ret_10d'].mean()
                types = sub['point_type'].value_counts().to_dict()
                types_str = ' '.join(f'{k}:{v}' for k, v in sorted(types.items()))
                print(f'  {label:10s} | {len(sub):5d} | {wr:5.1%} | {ar:+7.2%} | {types_str}')

        # 全段
        wr_all = (df_sells['ret_10d'] < 0).mean()
        ar_all = df_sells['ret_10d'].mean()
        print(f'  {"ALL":10s} | {len(df_sells):5d} | {wr_all:5.1%} | {ar_all:+7.2%} |')

    # ==================== 最终判定 ====================
    total_time = time.time() - t0
    print(f'\n{"=" * 100}')
    print(f'  Final Verdict ({total_time:.0f}s)')
    print(f'{"=" * 100}')

    # 1. 信号稳定性
    wr_std = np.std(wrs_10d) if wrs_10d else 0
    print(f'\n  [1] Signal Quality Stability:')
    print(f'      Period 10d WR std: {wr_std:.1%}')
    if wr_std < 0.10:
        print(f'      => STABLE (std < 10%)')
    elif wr_std < 0.15:
        print(f'      => FAIRLY STABLE (std 10-15%)')
    else:
        print(f'      => VOLATILE (std > 15%)')

    # 2. Walk-Forward衰减
    if valid_decays:
        avg_d = np.mean(valid_decays)
        max_d = max(valid_decays)
        print(f'\n  [2] Walk-Forward Decay:')
        print(f'      Avg: {avg_d:+.1f}%  Max: {max_d:+.1f}%')
        if avg_d < 0:
            print(f'      => EXCELLENT (test outperforms train)')
        elif avg_d < 15:
            print(f'      => NO OVERFIT (decay < 15%)')
        elif avg_d < 30:
            print(f'      => MILD OVERFIT (15-30%)')
        else:
            print(f'      => OVERFIT RISK (>30%)')

    # 3. 最弱时期
    if period_metrics:
        weakest = min([m for m in period_metrics if m['count'] >= 3],
                      key=lambda x: x['wr_10d'] if x['wr_10d'] is not None else 1,
                      default=None)
        if weakest:
            print(f'\n  [3] Weakest Period:')
            print(f'      {weakest["label"]}: {weakest["count"]} signals, '
                  f'10d WR={weakest["wr_10d"]:.1%}, 10d Ret={weakest["ret_10d"]:+.2%}')

    print(f'\n{"=" * 100}')
    print('  Walk-Forward Validation Complete')


if __name__ == '__main__':
    main()
