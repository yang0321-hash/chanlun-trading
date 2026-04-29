"""回测: 日线强势顶分型 + 30min 5笔背驰入场模式 (多参数网格)

逻辑:
  1. 日线检测强势顶分型
  2. 30min级别: 从顶开始计数下跌笔
  3. 需≥5笔下跌 + 中间形成中枢(ZG>ZD)
  4. 第5笔 vs 第1笔: 幅度/MACD/缩量 背驰
  5. 入场价: 第5笔结束时间之后的下一根30min open
  6. 出场: 止损 / 跟踪止盈 / 时间止损
"""
import os, sys, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
from collections import defaultdict
from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD

hs = HybridSource()
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else '.claude/temp/sample_300.txt'
COMMISSION = 0.001
TRAILING_TIERS = [(0.05, 0.02), (0.10, 0.05), (0.15, 0.08)]
TIME_STOP_BARS = 60

print(f'=== 日线顶分型+30min 5笔背驰 参数网格回测 ===')

with open(SAMPLE) as f:
    raw_codes = [l.strip() for l in f if l.strip()]


def to_hs_code(c):
    parts = c.split('.')
    return parts[1].lower() + parts[0] if len(parts) == 2 else c


# ===== Phase 1: 扫描日线强势顶分型 =====
print('\n[1] 扫描日线强势顶分型...')
t0 = time.time()
top_fractals = []

for i, raw_code in enumerate(raw_codes):
    hs_code = to_hs_code(raw_code)
    try:
        df = hs.get_kline(hs_code, period='daily')
        if df is None or len(df) < 100:
            continue
        n = len(df)
        h, l, c, o = df['high'].values, df['low'].values, df['close'].values, df['open'].values
        vol = df['volume'].values if 'volume' in df.columns else None

        for j in range(max(0, n-200)+2, n):
            if not (h[j-1] > h[j-2] and h[j-1] > h[j] and l[j-1] > l[j-2] and l[j-1] > l[j]):
                continue
            # 强势: 中间K线实体 >= 前20日均实体*0.5
            body = abs(c[j-1] - o[j-1])
            avg_body = np.mean([abs(c[k] - o[k]) for k in range(max(0, j-20), j)])
            if avg_body > 0 and body < avg_body * 0.5:
                continue
            top_fractals.append({
                'code': hs_code,
                'date': df.index[j-1],
                'price': float(h[j-1]),
            })
    except Exception:
        pass

    total = len(raw_codes)
    step = max(1, total // 5)
    if (i + 1) % step == 0:
        print(f'  [{i+1}/{total}] 顶分型: {len(top_fractals)} ({time.time()-t0:.0f}s)')

print(f'  日线强势顶分型: {len(top_fractals)}个')

# 过滤30min数据范围
df_ref = hs.get_kline('sh600519', period='30min')
if df_ref is not None:
    min_date = df_ref.index[0]
    top_fractals = [tf for tf in top_fractals if tf['date'] >= min_date]
    print(f'  30min范围内: {len(top_fractals)}个 (>{min_date.strftime("%Y-%m-%d")})')


# ===== Phase 2: 30min分析, 收集原始信号(含所有指标) =====
print('\n[2] 30min 5笔下跌+中枢检测...')
raw_signals = []
cache_30min = {}  # code -> (strokes, macd, df)

def get_30min(code):
    if code in cache_30min:
        return cache_30min[code]
    try:
        df = hs.get_kline(code, period='30min')
        if df is None or len(df) < 80:
            cache_30min[code] = None
            return None
        close_s = pd.Series(df['close'].values)
        macd = MACD(close_s)
        kline = KLine.from_dataframe(df, strict_mode=False)
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        if len(fractals) < 4:
            cache_30min[code] = None
            return None
        strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
        if len(strokes) < 3:
            cache_30min[code] = None
            return None
        pivots = PivotDetector(kline, strokes).get_pivots()
        result = (strokes, macd, df, pivots)
        cache_30min[code] = result
        return result
    except Exception:
        cache_30min[code] = None
        return None

for i, tf in enumerate(top_fractals):
    code = tf['code']
    data = get_30min(code)
    if data is None:
        continue
    strokes, macd, df_30, pivots = data

    # 找日线顶分型对应的30min高点位置
    top_price = tf['price']
    top_idx = len(strokes) - 1
    best_dist = float('inf')
    for si, s in enumerate(strokes):
        high_val = max(s.start_value, s.end_value)
        dist = abs(high_val - top_price)
        if dist < best_dist:
            best_dist = dist
            top_idx = si

    # 从top之后收集下跌笔
    down_strokes = []
    for si in range(top_idx, len(strokes)):
        s = strokes[si]
        if s.end_value < s.start_value:
            down_strokes.append(s)

    if len(down_strokes) < 5:
        continue

    s1, s2, s3, s4, s5 = down_strokes[0], down_strokes[1], down_strokes[2], down_strokes[3], down_strokes[4]

    # 中枢检查
    def sr(s):
        return max(s.start_value, s.end_value), min(s.start_value, s.end_value)
    h2, l2 = sr(s2); h3, l3 = sr(s3); h4, l4 = sr(s4)
    zg = min(h2, h3, h4)
    zd = max(l2, l3, l4)
    if zg <= zd:
        continue  # 无中枢

    # 幅度比
    amp1 = abs(s1.start_value - s1.end_value)
    amp5 = abs(s5.start_value - s5.end_value)
    amp_ratio = amp5 / amp1 if amp1 > 0 else 1.0

    # MACD面积比
    macd_ratio = 1.0
    try:
        hist = macd.get_histogram_series()
        offset = macd._kline_offset
        def sarea(s):
            si, ei = max(0, s.start_index - offset), min(len(hist)-1, s.end_index - offset)
            return sum(abs(float(hist.iloc[j])) for j in range(si, ei+1) if float(hist.iloc[j]) < 0) if ei > si else 0
        a1, a5 = sarea(s1), sarea(s5)
        macd_ratio = a5 / a1 if a1 > 0 else 1.0
    except Exception:
        pass

    # 缩量比
    vol_ratio = 1.0
    if 'volume' in df_30.columns:
        try:
            v = df_30['volume'].values
            v1s, v1e = max(0, s1.start_index), min(len(v)-1, s1.end_index)
            v5s, v5e = max(0, s5.start_index), min(len(v)-1, s5.end_index)
            if v1e > v1s and v5e > v5s:
                vol_ratio = np.mean(v[v5s:v5e+1]) / np.mean(v[v1s:v1e+1]) if np.mean(v[v1s:v1e+1]) > 0 else 1.0
        except Exception:
            pass

    # 用时间定位entry: 第5笔end_datetime之后的下一根30min
    s5_end_dt = s5.end_datetime
    future_mask = df_30.index > pd.Timestamp(s5_end_dt)
    future_df = df_30[future_mask]
    if len(future_df) == 0:
        continue
    entry_date = future_df.index[0]
    entry_price = float(future_df['open'].iloc[0])
    stop_price = min(s5.start_value, s5.end_value)

    if entry_price <= 0 or stop_price >= entry_price:
        continue

    raw_signals.append({
        'code': code,
        'daily_top_date': tf['date'],
        'daily_top_price': top_price,
        'entry_date': entry_date,
        'entry_price': entry_price,
        'stop_price': stop_price,
        'amp_ratio': amp_ratio,
        'macd_ratio': macd_ratio,
        'vol_ratio': vol_ratio,
        'pivot_zg': round(zg, 2),
        'pivot_zd': round(zd, 2),
        'down_count': len(down_strokes),
    })

    if (i + 1) % 200 == 0:
        print(f'  [{i+1}/{len(top_fractals)}] 中枢信号: {len(raw_signals)}')

print(f'  5笔下跌+中枢信号: {len(raw_signals)}个')


# ===== Phase 3: 参数网格搜索 =====
print('\n[3] 参数网格回测...')

def simulate(signals, amp_th, macd_th, vol_th, require_both_div=False):
    """按参数筛选信号并模拟交易"""
    trades = []
    for sig in signals:
        amp_div = sig['amp_ratio'] < amp_th
        macd_div = sig['macd_ratio'] < macd_th
        vol_ok = sig['vol_ratio'] < vol_th

        if require_both_div:
            if not (amp_div and macd_div):
                continue
        else:
            if not (amp_div or macd_div):
                continue

        # 模拟出场
        code = sig['code']
        data = cache_30min.get(code)
        if data is None:
            continue
        _, _, df_30, _ = data

        future_mask = df_30.index > sig['entry_date']
        future = df_30[future_mask]
        if len(future) < 2:
            continue

        ep = sig['entry_price']
        sp = sig['stop_price']
        shares = int(100000 * 0.2 / (ep * 100)) * 100
        if shares <= 0:
            continue
        cost = ep * shares * (1 + COMMISSION)
        max_p = ep

        for idx in range(len(future)):
            bar = future.iloc[idx]
            high, low, close = float(bar['high']), float(bar['low']), float(bar['close'])
            hold = idx + 1
            max_p = max(max_p, high)

            if low <= sp:
                trades.append({'ret': (sp-ep)/ep*100, 'reason': 'stop', 'hold': hold})
                break
            gain = (max_p - ep) / ep
            exited = False
            for tg, trail in reversed(TRAILING_TIERS):
                if gain >= tg:
                    tp = ep * (1 + tg - trail)
                    if low <= tp:
                        trades.append({'ret': (tp-ep)/ep*100, 'reason': f'trail_{tg:.0%}', 'hold': hold})
                        exited = True
                        break
            if exited:
                break
            if hold >= TIME_STOP_BARS and close <= ep:
                trades.append({'ret': (close-ep)/ep*100, 'reason': 'time', 'hold': hold})
                break
        else:
            last_c = float(future['close'].iloc[-1])
            trades.append({'ret': (last_c-ep)/ep*100, 'reason': 'eod', 'hold': len(future)})
    return trades


def stats(trades):
    if not trades:
        return {}
    rets = [t['ret'] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    gp = sum(wins)
    gl = abs(sum(losses)) if losses else 0.01
    return {
        'n': len(trades),
        'wr': len(wins)/len(rets)*100,
        'avg': np.mean(rets),
        'pf': gp/gl,
        'stops': sum(1 for t in trades if t['reason'] == 'stop'),
        'avg_hold': np.mean([t['hold'] for t in trades]),
    }


def pr(label, s):
    if not s:
        print(f'{label:<45} 无信号')
        return
    print(f'{label:<45} {s["n"]:>4}笔  WR:{s["wr"]:>5.1f}%  '
          f'Avg:{s["avg"]:>+6.2f}%  PF:{s["pf"]:>5.2f}  '
          f'止损:{s["stops"]:>3}笔  均{s["avg_hold"]:.0f}bars')


# 网格参数
configs = [
    # (label, amp_th, macd_th, vol_th, require_both)
    ('原始(amp<90% OR macd<80%)',           0.90, 0.80, 2.0, False),
    ('amp<60% OR macd<60%',                 0.60, 0.60, 2.0, False),
    ('amp<40% OR macd<40%',                 0.40, 0.40, 2.0, False),
    ('amp<60% AND macd<80%',                0.60, 0.80, 2.0, True),
    ('amp<60% AND macd<60%',                0.60, 0.60, 2.0, True),
    ('amp<40% AND macd<80%',                0.40, 0.80, 2.0, True),
    ('(amp<60% OR macd<60%) + 缩量<80%',   0.60, 0.60, 0.80, False),
    ('(amp<60% AND macd<80%) + 缩量<80%',  0.60, 0.80, 0.80, True),
    ('amp<40% OR macd<40% + 缩量<80%',     0.40, 0.40, 0.80, False),
    ('amp<40% AND macd<60% + 缩量<80%',    0.40, 0.60, 0.80, True),
]

print()
print('='*100)
print(f'{"参数组合":<45} {"笔数":>4}  {"胜率":>6}  {"均收益":>7}  {"盈亏比":>5}  {"止损":>5}  {"持仓":>5}')
print('='*100)

best_avg = -99
best_label = ''
for label, at, mt, vt, both in configs:
    trades = simulate(raw_signals, at, mt, vt, both)
    s = stats(trades)
    pr(label, s)
    if s and s['avg'] > best_avg and s['n'] >= 10:
        best_avg = s['avg']
        best_label = label

print(f'\n最佳(均收益>=10笔): {best_label} (Avg:{best_avg:+.2f}%)')

# 最佳参数详细出场分析
print('\n--- 最佳参数出场分布 ---')
for label, at, mt, vt, both in configs:
    if label != best_label:
        continue
    trades = simulate(raw_signals, at, mt, vt, both)
    reasons = defaultdict(list)
    for t in trades:
        reasons[t['reason']].append(t['ret'])
    for r, rs in sorted(reasons.items(), key=lambda x: -len(x[1])):
        wr = sum(1 for x in rs if x > 0) / len(rs) * 100
        print(f'  {r:<15} {len(rs):>3}笔  WR:{wr:.0f}%  Avg:{np.mean(rs):+.2f}%')

print(f'\n总耗时: {time.time()-t0:.0f}s')
