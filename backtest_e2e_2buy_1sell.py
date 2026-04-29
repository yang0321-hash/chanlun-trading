"""端到端回测: 日线买点→30min 2买入场→1卖减70%→紧跟踪清仓

验证完整生产策略:
  入场: 日线任意买点 + 30min 2买确认, 止损=1买低点
  出场: 30min 1卖减仓70%, 剩余30%用紧跟踪(2/4/7%)
  兜底: 止损/时间止损(20bars)
"""
import os, sys, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD

hs = HybridSource()
SAMPLE = '.claude/temp/sample_300.txt'
COMMISSION = 0.001

# 出场参数
REDUCE_PCT = 0.70
TIGHT_TIERS = [(0.02, 0.008), (0.04, 0.015), (0.07, 0.030)]
TIME_STOP_BARS = 20

with open(SAMPLE) as f:
    raw_codes = [l.strip() for l in f if l.strip()]
try:
    with open('.claude/skills/stock-name-matcher/stock_data.json', encoding='utf-8') as f:
        _names = json.load(f)
except:
    _names = {}

def to_hs(c):
    p = c.split('.')
    return p[1].lower() + p[0] if len(p) == 2 else c

def find_1buy_stop(bps, after_idx):
    for bp in reversed(bps):
        if bp.index >= after_idx:
            continue
        if bp.point_type in ('1buy',):
            return bp.stop_loss if bp.stop_loss > 0 else 0
    return 0

# === Phase 1: 数据加载 + 信号检测 ===
print('=== 端到端回测: 2买入场 + 1卖减仓 + 紧跟踪 ===')
print()
t0 = time.time()
entries = []
cache = {}

for ri, raw in enumerate(raw_codes):
    hc = to_hs(raw)
    name = _names.get(raw, '')
    if 'ST' in name:
        continue
    try:
        df_d = hs.get_kline(hc, period='daily')
        if df_d is None or len(df_d) < 200:
            continue
        if float(df_d['close'].iloc[-1]) < 3.0:
            continue
        if 'volume' in df_d.columns and float(df_d['volume'].tail(60).mean()) < 50000:
            continue

        cs = pd.Series(df_d['close'].values)
        mc = MACD(cs)
        kl = KLine.from_dataframe(df_d, strict_mode=False)
        fr = FractalDetector(kl, confirm_required=False).get_fractals()
        if len(fr) < 4:
            continue
        st = StrokeGenerator(kl, fr, min_bars=3).get_strokes()
        if len(st) < 3:
            continue
        pv = PivotDetector(kl, st).get_pivots()
        if not pv:
            continue
        det = BuySellPointDetector(fr, st, [], pv, macd=mc)
        buys_d, _ = det.detect_all()
        seen_d = {}
        for b in buys_d:
            if b.index not in seen_d or b.confidence > seen_d[b.index].confidence:
                seen_d[b.index] = b

        df_30 = hs.get_kline(hc, period='30min')
        if df_30 is None or len(df_30) < 80:
            continue
        c30 = pd.Series(df_30['close'].values)
        m30 = MACD(c30)
        k30 = KLine.from_dataframe(df_30, strict_mode=False)
        f30 = FractalDetector(k30, confirm_required=False).get_fractals()
        if len(f30) < 4:
            continue
        s30 = StrokeGenerator(k30, f30, min_bars=3).get_strokes()
        if len(s30) < 3:
            continue
        p30 = PivotDetector(k30, s30).get_pivots()
        d30 = BuySellPointDetector(f30, s30, [], p30, macd=m30)
        buys_30, sells_30 = d30.detect_all()

        seen_b = {}
        for b in buys_30:
            if b.index not in seen_b or b.confidence > seen_b[b.index].confidence:
                seen_b[b.index] = b
        seen_s = {}
        for s in sells_30:
            if s.index not in seen_s or s.confidence > seen_s[s.index].confidence:
                seen_s[s.index] = s

        bp30 = sorted(seen_b.values(), key=lambda x: x.index)
        sp30 = sorted(seen_s.values(), key=lambda x: x.index)
        cache[hc] = (df_30, bp30, sp30)
        min_30date = df_30.index[0]

        for b in seen_d.values():
            if b.index < 50 or b.index + 1 >= len(df_d):
                continue
            sig_date = df_d.index[b.index]
            if sig_date < min_30date:
                continue
            stop_d = b.stop_loss if b.stop_loss > 0 else float(df_d['close'].iloc[b.index]) * 0.95

            later_buys = [bp for bp in bp30
                         if df_30.index[min(bp.index, len(df_30)-1)] > pd.Timestamp(sig_date)]
            entry_bp = None
            for bp in later_buys:
                if bp.point_type in ('2buy', '2buy_strong', 'class2buy', '2b3bbuy'):
                    entry_bp = bp
                    break
            if not entry_bp:
                continue

            eidx = min(entry_bp.index + 1, len(df_30) - 1)
            ep = float(df_30['open'].iloc[eidx])
            s1 = find_1buy_stop(bp30, entry_bp.index)
            fs = max(s1, stop_d) if s1 > 0 else stop_d
            if ep <= 0 or fs >= ep:
                continue

            entries.append({
                'code': hc,
                'name': name[:6],
                'daily_type': b.point_type,
                '30m_type': entry_bp.point_type,
                'ep': ep,
                'stop': fs,
                'eidx': eidx,
                'sig_date': sig_date,
            })
    except Exception:
        pass
    if (ri + 1) % 60 == 0:
        print(f'  [{ri+1}/300] entries={len(entries)} ({time.time()-t0:.0f}s)')

print(f'入场信号: {len(entries)}个 ({time.time()-t0:.0f}s)')


# === Phase 2: 完整策略模拟 ===
def simulate_full(ents):
    trades = []
    for e in ents:
        df_30, bp30, sp30 = cache[e['code']]
        future = df_30.iloc[e['eidx'] + 1:]
        if len(future) < 2:
            continue
        ep, sp = e['ep'], e['stop']
        entry_date = df_30.index[e['eidx']]
        max_p = ep
        reduced = False
        reduce_price = 0
        reduce_idx = 0

        for idx in range(len(future)):
            h = float(future['high'].iloc[idx])
            l = float(future['low'].iloc[idx])
            c = float(future['close'].iloc[idx])
            fi = e['eidx'] + 1 + idx
            hold = idx + 1
            max_p = max(max_p, h)

            # 全额止损
            if l <= sp:
                if reduced:
                    ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (sp - ep) / ep * 100
                else:
                    ret = (sp - ep) / ep * 100
                trades.append({'ret': ret, 'reason': 'stop', 'hold': hold,
                              'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                              'reduced': reduced})
                break

            # 30min 1卖 → 减仓70%
            if not reduced and sp30:
                for s in sp30:
                    s_date = df_30.index[min(s.index, len(df_30) - 1)]
                    if s_date <= entry_date:
                        continue
                    if s.index > e['eidx'] and s.index <= fi:
                        if s.point_type in ('1sell',):
                            reduce_price = c
                            reduce_idx = idx
                            reduced = True
                            break

            # 紧跟踪止盈 (减仓后)
            if reduced:
                for tg, trail in reversed(TIGHT_TIERS):
                    if (max_p - ep) / ep >= tg:
                        tp = ep * (1 + tg - trail)
                        if l <= tp:
                            ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (tp - ep) / ep * 100
                            trades.append({'ret': ret, 'reason': f'tight_trail_{tg:.0%}', 'hold': hold,
                                          'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                                          'reduced': True})
                            break
                else:
                    # 紧跟踪没触发，继续
                    pass
                if 'ret' in trades[-1] if trades else False:
                    continue
                # check if the last append was from tight_trail
                if trades and 'reason' in trades[-1] and trades[-1].get('reason','').startswith('tight_trail'):
                    break
            # time stop
            if hold >= TIME_STOP_BARS and c <= ep:
                if reduced:
                    ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (c - ep) / ep * 100
                else:
                    ret = (c - ep) / ep * 100
                trades.append({'ret': ret, 'reason': 'time', 'hold': hold,
                              'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                              'reduced': reduced})
                break
        else:
            c = float(future['close'].iloc[-1])
            if reduced:
                ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (c - ep) / ep * 100
            else:
                ret = (c - ep) / ep * 100
            trades.append({'ret': ret, 'reason': 'eod', 'hold': len(future),
                          'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                          'reduced': reduced})

    return trades


# 修复: 紧跟踪break逻辑
def simulate_full_v2(ents):
    trades = []
    for e in ents:
        df_30, bp30, sp30 = cache[e['code']]
        future = df_30.iloc[e['eidx'] + 1:]
        if len(future) < 2:
            continue
        ep, sp = e['ep'], e['stop']
        entry_date = df_30.index[e['eidx']]
        max_p = ep
        reduced = False
        reduce_price = 0
        exited = False

        for idx in range(len(future)):
            h = float(future['high'].iloc[idx])
            l = float(future['low'].iloc[idx])
            c = float(future['close'].iloc[idx])
            fi = e['eidx'] + 1 + idx
            hold = idx + 1
            max_p = max(max_p, h)

            # 1. 全额止损
            if l <= sp:
                if reduced:
                    ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (sp - ep) / ep * 100
                else:
                    ret = (sp - ep) / ep * 100
                trades.append({'ret': ret, 'reason': 'stop', 'hold': hold,
                              'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                              'reduced': reduced})
                exited = True
                break

            # 2. 30min 1卖 → 减仓
            if not reduced and sp30:
                for s in sp30:
                    s_date = df_30.index[min(s.index, len(df_30) - 1)]
                    if s_date <= entry_date:
                        continue
                    if s.index > e['eidx'] and s.index <= fi:
                        if s.point_type in ('1sell',):
                            reduce_price = c
                            reduced = True
                            break

            # 3. 紧跟踪 (减仓后)
            if reduced:
                hit = False
                for tg, trail in reversed(TIGHT_TIERS):
                    if (max_p - ep) / ep >= tg:
                        tp = ep * (1 + tg - trail)
                        if l <= tp:
                            ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (tp - ep) / ep * 100
                            trades.append({'ret': ret, 'reason': f'tight_trail', 'hold': hold,
                                          'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                                          'reduced': True})
                            exited = True
                            hit = True
                            break
                if hit:
                    break

            # 4. 时间止损
            if hold >= TIME_STOP_BARS and c <= ep:
                if reduced:
                    ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (c - ep) / ep * 100
                else:
                    ret = (c - ep) / ep * 100
                trades.append({'ret': ret, 'reason': 'time', 'hold': hold,
                              'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                              'reduced': reduced})
                exited = True
                break

        if not exited:
            c = float(future['close'].iloc[-1])
            if reduced:
                ret = REDUCE_PCT * (reduce_price - ep) / ep * 100 + (1-REDUCE_PCT) * (c - ep) / ep * 100
            else:
                ret = (c - ep) / ep * 100
            trades.append({'ret': ret, 'reason': 'eod', 'hold': len(future),
                          'daily_type': e['daily_type'], '30m_type': e['30m_type'],
                          'reduced': reduced})

    return trades


print()
print('=' * 100)
print('  端到端回测: 日线买点 → 30min 2买入场 → 1卖减70% → 紧跟踪(2/4/7%) → 止损/20bars')
print('=' * 100)

trades = simulate_full_v2(entries)
if not trades:
    print('无交易')
    sys.exit(0)

rets = [t['ret'] for t in trades]
wins = [r for r in rets if r > 0]
losses = [r for r in rets if r < 0]
gp = sum(wins)
gl = abs(sum(losses)) if losses else 0.01

print(f'\n--- 总体 ---')
print(f'  交易: {len(trades)}笔')
print(f'  胜率: {len(wins)/len(rets)*100:.1f}%')
print(f'  均收益: {np.mean(rets):+.2f}%')
print(f'  PF: {gp/gl:.2f}')
print(f'  总PnL: {sum(rets):+.0f}% (加权)')
print(f'  均持仓: {np.mean([t["hold"] for t in trades]):.0f}根30min')

print(f'\n--- 出场原因 ---')
reasons = defaultdict(list)
for t in trades:
    reasons[t['reason']].append(t['ret'])
for r, rs in sorted(reasons.items(), key=lambda x: -len(x[1])):
    wr = sum(1 for x in rs if x > 0) / len(rs) * 100
    print(f'  {r:<20} {len(rs):>3}笔  WR:{wr:.0f}%  Avg:{np.mean(rs):+.2f}%')

print(f'\n--- 1卖减仓触发率 ---')
triggered = sum(1 for t in trades if t.get('reduced'))
print(f'  触发: {triggered}/{len(trades)} ({triggered/len(trades)*100:.0f}%)')

print(f'\n--- 按日线买点类型 ---')
by_daily = defaultdict(list)
for t in trades:
    by_daily[t['daily_type']].append(t['ret'])
for dt, rs in sorted(by_daily.items(), key=lambda x: -len(x[1])):
    wr = sum(1 for r in rs if r > 0) / len(rs) * 100
    print(f'  {dt:<12} {len(rs):>3}笔  WR:{wr:.0f}%  Avg:{np.mean(rs):+.2f}%  PF:{sum(r for r in rs if r>0)/max(abs(sum(r for r in rs if r<0)),0.01):.2f}')

print(f'\n--- 按30min 2买类型 ---')
by_30m = defaultdict(list)
for t in trades:
    by_30m[t['30m_type']].append(t['ret'])
for dt, rs in sorted(by_30m.items(), key=lambda x: -len(x[1])):
    wr = sum(1 for r in rs if r > 0) / len(rs) * 100
    print(f'  {dt:<12} {len(rs):>3}笔  WR:{wr:.0f}%  Avg:{np.mean(rs):+.2f}%  PF:{sum(r for r in rs if r>0)/max(abs(sum(r for r in rs if r<0)),0.01):.2f}')

print(f'\n--- TOP10盈利 ---')
for t in sorted(trades, key=lambda x: -x['ret'])[:10]:
    print(f'  +{t["ret"]:.1f}% 日线{t["daily_type"]}→30m{t["30m_type"]} '
          f'hold:{t["hold"]}bars {t["reason"]} {"(1卖减仓)" if t["reduced"] else ""}')

print(f'\n--- TOP10亏损 ---')
for t in sorted(trades, key=lambda x: x['ret'])[:10]:
    print(f'  {t["ret"]:.1f}% 日线{t["daily_type"]}→30m{t["30m_type"]} '
          f'hold:{t["hold"]}bars {t["reason"]} {"(1卖减仓)" if t["reduced"] else ""}')

print(f'\n总耗时: {time.time()-t0:.0f}s')
