"""混合出场: 跟踪止盈为主 + 30min 2卖减半仓"""
import os, sys, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD

hs = HybridSource()
with open('.claude/temp/sample_300.txt') as f:
    raw_codes = [l.strip() for l in f if l.strip()]

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

print('加载 + 检测信号...')
t0 = time.time()
entries = []
cache = {}

for ri, raw in enumerate(raw_codes):
    hc = to_hs(raw)
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
                'ep': ep,
                'stop': fs,
                'eidx': eidx,
            })
    except Exception:
        pass
    if (ri + 1) % 60 == 0:
        print(f'  [{ri+1}/300] entries={len(entries)} ({time.time()-t0:.0f}s)')

print(f'入场信号: {len(entries)}个 ({time.time()-t0:.0f}s)')


# === 纯跟踪止盈 (baseline) ===
def sim_trailing(ents, ts_bars, tiers):
    trades = []
    for e in ents:
        df_30 = cache[e['code']][0]
        future = df_30.iloc[e['eidx'] + 1:]
        if len(future) < 2:
            continue
        ep, sp = e['ep'], e['stop']
        max_p = ep
        for idx in range(len(future)):
            h = float(future['high'].iloc[idx])
            l = float(future['low'].iloc[idx])
            c = float(future['close'].iloc[idx])
            hold = idx + 1
            max_p = max(max_p, h)
            if l <= sp:
                trades.append((sp - ep) / ep * 100)
                break
            if tiers:
                done = False
                for tg, trail in reversed(tiers):
                    if (max_p - ep) / ep >= tg:
                        tp = ep * (1 + tg - trail)
                        if l <= tp:
                            trades.append((tp - ep) / ep * 100)
                            done = True
                            break
                if done:
                    break
            if hold >= ts_bars and c <= ep:
                trades.append((c - ep) / ep * 100)
                break
        else:
            trades.append((float(future['close'].iloc[-1]) - ep) / ep * 100)
    return trades


# === 混合: 2卖减半 + 跟踪止盈清仓 ===
def sim_hybrid(ents, ts_bars, tiers, reduce_pct=0.5):
    """2卖出现 → 减半仓(锁定部分利润), 剩余继续跟踪止盈"""
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

        for idx in range(len(future)):
            h = float(future['high'].iloc[idx])
            l = float(future['low'].iloc[idx])
            c = float(future['close'].iloc[idx])
            fi = e['eidx'] + 1 + idx
            hold = idx + 1
            max_p = max(max_p, h)

            # 止损 (全额)
            if l <= sp:
                if reduced:
                    # 已减仓: 减半部分用reduce_price, 剩余用stop
                    ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (sp - ep) / ep * 100
                else:
                    ret = (sp - ep) / ep * 100
                trades.append(ret)
                break

            # 检查是否出现2卖 → 减仓
            if not reduced and sp30:
                bar_time = df_30.index[min(fi, len(df_30) - 1)]
                for s in sp30:
                    s_date = df_30.index[min(s.index, len(df_30) - 1)]
                    if s_date <= entry_date:
                        continue
                    if s.index > e['eidx'] and s.index <= fi:
                        if s.point_type in ('2sell', '2sell_strong', 'class2sell', '2b3bsell'):
                            reduce_price = c
                            reduced = True
                            break

            # 跟踪止盈 (全额清仓)
            if tiers:
                done = False
                for tg, trail in reversed(tiers):
                    if (max_p - ep) / ep >= tg:
                        tp = ep * (1 + tg - trail)
                        if l <= tp:
                            if reduced:
                                ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (tp - ep) / ep * 100
                            else:
                                ret = (tp - ep) / ep * 100
                            trades.append(ret)
                            done = True
                            break
                if done:
                    break

            # 时间止损
            if hold >= ts_bars and c <= ep:
                if reduced:
                    ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (c - ep) / ep * 100
                else:
                    ret = (c - ep) / ep * 100
                trades.append(ret)
                break
        else:
            c = float(future['close'].iloc[-1])
            if reduced:
                ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (c - ep) / ep * 100
            else:
                ret = (c - ep) / ep * 100
            trades.append(ret)
    return trades


def pr(label, trades):
    if not trades:
        print(f'{label:<50} 无信号')
        return
    wins = [r for r in trades if r > 0]
    losses = [r for r in trades if r < 0]
    gp = sum(wins)
    gl = abs(sum(losses)) if losses else 0.01
    wr = len(wins) / len(trades) * 100
    avg = np.mean(trades)
    pf = gp / gl
    print(f'{label:<50} {len(trades):>3}笔  WR:{wr:>5.1f}%  '
          f'Avg:{avg:>+6.2f}%  PF:{pf:>4.2f}  PnL:{avg*len(trades):>+7.0f}')


print()
print('=' * 100)
print(f'{"出场策略":<50} {"笔":>3}  {"胜率":>6}  {"均收益":>7}  {"PF":>5}  {"总PnL":>7}')
print('=' * 100)

# baseline
pr('跟踪止盈(20bars, 3/6/10%) [baseline]', sim_trailing(entries, 20, [(0.03,0.01),(0.06,0.02),(0.10,0.04)]))
pr('跟踪止盈(30bars, 3/6/10%)', sim_trailing(entries, 30, [(0.03,0.01),(0.06,0.02),(0.10,0.04)]))

print('--- 混合: 2卖减半仓 + 跟踪止盈清仓 ---')
pr('2卖减50% + 跟踪(20bars, 3/6/10%)', sim_hybrid(entries, 20, [(0.03,0.01),(0.06,0.02),(0.10,0.04)], 0.5))
pr('2卖减50% + 跟踪(30bars, 3/6/10%)', sim_hybrid(entries, 30, [(0.03,0.01),(0.06,0.02),(0.10,0.04)], 0.5))
pr('2卖减50% + 跟踪(40bars, 3/6/10%)', sim_hybrid(entries, 40, [(0.03,0.01),(0.06,0.02),(0.10,0.04)], 0.5))

print('--- 混合: 2卖减30% (保守减仓) ---')
pr('2卖减30% + 跟踪(20bars, 3/6/10%)', sim_hybrid(entries, 20, [(0.03,0.01),(0.06,0.02),(0.10,0.04)], 0.3))
pr('2卖减30% + 跟踪(30bars, 3/6/10%)', sim_hybrid(entries, 30, [(0.03,0.01),(0.06,0.02),(0.10,0.04)], 0.3))

print('--- 混合: 2卖减70% (激进减仓) ---')
pr('2卖减70% + 跟踪(20bars, 3/6/10%)', sim_hybrid(entries, 20, [(0.03,0.01),(0.06,0.02),(0.10,0.04)], 0.7))
pr('2卖减70% + 跟踪(30bars, 3/6/10%)', sim_hybrid(entries, 30, [(0.03,0.01),(0.06,0.02),(0.10,0.04)], 0.7))

# 2卖触发后换更紧的跟踪
print('--- 混合: 2卖后换紧跟踪(2/4/7%) ---')
def sim_hybrid_tighten(ents, ts_bars, tiers, tight_tiers, reduce_pct=0.5):
    """2卖出现 → 减半仓, 剩余换更紧的跟踪止盈"""
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
        active_tiers = tiers

        for idx in range(len(future)):
            h = float(future['high'].iloc[idx])
            l = float(future['low'].iloc[idx])
            c = float(future['close'].iloc[idx])
            fi = e['eidx'] + 1 + idx
            hold = idx + 1
            max_p = max(max_p, h)

            if l <= sp:
                if reduced:
                    ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (sp - ep) / ep * 100
                else:
                    ret = (sp - ep) / ep * 100
                trades.append(ret)
                break

            if not reduced and sp30:
                bar_time = df_30.index[min(fi, len(df_30) - 1)]
                for s in sp30:
                    s_date = df_30.index[min(s.index, len(df_30) - 1)]
                    if s_date <= entry_date:
                        continue
                    if s.index > e['eidx'] and s.index <= fi:
                        if s.point_type in ('2sell', '2sell_strong', 'class2sell', '2b3bsell'):
                            reduce_price = c
                            reduced = True
                            active_tiers = tight_tiers
                            max_p = h
                            break

            if active_tiers:
                done = False
                for tg, trail in reversed(active_tiers):
                    if (max_p - ep) / ep >= tg:
                        tp = ep * (1 + tg - trail)
                        if l <= tp:
                            if reduced:
                                ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (tp - ep) / ep * 100
                            else:
                                ret = (tp - ep) / ep * 100
                            trades.append(ret)
                            done = True
                            break
                if done:
                    break

            if hold >= ts_bars and c <= ep:
                if reduced:
                    ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (c - ep) / ep * 100
                else:
                    ret = (c - ep) / ep * 100
                trades.append(ret)
                break
        else:
            c = float(future['close'].iloc[-1])
            if reduced:
                ret = reduce_pct * (reduce_price - ep) / ep * 100 + (1 - reduce_pct) * (c - ep) / ep * 100
            else:
                ret = (c - ep) / ep * 100
            trades.append(ret)
    return trades

pr('2卖减50% + 剩余换紧跟踪(2/4/7%, 20bars)',
   sim_hybrid_tighten(entries, 20, [(0.03,0.01),(0.06,0.02),(0.10,0.04)],
                      [(0.02,0.008),(0.04,0.015),(0.07,0.03)], 0.5))
pr('2卖减50% + 剩余换紧跟踪(2/4/7%, 30bars)',
   sim_hybrid_tighten(entries, 30, [(0.03,0.01),(0.06,0.02),(0.10,0.04)],
                      [(0.02,0.008),(0.04,0.015),(0.07,0.03)], 0.5))
pr('2卖减30% + 剩余换紧跟踪(2/4/7%, 30bars)',
   sim_hybrid_tighten(entries, 30, [(0.03,0.01),(0.06,0.02),(0.10,0.04)],
                      [(0.02,0.008),(0.04,0.015),(0.07,0.03)], 0.3))

print(f'\n总耗时: {time.time() - t0:.0f}s')
