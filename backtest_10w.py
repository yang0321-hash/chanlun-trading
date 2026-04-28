"""10万实盘模拟回测 v2 — 修复回撤计算

修复:
  1. 预加载所有股票数据, 用日期dict查价, 不重复get_kline
  2. 用统一日历遍历所有交易日
  3. 正确计算权益和回撤
"""
import os, sys, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

import pandas as pd
import numpy as np
import json
from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD
from core.buy_sell_points import BuySellPointDetector
from agents.committee_agents import analyze_weekly_chanlun
from collections import defaultdict

# 基本面过滤
MIN_PRICE = 3.0
MIN_AVG_VOL = 50000
try:
    with open('.claude/skills/stock-name-matcher/stock_data.json', encoding='utf-8') as f:
        _stock_names = json.load(f)
except:
    _stock_names = {}

def is_filtered(raw_code, df):
    if 'volume' not in df.columns:
        return False
    last_close = float(df['close'].iloc[-1])
    if last_close < MIN_PRICE:
        return True
    avg_vol = float(df['volume'].tail(60).mean())
    if avg_vol < MIN_AVG_VOL:
        return True
    name = _stock_names.get(raw_code, '')
    if 'ST' in name or 'st' in name.lower():
        return True
    return False

CAPITAL = 100000
FIXED_STOP = -0.05
TIME_STOP_DAYS = 60
TRAILING_TIERS = [(0.03, 0.01), (0.08, 0.03), (0.15, 0.08), (0.25, 0.15)]
MAX_POSITIONS = 5
COMMISSION = 0.001

hs = HybridSource()
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else '.claude/temp/sample_50.txt'
print(f'使用样本: {SAMPLE}')
with open(SAMPLE) as f:
    raw_codes = [l.strip() for l in f if l.strip()]


def to_hs_code(c):
    parts = c.split('.')
    return parts[1].lower() + parts[0] if len(parts) == 2 else c


# ===== Phase 1: 预加载所有数据 + 检测信号 =====
print('=== Phase 1: 加载数据 + 检测信号 ===')
t0 = time.time()

# {code: {date: {open, high, low, close}}}
price_data = {}
# 所有交易信号
all_signals = []
# 所有出现的交易日
all_dates = set()

for i, raw_code in enumerate(raw_codes):
    hs_code = to_hs_code(raw_code)
    try:
        df = hs.get_kline(hs_code, period='daily')
        if df is None or len(df) < 200: continue
        if is_filtered(raw_code, df): continue
        df_w = hs.get_kline(hs_code, period='weekly')

        # 存价格数据: date -> OHLCV
        pmap = {}
        for idx in range(len(df)):
            d = df.index[idx]
            pmap[d] = {
                'open': float(df['open'].iloc[idx]),
                'high': float(df['high'].iloc[idx]),
                'low': float(df['low'].iloc[idx]),
                'close': float(df['close'].iloc[idx]),
            }
            all_dates.add(d)
        price_data[hs_code] = pmap

        n = len(df)
        close_arr = df['close'].values
        open_arr = df['open'].values

        close_s = pd.Series(close_arr)
        macd_obj = MACD(close_s)
        kline = KLine.from_dataframe(df, strict_mode=False)
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        if len(fractals) < 4: continue
        strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
        if len(strokes) < 3: continue
        pivots = PivotDetector(kline, strokes).get_pivots()
        if not pivots: continue
        det = BuySellPointDetector(fractals, strokes, [], pivots, macd=macd_obj)
        buys, _ = det.detect_all()

        last_pivot = pivots[-1]
        support_strong = last_pivot.zd
        for s in reversed(strokes):
            if s.is_down:
                support_strong = max(support_strong, s.end_value)
                break
        support_medium = (last_pivot.zg + last_pivot.zd) / 2

        seen = {}
        for b in buys:
            if b.index not in seen or b.confidence > seen[b.index].confidence:
                seen[b.index] = b

        for b in seen.values():
            if b.index < 50 or b.index + 2 >= n: continue
            entry_idx = b.index + 1
            if entry_idx >= n: continue

            ss, sm = support_strong, support_medium
            trade_date = df.index[b.index]
            entry_date = df.index[entry_idx]
            w_phase, w_zg, w_zd = '', 0, 0
            if df_w is not None and len(df_w) >= 30:
                df_w_cut = df_w[df_w.index <= trade_date]
                if len(df_w_cut) >= 30:
                    wr = analyze_weekly_chanlun(df_w_cut)
                    if wr:
                        w_zg, w_zd = wr['zg'], wr['zd']
                        if w_zd > 0: ss = max(ss, w_zd)
                        w_mid = (w_zg + w_zd) / 2
                        if w_mid > 0: sm = max(sm, w_mid)

            has_top = False
            if b.index >= 2:
                h1, h2, h3 = float(df['high'].iloc[b.index-2]), float(df['high'].iloc[b.index-1]), float(df['high'].iloc[b.index])
                l1, l2, l3 = float(df['low'].iloc[b.index-2]), float(df['low'].iloc[b.index-1]), float(df['low'].iloc[b.index])
                has_top = h2 > h1 and h2 > h3 and l2 > l1 and l2 > l3

            all_signals.append({
                'code': hs_code,
                'entry_date': entry_date,
                'open_price': open_arr[entry_idx],
                'stop_base': b.stop_loss if b.stop_loss > 0 else open_arr[entry_idx] * 0.95,
                'type': b.point_type,
                'confidence': b.confidence,
                'has_top': has_top,
                'sm': sm,
                'ss': ss,
                'w_zg': w_zg,
                'w_zd': w_zd,
            })
    except Exception:
        pass
    total = len(raw_codes)
    step = max(1, total // 5)
    if (i + 1) % step == 0 or i + 1 == total:
        print(f'  [{i+1}/{total}] data={len(price_data)} signals={len(all_signals)} ({time.time()-t0:.0f}s)')

print(f'Phase 1完成: {len(price_data)}只股票, {len(all_signals)}信号 ({time.time()-t0:.0f}s)')

# 统一日历
sorted_dates = sorted(all_dates)
date_set = set(sorted_dates)
print(f'交易日历: {len(sorted_dates)}天 ({sorted_dates[0].strftime("%Y-%m-%d")} ~ {sorted_dates[-1].strftime("%Y-%m-%d")})')


# ===== Phase 2: 模拟交易 =====
class Position:
    __slots__ = ['code', 'entry_date', 'entry_price', 'shares', 'stop_price',
                 'pos_mult', 'max_price', 'cost', 'hold_days']

    def __init__(self, code, entry_date, entry_price, shares, stop_price, pos_mult):
        self.code = code
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.stop_price = stop_price
        self.pos_mult = pos_mult
        self.max_price = entry_price
        self.cost = entry_price * shares * (1 + COMMISSION)
        self.hold_days = 0

    def check_exit(self, high, low, close):
        self.max_price = max(self.max_price, high)
        if low <= self.stop_price:
            return self.stop_price, 'stop_loss'
        gain = (self.max_price - self.entry_price) / self.entry_price
        for tg, trail in reversed(TRAILING_TIERS):
            if gain >= tg:
                tp = self.entry_price * (1 + tg - trail)
                if low <= tp:
                    return tp, f'trail_{tg:.0%}'
                break
        if self.hold_days >= TIME_STOP_DAYS and close <= self.entry_price:
            return close, 'time_stop'
        return None


def simulate(use_pos_adjust):
    # 信号按日期排序
    signals = sorted(all_signals, key=lambda x: x['entry_date'])
    sig_idx = 0

    cash = CAPITAL
    positions = []
    trades = []
    equity_curve = []
    max_equity = CAPITAL
    max_drawdown = 0.0

    for date in sorted_dates:
        # 1. 检查持仓出场
        to_remove = []
        for pi, pos in enumerate(positions):
            pos.hold_days += 1
            pdata = price_data.get(pos.code, {}).get(date)
            if pdata is None:
                continue
            result = pos.check_exit(pdata['high'], pdata['low'], pdata['close'])
            if result:
                exit_price, reason = result
                proceeds = exit_price * pos.shares * (1 - COMMISSION)
                pnl = proceeds - pos.cost
                pct_ret = (exit_price - pos.entry_price) / pos.entry_price * 100
                cash += proceeds
                trades.append({
                    'code': pos.code,
                    'entry_date': pos.entry_date,
                    'exit_date': date,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'pct_ret': pct_ret,
                    'pnl': pnl,
                    'hold_days': pos.hold_days,
                    'reason': reason,
                    'pos_mult': pos.pos_mult,
                })
                to_remove.append(pi)

        for pi in sorted(to_remove, reverse=True):
            positions.pop(pi)

        # 2. 买入信号
        while sig_idx < len(signals) and signals[sig_idx]['entry_date'] <= date:
            sig = signals[sig_idx]
            sig_idx += 1
            if sig['entry_date'] != date: continue
            if len(positions) >= MAX_POSITIONS: continue
            if any(p.code == sig['code'] for p in positions): continue

            entry_price = sig['open_price']
            if entry_price <= 0: continue

            # 仓位系数
            pos_mult = 1.0
            if use_pos_adjust:
                sm, ss = sig['sm'], sig['ss']
                # 获取当天收盘价判断
                pdata = price_data.get(sig['code'], {}).get(date)
                cur_price = pdata['close'] if pdata else entry_price
                if sm > 0 and ss > 0 and cur_price > sm * 1.02:
                    pos_mult *= 0.8
                if sig['has_top']:
                    pos_mult *= 0.7
                w_zg, w_zd = sig['w_zg'], sig['w_zd']
                if w_zg > 0 and w_zd > 0 and cur_price > 0:
                    pos_pct = (cur_price - w_zd) / (w_zg - w_zd)
                    if pos_pct > 0.7:
                        pos_mult *= 0.8

            # 分配资金
            available_slots = MAX_POSITIONS - len(positions)
            if available_slots <= 0: continue
            slot = cash / available_slots
            if use_pos_adjust:
                slot *= pos_mult
            if slot < entry_price * 100: continue

            shares = int(slot / (entry_price * 100)) * 100
            if shares <= 0: continue
            cost = entry_price * shares * (1 + COMMISSION)
            if cost > cash: continue

            stop_price = min(max(sig['stop_base'], entry_price * (1 + FIXED_STOP)), entry_price * 0.99)
            pos = Position(sig['code'], date, entry_price, shares, stop_price, pos_mult)
            positions.append(pos)
            cash -= cost

        # 3. 计算权益
        equity = cash
        for pos in positions:
            pdata = price_data.get(pos.code, {}).get(date)
            if pdata:
                equity += pdata['close'] * pos.shares
            else:
                equity += pos.entry_price * pos.shares

        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100 if max_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        equity_curve.append((date, equity))

    # 清算剩余
    for pos in positions:
        trades.append({
            'code': pos.code, 'entry_date': pos.entry_date,
            'entry_price': pos.entry_price, 'exit_price': 0,
            'pct_ret': 0, 'pnl': 0, 'hold_days': pos.hold_days,
            'reason': 'still_holding', 'pos_mult': pos.pos_mult,
        })

    return trades, equity_curve, max_drawdown


# ===== 运行 =====
print('\n=== Phase 2: 模拟交易 ===')

print('[1] 满仓策略...')
t1 = time.time()
trades_old, eq_old, dd_old = simulate(use_pos_adjust=False)
print(f'  完成: {len(trades_old)}笔 ({time.time()-t1:.0f}s)')

print('[2] 仓位调整策略...')
t2 = time.time()
trades_new, eq_new, dd_new = simulate(use_pos_adjust=True)
print(f'  完成: {len(trades_new)}笔 ({time.time()-t2:.0f}s)')


# ===== 分析 =====
def trade_stats(trades):
    closed = [t for t in trades if t['reason'] != 'still_holding']
    if not closed: return {}
    rets = [t['pct_ret'] for t in closed]
    pnls = [t['pnl'] for t in closed]
    gp = sum(r for r in rets if r > 0)
    gl = abs(sum(r for r in rets if r < 0))
    stops = [t for t in closed if t['reason'] == 'stop_loss']
    return {
        'n': len(closed), 'wr': sum(1 for r in rets if r > 0) / len(rets) * 100,
        'avg': np.mean(rets), 'pf': gp / gl if gl > 0 else 99.9,
        'total_pnl': sum(pnls),
        'avg_hold': np.mean([t['hold_days'] for t in closed]),
        'stops': len(stops), 'stop_pct': len(stops) / len(closed) * 100,
    }


def pr_stats(label, s):
    if not s:
        print(f'{label:<35} 无交易')
        return
    print(f'{label:<35} {s["n"]:>3}笔  WR:{s["wr"]:>5.1f}%  Avg:{s["avg"]:>+6.2f}%  '
          f'PF:{s["pf"]:>5.2f}  PnL:{s["total_pnl"]:>+12,.0f}  '
          f'均{s["avg_hold"]:.0f}天  止损:{s["stops"]}笔({s["stop_pct"]:.0f}%)')


print('\n' + '='*80)
print('  10万实盘模拟结果')
print('='*80)

print('\n--- 交易统计 ---')
so = trade_stats(trades_old)
sn = trade_stats(trades_new)
pr_stats('满仓策略', so)
pr_stats('仓位调整策略', sn)

print(f'\n--- 风险指标 ---')
print(f'满仓策略最大回撤: {dd_old:.1f}%')
print(f'仓位调整最大回撤: {dd_new:.1f}%')

# 按出场原因
print('\n--- 出场原因分布 ---')
for name, trades in [('满仓', trades_old), ('仓位调整', trades_new)]:
    closed = [t for t in trades if t['reason'] != 'still_holding']
    reasons = defaultdict(list)
    for t in closed:
        reasons[t['reason']].append(t['pct_ret'])
    print(f'\n  {name}:')
    for r, rets in sorted(reasons.items(), key=lambda x: -len(x[1])):
        wr = sum(1 for x in rets if x > 0) / len(rets) * 100
        print(f'    {r:<15} {len(rets):>3}笔  WR:{wr:.1f}%  Avg:{np.mean(rets):>+6.2f}%')

# 按仓位系数
print('\n--- 仓位系数分布(新策略) ---')
mg = defaultdict(list)
for t in trades_new:
    if t['reason'] == 'still_holding': continue
    mg[t['pos_mult']].append(t)
for mult in sorted(mg.keys(), reverse=True):
    sub = mg[mult]
    rets = [t['pct_ret'] for t in sub]
    wr = sum(1 for r in rets if r > 0) / len(rets) * 100
    pnl = sum(t['pnl'] for t in sub)
    print(f'  仓位{mult:.0%}: {len(sub):>3}笔  WR:{wr:.1f}%  Avg:{np.mean(rets):+.2f}%  PnL:{pnl:+,.0f}')

# 年化收益
if eq_old and len(eq_old) > 1:
    days = (eq_old[-1][0] - eq_old[0][0]).days
    years = days / 365.25

    final_old = eq_old[-1][1]
    ret_old = (final_old / CAPITAL - 1) * 100
    ann_old = ((final_old / CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0

    final_new = eq_new[-1][1]
    ret_new = (final_new / CAPITAL - 1) * 100
    ann_new = ((final_new / CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0

    print(f'\n--- 最终结果 ({years:.1f}年) ---')
    print(f'满仓策略: 终值{final_old:>12,.0f}  总收益{ret_old:>+8.1f}%  年化{ann_old:>+6.1f}%  最大回撤{dd_old:.1f}%')
    print(f'仓位调整: 终值{final_new:>12,.0f}  总收益{ret_new:>+8.1f}%  年化{ann_new:>+6.1f}%  最大回撤{dd_new:.1f}%')
    print(f'差异: 总收益{ret_new-ret_old:+.1f}pp  回撤{dd_new-dd_old:+.1f}pp')

    # 找到最大回撤区间
    for eq_data, label in [(eq_old, '满仓'), (eq_new, '调整')]:
        me = CAPITAL
        peak_d = eq_data[0][0]
        worst_dd = 0
        worst_peak = eq_data[0][0]
        worst_trough = eq_data[0][0]
        for d, e in eq_data:
            if e >= me:
                me = e
                peak_d = d
            dd = (me - e) / me * 100
            if dd > worst_dd:
                worst_dd = dd
                worst_peak = peak_d
                worst_trough = d
        print(f'{label}最大回撤: {worst_peak.strftime("%Y-%m-%d")}→{worst_trough.strftime("%Y-%m-%d")} {worst_dd:.1f}%')

print(f'\n总耗时: {time.time()-t0:.0f}s')
