#!/usr/bin/env python3
"""只跑v12vp，v12基线结果已知"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_system/code')
sys.path.insert(0, '/workspace/chanlun_system')
sys.path.insert(0, '/workspace/indicator')
sys.path.insert(0, '/workspace/backtest')

from dotenv import load_dotenv
load_dotenv('/opt/data/.env')
import tushare as ts
import importlib
from backtest.daily_portfolio import DailyPortfolio, CommissionConfig
from collections import defaultdict

pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))
pro._DataApi__http_url = "http://111.170.34.57:8010/"

with open('/workspace/chanlun_system/config_v12.json') as f:
    codes = json.load(f)['codes'][:50]

start_date, end_date = '2020-01-01', '2025-12-31'

print(f"拉取{len(codes)}只数据...")
data_map = {}
for idx, code in enumerate(codes):
    try:
        df = pro.daily(ts_code=code, start_date=start_date.replace('-',''), end_date=end_date.replace('-',''))
        if df is None or len(df) < 150:
            continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['trade_date'])
        df.set_index('date', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'vol']].astype(float)
        df.rename(columns={'vol': 'volume'}, inplace=True)
        data_map[code] = df
    except: pass

print(f"成功: {len(data_map)}只")

# 只跑v12vp
mod = importlib.import_module('signal_engine_v12vp')
engine = mod.SignalEngine()

t0 = time.time()
all_signals = engine.generate(data_map)
print(f"信号: {time.time()-t0:.1f}s")

all_dates = sorted(set(dt for df in data_map.values() for dt in df.index))
aligned = {code: sig.reindex(all_dates).fillna(0.0) for code, sig in all_signals.items()}

portfolio = DailyPortfolio(1_000_000, CommissionConfig(0.0003, 0.0013))
eq_curve, trades, prev_w = [], [], {}
equity = 1_000_000

for dt in all_dates:
    target = {code: float(aligned[code].loc[dt]) for code in aligned if aligned[code].loc[dt] > 0.001}
    if target:
        target = portfolio.align_positions(target)
    
    pr, px = {}, {}
    for code, df in data_map.items():
        if dt not in df.index: continue
        loc = df.index.get_loc(dt)
        if loc < 1: continue
        pc = df['close'].iloc[loc-1]
        cc = df['close'].iloc[loc]
        if pc > 0:
            pr[code] = (cc - pc) / pc
            px[code] = cc
    
    equity = portfolio._portfolio_equity(equity, prev_w, target, pr)
    
    if target and px:
        new_w = portfolio.update_positions(target, px)
        for c in set(list(prev_w.keys()) + list(target.keys())):
            ow, nw = prev_w.get(c, 0.0), new_w.get(c, 0.0)
            if abs(ow - nw) > 0.001:
                trades.append({'date': dt, 'code': c, 'type': 'BUY' if nw > ow else 'SELL'})
        prev_w = new_w
    elif not target and prev_w:
        portfolio.update_positions({c: 0.0 for c in prev_w}, px)
        prev_w = {}
    
    eq_curve.append(equity)

eq = pd.Series(eq_curve, index=all_dates)
total_ret = eq.iloc[-1] / 1_000_000 - 1.0
max_dd = abs(((eq - eq.expanding().max()) / eq.expanding().max()).min())
dr = eq.pct_change().dropna()
sharpe = (dr.mean() * 252 - 0.02) / (dr.std() * np.sqrt(252)) if dr.std() > 0 else 0

eq_map = {eq.index[i]: eq.iloc[i] for i in range(len(eq))}
ct = defaultdict(list)
for t in trades: ct[t['code']].append(t)
wins, profits = 0, []
for code, tl in ct.items():
    tl.sort(key=lambda x: x['date'])
    bs = []
    for t in tl:
        if t['type'] == 'BUY': bs.append(t['date'])
        elif t['type'] == 'SELL' and bs:
            bd = bs.pop(0)
            be, se = eq_map.get(bd, 1e6), eq_map.get(t['date'], 1e6)
            pnl = (se - be) / be
            profits.append(pnl)
            if pnl > 0: wins += 1

total_t = len(profits)
wr = wins / total_t if total_t > 0 else 0
avg_w = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
avg_l = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 0.0001
plr = avg_w / avg_l if avg_l > 0 else 0

print(f"\nv12+量价结果:")
print(f"  Sharpe={sharpe:.3f} 收益={total_ret:.2%} 回撤={max_dd:.2%} 胜率={wr:.2%} 交易={total_t} 盈亏比={plr:.2f}")

print(f"\n对比:")
print(f"  {'指标':<10} {'v12基线':>12} {'v12+量价':>12} {'变化':>12}")
print(f"  {'-'*48}")
b_sharpe, b_ret, b_dd = 1.990, 7.5375, 0.1577
for name, bv, vv in [
    ('Sharpe', b_sharpe, sharpe),
    ('收益%', b_ret, total_ret),
    ('回撤%', b_dd, max_dd),
    ('胜率%', 0.9649, wr),
    ('交易', 114, total_t),
    ('盈亏比', 19.37, plr),
]:
    d = vv - bv
    if name == '交易':
        print(f"  {name:<8} {bv:>12d} {vv:>12d} {d:>+12d}")
    elif name in ('收益%', '回撤%', '胜率%'):
        print(f"  {name:<8} {bv:>12.2%} {vv:>12.2%} {d:>+12.2%}")
    else:
        print(f"  {name:<8} {bv:>12.3f} {vv:>12.3f} {d:>+12.3f}")
