#!/usr/bin/env python3
"""快速验证: 50只tushare数据对比v12 vs v12vp"""
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

# 50只测试池
with open('/workspace/chanlun_system/config_v12.json') as f:
    codes = json.load(f)['codes'][:50]

start_date, end_date = '2020-01-01', '2025-12-31'

# 拉数据
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
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(codes)}", flush=True)
    except Exception as e:
        print(f"  {code}: {e}")

print(f"成功: {len(data_map)}只")

def run_bt(engine_name, data_map):
    mod = importlib.import_module(engine_name)
    engine = mod.SignalEngine()
    
    t0 = time.time()
    all_signals = engine.generate(data_map)
    print(f"  信号: {time.time()-t0:.1f}s")
    
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
            if dt not in df.index:
                continue
            loc = df.index.get_loc(dt)
            if loc < 1:
                continue
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
    
    # 配对胜率
    eq_map = {eq.index[i]: eq.iloc[i] for i in range(len(eq))}
    ct = defaultdict(list)
    for t in trades:
        ct[t['code']].append(t)
    
    wins, profits = 0, []
    for code, tl in ct.items():
        tl.sort(key=lambda x: x['date'])
        bs = []
        for t in tl:
            if t['type'] == 'BUY':
                bs.append(t['date'])
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
    
    return {
        'sharpe': sharpe, 'return': total_ret, 'max_dd': max_dd,
        'trades': total_t, 'win_rate': wr, 'pl_ratio': avg_w / avg_l if avg_l > 0 else 0,
        'final_eq': eq.iloc[-1]
    }

print("\n--- v12基线 ---")
b = run_bt('signal_engine', data_map)
print(f"  Sharpe={b['sharpe']:.3f} 收益={b['return']:.2%} 回撤={b['max_dd']:.2%} 胜率={b['win_rate']:.2%} 交易={b['trades']} 盈亏比={b['pl_ratio']:.2f}")

print("\n--- v12+量价 ---")
v = run_bt('signal_engine_v12vp', data_map)
print(f"  Sharpe={v['sharpe']:.3f} 收益={v['return']:.2%} 回撤={v['max_dd']:.2%} 胜率={v['win_rate']:.2%} 交易={v['trades']} 盈亏比={v['pl_ratio']:.2f}")

print(f"\n{'指标':<12} {'v12':>10} {'v12+量价':>10} {'变化':>10}")
print("-" * 44)
for name, key in [('Sharpe','sharpe'),('收益%','return'),('回撤%','max_dd'),('胜率%','win_rate'),('交易','trades'),('盈亏比','pl_ratio')]:
    bv, vv = b[key], v[key]
    d = vv - bv
    if key in ('trades',):
        print(f"{name:<10} {bv:>10d} {vv:>10d} {d:>+10d}")
    elif key in ('return', 'max_dd'):
        print(f"{name:<10} {bv:>10.2%} {vv:>10.2%} {d:>+10.2%}")
    else:
        print(f"{name:<10} {bv:>10.3f} {vv:>10.3f} {d:>+10.3f}")

with open('/workspace/backtest_vp_real_50.txt', 'w') as f:
    f.write(f"v12: {b}\nv12vp: {v}\n")
print("\n保存到 backtest_vp_real_50.txt")
