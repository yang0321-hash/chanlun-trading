#!/usr/bin/env python3
"""
Part A/B 缠论框架回测
- 2买为主（止损=1买点下方2%）
- 3买次之（止损=中枢下沿下方2%）
- 仓位：单票≤30%
- SL=2%，TP=3%+8%移动止盈
- 0.618过滤

参考: /workspace/backtest_v15_all_a.py
"""
import sys, os, json, time, glob
import numpy as np
import pandas as pd
from collections import defaultdict

# 清代理
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_system/code')
sys.path.insert(0, '/workspace/chanlun_system')
sys.path.insert(0, '/workspace/backtest')

import importlib
import tushare as ts
from dotenv import load_dotenv
load_dotenv('/opt/data/.env')

pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))
pro._DataApi__http_url = "http://111.170.34.57:8010/"

# ── 全A池 ──────────────────────────────────────────────────────
try:
    with open('/workspace/chanlun_system/csi500_pool_all_a.json') as f:
        codes = json.load(f)
    print(f"全A池: {len(codes)}只")
except:
    # 备用：从TDX数据目录扫描
    tdx_root = '/workspace/tdx_data'
    codes = []
    for market in ['sz', 'sh']:
        d = f'{tdx_root}/{market}/lday'
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.day'):
                    codes.append(f.replace('.day', '').upper())
    print(f"TDX池: {len(codes)}只")

start_date, end_date = '20240601', '20260414'

# ── 数据加载 ──────────────────────────────────────────────────
cache_path = '/workspace/backtest_part_ab_data.pkl'
if os.path.exists(cache_path):
    print("加载缓存数据...")
    data_map = pd.read_pickle(cache_path)
    print(f"缓存: {len(data_map)}只")
else:
    print(f"拉取数据 {start_date}~{end_date}...")
    data_map = {}
    fail_count = 0
    t0 = time.time()
    for idx, code in enumerate(codes):
        try:
            # 转换代码格式
            ts_code = code.replace('SZ', '').replace('SH', '')
            suffix = 'SZ' if code.startswith('SZ') else 'SH'
            ts_code = f'{ts_code}.{suffix}'
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or len(df) < 100:
                fail_count += 1
                continue
            df = df.sort_values('trade_date').reset_index(drop=True)
            df['date'] = pd.to_datetime(df['trade_date'])
            df.set_index('date', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'vol']].astype(float)
            df.rename(columns={'vol': 'volume'}, inplace=True)
            data_map[code] = df
            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{len(codes)} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            fail_count += 1
    print(f"成功: {len(data_map)}只, 失败: {fail_count}只, 耗时: {time.time()-t0:.0f}s")
    pd.to_pickle(data_map, cache_path)

# ── 加载引擎并设置Part A/B参数 ────────────────────────────────
spec = importlib.util.spec_from_file_location("signal_engine",
    '/workspace/chanlun_system/code/signal_engine.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
engine = mod.SignalEngine()

# Part A/B 参数覆盖
engine.max_position = 0.30          # 单票最大30%（原20%）
engine.max_stop_pct = 0.02           # 最大止损2%（原10%）
engine.trailing_start = 0.03         # TP1=3%触发移动止盈
engine.trailing_tight = 0.03        # 回撤3%触发
engine.trailing_medium = 0.05        # 回撤5%触发
engine.trailing_wide = 0.08          # >15%盈利回撤8%
engine.base_position = 0.25          # 标准仓25%（原15%）

print(f"\nPart A/B 参数: max_pos=30%, max_SL=2%, TP1=3%, TSL=3%/5%/8%")

# ── 生成信号 ──────────────────────────────────────────────────
# 清缓存
for f in glob.glob('/workspace/chanlun_system/live_signals/*.pkl'):
    try:
        open(f, 'wb').close()
    except: pass

print("\n生成信号（2买/3买缠论框架）...")
t0 = time.time()
all_signals = engine.generate(data_map)
print(f"信号计算: {time.time()-t0:.1f}s, {len(all_signals)}只有信号")

# ── 对齐日期 ──────────────────────────────────────────────────
all_dates = sorted(set(dt for df in data_map.values() for dt in df.index))
aligned = {code: sig.reindex(all_dates).fillna(0.0) for code, sig in all_signals.items()}

# ── 基准: 等权全A ─────────────────────────────────────────────
bm_ret = pd.Series(0.0, index=all_dates)
for dt in all_dates:
    rets = []
    for code, df in data_map.items():
        if dt in df.index:
            loc = df.index.get_loc(dt)
            if loc >= 1:
                pc, cc = df['close'].iloc[loc-1], df['close'].iloc[loc]
                if pc > 0:
                    rets.append((cc - pc) / pc)
    bm_ret[dt] = np.mean(rets) if rets else 0.0
bm_eq = (1 + bm_ret).cumprod() * 1_000_000

# ── 组合回测 ─────────────────────────────────────────────────
from backtest.daily_portfolio import DailyPortfolio, CommissionConfig
portfolio = DailyPortfolio(1_000_000, CommissionConfig(0.0003, 0.0013))
equity = 1_000_000
prev_w = {}
trades = []
eq_curve = []

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

# ── 计算绩效指标 ───────────────────────────────────────────────
total_ret = eq.iloc[-1] / 1_000_000 - 1.0
max_dd = abs(((eq - eq.expanding().max()) / eq.expanding().max()).min())
dr = eq.pct_change().dropna()
sharpe = (dr.mean() * 252 - 0.02) / (dr.std() * np.sqrt(252)) if dr.std() > 0 else 0

years = (eq.index[-1] - eq.index[0]).days / 365.25
annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
calmar = annual_ret / max_dd if max_dd > 0 else 0

# Sortino
downside = dr[dr < 0]
downside_std = downside.std() * np.sqrt(252)
sortino = (dr.mean() * 252 - 0.02) / downside_std if downside_std > 0 else 0

# 基准对比
bm_total = bm_eq.iloc[-1] / 1_000_000 - 1.0
excess_ret = total_ret - bm_total
active = dr - bm_ret
ir = (active.mean() * 252) / (active.std() * np.sqrt(252)) if active.std() > 0 else 0

# 配对胜率
eq_map = {eq.index[i]: eq.iloc[i] for i in range(len(eq))}
ct = defaultdict(list)
for t in trades:
    if t['type'] == 'SELL':
        dt = t['date']
        code = t['code']
        # 找对应的买入
        for i, t2 in enumerate(trades):
            if t2['type'] == 'BUY' and t2['code'] == code and t2['date'] < dt:
                prev_dt = t2['date']
        if prev_dt and prev_dt in eq_map and dt in eq_map:
            pct = (eq_map[dt] - eq_map[prev_dt]) / eq_map[prev_dt]
            ct[code].append(pct)

wins = [v for vals in ct.values() for v in vals if v > 0]
losses = [v for vals in ct.values() for v in vals if v <= 0]
win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0
avg_win = np.mean(wins) if wins else 0
avg_loss = np.mean(losses) if losses else 0

# ── 输出结果 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  Part A/B 缠论框架回测结果")
print("="*60)
print(f"  回测期: {str(all_dates[0])[:10]} ~ {str(all_dates[-1])[:10]}")
print(f"  交易天数: {len(all_dates)}")
print(f"  总交易次数: {len(trades)}")
print(f"  盈利交易: {len(wins)} | 亏损交易: {len(losses)}")
print(f"  胜率: {win_rate:.1%}")
print(f"  平均盈利: {avg_win:+.2%} | 平均亏损: {avg_loss:+.2%}")
print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss else "  盈亏比: N/A")
print("-"*60)
print(f"  总收益: {total_ret:+.2%}")
print(f"  年化收益: {annual_ret:+.2%}")
print(f"  最大回撤: {max_dd:+.2%}")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  Sortino: {sortino:.2f}")
print(f"  Calmar: {calmar:.2f}")
print(f"  信息比率: {ir:.2f}")
print("-"*60)
print(f"  基准收益: {bm_total:+.2%}")
print(f"  超额收益: {excess_ret:+.2%}")
print(f"  策略波动: {dr.std()*np.sqrt(252):.2%}")
print("="*60)

# 保存结果
result = {
    'total_ret': float(total_ret), 'annual_ret': float(annual_ret),
    'max_dd': float(max_dd), 'sharpe': float(sharpe), 'sortino': float(sortino),
    'win_rate': float(win_rate), 'total_trades': len(trades),
    'n_wins': len(wins), 'n_losses': len(losses),
    'avg_win': float(avg_win), 'avg_loss': float(avg_loss),
    'benchmark_ret': float(bm_total), 'excess_ret': float(excess_ret),
    'calmar': float(calmar), 'ir': float(ir),
}
with open('/tmp/backtest_part_ab_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\n结果已保存到 /tmp/backtest_part_ab_result.json")
