#!/usr/bin/env python3
"""
backtest_v2大盘分层.py
用scan_with_signal_dates.pkl的CC15信号 + 沪指MA5/MA10大盘状态
验证v2.0: 大盘分层过滤是否改善信号质量
"""
import sys, os
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 60)
print("v2.0 大盘分层回测验证")
print("=" * 60)

# ── 1. 加载 CC15 信号 ────────────────────────────────────────────────
sig_df = pd.read_pickle('/workspace/scanner_signal_dates.pkl')
sig_df['last_signal_date'] = pd.to_datetime(sig_df['last_signal_date'])
print(f"\nCC15信号股票: {len(sig_df)} 只")
print(f"日期范围: {sig_df['last_signal_date'].min().date()} ~ {sig_df['last_signal_date'].max().date()}")
print(f"近30天有信号: {(sig_df['recent_30d'] > 0).sum()} 只")

# 近30天有信号的股票
recent_df = sig_df[sig_df['recent_30d'] > 0].copy()
print(f"\n近30天有信号的: {len(recent_df)} 只股票")

# ── 2. 沪指日线数据 ───────────────────────────────────────────────────
import tushare as ts
os.environ.pop('TOKEN_TUSHARE', None)
pro = ts.pro_api()

end = datetime.today().strftime('%Y%m%d')
start = (datetime.today() - pd.Timedelta(days=1095)).strftime('%Y%m%d')
try:
    df_idx = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
    df_idx['trade_date'] = pd.to_datetime(df_idx['trade_date'])
    df_idx = df_idx.sort_values('trade_date').set_index('trade_date')
    df_idx['close'] = df_idx['close'].astype(float)
    df_idx['MA5'] = df_idx['close'].rolling(5).mean()
    df_idx['MA10'] = df_idx['close'].rolling(10).mean()
    print(f"\n沪指数据: {len(df_idx)} bars ({df_idx.index[0].date()} ~ {df_idx.index[-1].date()})")
except Exception as ex:
    print(f"获取沪指失败: {ex}")
    sys.exit(1)

def get_market_state(date):
    """返回大盘状态: up/震荡/down"""
    if date not in df_idx.index:
        available = df_idx.index[df_idx.index <= date]
        if len(available) == 0:
            return 'unknown'
        date = available[-1]
    row = df_idx.loc[date]
    ma5, ma10, close = row['MA5'], row['MA10'], row['close']
    if pd.isna(ma5) or pd.isna(ma10):
        return 'unknown'
    if ma5 > ma10 and close > ma5:
        return 'up'
    elif ma5 < ma10 and close < ma5:
        return 'down'
    else:
        return '震荡'

# ── 3. 对近30天信号股票, 生成精确信号日期 + 大盘状态 ─────────────────
spec = importlib.util.spec_from_file_location('cc15', '/workspace/chanlun_unified/signal_engine_cc15.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

tdx_root = Path('/workspace/tdx_data')
min_bars = 120
data_map = {}
sig_cutoff = pd.Timestamp('today') - pd.Timedelta(days=30)
sig_cutoff_date = sig_cutoff.date()
t0 = datetime.now()

print(f"\n加载 TDX 数据...")
for _, row in recent_df.iterrows():
    code = row['code']  # e.g. SZ300003.SZ or SH600519.SH
    mkt = 'sz' if code.endswith('.SZ') else 'sh'
    # 去掉.SZ/.SH后取前7位: SZ300003.SZ → sz300003.day
    fname = code.replace('.SZ', '').replace('.SH', '').lower() + '.day'
    fpath = tdx_root / mkt / 'lday' / fname
    if not fpath.exists():
        continue
    try:
        data = fpath.read_bytes()
        n = len(data) // 32
        if n < min_bars:
            continue
        arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
        dates = pd.to_datetime(arr[:, 0].astype(str), format='%Y%m%d')
        prices = arr[:, 1:5] / 100.0
        volumes = arr[:, 6].astype(np.int64)
        df = pd.DataFrame({
            'open': prices[:, 0], 'high': prices[:, 1],
            'low': prices[:, 2], 'close': prices[:, 3],
            'volume': volumes
        }, index=dates).sort_index()
        data_map[code] = df
    except:
        pass

print(f"加载数据: {len(data_map)} 只 ({datetime.now()-t0})")

# ── 4. 生成信号 + 大盘状态标注 ──────────────────────────────────────
all_signals = []
done = 0
total = len(data_map)
t0 = datetime.now()

for code, df in data_map.items():
    try:
        engine = mod.SignalEngine()
        engine.dynamic_pool_enabled = False
        sigs = engine.generate({code: df}, live_mode=False, use_pivots=False)
        s = list(sigs.values())[0]
        dates_list = s.index.tolist()
        for i in range(len(s)):
            if s.iloc[i] > 0:
                sig_date = pd.Timestamp(dates_list[i]).normalize()
                if sig_date < sig_cutoff:
                    break
                mkt = get_market_state(sig_date)
                pos_val = float(s.iloc[i])
                all_signals.append({
                    'code': code,
                    'signal_date': sig_date,
                    'market_state': mkt,
                    'pos_value': pos_val,
                })
    except:
        pass
    done += 1
    if done % 500 == 0:
        print(f"  {done}/{total} ({len(all_signals)} signals, {datetime.now()-t0})")

print(f"\n生成完成: {len(all_signals)} 个近30天信号 ({datetime.now()-t0})")
if not all_signals:
    print("无信号，退出")
    sys.exit(0)

sig_all = pd.DataFrame(all_signals)

# ── 5. 大盘状态分布 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 近30天信号 × 大盘状态分布 ===")
print("=" * 60)
total_sig = len(sig_all)
for state in ['up', '震荡', 'down', 'unknown']:
    sub = sig_all[sig_all['market_state'] == state]
    if len(sub) == 0:
        continue
    avg_pos = sub['pos_value'].mean()
    high_q = (sub['pos_value'] > 0.15).sum()
    very_high_q = (sub['pos_value'] > 0.20).sum()
    print(f"\n【{state}】{len(sub)} 个信号 ({len(sub)/total_sig*100:.1f}%)")
    print(f"  平均仓位: {avg_pos:.4f}")
    print(f"  高质量>0.15: {high_q} 个 ({high_q/len(sub)*100:.1f}%)")
    print(f"  高质量>0.20: {very_high_q} 个 ({very_high_q/len(sub)*100:.1f}%)")
    # 月度分布
    monthly = sub.groupby(sub['signal_date'].dt.to_period('M')).size()
    print(f"  月度分布:", end='')
    for period, cnt in monthly.tail(4).items():
        print(f" {period}:{cnt}", end='')
    print()

# ── 6. v2.0 过滤效果对比 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 大盘过滤效果 ===")
print("=" * 60)
baseline = sig_all[sig_all['market_state'] != 'unknown']
print(f"\n基线(排除unknown): {len(baseline)} 个信号")
print(f"  平均仓位: {baseline['pos_value'].mean():.4f}")
print(f"  高质量(>0.15): {(baseline['pos_value']>0.15).sum()} ({(baseline['pos_value']>0.15).mean()*100:.1f}%)")

# 过滤1: 仅上涨笔(up状态)
up_sigs = baseline[baseline['market_state'] == 'up']
print(f"\n过滤1[仅上涨笔]: {len(up_sigs)} 个 ({len(up_sigs)/len(baseline)*100:.1f}%)")
if len(up_sigs) > 0:
    print(f"  平均仓位: {up_sigs['pos_value'].mean():.4f} (vs 基线 {baseline['pos_value'].mean():.4f})")
    print(f"  高质量>0.15: {(up_sigs['pos_value']>0.15).sum()} ({(up_sigs['pos_value']>0.15).mean()*100:.1f}%)")

# 过滤2: 上涨+震荡
safe_sigs = baseline[baseline['market_state'].isin(['up', '震荡'])]
print(f"\n过滤2[上涨+震荡]: {len(safe_sigs)} 个 ({len(safe_sigs)/len(baseline)*100:.1f}%)")
if len(safe_sigs) > 0:
    print(f"  平均仓位: {safe_sigs['pos_value'].mean():.4f} (vs 基线 {baseline['pos_value'].mean():.4f})")
    print(f"  高质量>0.15: {(safe_sigs['pos_value']>0.15).sum()} ({(safe_sigs['pos_value']>0.15).mean()*100:.1f}%)")

# 过滤3: 下跌笔（只做1买≤30%）
down_sigs = baseline[baseline['market_state'] == 'down']
print(f"\n下跌笔信号: {len(down_sigs)} 个 ({len(down_sigs)/len(baseline)*100:.1f}%)")
if len(down_sigs) > 0:
    print(f"  平均仓位: {down_sigs['pos_value'].mean():.4f}")
    print(f"  高质量>0.15: {(down_sigs['pos_value']>0.15).sum()} ({(down_sigs['pos_value']>0.15).mean()*100:.1f}%)")

# ── 7. 当前大盘状态 ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 当前大盘状态 ===")
print("=" * 60)
today_state = get_market_state(pd.Timestamp('today').normalize())
print(f"今日大盘状态: {today_state}")
if today_state == 'up':
    print("→ 建议仓位上限: 70-80%, 可重仓2买/3买")
elif today_state == '震荡':
    print("→ 建议仓位上限: 40-50%, 谨慎做2买")
else:
    print("→ 建议仓位上限: 20-30%, 只做1买快进快出")

# ── 8. 质量提升结论 ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 回测结论 ===")
print("=" * 60)
if len(up_sigs) > 0 and len(baseline) > 0:
    quality_up = (up_sigs['pos_value'] > 0.15).mean()
    quality_all = (baseline['pos_value'] > 0.15).mean()
    quality_diff = (quality_up - quality_all) * 100
    pos_diff = up_sigs['pos_value'].mean() - baseline['pos_value'].mean()
    print(f"\n上涨笔信号质量 vs 全市场:")
    print(f"  高质量(>0.15)比例: +{quality_diff:.1f}pp ({quality_up*100:.1f}% vs {quality_all*100:.1f}%)")
    print(f"  平均仓位差异: {pos_diff:+.4f}")
    if quality_diff > 5:
        print(f"\n✅ v2.0大盘过滤有效: 上涨笔中高质量信号比例+{quality_diff:.0f}pp")
    else:
        print(f"\n⚠️ v2.0大盘过滤差异较小: {quality_diff:.1f}pp")

if today_state == 'up':
    up_pct = len(up_sigs)/len(baseline)*100
    print(f"  当前为上涨笔市场, 可适当提高仓位")
elif today_state == '震荡':
    safe_pct = len(safe_sigs)/len(baseline)*100
    print(f"  当前为震荡市场, 建议仓位上限40-50%")
else:
    print(f"  当前为下跌笔市场, 建议仓位上限20-30%, 只做1买")

# ── 9. 实际回测 (简化版) ────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 简化回测 (各层 Sharpe 对比) ===")
print("=" * 60)

INITIAL = 1_000_000
CAPITAL_PER_TRADE = INITIAL / 5  # 最多5只
SL = 0.06

def simple_backtest(signals_df, label, win_rate_override=None):
    """简化回测: 用仓位值映射方向，随机模拟"""
    equity = INITIAL
    equity_curve = [equity]
    pnl_list = []
    wins, losses = 0, 0
    for _, row in signals_df.iterrows():
        pos = row['pos_value']
        # 用pos_value估算胜率: pos越大质量越高，胜率越高
        if win_rate_override:
            wr = win_rate_override
        else:
            wr = min(0.40 + pos * 2.5, 0.80)  # 0.30~0.30pos → 0.40~0.80 wr
        ret = np.random.choice([SL, -SL], p=[wr, 1-wr])
        equity += CAPITAL_PER_TRADE * ret
        equity_curve.append(equity)
        pnl_list.append(ret)
        if ret > 0: wins += 1
        else: losses += 1
    if not pnl_list: return None
    total_ret = (equity - INITIAL) / INITIAL * 100
    wr = wins / (wins + losses) * 100
    daily_rets = [(equity_curve[i]-equity_curve[i-1])/equity_curve[i-1]
                  for i in range(1, len(equity_curve))]
    if daily_rets:
        ann_ret = np.mean(daily_rets) * 250
        ann_std = np.std(daily_rets) * np.sqrt(250)
        sharpe = ann_ret / ann_std if ann_std > 0 else 0
        peak = np.maximum.accumulate(np.array(equity_curve))
        dd = ((peak - np.array(equity_curve)) / peak * 100)
        max_dd = dd.max()
    else:
        sharpe = max_dd = ann_ret = 0
    return {'label': label, 'signals': len(pnl_list), 'win_rate': wr,
            'total_ret': total_ret, 'sharpe': sharpe, 'max_dd': max_dd}

# 跑3个场景各5次取平均
np.random.seed(42)
results = []
for label, subset in [('全市场', baseline),
                       ('仅上涨笔', up_sigs),
                       ('上涨+震荡', safe_sigs)]:
    if len(subset) == 0: continue
    runs = [simple_backtest(subset, label) for _ in range(5)]
    runs = [r for r in runs if r]
    if runs:
        avg = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
        results.append(avg)
        print(f"\n{label} ({len(subset)} 个信号):")
        print(f"  Sharpe: {avg['sharpe']:.2f}, 胜率: {avg['win_rate']:.1f}%, "
              f"总收益: {avg['total_ret']:.1f}%, 最大回撤: {avg['max_dd']:.1f}%")

# 保存
sig_all.to_pickle('/workspace/backtest_v2_signals.pkl')
print(f"\n信号已保存: /workspace/backtest_v2_signals.pkl ({len(sig_all)} 条)")
