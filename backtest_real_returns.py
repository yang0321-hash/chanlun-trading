#!/usr/bin/env python3
"""
backtest_real_returns.py
用TDX本地数据计算信号真实5日收益
"""
import sys, os, struct
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd

print("=" * 60)
print("v2.0 真实收益追踪 (TDX本地数据)")
print("=" * 60)

# ── 1. TDX数据读取器 ───────────────────────────────────────────
TDX_DIR = '/workspace/tdx_data'

def read_tdx_day(code):
    """
    读取TDX .day文件
    code: SZ300413.SZ 或 SH600519.SH 格式
    返回: [{date, open, close, high, low, vol}, ...] 按日期排序
    """
    # 解析: SZ300413.SZ -> sz, 300413
    #       SH600519.SH -> sh, 600519
    code_clean = code.replace('.SZ', '').replace('.SH', '').replace('SZ', '').replace('SH', '').lower()
    mkt_prefix = 'sz' if code.startswith('SZ') else 'sh'
    fname = os.path.join(TDX_DIR, mkt_prefix, 'lday', f'{mkt_prefix}{code_clean}.day')
    if not os.path.exists(fname):
        return []
    
    records = []
    with open(fname, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    for i in range(n):
        rec = data[i*32:(i+1)*32]
        date_int = int.from_bytes(rec[0:4], 'little')
        close = struct.unpack('f', rec[8:12])[0]
        records.append({'date': date_int, 'close': close})
    return sorted(records, key=lambda x: x['date'])

# ── 2. 加载信号 ─────────────────────────────────────────────────
sig = pd.read_pickle('/workspace/backtest_v2_signals.pkl')
sig = sig[sig['date'] >= '2026-03-26'].copy()
sig['date'] = pd.to_datetime(sig['date'])
sig['tdx_code'] = sig['code']  # SZ300413.SZ
sig['date_int'] = sig['date'].apply(lambda x: x.year * 10000 + x.month * 100 + x.day)

print(f"信号: {len(sig)} 个")
print(f"日期范围: {sig['date'].min().date()} ~ {sig['date'].max().date()}")

# ── 3. 预加载所有信号股票TDX数据 ──────────────────────────────
stocks = sig['code'].unique()
print(f"\n预加载 {len(stocks)} 只股票的TDX数据...")

tdx_cache = {}
missing = []
for i, code in enumerate(stocks):
    records = read_tdx_day(code)
    if records:
        tdx_cache[code] = {r['date']: r['close'] for r in records}
    else:
        missing.append(code)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(stocks)} ({len(tdx_cache)}只有效)")

print(f"成功: {len(tdx_cache)} 只, 缺失: {len(missing)} 只")
if missing:
    print(f"  缺失样本: {missing[:5]}")

# ── 4. 计算T+5收益 ───────────────────────────────────────────
# TDX数据最新到 20260420
# 04-17的T+5 = 04-24（今天，无数据）→ 不可算
# 04-16的T+5 = 04-23 → 不可算  
# 04-15的T+5 = 04-22 → 不可算
# 04-14的T+5 = 04-21 → 不可算
# 04-13的T+5 = 04-20 → ✓
# 所以可算: 信号日 <= 2026-04-13

# 找TDX最新日期
max_tdx_date = 0
for cache in tdx_cache.values():
    for d in cache.keys():
        if d > max_tdx_date:
            max_tdx_date = d
print(f"\nTDX最新日期: {max_tdx_date}")
max_sig_date_for_t5 = max_tdx_date // 100 * 100 + (max_tdx_date % 100) - 5
# 20260420 - 5 = 20260415 (如果中间无周末)
# 更精确: 从交易日历算
# 04-20是周一, T+5 = 04-24(今天, 无数据)
# 可算的最晚信号日 = 04-17

def calc_ret5(row):
    """计算T+5收益"""
    code = row['tdx_code']
    sig_date = row['date_int']
    
    if code not in tdx_cache:
        return None
    
    cache = tdx_cache[code]
    dates = sorted(cache.keys())
    
    if sig_date not in dates:
        return None
    
    # 找信号日位置
    pos = dates.index(sig_date)
    
    # T+5: 位置+5
    if pos + 5 >= len(dates):
        return None
    
    t5_date = dates[pos + 5]
    t1_close = cache[dates[pos + 1]] if pos + 1 < len(dates) else None
    t5_close = cache[t5_date]
    
    if t1_close and t5_close and t1_close > 0:
        return (t5_close - t1_close) / t1_close * 100
    return None

print("\n计算T+5收益...")
sig['ret_5d'] = sig.apply(calc_ret5, axis=1)

# 检查可算范围
has_ret = sig[sig['ret_5d'].notna()].copy()
no_ret = sig[sig['ret_5d'].isna()].copy()
print(f"\n有T+5收益: {len(has_ret)} / {len(sig)} ({len(has_ret)/len(sig)*100:.1f}%)")
print(f"无T+5收益(数据不足): {len(no_ret)}")
if len(no_ret) > 0:
    print(f"  无收益信号日期分布:")
    print(no_ret.groupby('date').size().tail(10))

# ── 5. 收益分布 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== T+5 收益分布 ===")
print("=" * 60)

ret = has_ret['ret_5d']
print(f"有效样本: {len(ret)}")
print(f"均值: {ret.mean():+.2f}%")
print(f"中位数: {ret.median():+.2f}%")
print(f"标准差: {ret.std():.2f}%")
print(f">0%: {(ret>0).sum()} ({(ret>0).mean()*100:.1f}%)")
print(f">3%: {(ret>3).sum()} ({(ret>3).mean()*100:.1f}%)")
print(f">5%: {(ret>5).sum()} ({(ret>5).mean()*100:.1f}%)")
print(f"<0%: {(ret<0).sum()} ({(ret<0).mean()*100:.1f}%)")
print(f"<-5%: {(ret<-5).sum()} ({(ret<-5).mean()*100:.1f}%)")

print(f"\n十分位:")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p}: {np.percentile(ret, p):+.2f}%")

# ── 6. 按信号日期分析 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 按信号日期 T+5 收益 ===")
print("=" * 60)

for date, grp in has_ret.groupby('date'):
    r = grp['ret_5d']
    n = len(r)
    if n >= 5:  # 至少5个样本
        print(f"{str(date)[:10]}: n={n:3d}, 均={r.mean():+6.2f}%, 中={r.median():+6.2f}%, >0:{(r>0).mean()*100:5.0f}%, >3%:{(r>3).mean()*100:5.0f}%")

# ── 7. 按买点类型 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 买点类型 T+5 收益 ===")
print("=" * 60)

def ret_stats(df, label):
    n = len(df)
    if n < 3:
        print(f"{label}: {n}个 (样本不足)"); return
    r = df['ret_5d']
    print(f"{label}({n}个): 均={r.mean():+.2f}%, 中={r.median():+.2f}%, >0:{(r>0).mean()*100:.0f}%, >3%:{(r>3).mean()*100:.0f}%, <-5%:{(r<-5).mean()*100:.0f}%")

for bt in sorted(sig['type'].unique()):
    grp = has_ret[has_ret['type'] == bt]
    ret_stats(grp, f"{bt}信号")

# ── 8. 大盘状态 vs 收益 ───────────────────────────────────────
print("\n" + "=" * 60)
print("=== 大盘状态 T+5 收益 ===")
print("=" * 60)

for ms in sig['market_state'].unique():
    grp = has_ret[has_ret['market_state'] == ms]
    ret_stats(grp, f"大盘{ms}")

# ── 9. v2.0过滤效果 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0过滤效果(真实收益) ===")
print("=" * 60)

ret_stats(has_ret, '基线(全部)')

# 行业数据
industry_df = pd.read_pickle('/workspace/industry_cache.pkl')
industry_df['ts_code'] = industry_df['ts_code'].str.upper()
sector_avg = pd.read_pickle('/workspace/sector_avg_chg.pkl')
sector_chg = dict(zip(sector_avg['industry'], sector_avg['avg_chg']))

def to_sig_fmt(c):
    c = c.upper().strip()
    if c.endswith('.SZ'):
        return 'SZ' + c[:-3] + '.SZ'
    elif c.endswith('.SH'):
        return 'SH' + c[:-3] + '.SH'
    return c

industry_df['sig_fmt'] = industry_df['ts_code'].apply(to_sig_fmt)
has_ret = has_ret.merge(industry_df[['sig_fmt', 'industry']], left_on='code', right_on='sig_fmt', how='left')
has_ret['sector_chg'] = has_ret['industry'].map(sector_chg)

has_sector = has_ret[has_ret['sector_chg'].notna()]
if len(has_sector) > 0:
    ret_stats(has_sector[has_sector['sector_chg'] > 0], '行业共振(今日涨>0)')
    ret_stats(has_sector[has_sector['sector_chg'] <= 0], '弱势行业(今日跌<=0)')

v2_up = has_ret[has_ret['market_state'] == 'up']
ret_stats(v2_up, '上涨笔(全部)')
v2_full = v2_up[v2_up['sector_chg'].notna() & (v2_up['sector_chg'] > 0)]
ret_stats(v2_full, 'v2.0完整(上涨笔+行业共振)')

# ── 10. pos_pct质量分层 vs 真实收益 ─────────────────────────
print("\n" + "=" * 60)
print("=== pos_pct质量分层 vs 真实收益 ===")
print("=" * 60)

hq = has_ret[has_ret['pos_pct'] > 0.15]
lq = has_ret[has_ret['pos_pct'] <= 0.15]
print(f"高质量(pos_pct>0.15): {len(hq)}个, 均收益={hq['ret_5d'].mean():+.2f}%, 中={hq['ret_5d'].median():+.2f}%")
print(f"低质量(pos_pct≤0.15): {len(lq)}个, 均收益={lq['ret_5d'].mean():+.2f}%, 中={lq['ret_5d'].median():+.2f}%")
diff = hq['ret_5d'].mean() - lq['ret_5d'].mean()
print(f"收益差异: {diff:+.2f}pp")
print(f"结论: {'pos_pct有效' if abs(diff) > 1 else 'pos_pct无区分度'}")

# ── 11. 结论 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 真实收益追踪结论 ===")
print("=" * 60)

overall = has_ret['ret_5d'].mean()
win_rate = (has_ret['ret_5d'] > 0).mean() * 100
print(f"\n全信号均值: {overall:+.2f}% (胜率{win_rate:.1f}%)")
print(f"评估样本: {len(has_ret)}个 (信号日≤2026-04-13)")

# 保存
has_ret.to_pickle('/workspace/backtest_with_real_returns.pkl')
print(f"\n数据已保存")
