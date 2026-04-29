#!/usr/bin/env python3
"""
backtest_v2_final.py - v2.0板块共振最终验证
数据源: 腾讯行情批量API(1036只) × 行业均涨 × 大盘分层
"""
import sys, os
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd

print("=" * 60)
print("v2.0 板块共振验证 (腾讯行情 × 行业均涨)")
print("=" * 60)

# ── 1. 加载数据 ─────────────────────────────────────────────────
sector_avg = pd.read_pickle('/workspace/sector_avg_chg.pkl')
df_price = pd.read_pickle('/workspace/tencent_price_cache.pkl')
industry_df = pd.read_pickle('/workspace/industry_cache.pkl')
industry_df['ts_code'] = industry_df['ts_code'].str.upper()

sig = pd.read_pickle('/workspace/backtest_v2_signals.pkl')
sig = sig[sig['date'] >= '2026-03-26'].copy()

# 标准化: 去掉SZ/SH前缀和.SZ/.SH后缀 -> 300413
def normalize_code(c):
    """去掉市场前缀和后缀: SZ300413.SZ -> 300413"""
    c = c.upper().strip()
    if c.startswith('SZ'):
        return c[2:].replace('.SZ','').replace('.SH','')
    elif c.startswith('SH'):
        return c[2:].replace('.SZ','').replace('.SH','')
    return c

# 行业 -> 信号: industry ts_code 000408.SZ -> sig_fmt SZ000408.SZ
def to_sig_fmt(c):
    c = c.upper().strip()
    if c.endswith('.SZ'):
        return 'SZ' + c[:-3] + '.SZ'
    elif c.endswith('.SH'):
        return 'SH' + c[:-3] + '.SH'
    return c

industry_df['sig_fmt'] = industry_df['ts_code'].apply(to_sig_fmt)

# 标准化用于价格匹配
sig['norm'] = sig['code'].apply(normalize_code)
df_price['norm'] = df_price['signal_code'].apply(normalize_code)

# 今日涨跌 -> 信号
sig = sig.merge(df_price[['norm', 'chg_pct']].rename(columns={'chg_pct': 'today_chg'}),
                on='norm', how='left')

# 行业 -> 信号
sig = sig.merge(industry_df[['sig_fmt', 'industry']], left_on='code', right_on='sig_fmt', how='left')

# 行业均涨
sector_chg = dict(zip(sector_avg['industry'], sector_avg['avg_chg']))
sig['sector_chg'] = sig['industry'].map(sector_chg)

print(f"\n信号总数: {len(sig)}")
print(f"今日有行情: {sig['today_chg'].notna().sum()}")
print(f"有行业均涨: {sig['sector_chg'].notna().sum()} ({sig['sector_chg'].notna().mean()*100:.1f}%)")

has_sector = sig[sig['sector_chg'].notna()].copy()
has_both = has_sector[has_sector['today_chg'].notna()].copy()

print(f"同时有行业+行情: {len(has_both)} 个")

# ── 2. 行业均涨过滤效果 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 行业均涨过滤效果 ===")
print("=" * 60)

def stats(df, label):
    n = len(df)
    if n == 0:
        print(f"{label}: 0个"); return
    avg = df['pos_pct'].mean()
    hq = (df['pos_pct'] > 0.15).sum()
    up = (df['market_state'] == 'up').sum()
    dn = (df['market_state'] == 'down').sum()
    avg_sc = df['sector_chg'].mean()
    avg_tc = df['today_chg'].mean()
    print(f"{label}({n}个): 仓位={avg:.4f} 高质量={hq}({hq/n*100:.0f}%) [up:{up} dn:{dn}] 行业均涨={avg_sc:+.2f}% 个股均涨={avg_tc:+.2f}%")

stats(has_both, '基线(全部)')
stats(has_both[has_both['sector_chg'] > 0], '行业均涨>0 (主线共振)')
stats(has_both[has_both['sector_chg'] <= 0], '行业均涨<=0 (弱势)')
stats(has_both[has_both['sector_chg'] > 1], '行业均涨>1%')
stats(has_both[has_both['sector_chg'] < -1], '行业均涨<-1%')

# ── 3. v2.0 完整过滤 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 完整过滤 ===")
print("=" * 60)

v2_up = sig[sig['market_state'] == 'up']
v2_up_s = v2_up[v2_up['sector_chg'].notna()]
v2_full = v2_up_s[v2_up_s['sector_chg'] > 0]

stats(v2_up, '上涨笔(全部)')
stats(v2_up_s, '上涨笔(有行业)')
stats(v2_full, 'v2.0完整(上涨笔+行业共振)')

if len(v2_up) > 0:
    print(f"\n过滤保留率: {len(v2_full)}/{len(v2_up)} = {len(v2_full)/len(v2_up)*100:.0f}%")

# ── 4. 核心对比 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 行业共振 vs 弱势行业 ===")
print("=" * 60)

strong_ind = has_both[has_both['sector_chg'] > 0]
weak_ind = has_both[has_both['sector_chg'] <= 0]

if len(strong_ind) > 0 and len(weak_ind) > 0:
    q_strong = (strong_ind['pos_pct'] > 0.15).mean() * 100
    q_weak = (weak_ind['pos_pct'] > 0.15).mean() * 100
    tc_strong = strong_ind['today_chg'].mean()
    tc_weak = weak_ind['today_chg'].mean()
    sc_strong = strong_ind['sector_chg'].mean()
    sc_weak = weak_ind['sector_chg'].mean()
    diff = q_strong - q_weak
    print(f"\n主线共振({len(strong_ind)}个): 高质量率={q_strong:.1f}%, 行业均涨={sc_strong:+.2f}%, 个股均涨={tc_strong:+.2f}%")
    print(f"弱势行业({len(weak_ind)}个): 高质量率={q_weak:.1f}%, 行业均涨={sc_weak:+.2f}%, 个股均涨={tc_weak:+.2f}%")
    print(f"质量差异: {diff:+.1f}pp")
    print(f"\n今日大盘状态: up={len(sig[sig['market_state']=='up'])}, down={len(sig[sig['market_state']=='down'])}, 震荡={len(sig[sig['market_state']=='震荡'])}")

# ── 5. 今日主线推荐 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 今日主线推荐 ===")
print("=" * 60)

strong_s = sector_avg[sector_avg['avg_chg'] > 0].sort_values('avg_chg', ascending=False)
weak_s = sector_avg[sector_avg['avg_chg'] < 0].sort_values('avg_chg')

print(f"\n强势行业TOP10 (个股均涨):")
for _, r in strong_s.head(10).iterrows():
    print(f"  {r['industry']}: 均涨{r['avg_chg']:+.2f}% ({r['count']}只)")

print(f"\n弱势行业BOTTOM10:")
for _, r in weak_s.head(10).iterrows():
    print(f"  {r['industry']}: 均涨{r['avg_chg']:+.2f}% ({r['count']}只)")

# ── 6. 结论 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 行业共振验证结论 ===")
print("=" * 60)

if len(strong_ind) > 0 and len(has_both) > 0:
    q_full = (strong_ind['pos_pct'] > 0.15).mean()
    q_base = (has_both['pos_pct'] > 0.15).mean()
    diff2 = (q_full - q_base) * 100
    print(f"\n主线共振高质量率: {q_full*100:.1f}%")
    print(f"全行业基线高质量率: {q_base*100:.1f}%")
    print(f"差异: {diff2:+.1f}pp")
    print(f"\n今日强势行业({len(strong_ind)}个信号) vs 弱势({len(weak_ind)}个信号)")
    print(f"主线共振效果: {'有效' if diff2 > 2 else '差异较小' if diff2 >= -2 else '负效果'}")
