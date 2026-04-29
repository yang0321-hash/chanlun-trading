#!/usr/bin/env python3
"""
backtest_v2_sector.py
验证v2.0板块共振过滤:
1. 获取同花顺50行业板块资金流
2. 用mootdx获取板块指数历史数据(5日涨幅)
3. 匹配个股信号到板块
4. 对比: 有板块共振 vs 无板块共振 的信号质量
"""
import sys, os, json, re
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

print("=" * 60)
print("v2.0 板块共振过滤验证")
print("=" * 60)

# ── 1. 获取同花顺行业资金流 ─────────────────────────────────────────
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
url = 'http://data.10jqka.com.cn/funds/hyzjl/board.html'

print("\n获取同花顺行业资金流...")
try:
    r = requests.get(url, headers=headers, timeout=10)
    r.encoding = 'gbk'
    soup = BeautifulSoup(r.text, 'html.parser')
    rows = soup.select('table.m-table tbody tr')
    sectors = []
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cols) >= 8:
            try:
                chg_pct = float(cols[3].replace('%', ''))
                net_flow = float(cols[6])
                sectors.append({
                    'sector_name': cols[1],  # e.g. "白酒", "煤炭开采加工"
                    'chg_pct': chg_pct,
                    'net_flow': net_flow,  # 亿元
                })
            except:
                pass
    print(f"获取到 {len(sectors)} 个行业板块")
except Exception as ex:
    print(f"获取板块资金流失败: {ex}")
    sectors = []

# ── 2. 用mootdx获取板块指数5日涨幅 ───────────────────────────────────
print("\n获取板块指数数据(5日涨幅)...")
try:
    import mootdx
    # mootdx板块指数(用akshare替代，略过)
    # 板块5日涨幅用同花顺实时涨跌作为proxy
    sector_5d_chg = {}

    print(f"板块资金流获取成功: {len(sectors)} 个")
    print(f"\n今日资金流入TOP5:")
    sorted_s = sorted(sectors, key=lambda x: x['net_flow'], reverse=True)
    for s in sorted_s[:5]:
        print(f"  {s['sector_name']}: 涨{s['chg_pct']:+.2f}% 净流入{s['net_flow']:+.2f}亿")
    print(f"\n今日资金流出TOP5:")
    for s in sorted_s[-5:]:
        print(f"  {s['sector_name']}: 涨{s['chg_pct']:+.2f}% 净流入{s['net_flow']:+.2f}亿")
except Exception as ex:
    print(f"mootdx板块数据失败: {ex}")
    sectors = []

# ── 3. 加载个股→行业映射 ────────────────────────────────────────────
with open('/workspace/chanlun_system/sector_mapping.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

industry_to_sector = mapping.get('industry_to_sector', {})
pool_mapping = mapping.get('pool_mapping', {})

# 同花顺行业名 → 今日涨跌
sector_chg = {s['sector_name']: s['chg_pct'] for s in sectors}
# 同花顺行业名 → 今日净流入
sector_flow = {s['sector_name']: s['net_flow'] for s in sectors}

# ── 4. 加载信号 ────────────────────────────────────────────────────
sig = pd.read_pickle('/workspace/backtest_v2_signals.pkl')
sig = sig[sig['date'] >= '2026-03-26'].copy()

# 匹配个股到板块
# pool_mapping: 股票代码 → 行业名(通达信格式)
# industry_to_sector: 行业名(通达信) → 板块名(同花顺)
def get_sector_chg(code):
    """返回个股所属板块的今日涨跌幅"""
    # 格式: SZ300413.SZ -> 300413.SZ; SH600522.SH -> 600522.SH
    industry = pool_mapping.get(code)  # 先试原始格式
    if not industry:
        # 去掉前2字符: SZ300413.SZ -> 300413.SZ
        stripped = code[2:]
        industry = pool_mapping.get(stripped)
    if not industry:
        return None
    sector = industry_to_sector.get(industry)
    if not sector:
        return None
    return sector_chg.get(sector, None)

def get_sector_flow(code):
    """返回个股所属板块的今日主力净流入(亿)"""
    industry = pool_mapping.get(code)
    if not industry:
        stripped = code[2:]
        industry = pool_mapping.get(stripped)
    if not industry:
        return None
    sector = industry_to_sector.get(industry)
    if not sector:
        return None
    return sector_flow.get(sector, None)

sig['sector_chg'] = sig['code'].apply(get_sector_chg)
sig['sector_flow'] = sig['code'].apply(get_sector_flow)

# 统计映射成功率
mapped = sig['sector_chg'].notna().sum()
print(f"\n信号板块映射: {mapped}/{len(sig)} ({mapped/len(sig)*100:.1f}%)")
if mapped > 0:
    print(f"有板块数据的信号: {mapped} 个")
    no_sector = sig[sig['sector_chg'].isna()]
    print(f"无板块数据: {len(no_sector)} 个 (前5个: {no_sector['code'].head().tolist()})")

# ── 5. 板块共振过滤效果 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 板块共振过滤效果 ===")
print("=" * 60)

has_sector = sig[sig['sector_chg'].notna()].copy()
print(f"\n有板块数据: {len(has_sector)} 个信号")

# 按板块涨跌幅分层
for threshold in [0, 2, -2]:
    if threshold >= 0:
        filtered = has_sector[has_sector['sector_chg'] >= threshold]
        label = f"板块涨>={threshold}%"
    else:
        filtered = has_sector[has_sector['sector_chg'] < abs(threshold)]
        label = f"板块涨<{abs(threshold)}%"
    n = len(filtered)
    pct = len(filtered) / len(has_sector) * 100 if len(has_sector) > 0 else 0
    avg_pos = filtered['pos_pct'].mean() if n > 0 else 0
    high_q = (filtered['pos_pct'] > 0.15).sum() if n > 0 else 0
    print(f"\n过滤[{label}]: {n} 个信号 ({pct:.0f}%)")
    if n > 0:
        print(f"  平均仓位: {avg_pos:.4f}")
        print(f"  高质量(>0.15): {high_q} ({high_q/n*100:.1f}%)")
        # 大盘分布
        for mkt in ['up', '震荡', 'down']:
            m = filtered[filtered['market_state'] == mkt]
            if len(m) > 0:
                print(f"  {mkt}: {len(m)}个")
        # 信号类型
        for t, cnt in filtered['type'].value_counts().head(3).items():
            print(f"  {t}: {cnt}")

# ── 6. v2.0 主线共振过滤 (板块涨>0 且 主力净流入>0) ─────────────────
print("\n" + "=" * 60)
print("=== v2.0 主线共振过滤 ===")
print("=" * 60)

mainline = has_sector[
    (has_sector['sector_chg'] > 0) &
    (has_sector['sector_flow'].notna()) &
    (has_sector['sector_flow'] > 0)
].copy()

print(f"\n主线共振(板块涨>0 且 净流入>0): {len(mainline)} 个 ({len(mainline)/len(has_sector)*100:.0f}%)")
if len(mainline) > 0:
    print(f"  平均仓位: {mainline['pos_pct'].mean():.4f}")
    print(f"  高质量>0.15: {(mainline['pos_pct']>0.15).sum()} ({(mainline['pos_pct']>0.15).mean()*100:.1f}%)")
    for mkt in ['up', 'down', '震荡']:
        m = mainline[mainline['market_state'] == mkt]
        if len(m) > 0:
            print(f"  {mkt}: {len(m)}个")
    print(f"  信号类型:")
    for t, cnt in mainline['type'].value_counts().head(3).items():
        print(f"    {t}: {cnt}")
    print(f"  板块分布:")
    for s, cnt in mainline.groupby('sector_chg').size().sort_index().tail(5).items():
        print(f"    涨{s:+.1f}%区间: {cnt}个")

# ── 7. 无板块共振(板块跌或无数据) ──────────────────────────────────
no_mainline = has_sector[
    (has_sector['sector_chg'] <= 0) | (has_sector['sector_flow'] <= 0)
].copy()
print(f"\n无主线共振(板块跌或净流出): {len(no_mainline)} 个 ({len(no_mainline)/len(has_sector)*100:.0f}%)")
if len(no_mainline) > 0:
    print(f"  平均仓位: {no_mainline['pos_pct'].mean():.4f}")
    print(f"  高质量>0.15: {(no_mainline['pos_pct']>0.15).sum()} ({(no_mainline['pos_pct']>0.15).mean()*100:.1f}%)")

# ── 8. v2.0 完整过滤 (大盘up + 主线共振) ────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 完整过滤 (大盘上涨笔 + 主线共振) ===")
print("=" * 60)

v2_full = sig[
    (sig['market_state'] == 'up') &
    (sig['sector_chg'].notna()) &
    (sig['sector_chg'] > 0) &
    (sig['sector_flow'].notna()) &
    (sig['sector_flow'] > 0)
].copy()

v2_up_only = sig[sig['market_state'] == 'up'].copy()

print(f"\n过滤前(仅大盘上涨笔): {len(v2_up_only)} 个")
print(f"v2.0完整(上涨笔+共振): {len(v2_full)} 个 ({len(v2_full)/len(v2_up_only)*100:.0f}%)")
if len(v2_full) > 0:
    print(f"  平均仓位: {v2_full['pos_pct'].mean():.4f}")
    print(f"  高质量>0.15: {(v2_full['pos_pct']>0.15).sum()} ({(v2_full['pos_pct']>0.15).mean()*100:.1f}%)")
    print(f"  信号类型:")
    for t, cnt in v2_full['type'].value_counts().head(3).items():
        print(f"    {t}: {cnt}")

# ── 9. 结论 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 板块共振验证结论 ===")
print("=" * 60)

if len(mainline) > 0 and len(no_mainline) > 0:
    quality_mainline = (mainline['pos_pct'] > 0.15).mean()
    quality_no = (no_mainline['pos_pct'] > 0.15).mean()
    diff = (quality_mainline - quality_no) * 100
    print(f"\n主线共振信号质量 vs 无共振:")
    print(f"  高质量比例: {quality_mainline*100:.1f}% vs {quality_no*100:.1f}%")
    print(f"  差异: {diff:+.1f}pp")
    if diff > 5:
        print(f"\n✅ 板块共振过滤有效: +{diff:.0f}pp")
    elif diff < -5:
        print(f"\n❌ 板块共振过滤负效果: {diff:.0f}pp")
    else:
        print(f"\n⚠️ 板块共振过滤差异较小: {diff:.1f}pp")

if len(v2_full) > 0 and len(v2_up_only) > 0:
    print(f"\nv2.0完整过滤(上涨笔+共振) vs 仅上涨笔:")
    print(f"  {len(v2_full)}/{len(v2_up_only)} ({len(v2_full)/len(v2_up_only)*100:.0f}%)")
    print(f"  建议: v2.0过滤后保留{int(len(v2_full)/len(v2_up_only)*100)}%信号")

# ── 10. 今日板块推荐 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 今日板块推荐(按v2.0主线共振) ===")
print("=" * 60)
if sectors:
    # 满足: 涨>0 且 净流入>0
    strong = [s for s in sectors if s['chg_pct'] > 0 and s['net_flow'] > 0]
    strong.sort(key=lambda x: x['net_flow'], reverse=True)
    print(f"\n主线共振板块(涨>0 且 净流入>0): {len(strong)} 个")
    for s in strong[:10]:
        print(f"  {s['sector_name']}: 涨{s['chg_pct']:+.2f}% 净流入{s['net_flow']:+.2f}亿")
