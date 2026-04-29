#!/usr/bin/env python3
"""
backtest_v2_sector_full.py
验证v2.0板块共振过滤(完整版):
1. 批量获取1038只信号个股的行业分类(tushare)
2. 用industry_to_sector映射到同花顺板块
3. 获取同花顺50行业资金流(今日涨跌+主力净流入)
4. 验证板块共振过滤对信号质量的影响
"""
import sys, os, json, time
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

print("=" * 60)
print("v2.0 板块共振过滤完整验证")
print("=" * 60)

import tushare as ts
os.environ.pop('TOKEN_TUSHARE', None)
ts.set_token(os.environ.get('TOKEN_TUSHARE'))
pro = ts.pro_api()

# ── 1. 加载信号 ────────────────────────────────────────────────────
sig = pd.read_pickle('/workspace/backtest_v2_signals.pkl')
sig = sig[sig['date'] >= '2026-03-26'].copy()
codes_unique = sig['code'].unique().tolist()
print(f"\n信号股票: {len(codes_unique)} 只 (去重后)")

# ── 2. 批量获取行业分类 ───────────────────────────────────────────
industry_cache = '/workspace/industry_cache.pkl'
if os.path.exists(industry_cache):
    industry_df = pd.read_pickle(industry_cache)
    print(f"加载行业缓存: {len(industry_df)} 条")
else:
    print("\n批量获取行业分类...")
    def to_tushare_code(code):
        mkt = 'SZ' if code.startswith('SZ') else 'SH'
        return code[2:8] + '.' + mkt

    all_results = []
    BATCH = 50
    for i in range(0, len(codes_unique), BATCH):
        batch = codes_unique[i:i+BATCH]
        ts_codes = [to_tushare_code(c) for c in batch]
        try:
            df = pro.stock_basic(ts_code=','.join(ts_codes), fields='ts_code,industry')
            if len(df) > 0:
                for _, row in df.iterrows():
                    if pd.notna(row['industry']):
                        all_results.append({'ts_code': row['ts_code'], 'industry': row['industry']})
        except Exception as ex:
            # 批量失败，单个补查
            for tc in ts_codes:
                try:
                    df = pro.stock_basic(ts_code=tc, fields='ts_code,industry')
                    if len(df) > 0 and pd.notna(df.iloc[0]['industry']):
                        all_results.append({'ts_code': df.iloc[0]['ts_code'], 'industry': df.iloc[0]['industry']})
                except:
                    pass
        if (i // BATCH + 1) % 5 == 0:
            print(f"  {min(i+BATCH, len(codes_unique))}/{len(codes_unique)} ({len(all_results)}条)")
        time.sleep(0.5)

    industry_df = pd.DataFrame(all_results)
    industry_df.to_pickle(industry_cache)
    print(f"行业获取完成: {len(industry_df)} 条，已缓存")

# ── 5. 行业→板块映射 ────────────────────────────────────────────
with open('/workspace/chanlun_system/sector_mapping.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)
industry_to_sector = mapping.get('industry_to_sector', {})
print(f"\n行业→板块映射: {len(industry_to_sector)} 条")

# 给sig加ts_code列
def to_tushare_code(code):
    mkt = 'SZ' if code.startswith('SZ') else 'SH'
    return code[2:8] + '.' + mkt

sig['ts_code'] = sig['code'].apply(to_tushare_code)
industry_df['ts_code'] = industry_df['ts_code'].str.upper()

sig = sig.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')
sig['sector'] = sig['industry'].map(industry_to_sector)
sector_mapped = sig['sector'].notna().sum()
print(f"板块映射: {sector_mapped}/{len(sig)} ({sector_mapped/len(sig)*100:.1f}%)")

# ── 4. 获取同花顺板块资金流 ──────────────────────────────────────
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
url = 'http://data.10jqka.com.cn/funds/hyzjl/board.html'
sectors = []
try:
    r = requests.get(url, headers=headers, timeout=10)
    r.encoding = 'gbk'
    soup = BeautifulSoup(r.text, 'html.parser')
    rows = soup.select('table.m-table tbody tr')
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cols) >= 8:
            try:
                sectors.append({
                    'sector_name': cols[1],
                    'chg_pct': float(cols[3].replace('%', '')),
                    'net_flow': float(cols[6]),
                })
            except:
                pass
except Exception as ex:
    print(f"获取板块资金流失败: {ex}")

print(f"\n同花顺板块数据: {len(sectors)} 个")
sector_chg = {s['sector_name']: s['chg_pct'] for s in sectors}
sector_flow = {s['sector_name']: s['net_flow'] for s in sectors}

# 今日强势板块
strong = sorted(sectors, key=lambda x: x['net_flow'], reverse=True)
print(f"\n今日主力资金流入TOP5:")
for s in strong[:5]:
    print(f"  {s['sector_name']}: 涨{s['chg_pct']:+.2f}% 净流入{s['net_flow']:+.2f}亿")
print(f"\n今日主力资金流出TOP5:")
for s in strong[-5:]:
    print(f"  {s['sector_name']}: 涨{s['chg_pct']:+.2f}% 净流入{s['net_flow']:+.2f}亿")

# ── 5. 合并板块涨跌到信号 ────────────────────────────────────────
sig['sector_chg'] = sig['sector'].map(sector_chg)
sig['sector_flow'] = sig['sector'].map(sector_flow)

has_sector = sig[sig['sector_chg'].notna()].copy()
print(f"\n有板块数据的信号: {len(has_sector)} 个 ({len(has_sector)/len(sig)*100:.1f}%)")

# ── 6. 板块共振过滤效果 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 板块共振过滤效果 ===")
print("=" * 60)

def print_filter_stats(subset, label):
    n = len(subset)
    if n == 0:
        print(f"\n{label}: 0 个")
        return
    avg_pos = subset['pos_pct'].mean()
    high_q = (subset['pos_pct'] > 0.15).sum()
    print(f"\n{label}: {n} 个信号")
    print(f"  平均仓位: {avg_pos:.4f}")
    print(f"  高质量>0.15: {high_q} ({high_q/n*100:.1f}%)")
    # 大盘分布
    for mkt in ['up', '震荡', 'down']:
        m = subset[subset['market_state'] == mkt]
        if len(m) > 0:
            print(f"  {mkt}: {len(m)}个 ({len(m)/n*100:.0f}%)")
    # 信号类型
    for t, cnt in subset['type'].value_counts().head(3).items():
        print(f"  {t}: {cnt}")

# 无过滤(基线)
print_filter_stats(has_sector, "基线(全部有板块信号)")

# 过滤1: 板块涨>=0
f1 = has_sector[has_sector['sector_chg'] >= 0]
print_filter_stats(f1, "过滤1[板块涨>=0%]")

# 过滤2: 板块涨>=2%
f2 = has_sector[has_sector['sector_chg'] >= 2]
print_filter_stats(f2, "过滤2[板块涨>=2%]")

# 过滤3: 板块涨>0 且 净流入>0 (主线共振)
f3 = has_sector[(has_sector['sector_chg'] > 0) & (has_sector['sector_flow'] > 0)]
print_filter_stats(f3, "过滤3[主线共振: 涨>0 且 净流入>0]")

# 过滤4: 板块涨<0 或 净流出<0 (弱势板块)
f4 = has_sector[(has_sector['sector_chg'] <= 0) | (has_sector['sector_flow'] <= 0)]
print_filter_stats(f4, "过滤4[弱势板块: 跌 或 净流出]")

# ── 7. v2.0 完整过滤(大盘up + 主线共振) ─────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 完整过滤效果 ===")
print("=" * 60)

v2_up = sig[sig['market_state'] == 'up'].copy()
v2_full = v2_up[(v2_up['sector_chg'].notna()) & (v2_up['sector_chg'] > 0) & (v2_up['sector_flow'] > 0)].copy()
print_filter_stats(v2_up, "仅大盘上涨笔(基线)")
print_filter_stats(v2_full, "v2.0完整(上涨笔+共振)")

if len(v2_up) > 0:
    pct_keep = len(v2_full) / len(v2_up) * 100
    print(f"\nv2.0过滤保留率: {len(v2_full)}/{len(v2_up)} = {pct_keep:.0f}%")

# ── 8. 核心对比: 主线共振 vs 弱势板块 ────────────────────────────
print("\n" + "=" * 60)
print("=== 主线共振 vs 弱势板块 对比 ===")
print("=" * 60)

if len(f3) > 0 and len(f4) > 0:
    q_main = (f3['pos_pct'] > 0.15).mean() * 100
    q_weak = (f4['pos_pct'] > 0.15).mean() * 100
    pos_main = f3['pos_pct'].mean()
    pos_weak = f4['pos_pct'].mean()
    diff = q_main - q_weak
    print(f"\n主线共振({len(f3)}个): 高质量率={q_main:.1f}%, 平均仓位={pos_main:.4f}")
    print(f"弱势板块({len(f4)}个): 高质量率={q_weak:.1f}%, 平均仓位={pos_weak:.4f}")
    print(f"质量差异: {diff:+.1f}pp")
    if diff > 5:
        print(f"\n✅ 板块共振过滤有效: +{diff:.0f}pp")
    elif diff < -5:
        print(f"\n❌ 板块共振过滤负效果: {diff:.0f}pp")
    else:
        print(f"\n⚠️ 板块共振过滤差异较小: {diff:.1f}pp")

# ── 9. 结论 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 板块共振验证结论 ===")
print("=" * 60)
if len(f3) > 0 and len(has_sector) > 0:
    q_full = (f3['pos_pct'] > 0.15).mean()
    q_base = (has_sector['pos_pct'] > 0.15).mean()
    print(f"\n主线共振 vs 基线高质量率: {q_full*100:.1f}% vs {q_base*100:.1f}%")
    print(f"v2.0板块共振过滤{'有效' if q_full > q_base else '无明显效果'}")
    print(f"当前市场强势板块: {[s['sector_name'] for s in strong[:5]]}")
    print(f"建议: 今日重点关注{[s['sector_name'] for s in strong[:3]]}等主线板块")

sig.to_pickle('/workspace/backtest_v2_signals_with_sector.pkl')
print(f"\n数据已保存: backtest_v2_signals_with_sector.pkl")
