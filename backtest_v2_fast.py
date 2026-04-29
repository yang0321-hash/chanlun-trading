#!/usr/bin/env python3
"""
backtest_v2_fast.py
用已有scanner信号 + 沪指MA5/MA10大盘状态
验证v2.0: 大盘分层过滤是否改善信号质量
"""
import sys, os
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 60)
print("v2.0 大盘分层回测验证 (快速版)")
print("=" * 60)

# ── 1. 加载 scanner 信号 ──────────────────────────────────────────────
sig_df = pd.read_pickle('/workspace/scanner_new_fw_signals_live.pkl')
sig_df['date'] = pd.to_datetime(sig_df['date'])
print(f"\n信号总数: {len(sig_df)}")
print(f"日期范围: {sig_df['date'].min().date()} ~ {sig_df['date'].max().date()}")
print(f"信号类型:\n{sig_df['type'].value_counts()}")

# 排除ETF/基金
etf_mask = sig_df['code'].str.match(r'^(SZ159|SH51|SZ123|SH88[0-9]|SZ16[0-9]|SH56[0-9])')
sig_stock = sig_df[~etf_mask].copy()
print(f"\n排除ETF/基金后: {len(sig_stock)} 个股票信号")

# 近30天信号
cutoff = pd.Timestamp('today') - pd.Timedelta(days=30)
recent = sig_stock[sig_stock['date'] >= cutoff].copy()
print(f"近30天股票信号: {len(recent)} 个")
print(f"日期分布:\n{recent.groupby('date').size()}")

# ── 2. 沪指大盘状态 ───────────────────────────────────────────────────
import tushare as ts
os.environ.pop('TOKEN_TUSHARE', None)
pro = ts.pro_api()

end = datetime.today().strftime('%Y%m%d')
start = (datetime.today() - pd.Timedelta(days=730)).strftime('%Y%m%d')
df_idx = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
df_idx['trade_date'] = pd.to_datetime(df_idx['trade_date'])
df_idx = df_idx.sort_values('trade_date').set_index('trade_date')
df_idx['close'] = df_idx['close'].astype(float)
df_idx['MA5'] = df_idx['close'].rolling(5).mean()
df_idx['MA10'] = df_idx['close'].rolling(10).mean()
print(f"\n沪指数据: {len(df_idx)} bars")

def get_market_state(date):
    """返回大盘状态: up/震荡/down"""
    date = pd.Timestamp(date).normalize()
    if date not in df_idx.index:
        available = df_idx.index[df_idx.index <= date]
        if len(available) == 0: return 'unknown'
        date = available[-1]
    row = df_idx.loc[date]
    ma5, ma10, close = row['MA5'], row['MA10'], row['close']
    if pd.isna(ma5) or pd.isna(ma10): return 'unknown'
    if ma5 > ma10 and close > ma5: return 'up'
    elif ma5 < ma10 and close < ma5: return 'down'
    else: return '震荡'

# ── 3. 标注大盘状态 ──────────────────────────────────────────────────
recent['market_state'] = recent['date'].apply(get_market_state)
print(f"\n大盘状态标注完成")

# ── 4. 大盘状态分布 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 近30天信号 × 大盘状态 ===")
print("=" * 60)
total = len(recent)
for state in ['up', '震荡', 'down', 'unknown']:
    sub = recent[recent['market_state'] == state]
    if len(sub) == 0: continue
    avg_pos = sub['pos_pct'].mean()
    high_q = (sub['pos_pct'] > 0.15).sum()
    print(f"\n【{state}】{len(sub)} 个信号 ({len(sub)/total*100:.1f}%)")
    print(f"  平均仓位: {avg_pos:.4f}")
    print(f"  高质量>0.15: {high_q} 个 ({high_q/len(sub)*100:.1f}%)")
    # 信号类型
    print(f"  信号类型:")
    for t, cnt in sub['type'].value_counts().head(3).items():
        print(f"    {t}: {cnt}")
    # 月度
    monthly = sub.groupby(sub['date'].dt.to_period('M')).size()
    print(f"  月度: ", end='')
    for p, c in monthly.items():
        print(f"{p}:{c} ", end='')
    print()

# ── 5. v2.0 过滤效果 ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 大盘过滤效果 ===")
print("=" * 60)
baseline = recent[recent['market_state'] != 'unknown']
print(f"\n基线(排除unknown): {len(baseline)} 个信号")
print(f"  平均仓位: {baseline['pos_pct'].mean():.4f}")

# 过滤1: 仅上涨笔
up_sigs = baseline[baseline['market_state'] == 'up']
print(f"\n过滤1[仅上涨笔]: {len(up_sigs)} 个 ({len(up_sigs)/len(baseline)*100:.1f}%)")
if len(up_sigs) > 0:
    print(f"  平均仓位: {up_sigs['pos_pct'].mean():.4f} (vs 基线 {baseline['pos_pct'].mean():.4f})")

# 过滤2: 上涨+震荡
safe_sigs = baseline[baseline['market_state'].isin(['up', '震荡'])]
print(f"\n过滤2[上涨+震荡]: {len(safe_sigs)} 个 ({len(safe_sigs)/len(baseline)*100:.1f}%)")
if len(safe_sigs) > 0:
    print(f"  平均仓位: {safe_sigs['pos_pct'].mean():.4f} (vs 基线 {baseline['pos_pct'].mean():.4f})")

# 过滤3: 下跌笔
down_sigs = baseline[baseline['market_state'] == 'down']
print(f"\n过滤3[下跌笔]: {len(down_sigs)} 个 ({len(down_sigs)/len(baseline)*100:.1f}%)")
if len(down_sigs) > 0:
    print(f"  平均仓位: {down_sigs['pos_pct'].mean():.4f}")

# ── 6. 当前大盘状态 ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 当前大盘状态 ===")
print("=" * 60)
today_state = get_market_state(pd.Timestamp('today'))
print(f"今日大盘: {today_state}")
if today_state == 'up':
    print("→ 仓位上限: 70-80%, 可重仓2买/3买")
    print("→ 建议: 当前上涨笔市场，信号质量高，建议仓位40-50%")
elif today_state == '震荡':
    print("→ 仓位上限: 40-50%, 谨慎做2买")
else:
    print("→ 仓位上限: 20-30%, 只做1买快进快出")

# ── 7. 质量对比 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 回测结论 ===")
print("=" * 60)
if len(up_sigs) > 0 and len(baseline) > 0:
    quality_up = (up_sigs['pos_pct'] > 0.15).mean()
    quality_all = (baseline['pos_pct'] > 0.15).mean()
    quality_diff = (quality_up - quality_all) * 100
    pos_up = up_sigs['pos_pct'].mean()
    pos_all = baseline['pos_pct'].mean()
    print(f"\n上涨笔信号质量 vs 全市场:")
    print(f"  高质量(>0.15)比例: +{quality_diff:.1f}pp ({quality_up*100:.1f}% vs {quality_all*100:.1f}%)")
    print(f"  平均仓位: {pos_up:.4f} vs {pos_all:.4f}")
    if quality_diff > 5:
        print(f"\n✅ v2.0大盘过滤有效: 上涨笔中高质量信号比例+{quality_diff:.0f}pp")
    else:
        print(f"\n⚠️ v2.0大盘过滤差异较小: {quality_diff:.1f}pp")
else:
    print("上涨笔信号数量不足，无法比较")

# 按日期信号分布
print(f"\n近30天信号日期分布:")
daily = recent.groupby(['date', 'market_state']).size().unstack(fill_value=0)
print(daily.to_string())

# ── 8. v2.0 仓位建议 ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 实盘仓位建议 ===")
print("=" * 60)
for state, max_pos in [('up', '70-80%'), ('震荡', '40-50%'), ('down', '20-30%')]:
    n = len(recent[recent['market_state'] == state])
    print(f"{state}: 仓位上限{max_pos}, 当前信号{n}个")
if today_state == 'up':
    pct = len(up_sigs)/len(baseline)*100 if len(baseline) > 0 else 0
    print(f"\n当前为上涨笔: {pct:.0f}%信号来自上涨笔，建议仓位60%")
elif today_state == '震荡':
    pct = len(safe_sigs)/len(baseline)*100 if len(baseline) > 0 else 0
    print(f"\n当前为震荡: {pct:.0f}%信号适合操作，建议仓位40%")
else:
    print(f"\n当前为下跌笔: 建议仓位≤20%, 只做1买快进快出")

recent.to_pickle('/workspace/backtest_v2_signals.pkl')
print(f"\n信号已保存: /workspace/backtest_v2_signals.pkl")
