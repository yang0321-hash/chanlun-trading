#!/usr/bin/env python3
"""
backtest_v2_eastmoney_sector.py
验证v2.0板块共振过滤:
1. 新浪行业84个行业实时涨跌
2. 与tushare行业分类匹配(tushare返回的行业名)
3. 验证主线共振过滤效果
"""
import sys, os, json, re
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import requests
import tushare as ts
os.environ.pop('TOKEN_TUSHARE', None)
ts.set_token(os.environ.get('TOKEN_TUSHARE'))
pro = ts.pro_api()

print("=" * 60)
print("v2.0 板块共振验证 (新浪行业+东方财富资金流)")
print("=" * 60)

# ── 1. 获取新浪行业数据 ───────────────────────────────────────────
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
r = requests.get('http://vip.stock.finance.sina.com.cn/q/view/newFLJK.php?param=hy',
                 headers=headers, timeout=8)
r.encoding = 'gbk'
text = r.text

entries = re.findall(r'"(hangye_[^"]+)"\s*:\s*"([^"]+)"', text)
sina_sectors = []
for code, val in entries:
    parts = val.split(',')
    if len(parts) >= 6:
        try:
            name = parts[1].strip()
            chg_pct = float(parts[4]) * 100
            sina_sectors.append({
                'sina_code': code,
                'sina_name': name,
                'chg_pct': chg_pct,
            })
        except:
            pass

sina_sectors.sort(key=lambda x: x['chg_pct'], reverse=True)
print(f"新浪行业: {len(sina_sectors)} 个")
print(f"强势TOP5: {[s['sina_name']+':'+str(round(s['chg_pct'],1))+'%' for s in sina_sectors[:5]]}")
print(f"弱势BOTTOM5: {[s['sina_name']+':'+str(round(s['chg_pct'],1))+'%' for s in sina_sectors[-5:]]}")

# ── 2. 新浪行业名 vs tushare行业名 映射 ─────────────────────────
# tushare行业名(如"白酒","化学制药") → 新浪行业名(如"酒、饮料和精制茶制造业")
# 建立双向映射
sina_to_tushare = {
    '农、林、牧、渔专业及辅助性活动': '农业综合',
    '农业': '农用机械',
    '畜牧业': '养殖业',
    '渔业': '农业综合',
    '木材加工和木、竹、藤、棕、草制品业': '木材家具',
    '家具制造业': '家居用品',
    '造纸和纸制品业': '造纸',
    '印刷和记录媒介复制业': '广告包装',
    '文教、工美、体育和娱乐用品制造业': '文教休闲',
    '石油、煤炭及其他燃料加工业': '炼化及贸易',
    '化学原料和化学制品制造业': '化学原料',
    '医药制造业': '化学制药',
    '化学纤维制造业': '化纤',
    '橡胶和塑料制品业': '橡胶',
    '非金属矿物制品业': '建筑材料',
    '黑色金属冶炼和压延加工业': '钢铁',
    '有色金属冶炼和压延加工业': '小金属',
    '金属制品业': '金属制品',
    '通用设备制造业': '机械基件',
    '专用设备制造业': '专用设备',
    '汽车制造业': '汽车整车',
    '铁路、船舶、航空航天和其他运输设备制造业': '运输设备',
    '电气机械和器材制造业': '电气设备',
    '计算机、通信和其他电子设备制造业': '半导体',
    '仪器仪表制造业': '电器仪表',
    '其他制造业': '其他商业',
    '废弃资源综合利用业': '环保',
    '金属制品、机械和设备修理业': '专用设备',
    '电力、热力生产和供应业': '电力',
    '燃气生产和供应业': '供气供热',
    '水的生产和供应业': '水务',
    '房屋建筑业': '建筑装饰',
    '土木工程建筑业': '建筑工程',
    '建筑安装业': '装修装饰',
    '建筑装饰和其他建筑业': '装修装饰',
    '批发业': '商业百货',
    '零售业': '超市连锁',
    '铁路运输业': '铁路',
    '道路运输业': '公路',
    '水路运输业': '水运',
    '航空运输业': '空运',
    '管道运输业': '仓储物流',
    '装卸搬运和运输代理业': '仓储物流',
    '仓储业': '仓储物流',
    '邮政业': '仓储物流',
    '住宿业': '酒店餐饮',
    '餐饮业': '酒店餐饮',
    '电信、广播电视和卫星传输服务': '电信运营',
    '互联网和相关服务': '互联网',
    '软件和信息技术服务业': '软件开发',
    '货币金融服务': '银行',
    '资本市场服务': '证券',
    '保险业': '保险',
    '其他金融业': '多元金融',
    '房地产业': '房地产',
    '租赁业': '其他商业',
    '商务服务业': '其他社会服务',
    '研究和试验发展': '技术服务',
    '专业技术服务业': '技术服务',
    '科技推广和应用服务业': '技术服务',
    '水利管理业': '环保',
    '生态保护和环境治理业': '环保',
    '居民服务业': '其他社会服务',
    '修理和其他服务业': '其他社会服务',
    '教育': '文教休闲',
    '卫生': '医疗服务',
    '文化艺术业': '文教休闲',
    '娱乐业': '游戏',
    '综合': '综合',
}
# 反向映射
tushare_to_sina = {v: k for k, v in sina_to_tushare.items()}

# 新浪行业涨跌
sina_chg = {s['sina_name']: s['chg_pct'] for s in sina_sectors}

# ── 3. 获取tushare股票行业分类 ──────────────────────────────────
industry_cache = '/workspace/industry_cache.pkl'
if os.path.exists(industry_cache):
    industry_df = pd.read_pickle(industry_cache)
    print(f"行业缓存: {len(industry_df)} 条")
else:
    print("需要先获取行业缓存"); sys.exit(1)

industry_df['ts_code'] = industry_df['ts_code'].str.upper()

# ── 4. 加载信号 ────────────────────────────────────────────────
sig = pd.read_pickle('/workspace/backtest_v2_signals.pkl')
sig = sig[sig['date'] >= '2026-03-26'].copy()

def to_tushare_code(code):
    mkt = 'SZ' if code.startswith('SZ') else 'SH'
    return code[2:8] + '.' + mkt

sig['ts_code'] = sig['code'].apply(to_tushare_code)
sig = sig.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')

# ── 5. 匹配新浪行业涨跌 ────────────────────────────────────────
def get_sina_chg(industry):
    if pd.isna(industry): return None
    sina_name = tushare_to_sina.get(industry)  # tushare行业 → 新浪行业
    if not sina_name: return None
    return sina_chg.get(sina_name)

sig['sina_chg'] = sig['industry'].apply(get_sina_chg)

has_chg = sig[sig['sina_chg'].notna()].copy()
print(f"\n有行业涨跌信号: {len(has_chg)} / {len(sig)} ({len(has_chg)/len(sig)*100:.1f}%)")

# ── 6. 过滤效果 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 新浪行业共振过滤效果 ===")
print("=" * 60)

def stats(df, label):
    n = len(df)
    if n == 0:
        print(f"{label}: 0个"); return
    avg = df['pos_pct'].mean()
    hq = (df['pos_pct'] > 0.15).sum()
    up = (df['market_state'] == 'up').sum()
    dn = (df['market_state'] == 'down').sum()
    avg_chg = df['sina_chg'].mean()
    print(f"{label}({n}个): 仓位={avg:.4f} 高质量={hq}({hq/n*100:.0f}%) [up:{up} dn:{dn}] 行业均涨跌={avg_chg:+.2f}%")

stats(has_chg, '基线(全部)')
stats(has_chg[has_chg['sina_chg'] > 0], '行业涨>0')
stats(has_chg[has_chg['sina_chg'] <= 0], '行业跌<=0')
stats(has_chg[has_chg['sina_chg'] > 2], '行业涨>2%')
stats(has_chg[has_chg['sina_chg'] < -2], '行业跌<-2%')

# ── 7. v2.0完整过滤 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 完整过滤 ===")
print("=" * 60)

v2_up = sig[sig['market_state'] == 'up']
v2_up_chg = v2_up[v2_up['sina_chg'].notna()]
v2_full = v2_up_chg[v2_up_chg['sina_chg'] > 0]

stats(v2_up, '上涨笔(全部)')
stats(v2_up_chg, '上涨笔(有行业)')
stats(v2_full, 'v2.0完整(上涨笔+行业涨>0)')

if len(v2_up) > 0:
    print(f"\n过滤保留率: {len(v2_full)}/{len(v2_up)} = {len(v2_full)/len(v2_up)*100:.0f}%")

# ── 8. 今日强势行业 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 今日强势行业(按v2.0主线共振) ===")
print("=" * 60)
for s in sina_sectors[:10]:
    # 找有多少信号属于该行业
    tushare_ind = {v: k for k, v in tushare_to_sina.items() if k == s['sina_name']}
    n = len(has_chg[has_chg['industry'].isin([k for k, v in tushare_to_sina.items() if v == s['sina_name']])])
    if n > 0:
        print(f"  {s['sina_name']}: {s['chg_pct']:+.2f}% ({n}个信号)")
    else:
        print(f"  {s['sina_name']}: {s['chg_pct']:+.2f}% (0个信号)")

# ── 9. 核心结论 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== v2.0 行业共振验证结论 ===")
print("=" * 60)

if len(has_chg[has_chg['sina_chg']>0]) > 0 and len(has_chg) > 0:
    q_pos = (has_chg[has_chg['sina_chg']>0]['pos_pct']>0.15).mean()
    q_all = (has_chg['pos_pct']>0.15).mean()
    diff = (q_pos - q_all) * 100
    print(f"\n行业涨>0 vs 基线高质量率: {q_pos*100:.1f}% vs {q_all*100:.1f}% ({diff:+.1f}pp)")
    if diff > 3:
        print("✅ 行业共振过滤有效")
    elif diff < -3:
        print("❌ 行业共振过滤负效果")
    else:
        print("⚠️ 行业共振过滤差异较小")

# ── 10. 今日主线推荐 ───────────────────────────────────────────
print("\n=== 今日主线推荐(行业共振) ===")
for s in sina_sectors:
    if s['chg_pct'] > 2:
        print(f"  {s['sina_name']}: {s['chg_pct']:+.2f}%")

sig.to_pickle('/workspace/backtest_v2_signals_with_sina_sector.pkl')
print(f"\n数据已保存")
