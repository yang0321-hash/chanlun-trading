#!/usr/bin/env python3
"""缠师飞书v1.0 实盘诊断脚本"""
import sys, os, glob, struct
sys.path.insert(0, '/workspace')

# ---- 持仓数据 ----
POSITIONS = [
    {'code': '300936', 'name': '英特科技', 'market': 'SZ', 'cost': 38.80, 'shares': 100},
    {'code': '002600', 'name': '领益智造', 'market': 'SZ', 'cost': 12.30, 'shares': 100},
    {'code': '301062', 'name': '上海艾录', 'market': 'SZ', 'cost': 7.88,  'shares': 100},
    {'code': '688613', 'name': '奥精医疗', 'market': 'SH', 'cost': 18.20, 'shares': 100},
    {'code': '002951', 'name': '金时科技', 'market': 'SZ', 'cost': 14.90, 'shares': 100},
    {'code': '000826', 'name': '启迪环境', 'market': 'SZ', 'cost': 3.40,  'shares': 100},
    {'code': '301128', 'name': '强瑞技术', 'market': 'SZ', 'cost': 141.30,'shares': 100},
]

# 实时价格（手动更新）
PRICES = {
    '300936': 76.00, '002600': 14.16, '301062': 8.10,
    '688613': 19.84, '002951': 15.29, '000826': 1.86, '301128': 189.05,
}

# ========================
# 缠论引擎（简化版，用于诊断）
# ========================

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, dif - dea

def find_pivots(h, l):
    tops, bots = [], []
    for i in range(1, len(h) - 1):
        if h[i] > h[i-1] and h[i] > h[i+1] and l[i] > l[i-1] and l[i] > l[i+1]:
            tops.append((i, h[i]))
        elif l[i] < l[i-1] and l[i] < l[i+1] and h[i] < h[i-1] and h[i] < h[i+1]:
            bots.append((i, l[i]))
    return tops, bots

def build_bi(tops, bots, min_gap=3):
    all_f = sorted([(i, p, 'top') for i, p in tops] + [(i, p, 'bot') for i, p in bots], key=lambda x: x[0])
    bi_list = []
    i = 0
    while i < len(all_f) - 1:
        f1, f2 = all_f[i], all_f[i+1]
        if f1[2] == f2[2]: i += 1; continue
        if abs(f2[0] - f1[0]) < min_gap: i += 1; continue
        stype = 'down' if f1[2] == 'top' else 'up'
        bi_list.append({'start': f1[0], 'end': f2[0],
                        'high': max(f1[1], f2[1]), 'low': min(f1[1], f2[1]),
                        'type': stype,
                        'start_price': f1[1], 'end_price': f2[1]})
        i += 1
    return bi_list

def find_zs(bi_list):
    if len(bi_list) < 3: return []
    zones = []
    i = 0
    while i <= len(bi_list) - 3:
        b1, b2, b3 = bi_list[i], bi_list[i+1], bi_list[i+2]
        zg = min(b1['high'], b2['high'], b3['high'])
        zd = max(b1['low'], b2['low'], b3['low'])
        if zd < zg:
            zones.append({'zg': zg, 'zd': zd, 'dir': b1['type'],
                         'start': b1['start'], 'end': b3['end'],
                         'gg': max(b1['high'], b2['high'], b3['high']),
                         'dd': min(b1['low'], b2['low'], b3['low'])})
        i += 1
    return zones

def detect_2buy(klines, zones, dif, hist):
    """2买检测"""
    results = []
    down_zs = [z for z in zones if z['dir'] == 'down']
    if len(down_zs) < 1: return results
    h, l = klines['high'].values, klines['low'].values
    for z in down_zs:
        # 找中枢后的上升笔
        for i in range(z['end']+1, min(len(klines)-2, z['end']+20)):
            s = bi_list_stroke(i, klines) if False else None
            # 简化：直接用中枢区间后K线判断
            break
    return results

def diagnose_stock(code, market, name, cost, current_price, shares):
    """诊断单只股票"""
    print(f"\n{'='*60}")
    print(f"【{code} {name}】成本={cost} 现价={current_price} 盈亏={(current_price-cost)/cost*100:+.1f}%")
    print(f"{'='*60}")

    # 尝试加载数据
    # 1. 30分CSV
    csv_pattern = f'/workspace/chanlun_system/artifacts/min30_{code}.csv'
    matches = glob.glob(csv_pattern)
    if not matches:
        csv_pattern = f'/workspace/chanlun_system/artifacts/min30_{code}.{market.lower()}.csv'
        matches = glob.glob(csv_pattern)
    
    # 2. TDX日线
    tdx_path = f'/workspace/tdx_data/{market.lower()}/lday/{market.lower()}{code}.day'
    
    df = None
    data_source = ""
    if matches:
        try:
            df = load_30min(matches[0])
            data_source = f"30分CSV ({len(df)}条)"
        except: pass
    
    if df is None and os.path.exists(tdx_path):
        try:
            df = load_tdx_day(tdx_path)
            data_source = f"TDX日线 ({len(df)}条)"
        except: pass
    
    if df is None or len(df) < 60:
        print(f"  ⚠ 无充足数据 ({data_source or '无数据'})，跳过缠论分析")
        # 只做价格诊断
        do_price_diagnosis(code, name, cost, current_price, shares)
        return

    print(f"  数据源: {data_source}")

    # 缠论分析
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    tops, bots = find_pivots(h, l)
    bi = build_bi(tops, bots, min_gap=3)
    zones = find_zs(bi)
    
    dif, dea, hist = calc_macd(df['close'])
    last_dif = float(dif.iloc[-1]) if len(dif) > 0 else 0
    last_dea = float(dea.iloc[-1]) if len(dea) > 0 else 0
    last_hist = float(hist.iloc[-1]) if len(hist) > 0 else 0
    
    print(f"\n  缠论结构:")
    print(f"    分型: {len(tops)}顶 / {len(bots)}底")
    print(f"    笔: {len(bi)}段", end="")
    if bi:
        last_bi = bi[-1]
        arrow = "↑" if last_bi['type'] == 'up' else "↓"
        print(f" (最近: {arrow} {last_bi['start_price']:.2f}→{last_bi['end_price']:.2f})")
    else:
        print()
    print(f"    中枢: {len(zones)}个", end="")
    if zones:
        last_z = zones[-1]
        print(f" (ZG={last_z['zg']:.2f} ZD={last_z['zd']:.2f} 宽={last_z['zg']-last_z['zd']:.2f})")
    else:
        print()
    
    print(f"\n  MACD状态:")
    print(f"    DIF={last_dif:+.4f} DEA={last_dea:+.4f} MACD={last_hist*2:+.4f}")
    if last_dif > 0:
        print(f"    状态: DIF在零轴上方 ↑（多头）")
    else:
        print(f"    状态: DIF在零轴下方 ↓（空头）")
    
    # 位置判断
    if zones:
        last_z = zones[-1]
        zg, zd = last_z['zg'], last_z['zd']
        pivot_width = zg - zd
        gi = None  # 默认未定义
        if current_price > zg:
            dist = current_price - zg
            gi = pivot_width / dist if dist > 0 else 999
            print(f"\n  中枢位置: 价格>ZG({zg:.2f}) 距离={dist:.2f} 引力指数={gi:.2f}")
            if gi < 0.5:
                print(f"    ⚡ 已脱离引力区（强势）")
            elif gi < 1.0:
                print(f"    → 正在脱离，留意回调")
            else:
                print(f"    引力较强，有拉回可能")
        elif current_price < zd:
            dist = zd - current_price
            gi = pivot_width / dist if dist > 0 else 999
            print(f"\n  中枢位置: 价格<ZD({zd:.2f}) 距离={dist:.2f} 引力指数={gi:.2f}")
            if gi < 0.5:
                print(f"    ↓ 已大幅脱离，看空")
            else:
                print(f"    ← 中枢下方，留意是否止跌")
        else:
            print(f"\n  中枢位置: 价格在ZG-ZD之间（震荡）")
    else:
        gi = None
    
    # 风险评估
    cost_pct = (current_price - cost) / cost * 100
    print(f"\n  风险评估:")
    
    # 检查是否在ZG-ZD范围内
    in_pivot = False
    if zones:
        last_z = zones[-1]
        in_pivot = last_z['zd'] <= current_price <= last_z['zg']
    
    if cost_pct > 50:
        print(f"    ⚠️ 盈利{+cost_pct:.1f}% 已丰厚，注意保护")
        if last_hist < 0 and last_dif < 0:
            print(f"    MACD空头+盈利丰厚，建议设移动止盈")
        elif zones and gi < 0.5:
            print(f"    已脱离中枢，可将止损提高到成本价")
    elif cost_pct > 0:
        print(f"    ✓ 盈利{+cost_pct:.1f}%，正常持有")
    else:
        print(f"    ⚠️ 亏损{cost_pct:.1f}%，需关注是否破止损")
        if current_price < cost * 0.95:
            print(f"    🔴 已接近止损位(成本×0.95={cost*0.95:.2f})！必须检查")
        elif current_price < cost * 0.97:
            print(f"    ⚠️ 接近止损位(成本×0.97={cost*0.97:.2f})，留意")
        if zones and not in_pivot and cost_pct < -10:
            print(f"    🔴 亏损+价格在ZG-ZD范围外，双重风险，建议减仓")

def do_price_diagnosis(code, name, cost, current_price, shares):
    """无缠论数据时的纯价格诊断"""
    cost_pct = (current_price - cost) / cost * 100
    unrealized = (current_price - cost) * shares
    print(f"\n  盈亏: {cost_pct:+.1f}% (未实现盈亏: {unrealized:+.0f})")
    
    if cost_pct > 50:
        print(f"  ⚠️ 盈利丰厚，强烈建议设移动止盈（TSL）")
        print(f"  建议: 将止损提高到成本价，保住利润")
    elif cost_pct > 20:
        print(f"  ✓ 已有盈利，继续持有")
    elif cost_pct > 0:
        print(f"  ✓ 小幅盈利")
    else:
        print(f"  ⚠️ 亏损中")
        sl = cost * 0.95
        if current_price < sl:
            print(f"  🔴 已破止损！成本×0.95={sl:.2f}，现价={current_price}")
        elif current_price < cost * 0.97:
            print(f"  ⚠️ 接近止损位，需密切关注")

def load_30min(path):
    import pandas as pd
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df = df[df['volume'] > 0]
    return df

def load_tdx_day(path):
    """读取TDX日线文件
    格式: <IffffffI (32字节)
    date(I) + low(f) + open(f) + close(f) + high(f) + amount(f) + volume(I)
    """
    import struct, pandas as pd
    records = []
    with open(path, 'rb') as f:
        while True:
            data = f.read(32)
            if not data or len(data) < 32: break
            vals = struct.unpack('<IffffffI', data)
            date_val = vals[0]
            if not (19000101 <= date_val <= 21001231): break
            year = date_val // 10000
            month = (date_val % 10000) // 100
            day = date_val % 100
            records.append({
                'date': f"{year:04d}-{month:02d}-{day:02d}",
                'open': vals[2],
                'close': vals[3],
                'high': vals[4],
                'low': vals[1],
                'volume': vals[6]
            })
    df = pd.DataFrame(records)
    if 'date' in df.columns and len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
    return df

# ========================
# 大盘环境检查
# ========================
def check_market():
    """检查沪指大盘环境 — 使用腾讯实时API"""
    import pandas as pd
    print(f"\n{'='*60}")
    print("【大盘环境】沪指")
    print(f"{'='*60}")
    
    import urllib.request
    url = 'http://qt.gtimg.cn/q=sh000001'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        data = resp.read().decode('gbk')
        parts = data.split('~')
        if len(parts) > 34:
            price = float(parts[3])
            yesterday = float(parts[4])
            high = float(parts[33])
            low = float(parts[34])
            pct = (price - yesterday) / yesterday * 100
            print(f"  现价: {price:.2f} ({pct:+.2f}%)  最高: {high:.2f} 最低: {low:.2f}")
    except Exception as e:
        print(f"  ⚠ 无法获取大盘数据: {e}")
        price = None

    # 尝试沪指日线（格式: <IIIIIIII，int价格×100）
    sh_tdx = '/workspace/tdx_data/sh/lday/sh000001.day'
    if os.path.exists(sh_tdx):
        try:
            recs = []
            with open(sh_tdx, 'rb') as f:
                while True:
                    d = f.read(32)
                    if not d or len(d) < 32: break
                    vals = struct.unpack('<IIIIIIII', d)
                    date_val = vals[0]
                    if not (19000101 <= date_val <= 21001231): break
                    recs.append({
                        'date': f'{date_val // 10000}-{(date_val % 10000) // 100:02d}-{date_val % 100:02d}',
                        'current': vals[1] / 100.0,
                        'open': vals[2] / 100.0,
                        'high': vals[3] / 100.0,
                        'low': vals[4] / 100.0,
                        'close': vals[5] / 100.0,
                        'vol': vals[6],
                    })
            df = pd.DataFrame(recs)
            if len(df) > 20 and df.iloc[-1]['close'] > 0:
                c = df['close'].values
                ma5 = c[-5:].mean(); ma10 = c[-10:].mean(); ma20 = c[-20:].mean()
                last = df.iloc[-1]
                print(f"  日线: {last['date']} 收盘={last['close']:.2f} 当前={last['current']:.2f}")
                print(f"  均线: MA5={ma5:.2f} MA10={ma10:.2f} MA20={ma20:.2f}")
                regime = None
                if ma5 > ma10 and last['current'] > ma5: regime = "日线上涨笔"; maxpos = "70-80%"
                elif ma5 < ma10 and last['current'] < ma5: regime = "日线下跌笔"; maxpos = "20-30%"
                else: regime = "日线震荡"; maxpos = "40-50%"
                print(f"  → 大盘: {regime} | 最大仓位: {maxpos}")
                return regime
            else:
                print(f"  ⚠ 沪指TDX数据close=0或不足，使用实时判断")
        except Exception as e:
            print(f"  ⚠ 沪指TDX读取失败: {e}")
    return "日线震荡"

# ========================
# 主程序
# ========================
if __name__ == '__main__':
    print("=" * 60)
    print("缠师飞书 v1.0 — 实盘诊断")
    print("=" * 60)
    
    regime = check_market()
    
    # 汇总
    total_cost = sum(p['cost'] * p['shares'] for p in POSITIONS)
    total_value = sum(PRICES.get(p['code'], p['cost']) * p['shares'] for p in POSITIONS)
    total_pnl = total_value - total_cost
    total_pct = total_pnl / total_cost * 100
    
    print(f"\n{'='*60}")
    print("【持仓汇总】")
    print(f"{'='*60}")
    print(f"{'代码':<8} {'名称':<8} {'成本':>6} {'现价':>6} {'盈亏%':>7} {'未实现':>8}")
    print("-" * 50)
    for p in POSITIONS:
        cp = PRICES.get(p['code'], p['cost'])
        pct = (cp - p['cost']) / p['cost'] * 100
        pnl = (cp - p['cost']) * p['shares']
        print(f"{p['code']:<8} {p['name']:<8} {p['cost']:>6.2f} {cp:>6.2f} {pct:>+7.1f}% {pnl:>+8.0f}")
    print("-" * 50)
    print(f"{'合计':<16} {total_cost:>6.0f} {total_value:>6.0f} {total_pct:>+7.1f}% {total_pnl:>+8.0f}")
    
    # 各持仓诊断
    for p in POSITIONS:
        diagnose_stock(p['code'], p['market'], p['name'], p['cost'],
                      PRICES.get(p['code'], p['cost']), p['shares'])
    
    print(f"\n{'='*60}")
    print("【综合建议】")
    print(f"{'='*60}")
    if regime:
        print(f"大盘: {regime}，建议总仓位 ≤50%")
    for p in POSITIONS:
        cp = PRICES.get(p['code'], p['cost'])
        pct = (cp - p['cost']) / p['cost'] * 100
        if pct > 50:
            print(f"  {p['name']}: 盈利{pct:.0f}%，设TSL保护利润")
        elif pct < -15:
            print(f"  {p['name']}: 亏损{pct:.0f}%，检查是否破止损")
    print()
