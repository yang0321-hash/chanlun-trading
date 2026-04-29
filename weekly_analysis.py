#!/usr/bin/env python3
"""
周线缠论大盘分析 v1.1
使用scanner已验证的TDX读取方式
"""
import sys, os
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_system')
import numpy as np
import pandas as pd
from pathlib import Path


def read_tdx_index_fast():
    """用scanner已验证的numpy方式读取沪指日线"""
    base = Path('/workspace/tdx_data/sh/lday/sh000001.day')
    data = base.read_bytes()
    n = len(data) // 32
    arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
    dates = pd.to_datetime(arr[:, 0].astype(str), format='%Y%m%d')

    first_price_raw = float(arr[0, 1])
    if first_price_raw > 10_000_000:
        # float32格式
        prices = np.frombuffer(arr[:, 1:5].tobytes(), dtype=np.float32).reshape(n, 4)
    else:
        # int分格式
        prices = arr[:, 1:5] / 100.0

    # arr列: 0=date, 1=open, 2=close, 3=high, 4=low, 5=?, 6=vol, 7=?
    # TDX官方文档: open=1, close=2, high=3, low=4
    # 注意: scanner里high/low映射有bug，这里用正确顺序
    df = pd.DataFrame({
        'open':  prices[:, 0],  # arr[:,1]=open
        'high':  prices[:, 2],  # arr[:,3]=high
        'low':   prices[:, 3],  # arr[:,4]=low
        'close': prices[:, 1],  # arr[:,2]=close
        'volume': arr[:, 6].astype(np.int64)
    }, index=dates)
    df = df.sort_index()
    return df


def daily_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """日线 → 周线聚合"""
    df = df_daily.copy()
    df['week'] = df.index.to_period('W')
    return df.groupby('week').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index(drop=True)


def detect_bi(highs, lows, min_gap=4):
    """
    周线笔检测（贪心配对顶底分型）
    """
    n = len(highs)
    def is_top(i):
        if i <= 0 or i >= n-1: return False
        return float(highs.iloc[i]) > float(highs.iloc[i-1]) and float(highs.iloc[i]) > float(highs.iloc[i+1])
    def is_bottom(i):
        if i <= 0 or i >= n-1: return False
        return float(lows.iloc[i]) < float(lows.iloc[i-1]) and float(lows.iloc[i]) < float(lows.iloc[i+1])

    tops = [i for i in range(1, n-1) if is_top(i)]
    bottoms = [i for i in range(1, n-1) if is_bottom(i)]

    all_pts = sorted([(i, 'top') for i in tops] + [(i, 'bottom') for i in bottoms])
    strokes = []
    i = 0
    while i < len(all_pts) - 1:
        idx1, t1 = all_pts[i]
        idx2, t2 = all_pts[i+1]
        if t1 == 'top' and t2 == 'bottom':
            strokes.append((idx1, idx2, 'down'))
            i += 2
        elif t1 == 'bottom' and t2 == 'top':
            strokes.append((idx1, idx2, 'up'))
            i += 2
        else:
            i += 1

    # 合并相邻同向笔
    merged = []
    for s in strokes:
        if not merged:
            merged.append(s)
        elif s[2] == merged[-1][2]:
            merged[-1] = (merged[-1][0], s[1], s[2])
        else:
            merged.append(s)
    return merged


def detect_zs(strokes, highs, lows):
    """
    简化中枢检测（需3笔有重叠）
    """
    zhouqi = []
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i+1], strokes[i+2]
        d1, d2, d3 = s1[2], s2[2], s3[2]
        if not (d1 != d2 and d2 != d3 and d1 == d3):
            i += 1
            continue
        # 三笔的高低
        h1 = float(highs.iloc[s1[0]:s1[1]+1].max())
        l1 = float(lows.iloc[s1[0]:s1[1]+1].min())
        h2 = float(highs.iloc[s2[0]:s2[1]+1].max())
        l2 = float(lows.iloc[s2[0]:s2[1]+1].min())
        h3 = float(highs.iloc[s3[0]:s3[1]+1].max())
        l3 = float(lows.iloc[s3[0]:s3[1]+1].min())
        overlap_h = min(h1, h2, h3)
        overlap_l = max(l1, l2, l3)
        if overlap_h > overlap_l:
            zhouqi.append({
                'zg': round(overlap_h, 2),
                'zd': round(overlap_l, 2),
                'start': s1[0],
                'end': s3[1],
                'dir': d1,
                'width': round(overlap_h - overlap_l, 2)
            })
            i += 3
        else:
            i += 1
    return zhouqi


def analyze_weekly(df_w: pd.DataFrame) -> dict:
    """
    周线缠论大盘综合分析
    """
    n = len(df_w)
    if n < 20:
        return {'error': '数据不足'}

    close = df_w['close'].astype(float)
    highs = df_w['high'].astype(float)
    lows = df_w['low'].astype(float)
    vol = df_w['volume'].astype(float)

    # MA
    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()

    c = float(close.iloc[-1])
    m5 = float(ma5.iloc[-1])
    m10 = float(ma10.iloc[-1])
    m20 = float(ma20.iloc[-1])
    prev_m5 = float(ma5.iloc[-2]) if n > 1 else m5

    # 笔
    strokes = detect_bi(highs, lows)
    zs_list = detect_zs(strokes, highs, lows)
    current_zs = zs_list[-1] if zs_list else None

    # 近5周涨跌
    if n >= 6:
        c_prev = float(close.iloc[-6])
        last_5w_chg = (c - c_prev) / c_prev * 100 if c_prev > 100 else 0.0
    else:
        last_5w_chg = 0.0

    # 量能
    if n >= 10:
        vol_now = float(vol.iloc[-5:].mean())
        vol_prv = float(vol.iloc[-10:-5].mean())
        vol_chg = (vol_now - vol_prv) / vol_prv * 100 if vol_prv > 0 else 0
    else:
        vol_chg = 0

    # MA判断
    ma5_above = m5 > m10
    price_above = c > m5
    ma5_up = m5 > prev_m5

    # 近5笔方向
    recent = strokes[-5:] if len(strokes) >= 5 else strokes
    up_bi = sum(1 for s in recent if s[2] == 'up')
    dn_bi = sum(1 for s in recent if s[2] == 'down')
    bi_dir = strokes[-1][2] if strokes else 'unknown'

    # 综合得分
    score = sum([
        ma5_above, price_above, ma5_up,
        last_5w_chg > 0, vol_chg > 5
    ])

    if score >= 4:
        grade, grade_desc, cap, action = 'A', '周线强势', 0.75, '可重仓，2买/3买均可'
    elif score >= 2:
        grade, grade_desc, cap, action = 'B', '周线中性', 0.45, '谨慎，2买为主控仓'
    else:
        grade, grade_desc, cap, action = 'C', '周线弱势', 0.25, '轻仓，只做1买快出'

    # 中枢方向
    if zs_list:
        bi_type = '中枢震荡'
        cur_zs_dir = current_zs['dir'] if current_zs else 'unknown'
    elif up_bi > dn_bi:
        bi_type = '上涨延续'
        cur_zs_dir = 'up'
    else:
        bi_type = '下跌延续'
        cur_zs_dir = 'down'

    return {
        'close': round(c, 2),
        'ma5': round(m5, 2),
        'ma10': round(m10, 2),
        'ma20': round(m20, 2),
        'ma5_above': ma5_above,
        'price_above': price_above,
        'ma5_up': ma5_up,
        'last_5w_chg': round(last_5w_chg, 1),
        'vol_chg': round(vol_chg, 1),
        'bi_count': len(strokes),
        'recent_bi': f'上{up_bi}笔/下{dn_bi}笔',
        'bi_type': bi_type,
        'bi_dir': bi_dir,
        'zs_count': len(zs_list),
        'current_zs': current_zs,
        'score': score,
        'grade': grade,
        'grade_desc': grade_desc,
        'cap': cap,
        'action': action,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("  周线缠论大盘分析  (数据截止: 2026-04-14)")
    print("=" * 60)

    df_d = read_tdx_index_fast()
    print(f"\n日线数据: {len(df_d)} 天  {df_d.index[0].date()} ~ {df_d.index[-1].date()}")
    df_w = daily_to_weekly(df_d)
    print(f"周线数据: {len(df_w)} 周")

    env = analyze_weekly(df_w)

    print(f"\n📊 周线基础数据")
    print(f"  收盘: {env['close']}  |  MA5={env['ma5']}  MA10={env['ma10']}  MA20={env['ma20']}")
    ma5_s = '✓' if env['ma5_above'] else '✗'
    price_s = '✓' if env['price_above'] else '✗'
    m5_s = '↑' if env['ma5_up'] else '↓'
    print(f"  MA5{' >' if env['ma5_above'] else ' <'}MA10 [{ma5_s}]  价格{' >' if env['price_above'] else ' <'}MA5 [{price_s}]  MA5方向[{m5_s}]")

    print(f"\n📈 趋势")
    print(f"  近5周涨跌: {env['last_5w_chg']:+.1f}%")
    print(f"  量能变化: {env['vol_chg']:+.1f}%")
    print(f"  周线笔: {env['recent_bi']}  ({env['bi_type']})")

    print(f"\n🔶 缠论结构")
    print(f"  周线笔总数: {env['bi_count']}  中枢数: {env['zs_count']}")
    if env['current_zs']:
        zs = env['current_zs']
        print(f"  当前中枢: ZG={zs['zg']}  ZD={zs['zd']}  方向={zs['dir']}  宽度={zs['width']}")
    else:
        print(f"  当前中枢: 无（趋势延续）")

    print(f"\n🎯 综合判断  (得分 {env['score']}/5)")
    print(f"  评级: 【{env['grade']}级 {env['grade_desc']}】")
    print(f"  仓位上限: {env['cap']*100:.0f}%")
    print(f"  操作建议: {env['action']}")

    # 最近10周K线
    print(f"\n📅 最近10周K线:")
    print(f"{'日期':<10} {'开':>8} {'高':>8} {'低':>8} {'收':>8} {'涨跌%':>7}")
    for i in range(max(0, len(df_w)-10), len(df_w)):
        row = df_w.iloc[i]
        prev = df_w.iloc[i-1] if i > 0 else row
        chg = (float(row['close']) - float(prev['close'])) / float(prev['close']) * 100 if float(prev['close']) > 100 else 0
        week_str = str(df_w.index[i])[:10]
        print(f"{week_str:<10} {float(row['open']):>8.2f} {float(row['high']):>8.2f} "
              f"{float(row['low']):>8.2f} {float(row['close']):>8.2f} {chg:>+6.1f}%")

    # 日线当前状态（对比）
    print(f"\n📊 日线对照:")
    from chanlun_system.scanner_new_framework import _get_index_status
    idx = _get_index_status()
    above5 = '✓' if idx.get('above_ma5') else '✗'
    above10 = '✓' if idx.get('above_ma10') else '✗'
    ma5a10 = '✓' if idx.get('ma5_above_ma10') else '✗'
    print(f"  收盘={idx['close']:.0f}  MA5={idx['ma5']:.0f}  MA10={idx['ma10']:.0f}  MA20={idx['ma20']:.0f}")
    print(f"  MA5>{'✓' if idx.get('ma5_above_ma10') else '✗'}MA10[{ma5a10}]  价格{'↑' if idx.get('above_ma5') else '↓'}MA5[{above5}]")
