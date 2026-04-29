#!/usr/bin/env python3
"""
周线缠论分析模块
功能:
1. 日线 → 周线聚合
2. 周线笔/中枢/买卖点检测
3. 大盘周线环境判断
"""
import sys, os
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/chanlun_system')

import numpy as np
import pandas as pd
from chanlun.units import Market, Stock
import chanlun as cl


def daily_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """日线 → 周线聚合（按日历周）"""
    df = df_daily.copy()
    df['week'] = df.index.to_period('W')
    weekly = df.groupby('week').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    )
    return weekly.reset_index(drop=True)


def analyze_weekly(df_weekly: pd.DataFrame, code: str = 'SH000001') -> dict:
    """对周线数据进行缠论分析"""
    if len(df_weekly) < 20:
        return {'error': '数据不足', 'weeks': len(df_weekly)}

    # 用chanlun的Stock分析（指定周线频率）
    try:
        s = Stock(
            code=code,
            name=code,
            data=df_weekly,
            freq='W',  # 周线
           last_n=104  # 约2年104周
        )
        cl = ChanlunAnalysis()
        res = cl.analysis(s)
        return res
    except Exception as e:
        return {'error': str(e)}


def get_weekly_market_env(df_weekly: pd.DataFrame) -> dict:
    """
    周线大盘环境判断
    返回: {
        'trend': 'up'/'down'/'neutral',
        'ma5_above_ma10': bool,
        'price_above_ma5': bool,
        'ma5_direction': 'up'/'down'/'flat',
        'last_5_weeks_chg': float,  # 近5周涨跌幅
        'desc': str,
        '仓位上限': float
    }
    """
    if len(df_weekly) < 10:
        return {'error': '数据不足'}

    close = df_weekly['close'].astype(float)
    vol = df_weekly['volume'].astype(float)

    # 周线MA
    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()

    cur_close = float(close.iloc[-1])
    cur_ma5 = float(ma5.iloc[-1])
    cur_ma10 = float(ma10.iloc[-1])
    cur_ma20 = float(ma20.iloc[-1])
    prev_ma5 = float(ma5.iloc[-2]) if len(ma5) > 1 else cur_ma5

    # 趋势判断
    ma5_above_ma10 = cur_ma5 > cur_ma10
    price_above_ma5 = cur_close > cur_ma5
    ma5_direction = 'up' if cur_ma5 > prev_ma5 else ('down' if cur_ma5 < prev_ma5 else 'flat')

    # 近5周涨跌
    if len(close) >= 6:
        last_5_close = float(close.iloc[-6])
        last_5_chg = (cur_close - last_5_close) / last_5_close * 100
    else:
        last_5_chg = 0.0

    # 近5周成交量变化
    if len(vol) >= 6:
        avg_vol_5 = float(vol.iloc[-6:].mean())
        prev_avg_vol = float(vol.iloc[-11:-6].mean()) if len(vol) > 10 else avg_vol_5
        vol_chg = (avg_vol_5 - prev_avg_vol) / prev_avg_vol * 100 if prev_avg_vol > 0 else 0
    else:
        vol_chg = 0.0

    # 周线缠论趋势（简化：用近5笔的方向）
    # 通过高低点判断笔方向
    highs = df_weekly['high'].astype(float)
    lows = df_weekly['low'].astype(float)
    n = len(highs)
    recent_bi_directions = []
    for i in range(max(0, n-10), n-2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
            if lows.iloc[i] > lows.iloc[i-1] and lows.iloc[i] > lows.iloc[i+1]:
                recent_bi_directions.append('up')
            elif lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                recent_bi_directions.append('down')
    up_count = recent_bi_directions.count('up')
    down_count = recent_bi_directions.count('down')

    # 合成判断
    if ma5_above_ma10 and price_above_ma5 and ma5_direction == 'up':
        trend = 'up'
        cap = 0.75
        desc = '周线上涨笔(多头)'
    elif not ma5_above_ma10 and not price_above_ma5 and ma5_direction == 'down':
        trend = 'down'
        cap = 0.25
        desc = '周线下跌笔(空头)'
    elif ma5_above_ma10 and not price_above_ma5:
        trend = 'neutral'
        cap = 0.40
        desc = '周线反弹后震荡偏弱'
    elif not ma5_above_ma10 and price_above_ma5:
        trend = 'neutral'
        cap = 0.45
        desc = '周线反弹中'
    else:
        trend = 'neutral'
        cap = 0.40
        desc = '周线震荡'

    return {
        'trend': trend,
        'ma5_above_ma10': ma5_above_ma10,
        'price_above_ma5': price_above_ma5,
        'ma5_direction': ma5_direction,
        'last_5_weeks_chg': round(last_5_chg, 2),
        'vol_chg': round(vol_chg, 1),
        'cur_close': round(cur_close, 2),
        'ma5': round(cur_ma5, 2),
        'ma10': round(cur_ma10, 2),
        'ma20': round(cur_ma20, 2),
        'desc': desc,
        '仓位上限': cap,
        'recent_bi_up': up_count,
        'recent_bi_down': down_count,
    }


def get_index_weekly():
    """获取沪指周线数据 + 大盘环境"""
    base = '/workspace/tdx_data/sh/lday/sh000001.day'
    import struct

    rows = []
    with open(base, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    for i in range(n):
        b = data[i*32:(i+1)*32]
        date = struct.unpack('<I', b[0:4])[0]
        vals = struct.unpack('<6I', b[4:28])
        o, h, l, c, vol = vals[0], vals[2], vals[3], vals[4], vals[5]
        # float format check
        if 50 < o < 100000:
            rows.append({'date': date, 'open': o/100.0, 'high': h/100.0,
                         'low': l/100.0, 'close': c/100.0, 'volume': vol})
        else:
            rows.append({'date': date, 'open': float(o), 'high': float(h),
                         'low': float(l), 'close': float(c), 'volume': float(vol)})

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # 过滤到2026
    df = df[df.index <= '2026-04-27']

    df_weekly = daily_to_weekly(df)
    env = get_weekly_market_env(df_weekly)
    return df_weekly, env


if __name__ == '__main__':
    print("=== 周线大盘分析 ===\n")
    df_w, env = get_index_weekly()
    print(f"周线数据: {len(df_w)} 周")
    print(f"最后一周: {df_w.index[-1] if hasattr(df_w.index[-1], 'year') else 'N/A'}")
    print(f"最后收盘: {env.get('cur_close')}  MA5={env.get('ma5')}  MA10={env.get('ma10')}  MA20={env.get('ma20')}")
    print(f"周线趋势: {env.get('desc')}")
    print(f"MA5方向: {env.get('ma5_direction')}  {'↑' if env.get('ma5_direction')=='up' else '↓' if env.get('ma5_direction')=='down' else '→'}")
    print(f"近5周涨跌: {env.get('last_5_weeks_chg'):+.1f}%")
    print(f"量能变化: {env.get('vol_chg'):+.1f}%")
    print(f"周线笔(近10周): 上涨={env.get('recent_bi_up')}笔  下跌={env.get('recent_bi_down')}笔")
    print(f"建议仓位上限: {env.get('仓位上限')*100:.0f}%")

    # 显示最近10周K线
    print(f"\n最近10周K线:")
    print(f"{'周':<8} {'开':>8} {'高':>8} {'低':>8} {'收':>8} {'涨跌%':>7}")
    for i in range(-10, 0):
        row = df_w.iloc[i]
        prev = df_w.iloc[i-1] if i < -1 else row
        chg = (float(row['close']) - float(prev['close'])) / float(prev['close']) * 100
        week_label = str(df_w.index[i])[:10] if hasattr(df_w.index[i], 'year') else f'W{i}'
        print(f"{week_label:<8} {float(row['open']):>8.2f} {float(row['high']):>8.2f} {float(row['low']):>8.2f} {float(row['close']):>8.2f} {chg:>+6.1f}%")
