"""
大盘环境判断模块

自动识别沪指当前状态，返回:
  - regime: 'bull' | 'sideways' | 'bear'
  - ma5, ma10, ma20, close: 最新值
  - action: 操作建议
  - param_set: 推荐的缠论参数组 ('C' | 'A' | 'B')
"""

import numpy as np

def get_market_regime(sh000001_data=None):
    """
    计算沪指当前状态

    sh000001_data: DataFrame with columns [date, open, close, high, low, volume]
                   If None, will try to load from cache or tushare
    返回:
      dict with regime, prices, signals, action, param_set
    """
    if sh000001_data is None:
        sh000001_data = _load_index_data()

    if sh000001_data is None or len(sh000001_data) < 60:
        return _fallback_regime()

    df = sh000001_data.tail(60).copy()
    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    n = len(closes)

    # MA
    ma5 = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    ma20 = np.mean(closes[-20:])
    latest_close = closes[-1]
    prev_close = closes[-2] if n >= 2 else closes[-1]

    # 前5日大致方向（用线性回归斜率）
    x = np.arange(5)
    if n >= 5:
        slope = np.polyfit(x, closes[-5:], 1)[0]
    else:
        slope = 0

    # 今日涨跌
    chg_pct = (latest_close - prev_close) / prev_close * 100 if prev_close > 0 else 0

    # ── 判断规则 ──────────────────────────────────────────
    # 1. 趋势方向: MA5 vs MA10
    if ma5 > ma10:
        trend = '上涨笔'
    elif ma5 < ma10:
        trend = '下跌笔'
    else:
        trend = '震荡'

    # 2. 位置关系
    if latest_close > ma5:
        price_pos = 'MA5上方'
    elif latest_close > ma20:
        price_pos = 'MA5下方/MA20上方'
    elif latest_close > ma10:
        price_pos = 'MA20下方/MA10上方'
    else:
        price_pos = 'MA10下方（弱势）'

    # 3. MA多空排列
    if ma5 > ma10 > ma20:
        ma排列 = '多头排列'
    elif ma5 < ma10 < ma20:
        ma排列 = '空头排列'
    else:
        ma排列 = '纠缠'

    # ── Regime 分类 ─────────────────────────────────────
    if ma5 > ma10 and latest_close > ma5 and slope > 0:
        regime = 'bull'
        param_set = 'C'   # +5日动量≥3%
        action = '重仓参与，选强势股回调'
        max_pos = 70
    elif ma5 < ma10 and latest_close < ma5:
        regime = 'bear'
        param_set = 'B'   # 只做1B轻仓
        action = '轻仓快出/暂停，防御为主'
        max_pos = 20
    else:
        # 震荡: MA5和MA10缠绕，方向不明
        regime = 'sideways'
        param_set = 'A'   # v3b原始，不过滤
        action = '半仓操作，高抛低吸'
        max_pos = 40

    # 5日动量（大致判断）
    mom5d = (closes[-1] - closes[-6]) / closes[-6] * 100 if n >= 6 else 0

    # 成交额趋势（如果有amount）
    vol_trend = ''
    if 'amount' in df.columns and len(df) >= 5:
        vol5_avg = df['amount'].iloc[-5:].mean()
        vol10_avg = df['amount'].iloc[-10:].mean()
        if vol5_avg > vol10_avg * 1.1:
            vol_trend = '放量'
        elif vol5_avg < vol10_avg * 0.9:
            vol_trend = '缩量'
        else:
            vol_trend = '平量'

    return {
        'regime': regime,
        'trend': trend,
        'ma5': round(ma5, 2),
        'ma10': round(ma10, 2),
        'ma20': round(ma20, 2),
        'close': round(latest_close, 2),
        'chg_pct': round(chg_pct, 2),
        'mom5d': round(mom5d, 2),
        'slope': round(slope, 3),
        'price_pos': price_pos,
        'ma_arrange': ma排列,
        'vol_trend': vol_trend,
        'action': action,
        'param_set': param_set,
        'max_pos': max_pos,  # 最大仓位建议%
    }


def _load_index_data():
    """加载沪指数据，优先腾讯行情"""
    import pickle, os
    cache_file = '/tmp/sh000001_daily_cache.pkl'

    # 缓存1小时内有效
    if os.path.exists(cache_file):
        import time
        if time.time() - os.path.getmtime(cache_file) < 3600:
            try:
                return pickle.load(open(cache_file, 'rb'))
            except:
                pass

    data = _fetch_tencent_index()
    if data is not None:
        try:
            pickle.dump(data, open(cache_file, 'wb'))
        except:
            pass
    return data


def _fetch_tencent_index():
    """从腾讯行情获取沪指日线 (近120天)"""
    import requests
    try:
        url = 'https://web.ifzq.gtimg.cn/appstock/app/kline/kline?_var=kline_day&param=sh000001,day,,,120'
        r = requests.get(url, timeout=8)
        text = r.text
        # 格式: kline_day={"code":0,"data":{"sh000001":{"day":[["date","open","close","high","low","vol"],...]}}}
        import json
        # 去掉变量名前缀
        if text.startswith('kline_day='):
            text = text[len('kline_day='):]
        obj = json.loads(text)
        days = obj['data']['sh000001']['day']
        import pandas as pd
        df = pd.DataFrame(days, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        for c in ['open', 'close', 'high', 'low', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        df = df.sort_values('date').reset_index(drop=True)
        return df[['date', 'open', 'close', 'high', 'low', 'volume']]
    except:
        return None


def pd_to_datetime(date_series):
    """兼容处理pandas日期转换"""
    try:
        return pd.to_datetime(date_series, format='%Y%m%d')
    except:
        return pd.to_datetime(date_series)


def _fallback_regime():
    """数据不足时的fallback"""
    return {
        'regime': 'sideways',
        'trend': '未知',
        'ma5': 0, 'ma10': 0, 'ma20': 0, 'close': 0,
        'chg_pct': 0, 'mom5d': 0, 'slope': 0,
        'price_pos': '数据不足',
        'ma_arrange': '数据不足',
        'vol_trend': '',
        'action': '数据不足，请手动确认大盘',
        'param_set': 'A',
        'max_pos': 30,
    }


def format_regime_report(r):
    """格式化输出大盘状态报告"""
    param_labels = {
        'A': 'v3b原始（无过滤，信号多）',
        'B': 'v3b+1B轻仓（防御模式）',
        'C': 'v3b+5日动量≥3%（强势选股）',
    }

    regime_labels = {
        'bull': '🟢 牛市/上涨笔',
        'sideways': '🟡 震荡市',
        'bear': '🔴 熊市/下跌笔',
    }

    report = f"""
═══════════════════════════════════════════════════════
【大盘环境】沪指
═══════════════════════════════════════════════════════
  收盘价: {r['close']}  涨跌: {r['chg_pct']:+.2f}%
  MA5:  {r['ma5']}  MA10: {r['ma10']}  MA20: {r['ma20']}
  趋势: {r['trend']}  |  {r['price_pos']}  |  {r['ma_arrange']}
  5日动量: {r['mom5d']:+.2f}%  |  成交: {r['vol_trend']}
───────────────────────────────────────────────────────
  大盘状态: {regime_labels.get(r['regime'], r['regime'])}
  操作建议: {r['action']}
  最大仓位: {r['max_pos']}%
───────────────────────────────────────────────────────
  推荐参数: {param_labels.get(r['param_set'], r['param_set'])}
═══════════════════════════════════════════════════════
"""
    return report


if __name__ == '__main__':
    r = get_market_regime()
    print(format_regime_report(r))
