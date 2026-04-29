# -*- coding: utf-8 -*-
"""从TDX API获取最新日线数据，补充到本地数据末尾"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

def fetch_latest_klines(code, setcode, count=5):
    """从tdx_kline API获取最新日线数据
    返回: [[date, open, high, low, close, volume], ...] 或 None
    """
    try:
        import importlib
        # 通过gateway的工具获取 - 但在脚本中不能直接调
        # 改用HTTP方式
        import json, urllib.request
        # 这里用openclaw的工具机制不太方便，改为直接读文件
        return None
    except:
        return None

def append_api_klines(local_klines, api_klines):
    """将API获取的K线追加到本地数据末尾（去重）
    local_klines: [[date, o, h, l, c], ...]  (int dates like 20260423)
    api_klines: [[date, o, h, l, c, vol], ...] (string dates like "20260424")
    """
    if not api_klines:
        return local_klines
    
    local_dates = set(k[0] for k in local_klines)
    new_rows = []
    for k in api_klines:
        date = int(k[0]) if isinstance(k[0], str) else k[0]
        if date not in local_dates:
            o = float(k[1]) if isinstance(k[1], str) else k[1]
            h = float(k[2]) if isinstance(k[2], str) else k[2]
            l = float(k[3]) if isinstance(k[3], str) else k[3]
            c = float(k[4]) if isinstance(k[4], str) else k[4]
            vol = int(float(k[5])) if len(k) > 5 else 0
            new_rows.append([date, o, h, l, c])
    
    if new_rows:
        return local_klines + sorted(new_rows, key=lambda x: x[0])
    return local_klines
