# -*- coding: utf-8 -*-
"""
最新K线自动获取与合并工具

机制:
1. 先检查vipdoc最后日期
2. 如果数据过期（>2天），输出待获取列表
3. 手动或通过cron调openclaw tdx_kline获取数据
4. 数据存入 _latest_klines.json

在scan脚本中:
  from kline_supplement import auto_supplement
  klines = auto_supplement(code, setcode, local_klines)
  
  auto_supplement会:
  - 检查_latest_klines.json是否有该股票数据
  - 如果没有，标记为"待获取"
  - 如果有，合并到local_klines
"""
import json, os
from datetime import datetime, timedelta

LATEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_latest_klines.json')
FETCH_QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_fetch_queue.json')

def load_latest():
    if os.path.exists(LATEST_FILE):
        with open(LATEST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_latest(data):
    with open(LATEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def supplement_klines(code, setcode, local_klines, latest_data=None):
    """用API获取的最新K线补充本地数据"""
    if latest_data is None:
        latest_data = load_latest()
    
    key = f"{code}.{setcode}"
    api_bars = latest_data.get(key, [])
    if not api_bars:
        return local_klines
    
    local_dates = set(k[0] for k in local_klines)
    new = []
    for bar in api_bars:
        d = str(bar[0])
        if d not in local_dates:
            new.append([d, float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])])
    
    if new:
        return local_klines + sorted(new, key=lambda x: x[0])
    return local_klines

def auto_supplement(code, setcode, local_klines):
    """自动补充：从_latest_klines.json读取并合并"""
    return supplement_klines(code, setcode, local_klines)

def mark_stale(code, setcode):
    """标记需要获取的股票"""
    queue = []
    if os.path.exists(FETCH_QUEUE_FILE):
        with open(FETCH_QUEUE_FILE, 'r', encoding='utf-8') as f:
            queue = json.load(f)
    
    key = f"{code}.{setcode}"
    if key not in queue:
        queue.append(key)
        with open(FETCH_QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, ensure_ascii=False)

def get_fetch_queue():
    """获取待获取列表"""
    if os.path.exists(FETCH_QUEUE_FILE):
        with open(FETCH_QUEUE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def clear_fetch_queue():
    """清空待获取列表"""
    if os.path.exists(FETCH_QUEUE_FILE):
        os.remove(FETCH_QUEUE_FILE)

def update_stock(code, setcode, bars):
    """更新单只股票的API数据
    bars: [[date, open, high, low, close], ...]
    """
    data = load_latest()
    key = f"{code}.{setcode}"
    data[key] = bars
    save_latest(data)
