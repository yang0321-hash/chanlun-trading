# -*- coding: utf-8 -*-
"""
通达信 .day 文件读取器
文件路径: vipdoc/sh/lday/sh600519.day (沪市)
         vipdoc/sz/lday/sz000858.day (深市)
         
文件格式(每条32字节):
  int32  date     (YYYYMMDD)
  int32  open     (×100)
  int32  high     (×100)
  int32  low      (×100)
  int32  close    (×100)
  float32 amount  (成交额,元)
  int32  volume   (成交量,手)
  int32  reserved
"""
import struct, os, sys, json
sys.stdout.reconfigure(encoding='utf-8')

TDX_BASE = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc"

def read_day_file(filepath, want_last=0):
    """读取.day文件, 返回 [[date,open,high,low,close,amount,volume], ...]
    want_last: 取最后N条, 0=全部
    """
    records = []
    with open(filepath, 'rb') as f:
        data = f.read()
    
    record_size = 32
    count = len(data) // record_size
    
    start = 0
    if want_last > 0 and want_last < count:
        start = count - want_last
    
    for i in range(start, count):
        offset = i * record_size
        rec = data[offset:offset+record_size]
        if len(rec) < 32:
            break
        date_raw, open_raw, high_raw, low_raw, close_raw, amount, volume, _ = struct.unpack('<IIIIIfII', rec)
        
        date_str = str(date_raw)
        if len(date_str) == 8:
            o = open_raw / 100.0
            h = high_raw / 100.0
            l = low_raw / 100.0
            c = close_raw / 100.0
            records.append([date_str, o, h, l, c, amount, volume])
    
    return records

def get_day_path(code, setcode):
    """获取.day文件路径
    setcode: '1'=沪市(sh), '0'=深市(sz)
    """
    market = 'sh' if setcode == '1' else 'sz'
    return os.path.join(TDX_BASE, market, 'lday', f'{market}{code}.day')

def load_stock_klines(code, setcode, want_last=80):
    """加载股票日线数据, 返回 [[date,open,high,low,close], ...]"""
    path = get_day_path(code, setcode)
    if not os.path.exists(path):
        print(f"  文件不存在: {path}")
        return []
    raw = read_day_file(path, want_last)
    # 转为引擎需要的格式
    return [[r[0], r[1], r[2], r[3], r[4]] for r in raw]


if __name__ == '__main__':
    # 测试读取600519
    data = load_stock_klines('600519', '1', 80)
    print(f"600519 贵州茅台: {len(data)} bars")
    if data:
        print(f"  最新: {data[-1]}")
        print(f"  最早: {data[0]}")
    
    # 测试000858
    data2 = load_stock_klines('000858', '0', 80)
    print(f"000858 五粮液: {len(data2)} bars")
    if data2:
        print(f"  最新: {data2[-1]}")
