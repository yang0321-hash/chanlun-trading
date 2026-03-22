"""
检查市场最高连板数的分布
"""
import os
import struct
import pandas as pd

def is_valid_stock(code: str) -> bool:
    if code.startswith('sh6') and not code.startswith('sh688'): return True
    if code.startswith('sh688'): return True
    if code.startswith('sz0'): return True
    if code.startswith('sz3'): return True
    return False

def get_sector(code: str) -> str:
    if code.startswith('sh688'): return '科创板'
    if code.startswith('sz300'): return '创业板'
    if code.startswith('sh60'): return '沪主板'
    if code.startswith('sz0'): return '深主板'
    return '其他'

def calc_max_consecutive(df, idx):
    cons = 0
    for i in range(idx, -1, -1):
        row = df.iloc[i]
        if row['open'] > 0 and (row['close'] - row['open']) / row['open'] >= 0.095:
            cons += 1
        else:
            break
    return cons

def read_day_file(filepath):
    data = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(32)
            if len(chunk) < 32: break
            v = struct.unpack('IIIIIfII', chunk)
            try:
                date = pd.to_datetime(str(v[0]), format='%Y%m%d')
                open_p, high, low, close = v[1]/100, v[2]/100, v[3]/100, v[4]/100
                if 0 < close < 200:
                    data.append({'date': date, 'open': open_p, 'high': high, 'low': low, 'close': close})
            except: pass
    if data:
        return pd.DataFrame(data).set_index('date')
    return pd.DataFrame()

tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"
vipdoc_path = os.path.join(tdx_path, 'vipdoc')

print("正在加载数据...")
stock_data = {}
for market in ['sh', 'sz']:
    lday_path = os.path.join(vipdoc_path, market, 'lday')
    if not os.path.exists(lday_path): continue
    for f in os.listdir(lday_path):
        if f.endswith('.day'):
            full_code = f.replace('.day', '')
            if is_valid_stock(full_code):
                try:
                    df = read_day_file(os.path.join(lday_path, f))
                    if len(df) > 50:
                        stock_data[full_code] = df
                except: pass

print(f"已加载 {len(stock_data)} 只股票")

# 获取所有日期
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index)
all_dates = sorted(list(all_dates))[-30:]

print(f"\n最近30天的最高连板数:")
print("-" * 50)

for date in all_dates:
    max_cons = 0
    max_code = ''
    for code, df in stock_data.items():
        if date in df.index:
            idx = df.index.get_loc(date)
            cons = calc_max_consecutive(df, idx)
            if cons > max_cons:
                max_cons = cons
                max_code = code

    status = "✓" if max_cons >= 4 else " "
    print(f"{date.strftime('%Y-%m-%d')} {status}  最高连板: {max_cons}天 ({max_code})")

# 统计
ge_4 = sum(1 for d in all_dates if any(
    calc_max_consecutive(df, df.index.get_loc(d)) >= 4
    for df in stock_data.values() if d in df.index
))
print(f"\n达到4天及以上连板的天数: {ge_4}/{len(all_dates)} ({ge_4/len(all_dates)*100:.1f}%)")
