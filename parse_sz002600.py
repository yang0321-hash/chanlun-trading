"""
直接解析sz002600的day文件
"""
import struct
import json
import os
import pandas as pd
from datetime import datetime

def parse_day_file(file_path):
    """解析.day文件"""
    records = []

    with open(file_path, 'rb') as f:
        data = f.read()

    record_size = 32
    num_records = len(data) // record_size

    print(f"文件大小: {len(data)} bytes")
    print(f"预期记录数: {num_records}")

    for i in range(num_records):
        offset = i * record_size
        if offset + record_size > len(data):
            break

        # 解析32字节记录
        date = struct.unpack('<I', data[offset:offset+4])[0]
        open_p = struct.unpack('<i', data[offset+4:offset+8])[0] / 100
        high_p = struct.unpack('<i', data[offset+8:offset+12])[0] / 100
        low_p = struct.unpack('<i', data[offset+12:offset+16])[0] / 100
        close_p = struct.unpack('<i', data[offset+16:offset+20])[0] / 100
        amount = struct.unpack('<d', data[offset+20:offset+28])[0]
        volume = struct.unpack('<I', data[offset+28:offset+32])[0]

        records.append({
            'date': f"{date:04d}",
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'amount': amount,
            'volume': volume
        })

    return records

# 解析sz002600
file_path = "D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc/sz/lday/sz002600.day"
records = parse_day_file(file_path)

print(f"成功解析 {len(records)} 条记录")
if records:
    print(f"日期范围: {records[0]['date']} ~ {records[-1]['date']}")

# 保存为JSON
os.makedirs('.claude/temp', exist_ok=True)
with open('.claude/temp/sz002600.day.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("已保存到: .claude/temp/sz002600.day.json")
