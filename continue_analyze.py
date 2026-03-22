"""继续分析 002600.day 文件格式"""
import os
import struct

file_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc\sz\lday\sz002600.day"

print("继续分析...")

with open(file_path, 'rb') as f:
    data = f.read()

print(f"文件大小: {len(data)} 字节")
print(f"每条记录: {len(data)//3308} 字节 (如果3308条记录)")

# 检查是否有规律
print("\n分析文件结构...")

# 文件头
print(f"\n文件头(前32字节):")
print(f"  Hex: {data[:32].hex()}")
print(f"  Dec: {list(data[:32])}")

# 尝试找到记录的分隔
print(f"\n搜索可能的日期格式...")

# 搜索类似2024, 2025的数字
import re
dates_found = []
for i in range(0, len(data)-4, 4):
    val = int.from_bytes(data[i:i+4], 'little')
    if 20200101 <= val <= 20991231:
        dates_found.append((i, val))

print(f"找到 {len(dates_found)} 个可能是日期的值:")
for pos, val in dates_found[:10]:
    print(f"  位置{pos}: {val}")

# 查看文件中间部分
print(f"\n文件中间(字节100-200):")
mid = data[100:200]
print(f"  Hex: {mid.hex()}")

# 检查是否是某种压缩或编码
print(f"\n文件特征分析:"")
print(f"  总大小: {len(data)} 字节")
print(f"  可能的记录数: {len(data)//32}, {len(data)//64}, {len(data)//128}")

# 检查特定字节
print(f"\n检查特殊字节:")
if data[:2] == b'7bdd':
    print(f"  文件头 7bdd - 可能是某种自定义格式")
    print(f"  这是大侠神器的特殊格式，需要特定的解析方式")

# 查看深圳市场的另一个文件对比
other_file = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc\sz\lday\sz000001.day"

print(f"\n对比 sz000001.day:")
if os.path.exists(other_file):
    with open(other_file, 'rb') as f:
        other_data = f.read()
    print(f"  大小: {len(other_data)} 字节")
    print(f"  文件头: {other_data[:32].hex()}")
    print(f"  002600头: {data[:32].hex()}")
    print(f"  是否相同: {data[:32] == other_data[:32]}")
else:
    print(f"  文件不存在: {other_file}")

print("\n" + "="*50)
print("结论:")
print("文件头为 '7bdd' 不是标准通达信格式")
print("这可能是大侠神器的特殊编码格式")
print("需要大侠神器的解析接口或文档")
print("="*50)

input("\n按回车键退出...")
