"""分析 002600.day 文件格式"""
import os

file_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc\sz\lday\sz002600.day"

print("分析文件格式...")

with open(file_path, 'rb') as f:
    data = f.read()

print(f"文件大小: {len(data)} 字节")
print(f"记录数(假设32字节): {len(data)//32}")
print(f"记录数(假设64字节): {len(data)//64}")

# 检查前200字节
print("\n前200字节(hex):")
print(data[:200].hex())

print("\n前200字节(ascii尝试):")
try:
    print(data[:200])
except:
    pass

# 尝试解析为32字节记录
print("\n" + "="*50)
print("尝试解析为32字节记录...")
print("="*50)

# 方法1: 标准通达信格式
print("\n方法1: 32字节标准格式")
try:
    for i in range(min(3, len(data)//32)):
        offset = i * 32
        chunk = data[offset:offset+32]
        # 尝试多种解析方式
        print(f"\n记录 {i+1} (原始hex): {chunk.hex()}")

        # 作为8个int解析
        vals = struct.unpack('I' * 8, chunk)
        print(f"  8个int: {vals}")

        # 检查日期是否合理
        if 19900101 <= vals[0] <= 20991231:
            print(f"  日期: {vals[0]} (有效)")
        if vals[1] > 0 and vals[1] < 1000000:  # 开价(可能需要除100)
            print(f"  开: {vals[1]}/{vals[1]/100:.2f}")

except Exception as e:
    print(f"  解析失败: {e}")

# 方法2: 浮点数格式
print("\n" + "="*50)
print("方法2: 浮点数格式")
try:
    for i in range(1):
        chunk = data[i*32:(i+1)*32]
        vals = struct.unpack('f' * 8, chunk)
        print(f"记录1: {vals}")
        break
except Exception as e:
    print(f"  解析失败: {e}")

# 方法3: 64字节记录
print("\n" + "="*50)
print("方法3: 64字节记录格式")
try:
    record_size = 64
    for i in range(min(2, len(data)//record_size)):
        offset = i * record_size
        chunk = data[offset:offset+record_size]
        print(f"\n记录 {i+1} (前64字节): {chunk[:32].hex()}...")

        # 试试8个double
        vals = struct.unpack('d' * 8, chunk)
        print(f"  8个double: {vals}")

        # 试试16个int
        vals = struct.unpack('I' * 16, chunk)
        print(f"  16个int: {vals}")

except Exception as e:
    print(f"  解析失败: {e}")

input("\n按回车键退出...")
