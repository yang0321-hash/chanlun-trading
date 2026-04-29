import struct

path = '/workspace/tdx_data/sz/lday/sz000826.day'
with open(path, 'rb') as f:
    data = f.read()

print(f"总字节={len(data)}, 每条32B, 条数={len(data)//32}")

# 读最后一条
rec = data[-32:]
print(f"最后一条原始hex: {rec.hex()}")

# 尝试各种格式
formats = [
    ('<IIffffII', 'little IIfIII'),
    ('>IIffffII', 'big IIfIII'),
    ('<IIIIIIII', 'little 8int'),
    ('>IIIIIIII', 'big 8int'),
    ('<HHffffII', 'little HHffffII'),
]
for fmt, name in formats:
    try:
        vals = struct.unpack(fmt, rec)
        print(f"{name}: {vals}")
    except Exception as e:
        print(f"{name}: ERROR {e}")

# 读最后3条的date字段
print("\n--- 最后5条 date字段 ---")
for i in range(-5, 0):
    rec = data[i*32:(i+1)*32 if i != -1 else None]
    date_val = int.from_bytes(rec[:4], 'little')
    print(f"  offset{i}: date_raw=0x{date_val:08x}={date_val}")