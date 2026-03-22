"""测试读取 002600 数据文件 - 多种格式"""
import struct
import os
import array

file_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc\sz\lday\sz002600.day"

print("测试读取 002600.day 文件")
print(f"文件: {file_path}")

if not os.path.exists(file_path):
    print("文件不存在")
else:
    size = os.path.getsize(file_path)
    print(f"文件大小: {size} 字节 = {size // 32} 条记录(如果32字节)")

    # 尝试读取
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        print(f"读取数据: {len(data)} 字节")

        # 尝试不同的格式
        formats = [
            ('IIIIIIFI', 32, '标准格式'),
            ('iiiiiifi', 32, '小写格式'),
            ('IIIIIIII', 32, '8个int'),
            ('ffffffff', 32, '8个float'),
            ('I'*8, 32, '8个int(简化)'),
        ]

        for fmt, size, desc in formats:
            try:
                print(f"\n尝试格式: {fmt} ({desc})")
                record_size = struct.calcsize(fmt)
                num_records = len(data) // record_size
                print(f"  记录大小: {record_size}, 记录数: {num_records}")

                if num_records > 0:
                    # 读取第一条
                    values = struct.unpack(fmt, data[:record_size])
                    print(f"  第一条: {values[:5]}")
                    break
            except Exception as e:
                print(f"  失败: {e}")

        # 检查文件前100字节
        print("\n文件前100字节(hex):")
        print(data[:100].hex())

        print("\n文件前100字节(十进制):")
        for i in range(0, min(100, len(data)), 4):
            chunk = data[i:i+4]
            print(f"  {i//4}: {chunk.hex()}")

    except Exception as e:
        print(f"错误: {e}")

input("\n按回车键退出...")
