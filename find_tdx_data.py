"""通达信数据全面诊断 - 查找002600"""
import os
import sys
sys.path.insert(0, '.')


def scan_all_drives():
    """扫描所有可能的通达信路径"""
    print("="*70)
    print("         扫描通达信数据路径")
    print("="*70)

    # 用户提供的路径
    base_paths = [
        r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc",
        r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)",
        r"D:\new_tdx",
        r"D:\TdxW_HuaTai",
        r"D:\zd_zsone",
        r"D:\zd_tdx",
    ]

    # 搜索包含关键词的目录
    search_keywords = ['tdx', '通达信', 'vipdoc', 'TdxW']

    print("\n[1] 检查预设路径")
    for path in base_paths:
        if os.path.exists(path):
            print(f"  找到: {path}")
            scan_directory(path, "002600")
        else:
            print(f"  不存在: {path}")

    print("\n[2] 扫描D盘根目录")
    try:
        d_drive = []
        for item in os.listdir("D:\\"):
            full_path = os.path.join("D:\\", item)
            if os.path.isdir(full_path):
                d_drive.append(full_path.lower())

        # 查找包含关键词的目录
        for path in d_drive:
            for keyword in search_keywords:
                if keyword.lower() in path:
                    print(f"  发现: {path}")
                    scan_directory(path, "002600")
                    break
    except Exception as e:
        print(f"  扫描失败: {e}")

    print("\n[3] 搜索 .day 和 .min 文件")
    # 直接搜索包含002600的文件
    search_files("D:\\", "002600")


def scan_directory(base_path: str, target_code: str):
    """扫描目录查找目标股票文件"""
    try:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if target_code in file:
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    ext = os.path.splitext(file)[1]
                    kline_count = size // 32 if ext in ['.day', '.min'] else 0

                    print(f"\n    找到文件: {file}")
                    print(f"      路径: {root}")
                    print(f"      大小: {size} 字节")
                    if kline_count > 0:
                        print(f"      K线: {kline_count} 条")

                    # 尝试读取
                    test_read_file(full_path)

    except Exception as e:
        pass


def search_files(root_path: str, target: str):
    """递归搜索包含目标的文件"""
    try:
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if target in file or target.replace('0', '') in file:
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    ext = os.path.splitext(file)[1]

                    if ext in ['.day', '.min', '.lc1', '.lc5']:
                        kline_count = size // 32
                        print(f"\n  找到: {file}")
                        print(f    路径: {root}")
                        print(f    大小: {size} 字节, {kline_count} 条K线)

                        test_read_file(full_path)
    except Exception as e:
        pass


def test_read_file(file_path: str):
    """测试读取文件"""
    import struct

    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        record_size = 32
        num_records = len(data) // record_size

        if num_records > 0:
            # 读取最后一条记录
            offset = (num_records - 1) * record_size
            date_val, open_p, high_p, low_p, close_p, amount, vol, _ = \
                struct.unpack('IIIIIIFI', data[offset:offset + record_size])

            print(f"    最新: 日期={date_val}, 收盘={close_p/100:.2f}")
    except Exception as e:
        print(f"    读取失败: {e}")


def list_sample_files(base_path: str):
    """列出样本文件了解数据格式"""
    print(f"\n[4] 列出样本文件")

    if not os.path.exists(base_path):
        return

    count = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.day') or file.endswith('.min'):
                if count < 10:
                    print(f"  {os.path.join(root, file)}")
                    count += 1
                else:
                    print(f"  ... 还有更多文件")
                    return


if __name__ == "__main__":
    scan_all_drives()

    print("\n" + "="*70)
    print("如果找到文件，请复制文件路径，我来修改代码")
    print("="*70)

    input("\n按回车键退出...")
