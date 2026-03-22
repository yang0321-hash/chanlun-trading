"""扫描通达信目录结构"""
import sys
sys.path.insert(0, '.')

import os

def scan_tdx_structure():
    """扫描通达信目录结构"""
    tdx_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc"

    print("="*70)
    print("         通达信目录结构扫描")
    print("="*70)

    if not os.path.exists(tdx_path):
        print(f"\n路径不存在: {tdx_path}")
        return

    print(f"\n扫描路径: {tdx_path}")
    print("\n目录结构:")
    print("-"*70)

    # 递归扫描显示结构
    print_structure(tdx_path, "", 3)


def print_structure(path: str, prefix: str, max_depth: int):
    """打印目录结构"""
    try:
        items = os.listdir(path)
    except Exception as e:
        print(f"{prefix}[无法访问: {e}]")
        return

    # 分类：目录和文件
    dirs = []
    files = []

    for item in items:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            dirs.append(item)
        else:
            files.append(item)

        # 显示目录
        for d in sorted(dirs):
            full_path = os.path.join(path, d)
            print(f"{prefix}📁 {d}/")

            # 递归显示子目录
            if max_depth > 0:
                new_prefix = prefix + "    "
                print_structure(full_path, new_prefix, max_depth - 1)

        # 显示文件（只显示前几个）
        for f in sorted(files):
            full_path = os.path.join(path, f)
            size = os.path.getsize(full_path)

            # 判断文件类型
            if f.endswith('.day'):
                klines = size // 32
                print(f"{prefix}📄 {f} ({klines}条K线)")
            elif f.endswith('.min'):
                klines = size // 32
                print(f"{prefix}📄 {f} ({klines}条K线)")
            elif f.endswith('.lc1') or f.endswith('.lc5'):
                klines = size // 32
                print(f"{prefix}📄 {f} ({klines}条K线)")
            else:
                print(f"{prefix}📄 {f} ({size:,}字节)")

        # 如果文件太多，显示省略信息
        if len(files) > 10:
            remaining = len(files) - 10
            print(f"{prefix}    ... 还有 {remaining} 个文件")

    except Exception as e:
        print(f"{prefix}[错误: {e}]")


def find_002600_files():
    """查找002600相关文件"""
    tdx_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc"

    print("\n" + "="*70)
    print("         查找 002600 数据文件")
    print("="*70)

    search_codes = ['002600', 'sz002600', '600260']

    found_files = []

    for root, dirs, files in os.walk(tdx_path):
        for file in files:
            file_lower = file.lower()
            for code in search_codes:
                if code in file_lower:
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    found_files.append({
                        'file': file,
                        'path': root,
                        'full_path': full_path,
                        'size': size
                    })

    if found_files:
        print(f"\n找到 {len(found_files)} 个相关文件:\n")
        for i, f in enumerate(found_files, 1):
            print(f"{i}. {f['file']}")
            print(f"   路径: {f['path']}")
            print(f"   大小: {f['size']:,} 字节")

            # 分析文件内容
            analyze_file(f['full_path'])
    else:
        print("\n未找到 002600 相关文件")
        print("\n显示所有.day文件目录:")
        list_day_dirs(tdx_path)


def analyze_file(file_path: str):
    """分析数据文件"""
    import struct

    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        record_size = 32
        num_records = len(data) // record_size

        if num_records > 0:
            # 读取第一条和最后一条
            first_date, *_ = struct.unpack('I', data[:4])
            last_date, *_ = struct.unpack('I', data[-32:-28])

            print(f"   记录数: {num_records}")
            print(f"   起始日期: {first_date}")
            print(f"   结束日期: {last_date}")
    except Exception as e:
        print(f"   分析失败: {e}")


def list_day_dirs(base_path: str):
    """列出所有包含.day文件的目录"""
    day_dirs = []

    for root, dirs, files in os.walk(base_path):
        if any(f.endswith('.day') for f in files):
            rel_path = os.path.relpath(root, base_path)
            day_count = len([f for f in files if f.endswith('.day')])
            day_dirs.append((rel_path, day_count))

    if day_dirs:
        day_dirs.sort()
        print(f"\n发现 {len(day_dirs)} 个包含日线数据的目录:\n")
        for path, count in day_dirs[:10]:
            print(f"  {path} ({count}个文件)")
        if len(day_dirs) > 10:
            print(f"  ... 还有 {len(day_dirs)-10} 个目录")
    else:
        print("\n未找到 .day 文件目录")


if __name__ == "__main__":
    scan_tdx_structure()
    find_002600_files()

    print("\n" + "="*70)
    print("请查看上面的目录结构，确认后告诉我")
    print("="*70)

    input("\n按回车键退出...")
