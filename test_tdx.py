"""简单测试 - 扫描通达信目录"""
import sys
import os

def main():
    tdx_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc"

    print("扫描通达信目录...")
    print(f"路径: {tdx_path}")
    print()

    if not os.path.exists(tdx_path):
        print("路径不存在")
        input()
        return

    # 只扫描第一层
    print("第一层目录:")
    try:
        items = os.listdir(tdx_path)
        for item in items:
            full = os.path.join(tdx_path, item)
            if os.path.isdir(full):
                print(f"  [DIR] {item}")
            else:
                print(f"  [FILE] {item}")
    except Exception as e:
        print(f"错误: {e}")

    # 搜索002600
    print()
    print("搜索002600相关文件...")
    found = []
    for root, dirs, files in os.walk(tdx_path):
        for file in files:
            if '002600' in file or '600260' in file:
                found.append(os.path.join(root, file))

    if found:
        print(f"找到 {len(found)} 个文件:")
        for f in found[:10]:
            print(f"  {f}")
    else:
        print("未找到")

    # 尝试直接读取sz\lday目录
    print()
    print("尝试读取 sz\\lday 目录:")
    sz_lday = os.path.join(tdx_path, "sz", "lday")
    if os.path.exists(sz_lday):
        print(f"  目录存在: {sz_lday}")
        files = os.listdir(sz_lday)
        print(f"  文件数: {len(files)}")
        print(f"  前5个文件:")
        for f in files[:5]:
            print(f"    {f}")
    else:
        print(f"  目录不存在: {sz_lday}")

    print()
    print("按回车键退出...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
    input()
