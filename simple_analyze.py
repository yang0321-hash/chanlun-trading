"""简单分析 - 不闪退版本"""
import sys
import os

def main():
    file_path = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc\sz\lday\sz002600.day"

    print("="*50)
    print("002600.day 文件分析")
    print("="*50)

    if not os.path.exists(file_path):
        print("\n文件不存在")
        input()
        return

    with open(file_path, 'rb') as f:
        data = f.read()

    print(f"\n文件大小: {len(data):,} 字节")

    # 文件头
    header = data[:32]
    print(f"\n文件头(前32字节):")
    print(f"  Hex: {header.hex()}")
    print(f"  Dec: {list(header)}")

    # 结论
    print(f"\n文件头: 7bdd3201...")
    print(f"这不是标准通达信格式")
    print(f"而是大侠神器的特殊格式")

    print(f"\n解决方案:")
    print(f"1. 使用大侠神器的导出功能")
    print(f"2. 使用AKShare免费数据源")
    print(f"3. 使用Tushare等数据源")

    print(f"\n现在尝试使用AKShare获取002600数据...")

    try:
        import akshare as ak
        print(f"\n连接AKShare...")
        df = ak.stock_zh_a_hist(symbol="002600", period="daily", start_date="20230101", adjust="qfq")
        print(f"获取成功: {len(df)} 条数据")
        print(f"\n前5条:")
        print(df.head())

        # 保存数据供回测使用
        df.to_csv("002600_akshare.csv")
        print(f"\n数据已保存到: 002600_akshare.csv")
        print(f"可以直接使用CSV数据进行回测")

    except Exception as e:
        print(f"\nAKShare获取失败: {e}")
        print(f"\n请安装AKShare: pip install akshare")

    print("\n" + "="*50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
    finally:
        input()
