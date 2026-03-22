"""
缠论选股器演示脚本
"""

from scanner.chanlun_screener import ChanLunScreener, print_scan_result
import pandas as pd
import json
import os

def demo_local_scan():
    """演示本地数据扫描"""
    print("=" * 60)
    print("缠论选股器 - 演示")
    print("=" * 60)

    # 初始化选股器
    screener = ChanLunScreener(
        use_macd=True,    # 使用MACD背驰判断
        min_klines=60     # 最少需要60根K线
    )

    # 检查本地数据目录
    data_dir = "test_output"

    if not os.path.exists(data_dir):
        print(f"\n[错误] 数据目录不存在: {data_dir}")
        print("\n请先运行以下命令获取TDX数据：")
        print("  node .claude/skills/tdx-parser/scripts/parse_tdx.js --all")
        return

    # 扫描前100个文件作为演示
    import glob
    files = glob.glob(os.path.join(data_dir, "*.json"))[:100]

    print(f"\n扫描 {len(files)} 个文件...\n")

    result = screener.scan_local_files(data_dir, "*.json")

    # 打印结果
    print_scan_result(result)

    # 保存结果
    if result.total_signals > 0:
        with open("signals.txt", "w", encoding="utf-8") as f:
            f.write("# 缠论选股信号\n")
            f.write(f"# 扫描时间: {result.scan_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 扫描数量: {result.total_scanned}\n")
            f.write(f"# 信号数量: {result.total_signals}\n\n")

            for signal in result.get_all_signals():
                f.write(f"{signal.symbol},{signal.name},{signal.signal_type},")
                f.write(f"{signal.price:.2f},{signal.confidence:.0%},{signal.reason}\n")

        print(f"\n结果已保存到: signals.txt")

if __name__ == '__main__':
    demo_local_scan()
