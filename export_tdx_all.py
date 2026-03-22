#!/usr/bin/env python3
"""通达信所有日线数据导出到CSV

通达信日线数据格式 (.day 文件):
每条记录32字节:
- 日期 (4字节 int): 格式如 20250321
- 开盘价 (4字节 int): 实际值需除以100
- 最高价 (4字节 int): 实际值需除以100
- 最低价 (4字节 int): 实际值需除以100
- 收盘价 (4字节 int): 实际值需除以100
- 成交额 (4字节 float)
- 成交量 (4字节 int)
- 保留 (4字节 int)
"""

import struct
import os
import csv
from pathlib import Path
from datetime import datetime

# 通达信数据路径
TDX_VIPDOC_PATH = r"D:\new_tdx\vipdoc"

# 输出目录
OUTPUT_DIR = r"D:\新建文件夹\claude\tdx_output"

# CSV表头
CSV_HEADER = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']


def parse_day_file(filepath):
    """解析通达信 .day 日线文件"""
    data = []
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(32)
                if len(chunk) < 32:
                    break
                # 解包: 7个数值 (i=int, f=float)
                vals = struct.unpack('iiiiiifi', chunk)

                date_val = vals[0]
                # 日期格式转换为 YYYY-MM-DD
                if date_val > 0:
                    date_str = str(date_val)
                    if len(date_str) == 8:
                        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        date_formatted = date_str
                else:
                    date_formatted = ""

                open_p = vals[1] / 100.0
                high = vals[2] / 100.0
                low = vals[3] / 100.0
                close = vals[4] / 100.0
                amount = vals[5]
                volume = vals[6]

                data.append([date_formatted, open_p, high, low, close, volume, amount])
    except Exception as e:
        print(f"  解析错误: {e}")
        return []
    return data


def to_csv(data, output_path, stock_code):
    """保存为CSV文件"""
    if not data:
        return False

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"  保存CSV失败: {e}")
        return False


def export_market(market, output_subdir, sample_dir=None):
    """导出单个市场的所有日线数据

    Args:
        market: 市场代码 ('sh' 或 'sz')
        output_subdir: 输出子目录名
        sample_dir: 样本输出目录 (保存前几个文件)
    """
    lday_path = os.path.join(TDX_VIPDOC_PATH, market, 'lday')

    if not os.path.exists(lday_path):
        print(f"路径不存在: {lday_path}")
        return 0

    day_files = [f for f in os.listdir(lday_path) if f.endswith('.day')]
    total = len(day_files)
    print(f"\n{market.upper()} 市场: 发现 {total} 个日线文件")

    success_count = 0
    error_count = 0
    sample_count = 0  # 样本文件计数

    for i, filename in enumerate(day_files, 1):
        filepath = os.path.join(lday_path, filename)
        stock_code = filename.replace('.day', '')

        # 解析数据
        data = parse_day_file(filepath)

        if data:
            # 输出到主目录
            csv_filename = f"{stock_code}.csv"
            output_path = os.path.join(OUTPUT_DIR, output_subdir, csv_filename)

            if to_csv(data, output_path, stock_code):
                success_count += 1

                # 保存前几个文件到样本目录
                if sample_dir and sample_count < 5:
                    sample_path = os.path.join(sample_dir, f"{market}_{csv_filename}")
                    to_csv(data, sample_path, stock_code)
                    sample_count += 1
            else:
                error_count += 1
        else:
            error_count += 1

        # 进度显示
        if i % 500 == 0 or i == total:
            print(f"  进度: {i}/{total} ({i*100//total}%)")

    print(f"  完成: 成功 {success_count}, 失败 {error_count}")
    return success_count


def main():
    print("=" * 60)
    print("       通达信日线数据导出工具")
    print("=" * 60)
    print(f"源路径: {TDX_VIPDOC_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 样本输出目录
    sample_dir = r"D:\新建文件夹\claude\tdx-parser-workspace\iteration-1\eval-3\without_skill\outputs"
    os.makedirs(sample_dir, exist_ok=True)
    print(f"样本目录: {sample_dir}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 统计开始时间
    start_time = datetime.now()

    # 导出上海市场
    sh_count = export_market('sh', 'sh', sample_dir)

    # 导出深圳市场
    sz_count = export_market('sz', 'sz', sample_dir)

    # 统计结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("导出完成!")
    print(f"上海市场: {sh_count} 个文件")
    print(f"深圳市场: {sz_count} 个文件")
    print(f"总计: {sh_count + sz_count} 个文件")
    print(f"耗时: {duration:.1f} 秒")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"样本文件: {sample_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
