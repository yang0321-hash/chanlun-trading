"""
将TDX导出的CSV文件转换为JSON格式

兼容原始选股器的数据格式
"""

import os
import sys
import csv
import json
import pandas as pd
import glob
from datetime import datetime

# 确保输出编码正确
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def convert_csv_to_json(tdx_dir: str, output_dir: str = "test_output"):
    """
    将TDX导出的CSV文件转换为JSON格式

    Args:
        tdx_dir: TDX数据目录 (包含sh/, sz/, bj/ 子目录)
        output_dir: 输出目录
    """
    print("=" * 60)
    print("TDX CSV转JSON转换器")
    print(f"输入目录: {tdx_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    failed = 0

    # 处理sh目录
    sh_dir = os.path.join(tdx_dir, "sh")
    if os.path.exists(sh_dir):
        csv_files = glob.glob(os.path.join(sh_dir, "*.csv"))
        print(f"\n处理上海市场: {len(csv_files)} 个文件")

        for csv_file in csv_files:
            try:
                # 读取CSV
                df = pd.read_csv(csv_file, encoding='gbk')

                # 转换列名
                if '日期' in df.columns:
                    df = df.rename(columns={'日期': 'date'})
                if '开盘' in df.columns:
                    df = df.rename(columns={'开盘': 'open', '最高': 'high',
                                              '最低': 'low', '收盘': 'close',
                                              '成交量': 'volume', '成交额': 'amount'})

                # 添加前缀
                symbol = os.path.basename(csv_file).replace('.csv', '')

                # 确保必需列存在
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"  {symbol}: 缺少必需列")
                    failed += 1
                    continue

                # 按日期排序
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df = df.sort_values('date').reset_index(drop=True)

                # 保存为JSON
                output_file = os.path.join(output_dir, f"sh{symbol}.day.json")
                df[['date', 'open', 'high', 'low', 'close', 'amount', 'volume']].to_json(
                    output_file,
                    orient='records',
                    date_format='iso',
                    force_ascii=False
                )

                converted += 1
                if len(df) > 0:
                    latest = df.iloc[-1]
                    print(f"  {symbol} ✓ {latest['date']} 收盘¥{latest['close']:.2f}")

            except Exception as e:
                print(f"  {symbol} ✗ 转换失败: {e}")
                failed += 1

    # 处理sz目录
    sz_dir = os.path.join(tdx_dir, "sz")
    if os.path.exists(sz_dir):
        csv_files = glob.glob(os.path.join(sz_dir, "*.csv"))
        print(f"\n处理深圳市场: {len(csv_files)} 个文件")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='gbk')

                if '日期' in df.columns:
                    df = df.rename(columns={'日期': 'date'})
                df = df.rename(columns={'开盘': 'open', '最高': 'high',
                                              '最低': 'low', '收盘': 'close',
                                              '成交量': 'volume', '成交额': 'amount'})

                symbol = os.path.basename(csv_file).replace('.csv', '')

                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    failed += 1
                    continue

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df = df.sort_values('date').reset_index(drop=True)

                output_file = os.path.join(output_dir, f"sz{symbol}.day.json")
                df[['date', 'open', 'high', 'low', 'close', 'amount', 'volume']].to_json(
                    output_file,
                    orient='records',
                    date_format='iso',
                    force_ascii=False
                )

                converted += 1
                if len(df) > 0:
                    latest = df.iloc[-1]
                    print(f"  {symbol} ✓ {latest['date']} 收盘¥{latest['close']:.2f}")

            except Exception as e:
                failed += 1

    # 处理bj目录（已有JSON，检查日期）
    bj_dir = os.path.join(tdx_dir, "bj")
    if os.path.exists(bj_dir):
        json_files = glob.glob(os.path.join(bj_dir, "*.day.json"))
        print(f"\n处理北京市场: {len(json_files)} 个文件")

        for json_file in json_files[:20]:  # 只检查前20个
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data:
                    symbol = os.path.basename(json_file).replace('.day.json', '')
                    latest = data[-1]
                    print(f"  {symbol}: {latest['date']} 收盘¥{latest['close']:.2f}")

                    # 复制到test_output
                    if latest['date'] >= '2025-01-01':  # 只复制较新的数据
                        output_file = os.path.join(output_dir, f"{symbol}.day.json")
                        import shutil
                        shutil.copy2(json_file, output_file)
                        converted += 1
            except:
                pass

    print("\n" + "=" * 60)
    print(f"转换完成: 成功 {converted}, 失败 {failed}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TDX CSV转JSON转换器')
    parser.add_argument('--tdx_dir', type=str,
                       default='D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc',
                       help='TDX数据目录')
    parser.add_argument('--output', type=str, default='test_output',
                       help='输出目录')
    parser.add_argument('--limit', type=int, default=None,
                       help='转换数量限制')

    args = parser.parse_args()

    # 检查TDX目录
    if not os.path.exists(args.tdx_dir):
        print(f"错误: TDX目录不存在: {args.tdx_dir}")
        print("\n请指定正确的通达信数据目录，通常格式为:")
        print("  D:/通达信/vipdoc")
        print("  或:")
        print("  D:/大侠神器*/new_tdx(V770)/vipdoc")
        sys.exit(1)

    convert_csv_to_json(args.tdx_dir, args.output)
