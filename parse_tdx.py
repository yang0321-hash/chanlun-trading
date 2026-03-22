import struct
import os
import csv

# 设置基础路径
BASE_PATH = r'D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc'

def parse_day_file(filepath):
    """解析通达信 .day 日线文件
    格式: 每条记录32字节
    - 日期 (4字节 int): 格式如 20250321
    - 开盘价 (4字节 int): 实际值需除以100
    - 最高价 (4字节 int): 实际值需除以100
    - 最低价 (4字节 int): 实际值需除以100
    - 收盘价 (4字节 int): 实际值需除以100
    - 成交额 (4字节 float)
    - 成交量 (4字节 int)
    - 保留 (4字节 int)
    """
    data = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(32)
            if len(chunk) < 32:
                break
            vals = struct.unpack('iiiiiifi', chunk)
            date_str = str(vals[0])
            open_p = vals[1] / 100.0
            high = vals[2] / 100.0
            low = vals[3] / 100.0
            close = vals[4] / 100.0
            amount = vals[5]
            volume = vals[6]
            data.append([date_str, open_p, high, low, close, volume, amount])
    return data

def parse_min_file(filepath):
    """解析通达信分钟线文件 (.lc1, .lc5等)
    格式: 每条记录32字节
    - 日期时间 (4字节 int): 格式如 202503211030 (YYYYMMDDHHMM)
    - 开盘价 (4字节 int): 实际值需除以100
    - 最高价 (4字节 int): 实际值需除以100
    - 最低价 (4字节 int): 实际值需除以100
    - 收盘价 (4字节 int): 实际值需除以100
    - 成交额 (4字节 float)
    - 成交量 (4字节 int)
    - 保留 (4字节 int)
    """
    data = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(32)
            if len(chunk) < 32:
                break
            vals = struct.unpack('iiiiiifi', chunk)
            dt_str = str(vals[0])
            open_p = vals[1] / 100.0
            high = vals[2] / 100.0
            low = vals[3] / 100.0
            close = vals[4] / 100.0
            amount = vals[5]
            volume = vals[6]
            data.append([dt_str, open_p, high, low, close, volume, amount])
    return data

def to_csv(data, output_path, header=['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']):
    """保存为CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f'已保存: {output_path}, 共{len(data)}条记录')

def list_all_day_files():
    """列出所有day文件"""
    markets = ['sh', 'sz']
    for market in markets:
        lday_path = os.path.join(BASE_PATH, market, 'lday')
        if os.path.exists(lday_path):
            files = [f for f in os.listdir(lday_path) if f.endswith('.day')]
            print(f'{market}/lday: {len(files)} 个文件')
            for f in files[:10]:
                print(f'  - {f}')
            if len(files) > 10:
                print(f'  ... 还有 {len(files)-10} 个文件')

if __name__ == '__main__':
    print("=== 通达信数据解析器 ===\n")

    # 列出所有文件
    print("1. 扫描日线文件...")
    list_all_day_files()

    # 解析示例文件
    print("\n2. 解析示例文件...")

    # 上证指数
    sh000001 = os.path.join(BASE_PATH, 'sh', 'lday', 'sh000001.day')
    if os.path.exists(sh000001):
        data = parse_day_file(sh000001)
        print(f"\n上证指数(sh000001): {len(data)}条记录")
        print("最新5条:")
        for row in data[-5:]:
            print(f"  {row}")
        # 保存
        to_csv(data, 'D:/新建文件夹/claude/sh000001.csv')

    # 平安银行
    sz000001 = os.path.join(BASE_PATH, 'sz', 'lday', 'sz000001.day')
    if os.path.exists(sz000001):
        data = parse_day_file(sz000001)
        print(f"\n平安银行(sz000001): {len(data)}条记录")
        print("最新5条:")
        for row in data[-5:]:
            print(f"  {row}")
        # 保存
        to_csv(data, 'D:/新建文件夹/claude/sz000001.csv')

    print("\n完成! CSV文件已保存到当前目录")
