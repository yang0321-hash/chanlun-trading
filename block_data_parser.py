"""
板块数据解析器 - 从通达信本地数据读取概念板块
"""

import os
import struct
import pandas as pd
from typing import Dict, List
from collections import defaultdict


class BlockParser:
    """通达信板块数据解析器"""

    def __init__(self, tdx_path: str):
        self.tdx_path = tdx_path
        self.block_path = os.path.join(tdx_path, "T0002", "blocknew")

        # 板块数据
        self.blocks = {}  # 板块名称 -> 股票列表
        self.stock_to_blocks = defaultdict(list)  # 股票代码 -> 所属板块列表

    def parse_block_files(self) -> Dict[str, List[str]]:
        """解析板块文件"""
        print("正在解析板块文件...")

        # 获取所有.blk文件
        blk_files = [f for f in os.listdir(self.block_path) if f.endswith('.blk')]

        for blk_file in blk_files:
            filepath = os.path.join(self.block_path, blk_file)

            try:
                with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
                    content = f.read()

                # 每行是一个7位数字的股票代码
                codes = []
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if len(line) >= 6:
                        code = line[:6]
                        # 判断市场
                        if code.startswith('6'):
                            full_code = f'sh{code}'
                        elif code.startswith('0') or code.startswith('3'):
                            full_code = f'sz{code}'
                        elif code.startswith('8') or code.startswith('4'):
                            full_code = f'bj{code}'
                        else:
                            continue

                        codes.append(full_code)

                block_name = blk_file.replace('.blk', '')
                if codes:
                    self.blocks[block_name] = codes
                    for code in codes:
                        self.stock_to_blocks[code].append(block_name)

            except Exception as e:
                print(f"  跳过 {blk_file}: {e}")
                continue

        print(f"解析完成: {len(self.blocks)} 个板块")
        return self.blocks

    def get_block_stocks(self, block_name: str) -> List[str]:
        """获取板块内股票列表"""
        return self.blocks.get(block_name, [])

    def get_stock_blocks(self, stock_code: str) -> List[str]:
        """获取股票所属板块列表"""
        return self.stock_to_blocks.get(stock_code, [])

    def get_block_names(self) -> List[str]:
        """获取所有板块名称"""
        return list(self.blocks.keys())

    def print_summary(self):
        """打印板块摘要"""
        print(f"\n板块摘要:")
        print(f"  总板块数: {len(self.blocks)}")

        # 按股票数量排序
        sorted_blocks = sorted(self.blocks.items(), key=lambda x: len(x[1]), reverse=True)

        print(f"\n股票数量最多的板块:")
        for name, codes in sorted_blocks[:10]:
            print(f"  {name}: {len(codes)} 只股票")


# 读取板块配置获取板块名称
def parse_block_cfg(tdx_path: str) -> Dict[str, str]:
    """解析板块配置文件，获取板块名称"""
    cfg_path = os.path.join(tdx_path, "T0002", "blocknew", "blocknew.cfg")

    block_names = {}

    try:
        with open(cfg_path, 'rb') as f:
            data = f.read()

        # 通达信板块配置文件格式：每条记录可能包含板块代码和名称
        # 尝试查找可读的板块名称
        i = 0
        while i < len(data) - 50:
            chunk = data[i:i+50]

            # 尝试GBK解码
            try:
                text = chunk.decode('gbk', errors='ignore')

                # 查找可能的板块名称 (中文字符)
                if '\u4e00' <= text <= '\u9fff':
                    # 提取前面的数字作为板块代码
                    code_match = ''
                    for j in range(max(0, i-10), i):
                        if chr(data[j]).isdigit():
                            code_match = chr(data[j]) + code_match

                    if code_match:
                        block_names[code_match] = text.strip()
            except:
                pass

            i += 1

    except Exception as e:
        print(f"解析板块配置失败: {e}")

    return block_names


def main():
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    parser = BlockParser(tdx_path)

    # 解析板块文件
    parser.parse_block_files()

    # 打印摘要
    parser.print_summary()

    # 测试查询
    print(f"\n测试查询:")

    # 查看GS板块
    gs_stocks = parser.get_block_stocks('GS')
    print(f"  GS板块: {len(gs_stocks)} 只股票")
    print(f"  前10只: {gs_stocks[:10]}")

    # 查看某只股票所属板块
    test_code = gs_stocks[0] if gs_stocks else 'sh600000'
    blocks = parser.get_stock_blocks(test_code)
    print(f"\n  {test_code} 所属板块: {blocks}")

    # 查看zxg板块 (自选股)
    zxg_stocks = parser.get_block_stocks('zxg')
    print(f"\n  zxg自选股: {len(zxg_stocks)} 只")
    if zxg_stocks:
        print(f"  前10只: {zxg_stocks[:10]}")


if __name__ == "__main__":
    main()
