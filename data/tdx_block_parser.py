"""
通达信板块数据解析器

解析通达信板块配置和成分股文件
"""

import os
import struct
from typing import Dict, List
from collections import defaultdict
from loguru import logger


class TDXBlockParser:
    """
    通达信板块数据解析器

    功能：
    1. 解析板块配置文件(blocknew.cfg)获取板块名称
    2. 解析板块成分股文件(.blk)获取股票列表
    3. 建立股票与板块的映射关系
    """

    def __init__(self, tdx_path: str):
        """
        初始化解析器

        Args:
            tdx_path: 通达信根目录
        """
        self.tdx_path = tdx_path
        self.blocknew_path = os.path.join(tdx_path, "T0002", "blocknew")

        # 板块数据
        self.blocks = {}  # 板块代码 -> 板块信息
        self.stock_to_blocks = defaultdict(list)  # 股票代码 -> 板块列表

        # 缓存的板块名称
        self._block_names: Dict[str, str] = {}

    def parse_block_config(self) -> Dict[str, str]:
        """
        解析板块配置文件，获取板块名称

        Returns:
            {板块代码: 板块名称}
        """
        cfg_path = os.path.join(self.blocknew_path, "blocknew.cfg")

        if not os.path.exists(cfg_path):
            logger.warning(f"板块配置文件不存在: {cfg_path}")
            return {}

        block_names = {}

        try:
            with open(cfg_path, 'rb') as f:
                data = f.read()

            # 通达信板块配置格式
            # 每条记录: 板块代码(8字节ASCII) + 板块名称(42字节GBK编码)
            record_size = 50
            num_records = len(data) // record_size

            for i in range(num_records):
                offset = i * record_size
                record = data[offset:offset + record_size]

                # 提取板块代码
                code_bytes = record[:8]
                code = code_bytes.decode('ascii', errors='ignore').strip('\x00')

                # 提取板块名称
                name_bytes = record[8:50]
                try:
                    name = name_bytes.decode('gbk', errors='ignore').strip('\x00')
                except:
                    name = ""

                if code and name:
                    block_names[code] = name

            logger.info(f"解析板块配置: {len(block_names)}个板块")
            self._block_names = block_names

            # 打印部分板块名称
            for code, name in list(block_names.items())[:10]:
                logger.debug(f"  {code}: {name}")

        except Exception as e:
            logger.error(f"解析板块配置失败: {e}")

        return block_names

    def parse_block_file(self, block_file: str) -> List[str]:
        """
        解析单个板块成分股文件

        Args:
            block_file: .blk文件名（如"118.blk"）

        Returns:
            股票代码列表（如['sh600000', 'sz000001']）
        """
        filepath = os.path.join(self.blocknew_path, block_file)

        if not os.path.exists(filepath):
            return []

        codes = []

        try:
            with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()

            # 通达信.blk文件格式：每行一个7位代码
            # 格式: 市场标识(1位) + 股票代码(6位)
            for line in content.strip().split('\n'):
                line = line.strip()

                if len(line) >= 6:
                    # 提取6位股票代码
                    stock_code = line[:6]

                    # 判断市场
                    # 第1位可能是市场标识
                    if len(line) >= 7:
                        market_flag = line[0]
                        # 根据第1位或股票代码前缀判断市场
                        if stock_code.startswith('6'):
                            full_code = f'sh{stock_code}'
                        elif stock_code.startswith('0') or stock_code.startswith('3'):
                            full_code = f'sz{stock_code}'
                        elif stock_code.startswith('8') or stock_code.startswith('4'):
                            full_code = f'bj{stock_code}'
                        else:
                            continue
                    else:
                        if stock_code.startswith('6'):
                            full_code = f'sh{stock_code}'
                        elif stock_code.startswith('0') or stock_code.startswith('3'):
                            full_code = f'sz{stock_code}'
                        else:
                            continue

                    codes.append(full_code)

        except Exception as e:
            logger.debug(f"解析板块文件失败 {block_file}: {e}")

        return codes

    def parse_all_blocks(self) -> Dict[str, Dict]:
        """
        解析所有板块

        Returns:
            {板块代码: {'name': 板块名称, 'stocks': [股票列表]}}
        """
        logger.info("开始解析板块文件...")

        # 1. 解析板块配置
        self.parse_block_config()

        # 2. 获取所有.blk文件
        if not os.path.exists(self.blocknew_path):
            logger.error(f"板块目录不存在: {self.blocknew_path}")
            return {}

        blk_files = [f for f in os.listdir(self.blocknew_path) if f.endswith('.blk')]

        logger.info(f"找到{len(blk_files)}个板块文件")

        # 3. 解析每个板块
        for blk_file in blk_files:
            # 提取板块代码（去掉.blk扩展名）
            block_code = blk_file.replace('.blk', '')

            # 解析成分股
            stocks = self.parse_block_file(blk_file)

            if stocks:
                # 获取板块名称
                block_name = self._block_names.get(block_code, block_code)

                self.blocks[block_code] = {
                    'code': block_code,
                    'name': block_name,
                    'stocks': stocks
                }

                # 建立股票到板块的映射
                for stock in stocks:
                    self.stock_to_blocks[stock].append(block_code)

        logger.info(f"解析完成: {len(self.blocks)}个板块")

        return self.blocks

    def get_block_stocks(self, block_code: str) -> List[str]:
        """获取指定板块的成分股"""
        if block_code in self.blocks:
            return self.blocks[block_code]['stocks']
        return []

    def get_stock_blocks(self, stock_code: str) -> List[str]:
        """获取股票所属的板块代码列表"""
        return self.stock_to_blocks.get(stock_code, [])

    def get_block_name(self, block_code: str) -> str:
        """获取板块名称"""
        if block_code in self.blocks:
            return self.blocks[block_code]['name']
        return self._block_names.get(block_code, block_code)

    def print_summary(self):
        """打印板块摘要"""
        print(f"\n板块摘要:")
        print(f"  总板块数: {len(self.blocks)}")

        # 按股票数量排序
        sorted_blocks = sorted(
            self.blocks.items(),
            key=lambda x: len(x[1]['stocks']),
            reverse=True
        )

        print(f"\n股票数量最多的板块TOP20:")
        print(f"{'代码':<10} {'名称':<30} {'股票数':<10}")
        print("-" * 60)

        for code, info in sorted_blocks[:20]:
            print(f"{code:<10} {info['name']:<30} {len(info['stocks']):<10}")

    def create_sector_mapping(self) -> Dict[str, str]:
        """
        创建股票到板块的映射（用于快速查询）

        对于属于多个板块的股票，选择股票数量最多的板块
        """
        mapping = {}

        for stock, blocks in self.stock_to_blocks.items():
            if len(blocks) == 1:
                mapping[stock] = self.get_block_name(blocks[0])
            elif len(blocks) > 1:
                # 选择成分股最多的板块
                block_counts = [(b, len(self.get_block_stocks(b))) for b in blocks]
                best_block = max(block_counts, key=lambda x: x[1])[0]
                mapping[stock] = self.get_block_name(best_block)

        return mapping


def main():
    """测试"""
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    parser = TDXBlockParser(tdx_path)

    # 解析所有板块
    parser.parse_all_blocks()

    # 打印摘要
    parser.print_summary()

    # 测试查询
    print(f"\n测试查询:")

    # 查看某个板块的股票
    if parser.blocks:
        first_block_code = list(parser.blocks.keys())[0]
        stocks = parser.get_block_stocks(first_block_code)
        print(f"  {first_block_code}({parser.get_block_name(first_block_code)}): {len(stocks)}只股票")
        if stocks:
            print(f"    前10只: {stocks[:10]}")

    # 查看某只股票所属板块
    test_stock = 'sh600000'
    blocks = parser.get_stock_blocks(test_stock)
    print(f"\n  {test_stock} 所属板块:")
    for b in blocks:
        print(f"    {b} ({parser.get_block_name(b)})")


if __name__ == '__main__':
    main()
