"""
游资打板策略 - 使用本地TDX数据

基于通达信本地数据的涨停板游资打板策略
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tdx_sector_source import TDXSectorAnalyzer
from backtest.strategy import Strategy, Signal, SignalType


@dataclass
class DragonLeader:
    """龙一信息"""
    code: str
    name: str
    sector: str
    consecutive_limit_up: int
    price: float
    amount: float
    date: str


class HotMoneyTDXStrategy:
    """
    游资打板策略 - TDX版本

    使用本地通达信数据进行选股和分析
    """

    def __init__(
        self,
        tdx_path: str,
        min_consecutive_boards: int = 3,
        top_sectors: int = 5,
        use_contrarian_filter: bool = True
    ):
        """
        初始化策略

        Args:
            tdx_path: 通达信vipdoc路径
            min_consecutive_boards: 游资入场最小板数
            top_sectors: 考虑前N个板块
            use_contrarian_filter: 是否使用逆势筛选
        """
        self.tdx_path = tdx_path
        self.min_consecutive_boards = min_consecutive_boards
        self.top_sectors = top_sectors
        self.use_contrarian_filter = use_contrarian_filter

        # 分析器
        self.analyzer = TDXSectorAnalyzer(tdx_path)

        # 选中的股票
        self._selected_dragons: List[DragonLeader] = []

    def select(self, date: datetime = None) -> List[DragonLeader]:
        """
        执行选股

        Args:
            date: 选股日期，None表示最新

        Returns:
            龙一列表
        """
        if date is None:
            date = datetime.now()

        logger.info(f"游资打板选股: {date.strftime('%Y-%m-%d')}")

        # 执行每日分析
        result = self.analyzer.run_daily_analysis(
            date,
            min_consecutive_boards=self.min_consecutive_boards
        )

        # 检查条件
        if self.use_contrarian_filter and not result['market_status']['is_weak']:
            logger.info("大盘非弱势，跳过选股")
            self._selected_dragons = []
            return []

        if not result['hot_money_entered']:
            logger.info(f"最高板{result['max_boards']}未达到{self.min_consecutive_boards}，游资未入场")
            self._selected_dragons = []
            return []

        # 转换龙一数据
        dragons = []
        for d in result['dragons']:
            dragons.append(DragonLeader(
                code=d['code'],
                name='',  # TDX数据没有名称
                sector=d.get('sector', '龙头'),
                consecutive_limit_up=d['consecutive_boards'],
                price=d['price'],
                amount=d['amount'],
                date=d['date']
            ))

        self._selected_dragons = dragons

        logger.info(f"选出{len(dragons)}只龙一: {[d.code for d in dragons]}")

        return dragons

    def get_signals(self, date: datetime = None) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            date: 日期

        Returns:
            信号字典
        """
        if date is None:
            date = datetime.now()

        dragons = self.select(date)

        return {
            'date': date.strftime('%Y-%m-%d'),
            'buy_signals': [
                {
                    'code': d.code,
                    'sector': d.sector,
                    'boards': d.consecutive_limit_up,
                    'price': d.price,
                    'action': '打板买入',
                    'reason': f'{d.sector}龙一，{d.consecutive_limit_up}板'
                }
                for d in dragons
            ],
            'dragons': dragons
        }

    def print_signals(self, signals: Dict):
        """打印信号"""
        print(f"\n{'='*80}")
        print(f"游资打板信号 - {signals['date']}")
        print(f"{'='*80}")

        buy_signals = signals['buy_signals']
        print(f"\n【买入信号】({len(buy_signals)}个)")

        if buy_signals:
            print(f"{'代码':<12} {'板块':<20} {'连板':<6} {'价格':<10} {'理由':<30}")
            print("-" * 80)

            for sig in buy_signals:
                print(f"{sig['code']:<12} {sig['sector']:<20} "
                      f"{sig['boards']:<6} {sig['price']:<10.2f} {sig['reason']:<30}")
        else:
            print("  无买入信号（大盘强势或游资未入场）")

        print(f"\n{'='*80}\n")


def run_hot_money_selection(
    tdx_path: str,
    date: str = None,
    min_boards: int = 3
):
    """
    运行游资打板选股

    Args:
        tdx_path: 通达信路径
        date: 日期 YYYYMMDD
        min_boards: 最小连板数
    """
    # 解析日期
    if date:
        target_date = datetime.strptime(date, '%Y%m%d')
    else:
        target_date = datetime.now()

    # 创建策略
    strategy = HotMoneyTDXStrategy(
        tdx_path=tdx_path,
        min_consecutive_boards=min_boards,
        use_contrarian_filter=False  # 暂时禁用逆势筛选
    )

    # 执行选股
    dragons = strategy.select(target_date)

    # 输出结果
    print(f"\n{'='*80}")
    print(f"游资打板选股结果 - {target_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}")

    if dragons:
        print(f"\n选出{len(dragons)}只龙一:\n")

        print(f"{'代码':<12} {'板块':<20} {'连板':<6} {'价格':<10} {'成交(亿)':<12}")
        print("-" * 70)

        for d in dragons:
            print(f"{d.code:<12} {d.sector:<20} {d.consecutive_limit_up:<6} "
                  f"{d.price:<10.2f} {d.amount:<12.2f}")

        # 打印完整分析
        print(f"\n详细分析:")
        signals = strategy.get_signals(target_date)
        strategy.print_signals(signals)
    else:
        print("\n无符合条件的龙一")
        print("原因:")
        print("  - 大盘非弱势")
        print("  - 或游资未入场（最高板数不足）")
        print("  - 或板块无明确龙一")

    print(f"\n{'='*80}\n")

    return dragons


def main():
    import argparse

    parser = argparse.ArgumentParser(description='游资打板选股 - TDX版本')
    parser.add_argument('--date', type=str, help='日期 YYYYMMDD')
    parser.add_argument('--boards', type=int, default=3, help='最小连板数')
    parser.add_argument('--tdx-path', type=str,
                       help='通达信vipdoc路径')

    args = parser.parse_args()

    # 默认路径（通达信根目录，不是vipdoc子目录）
    if args.tdx_path:
        tdx_path = args.tdx_path
    else:
        tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    run_hot_money_selection(tdx_path, args.date, args.boards)


if __name__ == '__main__':
    main()
