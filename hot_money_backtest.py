"""
游资打板策略回测脚本

使用方法:
    python hot_money_backtest.py
    python hot_money_backtest.py --date 20240318
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data.sector_source import SectorDataSource, analyze_daily_limit_up
from strategies.hot_money_limit_up_strategy import (
    HotMoneyLimitUpStrategy,
    HotMoneyLimitUpSelector,
    DragonLeader
)
from backtest.engine import BacktestEngine, BacktestConfig


def run_daily_selector(date: str = None) -> list:
    """
    运行单日选股

    Args:
        date: 日期字符串 YYYYMMDD

    Returns:
        选中的龙一列表
    """
    if date:
        target_date = datetime.strptime(date, '%Y%m%d')
    else:
        target_date = datetime.now()

    logger.info(f"{'='*60}")
    logger.info(f"游资打板选股: {target_date.strftime('%Y-%m-%d')}")
    logger.info(f"{'='*60}")

    selector = HotMoneyLimitUpSelector(
        min_consecutive_boards=3,
        top_sectors=5,
        use_contrarian_filter=True
    )

    dragons = selector.select(target_date)

    # 输出结果
    print(f"\n选股结果 ({len(dragons)}只):")
    print("-" * 60)

    for i, dragon in enumerate(dragons, 1):
        print(f"{i}. {dragon.name} ({dragon.code})")
        print(f"   板块: {dragon.sector}")
        print(f"   连板: {dragon.consecutive_limit_up}板")
        print(f"   价格: ¥{dragon.price:.2f}")
        print()

    return dragons


def run_multi_day_backtest(start_date: str, end_date: str, initial_capital: float = 100000):
    """
    运行多日回测

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        initial_capital: 初始资金
    """
    logger.info(f"{'='*60}")
    logger.info(f"游资打板策略回测")
    logger.info(f"期间: {start_date} - {end_date}")
    logger.info(f"初始资金: ¥{initial_capital:,.0f}")
    logger.info(f"{'='*60}")

    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    strategy = HotMoneyLimitUpStrategy(
        name='游资打板策略',
        min_consecutive_boards=3,
        top_sectors=5,
        position_size=0.3
    )

    # 这里简化处理，实际回测需要历史数据
    # 由于涨停板数据获取限制，我们改用模拟回测

    trades = []
    current_date = start

    while current_date <= end:
        # 跳过周末
        if current_date.weekday() < 5:
            logger.info(f"\n处理日期: {current_date.strftime('%Y-%m-%d')}")

            # 执行选股
            selector = HotMoneyLimitUpSelector()
            dragons = selector.select(current_date)

            if dragons:
                logger.info(f"  选中{len(dragons)}只龙一")

                # 模拟买入
                for dragon in dragons:
                    trades.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'code': dragon.code,
                        'name': dragon.name,
                        'sector': dragon.sector,
                        'boards': dragon.consecutive_limit_up,
                        'price': dragon.price,
                        'action': '买入'
                    })

        current_date += timedelta(days=1)

    # 输出交易记录
    if trades:
        df = pd.DataFrame(trades)
        print("\n交易记录:")
        print(df.to_string(index=False))

        # 统计
        print(f"\n回测统计:")
        print(f"  总交易次数: {len(trades)}")
        print(f"  涉及股票: {df['name'].nunique()}只")
        print(f"  涉及板块: {df['sector'].nunique()}个")
        print(f"\n板块分布:")
        print(df['sector'].value_counts())


def generate_daily_report(date: str = None) -> dict:
    """
    生成每日分析报告

    Args:
        date: 日期 YYYYMMDD

    Returns:
        分析报告字典
    """
    if date:
        target_date = datetime.strptime(date, '%Y%m%d')
    else:
        target_date = datetime.now()

    logger.info(f"生成每日报告: {target_date.strftime('%Y-%m-%d')}")

    source = SectorDataSource()

    report = {
        'date': target_date.strftime('%Y-%m-%d'),
        'market_condition': source.is_market_weak(target_date),
        'strong_sectors': source.get_strong_sectors(top_n=10),
        'limit_up_stocks': [],
        'signals': []
    }

    # 获取涨停板数据
    limit_up_df = source.get_limit_up_stocks(date)

    if not limit_up_df.empty:
        report['limit_up_count'] = len(limit_up_df)

        # 按成交额排序
        limit_up_df = limit_up_df.sort_values('amount', ascending=False)

        for _, row in limit_up_df.head(20).iterrows():
            report['limit_up_stocks'].append({
                'code': row.get('code', ''),
                'name': row.get('name', ''),
                'price': row.get('price', 0),
                'change_pct': row.get('change_pct', 0),
                'amount': row.get('amount', 0) / 1e8 if 'amount' in row else 0,  # 转换为亿
                'turnover': row.get('turnover', 0),
            })

    return report


def print_daily_report(report: dict):
    """打印每日报告"""
    print(f"\n{'='*80}")
    print(f"涨停板每日分析报告 - {report['date']}")
    print(f"{'='*80}")

    # 大盘环境
    market = report['market_condition']
    print(f"\n【大盘环境】")
    print(f"  状态: {'[弱势]' if market['is_weak'] else '[强势]'}")
    print(f"  原因: {market['reason']}")

    # 最强板块
    print(f"\n【最强板块TOP10】")
    print(f"{'排名':<4} {'板块名称':<20} {'成交额(亿)':<15} {'涨跌幅':<10}")
    print("-" * 60)
    for i, sector in enumerate(report.get('strong_sectors', [])[:10], 1):
        print(f"{i:<4} {sector['name']:<20} {sector['amount']:<15.2f} {sector['change_pct']:<10.2f}%")

    # 涨停板股票
    print(f"\n【涨停板股票TOP20】")
    print(f"{'代码':<10} {'名称':<10} {'价格':<10} {'涨幅%':<10} {'成交额(亿)':<15} {'换手率%':<10}")
    print("-" * 80)
    for stock in report.get('limit_up_stocks', [])[:20]:
        print(f"{stock['code']:<10} {stock['name']:<10} {stock['price']:<10.2f} "
              f"{stock['change_pct']:<10.2f} {stock['amount']:<15.2f} {stock['turnover']:<10.2f}")

    # 游资打板信号
    print(f"\n【游资打板信号】")
    signals = report.get('signals', [])
    if signals:
        for signal in signals:
            print(f"  [*] {signal}")
    else:
        print("  无信号")


def main():
    parser = argparse.ArgumentParser(description='游资打板策略工具')
    parser.add_argument('--date', type=str, help='日期 YYYYMMDD')
    parser.add_argument('--mode', type=str, default='report',
                       choices=['report', 'select', 'backtest'],
                       help='运行模式: report=每日报告, select=选股, backtest=回测')
    parser.add_argument('--start-date', type=str, help='回测开始日期 YYYYMMDD')
    parser.add_argument('--end-date', type=str, help='回测结束日期 YYYYMMDD')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')

    args = parser.parse_args()

    if args.mode == 'report':
        report = generate_daily_report(args.date)
        print_daily_report(report)

    elif args.mode == 'select':
        dragons = run_daily_selector(args.date)

    elif args.mode == 'backtest':
        if not args.start_date or not args.end_date:
            logger.error("回测模式需要指定 --start-date 和 --end-date")
            return
        run_multi_day_backtest(args.start_date, args.end_date, args.capital)


if __name__ == '__main__':
    main()
