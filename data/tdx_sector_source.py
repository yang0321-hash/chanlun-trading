"""
基于本地TDX数据的板块和涨停板分析

完整版策略：
- 检测涨停板
- 计算连续涨停
- 按板块分类统计
- 在板块内找龙一
"""

import os
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from loguru import logger

from data.tdx_source import TDXDataSource
from data.tdx_block_parser import TDXBlockParser


def is_valid_stock(code: str) -> bool:
    """判断是否有效股票"""
    if code.startswith('sh6') and not code.startswith('sh688'):
        return True  # 上海主板
    if code.startswith('sh688'):
        return True  # 科创板
    if code.startswith('sz0'):
        return True  # 深圳主板
    if code.startswith('sz3'):
        return True  # 创业板
    return False


def calc_limit_up_threshold(code: str) -> float:
    """计算涨停阈值"""
    if code.startswith('sh688') or code.startswith('sz3'):
        return 19.5  # 科创板/创业板 20%
    return 9.5      # 普通股 10%


class TDXSectorAnalyzer:
    """
    基于通达信本地数据的板块和涨停板分析器
    """

    def __init__(self, tdx_path: str):
        """
        初始化分析器

        Args:
            tdx_path: 通达信根目录（不是vipdoc子目录）
        """
        self.tdx_path = tdx_path
        self.vipdoc_path = os.path.join(tdx_path, 'vipdoc')

        # 数据源
        self.tdx_source = TDXDataSource(self.vipdoc_path)

        # 板块解析器
        self.block_parser = TDXBlockParser(tdx_path)
        self._blocks_parsed = False

        # 缓存数据
        self._stock_data: Dict[str, pd.DataFrame] = {}
        self._index_data: Optional[pd.DataFrame] = None

    def parse_blocks(self):
        """解析板块数据"""
        if not self._blocks_parsed:
            self.block_parser.parse_all_blocks()
            self._blocks_parsed = True

    def load_stock_data(
        self,
        date: datetime,
        lookback_days: int = 60,
        max_stocks: int = None
    ) -> Dict[str, pd.DataFrame]:
        """加载股票数据"""
        logger.info(f"加载股票数据: {date.strftime('%Y-%m-%d')}")

        self._stock_data = {}
        files = []

        # 收集所有.day文件
        for market in ['sh', 'sz']:
            lday_path = os.path.join(self.vipdoc_path, market, 'lday')
            if not os.path.exists(lday_path):
                continue

            for f in os.listdir(lday_path):
                if f.endswith('.day'):
                    full_code = f.replace('.day', '')
                    if is_valid_stock(full_code):
                        files.append((full_code, os.path.join(lday_path, f)))

        logger.info(f"  找到 {len(files)} 只有效股票")

        start_date = date - timedelta(days=lookback_days)

        for i, (code, path) in enumerate(files):
            if max_stocks and i >= max_stocks:
                break

            if i % 500 == 0 and i > 0:
                logger.info(f"  加载进度: {i}/{min(len(files), max_stocks or len(files))}")

            try:
                df = self.tdx_source.get_kline(
                    code,
                    start_date=start_date,
                    end_date=date
                )

                if not df.empty and len(df) > 10:
                    self._stock_data[code] = df
            except Exception as e:
                continue

        logger.info(f"  成功加载 {len(self._stock_data)} 只股票")

        return self._stock_data

    def load_index_data(self, date: datetime, lookback_days: int = 30) -> pd.DataFrame:
        """加载指数数据"""
        start_date = date - timedelta(days=lookback_days)

        try:
            self._index_data = self.tdx_source.get_kline(
                'sh000001',
                start_date=start_date,
                end_date=date
            )
        except Exception as e:
            logger.warning(f"加载指数数据失败: {e}")

        return self._index_data

    def is_limit_up(self, row: pd.Series, code: str) -> bool:
        """判断是否涨停"""
        if 'open' not in row or 'close' not in row:
            return False

        open_p = row['open']
        close_p = row['close']

        if open_p <= 0:
            return False

        change_pct = (close_p - open_p) / open_p * 100
        threshold = calc_limit_up_threshold(code)

        return change_pct >= threshold

    def calc_consecutive_limit_up(
        self,
        df: pd.DataFrame,
        code: str,
        end_date: datetime
    ) -> int:
        """计算连续涨停板数"""
        if df.empty:
            return 0

        df_filtered = df[df.index <= pd.Timestamp(end_date)]

        if df_filtered.empty:
            return 0

        consecutive = 0
        for i in range(len(df_filtered) - 1, -1, -1):
            row = df_filtered.iloc[i]
            if self.is_limit_up(row, code):
                consecutive += 1
            else:
                break

        return consecutive

    def detect_limit_up_stocks(
        self,
        date: datetime
    ) -> List[Dict]:
        """检测指定日期的涨停板股票"""
        logger.info(f"检测涨停板: {date.strftime('%Y-%m-%d')}")

        limit_up_stocks = []
        target_date = pd.Timestamp(date)

        for code, df in self._stock_data.items():
            if df.empty:
                continue

            df_filtered = df[df.index <= target_date]

            if df_filtered.empty:
                continue

            latest = df_filtered.iloc[-1]

            # 检查日期是否匹配（允许1天误差）
            if abs((latest.name - target_date).days) > 1:
                continue

            if self.is_limit_up(latest, code):
                consecutive = self.calc_consecutive_limit_up(df, code, date)
                amount = latest.get('amount', 0) / 1e8 if 'amount' in latest else 0

                limit_up_stocks.append({
                    'code': code,
                    'price': latest['close'],
                    'open': latest['open'],
                    'high': latest['high'],
                    'low': latest['low'],
                    'change_pct': (latest['close'] - latest['open']) / latest['open'] * 100,
                    'amount': amount,
                    'volume': latest.get('volume', 0),
                    'consecutive_boards': consecutive,
                    'date': latest.name.strftime('%Y-%m-%d')
                })

        # 按成交额排序
        limit_up_stocks.sort(key=lambda x: x['amount'], reverse=True)

        logger.info(f"  发现 {len(limit_up_stocks)} 只涨停股")

        return limit_up_stocks

    def analyze_sector_strength(
        self,
        limit_up_stocks: List[Dict]
    ) -> List[Dict]:
        """
        分析板块强度

        Args:
            limit_up_stocks: 涨停板股票列表

        Returns:
            板块强度列表，按成交额排序
        """
        if not self._blocks_parsed:
            self.parse_blocks()

        # 按板块统计
        sector_stats = defaultdict(lambda: {
            'stocks': [],
            'total_amount': 0,
            'count': 0,
            'max_consecutive': 0
        })

        for stock in limit_up_stocks:
            code = stock['code']

            # 获取股票所属板块
            blocks = self.block_parser.get_stock_blocks(code)

            if not blocks:
                # 没有板块信息的单独处理
                blocks = ['无板块']

            for block_code in blocks:
                block_name = self.block_parser.get_block_name(block_code)

                sector_stats[block_code]['stocks'].append(stock)
                sector_stats[block_code]['total_amount'] += stock['amount']
                sector_stats[block_code]['count'] += 1
                sector_stats[block_code]['max_consecutive'] = max(
                    sector_stats[block_code]['max_consecutive'],
                    stock['consecutive_boards']
                )
                sector_stats[block_code]['name'] = block_name

        # 转换为列表并排序
        sectors = []
        for code, stats in sector_stats.items():
            if stats['count'] > 0:
                sectors.append({
                    'code': code,
                    'name': stats.get('name', code),
                    'stock_count': stats['count'],
                    'total_amount': stats['total_amount'],
                    'avg_amount': stats['total_amount'] / stats['count'],
                    'max_consecutive': stats['max_consecutive'],
                    'stocks': stats['stocks']
                })

        # 按总成交额排序
        sectors.sort(key=lambda x: x['total_amount'], reverse=True)

        return sectors

    def find_dragon_leader(
        self,
        sector_code: str,
        sector_name: str,
        limit_up_stocks: List[Dict]
    ) -> Optional[Dict]:
        """
        在板块内寻找龙一

        规则：
        1. 连续涨停板数最多
        2. 如果连板数相同，成交额最大
        3. 成交额必须明显大于第二（1.5倍以上）

        Args:
            sector_code: 板块代码
            sector_name: 板块名称
            limit_up_stocks: 涨停板股票列表

        Returns:
            龙一信息，如果找不到则返回None
        """
        if not self._blocks_parsed:
            self.parse_blocks()

        # 获取板块成分股
        block_stocks = self.block_parser.get_block_stocks(sector_code)

        if not block_stocks and sector_code != '无板块':
            return None

        # 筛选出涨停的成分股
        if sector_code == '无板块':
            sector_limit_stocks = [
                s for s in limit_up_stocks
                if not self.block_parser.get_stock_blocks(s['code'])
            ]
        else:
            sector_limit_stocks = [
                s for s in limit_up_stocks
                if s['code'] in block_stocks
            ]

        if not sector_limit_stocks:
            return None

        # 按连板数和成交额排序
        sector_limit_stocks.sort(
            key=lambda x: (x['consecutive_boards'], x['amount']),
            reverse=True
        )

        dragon = sector_limit_stocks[0]

        # 检查是否有明确龙一
        if len(sector_limit_stocks) >= 2:
            second = sector_limit_stocks[1]

            # 如果连板数相同且成交额接近，则无明确龙一
            if (dragon['consecutive_boards'] == second['consecutive_boards'] and
                dragon['amount'] < second['amount'] * 1.5):
                logger.info(f"    {sector_name}无明确龙一（成交额接近）")
                return None

        return {
            **dragon,
            'sector_code': sector_code,
            'sector': sector_name,
            'is_dragon_leader': True
        }

    def is_market_weak(self, date: datetime) -> Dict[str, any]:
        """判断大盘是否弱势"""
        if self._index_data is None or self._index_data.empty:
            return {'is_weak': False, 'reason': '无指数数据'}

        recent = self._index_data[self._index_data.index <= pd.Timestamp(date)].tail(5)

        if len(recent) < 3:
            return {'is_weak': False, 'reason': '数据不足'}

        is_falling = recent['close'].iloc[-1] < recent['close'].iloc[-3]
        volume_shrinking = recent['volume'].iloc[-1] < recent['volume'].iloc[-3:].mean()
        change_negative = (recent['close'].iloc[-1] - recent['close'].iloc[-2]) / recent['close'].iloc[-2] < 0

        is_weak = is_falling and (volume_shrinking or change_negative)

        return {
            'is_weak': is_weak,
            'reason': f'下跌={is_falling}, 缩量={volume_shrinking}, 跌幅={change_negative}'
        }

    def run_daily_analysis(
        self,
        date: datetime,
        min_consecutive_boards: int = 3,
        require_clear_leader: bool = True
    ) -> Dict:
        """
        执行每日分析

        Args:
            date: 分析日期
            min_consecutive_boards: 最小连板数（游资入场条件）
            require_clear_leader: 是否要求明确龙一

        Returns:
            分析结果
        """
        logger.info(f"{'='*60}")
        logger.info(f"每日分析: {date.strftime('%Y-%m-%d')}")
        logger.info(f"{'='*60}")

        # 1. 加载数据
        self.load_stock_data(date, max_stocks=5000)
        self.load_index_data(date)

        # 2. 检测涨停板
        limit_up_stocks = self.detect_limit_up_stocks(date)

        # 3. 判断大盘环境
        market_status = self.is_market_weak(date)
        logger.info(f"  大盘: {'弱势' if market_status['is_weak'] else '强势'}")

        # 4. 分析板块强度
        sectors = self.analyze_sector_strength(limit_up_stocks)
        logger.info(f"  涉及板块: {len(sectors)}个")

        # 5. 检查游资入场条件
        max_boards = max([s['consecutive_boards'] for s in limit_up_stocks], default=0)
        logger.info(f"  最高板: {max_boards}")

        hot_money_entered = max_boards >= min_consecutive_boards

        # 6. 寻找龙一
        dragons = []
        for sector in sectors[:10]:  # 只看前10个板块
            dragon = self.find_dragon_leader(
                sector['code'],
                sector['name'],
                limit_up_stocks
            )
            if dragon:
                dragons.append(dragon)

        return {
            'date': date.strftime('%Y-%m-%d'),
            'market_status': market_status,
            'limit_up_count': len(limit_up_stocks),
            'max_boards': max_boards,
            'hot_money_entered': hot_money_entered,
            'sectors': sectors[:10],
            'limit_up_stocks': limit_up_stocks[:50],
            'dragons': dragons
        }

    def print_analysis_result(self, result: Dict):
        """打印分析结果"""
        print(f"\n{'='*80}")
        print(f"涨停板分析报告 - {result['date']}")
        print(f"{'='*80}")

        # 大盘环境
        market = result['market_status']
        print(f"\n【大盘环境】")
        print(f"  状态: {'[弱势]' if market['is_weak'] else '[强势]'}")
        print(f"  原因: {market['reason']}")

        # 涨停统计
        print(f"\n【涨停统计】")
        print(f"  涨停数量: {result['limit_up_count']}只")
        print(f"  最高板数: {result['max_boards']}板")
        print(f"  游资入场: {'是' if result['hot_money_entered'] else '否'}")

        # 板块强度
        print(f"\n【板块强度TOP10】")
        print(f"{'排名':<4} {'板块代码':<10} {'板块名称':<20} {'数量':<6} {'成交额(亿)':<15}")
        print("-" * 70)

        for i, sector in enumerate(result['sectors'][:10], 1):
            print(f"{i:<4} {sector['code']:<10} {sector['name']:<20} "
                  f"{sector['stock_count']:<6} {sector['total_amount']:<15.2f}")

        # 涨停股
        print(f"\n【涨停股TOP30】")
        print(f"{'代码':<12} {'价格':<10} {'涨幅%':<10} {'成交(亿)':<12} {'连板':<6}")
        print("-" * 60)

        for stock in result['limit_up_stocks'][:30]:
            print(f"{stock['code']:<12} {stock['price']:<10.2f} "
                  f"{stock['change_pct']:<10.2f} {stock['amount']:<12.2f} "
                  f"{stock['consecutive_boards']:<6}")

        # 龙一
        print(f"\n【龙一股票】({len(result['dragons'])}只)")
        if result['dragons']:
            print(f"{'代码':<12} {'板块':<25} {'连板':<6} {'成交(亿)':<12}")
            print("-" * 60)

            for dragon in result['dragons']:
                print(f"{dragon['code']:<12} {dragon['sector']:<25} "
                      f"{dragon['consecutive_boards']:<6} {dragon['amount']:<12.2f}")
        else:
            print("  无龙一（大盘强势或游资未入场）")

        print(f"\n{'='*80}\n")
