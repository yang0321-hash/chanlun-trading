"""
板块和涨停板数据源

使用AKShare获取板块数据、涨停板数据
"""

from typing import List, Optional, Dict
from datetime import datetime, timedelta
import os
import pandas as pd
import time

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

from loguru import logger


class SectorDataSource:
    """
    板块数据源

    提供板块行情、涨停板数据等功能
    """

    def __init__(self, delay: float = 0.5):
        """
        初始化板块数据源

        Args:
            delay: 请求间隔（秒）
        """
        if not AKSHARE_AVAILABLE:
            raise ImportError("AKShare未安装，请运行: pip install akshare")

        # 清除代理，避免国内API被墙
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(key, None)

        self.delay = delay

    def get_sector_list(self) -> pd.DataFrame:
        """
        获取板块列表

        Returns:
            板块列表DataFrame
        """
        time.sleep(self.delay)

        try:
            # 获取概念板块
            df = ak.stock_board_concept_name_em()
            return df
        except Exception as e:
            logger.warning(f"获取板块列表失败: {e}")
            return pd.DataFrame()

    def get_sector_stocks(self, sector_name: str) -> pd.DataFrame:
        """
        获取板块成分股

        Args:
            sector_name: 板块名称

        Returns:
            成分股DataFrame
        """
        time.sleep(self.delay)

        try:
            df = ak.stock_board_concept_cons_em(symbol=sector_name)
            return df
        except Exception as e:
            logger.warning(f"获取板块成分股失败 {sector_name}: {e}")
            return pd.DataFrame()

    def get_sector_quote(self, sector_name: str = None) -> pd.DataFrame:
        """
        获取板块行情

        Args:
            sector_name: 板块名称，None表示获取所有板块

        Returns:
            板块行情DataFrame，包含板块名称、最新价、涨跌幅、成交额等
        """
        time.sleep(self.delay)

        try:
            # 获取所有概念板块的实时行情
            df = ak.stock_board_concept_name_em()

            # 按成交额排序
            if '成交额' in df.columns:
                df = df.sort_values('成交额', ascending=False)

            return df
        except Exception as e:
            logger.warning(f"获取板块行情失败: {e}")
            return pd.DataFrame()

    def get_limit_up_stocks(self, date: str = None) -> pd.DataFrame:
        """
        获取涨停板股票

        Args:
            date: 日期 YYYYMMDD 格式，None表示最新

        Returns:
            涨停板股票DataFrame
        """
        time.sleep(self.delay)

        try:
            # 获取涨停板数据
            df = ak.stock_zt_pool_em(date=date)

            # 标准化列名
            column_map = {
                '代码': 'code',
                '名称': 'name',
                '最新价': 'price',
                '涨跌幅': 'change_pct',
                '成交额': 'amount',
                '流通市值': 'market_cap',
                '换手率': 'turnover',
                '封单额': 'seal_amount',
                '首次涨停时间': 'first_limit_time',
                '涨停封单': 'seal_ratio',
                '炸板次数': 'break_count',
                '最后封单额': 'last_seal_amount'
            }
            df = df.rename(columns=column_map)

            return df
        except Exception as e:
            logger.warning(f"获取涨停板数据失败: {e}")
            return pd.DataFrame()

    def get_strong_sectors(self, date: str = None, top_n: int = 10) -> List[Dict]:
        """
        获取最强板块列表

        Args:
            date: 日期（暂未使用，保留接口）
            top_n: 返回前N个板块

        Returns:
            板块信息列表
        """
        df = self.get_sector_quote()

        if df.empty:
            return []

        result = []
        for _, row in df.head(top_n).iterrows():
            result.append({
                'name': row.get('板块名称', ''),
                'amount': row.get('成交额', 0),
                'change_pct': row.get('涨跌幅', 0),
                'stock_count': row.get('成分股数量', 0),
                'leading_stock': row.get('领涨股', ''),
                'leader_price': row.get('领涨股-价', 0),
            })

        return result

    def get_index_data(self, index_code: str = 'sh000001') -> pd.DataFrame:
        """
        获取指数数据

        Args:
            index_code: 指数代码，默认上证指数

        Returns:
            指数K线数据
        """
        time.sleep(self.delay)

        try:
            # 获取指数历史数据
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code.replace('sh', '')}")

            # 标准化列名
            column_map = {
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            df = df.rename(columns=column_map)
            df['datetime'] = pd.to_datetime(df['datetime'])

            return df
        except Exception as e:
            logger.warning(f"获取指数数据失败: {e}")
            return pd.DataFrame()

    def calculate_consecutive_limit_up(
        self,
        symbol: str,
        end_date: datetime = None
    ) -> int:
        """
        计算连续涨停板数

        Args:
            symbol: 股票代码
            end_date: 截止日期

        Returns:
            连续涨停板数
        """
        time.sleep(self.delay)

        try:
            # 获取股票历史数据
            symbol_clean = symbol.replace('sh', '').replace('sz', '')
            df = ak.stock_zh_a_daily(
                symbol=f"sz{symbol_clean}" if symbol.startswith('sz') else symbol_clean,
                start_date=(end_date - timedelta(days=60)).strftime('%Y%m%d') if end_date else (datetime.now() - timedelta(days=60)).strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d') if end_date else datetime.now().strftime('%Y%m%d'),
                adjust="qfq"
            )

            if df.empty:
                return 0

            # 标准化列名
            column_map = {
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
            }
            df = df.rename(columns=column_map)
            df = df.sort_values('datetime').reset_index(drop=True)

            # 计算连续涨停
            consecutive = 0
            for i in range(len(df) - 1, -1, -1):
                row = df.iloc[i]
                # 判断是否涨停（涨幅接近10%或20%）
                change_pct = (row['close'] - row['open']) / row['open'] * 100
                if change_pct >= 9.5:  # 涨停
                    consecutive += 1
                else:
                    break

            return consecutive
        except Exception as e:
            logger.warning(f"计算连续涨停失败 {symbol}: {e}")
            return 0

    def get_stock_name(self, symbol: str) -> str:
        """
        获取股票名称

        Args:
            symbol: 股票代码

        Returns:
            股票名称
        """
        time.sleep(self.delay)

        try:
            symbol_clean = symbol.replace('sh', '').replace('sz', '')
            df = ak.stock_individual_info_em(symbol=symbol_clean)
            if not df.empty:
                name = df[df['item'] == '股票名称']['value'].values[0]
                return name
        except Exception as e:
            logger.debug(f"获取股票名称失败 {symbol}: {e}")
        return ""

    def is_market_weak(self, date: datetime = None) -> Dict[str, bool]:
        """
        判断大盘是否弱势

        Args:
            date: 判断日期

        Returns:
            判断结果字典
        """
        try:
            # 获取上证指数
            df = self.get_index_data('sh000001')

            if df.empty:
                return {'is_weak': False, 'reason': '无法获取指数数据'}

            # 获取最近5天数据
            recent = df.tail(5)

            # 判断条件：
            # 1. 最近3天下跌
            is_falling = (recent['close'].iloc[-1] < recent['close'].iloc[-3])

            # 2. 成交量萎缩
            volume_shrinking = (recent['volume'].iloc[-1] < recent['volume'].iloc[-3:].mean())

            # 3. 涨跌幅小于0
            change_negative = (recent['close'].iloc[-1] - recent['close'].iloc[-2]) / recent['close'].iloc[-2] < 0

            is_weak = is_falling and (volume_shrinking or change_negative)

            return {
                'is_weak': is_weak,
                'reason': f'下跌={is_falling}, 缩量={volume_shrinking}, 跌幅={change_negative}'
            }
        except Exception as e:
            logger.warning(f"判断大盘强弱失败: {e}")
            return {'is_weak': False, 'reason': f'判断失败: {e}'}


def analyze_daily_limit_up(date: str = None) -> Dict:
    """
    每日涨停板分析

    Args:
        date: 日期 YYYYMMDD

    Returns:
        分析结果字典
    """
    source = SectorDataSource()

    # 获取涨停板数据
    limit_up_df = source.get_limit_up_stocks(date)

    if limit_up_df.empty:
        return {'error': '无涨停板数据'}

    # 获取板块行情
    sector_df = source.get_sector_quote()

    # 统计
    result = {
        'date': date or datetime.now().strftime('%Y-%m-%d'),
        'limit_up_count': len(limit_up_df),
        'limit_down_count': 0,  # 需要另外获取
        'sectors': [],
        'top_stocks': []
    }

    # 按板块统计涨停股
    if not sector_df.empty:
        for _, row in sector_df.head(10).iterrows():
            result['sectors'].append({
                'name': row.get('板块名称', ''),
                'amount': row.get('成交额', 0),
                'change_pct': row.get('涨跌幅', 0)
            })

    # 涨停股详情
    for _, row in limit_up_df.head(20).iterrows():
        result['top_stocks'].append({
            'code': row.get('code', ''),
            'name': row.get('name', ''),
            'price': row.get('price', 0),
            'change_pct': row.get('change_pct', 0),
            'amount': row.get('amount', 0),
            'first_limit_time': row.get('first_limit_time', '')
        })

    return result
