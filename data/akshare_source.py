"""
AKShare数据源实现

使用AKShare免费获取A股数据
"""

from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import time

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

from .source import DataSource


class AKShareSource(DataSource):
    """
    AKShare数据源

    免费的A股数据接口
    """

    # 周期映射
    PERIOD_MAP = {
        '1min': '1',
        '5min': '5',
        '15min': '15',
        '30min': '30',
        '60min': '60',
        'daily': 'daily',
        'weekly': 'weekly',
        'monthly': 'monthly'
    }

    # 复权映射
    ADJUST_MAP = {
        'qfq': 'qfq',  # 前复权
        'hfq': 'hfq',  # 后复权
        '': ''         # 不复权
    }

    def __init__(self, retry: int = 3, delay: float = 0.5):
        """
        初始化AKShare数据源

        Args:
            retry: 重试次数
            delay: 请求间隔（秒）
        """
        if not AKSHARE_AVAILABLE:
            raise ImportError("AKShare未安装，请运行: pip install akshare")

        self.retry = retry
        self.delay = delay

    def get_kline(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = 'daily',
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取K线数据

        Args:
            symbol: 股票代码（如 '000001' 或 '000001.SZ'）
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期
            adjust: 复权方式

        Returns:
            K线数据DataFrame
        """
        # 标准化股票代码
        symbol = self._normalize_symbol(symbol)

        # 设置默认日期
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 5)  # 默认5年

        # 周期转换
        ak_period = self.PERIOD_MAP.get(period, 'daily')

        df = None
        for attempt in range(self.retry):
            try:
                if period in ['1min', '5min', '15min', '30min', '60min']:
                    # 分钟级数据
                    df = self._get_minute_kline(symbol, start_date, end_date, ak_period)
                else:
                    # 日线及以上数据
                    df = self._get_daily_kline(symbol, start_date, end_date, ak_period, adjust)

                if df is not None and not df.empty:
                    break

            except Exception as e:
                if attempt == self.retry - 1:
                    raise
                time.sleep(self.delay * (attempt + 1))

        # 请求延迟
        time.sleep(self.delay)

        # 标准化列名（使用类名调用避免编码问题）
        try:
            df = AKShareSource._standardize_columns(df)
        except Exception as e:
            # 重试一次，处理可能的编码问题
            df = AKShareSource._standardize_columns(df)

        # 确保日期排序
        df = df.sort_values('datetime').reset_index(drop=True)

        return df

    def _get_daily_kline(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        period: str,
        adjust: str
    ) -> pd.DataFrame:
        """获取日线数据"""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        # 使用stock_zh_a_hist接口
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_str,
            end_date=end_str,
            adjust=adjust if adjust in ['qfq', 'hfq'] else ''
        )

        return df

    def _get_minute_kline(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        period: str
    ) -> pd.DataFrame:
        """获取分钟K线数据"""
        # AKShare的分钟数据接口使用 start_date 和 end_date 参数
        # 格式: 'YYYY-MM-DD HH:MM:SS'

        start_str = start_date.strftime('%Y-%m-%d 09:30:00')
        end_str = end_date.strftime('%Y-%m-%d 15:00:00')

        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                period=period,
                adjust=''
            )
            return df
        except Exception as e:
            print(f"获取分钟数据失败: {e}")
            return pd.DataFrame()

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表

        Returns:
            股票列表DataFrame
        """
        time.sleep(self.delay)

        # 获取A股列表
        df = ak.stock_info_a_code_name()

        return df

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时行情

        Args:
            symbols: 股票代码列表

        Returns:
            实时行情数据
        """
        time.sleep(self.delay)

        quotes = []
        for symbol in symbols:
            try:
                symbol = self._normalize_symbol(symbol)
                df = ak.stock_zh_a_spot_em()
                quote = df[df['代码'] == symbol]
                if not quote.empty:
                    quotes.append(quote)
            except Exception:
                continue

        if quotes:
            return pd.concat(quotes, ignore_index=True)
        return pd.DataFrame()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """
        标准化股票代码

        Args:
            symbol: 原始股票代码

        Returns:
            标准化后的6位代码
        """
        # 移除后缀
        symbol = symbol.replace('.SH', '').replace('.SZ', '')
        symbol = symbol.replace('.XSHG', '').replace('.XSHE', '')

        # 确保是6位数字
        return symbol.zfill(6)

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名

        Args:
            df: 原始DataFrame

        Returns:
            标准化后的DataFrame
        """
        # 列名映射（使用 UTF-8 字节以确保兼容性）
        column_map = {
            '\xe6\x97\xa5\xe6\x9c\x9f': 'datetime',   # 日期
            '\xe5\xbc\x80\xe7\x9b\x98': 'open',       # 开盘
            '\xe6\x9c\x80\xe9\xab\x98': 'high',       # 最高
            '\xe6\x9c\x80\xe4\xbd\x8e': 'low',        # 最低
            '\xe6\x94\xb6\xe7\x9b\x98': 'close',      # 收盘
            '\xe6\x88\x90\xe4\xba\xa4\xe9\x87\x8f': 'volume',   # 成交量
            '\xe6\x88\x90\xe4\xba\xa4\xe9\xa2\x9d': 'amount',     # 成交额
            '\xe6\x8d\xa2\xe6\x89\x8b\xe7\x8e\x87': 'turnover',    # 换手率
            '\xe6\xb6\xa8\xe8\xb7\x8c\xe5\xb9\x85': 'change_pct',  # 涨跌幅
            '\xe6\xb6\xa8\xe8\xb7\x8c\xe9\xa2\x9d': 'change',      # 涨跌额
            '\xe6\x8c\xaf\xe5\xb9\x85': 'amplitude'    # 振幅
        }

        # 同时支持直接中文列名
        column_map_fallback = {
            '日期': 'datetime',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '换手率': 'turnover',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change',
            '振幅': 'amplitude'
        }

        # 重命名列（优先使用字节映射）
        try:
            df = df.rename(columns=column_map)
        except:
            df = df.rename(columns=column_map_fallback)

        # 确保必需列存在
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}, 当前列: {df.columns.tolist()}")

        # 转换datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        # 确保amount列存在
        if 'amount' not in df.columns:
            df['amount'] = df['volume'] * df['close']

        # 选择需要的列
        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
        existing_cols = [c for c in cols if c in df.columns]

        return df[existing_cols]
