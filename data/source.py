"""
数据源抽象接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
import pandas as pd


class DataSource(ABC):
    """
    数据源抽象基类

    所有数据源需要实现此接口
    """

    @abstractmethod
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
            symbol: 股票代码（如 '000001'）
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期 ('1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly')
            adjust: 复权方式 ('qfq'前复权, 'hfq'后复权, ''不复权)

        Returns:
            包含K线数据的DataFrame，列名: datetime, open, high, low, close, volume, amount
        """
        pass

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表

        Returns:
            包含股票信息的DataFrame
        """
        pass

    @abstractmethod
    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时行情

        Args:
            symbols: 股票代码列表

        Returns:
            实时行情数据
        """
        pass
