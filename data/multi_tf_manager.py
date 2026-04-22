"""
多周期数据管理器

统一管理周线/日线/30分钟数据的获取、同步和切片。

数据层级：
- 周线: 从日线resample（无额外API调用）
- 日线: AKShare直接获取
- 30分钟: AKShare分钟数据

同步规则：
- 回测模式：一次性加载所有数据，按日迭代时切片30分钟
- 日线bar触发周线resample
- 30分钟bars累积到当前日线bar
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class TimeFrameData:
    """单个周期的数据容器"""
    period: str            # 'weekly', 'daily', '30min'
    df: pd.DataFrame
    last_update_count: int = 0


class MultiTimeFrameManager:
    """
    多周期数据管理器

    使用方法（回测模式）：
        mgr = MultiTimeFrameManager(source, 'sh600519')
        mgr.initialize(start_date, end_date)
        weekly_df = mgr.get_weekly_df()
        daily_df = mgr.get_daily_df()

        for i in range(len(daily_df)):
            daily_bar = daily_df.iloc[i]
            min30_slice = mgr.get_min30_slice(i)
    """

    def __init__(self, symbol: str, daily_df: pd.DataFrame,
                 min30_df: Optional[pd.DataFrame] = None):
        """
        Args:
            symbol: 股票代码
            daily_df: 日线数据（必须）
            min30_df: 30分钟数据（可选，为None则仅日线+周线）
        """
        self.symbol = symbol
        self._daily_df = daily_df
        self._min30_df = min30_df
        self._weekly_df: Optional[pd.DataFrame] = None

        self._build()

    def _build(self):
        """构建所有周期数据"""
        # 周线从日线resample
        self._weekly_df = self._resample_to_weekly(self._daily_df)

        # 确保30分钟数据的datetime列是标准格式
        if self._min30_df is not None and len(self._min30_df) > 0:
            self._min30_df = self._normalize_datetime(self._min30_df)

        logger.debug(
            f"多周期数据初始化完成: {self.symbol} "
            f"日线={len(self._daily_df)}, 周线={len(self._weekly_df)}, "
            f"30分钟={len(self._min30_df) if self._min30_df is not None else 0}"
        )

    @property
    def daily_df(self) -> pd.DataFrame:
        return self._daily_df

    @property
    def weekly_df(self) -> pd.DataFrame:
        return self._weekly_df

    @property
    def min30_df(self) -> Optional[pd.DataFrame]:
        return self._min30_df

    def has_min30(self) -> bool:
        """是否有30分钟数据"""
        return self._min30_df is not None and len(self._min30_df) > 0

    def get_weekly_up_to(self, daily_index: int) -> pd.DataFrame:
        """
        获取到指定日线bar为止的周线数据

        Args:
            daily_index: 日线中的位置索引

        Returns:
            截止到该日的周线DataFrame
        """
        if daily_index >= len(self._daily_df):
            return self._weekly_df

        daily_date = self._daily_df.index[daily_index]
        return self._weekly_df[self._weekly_df.index <= daily_date]

    def get_min30_slice(self, daily_index: int) -> Optional[pd.DataFrame]:
        """
        获取指定日线bar对应的30分钟数据切片

        Args:
            daily_index: 日线中的位置索引

        Returns:
            当日的30分钟DataFrame，若无30分钟数据则返回None
        """
        if not self.has_min30() or daily_index >= len(self._daily_df):
            return None

        daily_date = self._daily_df.index[daily_index]
        daily_date_str = pd.Timestamp(daily_date).strftime('%Y-%m-%d')

        # 按日期筛选30分钟数据
        mask = self._min30_df.index.strftime('%Y-%m-%d') == daily_date_str
        slice_df = self._min30_df[mask]

        return slice_df if len(slice_df) > 0 else None

    def get_min30_up_to(self, daily_index: int) -> Optional[pd.DataFrame]:
        """
        获取到指定日线bar为止的所有30分钟数据

        Args:
            daily_index: 日线中的位置索引

        Returns:
            截止到该日的所有30分钟DataFrame
        """
        if not self.has_min30() or daily_index >= len(self._daily_df):
            return None

        daily_date = self._daily_df.index[daily_index]
        mask = self._min30_df.index <= daily_date
        result = self._min30_df[mask]

        return result if len(result) > 0 else None

    def _resample_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """日线→周线resample"""
        if len(daily_df) == 0:
            return daily_df

        weekly = daily_df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        if 'amount' in daily_df.columns:
            weekly_amount = daily_df.resample('W').agg({'amount': 'sum'}).dropna()
            weekly['amount'] = weekly_amount['amount']

        return weekly

    def _normalize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化datetime索引"""
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')
            df.index = pd.to_datetime(df.index)
        return df

    @classmethod
    def from_akshare(
        cls,
        symbol: str,
        daily_df: pd.DataFrame,
        min30_df: Optional[pd.DataFrame] = None,
    ) -> 'MultiTimeFrameManager':
        """
        从AKShare数据创建多周期管理器

        Args:
            symbol: 股票代码
            daily_df: 日线OHLCV数据
            min30_df: 30分钟OHLCV数据（可选）

        Returns:
            MultiTimeFrameManager实例
        """
        return cls(symbol, daily_df, min30_df)
