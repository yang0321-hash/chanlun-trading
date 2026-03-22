"""
K线数据处理模块

处理原始K线数据，包括包含关系合并等缠论基础操作
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class KLineData:
    """
    单根K线数据

    Attributes:
        datetime: K线时间
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        volume: 成交量
        amount: 成交额
    """
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0

    @property
    def is_up(self) -> bool:
        """是否阳线"""
        return self.close >= self.open

    @property
    def is_down(self) -> bool:
        """是否阴线"""
        return self.close < self.open

    @property
    def body(self) -> float:
        """实体长度"""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> float:
        """上影线长度"""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """下影线长度"""
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        """K线振幅"""
        return self.high - self.low

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'datetime': self.datetime,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'amount': self.amount
        }


@dataclass
class MergeInfo:
    """
    K线合并信息

    记录包含关系合并的详细信息
    """
    merged_indices: List[int] = field(default_factory=list)
    original_count: int = 0


class KLine:
    """
    K线处理类

    实现缠论中的K线处理，包括包含关系合并
    """

    def __init__(self, data: List[KLineData], strict_mode: bool = True):
        """
        初始化K线处理类

        Args:
            data: 原始K线数据列表
            strict_mode: 是否使用严格模式（处理包含关系）
        """
        self.raw_data: List[KLineData] = data.copy()
        self.strict_mode = strict_mode
        self.processed_data: List[KLineData] = []
        self.merge_info: List[MergeInfo] = []
        self._process()

    def _process(self) -> None:
        """处理K线数据，处理包含关系"""
        if not self.strict_mode:
            self.processed_data = self.raw_data.copy()
            return

        result = []
        i = 0
        n = len(self.raw_data)

        while i < n:
            current = self.raw_data[i]
            merged_indices = [i]

            # 向后查找包含关系
            while i + 1 < n:
                next_k = self.raw_data[i + 1]
                inclusion = self._check_inclusion(current, next_k)

                if not inclusion:
                    break

                # 存在包含关系，合并K线
                merged_indices.append(i + 1)
                current = self._merge_klines(current, next_k, result)
                i += 1

            result.append(current)
            self.merge_info.append(MergeInfo(
                merged_indices=merged_indices,
                original_count=len(merged_indices)
            ))
            i += 1

        self.processed_data = result

    def _check_inclusion(self, k1: KLineData, k2: KLineData) -> bool:
        """
        检查两根K线是否存在包含关系

        包含关系：一根K线的高低点完全在另一根K线的高低点范围内

        Args:
            k1: 第一根K线
            k2: 第二根K线

        Returns:
            是否存在包含关系
        """
        # k1包含k2
        k1_contains_k2 = (k1.high >= k2.high and k1.low <= k2.low)
        # k2包含k1
        k2_contains_k1 = (k2.high >= k1.high and k2.low <= k1.low)

        return k1_contains_k2 or k2_contains_k1

    def _merge_klines(
        self,
        k1: KLineData,
        k2: KLineData,
        processed: List[KLineData]
    ) -> KLineData:
        """
        合并两根存在包含关系的K线

        缠论规则：
        - 如果前一根是阳线，则取高点的高、低点的低
        - 如果前一根是阴线，则取低点的低、高点的高

        Args:
            k1: 第一根K线（合并目标）
            k2: 第二根K线（被合并）
            processed: 已处理的K线列表（用于判断趋势）

        Returns:
            合并后的K线
        """
        # 判断趋势方向
        # 需要参考之前处理过的K线来确定当前是上升趋势还是下降趋势
        direction = self._get_direction(processed)

        if direction == 'up':
            # 上升趋势：取高点的高，低点的低
            new_high = max(k1.high, k2.high)
            new_low = max(k1.low, k2.low)
        else:
            # 下降趋势：取低点的低，高点的高
            new_high = min(k1.high, k2.high)
            new_low = min(k1.low, k2.low)

        # 确定新的开盘和收盘价
        # 通常保留后一根K线的开收盘，或者根据规则调整
        return KLineData(
            datetime=k2.datetime,
            open=k2.open,
            high=new_high,
            low=new_low,
            close=k2.close,
            volume=k1.volume + k2.volume,
            amount=k1.amount + k2.amount
        )

    def _get_direction(self, processed: List[KLineData]) -> str:
        """
        获取当前趋势方向

        Args:
            processed: 已处理的K线列表

        Returns:
            'up' 或 'down'
        """
        if len(processed) < 2:
            # 默认按当前K线方向
            return 'up'

        # 比较最后两根K线的高低点关系
        k1, k2 = processed[-2], processed[-1]

        if k2.high > k1.high and k2.low > k1.low:
            return 'up'
        elif k2.high < k1.high and k2.low < k1.low:
            return 'down'
        else:
            # 震荡情况，参考更早的K线
            for i in range(len(processed) - 2, -1, -1):
                if processed[i].high != k1.high or processed[i].low != k1.low:
                    if k1.high > processed[i].high:
                        return 'up'
                    else:
                        return 'down'
            return 'up'

    @property
    def data(self) -> List[KLineData]:
        """获取处理后的K线数据"""
        return self.processed_data

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, index: int) -> KLineData:
        return self.processed_data[index]

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = [k.to_dict() for k in self.processed_data]
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, strict_mode: bool = True) -> 'KLine':
        """
        从DataFrame创建KLine对象

        Args:
            df: DataFrame，需包含 datetime, open, high, low, close 列
            strict_mode: 是否严格模式

        Returns:
            KLine对象
        """
        # 确保datetime是datetime类型
        if 'datetime' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])

        kline_data = []
        for _, row in df.iterrows():
            if isinstance(row.name, pd.Timestamp):
                # 去除纳秒避免警告
                ts = row.name
                dt = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
            else:
                dt = row.get('datetime', row.name)
                if not isinstance(dt, datetime):
                    dt_ts = pd.to_datetime(dt)
                    dt = datetime(dt_ts.year, dt_ts.month, dt_ts.day, dt_ts.hour, dt_ts.minute, dt_ts.second)

            kline_data.append(KLineData(
                datetime=dt,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                amount=float(row.get('amount', 0))
            ))

        return cls(kline_data, strict_mode=strict_mode)

    def get_high(self, index: int) -> float:
        """获取指定位置K线的最高价"""
        return self.processed_data[index].high

    def get_low(self, index: int) -> float:
        """获取指定位置K线的最低价"""
        return self.processed_data[index].low

    def get_close(self, index: int) -> float:
        """获取指定位置K线的收盘价"""
        return self.processed_data[index].close

    def slice(self, start: int, end: Optional[int] = None) -> List[KLineData]:
        """切片获取K线数据"""
        return self.processed_data[start:end]
