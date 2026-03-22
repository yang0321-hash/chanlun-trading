"""
通达信(TDX)数据源

读取本地通达信目录下的A股日线和分钟线数据
"""

import os
import struct
import array
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .source import DataSource


class TDXDataSource(DataSource):
    """
    通达信数据源

    读取本地通达信目录的数据
    """

    def __init__(self, tdx_path: str):
        """
        初始化通达信数据源

        Args:
            tdx_path: 通达信vipdoc目录路径
        """
        self.tdx_path = tdx_path

        # 检查路径
        if not os.path.exists(tdx_path):
            raise ValueError(f"通达信路径不存在: {tdx_path}")

        # 新版通达信路径格式
        self.sh_day_path = os.path.join(tdx_path, "sh", "lday")
        self.sz_day_path = os.path.join(tdx_path, "sz", "lday")

        # 分钟线路径
        self.sh_min_path = os.path.join(tdx_path, "sh", "minline")
        self.sz_min_path = os.path.join(tdx_path, "sz", "minline")

    def get_kline(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = 'daily',
        adjust: str = ''
    ) -> pd.DataFrame:
        """
        获取K线数据

        Args:
            symbol: 股票代码（如 '600000' 或 '000001'）
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期
            adjust: 复权方式 (通达信本地数据通常为前复权)

        Returns:
            K线数据DataFrame
        """
        symbol = self._normalize_symbol(symbol)

        if period in ['daily', 'day', 'd']:
            df = self._read_day_data(symbol)
        elif period in ['5min', '5m', '60min', '60m', '1min', '1m']:
            df = self._read_min_data(symbol, period)
        else:
            raise ValueError(f"不支持的周期: {period}")

        if df is None or df.empty:
            return pd.DataFrame()

        # 日期过滤
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df.sort_index()

    def _normalize_symbol(self, symbol: str) -> tuple:
        """
        标准化股票代码为(市场, 代码)格式

        Args:
            symbol: 股票代码

        Returns:
            (market, code) market='SH'或'SZ'
        """
        symbol = symbol.upper().replace('.SH', '').replace('.SZ', '')
        symbol = symbol.replace('SH', '').replace('SZ', '')

        if len(symbol) != 6:
            raise ValueError(f"无效的股票代码: {symbol}")

        if symbol.startswith('6'):
            return 'SH', symbol
        elif symbol.startswith(('0', '3')):
            return 'SZ', symbol
        elif symbol.startswith('8') or symbol.startswith('4'):
            return 'BJ', symbol
        else:
            # 默认上海
            return 'SH', symbol

    def _get_day_file_path(self, market: str, code: str) -> str:
        """获取日线数据文件路径"""
        # 实际通达信格式: vipdoc/sh/lday/sh600000.day 或 vipdoc/sz/lday/sz002600.day
        if market == 'SH':
            base_path = os.path.join(self.tdx_path, "sh", "lday", f"sh{code}")
        else:  # SZ
            base_path = os.path.join(self.tdx_path, "sz", "lday", f"sz{code}")

        # 尝试带.day扩展名和不带扩展名两种情况
        if os.path.exists(base_path):
            return base_path
        elif os.path.exists(base_path + ".day"):
            return base_path + ".day"
        else:
            return base_path  # 返回不带扩展名的路径，让调用者处理不存在的情况

    def _read_day_data(self, symbol: str) -> pd.DataFrame:
        """读取日线数据"""
        # symbol 可能是字符串或 (market, code) 元组
        if isinstance(symbol, tuple):
            market, code = symbol
        else:
            market, code = self._normalize_symbol(symbol)
        file_path = self._get_day_file_path(market, code)

        if not os.path.exists(file_path):
            return pd.DataFrame()

        try:
            with open(file_path, 'rb') as f:
                # 通达信日线数据格式：
                # 每条记录32字节：日期(4) + 开(4) + 高(4) + 低(4) + 收(4) + 成交额(4) + 成交量(4) + 保留(4)
                data = f.read()

            record_size = 32
            num_records = len(data) // record_size

            if num_records == 0:
                return pd.DataFrame()

            records = []
            for i in range(num_records):
                offset = i * record_size
                # 解包数据
                # 通达信格式: 8个4字节整数
                date_val, open_p, high_p, low_p, close_p, amount, vol, reserved = \
                    struct.unpack('IIIIIIII', data[offset:offset + record_size])

                # 日期转换：通达信日期是整数，格式为YYYYMMDD
                # 需要转换为可读日期
                date_str = str(date_val)
                if len(date_str) == 8:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    if year >= 1990:  # 过滤无效数据
                        try:
                            dt = pd.Timestamp(year, month, day)
                            # 价格需要除以100
                            records.append({
                                'datetime': dt,
                                'open': open_p / 100,
                                'high': high_p / 100,
                                'low': low_p / 100,
                                'close': close_p / 100,
                                'volume': vol,
                                'amount': amount
                            })
                        except:
                            continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            return df

        except Exception as e:
            print(f"读取日线数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _read_min_data(self, symbol: str, period: str) -> pd.DataFrame:
        """读取分钟线数据"""
        # symbol 可能是字符串或 (market, code) 元组
        if isinstance(symbol, tuple):
            market, code = symbol
        else:
            market, code = self._normalize_symbol(symbol)

        # 通达信5分钟线文件路径格式
        if period in ['5min', '5m']:
            file_name = f"{code}{market}_5.min"
        elif period in ['1min', '1m']:
            file_name = f"{code}{market}_1.min"
        elif period in ['30min', '30m']:
            file_name = f"{code}{market}_30.min"
        elif period in ['60min', '60m']:
            file_name = f"{code}{market}_60.min"
        else:
            return pd.DataFrame()

        if market == 'SH':
            base_path = self.sh_min5_path if os.path.exists(self.sh_min5_path) else os.path.join(self.tdx_path, "minline")
        else:
            base_path = self.sz_min5_path if os.path.exists(self.sz_min5_path) else os.path.join(self.tdx_path, "minline")

        file_path = os.path.join(base_path, file_name)

        if not os.path.exists(file_path):
            return pd.DataFrame()

        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            # 通达信分钟线格式：每条记录32字节
            record_size = 32
            num_records = len(data) // record_size

            if num_records == 0:
                return pd.DataFrame()

            records = []
            for i in range(num_records):
                offset = i * record_size
                date_val, time_val, open_p, high_p, low_p, close_p, amount, vol, _ = \
                    struct.unpack('IIIIIIFII', data[offset:offset + record_size])

                # 日期时间转换
                date_str = str(date_val)
                time_str = str(time_val).zfill(6)

                if len(date_str) == 8 and len(time_str) == 6:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])

                    if year >= 1990:
                        try:
                            dt = pd.Timestamp(year, month, day, hour, minute)
                            records.append({
                                'datetime': dt,
                                'open': open_p / 100,
                                'high': high_p / 100,
                                'low': low_p / 100,
                                'close': close_p / 100,
                                'volume': vol,
                                'amount': amount
                            })
                        except:
                            continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            return df

        except Exception as e:
            print(f"读取分钟线数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表

        扫描通达信目录获取所有可用股票
        """
        stocks = []

        # 扫描上海市场
        sh_path = self.sh_day_path if os.path.exists(self.sh_day_path) else os.path.join(self.tdx_path, "vipdoc", "ShTdzxDay")
        if os.path.exists(sh_path):
            for file in os.listdir(sh_path):
                if file.endswith('.day'):
                    code = file.replace('.day', '').replace('SH', '').replace('SZ', '')
                    if len(code) == 6:
                        stocks.append({
                            'code': code,
                            'market': 'SH',
                            'symbol': f"{code}.SH"
                        })

        # 扫描深圳市场
        sz_path = self.sz_day_path if os.path.exists(self.sz_day_path) else os.path.join(self.tdx_path, "vipdoc", "SzTdzxDay")
        if os.path.exists(sz_path):
            for file in os.listdir(sz_path):
                if file.endswith('.day'):
                    code = file.replace('.day', '').replace('SH', '').replace('SZ', '')
                    if len(code) == 6:
                        stocks.append({
                            'code': code,
                            'market': 'SZ',
                            'symbol': f"{code}.SZ"
                        })

        return pd.DataFrame(stocks)

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """通达信本地数据不支持实时行情"""
        return pd.DataFrame()

    def get_available_dates(self, symbol: str, period: str = 'daily') -> List[pd.Timestamp]:
        """获取指定股票的可用日期列表"""
        df = self.get_kline(symbol, period=period)
        return df.index.tolist() if not df.empty else []
