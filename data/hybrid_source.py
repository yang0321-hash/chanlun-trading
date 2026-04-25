"""
混合数据源 — TDX本地历史 + 新浪实时

整合两个数据源的优势:
  - TDX本地: 完整历史数据(日线+30分钟), 8900+只股票, 盘后更新
  - 新浪在线: 盘中实时30分钟/5分钟K线 + 实时报价

自动选择最优数据源:
  - 日线 → TDX本地
  - 盘中30分钟 → 新浪在线
  - 实时报价 → 新浪在线
  - TDX不可用时回退到新浪

用法:
  from data.hybrid_source import HybridSource
  source = HybridSource()

  # 日线(自动用TDX)
  df_daily = source.get_kline('002600', period='daily')

  # 30分钟线(盘中自动用新浪)
  df_30min = source.get_kline('002600', period='30min')

  # 实时报价
  quotes = source.get_realtime_quote(['002600', '600438'])
"""

import os
import struct
import pandas as pd
from typing import List, Optional
from datetime import datetime, time as dtime
from pathlib import Path

from .source import DataSource
from .sina_source import SinaSource


class HybridSource(DataSource):
    """混合数据源

    自动根据数据类型和时间选择最优数据源。
    """

    # 默认TDX路径 (优先原始TDX vipdoc, 项目内tdx_data排后)
    DEFAULT_TDX_PATHS = [
        r'D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc',
        r'D:\new_tdx\vipdoc',
        r'D:\新建文件夹\claude\tdx_data',
        r'D:\tdx_data',
    ]

    def __init__(self, tdx_path: Optional[str] = None):
        """初始化

        Args:
            tdx_path: TDX vipdoc路径，不指定则自动搜索
        """
        self.sina = SinaSource()
        self.tdx_path = tdx_path or self._find_tdx_path()
        self._tdx_available = self.tdx_path is not None and os.path.exists(self.tdx_path)

    def _find_tdx_path(self) -> Optional[str]:
        """自动搜索TDX数据路径"""
        for path in self.DEFAULT_TDX_PATHS:
            if os.path.exists(path):
                return path
        return None

    def _parse_code(self, symbol: str):
        """解析股票代码 → (market_dir, market_tag, pure_code)

        Returns: ('sh', 'SH', '600438')
        """
        code = symbol.strip()
        market_hint = None
        for pfx in ['sh', 'sz', 'bj', 'SH', 'SZ', 'BJ']:
            if code.startswith(pfx):
                market_hint = pfx.lower()
                code = code[len(pfx):]
                break
        for suffix in ['.SH', '.SZ', '.BJ']:
            code = code.replace(suffix, '')

        # If user explicitly specified market prefix, preserve it
        if market_hint:
            return market_hint, market_hint.upper(), code

        if code.startswith('6') or code.startswith('9'):
            return 'sh', 'SH', code
        elif code.startswith('0') or code.startswith('2') or code.startswith('3'):
            return 'sz', 'SZ', code
        else:
            return 'sz', 'SZ', code

    def _read_tdx_day(self, symbol: str) -> pd.DataFrame:
        """从TDX本地读取日线 .day 文件（自动前复权，向量化解析）"""
        import numpy as np
        market_dir, market_tag, code = self._parse_code(symbol)
        filepath = os.path.join(self.tdx_path, market_dir, 'lday', f'{market_dir}{code}.day')
        if not os.path.exists(filepath):
            return pd.DataFrame()

        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            record_size = 32
            n = len(data) // record_size
            if n == 0:
                return pd.DataFrame()

            arr = np.frombuffer(data[:n * record_size], dtype='<u4').reshape(n, 8)
            dates = arr[:, 0]
            valid = dates >= 19900101
            arr = arr[valid]
            dates = dates[valid]

            if len(arr) == 0:
                return pd.DataFrame()

            df = pd.DataFrame({
                'open': arr[:, 1] / 100.0,
                'high': arr[:, 2] / 100.0,
                'low': arr[:, 3] / 100.0,
                'close': arr[:, 4] / 100.0,
                'volume': arr[:, 6].astype(np.int64),
            }, index=pd.to_datetime(dates.astype(str), format='%Y%m%d'))
            df.index.name = 'datetime'
            df = df.sort_index()

            # 前复权处理
            try:
                from data.tdx_adj_factor import adjust_tdx_daily
                ts_code = f'{code}.{market_dir.upper()}'
                df = adjust_tdx_daily(df, ts_code)
            except Exception:
                pass

            return df
        except Exception:
            return pd.DataFrame()

    def load_all_daily(self, codes: list, min_price: float = 3.0,
                       max_price: float = 200.0, min_bars: int = 200,
                       use_cache: bool = True) -> dict:
        """批量加载日线数据（带缓存，向量化解析）

        Args:
            codes: 股票代码列表（纯数字格式）
            min_price/max_price: 价格范围
            min_bars: 最小K线数
            use_cache: 是否使用pickle缓存（同交易日只读一次文件）

        Returns:
            {code: DataFrame} 字典
        """
        import numpy as np
        import pickle

        if not self._tdx_available:
            return {}

        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.claude', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'daily_map_cache.pkl')

        # 尝试加载缓存
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                cache_date = cached.get('date', '')
                today = pd.Timestamp.now().strftime('%Y-%m-%d')
                if cache_date == today:
                    daily_map = cached['data']
                    # 过滤价格范围
                    filtered = {}
                    for code, df in daily_map.items():
                        if code in codes and len(df) >= min_bars:
                            last_close = df['close'].iloc[-1]
                            if min_price <= last_close <= max_price:
                                filtered[code] = df
                    print(f'   [缓存] 日线数据命中 ({len(filtered)}只, 日期={cache_date})',
                          flush=True)
                    return filtered
            except Exception:
                pass

        # 全量加载
        daily_map = {}
        for code in codes:
            try:
                df = self._read_tdx_day(code)
                if len(df) >= min_bars:
                    last_close = df['close'].iloc[-1]
                    if min_price <= last_close <= max_price:
                        daily_map[code] = df
            except Exception:
                pass

        # 保存缓存（保存全量数据，不过滤价格，下次按需过滤）
        if use_cache and daily_map:
            try:
                # 保存前先收集全量（包括价格范围外的）
                full_map = {}
                for code in codes:
                    try:
                        df = self._read_tdx_day(code)
                        if len(df) >= min_bars:
                            full_map[code] = df
                    except Exception:
                        pass
                today = pd.Timestamp.now().strftime('%Y-%m-%d')
                with open(cache_file, 'wb') as f:
                    pickle.dump({'date': today, 'data': full_map}, f)
            except Exception:
                pass

        return daily_map

    def _read_tdx_1min(self, symbol: str) -> pd.DataFrame:
        """从TDX本地读取1分钟线 (.lc1格式)

        格式: HHfffffII (32字节/条)
          H: 日期 (year=num//2048+2004, month=(num%2048)//100, day=num%100)
          H: 从0点开始的分钟数
          f: open, high, low, close, amount (5个float)
          I: volume, reserved
        """
        market_dir, market_tag, code = self._parse_code(symbol)
        filepath = os.path.join(self.tdx_path, market_dir, 'minline', f'{market_dir}{code}.lc1')
        if not os.path.exists(filepath):
            return pd.DataFrame()

        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            record_fmt = struct.Struct('<HHfffffII')
            num_records = len(data) // record_fmt.size
            if num_records == 0:
                return pd.DataFrame()

            records = []
            for i in range(num_records):
                vals = record_fmt.unpack_from(data, i * record_fmt.size)
                date_num, time_num = vals[0], vals[1]
                year = date_num // 2048 + 2004
                month = (date_num % 2048) // 100
                day = (date_num % 2048) % 100
                hour = time_num // 60
                minute = time_num % 60
                if year >= 1990:
                    try:
                        dt = pd.Timestamp(year, month, day, hour, minute)
                        records.append({
                            'datetime': dt,
                            'open': vals[2],
                            'high': vals[3],
                            'low': vals[4],
                            'close': vals[5],
                            'volume': float(vals[7]),
                        })
                    except Exception:
                        continue

            if not records:
                return pd.DataFrame()
            df = pd.DataFrame(records).set_index('datetime').sort_index()
            return df
        except Exception:
            return pd.DataFrame()

    def _read_tdx_5min(self, symbol: str) -> pd.DataFrame:
        """从TDX本地读取5分钟线 (.lc5格式)

        格式与.lc1相同: HHfffffII (32字节/条)
        """
        market_dir, market_tag, code = self._parse_code(symbol)
        filepath = os.path.join(self.tdx_path, market_dir, 'fzline', f'{market_dir}{code}.lc5')
        if not os.path.exists(filepath):
            return pd.DataFrame()

        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            record_fmt = struct.Struct('<HHfffffII')
            num_records = len(data) // record_fmt.size
            if num_records == 0:
                return pd.DataFrame()

            records = []
            for i in range(num_records):
                vals = record_fmt.unpack_from(data, i * record_fmt.size)
                date_num, time_num = vals[0], vals[1]
                year = date_num // 2048 + 2004
                month = (date_num % 2048) // 100
                day = (date_num % 2048) % 100
                hour = time_num // 60
                minute = time_num % 60
                if year >= 1990:
                    try:
                        dt = pd.Timestamp(year, month, day, hour, minute)
                        records.append({
                            'datetime': dt,
                            'open': vals[2],
                            'high': vals[3],
                            'low': vals[4],
                            'close': vals[5],
                            'volume': float(vals[7]),
                        })
                    except Exception:
                        continue

            if not records:
                return pd.DataFrame()
            df = pd.DataFrame(records).set_index('datetime').sort_index()
            return df
        except Exception:
            return pd.DataFrame()

    def _resample_to_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """将1分钟或5分钟数据合成为30分钟K线"""
        if df.empty:
            return df
        return df.resample('30min', closed='left', label='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

    def get_kline(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = 'daily',
        adjust: str = 'qfq',
    ) -> pd.DataFrame:
        """获取K线数据，自动选择数据源

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期
            adjust: 复权方式

        Returns:
            DataFrame: OHLCV数据
        """
        if period == 'daily' or period == 'weekly' or period == 'monthly':
            # 日线优先用TDX本地
            if self._tdx_available:
                df = self._read_tdx_day(symbol)
                if len(df) > 0:
                    if start_date:
                        df = df[df.index >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df.index <= pd.Timestamp(end_date)]
                    return df
            # TDX不可用，回退到mootdx/新浪不支持日线，返回空
            return pd.DataFrame()

        elif period in ('30min', '5min', '15min', '1min', '60min'):
            # 分钟线: TDX本地历史 + 新浪盘中实时
            df_tdx = pd.DataFrame()
            if self._tdx_available:
                if period == '1min':
                    df_tdx = self._read_tdx_1min(symbol)
                elif period == '5min':
                    df_tdx = self._read_tdx_5min(symbol)
                elif period == '30min':
                    # TDX没有30分钟线，从1分钟合成
                    df_1min = self._read_tdx_1min(symbol)
                    if not df_1min.empty:
                        df_tdx = self._resample_to_30min(df_1min)
                elif period == '15min':
                    # 从5分钟合成15分钟
                    df_5min = self._read_tdx_5min(symbol)
                    if not df_5min.empty:
                        df_tdx = df_5min.resample('15min', closed='left', label='left').agg({
                            'open': 'first', 'high': 'max', 'low': 'min',
                            'close': 'last', 'volume': 'sum',
                        }).dropna()
                elif period == '60min':
                    # 从5分钟合成60分钟
                    df_5min = self._read_tdx_5min(symbol)
                    if not df_5min.empty:
                        df_tdx = df_5min.resample('60min', closed='left', label='left').agg({
                            'open': 'first', 'high': 'max', 'low': 'min',
                            'close': 'last', 'volume': 'sum',
                        }).dropna()

            # 获取新浪实时数据
            df_sina = self.sina.get_kline(symbol, period=period, start_date=start_date, end_date=end_date)

            if len(df_tdx) > 0 and len(df_sina) > 0:
                # 合并：TDX历史 + 新浪最新
                # 去掉TDX中与新浪重叠的部分，保留TDX更早的数据
                overlap_start = df_sina.index[0]
                df_tdx_early = df_tdx[df_tdx.index < overlap_start]
                df = pd.concat([df_tdx_early, df_sina])
                df = df[~df.index.duplicated(keep='last')].sort_index()
            elif len(df_tdx) > 0:
                df = df_tdx
                if start_date:
                    df = df[df.index >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]
            elif len(df_sina) > 0:
                df = df_sina
            else:
                return pd.DataFrame()

            return df

        return pd.DataFrame()

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self.sina.get_stock_list()

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时报价(新浪)"""
        return self.sina.get_realtime_quote(symbols)

    def get_realtime_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        return self.sina.get_realtime_price(symbol)

    @property
    def tdx_status(self) -> str:
        """TDX数据状态"""
        if self._tdx_available:
            return f"TDX本地: {self.tdx_path}"
        return "TDX本地: 不可用"

    @property
    def info(self) -> str:
        """数据源信息"""
        lines = [
            f"混合数据源:",
            f"  {self.tdx_status}",
            f"  新浪在线: 可用 (30min/5min/1min + 实时报价)",
        ]
        if self._tdx_available:
            try:
                sh_count = len(os.listdir(os.path.join(self.tdx_path, 'sh', 'lday')))
                sz_count = len(os.listdir(os.path.join(self.tdx_path, 'sz', 'lday')))
                lines.append(f"  TDX日线: {sh_count + sz_count} 只股票")
                sh_min = os.path.join(self.tdx_path, 'sh', 'minline')
                sz_min = os.path.join(self.tdx_path, 'sz', 'minline')
                if os.path.exists(sh_min) and os.path.exists(sz_min):
                    min_count = len(os.listdir(sh_min)) + len(os.listdir(sz_min))
                    lines.append(f"  TDX30分钟: {min_count} 只股票")
            except Exception:
                pass
        return '\n'.join(lines)
