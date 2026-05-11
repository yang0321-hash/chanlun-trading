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
import urllib.request
import json
import time
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
        self._sina_available = True  # Sina始终可用（网络请求）

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
        """从TDX本地读取日线 .day 文件（自动前复权，向量化解析）

        自动检测OHLC格式:
        - int×100格式: 老数据, 价格字段如 170000 → 1700.00元
        - float格式: 新下载数据, 价格字段如 39.58 → 39.58元
        检测: 第一个open字段尝试int×100和float两种解读, 取价格合理的版本
        """
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

            # 自动检测OHLC格式:
            # - int×100格式: 老数据, uint32值如 170000 → 1700.00元
            # - float格式: 新数据, uint32值如 1109283308 → float32解读=39.58元
            # 判定: float32解读(open_f[0])在0.01~10000之间 → float格式; 否则 → int×100格式
            arr_int = np.frombuffer(data[:n * record_size], dtype='<u4').reshape(n, 8)
            dates = arr_int[:, 0]
            valid = dates >= 19900101
            if valid.sum() == 0:
                return pd.DataFrame()

            # 先截取有效行
            dates = dates[valid]
            close_u4 = arr_int[valid, 4]   # 收盘, uint32
            open_u4 = arr_int[valid, 1]

            # float32解读
            open_f = open_u4.view('<f4')
            close_f = close_u4.view('<f4')

            # 用收盘价的float32解读判断格式 (更稳定)
            # float格式: close_f应在0.01~10000; int×100格式: close_f解读会超出范围
            if 0.01 <= close_f[0] <= 10000:
                # float32格式, divisor=1.0
                divisor = 1.0
            else:
                # int×100格式, divisor=100.0
                divisor = 100.0

            df = pd.DataFrame({
                'open': open_u4 / divisor,
                'high': arr_int[valid, 2] / divisor,
                'low': arr_int[valid, 3] / divisor,
                'close': close_u4 / divisor,
                'volume': arr_int[valid, 6].astype(np.int64),
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

    def _fetch_sina_daily(self, symbol: str) -> pd.DataFrame:
        """通过新浪接口获取日线历史（urllib直连，绕过requests代理问题）"""
        import urllib.request, json

        # 解析代码
        sina_code = self.sina._normalize_code(symbol)
        url = (
            'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php'
            '/CN_MarketData.getKLineData'
        )
        params_str = f"symbol={sina_code}&scale=240&ma=no&datalen=500"
        try:
            req = urllib.request.Request(
                f"{url}?{params_str}",
                headers={'User-Agent': 'Mozilla/5.0'},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except Exception:
            return pd.DataFrame()

        if not data or not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()

        rows = []
        for k in data:
            try:
                rows.append({
                    'datetime': pd.Timestamp(k['day']),
                    'open': float(k['open']),
                    'high': float(k['high']),
                    'low': float(k['low']),
                    'close': float(k['close']),
                    'volume': float(k.get('volume', 0)),
                })
            except (KeyError, ValueError):
                continue

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index('datetime').sort_index()
        return df.tail(500)

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
            # TDX不可用 → tushare批量fallback（比Sina单请求快10x）
            print(f'   [Tushare Fallback] TDX不可用，改用tushare批量...', flush=True)
            fallback = {}
            try:
                import tushare as ts
                TOKEN = '445af3e7113dd4984a0ac217c32686ec6321161eac11a435529bc07d'
                ts.set_token(TOKEN)
                api = ts.pro_api()
                # 转换纯代码→完整ts_code
                def to_ts_code(code):
                    pure = code.split('.')[0]
                    if pure.startswith(('6', '9', '5')):
                        return f'{pure}.SH'
                    else:
                        return f'{pure}.SZ'

                # 获取最近交易日
                try:
                    cal = api.trade_cal(exchange='SSE', is_open='1',
                                         start_date=(pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y%m%d'),
                                         end_date=pd.Timestamp.now().strftime('%Y%m%d'))
                    last_trade = cal['cal_date'].iloc[-1]
                except Exception:
                    last_trade = pd.Timestamp.now().strftime('%Y%m%d')
                start_date = (pd.Timestamp(last_trade) - pd.Timedelta(days=300)).strftime('%Y%m%d')
                end_date = last_trade.replace('-', '')
                # 批量拉取: 每批50只
                BATCH = 50
                for i in range(0, len(codes), BATCH):
                    batch = codes[i:i+BATCH]
                    for code in batch:
                        try:
                            ts_code = to_ts_code(code)
                            df = api.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                            if df is not None and len(df) >= min_bars:
                                df = df.sort_values('trade_date').reset_index(drop=True)
                                last_close = df['close'].iloc[-1]
                                if min_price <= last_close <= max_price:
                                    df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                                    for col in ['open', 'high', 'low', 'close']:
                                        df[col] = pd.to_numeric(df[col], errors='coerce')
                                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                                    pure = code.split('.')[0]
                                    fallback[pure] = df
                        except Exception:
                            pass
                    if i % 200 == 0:
                        print(f'     {i}/{len(codes)} 成功{len(fallback)}只', flush=True)
            except Exception as e:
                print(f'     Tushare fallback失败: {e}', flush=True)
            print(f'   [Tushare Fallback] 成功获取 {len(fallback)} 只', flush=True)
            return fallback

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

        # 多进程并行加载（min_price/max_price过滤后作为daily_map；同时保留全量数据存缓存）
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _read_one(args):
            """独立函数，供线程池调用 — 向量化解析，与_read_tdx_day一致"""
            tdx_path, code = args
            try:
                import numpy as np
                if code.startswith(('6', '9', '5')):
                    market_dir = 'sh'
                else:
                    market_dir = 'sz'
                filepath = os.path.join(tdx_path, market_dir, 'lday', f'{market_dir}{code}.day')
                if not os.path.exists(filepath):
                    return code, None, None
                with open(filepath, 'rb') as f:
                    data = f.read()
                record_size = 32
                n = len(data) // record_size
                if n == 0:
                    return code, None, None
                arr_int = np.frombuffer(data[:n * record_size], dtype='<u4').reshape(n, 8)
                dates = arr_int[:, 0]
                valid = dates >= 19900101
                if valid.sum() == 0:
                    return code, None, None
                dates = dates[valid]
                close_u4 = arr_int[valid, 4]
                open_u4 = arr_int[valid, 1]
                close_f = close_u4.view('<f4')
                if 0.01 <= close_f[0] <= 10000:
                    divisor = 1.0
                else:
                    divisor = 100.0
                import pandas as pd
                df = pd.DataFrame({
                    'open': open_u4 / divisor,
                    'high': arr_int[valid, 2] / divisor,
                    'low': arr_int[valid, 3] / divisor,
                    'close': close_u4 / divisor,
                    'volume': arr_int[valid, 6].astype(np.int64),
                }, index=pd.to_datetime(dates.astype(str), format='%Y%m%d'))
                last_close = None
                if len(df) >= min_bars:
                    last_close = df['close'].iloc[-1]
                return code, df, last_close
            except Exception:
                return code, None, None

        daily_map = {}
        full_map = {}
        t0 = time.time()
        workers = min(32, len(codes))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_read_one, (self.tdx_path, code)): code for code in codes}
            done = 0
            for future in as_completed(futures):
                code, df, last_close = future.result()
                if df is not None and len(df) >= min_bars:
                    full_map[code] = df  # 存全量
                    if last_close is not None and min_price <= last_close <= max_price:
                        daily_map[code] = df
                done += 1
                if done % 500 == 0:
                    print(f'   加载进度: {done}/{len(codes)}', flush=True)
        print(f'   并行读取 {len(codes)} 只耗时 {time.time()-t0:.1f}s', flush=True)

        # 保存缓存（全量数据，供不同价格过滤条件复用）
        if use_cache and full_map:
            try:
                cache_dir = os.path.join(os.path.dirname(__file__), '..', '.claude', 'cache')
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, 'daily_map_cache.pkl')
                today = pd.Timestamp.now().strftime('%Y-%m-%d')
                with open(cache_file, 'wb') as f:
                    pickle.dump({'date': today, 'data': full_map}, f)
            except Exception:
                pass

        return daily_map

    def _read_tdx_1min(self, symbol: str) -> pd.DataFrame:
        """从TDX本地读取1分钟线 (.lc1格式, 修正: 36字节, 每8条聚合1条有效K线)

        TDX .lc1格式: 每条记录36字节
        [0:4)   date (uint32, 乱码)
        [4:8)   open (float32)
        [8:12)  high (float32)
        [12:16) low (float32)
        [16:20) close (float32)
        [20:24) amount (float32)
        [24:28) vol (uint32)
        [28:36) reserved

        实际存储: 每8条36字节记录中, 只有第1条含有效OHLCV,
        故每8条聚合为1条有效1分钟K线。
        """
        market_dir, market_tag, code = self._parse_code(symbol)
        filepath = os.path.join(self.tdx_path, market_dir, 'minline', f'{market_dir}{code}.lc1')
        if not os.path.exists(filepath):
            return pd.DataFrame()

        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            n_records = len(data) // 36
            if n_records == 0:
                return pd.DataFrame()

            bars = []
            n_bars = n_records // 8
            for bar_idx in range(n_bars):
                off = bar_idx * 8 * 36
                chunk = data[off:off+36]
                o = struct.unpack('<f', chunk[4:8])[0]
                h = struct.unpack('<f', chunk[8:12])[0]
                l = struct.unpack('<f', chunk[12:16])[0]
                c = struct.unpack('<f', chunk[16:20])[0]
                amt = struct.unpack('<f', chunk[20:24])[0]
                vol = struct.unpack('<I', chunk[24:28])[0]
                if not (0.1 < o < 100 and 0.1 < h < 100 and
                        0.1 < l < 100 and vol > 0):
                    continue
                bars.append({
                    'datetime': bar_idx,
                    'open': o, 'high': h, 'low': l,
                    'close': c, 'amount': amt, 'volume': float(vol),
                })

            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars).set_index('datetime').sort_index()
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

    def _resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线→周线resample"""
        if len(df) < 5:
            return df
        weekly = df.resample('W').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
        return weekly

    def _resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线→月线resample"""
        if len(df) < 20:
            return df
        monthly = df.resample('M').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
        return monthly

    def _resample_to_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """将1分钟或5分钟数据合成为30分钟K线"""
        if df.empty:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            elif 'date' in df.columns:
                df = df.set_index('date')
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    return pd.DataFrame()
        return df.resample('30min', closed='left', label='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

    def _synthesize_30min_from_daily(self, symbol: str) -> pd.DataFrame:
        """从日线合成30min K线 (Geometric Brownian Motion插值)

        每个交易日生成8根30min K线 (09:30~11:30, 13:00~15:00)
        价格路径用GBM插值,保持OHLC一致性和真实波动率
        """
        df_daily = self._read_tdx_day(symbol)
        if df_daily.empty or len(df_daily) < 10:
            return pd.DataFrame()

        import numpy as np
        np_rng = np.random.RandomState(hash(symbol) % 2**31)

        rows = []
        for date, row in df_daily.iterrows():
            o, h, lo, c, v = row['open'], row['high'], row['low'], row['close'], row.get('volume', 0)
            if v <= 0:
                continue

            date_str = date.strftime('%Y-%m-%d')
            time_slots = [
                f'{date_str} 09:30', f'{date_str} 10:00',
                f'{date_str} 10:30', f'{date_str} 11:00',
                f'{date_str} 13:00', f'{date_str} 13:30',
                f'{date_str} 14:00', f'{date_str} 14:30',
            ]
            n_bars = 8

            # GBM path from open to close, constrained to [low, high]
            log_return = np.log(c / o) if o > 0 and c > 0 else 0
            daily_vol = abs(np.log(h / lo)) if lo > 0 and h > 0 else 0.02
            per_bar_vol = daily_vol / np.sqrt(n_bars)
            per_bar_drift = log_return / n_bars

            prices = [o]
            for _ in range(n_bars - 1):
                shock = np_rng.normal(per_bar_drift, per_bar_vol * 0.5)
                new_p = prices[-1] * np.exp(shock)
                prices.append(new_p)

            # Normalize to hit exact close
            prices[-1] = c
            if prices[-2] != 0:
                scale = c / prices[-1] if prices[-1] != 0 else 1.0
                for k in range(n_bars - 1):
                    prices[k] *= (1 + (c / prices[-1] - 1) * (k + 1) / n_bars) if prices[-1] != 0 else prices[k]

            # Clamp to [low, high]
            prices = [max(lo, min(h, p)) for p in prices]
            prices[-1] = c

            # Distribute volume: U-shaped (higher at open/close)
            vol_weights = np.array([1.5, 1.0, 0.8, 0.7, 0.7, 0.8, 1.0, 1.5])
            vol_weights = vol_weights / vol_weights.sum()
            bar_volumes = (v * vol_weights).astype(int)

            for i, (ts, bar_vol) in enumerate(zip(time_slots, bar_volumes)):
                bar_open = prices[i]
                bar_close = prices[i + 1] if i < n_bars - 1 else c
                bar_high = max(bar_open, bar_close, max(prices[i:i+2]))
                bar_low = min(bar_open, bar_close, min(prices[i:i+2]))
                rows.append({
                    'datetime': pd.Timestamp(ts),
                    'open': round(bar_open, 2),
                    'high': round(min(bar_high, h), 2),
                    'low': round(max(bar_low, lo), 2),
                    'close': round(bar_close, 2),
                    'volume': float(bar_vol),
                })

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index('datetime').sort_index()
        return df

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
        if period in ('daily', 'weekly', 'monthly'):
            # 日线: 优先TDX本地，失败则用Sina历史接口
            df = pd.DataFrame()
            if self._tdx_available:
                df = self._read_tdx_day(symbol)
                if not (len(df) > 0
                        and df['close'].iloc[-1] > 0.01
                        and df['close'].iloc[-1] < 10000):
                    df = pd.DataFrame()
            if df.empty and self._sina_available:
                sina_df = self._fetch_sina_daily(symbol)
                df = sina_df if sina_df is not None and len(sina_df) > 0 else pd.DataFrame()
            if df.empty:
                return df
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
            if period == 'weekly':
                df = self._resample_to_weekly(df)
            elif period == 'monthly':
                df = self._resample_to_monthly(df)
            return df

        elif period in ('30min', '5min', '15min', '1min', '60min'):
            # 分钟线: TDX本地历史 + 新浪盘中实时
            df_tdx = pd.DataFrame()
            if self._tdx_available:
                if period == '1min':
                    df_tdx = self._read_tdx_1min(symbol)
                elif period == '5min':
                    df_tdx = self._read_tdx_5min(symbol)
                elif period == '30min':
                    # 30min: TDX真实分钟线 + 日线合成补历史
                    df_real = pd.DataFrame()
                    df_1min = self._read_tdx_1min(symbol)
                    if not df_1min.empty:
                        df_real = self._resample_to_30min(df_1min)
                    elif not self._read_tdx_5min(symbol).empty:
                        df_5min = self._read_tdx_5min(symbol)
                        df_real = self._resample_to_30min(df_5min)

                    # 日线合成30min (覆盖更早历史)
                    df_synth = self._synthesize_30min_from_daily(symbol)

                    if len(df_real) > 0 and len(df_synth) > 0:
                        # 合并: 合成的早期数据 + 真实近期数据
                        overlap_start = df_real.index[0]
                        df_synth_early = df_synth[df_synth.index < overlap_start]
                        df_tdx = pd.concat([df_synth_early, df_real])
                        df_tdx = df_tdx[~df_tdx.index.duplicated(keep='last')].sort_index()
                    elif len(df_real) > 0:
                        df_tdx = df_real
                    elif len(df_synth) > 0:
                        df_tdx = df_synth
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
