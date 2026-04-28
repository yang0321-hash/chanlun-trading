"""
新浪财经实时数据源

盘中实时数据获取，支持30分钟/5分钟/1分钟K线和实时报价。
适用于盘中缠论分析和实时监控。

数据能力:
  - 30分钟K线: 最多500 bars (~3个月)
  - 5分钟K线: 最多500 bars
  - 1分钟K线: 最多500 bars
  - 实时报价: 所有A股

用法:
  from data.sina_source import SinaSource
  source = SinaSource()

  # 获取30分钟K线
  df = source.get_kline('002600', period='30min')

  # 获取实时报价
  quotes = source.get_realtime_quote(['002600', '600438'])
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from .source import DataSource


def _clear_proxy():
    """清除代理设置，避免国内API被墙"""
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
        os.environ.pop(key, None)


class SinaSource(DataSource):
    """新浪财经实时数据源

    基于新浪财经API，提供盘中实时K线和报价数据。
    不依赖AKShare，直接HTTP请求，轻量快速。
    """

    # K线周期映射 (新浪API的scale参数)
    SCALE_MAP = {
        '1min': '1',
        '5min': '5',
        '15min': '15',
        '30min': '30',
        '60min': '60',
    }

    # 最大数据量
    MAX_BARS = 500

    def __init__(self, timeout: int = 10, delay: float = 0.2):
        """初始化

        Args:
            timeout: 请求超时(秒)
            delay: 请求间隔(秒)，避免被封
        """
        _clear_proxy()
        self.timeout = timeout
        self.delay = delay
        self._session = requests.Session()
        self._session.trust_env = False
        self._last_request = 0.0

    def _request(self, url: str, params: dict = None) -> Optional[dict]:
        """发送请求，带频率限制"""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        try:
            r = self._session.get(url, params=params, timeout=self.timeout)
            self._last_request = time.time()
            if r.status_code == 200 and r.text and r.text != 'null':
                return json.loads(r.text)
        except Exception:
            pass
        return None

    @staticmethod
    def _normalize_code(symbol: str) -> str:
        """标准化股票代码 → 新浪格式

        '600438' → 'sh600438'
        '000001' → 'sz000001'
        'sh600438' → 'sh600438'
        '600438.SH' → 'sh600438'
        """
        code = symbol.strip().upper()
        # 去掉后缀
        for suffix in ['.SH', '.SZ', '.BJ']:
            code = code.replace(suffix, '')
        code = code.lower()

        if code.startswith('sh') or code.startswith('sz') or code.startswith('bj'):
            return code

        # 根据代码判断市场
        if code.startswith('6') or code.startswith('9'):
            return f'sh{code}'
        elif code.startswith('0') or code.startswith('2') or code.startswith('3'):
            return f'sz{code}'
        elif code.startswith('8') or code.startswith('4'):
            return f'bj{code}'
        return f'sz{code}'

    @staticmethod
    def _strip_prefix(symbol: str) -> str:
        """去掉市场前缀 → 纯数字代码"""
        for prefix in ['sh', 'sz', 'bj', 'SH', 'SZ', 'BJ']:
            if symbol.startswith(prefix):
                return symbol[len(prefix):]
        return symbol

    def get_kline(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = '30min',
        adjust: str = 'qfq',
    ) -> pd.DataFrame:
        """获取K线数据

        Args:
            symbol: 股票代码
            start_date: 开始日期(可选，仅在返回数据中过滤)
            end_date: 结束日期(可选)
            period: K线周期 ('1min', '5min', '15min', '30min', '60min')
            adjust: 复权方式(新浪分钟线不复权)

        Returns:
            DataFrame: columns [datetime, open, high, low, close, volume]
        """
        scale = self.SCALE_MAP.get(period)
        if scale is None:
            raise ValueError(f"不支持的周期: {period}，支持: {list(self.SCALE_MAP.keys())}")

        sina_code = self._normalize_code(symbol)
        url = 'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
        params = {'symbol': sina_code, 'scale': scale, 'ma': 'no', 'datalen': str(self.MAX_BARS)}

        data = self._request(url, params)
        if not data or not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()

        rows = []
        for k in data:
            try:
                dt = pd.Timestamp(k['day'])
                # 日期过滤
                if start_date and dt < pd.Timestamp(start_date):
                    continue
                if end_date and dt > pd.Timestamp(end_date):
                    continue
                rows.append({
                    'datetime': dt,
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

        df = pd.DataFrame(rows)
        df = df.set_index('datetime').sort_index()
        return df

    def get_daily(self, symbol: str, lookback: int = 250) -> pd.DataFrame:
        """获取日线历史数据（通过新浪行情接口）

        新浪日线接口: scale=240 → 日K线（不是月度！）
        Returns: DataFrame with index=datetime, columns=[open, high, low, close, volume]
        """
        sina_code = self._normalize_code(symbol)
        url = (
            'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php'
            '/CN_MarketData.getKLineData'
        )
        params = {
            'symbol': sina_code,
            'scale': '240',   # 240 = 日K线（新浪约定）
            'ma': 'no',
            'datalen': str(min(lookback, 500)),
        }
        data = self._request(url, params)
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
        return df.tail(lookback)

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表(简化版，返回代码和名称)"""
        # 新浪没有直接的股票列表API，返回空DataFrame
        # 实际使用时可配合TDX本地数据获取完整列表
        return pd.DataFrame(columns=['code', 'name', 'market'])

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时报价

        Args:
            symbols: 股票代码列表

        Returns:
            DataFrame: columns [code, name, price, open, high, low, close,
                       volume, amount, bid1, ask1, change, pct_chg]
        """
        sina_codes = [self._normalize_code(s) for s in symbols]
        batch = ','.join(sina_codes)

        url = f'http://hq.sinajs.cn/list={batch}'
        headers = {'Referer': 'http://finance.sina.com.cn'}

        try:
            elapsed = time.time() - self._last_request
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            r = self._session.get(url, headers=headers, timeout=self.timeout)
            self._last_request = time.time()
        except Exception:
            return pd.DataFrame()

        rows = []
        for line in r.text.strip().split('\n'):
            try:
                parts = line.split('="')
                if len(parts) != 2:
                    continue
                sina_code = self._strip_prefix(parts[0].split('_')[-1])
                vals = parts[1].rstrip('";').split(',')
                if len(vals) < 32:
                    continue

                open_p = float(vals[1]) if vals[1] else 0
                prev_close = float(vals[2]) if vals[2] else 0
                price = float(vals[3]) if vals[3] else 0
                high = float(vals[4]) if vals[4] else 0
                low = float(vals[5]) if vals[5] else 0
                bid1 = float(vals[6]) if vals[6] else 0
                ask1 = float(vals[7]) if vals[7] else 0
                volume = float(vals[8]) if vals[8] else 0
                amount = float(vals[9]) if vals[9] else 0

                change = price - prev_close if prev_close > 0 else 0
                pct_chg = (change / prev_close * 100) if prev_close > 0 else 0

                rows.append({
                    'code': sina_code,
                    'name': vals[0],
                    'price': price,
                    'open': open_p,
                    'prev_close': prev_close,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume,
                    'amount': amount,
                    'bid1': bid1,
                    'ask1': ask1,
                    'change': round(change, 3),
                    'pct_chg': round(pct_chg, 2),
                })
            except (ValueError, IndexError):
                continue

        return pd.DataFrame(rows)

    def get_realtime_price(self, symbol: str) -> Optional[float]:
        """获取单只股票最新价格"""
        df = self.get_realtime_quote([symbol])
        if len(df) > 0:
            return float(df.iloc[0]['price'])
        return None
