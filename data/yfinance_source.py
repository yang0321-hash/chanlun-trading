"""
yfinance数据源实现

使用yfinance获取股票数据（需要科学上网）
"""

from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from .source import DataSource


class YFinanceSource(DataSource):
    """
    yfinance数据源

    支持A股、美股、港股等全球市场数据
    """

    # 周期映射
    PERIOD_MAP = {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '60min': '1h',
        'daily': '1d',
        'weekly': '1wk',
        'monthly': '1mo'
    }

    def __init__(self, proxy: Optional[str] = None):
        """
        初始化yfinance数据源

        Args:
            proxy: 代理设置，如 'http://127.0.0.1:7890'
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance未安装，请运行: pip install yfinance")

        self.proxy = proxy

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
            symbol: 股票代码（如 '000001.SZ' 代表平安银行）
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期
            adjust: 复权方式 (yfinance默认已复权)

        Returns:
            K线数据DataFrame
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 5)

        # 转换周期
        interval = self.PERIOD_MAP.get(period, '1d')

        # 添加后缀（如果需要）
        symbol = self._normalize_symbol(symbol)

        try:
            # 使用yfinance下载数据
            ticker = yf.Ticker(symbol)

            # 构建参数
            kwargs = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'interval': interval,
                'auto_adjust': True,  # 自动调整复权
            }

            df = ticker.history(**kwargs)

            if df.empty:
                raise ValueError(f"未获取到数据，请检查股票代码: {symbol}")

            # 标准化列名
            df = self._standardize_columns(df)

            return df

        except Exception as e:
            raise RuntimeError(f"获取数据失败: {e}")

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表

        注意: yfinance不提供完整的A股列表

        Returns:
            空DataFrame
        """
        return pd.DataFrame()

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时行情

        Args:
            symbols: 股票代码列表

        Returns:
            实时行情数据
        """
        quotes = []

        for symbol in symbols:
            try:
                symbol = self._normalize_symbol(symbol)
                ticker = yf.Ticker(symbol)
                info = ticker.info

                if info:
                    quote = {
                        'code': symbol,
                        'name': info.get('longName', ''),
                        'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'change': info.get('previousClose', 0) - info.get('currentPrice', 0),
                        'volume': info.get('volume', 0)
                    }
                    quotes.append(quote)

            except Exception:
                continue

        if quotes:
            return pd.DataFrame(quotes)
        return pd.DataFrame()

    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化股票代码为yfinance格式

        A股代码格式：
        - 上海: 600000.SS
        - 深圳: 000001.SZ

        Args:
            symbol: 原始代码

        Returns:
            yfinance格式代码
        """
        # 已经是yfinance格式
        if '.' in symbol and symbol.split('.')[1] in ['SS', 'SZ', 'HK']:
            return symbol

        # 根据代码判断交易所
        if len(symbol) == 6:
            if symbol.startswith('6'):
                return f"{symbol}.SS"  # 上海
            elif symbol.startswith(('0', '3')):
                return f"{symbol}.SZ"  # 深圳
            elif symbol.startswith('8') or symbol.startswith('4'):
                return f"{symbol}.BJ"  # 北京

        return symbol

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # yfinance列名: Open, High, Low, Close, Volume
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }

        df = df.rename(columns=column_map)

        # 重置索引，将Date变为datetime列
        df.reset_index(inplace=True)

        # 查找日期列（可能是Date或索引名的变化）
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'datetime']:
                date_col = col
                break

        if date_col is None:
            # 使用第一列作为日期列
            date_col = df.columns[0]

        df.rename(columns={date_col: 'datetime'}, inplace=True)

        # 确保datetime列是正确类型
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        # 移除时间为0的无效日期
        df = df[df['datetime'].notna()]
        df = df[df['datetime'] >= pd.Timestamp('2000-01-01')]

        # 设置为索引
        df.set_index('datetime', inplace=True)

        # 确保必需列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")

        # 添加amount列
        df['amount'] = df['volume'] * df['close']

        # 选择需要的列
        return df[required_cols + ['amount']]


# 常用A股代码示例
A_SHARE_EXAMPLES = {
    '000001.SZ': '平安银行',
    '000002.SZ': '万科A',
    '600000.SS': '浦发银行',
    '600036.SS': '招商银行',
    '600519.SS': '贵州茅台',
    '000858.SZ': '五粮液',
    '300750.SZ': '宁德时代',
    '002594.SZ': '比亚迪',
}

# 美股代码示例
US_STOCK_EXAMPLES = {
    'AAPL': '苹果',
    'MSFT': '微软',
    'GOOGL': '谷歌',
    'TSLA': '特斯拉',
    'NVDA': '英伟达',
}
