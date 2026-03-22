"""
缠论选股器

扫描A股市场，识别日线级别的1买、2买、3买信号
支持在线数据(AKShare)和离线数据(通达信TDX)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
from loguru import logger
import json
import os

from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator
from core.segment import SegmentGenerator
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD


@dataclass
class BuySignal:
    """买入信号"""
    symbol: str           # 股票代码
    name: str             # 股票名称
    signal_type: str      # 1buy/2buy/3buy
    price: float          # 当前价格
    datetime: datetime    # 信号时间
    confidence: float     # 信号强度 (0-1)
    reason: str           # 信号描述
    pivot_high: float = 0 # 中枢高点
    pivot_low: float = 0  # 中枢低点
    macd_divergence: bool = False  # MACD背驰

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'signal_type': self.signal_type,
            'price': self.price,
            'datetime': self.datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': f"{self.confidence:.0%}",
            'reason': self.reason,
            'pivot_high': self.pivot_high,
            'pivot_low': self.pivot_low
        }


@dataclass
class ScanResult:
    """扫描结果"""
    scan_time: datetime = field(default_factory=datetime.now)
    total_scanned: int = 0
    signals_1buy: List[BuySignal] = field(default_factory=list)
    signals_2buy: List[BuySignal] = field(default_factory=list)
    signals_3buy: List[BuySignal] = field(default_factory=list)

    @property
    def total_signals(self) -> int:
        return len(self.signals_1buy) + len(self.signals_2buy) + len(self.signals_3buy)

    def get_all_signals(self) -> List[BuySignal]:
        """获取所有信号，按优先级排序"""
        all_signals = []
        all_signals.extend(self.signals_1buy)
        all_signals.extend(self.signals_2buy)
        all_signals.extend(self.signals_3buy)
        # 按信号强度排序
        all_signals.sort(key=lambda x: x.confidence, reverse=True)
        return all_signals


class ChanLunScreener:
    """
    缠论选股器

    识别规则：
    - 第一类买点(1buy): 下跌趋势中，最后中枢下方出现底背驰
    - 第二类买点(2buy): 1买后，回抽不破前低的点
    - 第三类买点(3buy): 突破中枢后，回踩不破中枢上沿
    """

    def __init__(
        self,
        use_macd: bool = True,
        min_klines: int = 60,
        tdx_path: Optional[str] = None,
        exclude_bj: bool = True
    ):
        """
        初始化选股器

        Args:
            use_macd: 是否使用MACD判断背驰
            min_klines: 最少K线数量
            tdx_path: TDX数据路径，如果提供则使用本地数据
            exclude_bj: 是否排除北交所股票
        """
        self.use_macd = use_macd
        self.min_klines = min_klines
        self.tdx_path = tdx_path
        self.exclude_bj = exclude_bj

        # 尝试加载数据源
        self.data_source = None
        self._init_data_source()

        # 股票名称缓存
        self._stock_names: Dict[str, str] = {}

        # 已触发1买的股票缓存 {symbol: buy_price}
        self._first_buy_triggered: Dict[str, float] = {}

        # 加载股票名称映射
        self._load_stock_names()

    def _init_data_source(self):
        """初始化数据源"""
        if self.tdx_path:
            try:
                from data.tdx_source import TDXSource
                self.data_source = TDXSource(self.tdx_path)
                logger.info(f"使用TDX数据源: {self.tdx_path}")
                return
            except Exception as e:
                logger.warning(f"TDX数据源初始化失败: {e}")

        try:
            from data.akshare_source import AKShareSource
            self.data_source = AKShareSource(retry=2, delay=0.3)
            logger.info("使用AKShare在线数据源")
        except Exception as e:
            logger.warning(f"AKShare数据源初始化失败: {e}")

    def _load_stock_names(self):
        """加载股票名称映射"""
        # 从技能目录加载
        skill_path = ".claude/skills/stock-name-matcher/stock_data.json"
        if os.path.exists(skill_path):
            try:
                with open(skill_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for code, name in data.items():
                        self._stock_names[code] = name
                logger.info(f"加载了 {len(self._stock_names)} 个股票名称")
            except Exception as e:
                logger.debug(f"加载股票名称失败: {e}")

    def scan_from_dataframe(self, df: pd.DataFrame, symbol: str) -> Optional[BuySignal]:
        """
        从DataFrame扫描股票

        Args:
            df: K线数据
            symbol: 股票代码

        Returns:
            买入信号
        """
        # 过滤北交所股票
        if self.exclude_bj and (symbol.startswith('bj') or symbol[:2].upper() == 'BJ'):
            return None

        if df is None or len(df) < self.min_klines:
            return None

        try:
            # 确保有datetime列
            if 'datetime' not in df.columns:
                if 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                else:
                    logger.error(f"DataFrame缺少datetime或date列: {df.columns.tolist()}")
                    return None

            kline = KLine.from_dataframe(df, strict_mode=True)
            return self._analyze_buy_points(kline, symbol)
        except Exception as e:
            logger.error(f"分析 {symbol} 失败: {e}")
            return None

    def scan_stock(self, symbol: str) -> Optional[BuySignal]:
        """
        扫描单只股票

        Args:
            symbol: 股票代码 (如 '600000', '000001')

        Returns:
            买入信号，如果没有信号则返回None
        """
        if self.data_source is None:
            logger.error("数据源未初始化")
            return None

        try:
            df = self.data_source.get_kline(
                symbol=symbol,
                period='daily'
            )

            return self.scan_from_dataframe(df, symbol)

        except Exception as e:
            logger.debug(f"扫描 {symbol} 失败: {e}")
            return None

    def scan_multiple(
        self,
        symbols: List[str],
        show_progress: bool = True
    ) -> ScanResult:
        """
        扫描多只股票

        Args:
            symbols: 股票代码列表
            show_progress: 是否显示进度

        Returns:
            扫描结果
        """
        result = ScanResult()
        result.total_scanned = len(symbols)

        for i, symbol in enumerate(symbols):
            if show_progress and i % 10 == 0:
                logger.info(f"扫描进度: {i}/{len(symbols)}")

            signal = self.scan_stock(symbol)

            if signal:
                if signal.signal_type == '1buy':
                    result.signals_1buy.append(signal)
                    self._first_buy_triggered[symbol] = signal.price
                elif signal.signal_type == '2buy':
                    result.signals_2buy.append(signal)
                elif signal.signal_type == '3buy':
                    result.signals_3buy.append(signal)

        return result

    def scan_local_files(
        self,
        data_dir: str,
        pattern: str = "*.json"
    ) -> ScanResult:
        """
        扫描本地数据文件

        Args:
            data_dir: 数据目录
            pattern: 文件匹配模式

        Returns:
            扫描结果
        """
        import glob

        result = ScanResult()
        files = glob.glob(os.path.join(data_dir, pattern))
        result.total_scanned = len(files)

        logger.info(f"发现 {len(files)} 个数据文件")

        for i, filepath in enumerate(files):
            if i % 10 == 0:
                logger.info(f"扫描进度: {i}/{len(files)}")

            try:
                # 从文件名提取股票代码
                symbol = os.path.basename(filepath).replace('.day.json', '').replace('.json', '')

                # 读取JSON数据
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)
                # 兼容不同的日期字段名
                if 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                else:
                    continue

                signal = self.scan_from_dataframe(df, symbol)

                if signal:
                    if signal.signal_type == '1buy':
                        result.signals_1buy.append(signal)
                    elif signal.signal_type == '2buy':
                        result.signals_2buy.append(signal)
                    elif signal.signal_type == '3buy':
                        result.signals_3buy.append(signal)

            except Exception as e:
                logger.debug(f"处理文件 {filepath} 失败: {e}")

        return result

    def _analyze_buy_points(self, kline: KLine, symbol: str) -> Optional[BuySignal]:
        """
        分析买入点

        Args:
            kline: K线对象
            symbol: 股票代码

        Returns:
            买入信号
        """
        # 识别分型
        fractal_detector = FractalDetector(kline, confirm_required=False)
        fractals = fractal_detector.get_fractals()

        if len(fractals) < 3:
            return None

        # 生成笔
        stroke_gen = StrokeGenerator(kline, fractals)
        strokes = stroke_gen.get_strokes()

        if len(strokes) < 3:
            return None

        # 识别中枢
        pivot_detector = PivotDetector(kline, strokes)
        pivots = pivot_detector.get_pivots()

        # 获取当前价格
        df = kline.to_dataframe()
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        # datetime在索引中
        if isinstance(df.index[-1], pd.Timestamp):
            current_datetime = df.index[-1].to_pydatetime()
        else:
            current_datetime = df.index[-1]

        # 获取股票名称
        name = self._get_stock_name(symbol)

        # 计算MACD
        macd_divergence = False
        if self.use_macd:
            try:
                macd = MACD(df['close'])
                if len(macd) > 20:
                    has_div, _ = macd.check_divergence(
                        len(macd) - 20,
                        len(macd) - 1,
                        'down'
                    )
                    macd_divergence = has_div
            except Exception:
                pass

        # 检查1买
        signal_1buy = self._check_first_buy(
            symbol, name, current_price, current_datetime,
            strokes, pivots, macd_divergence
        )
        if signal_1buy:
            return signal_1buy

        # 检查2买
        signal_2buy = self._check_second_buy(
            symbol, name, current_price, current_datetime,
            strokes, pivots
        )
        if signal_2buy:
            return signal_2buy

        # 检查3买
        signal_3buy = self._check_third_buy(
            symbol, name, current_price, current_datetime,
            strokes, pivots
        )
        if signal_3buy:
            return signal_3buy

        return None

    def _check_first_buy(
        self,
        symbol: str,
        name: str,
        price: float,
        dt: datetime,
        strokes: List,
        pivots: List,
        macd_divergence: bool
    ) -> Optional[BuySignal]:
        """
        检查第一类买点

        条件：
        1. 存在中枢
        2. 当前价格低于中枢下沿
        3. 正在形成向上反弹或接近支撑
        """
        if not pivots:
            return None

        if len(strokes) < 2:
            return None

        last_pivot = pivots[-1]
        last_stroke = strokes[-1]

        # 价格必须在中枢下方
        if price >= last_pivot.low:
            return None

        # 检查是否正在反弹（向上笔）
        is_rebounding = last_stroke.is_up

        # 计算距离中枢的距离
        distance = (last_pivot.low - price) / last_pivot.low

        confidence = 0.6
        reason = "价格在中枢下方"

        if is_rebounding:
            confidence = 0.75
            reason = "跌破中枢后开始反弹"

            # 检查是否有连续向上笔（更强的反弹信号）
            if len(strokes) >= 2 and strokes[-2].is_up:
                confidence = 0.8
                reason = "连续向上反弹，确认支撑"
        elif distance < 0.1:  # 接近中枢下沿
            confidence = 0.7
            reason = "接近中枢下沿支撑位"

        if self.use_macd and macd_divergence:
            confidence = min(confidence + 0.1, 0.95)
            reason += " + MACD背驰"

        return BuySignal(
            symbol=symbol,
            name=name,
            signal_type='1buy',
            price=price,
            datetime=dt,
            confidence=confidence,
            reason=reason,
            pivot_high=last_pivot.high,
            pivot_low=last_pivot.low,
            macd_divergence=macd_divergence
        )

    def _check_second_buy(
        self,
        symbol: str,
        name: str,
        price: float,
        dt: datetime,
        strokes: List,
        pivots: List
    ) -> Optional[BuySignal]:
        """检查第二类买点"""
        if symbol not in self._first_buy_triggered:
            return None

        first_buy_price = self._first_buy_triggered[symbol]

        # 回抽不破前低
        if price < first_buy_price * 0.95:
            del self._first_buy_triggered[symbol]
            return None

        if first_buy_price * 0.95 <= price <= first_buy_price * 1.05:
            if strokes and not strokes[-1].is_up:
                return BuySignal(
                    symbol=symbol,
                    name=name,
                    signal_type='2buy',
                    price=price,
                    datetime=dt,
                    confidence=0.75,
                    reason=f"回抽不破前低，形成支撑"
                )

        return None

    def _check_third_buy(
        self,
        symbol: str,
        name: str,
        price: float,
        dt: datetime,
        strokes: List,
        pivots: List
    ) -> Optional[BuySignal]:
        """
        检查第三类买点

        条件：
        1. 价格突破中枢上沿
        2. 回踩不破中枢上沿
        """
        if not pivots or len(strokes) < 3:
            return None

        last_pivot = pivots[-1]

        # 检查最近是否有突破中枢
        recent_strokes = strokes[-5:] if len(strokes) >= 5 else strokes
        has_breakout = any(
            s.high > last_pivot.high * 1.01 for s in recent_strokes
        )

        if not has_breakout:
            return None

        # 当前价格在中枢上沿附近或上方（回踩确认）
        if price >= last_pivot.high * 0.97:
            # 检查是否形成向上笔
            if strokes[-1].is_up:
                return BuySignal(
                    symbol=symbol,
                    name=name,
                    signal_type='3buy',
                    price=price,
                    datetime=dt,
                    confidence=0.7,
                    reason=f"突破中枢后确认，中枢上沿支撑有效",
                    pivot_high=last_pivot.high,
                    pivot_low=last_pivot.low
                )

            # 或者是回踩后形成的底分型
            if len(strokes) >= 2 and strokes[-1].is_down and strokes[-2].is_up:
                if price >= last_pivot.high * 0.98:
                    return BuySignal(
                        symbol=symbol,
                        name=name,
                        signal_type='3buy',
                        price=price,
                        datetime=dt,
                        confidence=0.65,
                        reason=f"突破后回踩，中枢上沿支撑",
                        pivot_high=last_pivot.high,
                        pivot_low=last_pivot.low
                    )

        return None

    def _get_stock_name(self, symbol: str) -> str:
        """获取股票名称"""
        # 标准化代码
        if not symbol.startswith(('sh', 'sz', 'bj')):
            # 添加前缀进行查找
            for prefix in ['sh', 'sz', 'bj']:
                key = f"{prefix}{symbol}"
                if key in self._stock_names:
                    return self._stock_names[key]

        if symbol in self._stock_names:
            return self._stock_names[symbol]

        return "未知"


def print_scan_result(result: ScanResult):
    """打印扫描结果"""
    print("\n" + "=" * 60)
    print(f"[缠论选股扫描结果]")
    print(f"扫描时间: {result.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"扫描数量: {result.total_scanned} 只")
    print(f"发现信号: {result.total_signals} 个")
    print("=" * 60)

    # 1买信号
    if result.signals_1buy:
        print(f"\n[第一类买点 (1买)] - {len(result.signals_1buy)} 个:")
        print("-" * 60)
        for signal in sorted(result.signals_1buy, key=lambda x: x.confidence, reverse=True):
            print(f"  [{signal.symbol}] {signal.name}")
            print(f"  价格: ¥{signal.price:.2f} | 强度: {signal.confidence:.0%}")
            print(f"  描述: {signal.reason}")
            if signal.pivot_high > 0:
                print(f"  中枢: [{signal.pivot_low:.2f}, {signal.pivot_high:.2f}]")
            print()

    # 2买信号
    if result.signals_2buy:
        print(f"\n[第二类买点 (2买)] - {len(result.signals_2buy)} 个:")
        print("-" * 60)
        for signal in sorted(result.signals_2buy, key=lambda x: x.confidence, reverse=True):
            print(f"  [{signal.symbol}] {signal.name}")
            print(f"  价格: ¥{signal.price:.2f} | 强度: {signal.confidence:.0%}")
            print(f"  描述: {signal.reason}")
            print()

    # 3买信号
    if result.signals_3buy:
        print(f"\n[第三类买点 (3买)] - {len(result.signals_3buy)} 个:")
        print("-" * 60)
        for signal in sorted(result.signals_3buy, key=lambda x: x.confidence, reverse=True):
            print(f"  [{signal.symbol}] {signal.name}")
            print(f"  价格: ¥{signal.price:.2f} | 强度: {signal.confidence:.0%}")
            print(f"  描述: {signal.reason}")
            print()

    if result.total_signals == 0:
        print("\n暂无买入信号")

    print("=" * 60)


def save_scan_result(result: ScanResult, filepath: str):
    """保存扫描结果到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 缠论选股扫描结果\n")
        f.write(f"# 时间: {result.scan_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 扫描: {result.total_scanned} 只, 信号: {result.total_signals} 个\n\n")

        for signal in result.get_all_signals():
            f.write(f"{signal.symbol},{signal.name},{signal.signal_type},")
            f.write(f"{signal.price},{signal.confidence:.0%},{signal.reason}\n")

    logger.info(f"结果已保存到: {filepath}")


if __name__ == '__main__':
    # 示例：从本地TDX数据扫描
    screener = ChanLunScreener(use_macd=True)

    # 检查是否有本地TDX数据
    tdx_data_dir = "tdx_data"
    if os.path.exists(tdx_data_dir):
        result = screener.scan_local_files(tdx_data_dir, "*.json")
    else:
        # 扫描指定股票
        symbols = ['sh600519', 'sz000001', 'sz000002', 'sh600000', 'sh600036']
        result = screener.scan_multiple(symbols)

    print_scan_result(result)
