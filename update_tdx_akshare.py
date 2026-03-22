"""
通达信TDX数据更新脚本 - AKShare在线版

使用AKShare获取最新数据，替代TDX数据
兼容原有TDX数据格式
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import time
import json

# 确保输出编码正确
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class AKShareDataUpdater:
    """AKShare数据更新器"""

    def __init__(self, output_dir: str = "test_output"):
        """
        初始化更新器

        Args:
            output_dir: 数据输出目录
        """
        self.output_dir = output_dir
        self.updated_count = 0
        self.failed_count = 0
        self.failed_stocks = []

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 检查akshare是否安装
        try:
            import akshare as ak
            self.ak = ak
            self.ak_available = True
        except ImportError:
            self.ak_available = False
            print("警告: akshare未安装")
            print("安装命令: pip install akshare")

    def update_stock(self, symbol: str, period: str = 'daily') -> bool:
        """
        更新单只股票数据

        Args:
            symbol: 股票代码，如 'sh600000', 'sz000001'
            period: 周期类型 daily/weekly/monthly/1min/5min...

        Returns:
            bool: 是否成功
        """
        if not self.ak_available:
            return False

        try:
            # 标准化代码
            code = symbol.replace('sh', '').replace('sz', '')

            # 设置日期范围（获取最近3年数据）
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')

            # 获取数据
            if period == 'daily':
                df = self.ak.stock_zh_a_hist(
                    symbol=code,
                    period='daily',
                    start_date=start_date,
                    end_date=end_date,
                    adjust='qfq'
                )
            elif period == 'weekly':
                df = self.ak.stock_zh_a_hist(
                    symbol=code,
                    period='weekly',
                    start_date=start_date,
                    end_date=end_date,
                    adjust='qfq'
                )
            else:
                print(f"不支持的周期: {period}")
                return False

            if df is None or df.empty:
                return False

            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            })

            # 确保必需列存在
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return False

            # 添加datetime列
            df['datetime'] = pd.to_datetime(df['date'])

            # 保存为JSON（兼容TDX格式）
            filepath = os.path.join(self.output_dir, f"{symbol}.day.json")
            df[['date', 'open', 'high', 'low', 'close', 'amount', 'volume']].to_json(
                filepath,
                orient='records',
                date_format='iso',
                force_ascii=False,
                indent=None
            )

            return True

        except Exception as e:
            return False

    def update_batch(
        self,
        symbols: List[str],
        period: str = 'daily',
        delay: float = 0.5
    ) -> None:
        """
        批量更新股票数据

        Args:
            symbols: 股票代码列表
            period: 周期类型
            delay: 请求间隔（秒）
        """
        print("=" * 60)
        print("股票数据批量更新")
        print(f"数据源: AKShare在线")
        print(f"目标数量: {len(symbols)}")
        print(f"输出目录: {self.output_dir}")
        print(f"数据周期: {period}")
        print("=" * 60)

        start_time = datetime.now()

        for i, symbol in enumerate(symbols):
            if i % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                remaining = len(symbols) - i
                eta = elapsed / (i + 1) * remaining if i > 0 else 0
                print(f"进度: {i}/{len(symbols)} | "
                      f"成功: {self.updated_count} | "
                      f"失败: {self.failed_count} | "
                      f"预计剩余: {int(eta)}秒")

            success = self.update_stock(symbol, period)

            if success:
                self.updated_count += 1
                latest_price = self._get_latest_price(symbol)
                print(f"  {symbol} ✓ 最新价: ¥{latest_price:.2f}")
            else:
                self.failed_count += 1
                self.failed_stocks.append(symbol)
                print(f"  {symbol} ✗ 更新失败")

            # 请求间隔
            time.sleep(delay)

        # 总结
        elapsed = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("更新完成")
        print(f"总耗时: {int(elapsed)}秒 ({elapsed/60:.1f}分钟)")
        print(f"成功: {self.updated_count}")
        print(f"失败: {self.failed_count}")

        if self.failed_stocks:
            print(f"\n失败股票: {', '.join(self.failed_stocks[:10])}")
            if len(self.failed_stocks) > 10:
                print(f"  ... 等共 {len(self.failed_stocks)} 只")

        print("=" * 60)

        # 保存更新记录
        self._save_update_log(symbols)

    def _get_latest_price(self, symbol: str) -> float:
        """获取最新价格"""
        try:
            filepath = os.path.join(self.output_dir, f"{symbol}.day.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return float(data[-1]['close'])
        except:
            return 0.0

    def _save_update_log(self, symbols: List[str]) -> None:
        """保存更新日志"""
        log_file = os.path.join(self.output_dir, "update_log.txt")

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# 数据更新日志\n")
            f.write(f"# 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 成功: {self.updated_count}\n")
            f.write(f"# 失败: {self.failed_count}\n\n")

            if self.failed_stocks:
                f.write("# 失败股票:\n")
                for stock in self.failed_stocks:
                    f.write(f"{stock}\n")

            f.write(f"\n# 全部股票:\n")
            for stock in symbols:
                status = "OK" if stock not in self.failed_stocks else "FAIL"
                f.write(f"{stock} {status}\n")


def get_popular_stocks(limit: int = 100) -> List[str]:
    """获取热门股票列表"""
    try:
        import akshare as ak

        # 获取沪深300成分股
        print("获取沪深300成分股...")
        df = ak.index_stock_cons(symbol="沪深300")

        if df is not None and not df.empty:
            codes = df['品种代码'].tolist()
            # 添加市场前缀
            symbols = []
            for code in codes:
                if code.startswith('6'):
                    symbols.append(f"sh{code}")
                elif code.startswith(('0', '3')):
                    symbols.append(f"sz{code}")
            return symbols[:limit]

    except Exception as e:
        print(f"获取热门股票失败: {e}")

    # 备选：返回常用股票
    return [
        'sh600519', 'sh600000', 'sh600036', 'sh601318', 'sh601398',
        'sh600276', 'sh600030', 'sh601316', 'sh601766', 'sh600887',
        'sz000001', 'sz000002', 'sz300015', 'sz300059', 'sz300750'
    ]


def get_local_stock_list() -> List[str]:
    """从本地获取股票列表"""
    data_dir = "test_output"

    if not os.path.exists(data_dir):
        return []

    import glob
    files = glob.glob(os.path.join(data_dir, "*.json"))

    symbols = []
    for f in files:
        symbol = os.path.basename(f).replace('.day.json', '').replace('.json', '')
        symbols.append(symbol)

    return symbols


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='股票数据批量更新')
    parser.add_argument('--source', choices=['akshare', 'local'], default='akshare',
                       help='数据源: akshare在线/local更新')
    parser.add_argument('--limit', type=int, default=50,
                       help='更新数量限制 (默认50)')
    parser.add_argument('--output', default='test_output',
                       help='输出目录')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='请求间隔(秒)')

    args = parser.parse_args()

    if args.source == 'local':
        # 使用本地股票列表
        symbols = get_local_stock_list()
        if args.limit:
            symbols = symbols[:args.limit]
        print(f"本地股票列表: {len(symbols)} 只")
    else:
        # 使用热门股票列表
        symbols = get_popular_stocks(limit=args.limit)

    if not symbols:
        print("未获取到股票列表")
        sys.exit(1)

    # 执行更新
    updater = AKShareDataUpdater(output_dir=args.output)
    updater.update_batch(symbols, period='daily', delay=args.delay)

    # 更新完成后提示
    print("\n数据更新完成！可以使用以下命令扫描:")
    print(f"  python final_screener.py --limit 100")
