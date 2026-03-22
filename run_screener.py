"""
缠论选股器 - 主程序

使用方法:
    # 扫描本地数据（默认）
    python run_screener.py

    # 扫描指定股票 (在线数据)
    python run_screener.py --symbols sh600519,sz000001

    # 不使用MACD
    python run_screener.py --no-macd

    # 保存结果
    python run_screener.py --output signals.txt
"""

import argparse
import os
import sys

# 确保输出编码正确
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from loguru import logger
from scanner.chanlun_screener import ChanLunScreener, print_scan_result, save_scan_result


def main():
    parser = argparse.ArgumentParser(description='缠论选股器 - 识别1买/2买/3买信号')
    parser.add_argument(
        '--symbols',
        type=str,
        help='股票代码列表，逗号分隔，如: sh600519,sz000001'
    )
    parser.add_argument(
        '--local',
        type=str,
        nargs='?',
        const='test_output',
        help='本地数据目录（默认: test_output）'
    )
    parser.add_argument(
        '--no-macd',
        action='store_true',
        help='不使用MACD背驰判断'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出结果到文件'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，只输出结果'
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

    # 如果没有指定任何参数，默认扫描本地数据
    if not args.symbols and not args.local:
        args.local = 'test_output'

    # 初始化选股器
    screener = ChanLunScreener(
        use_macd=not args.no_macd,
        min_klines=60
    )

    result = None

    try:
        # 执行扫描
        if args.local:
            data_dir = args.local
            if not os.path.exists(data_dir):
                print(f"错误: 目录不存在 - {data_dir}")
                print("\n请先准备TDX数据，或使用 --symbols 参数扫描指定股票")
                return

            print(f"扫描本地数据: {data_dir}")
            result = screener.scan_local_files(data_dir, "*.json")

        elif args.symbols:
            symbols = args.symbols.split(',')
            symbols = [s.strip() for s in symbols if s.strip()]
            print(f"扫描指定股票: {symbols}")
            result = screener.scan_multiple(symbols)

        # 输出结果
        if result:
            print_scan_result(result)

            # 保存到文件
            if args.output:
                save_scan_result(result, args.output)

    except KeyboardInterrupt:
        print("\n\n扫描已中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
