#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
周线+日线 多级别缠论策略回测

交易规则：
1. 周线级别2买买入
2. 跌破1买低点止损
3. 日线级别MACD顶背离减仓50%
4. 日线级别2卖卖出剩余
"""

import json
import sys
import os
import subprocess
import argparse
import shutil
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datetime import datetime
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy


def parse_tdx_data(symbol: str) -> bool:
    """调用tdx-parser解析通达信数据"""
    project_root = Path(__file__).parent
    tdx_parser_dir = project_root / ".claude" / "skills" / "tdx-parser" / "scripts"
    parse_script = tdx_parser_dir / "parse_tdx.js"

    if not parse_script.exists():
        print(f"警告: tdx-parser脚本不存在: {parse_script}")
        return False

    if not shutil.which("node"):
        print("警告: Node.js未安装，无法运行tdx-parser")
        return False

    # 检查通达信数据目录是否存在
    tdx_paths = [
        'D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc',
        'D:/新建通达信/vipdoc',
        'D:/通达信/vipdoc',
        'C:/新建通达信/vipdoc',
        'C:/通达信/vipdoc',
    ]
    tdx_path = None
    for p in tdx_paths:
        if Path(p).exists():
            tdx_path = p
            break

    if not tdx_path:
        print("警告: 未找到通达信数据目录，无法自动解析")
        print("请确保通达信软件已安装，或手动将数据文件放到 test_output/ 目录")
        return False

    try:
        # 提取市场前缀和代码
        market = "sh" if symbol.startswith(("sh", "6", "900")) else "sz"
        code = symbol.replace("sh", "").replace("sz", "").lower()

        # 构建输出目录
        output_dir = project_root / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 调用tdx-parser
        cmd = [
            "node",
            str(parse_script),
            "--input", tdx_path,
            "--code", f"{market}{code}",
            "--output", str(output_dir),
            "--format", "json",
            "--date-format", "iso"
        ]

        print(f"正在调用 tdx-parser 解析 {symbol} 数据...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')

        if result.returncode == 0:
            print(f"成功解析: {output_dir}/{symbol.lower()}.day.json")
            return True
        else:
            print(f"解析失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"解析出错: {e}")
        return False


def load_tdx_data(json_file: str) -> pd.DataFrame:
    """加载通达信JSON数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 确保有必需的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")

    return df


def print_results(results: dict, symbol: str = "N/A"):
    """打印回测结果"""
    print("\n" + "=" * 60)
    print(f"回测结果 - {symbol}")
    print("=" * 60)

    print(f"\n核心指标:")
    print(f"  初始资金: {results.get('initial_capital', 100000):,.2f}")
    print(f"  最终资金: {results.get('final_equity', 0):,.2f}")
    print(f"  总收益率: {results.get('total_return', 0)*100:.2f}%")
    print(f"  年化收益: {results.get('annual_return', 0)*100:.2f}%")

    print(f"\n风险指标:")
    print(f"  夏普比率: {results.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {results.get('max_drawdown', 0)*100:.2f}%")

    print(f"\n交易统计:")
    print(f"  总交易次数: {results.get('total_trades', 0)}")
    print(f"  胜率: {results.get('win_rate', 0)*100:.2f}%")
    print(f"  盈亏比: {results.get('profit_loss_ratio', 0):.2f}")

    # 打印交易明细
    trades = results.get('trades', [])
    if trades:
        print(f"\n交易明细:")
        print(f"{'日期':<12} {'操作':<6} {'价格':<10} {'数量':<8} {'理由'}")
        print("-" * 60)
        for t in trades:
            action = "买入" if t.signal_type.value == 'buy' else "卖出"
            date_str = t.datetime.strftime('%Y-%m-%d') if hasattr(t.datetime, 'strftime') else str(t.datetime)[:10]
            print(f"{date_str:<12} {action:<6} CNY{t.price:>7.2f} {t.quantity:>6} {t.reason}")

    # 买入持有比较
    print("\n" + "=" * 60)


def run_single_symbol_backtest(symbol: str, data_file: str):
    """单个股票回测"""
    print(f"\n{'='*60}")
    print(f"回测标的: {symbol}")
    print(f"{'='*60}")

    # 加载数据
    df = load_tdx_data(data_file)
    print(f"\n数据概览:")
    print(f"  日期范围: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  K线数量: {len(df)} 条")
    print(f"  最新价格: CNY{df['close'].iloc[-1]:.2f}")
    print(f"  期间涨跌: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")

    # 买入持有收益
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    print(f"  买入持有收益: {buy_hold_return*100:.2f}%")

    # 创建策略
    strategy = WeeklyDailyChanLunStrategy(
        name='周日线缠论策略',
        weekly_min_strokes=3,
        daily_min_strokes=3,
        stop_loss_pct=0.08,
        exit_ratio=0.5
    )

    # 创建回测引擎
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100
    )

    engine = BacktestEngine(config)
    engine.add_data(symbol, df)
    engine.set_strategy(strategy)

    # 运行回测
    results = engine.run()
    results['initial_capital'] = config.initial_capital

    # 打印结果
    print_results(results, symbol)

    return results, df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='周线+日线缠论策略回测')
    parser.add_argument('symbols', nargs='*', help='股票代码 (如: sz002600)')
    parser.add_argument('--data-dir', default='test_output', help='数据目录')

    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")

    print("\n" + "="*60)
    print("周线+日线 多级别缠论策略回测")
    print("="*60)
    print("\n策略规则:")
    print("  1. 周线级别2买买入")
    print("  2. 跌破1买低点止损")
    print("  3. 日线MACD顶背离减仓50%")
    print("  4. 日线2卖清仓")

    # 确定回测标的
    if args.symbols:
        symbols = [(s, f"{args.data_dir}/{s.lower()}.day.json") for s in args.symbols]
    else:
        # 默认股票
        symbols = [
            ('sz002600', f'{args.data_dir}/sz002600.day.json'),
        ]

    all_results = {}

    for symbol, data_file in symbols:
        # 检查数据文件是否存在
        if not os.path.exists(data_file):
            print(f"\n数据文件不存在: {data_file}")
            print(f"尝试调用 tdx-parser 解析 {symbol} 数据...")
            if parse_tdx_data(symbol):
                print("继续回测...")
            else:
                print(f"无法获取 {symbol} 数据，跳过")
                continue

        try:
            results, df = run_single_symbol_backtest(symbol, data_file)
            all_results[symbol] = (results, df)

            # 与买入持有比较
            buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            strategy_return = results.get('total_return', 0) * 100
            print(f"\n策略 vs 买入持有:")
            print(f"  策略收益: {strategy_return:.2f}%")
            print(f"  买入持有: {buy_hold_return:.2f}%")
            print(f"  超额收益: {strategy_return - buy_hold_return:.2f}%")

        except Exception as e:
            print(f"\n回测失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("回测完成!")
    print("="*60)


if __name__ == "__main__":
    main()
