"""
选股器回测系统
策略：获得5分的个股，第二天涨停价买入，第三天计算盈利率
"""

import os
import struct
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import pickle


class BacktestScreener:
    """选股器回测"""

    def __init__(self, tdx_path: str):
        self.tdx_path = tdx_path
        self.vipdoc_path = os.path.join(tdx_path, "vipdoc")
        self.block_path = os.path.join(tdx_path, "T0002", "blocknew")

        # 涨停阈值
        self.limit_main = 0.095  # 主板
        self.limit_gem = 0.195   # 创业板/科创板

        # 存储数据
        self.stock_data = {}  # 股票代码 -> DataFrame
        self.index_data = None  # 大盘指数

        # 回测缓存
        self.cache_file = "screener_cache.pkl"

    def list_day_files(self) -> List[str]:
        """列出所有日线文件"""
        files = []
        for market in ['sh', 'sz']:
            lday_path = os.path.join(self.vipdoc_path, market, 'lday')
            if os.path.exists(lday_path):
                for f in os.listdir(lday_path):
                    if f.endswith('.day'):
                        code = f.replace('.day', '')
                        files.append((market, code, os.path.join(lday_path, f)))
        return files

    def read_day_file(self, filepath: str) -> pd.DataFrame:
        """读取通达信日线文件"""
        data = []
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(32)
                if len(chunk) < 32:
                    break
                values = struct.unpack('IIIIIfII', chunk)
                date_int = values[0]
                date_str = str(date_int)
                if len(date_str) == 8:
                    try:
                        date = pd.to_datetime(date_str, format='%Y%m%d')
                        open_p = values[1] / 100
                        high = values[2] / 100
                        low = values[3] / 100
                        close = values[4] / 100
                        if close > 0:
                            data.append({
                                'date': date,
                                'open': open_p,
                                'high': high,
                                'low': low,
                                'close': close,
                            })
                    except:
                        pass

        if data:
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df
        return pd.DataFrame()

    def load_data(self, max_stocks: int = None):
        """加载股票数据"""
        print("正在加载数据...")
        files = self.list_day_files()
        if max_stocks:
            files = files[:max_stocks]

        for i, (market, code, filepath) in enumerate(files):
            if i % 500 == 0:
                print(f"  进度: {i}/{len(files)}")
            try:
                df = self.read_day_file(filepath)
                if len(df) > 10:  # 至少10天数据
                    full_code = f"{market}{code}"
                    self.stock_data[full_code] = df
            except:
                pass

        # 加载大盘指数
        index_path = self.vipdoc_path + "/sh/lday/sh000001.day"
        if os.path.exists(index_path):
            self.index_data = self.read_day_file(index_path)

        print(f"已加载 {len(self.stock_data)} 只股票")

    def get_limit_threshold(self, code: str) -> float:
        """获取涨停阈值"""
        if code.startswith('3') or code.startswith('688'):
            return self.limit_gem
        return self.limit_main

    def is_limit_up(self, open_price, close_price) -> bool:
        """判断是否涨停"""
        if open_price <= 0:
            return False
        change = (close_price - open_price) / open_price
        return change >= 0.095

    def calc_daily_stats(self, date: pd.Timestamp) -> Dict:
        """计算某天的市场统计"""
        up_count = 0
        down_count = 0
        limit_up_count = 0
        max_consecutive = {}  # code -> consecutive days
        stock_change = {}

        for code, df in self.stock_data.items():
            if date not in df.index:
                continue

            idx = df.index.get_loc(date)
            if idx < 1:
                continue

            today = df.iloc[idx]
            yesterday = df.iloc[idx - 1]

            # 涨跌幅
            if yesterday['close'] > 0:
                change = (today['close'] - yesterday['close']) / yesterday['close']
                stock_change[code] = change

                if change > 0:
                    up_count += 1
                elif change < 0:
                    down_count += 1

                # 涨停判断
                if self.is_limit_up(today['open'], today['close']):
                    limit_up_count += 1

            # 连板统计
            consecutive = 0
            for j in range(idx, -1, -1):
                row = df.iloc[j]
                if self.is_limit_up(row['open'], row['close']):
                    consecutive += 1
                else:
                    break
            if consecutive > 0:
                max_consecutive[code] = max(max_consecutive.get(code, 0), consecutive)

        max_cons = max(max_consecutive.values()) if max_consecutive else 0

        return {
            'up_count': up_count,
            'down_count': down_count,
            'total_count': len(stock_change),
            'limit_up_count': limit_up_count,
            'max_consecutive': max_cons,
            'stock_change': stock_change
        }

    def score_stock(self, code: str, date: pd.Timestamp, stats: Dict) -> int:
        """对股票评分"""
        score = 0

        if code not in self.stock_data:
            return 0

        df = self.stock_data[code]
        if date not in df.index:
            return 0

        idx = df.index.get_loc(date)
        if idx < 1:
            return 0

        today = df.iloc[idx]
        yesterday = df.iloc[idx - 1]

        # 当日涨跌幅
        change = stats['stock_change'].get(code, 0)

        # 条件1: 个股所在板块是当天最强/第二强板块
        # (需要板块数据，暂跳过)

        # 条件2: 个股是该板块龙头
        # (需要板块数据，暂跳过)

        # 条件3: 个股和板块都在大盘下跌时上涨
        if self.index_data is not None and date in self.index_data.index:
            idx_idx = self.index_data.index.get_loc(date)
            if idx_idx >= 1:
                index_today = self.index_data.iloc[idx_idx]
                index_yest = self.index_data.iloc[idx_idx - 1]
                index_change = (index_today['close'] - index_yest['close']) / index_yest['close']

                if index_change < 0 and change > 0:
                    score += 1

        # 条件4: 逆势环境
        total = stats['total_count']
        if total > 0:
            up_ratio = stats['up_count'] / total
            if up_ratio < 0.3:  # 上涨少于30%
                score += 1

        # 条件5: 市场最高连板数 >= 4
        if stats['max_consecutive'] >= 4:
            score += 1

        # 暂时用其他条件补齐到5分用于测试
        # 条件1和2需要板块数据，先给所有股票加2分
        score += 2

        return score

    def backtest(self, start_date: str = None, end_date: str = None, min_score: int = 4):
        """执行回测"""
        # 获取所有交易日期
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))

        if start_date:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date)]

        # 结果记录
        trades = []  # (buy_date, code, buy_price, sell_date, sell_price, profit_pct)
        score_stats = defaultdict(int)  # 评分统计

        print(f"\n开始回测，共 {len(all_dates)} 个交易日，目标分数: {min_score}")

        # 遍历每个交易日
        for i, date in enumerate(all_dates[:-2]):  # 需要后两天数据
            if i % 50 == 0:
                print(f"  处理中: {i}/{len(all_dates)}")

            # 计算当天统计
            stats = self.calc_daily_stats(date)

            # 评分
            qualified_stocks = []
            for code in self.stock_data.keys():
                score = self.score_stock(code, date, stats)
                if score >= min_score:
                    qualified_stocks.append((code, score))
                    score_stats[score] += 1

            if not qualified_stocks:
                continue

            # 第二天买入
            next_day = all_dates[all_dates.index(date) + 1]

            for code, score in qualified_stocks:
                df = self.stock_data[code]
                if next_day not in df.index:
                    continue

                idx = df.index.get_loc(next_day)
                buy_day = df.iloc[idx]

                # 涨停价买入 (使用当天最高价作为涨停价)
                buy_price = buy_day['high']

                # 第三天卖出
                if i + 2 < len(all_dates):
                    sell_day = all_dates[all_dates.index(date) + 2]
                    if sell_day in df.index:
                        sell_idx = df.index.get_loc(sell_day)
                        sell_data = df.iloc[sell_idx]
                        sell_price = sell_data['close']

                        profit_pct = (sell_price - buy_price) / buy_price * 100
                        trades.append({
                            'signal_date': date,
                            'code': code,
                            'score': score,
                            'buy_date': next_day,
                            'buy_price': buy_price,
                            'sell_date': sell_day,
                            'sell_price': sell_price,
                            'profit_pct': profit_pct
                        })

        # 打印评分统计
        print(f"\n评分分布:")
        for s in sorted(score_stats.keys()):
            print(f"  {s}分: {score_stats[s]} 次")

        return trades

    def analyze_results(self, trades: List[Dict]):
        """分析回测结果"""
        if not trades:
            print("没有交易记录")
            return

        df = pd.DataFrame(trades)

        print("\n" + "="*60)
        print("回测结果分析")
        print("="*60)

        # 基本统计
        total_trades = len(df)
        profit_trades = len(df[df['profit_pct'] > 0])
        loss_trades = len(df[df['profit_pct'] < 0])

        print(f"\n总交易次数: {total_trades}")
        print(f"盈利次数: {profit_trades} ({profit_trades/total_trades*100:.1f}%)")
        print(f"亏损次数: {loss_trades} ({loss_trades/total_trades*100:.1f}%)")

        # 盈利率统计
        avg_profit = df[df['profit_pct'] > 0]['profit_pct'].mean() if profit_trades > 0 else 0
        avg_loss = df[df['profit_pct'] < 0]['profit_pct'].mean() if loss_trades > 0 else 0
        total_avg = df['profit_pct'].mean()

        print(f"\n平均盈利: {avg_profit:.2f}%")
        print(f"平均亏损: {avg_loss:.2f}%")
        print(f"总平均收益: {total_avg:.2f}%")

        # 分布统计
        print(f"\n最大盈利: {df['profit_pct'].max():.2f}%")
        print(f"最大亏损: {df['profit_pct'].min():.2f}%")

        # 区间统计
        print(f"\n盈利 > 5%: {len(df[df['profit_pct'] > 5])} 次")
        print(f"盈利 0-5%: {len(df[(df['profit_pct'] > 0) & (df['profit_pct'] <= 5)])} 次")
        print(f"亏损 0-5%: {len(df[(df['profit_pct'] < 0) & (df['profit_pct'] >= -5)])} 次")
        print(f"亏损 > 5%: {len(df[df['profit_pct'] < -5])} 次")

        # 胜率分析
        print(f"\n胜率: {profit_trades/total_trades*100:.2f}%")

        print("\n" + "="*60)

        # 显示最近交易
        print("\n最近10笔交易:")
        print(df.tail(10).to_string())


def main():
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    screener = BacktestScreener(tdx_path)

    # 加载数据
    screener.load_data(max_stocks=500)

    # 执行回测 (4分股票)
    trades = screener.backtest(min_score=4)

    # 分析结果
    screener.analyze_results(trades)


if __name__ == "__main__":
    main()
