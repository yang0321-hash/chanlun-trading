"""
完整选股器 - 包含真实板块评分
评分系统：
1. 个股所在板块是当天最强/第二强板块 (+1分)
2. 个股是该板块龙头 (涨得最好) (+1分)
3. 个股和板块都在大盘下跌时上涨 (+1分)
4. 逆势环境 (+1分)
5. 市场最高连板数 >= 4天 (+1分)
"""

import os
import struct
import pandas as pd
from collections import defaultdict
from typing import Dict
from block_data_parser import BlockParser


class FinalScreener:
    """完整选股器"""

    def __init__(self, tdx_path: str):
        self.tdx_path = tdx_path
        self.vipdoc_path = os.path.join(tdx_path, "vipdoc")
        self.block_parser = BlockParser(tdx_path)

        # 加载板块数据
        self.block_parser.parse_block_files()

        # 存储数据
        self.stock_data = {}
        self.index_data = None

    def load_stock_data(self, max_stocks: int = None):
        """加载股票数据"""
        print("正在加载股票数据...")
        files = []

        for market in ['sh', 'sz']:
            lday_path = os.path.join(self.vipdoc_path, market, 'lday')
            if os.path.exists(lday_path):
                for f in os.listdir(lday_path):
                    if f.endswith('.day'):
                        code = f.replace('.day', '').replace(market, '')
                        files.append((market, code, os.path.join(lday_path, f)))

        if max_stocks:
            files = files[:max_stocks]

        for i, (market, code, filepath) in enumerate(files):
            if i % 500 == 0:
                print(f"  进度: {i}/{len(files)}")

            try:
                df = self.read_day_file(filepath)
                if len(df) > 10:
                    full_code = f"{market}{code}"
                    self.stock_data[full_code] = df
            except:
                pass

        # 加载大盘指数
        index_path = self.vipdoc_path + "/sh/lday/sh000001.day"
        if os.path.exists(index_path):
            self.index_data = self.read_day_file(index_path)

        print(f"已加载 {len(self.stock_data)} 只股票")

    def read_day_file(self, filepath: str) -> pd.DataFrame:
        """读取日线文件"""
        data = []
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(32)
                if len(chunk) < 32:
                    break
                values = struct.unpack('IIIIIfII', chunk)
                date_int = values[0]
                try:
                    date = pd.to_datetime(str(date_int), format='%Y%m%d')
                    open_p = values[1] / 100
                    high = values[2] / 100
                    low = values[3] / 100
                    close = values[4] / 100
                    if close > 0:
                        data.append({'date': date, 'open': open_p, 'high': high, 'low': low, 'close': close})
                except:
                    pass
        if data:
            df = pd.DataFrame(data).set_index('date')
            return df
        return pd.DataFrame()

    def calc_block_strength(self, date: pd.Timestamp, stock_change: Dict[str, float]) -> Dict:
        """计算板块强度 (按平均涨跌幅)"""
        block_change = defaultdict(list)

        for code, change in stock_change.items():
            blocks = self.block_parser.get_stock_blocks(code)
            for block in blocks:
                block_change[block].append(change)

        # 计算板块平均涨跌幅
        block_avg = {}
        for block, changes in block_change.items():
            if len(changes) >= 3:  # 至少3只股票
                block_avg[block] = sum(changes) / len(changes)

        # 排序找出最强板块
        sorted_blocks = sorted(block_avg.items(), key=lambda x: x[1], reverse=True)
        top_blocks = [b[0] for b in sorted_blocks[:2]]  # 前两名

        return {
            'block_avg': block_avg,
            'top_blocks': top_blocks
        }

    def calc_market_stats(self, date: pd.Timestamp) -> Dict:
        """计算市场统计"""
        up_count = 0
        total_count = 0
        max_cons = 0
        stock_change = {}

        for code, df in self.stock_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < 1:
                continue

            today = df.iloc[idx]
            yest = df.iloc[idx-1]

            if yest['close'] > 0:
                change = (today['close'] - yest['close']) / yest['close']
                stock_change[code] = change
                total_count += 1
                if change > 0:
                    up_count += 1

                # 连板统计
                cons = 0
                for j in range(idx, -1, -1):
                    row = df.iloc[j]
                    if row['open'] > 0 and (row['close'] - row['open']) / row['open'] >= 0.095:
                        cons += 1
                    else:
                        break
                max_cons = max(max_cons, cons)

        return {
            'up_count': up_count,
            'total_count': total_count,
            'max_consecutive': max_cons,
            'stock_change': stock_change
        }

    def score_stock(self, code: str, date: pd.Timestamp, stats: Dict, block_strength: Dict) -> int:
        """对股票评分"""
        score = 0
        change = stats['stock_change'].get(code, 0)

        # 条件1: 个股所在板块是当天最强/第二强板块
        blocks = self.block_parser.get_stock_blocks(code)
        top_blocks = block_strength['top_blocks']
        in_top_block = any(b in top_blocks for b in blocks)
        if in_top_block:
            score += 1

        # 条件2: 个股是该板块龙头
        if blocks:
            block_stock_changes = []
            for other_code in stats['stock_change'].keys():
                other_blocks = self.block_parser.get_stock_blocks(other_code)
                if any(b in blocks for b in other_blocks):
                    block_stock_changes.append((other_code, stats['stock_change'][other_code]))

            if block_stock_changes:
                block_stock_changes.sort(key=lambda x: x[1], reverse=True)
                if block_stock_changes[0][0] == code:
                    score += 1

        # 条件3: 个股和板块都在大盘下跌时上涨
        if self.index_data is not None and date in self.index_data.index:
            idx_idx = self.index_data.index.get_loc(date)
            if idx_idx >= 1:
                index_change = (self.index_data.iloc[idx_idx]['close'] - self.index_data.iloc[idx_idx-1]['close']) / self.index_data.iloc[idx_idx-1]['close']

                if index_change < 0 and change > 0:
                    block_avg = block_strength['block_avg']
                    block_up = False
                    for block in blocks:
                        if block in block_avg and block_avg[block] > 0:
                            block_up = True
                            break

                    if block_up:
                        score += 1

        # 条件4: 逆势环境
        if stats['total_count'] > 0:
            up_ratio = stats['up_count'] / stats['total_count']
            if up_ratio < 0.3:
                score += 1

        # 条件5: 市场最高连板数 >= 4
        if stats['max_consecutive'] >= 4:
            score += 1

        return score

    def backtest(self, start_idx: int = -200, min_score: int = 5):
        """回测"""
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))[start_idx:]

        print(f"\n开始回测，共 {len(all_dates)} 个交易日，目标分数: {min_score}")

        trades = []
        score_distribution = defaultdict(int)

        for i, date in enumerate(all_dates[:-2]):
            if i % 20 == 0:
                print(f"  进度: {i}/{len(all_dates)}")

            stats = self.calc_market_stats(date)
            block_strength = self.calc_block_strength(date, stats['stock_change'])

            for code in stats['stock_change'].keys():
                score = self.score_stock(code, date, stats, block_strength)
                score_distribution[score] += 1

                if score >= min_score:
                    buy_day = all_dates[all_dates.index(date) + 1]
                    sell_day = all_dates[all_dates.index(date) + 2]

                    df = self.stock_data[code]
                    if buy_day in df.index and sell_day in df.index:
                        buy_price = df.loc[buy_day, 'high']
                        sell_price = df.loc[sell_day, 'close']

                        profit_pct = (sell_price - buy_price) / buy_price * 100
                        trades.append({
                            'signal_date': date,
                            'code': code,
                            'score': score,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit_pct': profit_pct
                        })

        print(f"\n评分分布:")
        for s in sorted(score_distribution.keys()):
            print(f"  {s}分: {score_distribution[s]} 次")

        return trades

    def analyze_results(self, trades: list):
        """分析结果"""
        if not trades:
            print("\n没有符合条件的交易")
            return

        df = pd.DataFrame(trades)

        print("\n" + "="*60)
        print("回测结果分析")
        print("="*60)

        total = len(df)
        profit = len(df[df['profit_pct'] > 0])
        loss = len(df[df['profit_pct'] < 0])

        print(f"\n总交易: {total}")
        print(f"盈利: {profit} ({profit/total*100:.1f}%)")
        print(f"亏损: {loss} ({loss/total*100:.1f}%)")
        print(f"平均收益: {df['profit_pct'].mean():.2f}%")
        print(f"最大盈利: {df['profit_pct'].max():.2f}%")
        print(f"最大亏损: {df['profit_pct'].min():.2f}%")

        print(f"\n最后20笔交易:")
        print(df[['code', 'score', 'buy_price', 'sell_price', 'profit_pct']].tail(20).to_string())


def main():
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    screener = FinalScreener(tdx_path)
    screener.load_stock_data(max_stocks=1000)

    trades = screener.backtest(min_score=5)
    screener.analyze_results(trades)


if __name__ == "__main__":
    main()
