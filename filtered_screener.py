"""
过滤后的选股器回测
只保留：沪深主板 + 科创板 + 创业板
"""

import os
import struct
import pandas as pd
from collections import defaultdict
from typing import Dict


def is_valid_stock(code: str) -> bool:
    """判断是否为有效股票（过滤基金、债券等）"""
    # 沪主板: 600xxx, 601xxx, 603xxx, 605xxx
    if code.startswith('sh60') and not code.startswith('sh688'):
        return True
    # 深主板: 000xxx, 001xxx
    if code.startswith('sz000') or code.startswith('sz001'):
        return True
    # 科创板: 688xxx
    if code.startswith('sh688'):
        return True
    # 创业板: 300xxx
    if code.startswith('sz300'):
        return True
    return False


def get_stock_sector(code: str) -> str:
    """获取股票所属板块"""
    if code.startswith('sh688'):
        return '科创板'
    elif code.startswith('sz300'):
        return '创业板'
    elif code.startswith('sh60'):
        return '沪主板'
    elif code.startswith('sz0'):
        return '深主板'
    return '其他'


class FilteredScreener:
    """过滤后的选股器"""

    def __init__(self, tdx_path: str):
        self.tdx_path = tdx_path
        self.vipdoc_path = os.path.join(tdx_path, "vipdoc")
        self.stock_data = {}
        self.index_data = None
        self.sectors = ['沪主板', '深主板', '创业板', '科创板']

    def load_stock_data(self, max_stocks: int = None):
        """加载股票数据（只加载有效股票）"""
        print("正在加载股票数据...")
        files = []

        for market in ['sh', 'sz']:
            lday_path = os.path.join(self.vipdoc_path, market, 'lday')
            if os.path.exists(lday_path):
                for f in os.listdir(lday_path):
                    if f.endswith('.day'):
                        code = f.replace('.day', '').replace(market, '')
                        full_code = f"{market}{code}"
                        # 只加载有效股票
                        if is_valid_stock(full_code):
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

        print(f"已加载 {len(self.stock_data)} 只有效股票")

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
                    if close > 0 and close < 200:  # 过滤异常高价
                        data.append({'date': date, 'open': open_p, 'high': high, 'low': low, 'close': close})
                except:
                    pass
        if data:
            df = pd.DataFrame(data).set_index('date')
            return df
        return pd.DataFrame()

    def calc_sector_stats(self, stock_change: Dict[str, float]) -> Dict:
        """计算板块统计"""
        sector_avg = {}
        for sector in self.sectors:
            changes = [v for k, v in stock_change.items() if get_stock_sector(k) == sector]
            if changes:
                sector_avg[sector] = sum(changes) / len(changes)

        sorted_sectors = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [s[0] for s in sorted_sectors[:2]]

        return {
            'sector_avg': sector_avg,
            'top_sectors': top_sectors
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

                # 连板
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

    def score_stock(self, code: str, date: pd.Timestamp, stats: Dict, sector_stats: Dict) -> int:
        """对股票评分"""
        score = 0
        change = stats['stock_change'].get(code, 0)
        sector = get_stock_sector(code)

        # 条件1: 个股所在板块是当天最强/第二强
        if sector in sector_stats['top_sectors']:
            score += 1

        # 条件2: 个股是该板块龙头
        sector_changes = [(k, v) for k, v in stats['stock_change'].items() if get_stock_sector(k) == sector]
        if sector_changes:
            sector_changes.sort(key=lambda x: x[1], reverse=True)
            if sector_changes[0][0] == code:
                score += 1

        # 条件3: 个股和板块都在大盘下跌时上涨
        if self.index_data is not None and date in self.index_data.index:
            idx_idx = self.index_data.index.get_loc(date)
            if idx_idx >= 1:
                index_change = (self.index_data.iloc[idx_idx]['close'] - self.index_data.iloc[idx_idx-1]['close']) / self.index_data.iloc[idx_idx-1]['close']

                if index_change < 0 and change > 0:
                    sector_avg = sector_stats['sector_avg']
                    if sector in sector_avg and sector_avg[sector] > 0:
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

    def backtest(self, start_idx: int = -200, min_score: int = 3, buy_at: str = 'close'):
        """
        回测
        buy_at: 'close'=信号日收盘价买入, 'open'=次日开盘价买入, 'high'=次日涨停价买入
        """
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))[start_idx:]

        buy_method = {'close': '信号日收盘价', 'open': '次日开盘价', 'high': '次日涨停价'}.get(buy_at, buy_at)
        print(f"\n开始回测，共 {len(all_dates)} 个交易日，目标分数: {min_score}，买入方式: {buy_method}")

        trades = []
        score_distribution = defaultdict(int)

        for i, date in enumerate(all_dates[:-2]):
            if i % 20 == 0:
                print(f"  进度: {i}/{len(all_dates)}")

            stats = self.calc_market_stats(date)
            sector_stats = self.calc_sector_stats(stats['stock_change'])

            for code in stats['stock_change'].keys():
                score = self.score_stock(code, date, stats, sector_stats)
                score_distribution[score] += 1

                if score >= min_score:
                    df = self.stock_data[code]

                    if buy_at == 'close':
                        # 信号日收盘价买入，次日收盘卖出
                        if date not in df.index:
                            continue
                        buy_date = date
                        sell_date = all_dates[all_dates.index(date) + 1]
                    else:
                        # 次日买入，第三天卖出
                        buy_date = all_dates[all_dates.index(date) + 1]
                        sell_date = all_dates[all_dates.index(date) + 2]

                    if buy_date not in df.index or sell_date not in df.index:
                        continue

                    if buy_at == 'close':
                        buy_price = df.loc[buy_date, 'close']
                    elif buy_at == 'open':
                        buy_price = df.loc[buy_date, 'open']
                    else:  # high
                        buy_price = df.loc[buy_date, 'high']

                    sell_price = df.loc[sell_date, 'close']

                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    trades.append({
                        'signal_date': date,
                        'code': code,
                        'score': score,
                        'sector': get_stock_sector(code),
                        'buy_date': buy_date,
                        'buy_price': buy_price,
                        'sell_date': sell_date,
                        'sell_price': sell_price,
                        'profit_pct': profit_pct
                    })

        # 打印评分分布
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

        # 按板块统计
        print(f"\n按板块统计:")
        for sector in self.sectors:
            sector_df = df[df['sector'] == sector]
            if len(sector_df) > 0:
                win_rate = len(sector_df[sector_df['profit_pct'] > 0]) / len(sector_df) * 100
                avg_ret = sector_df['profit_pct'].mean()
                print(f"  {sector}: {len(sector_df)}笔, 胜率{win_rate:.1f}%, 平均收益{avg_ret:.2f}%")

        print(f"\n最后20笔交易:")
        print(df[['code', 'sector', 'score', 'buy_price', 'sell_price', 'profit_pct']].tail(20).to_string())


def main():
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    screener = FilteredScreener(tdx_path)
    screener.load_stock_data()

    # 测试不同买入方式
    buy_methods = ['close', 'open']  # 不用high了，因为效果不好

    for buy_at in buy_methods:
        print(f"\n{'='*60}")
        print(f"买入方式: {buy_at}")
        print(f"{'='*60}")

        trades = screener.backtest(min_score=3, buy_at=buy_at)
        screener.analyze_results(trades)


if __name__ == "__main__":
    main()
