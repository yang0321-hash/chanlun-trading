"""
通达信选股器 - 评分系统
条件：
1. 个股所在板块是当天最强/第二强板块 (+1分)
2. 个股是该板块龙头 (+1分)
3. 个股和板块都在大盘下跌时上涨 (+1分)
4. 逆势环境 (+1分)
5. 市场最高连板数 >= 4天 (+1分)
"""

import os
import struct
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict


class TdxScreener:
    """通达信选股器"""

    def __init__(self, tdx_path: str):
        self.tdx_path = tdx_path
        self.vipdoc_path = os.path.join(tdx_path, "vipdoc")
        self.block_path = os.path.join(tdx_path, "T0002", "blocknew")

        # 涨停阈值
        self.limit_main = 0.095  # 主板 9.5% 以上算涨停 (考虑四舍五入)
        self.limit_gem = 0.195   # 创业板/科创板 19.5% 以上

        # 存储数据
        self.stock_data = {}  # 股票代码 -> DataFrame
        self.stock_change = {}  # 股票代码 -> 涨跌幅
        self.limit_up_stocks = []  # 当日涨停股票
        self.limit_up_count = 0  # 涨停数量
        self.max_consecutive = 0  # 最高连板数
        self.board_strength = {}  # 板块强度

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
                # 解析一条记录 (日期,开,高,低,收,成交额,成交量,保留)
                values = struct.unpack('IIIIIfII', chunk)
                date_int = values[0]
                # 日期转字符串 YYYYMMDD
                date_str = str(date_int)
                if len(date_str) == 8:
                    try:
                        date = pd.to_datetime(date_str, format='%Y%m%d')
                        open_p = values[1] / 100
                        high = values[2] / 100
                        low = values[3] / 100
                        close = values[4] / 100
                        amount = values[5]  # 成交额
                        volume = values[6]  # 成交量

                        if close > 0:  # 过滤无效数据
                            data.append({
                                'date': date,
                                'open': open_p,
                                'high': high,
                                'low': low,
                                'close': close,
                                'amount': amount,
                                'volume': volume
                            })
                    except:
                        pass

        if data:
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df
        return pd.DataFrame()

    def get_market_code(self, code: str) -> str:
        """获取市场代码"""
        if code.startswith('6'):
            return 'sh'
        elif code.startswith('0') or code.startswith('3'):
            return 'sz'
        elif code.startswith('8') or code.startswith('4'):
            return 'bj'
        return 'sh'

    def get_limit_threshold(self, code: str) -> float:
        """获取涨停阈值"""
        if code.startswith('3') or code.startswith('688'):
            return self.limit_gem  # 创业板/科创板
        return self.limit_main

    def is_limit_up(self, row: pd.Series, code: str) -> bool:
        """判断是否涨停"""
        threshold = self.get_limit_threshold(code)
        change = (row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0
        return change >= threshold

    def calc_consecutive_limit_ups(self, df: pd.DataFrame, code: str) -> int:
        """计算连续涨停天数"""
        if len(df) < 2:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        # 从最新数据往前检查
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if self.is_limit_up(row, code):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                # 判断是否接近涨停但没封住（比如涨幅>7%）
                change = (row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0
                if change < 0.07:  # 非涨停日，重置
                    current_consecutive = 0

        return max_consecutive

    def load_all_stock_data(self, max_stocks: int = None) -> Dict[str, pd.DataFrame]:
        """加载所有股票数据"""
        print("正在加载股票数据...")
        files = self.list_day_files()
        if max_stocks:
            files = files[:max_stocks]

        all_data = {}
        total = len(files)

        for i, (market, code, filepath) in enumerate(files):
            if i % 100 == 0:
                print(f"  进度: {i}/{total}")
            try:
                df = self.read_day_file(filepath)
                if len(df) > 0:
                    full_code = f"{market}{code}"
                    all_data[full_code] = df
            except Exception as e:
                pass

        print(f"已加载 {len(all_data)} 只股票")
        return all_data

    def get_index_data(self, index_code: str = 'sh000001') -> pd.DataFrame:
        """获取大盘指数数据"""
        # 上证指数文件
        if index_code.startswith('sh'):
            filepath = os.path.join(self.vipdoc_path, 'sh', 'lday', f'{index_code[2:]}.day')
        else:
            filepath = os.path.join(self.vipdoc_path, 'sz', 'lday', f'{index_code[2:]}.day')

        if os.path.exists(filepath):
            return self.read_day_file(filepath)
        return pd.DataFrame()

    def analyze_market(self, date: str = None):
        """分析市场状态"""
        print("正在分析市场状态...")

        # 获取最新交易日
        all_changes = []
        limit_ups = []

        for code, df in self.stock_data.items():
            if len(df) > 1:
                latest = df.iloc[-1]
                prev = df.iloc[-2]

                # 计算涨跌幅
                change = (latest['close'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0
                self.stock_change[code] = change
                all_changes.append(change)

                # 判断是否涨停
                if self.is_limit_up(latest, code):
                    self.limit_up_stocks.append(code)

                # 计算连板
                consecutive = self.calc_consecutive_limit_ups(df, code)
                if consecutive > 0:
                    limit_ups.append((code, consecutive))
                    self.max_consecutive = max(self.max_consecutive, consecutive)

        self.limit_up_count = len(self.limit_up_stocks)

        # 统计市场状态
        up_count = sum(1 for c in all_changes if c > 0)
        down_count = sum(1 for c in all_changes if c < 0)
        total_count = len(all_changes)

        print(f"涨跌统计: 上涨 {up_count}, 下跌 {down_count}, 平盘 {total_count - up_count - down_count}")
        print(f"涨停数量: {self.limit_up_count}")
        print(f"最高连板: {self.max_consecutive}")

        return {
            'up_count': up_count,
            'down_count': down_count,
            'total_count': total_count,
            'limit_up_count': self.limit_up_count,
            'max_consecutive': self.max_consecutive
        }

    def score_stock(self, code: str, market_status: dict) -> int:
        """对单个股票评分"""
        score = 0

        if code not in self.stock_data:
            return 0

        df = self.stock_data[code]
        if len(df) < 2:
            return 0

        latest = df.iloc[-1]
        change = self.stock_change.get(code, 0)

        # 条件1和2: 需要板块数据（待实现）
        # 暂时跳过板块评分

        # 条件3: 个股在大盘下跌时上涨
        # 需要获取大盘数据来判断
        index_df = self.get_index_data()
        if len(index_df) >= 2:
            index_change = (index_df.iloc[-1]['close'] - index_df.iloc[-2]['close']) / index_df.iloc[-2]['close']
            if index_change < 0 and change > 0:
                score += 1
                print(f"  +1: {code} 逆势上涨 (大盘{index_change*100:.2f}%, 个股{change*100:.2f}%)")

        # 条件4: 逆势环境
        up_ratio = market_status['up_count'] / market_status['total_count'] if market_status['total_count'] > 0 else 0
        if up_ratio < 0.3:  # 上涨股票少于30%
            score += 1

        # 条件5: 市场最高连板数 >= 4
        if self.max_consecutive >= 4:
            score += 1

        return score

    def screen(self, min_score: int = 5) -> List[Tuple[str, int]]:
        """执行选股"""
        print(f"\n开始选股，目标分数: {min_score}")

        # 加载股票数据
        self.stock_data = self.load_all_stock_data()

        # 分析市场
        market_status = self.analyze_market()

        # 评分
        results = []
        print("\n正在评分...")
        for code in self.stock_data.keys():
            score = self.score_stock(code, market_status)
            if score >= min_score:
                results.append((code, score))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results, market_status


def main():
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    screener = TdxScreener(tdx_path)

    # 限制加载500只股票进行测试
    screener.stock_data = screener.load_all_stock_data(max_stocks=500)

    # 分析市场
    market_status = screener.analyze_market()

    # 评分
    results = []
    print("\n正在评分...")
    for code in screener.stock_data.keys():
        score = screener.score_stock(code, market_status)
        if score >= 2:  # 先显示2分以上的
            results.append((code, score))

    # 排序
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== 结果 ===")
    print(f"市场状态: {market_status}")
    print(f"\n高分股票:")
    for code, score in results:
        print(f"  {code}: {score}分")


if __name__ == "__main__":
    main()
