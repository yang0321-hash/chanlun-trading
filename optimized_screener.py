"""
优化后的选股器 - 加权评分系统
"""

import os
import struct
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple


def is_valid_stock(code: str) -> bool:
    """只保留A股"""
    if code.startswith('sh6') and not code.startswith('sh688'): return True
    if code.startswith('sh688'): return True
    if code.startswith('sz0'): return True
    if code.startswith('sz3'): return True
    return False


def get_sector(code: str) -> str:
    """根据代码判断板块"""
    if code.startswith('sh688'): return '科创板'
    if code.startswith('sz300'): return '创业板'
    if code.startswith('sh60'): return '沪主板'
    if code.startswith('sz0'): return '深主板'
    return '其他'


def is_limit_up(open_p, close, code):
    """判断涨停"""
    if open_p <= 0: return False
    change = (close - open_p) / open_p
    threshold = 0.195 if code.startswith('sh688') or code.startswith('sz3') else 0.095
    return change >= threshold


def read_day_file(filepath):
    """读取日线文件"""
    data = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(32)
            if len(chunk) < 32: break
            v = struct.unpack('IIIIIfII', chunk)
            try:
                date = pd.to_datetime(str(v[0]), format='%Y%m%d')
                open_p, high, low, close = v[1]/100, v[2]/100, v[3]/100, v[4]/100
                if 0 < close < 200:
                    data.append({'date': date, 'open': open_p, 'high': high, 'low': low, 'close': close})
            except: pass
    if data:
        return pd.DataFrame(data).set_index('date')
    return pd.DataFrame()


def calc_max_consecutive(df, idx):
    """计算连板数"""
    cons = 0
    for i in range(idx, -1, -1):
        row = df.iloc[i]
        if row['open'] > 0 and (row['close'] - row['open']) / row['open'] >= 0.095:
            cons += 1
        else:
            break
    return cons


class WeightedScreener:
    """加权评分选股器"""

    def __init__(self, tdx_path: str):
        self.tdx_path = tdx_path
        self.vipdoc_path = os.path.join(tdx_path, 'vipdoc')
        self.stock_data = {}
        self.index_data = None

    def load_data(self, max_stocks=None):
        """加载数据"""
        print("正在加载数据...")
        files = []
        for market in ['sh', 'sz']:
            lday_path = os.path.join(self.vipdoc_path, market, 'lday')
            if not os.path.exists(lday_path): continue
            for f in os.listdir(lday_path):
                if f.endswith('.day'):
                    full_code = f.replace('.day', '')
                    if is_valid_stock(full_code):
                        files.append((full_code, os.path.join(lday_path, f)))

        print(f"找到 {len(files)} 只有效股票")

        for i, (code, path) in enumerate(files):
            if max_stocks and i >= max_stocks:
                break
            if i % 500 == 0:
                print(f"  加载进度: {i}/{min(len(files), max_stocks or len(files))}", flush=True)
            try:
                df = read_day_file(path)
                if len(df) > 50:
                    self.stock_data[code] = df
            except:
                pass

        # 加载大盘指数
        index_path = os.path.join(self.vipdoc_path, 'sh', 'lday', 'sh000001.day')
        if os.path.exists(index_path):
            self.index_data = read_day_file(index_path)

        print(f"已加载 {len(self.stock_data)} 只股票")

    def score_stock(self, code: str, date: pd.Timestamp,
                    market_stats: Dict = None, sector_stats: Dict = None) -> Dict:
        """对单只股票评分（使用预计算的统计数据）"""
        df = self.stock_data.get(code)
        if df is None or date not in df.index:
            return {'total_score': 0, 'details': {}}

        idx = df.index.get_loc(date)
        if idx < 1:
            return {'total_score': 0, 'details': {}}

        scores = {}
        details = {}

        # === 基础数据 ===
        today = df.iloc[idx]
        yest = df.iloc[idx-1]
        stock_change = (today['close'] - yest['close']) / yest['close']
        sector = get_sector(code)

        # === 使用预计算的数据 ===
        if market_stats is None:
            market_stats = self._calc_market_stats(date)
        if sector_stats is None:
            sector_stats = self._calc_sector_stats(date, market_stats['stock_change'])

        # ========== 【市场环境】最多4分 ==========

        # 1. 大盘下跌 + 个股上涨 (2分)
        if market_stats['index_change'] < 0 and stock_change > 0:
            scores['market_stock_up'] = 2
            details['大盘跌个股涨'] = 2
        else:
            scores['market_stock_up'] = 0

        # 2. 大盘下跌 + 板块上涨 (1分)
        sector_avg = sector_stats.get(sector, 0)
        if market_stats['index_change'] < 0 and sector_avg > 0:
            scores['market_sector_up'] = 1
            details['大盘跌板块涨'] = 1
        else:
            scores['market_sector_up'] = 0

        # 3. 全市涨跌比 < 30% (1分)
        up_ratio = market_stats['up_count'] / market_stats['total_count'] if market_stats['total_count'] > 0 else 0
        if up_ratio < 0.3:
            scores['market_down'] = 1
            details['弱势环境'] = 1
        else:
            scores['market_down'] = 0

        # 4. 市场最高连板 >= 4 (1分)
        if market_stats['max_consecutive'] >= 4:
            scores['max_consecutive'] = 1
            details['高连板'] = 1
        else:
            scores['max_consecutive'] = 0

        # ========== 【板块强度】最多4分 ==========

        # 板块涨幅排名
        sector_ranking = sector_stats.get('ranking', {})
        sector_rank = sector_ranking.get(sector, 99)

        if sector_rank <= 3:
            scores['sector_rank'] = 2
            details[f'板块第{sector_rank}'] = 2
        elif sector_rank <= 5:
            scores['sector_rank'] = 1
            details[f'板块第{sector_rank}'] = 1
        else:
            scores['sector_rank'] = 0

        # 板块逆势 (1分)
        if market_stats['index_change'] < 0 and sector_avg > 0:
            scores['sector_counter'] = 1
            details['板块逆势'] = 1
        else:
            scores['sector_counter'] = 0

        # 板块内有涨停股 (1分)
        sector_limit_count = sector_stats.get('limit_up_count', {}).get(sector, 0)
        if sector_limit_count >= 2:
            scores['sector_limit'] = 1
            details['板块多涨停'] = 1
        else:
            scores['sector_limit'] = 0

        # ========== 【个股地位】最多4分 ==========

        # 计算个股在板块内的排名
        sector_stocks = [(c, ch) for c, ch in market_stats['stock_change'].items()
                        if get_sector(c) == sector]
        sector_stocks.sort(key=lambda x: x[1], reverse=True)

        stock_rank = -1
        for i, (c, _) in enumerate(sector_stocks):
            if c == code:
                stock_rank = i + 1
                break

        if stock_rank == 1:
            scores['stock_rank'] = 3
            details['龙一'] = 3
        elif stock_rank == 2:
            scores['stock_rank'] = 2
            details['龙二'] = 2
        elif stock_rank == 3:
            scores['stock_rank'] = 1
            details['龙三'] = 1
        else:
            scores['stock_rank'] = 0

        # 个股涨幅 > 5% (1分)
        if stock_change > 0.05:
            scores['stock_change'] = 1
            details['涨幅>5%'] = 1
        else:
            scores['stock_change'] = 0

        # ========== 【量能形态】最多3分 ==========

        # 量比 (需要成交量数据)
        # if idx >= 5:
        #     vol_ratio = today['volume'] / df.iloc[idx-5:idx]['volume'].mean()
        #     if vol_ratio > 2:
        #         scores['volume'] = 1
        #         details['量比>2'] = 1

        # 突破形态 (1分)
        high_20 = df.iloc[idx-20:idx]['high'].max() if idx >= 20 else df.iloc[:idx]['high'].max()
        if today['close'] > high_20:
            scores['breakout'] = 1
            details['突破'] = 1
        else:
            scores['breakout'] = 0

        total = sum(scores.values())

        return {
            'total_score': total,
            'details': details,
            'stock_change': stock_change,
            'sector': sector,
            'sector_rank': sector_rank,
            'stock_rank_in_sector': stock_rank,
        }

    def _calc_market_stats(self, date: pd.Timestamp) -> Dict:
        """计算市场统计"""
        up_count = 0
        total_count = 0
        max_cons = 0
        stock_change = {}
        index_change = 0

        # 计算大盘涨跌
        if self.index_data is not None and date in self.index_data.index:
            idx_idx = self.index_data.index.get_loc(date)
            if idx_idx >= 1:
                index_change = (self.index_data.iloc[idx_idx]['close'] -
                              self.index_data.iloc[idx_idx-1]['close']) / self.index_data.iloc[idx_idx-1]['close']

        # 计算个股数据
        for code, df in self.stock_data.items():
            if date not in df.index: continue
            idx = df.index.get_loc(date)
            if idx < 1: continue

            today = df.iloc[idx]
            yest = df.iloc[idx-1]

            if yest['close'] > 0:
                change = (today['close'] - yest['close']) / yest['close']
                stock_change[code] = change
                total_count += 1
                if change > 0:
                    up_count += 1

                # 连板
                cons = calc_max_consecutive(df, idx)
                max_cons = max(max_cons, cons)

        return {
            'up_count': up_count,
            'total_count': total_count,
            'max_consecutive': max_cons,
            'stock_change': stock_change,
            'index_change': index_change
        }

    def _calc_sector_stats(self, date: pd.Timestamp, stock_change: Dict) -> Dict:
        """计算板块统计"""
        sector_avg = {}
        sector_limit_count = {}

        # 按板块分组计算平均涨幅
        for sector in ['沪主板', '深主板', '创业板', '科创板']:
            changes = [ch for code, ch in stock_change.items() if get_sector(code) == sector]
            if changes:
                sector_avg[sector] = sum(changes) / len(changes)

        # 板块排名
        sorted_sectors = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)
        ranking = {s[0]: i+1 for i, s in enumerate(sorted_sectors)}

        # 统计各板块涨停数
        for code, change in stock_change.items():
            if change >= 0.095:
                sector = get_sector(code)
                sector_limit_count[sector] = sector_limit_count.get(sector, 0) + 1

        return {
            'sector_avg': sector_avg,
            'ranking': ranking,
            'limit_up_count': sector_limit_count
        }

    def backtest(self, start_idx: int = -100, min_score: int = 8,
                 buy_at: str = 'open', hold_days: int = 1):
        """回测"""
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))[start_idx:]

        buy_method = {'close': '信号日', 'open': '次日开盘', 'high': '次日涨停'}.get(buy_at, buy_at)
        print(f"\n回测：{len(all_dates)}天，阈值≥{min_score}分，{buy_method}买入，持有{hold_days}天")

        trades = []

        for i, date in enumerate(all_dates[:-3]):
            if i % 10 == 0:
                print(f"  进度: {i}/{len(all_dates)}", flush=True)

            # 先计算市场统计（避免重复）
            market_stats = self._calc_market_stats(date)
            sector_stats = self._calc_sector_stats(date, market_stats['stock_change'])

            # 选股
            scores = []
            for code in self.stock_data.keys():
                result = self.score_stock(code, date, market_stats, sector_stats)
                if result['total_score'] >= min_score:
                    scores.append((code, result))

            if not scores:
                continue

            # 按分数排序，取前5
            scores.sort(key=lambda x: x[1]['total_score'], reverse=True)
            top_stocks = scores[:5]

            for code, result in top_stocks:
                df = self.stock_data[code]

                if buy_at == 'close':
                    buy_date = date
                    sell_date = all_dates[all_dates.index(date) + hold_days]
                else:
                    buy_date = all_dates[all_dates.index(date) + 1]
                    sell_date = all_dates[all_dates.index(date) + hold_days + 1]

                if buy_date not in df.index or sell_date not in df.index:
                    continue

                buy_price = df.loc[buy_date, 'open'] if buy_at == 'open' else df.loc[buy_date, 'close']
                sell_price = df.loc[sell_date, 'close']

                profit = (sell_price - buy_price) / buy_price * 100

                trades.append({
                    'signal_date': date,
                    'code': code,
                    'score': result['total_score'],
                    'sector': result['sector'],
                    'details': result['details'],
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': sell_date,
                    'sell_price': sell_price,
                    'profit_pct': profit
                })

        return trades

    def analyze(self, trades: List):
        """分析结果"""
        if not trades:
            print("无交易")
            return

        df = pd.DataFrame(trades)

        print("\n" + "="*60)
        print("回测结果")
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

        # 按分数统计
        print(f"\n按分数统计:")
        for score in sorted(df['score'].unique()):
            sub = df[df['score'] == score]
            win = len(sub[sub['profit_pct'] > 0])
            print(f"  {score}分: {len(sub)}笔, 胜率{win/len(sub)*100:.1f}%, 平均{sub['profit_pct'].mean():.2f}%")

        # 按板块统计
        print(f"\n按板块统计:")
        for sector in ['沪主板', '深主板', '创业板', '科创板']:
            sub = df[df['sector'] == sector]
            if len(sub) > 0:
                win = len(sub[sub['profit_pct'] > 0])
                print(f"  {sector}: {len(sub)}笔, 胜率{win/len(sub)*100:.1f}%, 平均{sub['profit_pct'].mean():.2f}%")

        # 显示最近交易
        print(f"\n最近10笔:")
        display = df[['code', 'score', 'sector', 'buy_price', 'sell_price', 'profit_pct']].tail(10)
        print(display.to_string())


def main():
    tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"

    screener = WeightedScreener(tdx_path)
    screener.load_data(max_stocks=2000)  # 限制加载2000只股票

    # 测试不同阈值 - 只用30天数据
    for min_score in [6, 8]:
        print(f"\n{'='*60}")
        print(f"评分阈值: {min_score}分")
        print(f"{'='*60}")

        trades = screener.backtest(start_idx=-30, min_score=min_score, buy_at='open', hold_days=1)
        screener.analyze(trades)


if __name__ == "__main__":
    main()
