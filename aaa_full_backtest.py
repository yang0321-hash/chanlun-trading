#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AAA 策略完整回测
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import akshare as ak


class AAAIndicators:
    """AAA 技术指标"""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标"""
        # MA
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # MACD
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['dif'] = df['ema12'] - df['ema26']
        df['dea'] = df['dif'].ewm(span=9).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # KDJ
        low = df['low'].rolling(9).min()
        high = df['high'].rolling(9).max()
        rsv = (df['close'] - low) / (high - low) * 100
        rsv = rsv.fillna(50)
        df['k'] = rsv.ewm(com=2).mean()
        df['d'] = df['k'].ewm(com=2).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

        # BOLL
        df['boll_mid'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * std
        df['boll_lower'] = df['boll_mid'] - 2 * std
        df['boll_pos'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower']) * 100

        return df


class AAAScorer:
    """AAA 评分系统"""

    @staticmethod
    def score(df: pd.DataFrame) -> dict:
        """计算评分"""
        latest = df.iloc[-1]

        scores = {
            'trend': 20 if latest['ma5'] > latest['ma10'] > latest['ma20'] else 10 if latest['ma5'] > latest['ma10'] else 0,
            'momentum': 20 if latest['dif'] > latest['dea'] and latest['macd'] > 0 else 10 if latest['dif'] > latest['dea'] else 0,
            'rsi': 15 if latest['rsi'] < 30 else 10 if 30 <= latest['rsi'] <= 70 else 0,
            'volume': 15 if latest['volume'] > df['volume'].iloc[-10:].mean() * 1.5 else 5,
            'boll': 15 if latest['boll_pos'] < 20 else 10 if 40 <= latest['boll_pos'] <= 60 else 0,
            'kdj': 15 if latest['k'] > latest['d'] and latest['k'] < 20 else 10 if latest['k'] > latest['d'] else 5
        }

        total = sum(scores.values())

        if total >= 70:
            level = "买入"
        elif total >= 55:
            level = "偏多"
        elif total >= 45:
            level = "观望"
        elif total >= 30:
            level = "偏空"
        else:
            level = "规避"

        return {'scores': scores, 'total': total, 'level': level}


class AAABacktest:
    """AAA 回测引擎"""

    def __init__(self, initial_capital=100000, commission=0.0003):
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []

    def get_stock_data(self, symbol: str, start_date: datetime) -> pd.DataFrame:
        """获取股票数据"""
        start_str = start_date.strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(
            symbol=symbol.replace('sz', '').replace('sh', ''),
            period='daily',
            start_date=start_str,
            adjust='qfq'
        )

        # 标准化列名
        column_map = {
            '日期': 'datetime',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
        }
        df = df.rename(columns=column_map)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    def run(self, symbol: str, start_date: datetime, buy_threshold=70, sell_threshold=30):
        """运行回测"""
        # 获取数据
        df = self.get_stock_data(symbol, start_date)
        print(f"数据范围: {df['datetime'].iloc[0].date()} ~ {df['datetime'].iloc[-1].date()}, 共 {len(df)} 条")

        # 计算指标
        df = AAAIndicators.calculate_all(df)

        # 回测
        signals = []
        entry_price = 0
        entry_score = 0

        for i in range(60, len(df)):
            current_df = df.iloc[:i+1]
            score_result = AAAScorer.score(current_df)
            current_bar = df.iloc[i]

            # 生成信号
            if self.position == 0:
                # 无持仓，检查买入
                if score_result['total'] >= buy_threshold:
                    # 检查评分变化
                    if i > 60:
                        prev_score = AAAScorer.score(df.iloc[:i])
                        score_change = score_result['total'] - prev_score['total']
                        if score_change < 10:
                            continue

                    # 买入
                    price = current_bar['close']
                    max_amount = self.cash * 0.95  # 保留5%现金
                    quantity = int(max_amount / price / 100) * 100

                    if quantity > 0:
                        self.position = quantity
                        entry_price = price
                        entry_score = score_result['total']
                        self.cash -= quantity * price * (1 + self.commission)

                        signals.append({
                            'date': current_bar['datetime'],
                            'action': '买入',
                            'price': price,
                            'quantity': quantity,
                            'score': score_result['total'],
                            'reason': score_result['level']
                        })

            else:
                # 有持仓，检查卖出
                should_sell = False
                reason = ""

                # 评分过低
                if score_result['total'] <= sell_threshold:
                    should_sell = True
                    reason = f"评分过低 ({score_result['total']} <= {sell_threshold})"

                # 止损
                elif current_bar['close'] <= entry_price * 0.95:
                    should_sell = True
                    reason = f"止损 ({current_bar['close']:.2f} <= {entry_price * 0.95:.2f})"

                # 止盈
                elif current_bar['close'] >= entry_price * 1.10:
                    should_sell = True
                    reason = f"止盈 ({current_bar['close']:.2f} >= {entry_price * 1.10:.2f})"

                # 评分下降
                elif entry_score - score_result['total'] >= 20:
                    should_sell = True
                    reason = f"评分下降 ({entry_score:.0f} -> {score_result['total']:.0f})"

                if should_sell:
                    price = current_bar['close']
                    self.cash += self.position * price * (1 - self.commission)
                    profit = (price - entry_price) * self.position

                    signals.append({
                        'date': current_bar['datetime'],
                        'action': '卖出',
                        'price': price,
                        'quantity': self.position,
                        'score': score_result['total'],
                        'reason': reason,
                        'profit': profit
                    })

                    self.position = 0
                    entry_price = 0
                    entry_score = 0

            # 记录权益
            equity = self.cash + self.position * df.iloc[i]['close']
            self.equity_curve.append({'date': df.iloc[i]['datetime'], 'equity': equity})

        return self.analyze_results(df, signals)

    def analyze_results(self, df, signals):
        """分析结果"""
        # 计算收益
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # 买入持有收益
        buy_hold_return = (df.iloc[-1]['close'] - df.iloc[60]['close']) / df.iloc[60]['close']

        # 交易统计
        buy_signals = [s for s in signals if s['action'] == '买入']
        sell_signals = [s for s in signals if s['action'] == '卖出']

        # 盈利交易
        profit_trades = [s for s in sell_signals if s.get('profit', 0) > 0]
        win_rate = len(profit_trades) / len(sell_signals) if sell_signals else 0

        # 最大回撤
        equity_series = [e['equity'] for e in self.equity_curve]
        max_drawdown = 0
        peak = equity_series[0]
        for val in equity_series:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        # 夏普比率（简化版）
        returns = pd.Series([e['equity'] for e in self.equity_curve]).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'win_rate': win_rate,
            'signals': signals,
            'equity_curve': self.equity_curve
        }


def main():
    symbol = 'sz002600'
    start_date = datetime(2023, 1, 1)

    print("=" * 60)
    print(f"AAA 策略回测 - {symbol}")
    print("=" * 60)
    print(f"回测期间: {start_date.date()} ~")
    print(f"初始资金: {100000:,.0f} 元")
    print(f"买入阈值: 70")
    print(f"卖出阈值: 30")
    print("=" * 60)
    print()

    # 运行回测
    backtest = AAABacktest(initial_capital=100000)
    results = backtest.run(symbol, start_date)

    # 打印结果
    print()
    print("=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"初始资金: {results['initial_capital']:,.0f} 元")
    print(f"最终资金: {results['final_equity']:,.0f} 元")
    print(f"总收益: {results['total_return']*100:.2f}%")
    print(f"买入持有收益: {results['buy_hold_return']*100:.2f}%")
    print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"买入次数: {results['buy_signals']}")
    print(f"卖出次数: {results['sell_signals']}")
    print(f"胜率: {results['win_rate']*100:.1f}%")
    print()

    # 显示交易记录
    if results['signals']:
        print("=" * 60)
        print("交易记录")
        print("=" * 60)
        for i, signal in enumerate(results['signals'], 1):
            action = signal['action']
            date = signal['date'].date()
            price = signal['price']
            qty = signal['quantity']
            score = signal['score']
            reason = signal['reason']

            if action == '买入':
                print(f"{i}. {date} {action} {qty}股 @ {price:.2f}元 (评分:{score:.0f} {reason})")
            else:
                profit = signal.get('profit', 0)
                profit_str = f"盈利{profit:,.0f}" if profit > 0 else f"亏损{profit:,.0f}"
                print(f"{i}. {date} {action} {qty}股 @ {price:.2f}元 (评分:{score:.0f} {reason}) [{profit_str}]")

    return results


if __name__ == '__main__':
    main()
