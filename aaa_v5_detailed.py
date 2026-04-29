#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AAA 策略 V5 详细版 - 显示买卖点
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import akshare as ak


class AAAIndicatorsV2:
    """AAA 技术指标 V2"""

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

        # ATR
        high_low = df['high'] - df['low']
        df['atr'] = high_low.rolling(14).mean()

        return df


class TrendFilter:
    """趋势过滤器"""

    @staticmethod
    def is_uptrend(df: pd.DataFrame) -> bool:
        """判断是否处于上升趋势"""
        if len(df) < 60:
            return False

        latest = df.iloc[-1]

        # 1. 价格在 MA60 之上
        if latest['close'] < latest['ma60']:
            return False

        # 2. MA5 > MA20
        if latest['ma5'] < latest['ma20']:
            return False

        # 3. MACD 多头 (DIF > DEA)
        if latest['dif'] <= latest['dea']:
            return False

        # 4. 近期价格高点抬升（20日新高）
        recent_20_high = df['high'].iloc[-20:].max()
        if latest['high'] >= recent_20_high * 0.98:
            return True

        return False


class AAAScorerV2:
    """AAA 评分系统 V2"""

    @staticmethod
    def calculate_dynamic_thresholds(df: pd.DataFrame) -> tuple:
        """根据市场波动率动态调整阈值"""
        if len(df) < 60:
            return 65, 35

        latest = df.iloc[-1]
        atr = latest.get('atr', df['close'] * 0.02)
        atr_pct = atr / latest['close']

        # 计算价格离散度
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.02

        # 动态调整
        if volatility > 0.03:  # 高波动
            return 60, 30
        elif volatility < 0.015:  # 低波动
            return 70, 40
        else:
            return 65, 35

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


class TrailingStop:
    """跟踪止损"""

    def __init__(self, atr_multiplier=2.5, activation_pct=0.03):
        self.atr_multiplier = atr_multiplier
        self.activation_pct = activation_pct

    def calculate_stop(self, entry_price: float, current_price: float,
                       highest_price: float, atr: float) -> tuple:
        """计算跟踪止损"""
        stop_loss = highest_price - atr * self.atr_multiplier
        initial_stop = entry_price * 0.95
        stop_loss = max(stop_loss, initial_stop)

        if current_price > highest_price:
            highest_price = current_price

        return stop_loss, highest_price


def main():
    symbol = 'sz002600'
    symbol_name = '领益智造'
    start_date = datetime(2023, 1, 1)
    initial_capital = 100000

    print("=" * 80)
    print(f"AAA V5 策略详细回测 - {symbol} ({symbol_name})")
    print("=" * 80)
    print(f"回测期间: {start_date.date()} ~")
    print(f"初始资金: {initial_capital:,.0f} 元")
    print(f"配置: 趋势过滤 + 跟踪止损 + 动态阈值 + 周线确认")
    print("=" * 80)
    print()

    # 获取数据
    df = ak.stock_zh_a_hist(
        symbol=symbol.replace('sz', '').replace('sh', ''),
        period='daily',
        start_date=start_date.strftime('%Y%m%d'),
        adjust='qfq'
    )

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

    print(f"日线数据: {len(df)} 条")

    # 获取周线数据
    df_copy = df.copy()
    df_copy.set_index('datetime', inplace=True)
    weekly_df = pd.DataFrame({
        'open': df_copy['open'].resample('W').first(),
        'high': df_copy['high'].resample('W').max(),
        'low': df_copy['low'].resample('W').min(),
        'close': df_copy['close'].resample('W').last(),
        'volume': df_copy['volume'].resample('W').sum()
    }).dropna().reset_index()
    print(f"周线数据: {len(weekly_df)} 条")
    print()

    # 计算指标
    df = AAAIndicatorsV2.calculate_all(df)

    # 回测状态
    cash = initial_capital
    position = 0
    entry_price = 0
    entry_score = 0
    highest_price = 0
    signals = []

    for i in range(60, len(df)):
        current_df = df.iloc[:i+1]
        current_bar = df.iloc[i]

        # 获取动态阈值
        buy_threshold, sell_threshold = AAAScorerV2.calculate_dynamic_thresholds(current_df)

        # 计算评分
        score_result = AAAScorerV2.score(current_df)

        # 趋势过滤
        if not TrendFilter.is_uptrend(current_df):
            if position > 0:
                # 有持仓但趋势转弱，检查卖出
                pass
            else:
                continue  # 不是上升趋势，跳过买入信号

        # 周线确认
        current_date = current_bar['datetime']
        weekly_slice = weekly_df[weekly_df['datetime'] <= current_date]
        weekly_score_info = None
        if len(weekly_slice) >= 60:
            weekly_with_indicators = AAAIndicatorsV2.calculate_all(weekly_slice.copy())
            weekly_score_info = AAAScorerV2.score(weekly_with_indicators)
            if weekly_score_info['total'] < 55 and position == 0:
                continue  # 周线不支持，跳过

        if position == 0:
            # 无持仓，检查买入
            if score_result['total'] >= buy_threshold:
                # 检查评分变化
                if i > 60:
                    prev_score = AAAScorerV2.score(df.iloc[:i])
                    score_change = score_result['total'] - prev_score['total']
                    if score_change < 5:
                        continue

                # 买入
                price = current_bar['close']
                max_amount = cash * 0.95
                quantity = int(max_amount / price / 100) * 100

                if quantity > 0:
                    position = quantity
                    entry_price = price
                    highest_price = price
                    entry_score = score_result['total']
                    cash -= quantity * price * (1 + 0.0003)

                    signals.append({
                        'date': current_bar['datetime'],
                        'action': 'BUY',
                        'price': price,
                        'quantity': quantity,
                        'score': score_result['total'],
                        'threshold': buy_threshold,
                        'weekly_score': weekly_score_info['total'] if weekly_score_info else 0,
                        'ma5': current_bar['ma5'],
                        'ma20': current_bar['ma20'],
                        'ma60': current_bar['ma60'],
                        'dif': current_bar['dif'],
                        'dea': current_bar['dea'],
                        'rsi': current_bar['rsi'],
                        'cash': cash
                    })

        else:
            # 有持仓，检查卖出
            should_sell = False
            reason = ""

            # 评分过低
            if score_result['total'] <= sell_threshold:
                should_sell = True
                reason = f"评分过低({score_result['total']}<=阈值{sell_threshold})"

            # 跟踪止损
            atr = current_df.iloc[-1]['atr']
            trailing_stop = TrailingStop()
            stop_price, _ = trailing_stop.calculate_stop(entry_price, current_bar['close'], highest_price, atr)

            if current_bar['close'] <= stop_price:
                should_sell = True
                reason = f"跟踪止损(价格{current_bar['close']:.2f}<=止损{stop_price:.2f})"

            # 更新最高价
            if current_bar['close'] > highest_price:
                highest_price = current_bar['close']

            # 止盈
            if current_bar['close'] >= entry_price * 1.15:
                should_sell = True
                reason = f"止盈(涨幅>=15%)"

            # 评分大幅下降
            if entry_score - score_result['total'] >= 25:
                should_sell = True
                reason = f"评分下降({entry_score:.0f}->{score_result['total']:.0f})"

            if should_sell:
                price = current_bar['close']
                profit = (price - entry_price) * position
                profit_pct = profit / (entry_price * position) * 100

                signals.append({
                    'date': current_bar['datetime'],
                    'action': 'SELL',
                    'price': price,
                    'quantity': position,
                    'score': score_result['total'],
                    'reason': reason,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'cash': cash + position * price * (1 - 0.0003)
                })

                cash += position * price * (1 - 0.0003)
                position = 0
                entry_price = 0
                entry_score = 0
                highest_price = 0

    # 打印详细交易记录
    print("=" * 80)
    print("详细交易记录")
    print("=" * 80)
    print()

    for i, signal in enumerate(signals, 1):
        if signal['action'] == 'BUY':
            print(f"[交易 #{(i+1)//2}] 买入")
            print(f"  日期: {signal['date'].date()}")
            print(f"  价格: {signal['price']:.2f} 元")
            print(f"  数量: {signal['quantity']} 股")
            print(f"  金额: {signal['price']*signal['quantity']:,.0f} 元")
            print(f"  日线评分: {signal['score']:.0f} (阈值: {signal['threshold']:.0f})")
            print(f"  周线评分: {signal['weekly_score']:.0f}")
            print(f"  MA5/MA20/MA60: {signal['ma5']:.2f} / {signal['ma20']:.2f} / {signal['ma60']:.2f}")
            print(f"  DIF/DEA: {signal['dif']:.3f} / {signal['dea']:.3f}")
            print(f"  RSI: {signal['rsi']:.1f}")
            print(f"  剩余现金: {signal['cash']:,.0f} 元")
            print()

        else:  # SELL
            print(f"[交易 #{i//2}] 卖出")
            print(f"  日期: {signal['date'].date()}")
            print(f"  价格: {signal['price']:.2f} 元")
            print(f"  数量: {signal['quantity']} 股")
            print(f"  盈亏: {signal['profit']:>+,.0f} 元 ({signal['profit_pct']:+.2f}%)")
            print(f"  原因: {signal['reason']}")
            print(f"  账户总值: {signal['cash']:,.0f} 元")
            print("-" * 80)
            print()

    # 汇总统计
    final_equity = cash if position == 0 else cash + position * df.iloc[-1]['close']
    total_return = (final_equity - initial_capital) / initial_capital * 100

    buy_signals = [s for s in signals if s['action'] == 'BUY']
    sell_signals = [s for s in signals if s['action'] == 'SELL']
    profit_trades = [s for s in sell_signals if s['profit'] > 0]

    print("=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"最终资金: {final_equity:,.0f} 元")
    print(f"总收益: {total_return:+.2f}%")
    print(f"买入次数: {len(buy_signals)}")
    print(f"卖出次数: {len(sell_signals)}")
    print(f"盈利次数: {len(profit_trades)}")
    print(f"胜率: {len(profit_trades)/len(sell_signals)*100:.1f}%" if sell_signals else "胜率: N/A")

    total_profit = sum(s['profit'] for s in sell_signals) if sell_signals else 0
    print(f"累计盈亏: {total_profit:>+,.0f} 元")

    if profit_trades:
        avg_profit = sum(s['profit'] for s in profit_trades) / len(profit_trades)
        print(f"平均盈利: {avg_profit:>+,.0f} 元")

    loss_trades = [s for s in sell_signals if s['profit'] < 0]
    if loss_trades:
        avg_loss = sum(s['profit'] for s in loss_trades) / len(loss_trades)
        print(f"平均亏损: {avg_loss:>+,.0f} 元")

    return signals


if __name__ == '__main__':
    main()
