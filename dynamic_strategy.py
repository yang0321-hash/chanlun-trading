"""
动态选股策略回测
逻辑：
1. 第二天涨停价买入
2. 第三天不涨停则卖出，涨停则继续持有并卖出30%止盈
"""

import os
import struct
import pandas as pd
from collections import defaultdict

tdx_path = r'D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)'
vipdoc_path = os.path.join(tdx_path, 'vipdoc')

def is_valid_stock(code):
    """只保留A股"""
    if code.startswith('sh6') and not code.startswith('sh688'): return True  # 沪主板
    if code.startswith('sh688'): return True  # 科创板
    if code.startswith('sz0'): return True  # 深主板
    if code.startswith('sz3'): return True  # 创业板
    return False

def is_limit_up(open_price, close_price, code):
    """判断是否涨停"""
    if open_price <= 0:
        return False
    change = (close_price - open_price) / open_price
    # 科创板和创业板20%，其他10%
    threshold = 0.195 if (code.startswith('sh688') or code.startswith('sz3')) else 0.095
    return change >= threshold

def read_day_file(filepath):
    """读取日线文件"""
    data = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(32)
            if len(chunk) < 32:
                break
            v = struct.unpack('IIIIIfII', chunk)
            try:
                date = pd.to_datetime(str(v[0]), format='%Y%m%d')
                open_p = v[1] / 100
                high = v[2] / 100
                low = v[3] / 100
                close = v[4] / 100
                if 0 < close < 200:
                    data.append({'date': date, 'open': open_p, 'high': high, 'low': low, 'close': close})
            except:
                pass
    if data:
        return pd.DataFrame(data).set_index('date')
    return pd.DataFrame()

print("正在加载数据...")
stock_data = {}
for market in ['sh', 'sz']:
    lday_path = os.path.join(vipdoc_path, market, 'lday')
    files = [f for f in os.listdir(lday_path) if f.endswith('.day')]
    for f in files:
        code = f.replace('.day', '')
        if is_valid_stock(code):
            try:
                df = read_day_file(os.path.join(lday_path, f))
                if len(df) > 50:
                    stock_data[code] = df
            except:
                pass

print(f"已加载 {len(stock_data)} 只股票")

# 获取所有交易日期
all_dates = sorted(list(set(d for df in stock_data.values() for d in df.index)))
start_idx = -100  # 最近100天
dates = all_dates[start_idx:]

# 回测参数
initial_cash = 100000  # 初始资金
cash = initial_cash
positions = {}  # {code: shares}
trades = []
trade_log = []

print(f"\n开始回测，共 {len(dates)} 个交易日")

for i, date in enumerate(dates[:-5]):
    # === 上午盘前：筛选信号 ===
    signal_changes = {}
    for code, df in stock_data.items():
        if date in df.index:
            idx = df.index.get_loc(date)
            if idx > 0:
                change = (df.iloc[idx]['close'] - df.iloc[idx-1]['close']) / df.iloc[idx-1]['close']
                if change > 0.05:  # 涨幅>5%作为信号
                    signal_changes[code] = change

    # === 根据信号买入（第二天涨停价买入）===
    if i + 1 < len(dates) and signal_changes and cash > 0:
        buy_date = dates[i + 1]

        # 简单分配：平均分配给前5个信号
        top_signals = sorted(signal_changes.items(), key=lambda x: x[1], reverse=True)[:5]

        if top_signals:
            per_stock_cash = cash / len(top_signals)

            for code, change in top_signals:
                df = stock_data[code]
                if buy_date in df.index:
                    buy_day_data = df.loc[buy_date]
                    buy_price = buy_day_data['high']  # 涨停价买入

                    if buy_price > 0:
                        shares = int(per_stock_cash / buy_price)
                        if shares > 0:
                            cash -= shares * buy_price
                            positions[code] = {
                                'shares': shares,
                                'buy_price': buy_price,
                                'buy_date': buy_date,
                                'entry_date': date
                            }
                            trade_log.append({
                                'date': buy_date,
                                'action': '买入',
                                'code': code,
                                'shares': shares,
                                'price': buy_price,
                                'cash': cash
                            })

    # === 处理持仓：第三天及之后 ===
    # 检查每个持仓股票
    codes_to_sell = []
    codes_to_hold = []

    for code, pos in positions.items():
        df = stock_data[code]
        sell_date = dates[i + 2] if i + 2 < len(dates) else None

        if not sell_date or sell_date not in df.index:
            continue

        day_data = df.loc[sell_date]
        is_limit = is_limit_up(day_data['open'], day_data['close'], code)

        if is_limit:
            # 涨停：继续持有，卖出30%止盈
            sell_shares = int(pos['shares'] * 0.3)
            if sell_shares > 0:
                sell_price = day_data['close']  # 涨停通常以收盘价卖出
                cash += sell_shares * sell_price
                profit = (sell_price - pos['buy_price']) / pos['buy_price'] * 100

                pos['shares'] -= sell_shares

                trade_log.append({
                    'date': sell_date,
                    'action': '部分止盈',
                    'code': code,
                    'shares': sell_shares,
                    'price': sell_price,
                    'profit_pct': profit,
                    'cash': cash
                })

                if pos['shares'] == 0:
                    codes_to_sell.append(code)
                else:
                    # 更新持仓信息
                    positions[code]['shares'] = pos['shares']

            # 记录涨停继续持有的天数
            if code not in codes_to_sell:
                trades.append({
                    'code': code,
                    'entry_date': pos['entry_date'],
                    'buy_price': pos['buy_price'],
                    'current_date': sell_date,
                    'is_limit': True,
                    'hold_days': (sell_date - pos['buy_date']).days
                })
        else:
            # 不涨停：全部卖出
            sell_shares = pos['shares']
            sell_price = day_data['close']
            cash += sell_shares * sell_price
            profit = (sell_price - pos['buy_price']) / pos['buy_price'] * 100

            trade_log.append({
                'date': sell_date,
                'action': '卖出',
                'code': code,
                'shares': sell_shares,
                'price': sell_price,
                'profit_pct': profit,
                'cash': cash,
                'hold_days': (sell_date - pos['buy_date']).days
            })

            codes_to_sell.append(code)

            trades.append({
                'code': code,
                'entry_date': pos['entry_date'],
                'buy_price': pos['buy_price'],
                'sell_price': sell_price,
                'profit_pct': profit,
                'hold_days': (sell_date - pos['buy_date']).days,
                'exit_reason': '不涨停'
            })

    # 清理已卖出持仓
    for code in codes_to_sell:
        del positions[code]

# === 统计结果 ===
print("\n" + "="*60)
print("回测结果")
print("="*60)

if trades:
    df = pd.DataFrame(trades)

    total = len(df)
    profit = len(df[df['profit_pct'] > 0])
    loss = len(df[df['profit_pct'] < 0])

    print(f"\n总交易: {total}")
    print(f"盈利: {profit} ({profit/total*100:.1f}%)")
    print(f"亏损: {loss} ({loss/total*100:.1f}%)")
    print(f"平均收益: {df['profit_pct'].mean():.2f}%")
    print(f"最大盈利: {df['profit_pct'].max():.2f}%")
    print(f"最大亏损: {df['profit_pct'].min():.2f}%")
    print(f"平均持仓天数: {df['hold_days'].mean():.1f}天")

    # 按退出原因统计
    if 'exit_reason' in df.columns:
        print(f"\n按退出原因:")
        for reason, group in df.groupby('exit_reason'):
            win_rate = (group['profit_pct'] > 0).sum() / len(group) * 100
            avg_profit = group['profit_pct'].mean()
            print(f"  {reason}: {len(group)}笔, 胜率{win_rate:.1f}%, 平均{avg_profit:.2f}%")

    print(f"\n最后20笔交易:")
    print(df[['code', 'buy_price', 'sell_price', 'profit_pct', 'hold_days']].tail(20).to_string())

print(f"\n最终现金: {cash:.2f}")
print(f"总收益率: {(cash - initial_cash) / initial_cash * 100:.2f}%")
