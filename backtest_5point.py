"""
筛选5分普通股票回测
"""

import os
import struct
import pandas as pd
from collections import defaultdict

tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"
vipdoc_path = os.path.join(tdx_path, "vipdoc")

# 读取股票数据
stock_data = {}
for market in ['sh', 'sz']:
    lday_path = os.path.join(vipdoc_path, market, 'lday')
    if os.path.exists(lday_path):
        files = [f for f in os.listdir(lday_path) if f.endswith('.day')][:1000]
        for f in files:
            code = f.replace('.day', '').replace('sh', '').replace('sz', '')
            filepath = os.path.join(lday_path, f)
            data = []
            with open(filepath, 'rb') as file:
                while True:
                    chunk = file.read(32)
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
                        if close > 0 and close < 100:  # 过滤掉基金/ETF（价格通常>100）
                            data.append({'date': date, 'open': open_p, 'high': high, 'low': low, 'close': close})
                    except:
                        pass
            if data:
                df = pd.DataFrame(data).set_index('date')
                if len(df) > 10:
                    full_code = f"{market}{code}"
                    stock_data[full_code] = df

print(f"加载了 {len(stock_data)} 只普通股票")

# 读取大盘指数
index_path = vipdoc_path + "/sh/lday/sh000001.day"
index_data = []
with open(index_path, 'rb') as f:
    while True:
        chunk = f.read(32)
        if len(chunk) < 32:
            break
        values = struct.unpack('IIIIIfII', chunk)
        date_int = values[0]
        try:
            date = pd.to_datetime(str(date_int), format='%Y%m%d')
            close = values[4] / 100
            open_p = values[1] / 100
            if close > 0:
                index_data.append({'date': date, 'open': open_p, 'high': values[2]/100, 'low': values[3]/100, 'close': close})
        except:
            pass
index_df = pd.DataFrame(index_data).set_index('date')

# 获取所有交易日期
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index)
all_dates = sorted(list(all_dates))[-200:]  # 最后200天

# 回测
trades = []
five_point_count = 0

for i, date in enumerate(all_dates[:-2]):
    if i % 20 == 0:
        print(f"  进度: {i}/{len(all_dates)}")

    # 市场统计
    up_count = 0
    total_count = 0
    max_cons = 0
    stock_change = {}

    for code, df in stock_data.items():
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
                if (row['close'] - row['open']) / row['open'] >= 0.095 and row['open'] > 0:
                    cons += 1
                else:
                    break
            max_cons = max(max_cons, cons)

    # 筛选5分股票
    five_point_stocks = []
    for code, change in stock_change.items():
        score = 0

        # 条件3: 逆势上涨
        if date in index_df.index:
            idx_idx = index_df.index.get_loc(date)
            if idx_idx >= 1:
                index_change = (index_df.iloc[idx_idx]['close'] - index_df.iloc[idx_idx-1]['close']) / index_df.iloc[idx_idx-1]['close']
                if index_change < 0 and change > 0:
                    score += 1

        # 条件4: 逆势环境
        if total_count > 0:
            up_ratio = up_count / total_count
            if up_ratio < 0.3:
                score += 1

        # 条件5: 最高连板 >= 4
        if max_cons >= 4:
            score += 1

        # 条件1和2: 板块相关 (暂加2分)
        score += 2

        if score >= 4:  # 改为4分
            five_point_stocks.append((code, score))

    if not five_point_stocks:
        continue

    five_point_count += len(five_point_stocks)

    # 第二天涨停价买入，第三天收盘卖出
    next_day = all_dates[all_dates.index(date) + 1]
    sell_day = all_dates[all_dates.index(date) + 2]

    for code, score in five_point_stocks:
        df = stock_data[code]
        if next_day not in df.index or sell_day not in df.index:
            continue

        buy_price = df.loc[next_day, 'high']  # 涨停价买入
        sell_price = df.loc[sell_day, 'close']

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

print(f"\n4+分信号总数: {five_point_count}")
print(f"有效交易数: {len(trades)}")

if trades:
    df = pd.DataFrame(trades)

    print("\n" + "="*50)
    print("5分股票回测结果")
    print("="*50)

    profit = len(df[df['profit_pct'] > 0])
    loss = len(df[df['profit_pct'] < 0])
    total = len(df)

    print(f"\n总交易: {total}")
    print(f"盈利: {profit} ({profit/total*100:.1f}%)")
    print(f"亏损: {loss} ({loss/total*100:.1f}%)")
    print(f"平均收益: {df['profit_pct'].mean():.2f}%")
    print(f"最大盈利: {df['profit_pct'].max():.2f}%")
    print(f"最大亏损: {df['profit_pct'].min():.2f}%")

    print(f"\n最近20笔交易:")
    print(df[['code', 'buy_price', 'sell_price', 'profit_pct']].tail(20).to_string())
