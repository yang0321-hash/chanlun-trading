"""
回测：不同持仓天数的效果
"""

import os
import struct
import pandas as pd
from collections import defaultdict

tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"
vipdoc_path = os.path.join(tdx_path, "vipdoc")

# 读取股票数据
print("正在加载数据...")
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
                        if close > 0 and close < 100:
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
                index_data.append({'date': date, 'open': open_p, 'close': close})
        except:
            pass
index_df = pd.DataFrame(index_data).set_index('date')

# 获取所有交易日期
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index)
all_dates = sorted(list(all_dates))[-200:]

# 测试不同持仓天数
hold_days_list = [1, 3, 5, 10]  # 买入后持有N天
results = {}

for hold_days in hold_days_list:
    print(f"\n{'='*50}")
    print(f"回测：持有 {hold_days} 天")
    print(f"{'='*50}")

    trades = []

    for i, date in enumerate(all_dates[:-hold_days-1]):
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

        # 筛选4分股票
        four_point_stocks = []
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

            if score >= 4:
                four_point_stocks.append(code)

        if not four_point_stocks:
            continue

        # 第二天涨停价买入，持有N天后收盘卖出
        buy_day = all_dates[all_dates.index(date) + 1]
        sell_day_idx = all_dates.index(date) + 1 + hold_days
        if sell_day_idx >= len(all_dates):
            continue
        sell_day = all_dates[sell_day_idx]

        for code in four_point_stocks:
            df = stock_data[code]
            if buy_day not in df.index or sell_day not in df.index:
                continue

            buy_price = df.loc[buy_day, 'high']  # 涨停价买入
            sell_price = df.loc[sell_day, 'close']

            profit_pct = (sell_price - buy_price) / buy_price * 100
            trades.append(profit_pct)

    if not trades:
        print("无交易数据")
        continue

    trades_df = pd.Series(trades)

    profit = len(trades_df[trades_df > 0])
    loss = len(trades_df[trades_df < 0])
    total = len(trades_df)

    print(f"\n结果 (持有{hold_days}天):")
    print(f"  总交易: {total}")
    print(f"  盈利: {profit} ({profit/total*100:.1f}%)")
    print(f"  亏损: {loss} ({loss/total*100:.1f}%)")
    print(f"  平均收益: {trades_df.mean():.2f}%")
    print(f"  最大盈利: {trades_df.max():.2f}%")
    print(f"  最大亏损: {trades_df.min():.2f}%")

    results[hold_days] = {
        'total': total,
        'profit': profit,
        'loss': loss,
        'win_rate': profit/total*100,
        'avg_return': trades_df.mean(),
        'max_profit': trades_df.max(),
        'max_loss': trades_df.min()
    }

# 汇总对比
print(f"\n{'='*60}")
print("持仓天数对比汇总")
print(f"{'='*60}")
print(f"{'持仓天数':<10} {'交易数':<10} {'胜率':<10} {'平均收益':<12} {'最大盈利':<12} {'最大亏损':<12}")
print("-"*60)
for days in hold_days_list:
    if days in results:
        r = results[days]
        print(f"{days:<10} {r['total']:<10} {r['win_rate']:<10.1f}% {r['avg_return']:<12.2f}% {r['max_profit']:<12.2f}% {r['max_loss']:<12.2f}%")
