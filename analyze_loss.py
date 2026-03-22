"""
分析最大亏损产生的原因
"""

import os
import struct
import pandas as pd

tdx_path = r"D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)"
vipdoc_path = os.path.join(tdx_path, "vipdoc")

# 读取股票数据
def read_day_file(filepath):
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
        return pd.DataFrame(data).set_index('date')
    return pd.DataFrame()

# 查找最大亏损案例
print("正在分析最大亏损...")

worst_trades = []

for market in ['sh', 'sz']:
    lday_path = os.path.join(vipdoc_path, market, 'lday')
    if os.path.exists(lday_path):
        files = [f for f in os.listdir(lday_path) if f.endswith('.day')][:500]

        for f in files:
            code = f.replace('.day', '').replace(market, '')
            filepath = os.path.join(lday_path, f)
            df = read_day_file(filepath)

            if len(df) < 5:
                continue

            # 遍历最近交易日
            start_idx = max(0, len(df) - 100)
            for i in range(start_idx, len(df) - 2):
                signal_date = df.index[i]
                buy_date = df.index[i + 1]
                sell_date = df.index[i + 2]

                # 信号日涨幅
                if i > 0:
                    signal_change = (df.iloc[i]['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
                else:
                    signal_change = 0

                # 涨停价买入
                buy_price = df.loc[buy_date, 'high']
                sell_price = df.loc[sell_date, 'close']

                profit_pct = (sell_price - buy_price) / buy_price * 100

                if profit_pct < -20:  # 亏损超过20%
                    worst_trades.append({
                        'code': f"{market}{code}",
                        'signal_date': signal_date,
                        'signal_change': signal_change * 100,
                        'buy_date': buy_date,
                        'buy_price': buy_price,
                        'buy_open': df.loc[buy_date, 'open'],
                        'buy_close': df.loc[buy_date, 'close'],
                        'buy_high': df.loc[buy_date, 'high'],
                        'sell_date': sell_date,
                        'sell_price': sell_price,
                        'profit_pct': profit_pct
                    })

# 按亏损排序
worst_trades.sort(key=lambda x: x['profit_pct'])

print(f"\n找到 {len(worst_trades)} 笔亏损超过20%的交易")
print("\n亏损最严重的10笔:")

print(f"{'代码':<12} {'信号日':<12} {'信号涨幅':<10} {'买入日期':<12} {'买入价':<10} {'买入开盘':<10} {'买入收盘':<10} {'卖出价':<10} {'亏损':<10}")
print("-" * 120)

for t in worst_trades[:10]:
    print(f"{t['code']:<12} {str(t['signal_date']):<12} {t['signal_change']:<10.2f} {str(t['buy_date']):<12} {t['buy_price']:<10.2f} {t['buy_open']:<10.2f} {t['buy_close']:<10.2f} {t['sell_price']:<10.2f} {t['profit_pct']:<10.2f}")

# 分析原因
print("\n" + "="*60)
print("亏损原因分析:")
print("="*60)

if worst_trades:
    # 统计买入价与开盘价的差异
    buy_vs_open = [(t['buy_price'] - t['buy_open']) / t['buy_open'] * 100 for t in worst_trades]
    avg_buy_over_open = sum(buy_vs_open) / len(buy_vs_open)

    print(f"\n1. 买入价(最高价)比开盘价平均高: {avg_buy_over_open:.2f}%")

    # 统计买入日下跌情况
    buy_day_drop = [(t['buy_close'] - t['buy_open']) / t['buy_open'] * 100 for t in worst_trades]
    avg_buy_day_drop = sum(buy_day_drop) / len(buy_day_drop)

    print(f"2. 买入日平均跌幅: {avg_buy_day_drop:.2f}%")

    # 统计次日继续下跌
    sell_day_drop = [(t['sell_price'] - t['buy_close']) / t['buy_close'] * 100 for t in worst_trades]
    avg_sell_day_drop = sum(sell_day_drop) / len(sell_day_drop)

    print(f"3. 卖出日相对买入日收盘平均跌幅: {avg_sell_day_drop:.2f}%")
