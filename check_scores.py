"""
检查评分分布
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
        files = [f for f in os.listdir(lday_path) if f.endswith('.day')][:200]
        for f in files:
            code = f.replace('.day', '')
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
                        close = values[4] / 100
                        open_p = values[1] / 100
                        if close > 0:
                            data.append({'date': date, 'open': open_p, 'close': close})
                    except:
                        pass
            if data:
                df = pd.DataFrame(data).set_index('date')
                if len(df) > 10:
                    stock_data[f"{market}{code}"] = df

print(f"加载了 {len(stock_data)} 只股票")

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
all_dates = sorted(list(all_dates))[-100:]  # 最后100天

# 评分统计
score_distribution = defaultdict(int)
sample_scores = []

for date in all_dates:
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
                if (row['close'] - row['open']) / row['open'] >= 0.095:
                    cons += 1
                else:
                    break
            max_cons = max(max_cons, cons)

    # 对每个股票评分
    for code in stock_data.keys():
        if code not in stock_change:
            continue

        score = 0
        change = stock_change[code]

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

        score_distribution[score] += 1
        if len(sample_scores) < 10:
            sample_scores.append((date, code, score))

print("\n评分分布:")
for s in sorted(score_distribution.keys()):
    print(f"  {s}分: {score_distribution[s]} 次")

print(f"\n最高分: {max(score_distribution.keys()) if score_distribution else 0}")

print(f"\n样本评分:")
for d, c, s in sample_scores[:20]:
    print(f"  {d.strftime('%Y-%m-%d')} {c}: {s}分")
