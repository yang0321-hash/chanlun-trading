"""
拉取/生成30分钟K线数据

策略：
1. 优先使用mootdx拉取（如可用）
2. 备选：从日线数据模拟生成30分钟K线（使用日内价格波动模型）

保存到 artifacts/min30_{code}.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = "/workspace/chanlun_system/artifacts"
CODES = ["000001.SZ", "000333.SZ", "000858.SZ", "002415.SZ",
         "002600.SZ", "600036.SH", "600519.SH", "601318.SH"]

# 尝试mootdx拉取
def try_mootdx_fetch(ts_code: str, mootdx_code: str) -> pd.DataFrame:
    """尝试通过mootdx拉取30分钟数据"""
    try:
        from mootdx.quotes import Quotes
        import time
        client = Quotes.factory(market='std')
        all_dfs = []
        for batch in range(5):
            offset = batch * 800
            df = client.bars(symbol=mootdx_code, frequency=2, offset=offset)
            if df is None or len(df) == 0:
                break
            all_dfs.append(df)
            if len(df) < 800:
                break
            time.sleep(0.3)
        if not all_dfs:
            return pd.DataFrame()
        result = pd.concat(all_dfs, ignore_index=False)
        result = result[~result.index.duplicated(keep='first')]
        result = result.sort_index()
        return result
    except Exception:
        return pd.DataFrame()


def generate_min30_from_daily(daily_path: str) -> pd.DataFrame:
    """
    从日线数据模拟生成30分钟K线
    
    使用日内价格波动模型：
    - 每天生成8根30分钟K线（A股交易时间 9:30-11:30, 13:00-15:00）
    - 日内价格路径使用几何布朗运动 + 均值回归
    - 开盘=日线开盘，收盘=日线收盘，最高=日线最高，最低=日线最低
    """
    daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    if len(daily) == 0:
        return pd.DataFrame()
    
    # 30分钟时间戳模板
    time_slots = [
        '09:30:00', '10:00:00', '10:30:00', '11:00:00',
        '13:00:00', '13:30:00', '14:00:00', '14:30:00'
    ]
    
    np.random.seed(42)  # 可复现
    
    all_rows = []
    for date_idx in range(len(daily)):
        d = daily.index[date_idx]
        date_str = pd.Timestamp(d).strftime('%Y-%m-%d')
        
        day_open = float(daily.iloc[date_idx]['open'])
        day_close = float(daily.iloc[date_idx]['close'])
        day_high = float(daily.iloc[date_idx]['high'])
        day_low = float(daily.iloc[date_idx]['low'])
        day_vol = float(daily.iloc[date_idx]['volume'])
        
        # 确保high >= max(open,close) and low <= min(open,close)
        day_high = max(day_high, day_open, day_close)
        day_low = min(day_low, day_open, day_close)
        
        if day_high == day_low:
            day_high = day_low + 0.01
        
        # 生成日内价格路径
        # 使用beta分布模拟价格在日内的路径
        # 8个时间点，价格从day_open到day_close
        n_bars = 8
        
        # 生成随机中间路径
        prices = np.zeros(n_bars + 1)
        prices[0] = day_open
        prices[-1] = day_close
        
        # 简单的插值 + 随机波动
        trend = (day_close - day_open) / n_bars
        for k in range(1, n_bars):
            # 添加趋势 + 随机波动
            noise = np.random.normal(0, (day_high - day_low) * 0.15)
            prices[k] = prices[k-1] + trend + noise
            # 约束在日线范围内（允许略微超出后修正）
            prices[k] = np.clip(prices[k], day_low * 0.998, day_high * 1.002)
        
        # 确保最高/最低点出现
        # 找到需要插入高低点的位置
        if day_high > max(prices):
            # 随机选一个中间bar作为最高点
            insert_idx = np.random.randint(1, n_bars)
            prices[insert_idx] = day_high
        if day_low < min(prices):
            insert_idx = np.random.randint(1, n_bars)
            prices[insert_idx] = day_low
        
        # 重新约束最终价格
        prices[-1] = day_close
        
        # 从价格路径构建30分钟K线
        per_bar_vol = day_vol / n_bars
        
        for bar_idx in range(n_bars):
            bar_open = prices[bar_idx]
            bar_close = prices[bar_idx + 1]
            
            # 30分钟内的高低点
            bar_range = abs(bar_close - bar_open)
            bar_high = max(bar_open, bar_close) + abs(np.random.normal(0, bar_range * 0.3))
            bar_low = min(bar_open, bar_close) - abs(np.random.normal(0, bar_range * 0.3))
            
            # 确保bar高低不超过日线范围（宽松约束）
            bar_high = min(bar_high, day_high * 1.001)
            bar_low = max(bar_low, day_low * 0.999)
            bar_high = max(bar_high, bar_open, bar_close)
            bar_low = min(bar_low, bar_open, bar_close)
            
            ts = f"{date_str} {time_slots[bar_idx]}"
            
            # 成交量分布：开盘和收盘时量大，中午量小
            vol_weights = [1.3, 1.0, 0.8, 0.6, 0.7, 0.8, 1.0, 1.4]
            bar_vol = per_bar_vol * vol_weights[bar_idx] * (0.8 + np.random.random() * 0.4)
            
            all_rows.append({
                'datetime': ts,
                'open': round(bar_open, 4),
                'close': round(bar_close, 4),
                'high': round(bar_high, 4),
                'low': round(bar_low, 4),
                'volume': round(bar_vol, 2),
            })
    
    result = pd.DataFrame(all_rows)
    result['datetime'] = pd.to_datetime(result['datetime'])
    result = result.set_index('datetime')
    result = result.sort_index()
    
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    mootdx_map = {
        "000001.SZ": "000001",
        "000333.SZ": "000333", 
        "000858.SZ": "000858",
        "002415.SZ": "002415",
        "002600.SZ": "002600",
        "600036.SH": "600036",
        "600519.SH": "600519",
        "601318.SH": "601318",
    }
    
    for code in CODES:
        output_path = os.path.join(OUTPUT_DIR, f"min30_{code}.csv")
        daily_path = os.path.join(OUTPUT_DIR, f"ohlcv_{code}.csv")
        
        # 1. 尝试mootdx
        print(f"[处理] {code}")
        df = try_mootdx_fetch(code, mootdx_map.get(code, code.replace('.SZ','').replace('.SH','')))
        
        if len(df) >= 3000:
            # 标准化列名
            if 'vol' in df.columns and 'volume' not in df.columns:
                df = df.rename(columns={'vol': 'volume'})
            df.index.name = 'datetime'
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.to_csv(output_path)
            print(f"  [mootdx] 保存 {len(df)} 根30分钟K线")
            continue
        
        # 2. 检查已有文件
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path, index_col=0, parse_dates=True)
            if len(existing) >= 3000:
                print(f"  [已有] {len(existing)} 根30分钟K线，跳过")
                continue
        
        # 3. 从日线生成
        if not os.path.exists(daily_path):
            print(f"  [跳过] 日线数据不存在: {daily_path}")
            continue
        
        print(f"  [生成] 从日线模拟30分钟K线...")
        df = generate_min30_from_daily(daily_path)
        
        if df.empty:
            print(f"  [失败] 无法生成30分钟数据")
            continue
        
        df.to_csv(output_path)
        print(f"  [保存] {output_path}: {len(df)} 根K线, "
              f"日期范围 {df.index[0]} ~ {df.index[-1]}")
    
    print("\n=== 完成 ===")
    # 验证
    for code in CODES:
        output_path = os.path.join(OUTPUT_DIR, f"min30_{code}.csv")
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, index_col=0, parse_dates=True, nrows=3)
            full_count = sum(1 for _ in open(output_path)) - 1
            print(f"  {code}: {full_count} 根K线")
        else:
            print(f"  {code}: 文件不存在!")


if __name__ == "__main__":
    main()
