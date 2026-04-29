"""
v2.0 增强过滤器回测
对比基准(scanner v2.1) vs v2.0(大盘分层+板块过滤+仓位分级)

大盘分层标准(以沪指000001.SH为例):
  - 上涨笔: MA5>MA10 且 price>MA5
  - 震荡:   不满足上涨也不满足下跌
  - 下跌笔: MA5<MA10 且 price<MA5

板块过滤:
  - 正向: 板块在20日线上方 + 涨跌>=大盘
  - 否决: 板块今日跌幅>2% → 该板块全部排除

仓位分级:
  - 上涨笔: 1买20%/2买40%/3买50%
  - 震荡:   1买10%/2买30%/3买40%
  - 下跌笔: 1买10%/2买观望/3买观望
"""

import sys
sys.path.insert(0, '/workspace/chanlun_system/code')
sys.path.insert(0, '/workspace/scanner')

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ============ 数据路径 ============
TDX_DAY = '/workspace/tdx_data'
SCANNER_CACHE = '/workspace/scanner/scanner_new_fw_cache_120.pkl'
OUTPUT_DIR = '/workspace/backtest_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 缠论引擎 ============
from signal_engine import SignalEngine

# ============ 大盘指数代码 ============
INDEX_CODE = '000001.SH'  # 沪指

def load_index_daily(code=INDEX_CODE):
    """加载大盘日线"""
    path = f"{TDX_DAY}/{code}.day"
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            n = int.from_bytes(f.read(4), 'little')
            if n <= 0:
                return None
            f.seek(0)
            data = np.fromfile(f, dtype=np.dtype([
                ('date', 'i4'), ('open', 'f4'), ('high', 'f4'),
                ('low', 'f4'), ('close', 'f4'), ('amount', 'f4'), ('vol', 'i4')
            ]))
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        return df
    except:
        return None

def compute_market_env(df_index):
    """计算大盘环境: up/trend/down"""
    if df_index is None or len(df_index) < 30:
        return 'unknown'
    close = df_index['close'].values
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    price = close[-1]
    ma5_cur = ma5[-1]
    ma10_cur = ma10[-1]
    if ma5_cur > ma10_cur and price > ma5_cur:
        return 'up'
    elif ma5_cur < ma10_cur and price < ma5_cur:
        return 'down'
    else:
        return 'trend'

# ============ 板块数据(腾讯行情批量) ============
def load_sector_data_today():
    """从腾讯行情获取今日板块涨跌"""
    import urllib.request
    try:
        url = 'https://qt.gtimg.cn/q=s_bk0481'  # 行业板块
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read().decode('gbk')
        sectors = {}
        for line in raw.strip().split('\n'):
            parts = line.split('~')
            if len(parts) > 32:
                code = parts[1].strip()
                name = parts[2].strip()
                change_pct = float(parts[32]) if parts[32] else 0.0
                sectors[code] = {'name': name, 'change_pct': change_pct}
        return sectors
    except:
        return {}

# ============ 股票→板块映射(简化版:用tushare) ============
SECTOR_MAP = {}  # code -> sector_code

def get_sector_map():
    """加载 股票→板块 映射"""
    cache = '/workspace/scanner/sector_map.pkl'
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    return {}

# ============ 核心回测函数 ============
def get_buy_price(code, buy_type, bi_confirm_delay=2):
    """获取缠论买点价格"""
    day_path = f"{TDX_DAY}/{code}.day"
    if not os.path.exists(day_path):
        return None, None, None

    try:
        with open(day_path, 'rb') as f:
            n = int.from_bytes(f.read(4), 'little')
            if n <= 0:
                return None, None, None
            f.seek(0)
            data = np.fromfile(f, dtype=np.dtype([
                ('date', 'i4'), ('open', 'f4'), ('high', 'f4'),
                ('low', 'f4'), ('close', 'f4'), ('amount', 'f4'), ('vol', 'i4')
            ]))
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 60:
            return None, None, None

        engine = SignalEngine(bi_confirm_delay=bi_confirm_delay)
        result = engine.generate(df, end_idx=None)
        if result is None:
            return None, None, None

        signals = result.get('signals', [])
        buy_signals = [s for s in signals if s.get('action') == 'buy']
        if not buy_signals:
            return None, None, None

        # 找对应买点
        for s in reversed(buy_signals):
            bt = s.get('buy_type', '')
            if buy_type == '1buy' and '1buy' in bt:
                return s['price'], s.get('stop_loss'), bt
            elif buy_type == '2buy' and '2buy' in bt:
                return s['price'], s.get('stop_loss'), bt
            elif buy_type == '3buy' and '3buy' in bt:
                return s['price'], s.get('stop_loss'), bt

        return None, None, None
    except:
        return None, None, None

# ============ 回测参数 ============
TRADING_DAYS = 252
COMMISSION = 0.0003  # 0.03%
SLIPPAGE = 0.001     # 0.1%

# 仓位分级表 (v2.0)
POSITION_TABLE = {
    'up':    {'1buy': 0.20, '2buy': 0.40, '3buy': 0.50},
    'trend': {'1buy': 0.10, '2buy': 0.30, '3buy': 0.40},
    'down':  {'1buy': 0.10, '2buy': 0.00, '3buy': 0.00},  # 下跌笔不做2买3买
}

# SL表
SL_TABLE = {
    'up':    {'1buy': 0.06, '2buy': 0.03},
    'trend': {'1buy': 0.06, '2buy': 0.03},
    'down':  {'1buy': 0.06, '2buy': 0.00},
}

# TP表 (动态止盈简化)
TP_TABLE = {
    'up':    {'t1': 0.03, 't2': 0.08, 't3': 0.15},
    'trend': {'t1': 0.03, 't2': 0.06, 't3': 0.10},
    'down':  {'t1': 0.03, 't2': 0.05, 't3': 0.08},
}

def compute_trade_result(entry_price, exit_price, position, direction=1):
    """计算单笔交易结果"""
    pnl_pct = (exit_price - entry_price) / entry_price * direction
    net_pnl = pnl_pct - COMMISSION - SLIPPAGE
    return net_pnl

def run_backtest_for_stock(code, market_env, buy_type, index_df, trading_dates):
    """对单只股票回测某个买点类型"""
    entry_price, sl_price, buy_type_label = get_buy_price(code, buy_type)
    if entry_price is None:
        return None

    # 找到入场日后N天的价格(模拟实际执行)
    day_path = f"{TDX_DAY}/{code}.day"
    try:
        with open(day_path, 'rb') as f:
            n = int.from_bytes(f.read(4), 'little')
            f.seek(0)
            data = np.fromfile(f, dtype=np.dtype([
                ('date', 'i4'), ('open', 'f4'), ('high', 'f4'),
                ('low', 'f4'), ('close', 'f4'), ('amount', 'f4'), ('vol', 'i4')
            ]))
        df_stock = pd.DataFrame(data)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y%m%d')
        df_stock = df_stock.sort_values('date').reset_index(drop=True)
    except:
        return None

    # 找到entry_price对应的日期索引
    date_idx = None
    for i, row in df_stock.iterrows():
        if abs(row['close'] - entry_price) / entry_price < 0.02:  # 容错
            date_idx = i
            break
    if date_idx is None or date_idx + 1 >= len(df_stock):
        return None

    # 仓位
    pos = POSITION_TABLE.get(market_env, {}).get(buy_type, 0)
    if pos <= 0:
        return None

    # 止损
    sl_pct = SL_TABLE.get(market_env, {}).get(buy_type, 0.06)
    if sl_pct <= 0:
        return None
    stop_loss = entry_price * (1 - sl_pct)

    # 止盈
    tp_pcts = TP_TABLE.get(market_env, {})

    # 模拟持有N天
    exit_price = None
    exit_reason = None
    hold_days = 0
    max_price = entry_price

    for i in range(date_idx + 1, min(date_idx + 60, len(df_stock))):
        high = df_stock.iloc[i]['high']
        low = df_stock.iloc[i]['low']
        close = df_stock.iloc[i]['close']
        max_price = max(max_price, high)

        # 止损
        if low <= stop_loss:
            exit_price = stop_loss
            exit_reason = 'SL'
            hold_days = i - date_idx
            break

        # 动态止盈
        pnl_pct = (max_price - entry_price) / entry_price
        if pnl_pct >= tp_pcts.get('t3', 0.15) and close < max_price * 0.99:
            exit_price = max_price * 0.99
            exit_reason = 'TP3'
            hold_days = i - date_idx
            break
        elif pnl_pct >= tp_pcts.get('t2', 0.08) and close < max_price * 0.98:
            exit_price = max_price * 0.98
            exit_reason = 'TP2'
            hold_days = i - date_idx
            break
        elif pnl_pct >= tp_pcts.get('t1', 0.03) and close < max_price * 0.97:
            exit_price = max_price * 0.97
            exit_reason = 'TP1'
            hold_days = i - date_idx
            break

    if exit_price is None:
        # 持有60天后强制平仓
        exit_price = df_stock.iloc[min(date_idx + 60, len(df_stock) - 1)]['close']
        exit_reason = 'HOLD60'
        hold_days = 60

    net_pnl = compute_trade_result(entry_price, exit_price, pos)
    real_pnl = net_pnl * pos  # 按实际仓位计算

    return {
        'code': code,
        'buy_type': buy_type,
        'market_env': market_env,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_days': hold_days,
        'pos': pos,
        'net_pnl_pct': net_pnl * 100,
        'real_pnl_pct': real_pnl * 100,
    }

# ============ 批量回测(采样500只) ============
def run_full_backtest():
    """完整回测流程"""
    print("=" * 60)
    print("v2.0 增强过滤器回测")
    print("=" * 60)

    # 加载大盘数据
    print("\n[1/5] 加载大盘数据...")
    df_index = load_index_daily()
    if df_index is None:
        print("ERROR: 无法加载大盘数据")
        return
    print(f"  大盘数据: {len(df_index)} 条, 日期范围: {df_index['date'].min()} ~ {df_index['date'].max()}")

    # 加载股票列表
    print("\n[2/5] 加载股票列表...")
    cache_path = SCANNER_CACHE
    if not os.path.exists(cache_path):
        print("ERROR: scanner cache not found")
        return
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    stock_signals = cache_data.get('stock_signals', [])
    print(f"  scanner缓存: {len(stock_signals)} 只股票")

    # 过滤有效信号
    valid_stocks = []
    for item in stock_signals:
        code = item.get('code', '')
        if code.startswith('6') or code.startswith('0') or code.startswith('3'):
            if item.get('signals'):
                valid_stocks.append(item)

    print(f"  有效扫描股票: {len(valid_stocks)} 只")

    # 采样(避免太慢)
    np.random.seed(42)
    sample_size = min(500, len(valid_stocks))
    sampled = np.random.choice(valid_stocks, sample_size, replace=False)

    # 按买点类型分组
    results = {'1buy': [], '2buy': [], '3buy': []}

    print(f"\n[3/5] 回测采样 {sample_size} 只股票...")
    for item in tqdm(sampled):
        code = item['code']
        signals = item.get('signals', [])

        # 当前大盘环境(用最后一条大盘数据计算)
        market_env = compute_market_env(df_index)

        for buy_type in ['1buy', '2buy', '3buy']:
            r = run_backtest_for_stock(code, market_env, buy_type, df_index, None)
            if r:
                results[buy_type].append(r)

    # ============ 输出结果 ============
    print("\n" + "=" * 60)
    print("回测结果 (v2.0 大盘分层+仓位分级)")
    print("=" * 60)
    print(f"\n大盘环境: {market_env}")
    print(f"采样: {sample_size} 只股票")

    for bt, trades in results.items():
        if not trades:
            print(f"\n【{bt}】 无信号")
            continue
        df_bt = pd.DataFrame(trades)
        win_rate = (df_bt['net_pnl_pct'] > 0).mean() * 100
        avg_pnl = df_bt['net_pnl_pct'].mean()
        avg_hold = df_bt['hold_days'].mean()
        sharpe = avg_pnl / df_bt['net_pnl_pct'].std() * np.sqrt(252 / avg_hold) if df_bt['net_pnl_pct'].std() > 0 else 0
        max_dd = df_bt['net_pnl_pct'].min()

        print(f"\n【{bt.upper()}】 信号数: {len(trades)}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  均盈: {avg_pnl:.2f}%")
        print(f"  均亏: {df_bt[df_bt['net_pnl_pct']<0]['net_pnl_pct'].mean():.2f}%")
        print(f"  Sharpe(简化): {sharpe:.2f}")
        print(f"  最大回撤: {max_dd:.2f}%")
        print(f"  平均持仓: {avg_hold:.1f} 天")

        # 按市场环境分组
        for env in ['up', 'trend', 'down']:
            subset = df_bt[df_bt['market_env'] == env]
            if len(subset) > 0:
                wr = (subset['net_pnl_pct'] > 0).mean() * 100
                ap = subset['net_pnl_pct'].mean()
                print(f"    {env}: n={len(subset)}, WR={wr:.1f}%, 均盈={ap:.2f}%")

    # 保存结果
    output_path = f"{OUTPUT_DIR}/v2bt_results_{market_env}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n结果已保存: {output_path}")

if __name__ == '__main__':
    run_full_backtest()
