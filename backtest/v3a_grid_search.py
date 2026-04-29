#!/usr/bin/env python3
"""v3a 30分钟策略 — Grid Search 参数优化

核心优化: czsc bi 只计算一次，参数只影响入场/出场逻辑。

阶段1: 加载数据 → czsc batch → 所有bi (一次性)
阶段2: 遍历参数组合 → 回放交易 → 计算metrics
阶段3: Top组合 → walk-forward验证

用法:
  python backtest/v3a_grid_search.py --stocks 50         # 50只快速搜索
  python backtest/v3a_grid_search.py --stocks 100        # 100只标准搜索
  python backtest/v3a_grid_search.py --stocks 100 --top 5  # 只验证top 5
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)


# ===================== 参数网格 =====================

PARAM_GRID = {
    'stop_loss_pct':  [0.06, 0.08, 0.10, 0.12, 0.15],
    'trailing_start': [0.03, 0.05, 0.07, 0.10],
    'trailing_dist':  [0.02, 0.03, 0.05, 0.08],
    'max_hold_bars':  [50, 80, 120, 160],
    'recent_bars':    [15, 20, 30, 45],
}

# 固定参数
FIXED_PARAMS = {
    'min_hold_bars': 6,
    'cooldown_bars': 3,
    'daily_ma_short': 20,
    'daily_ma_long': 60,
}


# ===================== 数据加载 =====================

def load_data(n_stocks=100):
    """加载30分钟和日线数据"""
    from data.hybrid_source import HybridSource
    from chanlun_unified.stock_pool import StockPoolManager
    from chanlun_unified.config import UnifiedConfig

    config = UnifiedConfig()
    hs = HybridSource(config.tdx_path)
    pool = StockPoolManager(config.tdx_path)
    codes = pool.get_pool('tdx_all')

    if n_stocks > 0:
        # 随机选 (但固定seed确保可复现)
        np.random.seed(42)
        codes = list(np.random.choice(codes, min(n_stocks * 3, len(codes)), replace=False))

    print(f'[1] 加载30分钟数据 ({len(codes)}只候选)...')
    data_30min = {}
    failed = 0
    for i, code in enumerate(codes):
        try:
            df = hs.get_kline(code, period='30min')
            if df is not None and len(df) >= 200:
                data_30min[code] = df
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(codes)}] 成功={len(data_30min)}')

    # 限制数量
    if n_stocks > 0 and len(data_30min) > n_stocks:
        items = list(data_30min.items())[:n_stocks]
        data_30min = dict(items)

    print(f'  30分钟数据: {len(data_30min)}只')

    # 日线数据
    print(f'[2] 加载日线数据...')
    daily_map = {}
    for code in data_30min:
        try:
            df = hs.get_kline(code, period='daily')
            if df is not None and len(df) >= 60:
                daily_map[code] = df
        except Exception:
            pass
    print(f'  日线数据: {len(daily_map)}只')

    return data_30min, daily_map


def precompute_czsc_bi(data_30min):
    """批量预计算czsc bi"""
    from core.czsc_bridge import get_czsc_bi_batch
    print(f'[3] 预计算czsc bi ({len(data_30min)}只)...')
    t0 = time.time()
    bi_map = get_czsc_bi_batch(data_30min, timeout=600)
    print(f'  完成: {len(bi_map)}/{len(data_30min)}只 ({time.time()-t0:.1f}s)')
    return bi_map


# ===================== 回测引擎 =====================

def check_daily_trend(daily_df, date, ma_short=20, ma_long=60):
    """日线趋势: MA_short > MA_long"""
    if daily_df is None or len(daily_df) < ma_long:
        return None
    mask = daily_df.index <= pd.Timestamp(date)
    subset = daily_df[mask]
    if len(subset) < ma_long:
        return None
    close = subset['close']
    return bool(close.rolling(ma_short).mean().iloc[-1] > close.rolling(ma_long).mean().iloc[-1])


def check_macd(close_series):
    """MACD确认: DIF>DEA 或 HIST递增 或 DIF递增"""
    if len(close_series) < 30:
        return False
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    i = len(close_series) - 1
    if dif.iloc[i] > dea.iloc[i]:
        return True
    if hist.iloc[i] <= 0 and hist.iloc[i] > hist.iloc[i-1]:
        return True
    if dif.iloc[i] > dif.iloc[i-1]:
        return True
    return False


def run_backtest(data_30min, daily_map, bi_map, params):
    """用给定参数跑完整回测

    参数只影响:
      - stop_loss_pct: 止损距离
      - trailing_start: 追踪止损启动
      - trailing_dist: 追踪止损距离
      - max_hold_bars: 最大持仓时间
      - recent_bars: 信号回看窗口
    """
    stop_pct = params['stop_loss_pct']
    trail_start = params['trailing_start']
    trail_dist = params['trailing_dist']
    max_hold = params['max_hold_bars']
    recent = params['recent_bars']
    min_hold = FIXED_PARAMS['min_hold_bars']

    capital = 1_000_000
    cash = capital
    positions = {}

    if not data_30min:
        return _compute_metrics([capital], [])
    trades = []
    max_positions = 5
    equity_curve = [cash]

    # 获取所有交易日
    sample = next(iter(data_30min.values()))
    all_dates = sorted(sample.index.unique())

    seen_days = set()
    for dt in all_dates:
        day_str = str(dt.date())
        if day_str in seen_days:
            continue
        seen_days.add(day_str)

        for code in list(positions.keys()):
            if code not in data_30min:
                continue
            df = data_30min[code]
            day_mask = df.index.date == dt.date()
            day_data = df[day_mask]
            if len(day_data) == 0:
                continue
            price = float(day_data.iloc[-1]['close'])
            pos = positions[code]

            profit = (price - pos['entry_price']) / pos['entry_price']
            if price > pos['highest']:
                pos['highest'] = price

            # 固定止损
            if price <= pos['stop']:
                pnl = (price - pos['entry_price']) / pos['entry_price']
                cash += pos['shares'] * price * (1 - 0.0013)
                trades.append(pnl)
                del positions[code]
                continue

            # 追踪止损
            if profit > trail_start:
                trail_stop = pos['highest'] * (1 - trail_dist)
                if price <= trail_stop:
                    pnl = (price - pos['entry_price']) / pos['entry_price']
                    cash += pos['shares'] * price * (1 - 0.0013)
                    trades.append(pnl)
                    del positions[code]
                    continue

            # 时间止损
            bars_held = len(df[df.index <= dt])
            if bars_held - pos['entry_idx'] >= max_hold:
                pnl = (price - pos['entry_price']) / pos['entry_price']
                cash += pos['shares'] * price * (1 - 0.0013)
                trades.append(pnl)
                del positions[code]
                continue

        # 买入扫描
        if len(positions) >= max_positions:
            continue

        klen_cache = {}
        for code in data_30min:
            if code in positions:
                continue
            if len(positions) >= max_positions:
                break

            df = data_30min[code]
            subset = df[df.index <= dt + pd.Timedelta(hours=15)]
            if len(subset) < 120:
                continue

            klen = len(subset)
            klen_cache[code] = klen
            close_s = subset['close']
            current_price = float(close_s.iloc[-1])

            # 日线过滤
            daily_ok = check_daily_trend(daily_map.get(code), day_str,
                                         FIXED_PARAMS['daily_ma_short'],
                                         FIXED_PARAMS['daily_ma_long'])
            if daily_ok is False:
                continue

            # bi信号: 最近recent根K线内有向下笔结束
            if code not in bi_map:
                continue

            recent_buy = False
            for bi in bi_map[code]:
                if bi['is_down'] and bi['end_idx'] >= klen - recent and bi['end_idx'] < klen:
                    recent_buy = True
                    break
            if not recent_buy:
                continue

            # MACD确认
            if not check_macd(close_s):
                continue

            # 止损计算
            low_s = subset['low']
            last_buy_idx = max(bi['end_idx'] for bi in bi_map[code]
                              if bi['is_down'] and bi['end_idx'] >= klen - recent and bi['end_idx'] < klen)
            lookback = min(30, last_buy_idx)
            recent_low = float(low_s.iloc[max(0, last_buy_idx - lookback):last_buy_idx + 1].min())
            stop = max(recent_low, current_price * (1 - stop_pct))

            # 买入
            weight = 1.0 / max_positions
            alloc = cash * weight
            shares = int(alloc * (1 - 0.0003) / current_price / 100) * 100
            if shares >= 100 and alloc <= cash:
                cash -= alloc
                positions[code] = {
                    'shares': shares,
                    'entry_price': current_price,
                    'stop': stop,
                    'highest': current_price,
                    'entry_idx': klen,
                }

        # 更新权益
        holdings = sum(pos['shares'] * float(data_30min[c].loc[dt, 'close'])
                      for c, pos in positions.items()
                      if c in data_30min and dt in data_30min[c].index)
        equity_curve.append(cash + holdings)

    return _compute_metrics(equity_curve, trades)


def _compute_metrics(equity_curve, trades):
    """计算回测指标"""
    if len(trades) == 0:
        return {'sharpe': 0, 'total_return': 0, 'max_dd': 0,
                'trades': 0, 'win_rate': 0, 'avg_pnl': 0,
                'profit_factor': 0, 'score': 0}

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)) if len(returns) > 1 else 0
    total_return = float((equity[-1] - equity[0]) / equity[0])
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_pnl = float(np.mean(trades))
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-8
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # 综合得分
    score = (0.4 * max(sharpe, 0) + 0.25 * win_rate +
             0.20 * min(profit_factor, 5) / 5 - 0.15 * abs(max_dd))

    return {
        'sharpe': round(sharpe, 2),
        'total_return': round(total_return, 4),
        'max_dd': round(max_dd, 4),
        'trades': len(trades),
        'win_rate': round(win_rate, 4),
        'avg_pnl': round(avg_pnl, 4),
        'profit_factor': round(profit_factor, 2),
        'score': round(score, 4),
    }


# ===================== Walk-Forward =====================

def walk_forward_validate(data_30min, daily_map, bi_map, params, n_splits=3):
    """Walk-forward验证: 用equity curve分段的Sharpe，与run_backtest一致

    跑完整回测，记录每日equity，按70/30分段计算Sharpe。
    """
    if not data_30min:
        return {'train_sharpe': 0, 'test_sharpe': 0, 'ratio': 0}

    sample = next(iter(data_30min.values()))
    all_dates = sorted(sample.index.unique())

    if len(all_dates) < 100:
        return {'train_sharpe': 0, 'test_sharpe': 0, 'ratio': 0}

    # 分2段: 前70% train, 后30% test
    split_idx = int(len(all_dates) * 0.7)
    split_date = all_dates[split_idx]

    # 复用run_backtest的逻辑，但记录每日equity和对应日期
    stop_pct = params['stop_loss_pct']
    trail_start = params['trailing_start']
    trail_dist = params['trailing_dist']
    max_hold = params['max_hold_bars']
    recent = params['recent_bars']

    capital = 1_000_000
    cash = capital
    positions = {}
    max_positions = 5
    train_equity = []
    test_equity = []
    train_trades = 0
    test_trades = 0

    seen_days = set()
    for dt in all_dates:
        day_str = str(dt.date())
        if day_str in seen_days:
            continue
        seen_days.add(day_str)
        is_test = dt >= split_date

        # 出场 (与run_backtest一致)
        for code in list(positions.keys()):
            if code not in data_30min:
                continue
            df = data_30min[code]
            day_mask = df.index.date == dt.date()
            day_data = df[day_mask]
            if len(day_data) == 0:
                continue
            price = float(day_data.iloc[-1]['close'])
            pos = positions[code]

            profit = (price - pos['entry_price']) / pos['entry_price']
            if price > pos['highest']:
                pos['highest'] = price

            sell = False
            if price <= pos['stop']:
                sell = True
            elif profit > trail_start:
                trail_stop = pos['highest'] * (1 - trail_dist)
                if price <= trail_stop:
                    sell = True

            if not sell:
                bars_held = len(df[df.index <= dt])
                if bars_held - pos['entry_idx'] >= max_hold:
                    sell = True

            if sell:
                pnl = (price - pos['entry_price']) / pos['entry_price']
                cash += pos['shares'] * price * (1 - 0.0013)
                if is_test:
                    test_trades += 1
                else:
                    train_trades += 1
                del positions[code]

        # 入场 (与run_backtest一致)
        if len(positions) < max_positions:
            for code in data_30min:
                if code in positions or len(positions) >= max_positions:
                    break
                df = data_30min[code]
                subset = df[df.index <= dt + pd.Timedelta(hours=15)]
                if len(subset) < 120:
                    continue
                klen = len(subset)
                close_s = subset['close']
                current_price = float(close_s.iloc[-1])

                daily_ok = check_daily_trend(daily_map.get(code), day_str,
                                             FIXED_PARAMS['daily_ma_short'],
                                             FIXED_PARAMS['daily_ma_long'])
                if daily_ok is False:
                    continue
                if code not in bi_map:
                    continue

                recent_buy = any(bi['is_down'] and bi['end_idx'] >= klen - recent
                               and bi['end_idx'] < klen for bi in bi_map[code])
                if not recent_buy:
                    continue
                if not check_macd(close_s):
                    continue

                low_s = subset['low']
                last_buy = max(bi['end_idx'] for bi in bi_map[code]
                              if bi['is_down'] and bi['end_idx'] >= klen - recent and bi['end_idx'] < klen)
                lookback = min(30, last_buy)
                recent_low = float(low_s.iloc[max(0, last_buy - lookback):last_buy + 1].min())
                stop = max(recent_low, current_price * (1 - stop_pct))

                weight = 1.0 / max_positions
                alloc = cash * weight
                shares = int(alloc * (1 - 0.0003) / current_price / 100) * 100
                if shares >= 100 and alloc <= cash:
                    cash -= alloc
                    positions[code] = {
                        'shares': shares, 'entry_price': current_price,
                        'stop': stop, 'highest': current_price, 'entry_idx': klen,
                    }

        # 计算当日equity
        holdings = sum(pos['shares'] * float(data_30min[c].loc[dt, 'close'])
                      for c, pos in positions.items()
                      if c in data_30min and dt in data_30min[c].index)
        day_equity = cash + holdings

        if is_test:
            test_equity.append(day_equity)
        else:
            train_equity.append(day_equity)

    # 用equity curve算Sharpe (与run_backtest一致)
    def equity_to_sharpe(equity_list):
        if len(equity_list) < 5:
            return 0.0
        arr = np.array(equity_list)
        returns = np.diff(arr) / arr[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0
        return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

    train_sharpe = equity_to_sharpe(train_equity)
    test_sharpe = equity_to_sharpe(test_equity)
    ratio = test_sharpe / train_sharpe if abs(train_sharpe) > 0.01 else 0

    return {
        'train_sharpe': round(train_sharpe, 2),
        'test_sharpe': round(test_sharpe, 2),
        'ratio': round(ratio, 2),
        'train_trades': train_trades,
        'test_trades': test_trades,
    }


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description='v3a Grid Search')
    parser.add_argument('--stocks', type=int, default=50, help='股票数量')
    parser.add_argument('--top', type=int, default=5, help='验证top N组合')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"v3a 参数优化 — Grid Search")
    print(f"{'='*60}")

    # 1. 加载数据
    data_30min, daily_map = load_data(args.stocks)
    if not data_30min:
        print('无数据')
        return

    # 2. 预计算czsc bi
    bi_map = precompute_czsc_bi(data_30min)

    # 3. Grid search
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f'\n[4] Grid Search: {total} 组合')

    all_results = []
    t0 = time.time()

    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        params.update(FIXED_PARAMS)

        metrics = run_backtest(data_30min, daily_map, bi_map, params)
        result = {**params, **metrics}
        all_results.append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            best_so_far = max(r['score'] for r in all_results)
            print(f'  [{i+1}/{total}] best_score={best_so_far:.3f} ({elapsed:.0f}s, ETA {eta:.0f}s)')

    elapsed = time.time() - t0
    print(f'  完成: {total}组 × {len(data_30min)}只 = {total*len(data_30min)}次回测 ({elapsed:.0f}s)')

    # 4. 排序
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('score', ascending=False)

    # 打印 Top 10
    print(f"\n{'='*60}")
    print(f"Top 10 参数组合")
    print(f"{'='*60}")
    display_cols = ['stop_loss_pct', 'trailing_start', 'trailing_dist',
                    'max_hold_bars', 'recent_bars', 'sharpe', 'win_rate',
                    'profit_factor', 'max_dd', 'trades', 'score']
    print(df_results[display_cols].head(10).to_string(index=False))

    # 5. Walk-forward 验证 Top N
    top_n = df_results.head(args.top)
    print(f"\n{'='*60}")
    print(f"Walk-Forward 验证 (Top {args.top})")
    print(f"{'='*60}")

    wf_results = []
    for idx, row in top_n.iterrows():
        params = {k: row[k] for k in PARAM_GRID.keys()}
        params.update(FIXED_PARAMS)
        wf = walk_forward_validate(data_30min, daily_map, bi_map, params)
        wf_result = {**params, **wf}
        wf_results.append(wf_result)
        verdict = 'OK' if wf['ratio'] > 0.5 else 'WARN' if wf['ratio'] > 0.3 else 'FAIL'
        print(f"  stop={params['stop_loss_pct']:.2f} trail={params['trailing_start']:.2f}/{params['trailing_dist']:.2f} "
              f"hold={params['max_hold_bars']} recent={params['recent_bars']} | "
              f"train={wf['train_sharpe']:.2f} test={wf['test_sharpe']:.2f} ratio={wf['ratio']:.2f} [{verdict}]")

    # 6. 保存结果
    os.makedirs('signals', exist_ok=True)

    # 全部结果
    df_results.to_json('signals/v3a_grid_search_results.json', orient='records', indent=2)

    # 最优参数 (取walk-forward ratio最高的)
    if wf_results:
        best_wf = max(wf_results, key=lambda x: x.get('ratio', 0))
        optimal = {k: best_wf[k] for k in PARAM_GRID.keys()}
        optimal.update(FIXED_PARAMS)
        optimal['walk_forward_ratio'] = best_wf['ratio']
        optimal['walk_forward_test_sharpe'] = best_wf['test_sharpe']
        with open('signals/v3a_optimal_params.json', 'w') as f:
            json.dump(optimal, f, indent=2)
        print(f'\n最优参数已保存: signals/v3a_optimal_params.json')
        print(f'  ratio={best_wf["ratio"]:.2f}, test_sharpe={best_wf["test_sharpe"]:.2f}')

    print(f"\n完成")


if __name__ == '__main__':
    main()
