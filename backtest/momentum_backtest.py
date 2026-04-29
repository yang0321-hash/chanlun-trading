#!/usr/bin/env python3
"""动量+均值回归策略 — 全A回测 + 滚动窗口验证

用法:
  python backtest/momentum_backtest.py                  # 全A两套参数
  python backtest/momentum_backtest.py --stocks 50      # 50只快速验证
  python backtest/momentum_backtest.py --preset short   # 只跑短线
"""

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.momentum_reversion import MomentumReversionEngine, PRESETS


def compute_metrics(signals, data_map, initial_capital=1_000_000, max_positions=5):
    """通用回测指标计算 — 带全局仓位限制和信号排序

    信号编码:
      0     = 空仓/卖出
      0.05  = 减半仓持有
      0.1   = 满仓持有
      3~6   = 入场信号(分数越高越好)
    """
    all_dates = set()
    for sig in signals.values():
        all_dates.update(sig.index)
    all_dates = sorted(all_dates)
    if not all_dates:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0, 'trade_count': 0, 'win_rate': 0}

    cash = initial_capital
    positions = {}  # code -> {shares, entry_price}
    equity_curve = [cash]
    trades = []

    for dt in all_dates:
        # === 第一轮: 处理卖出 ===
        for code in list(positions.keys()):
            if code not in signals or code not in data_map:
                continue
            sig = signals[code]
            if dt not in sig.index or dt not in data_map[code].index:
                continue

            val = float(sig.loc[dt])
            if val <= 0:
                # 清仓卖出
                price = float(data_map[code].loc[dt, 'close'])
                pos = positions[code]
                pnl_pct = (price - pos['entry_price']) / pos['entry_price']
                cash += pos['shares'] * price * (1 - 0.0003)
                trades.append({'code': code, 'pnl_pct': pnl_pct, 'date': str(dt.date())})
                del positions[code]

        # === 第二轮: 收集买入候选(信号>1 = 入场分数) ===
        buy_candidates = []
        for code, sig in signals.items():
            if code in positions or code not in data_map:
                continue
            if dt not in sig.index or dt not in data_map[code].index:
                continue
            val = float(sig.loc[dt])
            if val >= 3:  # 入场分数 3~6
                buy_candidates.append((code, val))

        # 按分数降序排序, 取 top N
        buy_candidates.sort(key=lambda x: -x[1])
        available_slots = max_positions - len(positions)

        for code, score in buy_candidates[:available_slots]:
            price = float(data_map[code].loc[dt, 'close'])
            weight = 1.0 / max_positions  # 等权
            alloc = cash * weight
            shares = (alloc * (1 - 0.0003)) / price
            if shares > 100 and alloc <= cash:
                cash -= alloc
                positions[code] = {'shares': shares, 'entry_price': price}

        # === 权益计算 ===
        holdings = sum(
            pos['shares'] * float(data_map[c].loc[dt, 'close'])
            for c, pos in positions.items()
            if c in data_map and dt in data_map[c].index
        )
        equity_curve.append(cash + holdings)

    eq = np.array(equity_curve)
    if len(eq) < 2:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0, 'trade_count': 0, 'win_rate': 0}

    returns = np.diff(eq) / eq[:-1]
    total_return = eq[-1] / eq[0] - 1
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    total = len(trades)
    win_rate = wins / total if total > 0 else 0

    return {
        'final_value': float(eq[-1]),
        'total_return': float(total_return),
        'annual_return': float((1 + total_return) ** (252 / max(len(equity_curve), 1)) - 1),
        'max_drawdown': float(max_dd),
        'sharpe': float(sharpe),
        'trade_count': total,
        'win_count': wins,
        'loss_count': total - wins,
        'win_rate': float(win_rate),
        'days': len(equity_curve),
        'avg_trades_per_day': total / max(len(all_dates), 1),
    }


def run_backtest(data_map, preset_name, start_date, end_date):
    """运行全量回测"""
    filtered = {}
    for code, df in data_map.items():
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        f = df[mask]
        if len(f) >= 80:
            filtered[code] = f

    engine = MomentumReversionEngine(preset=preset_name)
    signals = engine.generate(filtered)
    metrics = compute_metrics(signals, filtered)

    return signals, metrics


def main():
    parser = argparse.ArgumentParser(description='动量+均值回归策略回测')
    parser.add_argument('--stocks', type=int, default=0, help='限制股票数(0=全部)')
    parser.add_argument('--preset', default='both', help='参数组: short/mid/both')
    parser.add_argument('--start', default='2024-06-01')
    parser.add_argument('--end', default='2026-04-17')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"动量+均值回归策略回测")
    print(f"{'='*70}")

    # 加载数据
    print(f'\n[1] 加载数据...')
    t0 = time.time()

    try:
        from chanlun_unified.data_adapter import HybridSourceAdapter
        from chanlun_unified.stock_pool import StockPoolManager
        from chanlun_unified.config import UnifiedConfig

        config = UnifiedConfig(lookback=500)
        adapter = HybridSourceAdapter(config.tdx_path)
        pool = StockPoolManager(config.tdx_path)
        codes = pool.get_pool('tdx_all')
    except Exception as e:
        print(f'TDX数据加载失败: {e}')
        print('尝试AKShare...')
        # Fallback: 从tushare获取
        sys.path.insert(0, r'C:\Users\nick0\AppData\Local\Programs\Python\Python312\Lib\site-packages')
        import tushare as ts
        import os
        os.environ['NO_PROXY'] = '*'
        pro = ts.pro_api(os.environ.get('TUSHARE_TOKEN', ''))
        pro._DataApi__http_url = 'http://111.170.34.57:8010/'
        codes_df = pro.stock_basic(list_status='L', fields='ts_code')
        codes = codes_df['ts_code'].tolist()

    if args.stocks > 0:
        codes = codes[:args.stocks]

    print(f'  股票池: {len(codes)} 只')

    # 加载K线
    data_map = {}
    if 'adapter' in dir():
        data_map = adapter.fetch_daily_map(codes, lookback=500, max_workers=4, verbose=True)
    else:
        # tushare fallback
        from tqdm import tqdm
        for code in tqdm(codes, desc='拉取日线'):
            try:
                df = pro.daily(ts_code=code, start_date='20231001', end_date=args.end.replace('-',''))
                if df is not None and len(df) > 80:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df = df.set_index('trade_date').sort_index()
                    df = df.rename(columns={'vol': 'volume'})
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    data_map[code] = df
                time.sleep(0.15)
            except Exception:
                pass

    print(f'  加载成功: {len(data_map)} 只 ({time.time()-t0:.0f}s)')

    if not data_map:
        print('无数据')
        return

    # 运行回测
    presets_to_run = ['short', 'mid'] if args.preset == 'both' else [args.preset]
    results = {}

    for preset_name in presets_to_run:
        print(f"\n{'='*70}")
        print(f"[{preset_name}] {PRESETS[preset_name]['ma_short']}/{PRESETS[preset_name]['ma_long']}均线 | "
              f"时间止损{PRESETS[preset_name]['time_stop_bars']}天 | "
              f"止损{PRESETS[preset_name]['max_stop_pct']*100:.0f}%")
        print(f"{'='*70}")

        t1 = time.time()
        sig, metrics = run_backtest(data_map, preset_name, args.start, args.end)
        elapsed = time.time() - t1

        print(f'\n  结果:')
        print(f'    Sharpe: {metrics["sharpe"]:.2f}')
        print(f'    收益率: {metrics["total_return"]*100:.1f}%')
        print(f'    年化:   {metrics["annual_return"]*100:.1f}%')
        print(f'    回撤:   {metrics["max_drawdown"]*100:.1f}%')
        print(f'    交易:   {metrics["trade_count"]}笔')
        print(f'    胜率:   {metrics["win_rate"]*100:.1f}%')
        print(f'    耗时:   {elapsed:.0f}s')

        results[preset_name] = metrics

    # 对比
    if len(results) == 2:
        print(f"\n{'='*70}")
        print(f"对比")
        print(f"{'='*70}")
        for k in ['sharpe', 'total_return', 'max_drawdown', 'trade_count', 'win_rate']:
            sv = results['short'].get(k, 0)
            mv = results['mid'].get(k, 0)
            fmt = '.2f' if k == 'sharpe' else ('.1%' if isinstance(sv, float) and abs(sv) < 1 else 'd')
            if isinstance(sv, float) and abs(sv) < 1:
                print(f'  {k:<20} short={sv*100:.1f}%  mid={mv*100:.1f}%')
            else:
                print(f'  {k:<20} short={sv:{fmt}}  mid={mv:{fmt}}')

    # 保存
    out = {
        'config': {'stocks': len(data_map), 'period': f'{args.start}~{args.end}'},
        'results': {k: {kk: f'{vv:.4f}' if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in results.items()},
    }
    out_dir = PROJECT_ROOT / 'backtest'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'momentum_result.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f'\n[保存] {out_path}')


if __name__ == '__main__':
    main()
