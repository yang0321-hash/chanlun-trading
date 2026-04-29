#!/usr/bin/env python3
"""v3a 30分钟策略 — 滚动窗口回测验证

目的: 确认v3a策略是否存在 look-ahead bias

原理:
  全量回测: 一次性加载完整K线 → 计算缠论结构 → 匹配信号
  滚动窗口: 每个重平衡日 t, 只用 [t-window, t] 的数据
            重新计算缠论 → 检查是否有新信号

如果 滚动窗口Sharpe / 全量Sharpe > 0.7 → 策略基本有效
如果 < 0.4 → 策略严重失效 (大部分alpha来自look-ahead)

用法:
  python backtest/walk_forward_v3a.py --stocks 20          # 20只快速验证
  python backtest/walk_forward_v3a.py --stocks 50          # 50只标准验证
  python backtest/walk_forward_v3a.py                       # 全量(慢)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)


def load_30min_data(codes, adapter, lookback_days=120):
    """加载30分钟数据"""
    print(f'[1] 加载30分钟数据 ({len(codes)}只)...')
    data_map = {}
    failed = 0
    for i, code in enumerate(codes):
        try:
            df = adapter.get_kline(code, period='30min')
            if df is not None and len(df) >= 200:
                data_map[code] = df
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(codes)}] 成功={len(data_map)} 失败={failed}')
    print(f'  加载完成: {len(data_map)}只成功, {failed}只失败')
    return data_map


def load_daily_data(codes, adapter, lookback=500):
    """加载日线数据(用于趋势过滤)"""
    print(f'[2] 加载日线数据 ({len(codes)}只)...')
    daily_map = {}
    for code in codes:
        try:
            df = adapter.get_kline(code, period='daily')
            if df is not None and len(df) >= 60:
                daily_map[code] = df
        except Exception:
            pass
    print(f'  日线加载完成: {len(daily_map)}只')
    return daily_map


def _check_daily_trend(daily_df, date, ma_short=20, ma_long=60):
    """检查某日的日线趋势"""
    if daily_df is None or len(daily_df) < ma_long:
        return None
    mask = daily_df.index <= pd.Timestamp(date)
    subset = daily_df[mask]
    if len(subset) < ma_long:
        return None
    close = subset['close']
    ma_s = close.rolling(ma_short).mean().iloc[-1]
    ma_l = close.rolling(ma_long).mean().iloc[-1]
    return bool(ma_s > ma_l)


def scan_signals_30min(data_30min, daily_map, date, codes_with_data,
                       precomputed_bi=None):
    """在指定日期扫描所有股票的30分钟信号 (使用截止到date的数据)

    使用czsc bi (预计算或回退到core/ stroke end) 作为买卖信号

    Args:
        precomputed_bi: {code: bi_list} 预计算的czsc bi结果

    返回: {code: {'action': 'buy'/'sell'/'hold', 'signal_type': str, 'price': float, 'stop': float}}
    """
    from core.kline import KLine
    from core.fractal import FractalDetector
    from core.stroke import StrokeGenerator
    from indicator.macd import MACD

    signals = {}
    ts_date = pd.Timestamp(date)

    for code in codes_with_data:
        df = data_30min[code]
        # 截取到当前日期
        subset = df[df.index <= ts_date + pd.Timedelta(hours=15)]
        if len(subset) < 120:
            continue

        # 日线趋势过滤
        daily_ok = _check_daily_trend(daily_map.get(code), date)
        if daily_ok is False:
            continue

        try:
            close_s = pd.Series(subset['close'].values)
            low_s = pd.Series(subset['low'].values)
            macd = MACD(close_s)
            klen = len(close_s)
            current_price = float(close_s.iloc[-1])

            # 1. 使用预计算的czsc bi (按当前subset截取)
            bi_buy = []
            bi_sell = []
            if precomputed_bi and code in precomputed_bi:
                # 从预计算结果中截取到当前日期的bi
                for bi in precomputed_bi[code]:
                    if bi['end_idx'] < klen:
                        if bi['is_down']:
                            bi_buy.append(bi['end_idx'])
                        else:
                            bi_sell.append(bi['end_idx'])
            else:
                # 回退到core/ stroke
                kline = KLine.from_dataframe(subset, strict_mode=False)
                fractals = FractalDetector(kline, confirm_required=False).get_fractals()
                if len(fractals) < 6:
                    continue
                strokes = StrokeGenerator(kline, fractals, min_bars=5).get_strokes()
                if len(strokes) < 4:
                    continue
                for s in strokes:
                    if s.end_value < s.start_value:
                        bi_buy.append(s.end_index)
                    else:
                        bi_sell.append(s.end_index)

            # 检查买入: 最近10根K线内有向下笔结束
            recent_buy = [idx for idx in bi_buy if idx >= klen - 10]

            # MACD确认
            macd_ok = False
            if recent_buy:
                latest_macd = macd.get_latest()
                if latest_macd:
                    if latest_macd.macd > latest_macd.signal:
                        macd_ok = True
                    else:
                        hist_s = macd.get_histogram_series()
                        dif_s = macd.get_dif_series()
                        if len(hist_s) >= 2 and float(hist_s.iloc[-1]) <= 0 and float(hist_s.iloc[-1]) > float(hist_s.iloc[-2]):
                            macd_ok = True
                        elif len(dif_s) >= 2 and float(dif_s.iloc[-1]) > float(dif_s.iloc[-2]):
                            macd_ok = True

            if recent_buy and macd_ok:
                lookback = min(30, recent_buy[-1])
                recent_low = float(low_s.iloc[recent_buy[-1] - lookback:recent_buy[-1] + 1].min())
                stop = max(recent_low, current_price * 0.88)
                signals[code] = {'action': 'buy', 'signal_type': 'bi_buy',
                                 'price': current_price, 'stop': float(stop)}

            # 检查卖出: 最近10根K线内有向上笔结束
            recent_sell = [idx for idx in bi_sell if idx >= klen - 10]
            if recent_sell:
                signals[code] = {'action': 'sell', 'signal_type': 'bi_sell',
                                 'price': current_price, 'stop': 0}

        except Exception:
            continue

    return signals


def _precompute_czsc_bi(data_30min):
    """预计算所有股票的czsc bi — 一次subprocess"""
    try:
        from core.czsc_bridge import get_czsc_bi_batch
        print('  预计算czsc bi (批量)...')
        bi_map = get_czsc_bi_batch(data_30min, timeout=300)
        print(f'  czsc bi完成: {len(bi_map)}/{len(data_30min)}只')
        return bi_map
    except Exception as e:
        print(f'  czsc bi预计算失败: {e}, 使用core/回退')
        return None


def full_backtest_30min(data_30min, daily_map, start_date, end_date,
                        precomputed_bi=None):
    """全量回测 — 使用完整数据一次性计算"""
    print('\n[全量回测] 一次性计算所有K线...')

    capital = 1_000_000
    cash = capital
    positions = {}
    equity_curve = [cash]
    trades = []
    max_positions = 5

    # 获取所有交易日
    sample = next(iter(data_30min.values()))
    all_dates = sample.index.sort_values().unique()
    test_start = pd.Timestamp(start_date)
    test_end = pd.Timestamp(end_date)
    test_dates = pd.DatetimeIndex([d for d in all_dates if test_start <= d <= test_end])

    if len(test_dates) == 0:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0,
                'trade_count': 0, 'win_rate': 0}

    # 按天遍历 (每天只取最后一个信号)
    seen_days = set()
    for dt in test_dates:
        day_str = str(dt.date())

        # 每天只处理一次 (30min数据一天有多根)
        if day_str in seen_days:
            # 更新权益
            holdings = sum(pos['shares'] * float(data_30min[c].loc[dt, 'close'])
                         for c, pos in positions.items()
                         if c in data_30min and dt in data_30min[c].index)
            equity_curve.append(cash + holdings)
            continue
        seen_days.add(day_str)

        # 扫描信号
        signals = scan_signals_30min(data_30min, daily_map, day_str,
                                     list(data_30min.keys()),
                                     precomputed_bi=precomputed_bi)

        # 处理卖出
        for code in list(positions.keys()):
            if code in signals and signals[code]['action'] == 'sell':
                price = signals[code]['price']
                pos = positions[code]
                pnl = (price - pos['entry_price']) / pos['entry_price']
                cash += pos['shares'] * price * (1 - 0.0003)
                trades.append({'code': code, 'pnl': pnl})
                del positions[code]

        # 止损检查
        for code in list(positions.keys()):
            if code not in data_30min:
                continue
            pos = positions[code]
            # 获取当日收盘价
            day_data = data_30min[code][data_30min[code].index.date == dt.date()]
            if len(day_data) > 0:
                price = float(day_data.iloc[-1]['close'])
                if price <= pos['stop']:
                    pnl = (price - pos['entry_price']) / pos['entry_price']
                    cash += pos['shares'] * price * (1 - 0.0003)
                    trades.append({'code': code, 'pnl': pnl, 'reason': 'stop'})
                    del positions[code]

        # 买入
        buy_candidates = [(c, s) for c, s in signals.items() if s['action'] == 'buy' and c not in positions]
        available = max_positions - len(positions)
        for code, sig in buy_candidates[:available]:
            price = sig['price']
            weight = 1.0 / max_positions
            alloc = cash * weight
            shares = int(alloc * (1 - 0.0003) / price / 100) * 100
            if shares >= 100 and alloc <= cash:
                cash -= alloc
                positions[code] = {'shares': shares, 'entry_price': price, 'stop': sig['stop']}

        # 权益
        holdings = sum(pos['shares'] * float(data_30min[c].loc[dt, 'close'])
                     for c, pos in positions.items()
                     if c in data_30min and dt in data_30min[c].index)
        equity_curve.append(cash + holdings)

    return _compute_metrics(equity_curve, trades)


def walk_forward_30min(data_30min, daily_map, start_date, end_date,
                        rebalance_days=10, precomputed_bi=None):
    """滚动窗口回测 — 每个重平衡日只用历史数据"""
    print(f'\n[滚动窗口] 重平衡间隔={rebalance_days}天')

    capital = 1_000_000
    cash = capital
    positions = {}
    equity_curve = [cash]
    trades = []
    max_positions = 5

    # 重平衡日
    sample = next(iter(data_30min.values()))
    all_dates = sample.index.sort_values().unique()
    test_start = pd.Timestamp(start_date)
    test_end = pd.Timestamp(end_date)
    test_dates = pd.DatetimeIndex([d for d in all_dates if test_start <= d <= test_end])

    if len(test_dates) == 0:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0,
                'trade_count': 0, 'win_rate': 0}

    # 按交易日重平衡
    rebalance_dates = []
    last_rebal = None
    for d in test_dates:
        day = d.date()
        if last_rebal is None or (day - last_rebal).days >= rebalance_days:
            rebalance_dates.append(d)
            last_rebal = day

    print(f'  测试期: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)}根K线)')
    print(f'  重平衡: {len(rebalance_dates)}次')

    for ri, rebal_date in enumerate(rebalance_dates):
        t0 = time.time()
        rebal_day = str(rebal_date.date())

        # 截取数据到重平衡日 (不含未来数据!)
        window_30min = {}
        for code, df in data_30min.items():
            w = df[df.index <= rebal_date + pd.Timedelta(hours=15)]
            if len(w) >= 120:
                window_30min[code] = w

        # 用截断数据扫描信号
        signals = scan_signals_30min(window_30min, daily_map, rebal_day,
                                     list(window_30min.keys()),
                                     precomputed_bi=precomputed_bi)

        buys = 0
        sells = 0

        # 执行卖出
        for code in list(positions.keys()):
            if code in signals and signals[code]['action'] == 'sell':
                price = signals[code]['price']
                pos = positions[code]
                pnl = (price - pos['entry_price']) / pos['entry_price']
                cash += pos['shares'] * price * (1 - 0.0003)
                trades.append({'code': code, 'pnl': pnl})
                del positions[code]
                sells += 1

        # 止损检查
        for code in list(positions.keys()):
            pos = positions[code]
            if code in data_30min:
                day_data = data_30min[code][
                    data_30min[code].index.date == rebal_date.date()
                ]
                if len(day_data) > 0:
                    price = float(day_data.iloc[-1]['close'])
                    if price <= pos['stop']:
                        pnl = (price - pos['entry_price']) / pos['entry_price']
                        cash += pos['shares'] * price * (1 - 0.0003)
                        trades.append({'code': code, 'pnl': pnl, 'reason': 'stop'})
                        del positions[code]
                        sells += 1

        # 执行买入
        buy_candidates = [(c, s) for c, s in signals.items()
                         if s['action'] == 'buy' and c not in positions]
        available = max_positions - len(positions)
        for code, sig in buy_candidates[:available]:
            price = sig['price']
            weight = 1.0 / max_positions
            alloc = cash * weight
            shares = int(alloc * (1 - 0.0003) / price / 100) * 100
            if shares >= 100 and alloc <= cash:
                cash -= alloc
                positions[code] = {'shares': shares, 'entry_price': price, 'stop': sig['stop']}
                buys += 1

        # 模拟持仓期间权益
        next_rebal = rebalance_dates[ri + 1] if ri + 1 < len(rebalance_dates) else test_end
        hold_dates = test_dates[(test_dates > rebal_date) & (test_dates <= next_rebal)]
        for dt in hold_dates:
            holdings = sum(
                pos['shares'] * float(data_30min[c].loc[dt, 'close'])
                for c, pos in positions.items()
                if c in data_30min and dt in data_30min[c].index
            )
            equity_curve.append(cash + holdings)

        elapsed = time.time() - t0
        print(f'  [{ri+1}/{len(rebalance_dates)}] {rebal_day} | '
              f'{len(window_30min)}只 | 买{buys} 卖{sells} | 持仓{len(positions)}只 | {elapsed:.1f}s')

    return _compute_metrics(equity_curve, trades)


def _compute_metrics(equity_curve, trades):
    """计算回测指标"""
    if len(equity_curve) < 2:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0,
                'trade_count': 0, 'win_rate': 0, 'win_count': 0, 'loss_count': 0}

    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]
    total_return = eq[-1] / eq[0] - 1
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 8)) if np.std(returns) > 0 else 0

    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)
    total = len(trades)
    win_rate = wins / total if total > 0 else 0

    avg_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if losses > 0 else 0

    return {
        'final_value': float(eq[-1]),
        'total_return': float(total_return),
        'max_drawdown': float(max_dd),
        'sharpe': float(sharpe),
        'trade_count': total,
        'win_count': wins,
        'loss_count': losses,
        'win_rate': float(win_rate),
        'avg_pnl': float(avg_pnl),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'days': len(equity_curve),
    }


def main():
    parser = argparse.ArgumentParser(description='v3a 滚动窗口验证')
    parser.add_argument('--stocks', type=int, default=20, help='股票数(0=全部)')
    parser.add_argument('--rebalance', type=int, default=10, help='重平衡间隔(天)')
    parser.add_argument('--start', default='2025-06-01')
    parser.add_argument('--end', default='2026-04-17')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"v3a 30分钟策略 — 滚动窗口验证")
    print(f"{'='*70}")

    # 加载数据
    from data.hybrid_source import HybridSource
    from chanlun_unified.stock_pool import StockPoolManager
    from chanlun_unified.config import UnifiedConfig

    config = UnifiedConfig(lookback=500)
    adapter = HybridSource(config.tdx_path)
    pool = StockPoolManager(config.tdx_path)
    codes = pool.get_pool('tdx_all')

    if args.stocks > 0:
        # 选取有足够数据的股票
        codes = codes[:args.stocks * 3]  # 多取一些，过滤后可能剩20只
    print(f'候选股票池: {len(codes)}只')

    # 加载30分钟数据
    data_30min = load_30min_data(codes, adapter)

    # 限制最终数量
    if args.stocks > 0 and len(data_30min) > args.stocks:
        items = list(data_30min.items())
        data_30min = dict(items[:args.stocks])

    if not data_30min:
        print('无数据')
        return

    # 加载日线数据
    daily_map = load_daily_data(list(data_30min.keys()), adapter)
    print(f'最终测试: {len(data_30min)}只\n')

    # 预计算czsc bi (一次subprocess处理所有股票)
    precomputed_bi = _precompute_czsc_bi(data_30min)

    # 1. 全量回测
    full_metrics = full_backtest_30min(data_30min, daily_map, args.start, args.end,
                                        precomputed_bi=precomputed_bi)
    print(f'\n  全量回测结果:')
    _print_metrics(full_metrics)

    # 2. 滚动窗口回测
    wf_metrics = walk_forward_30min(data_30min, daily_map, args.start, args.end,
                                     rebalance_days=args.rebalance,
                                     precomputed_bi=precomputed_bi)
    print(f'\n  滚动窗口结果:')
    _print_metrics(wf_metrics)

    # 3. 对比
    print(f"\n{'='*70}")
    print(f"对比结果")
    print(f"{'='*70}")
    print(f"{'指标':<15} {'全量(有bias?)':>15} {'滚动窗口(无bias)':>18} {'差异':>10}")
    print('-' * 60)

    keys = [
        ('Sharpe', 'sharpe', '.2f'),
        ('收益率', 'total_return', '.1%'),
        ('最大回撤', 'max_drawdown', '.1%'),
        ('交易笔数', 'trade_count', 'd'),
        ('胜率', 'win_rate', '.1%'),
        ('平均盈亏', 'avg_pnl', '.2%'),
    ]

    for label, key, fmt in keys:
        fv = full_metrics.get(key, 0)
        wv = wf_metrics.get(key, 0)
        if key in ('total_return', 'max_drawdown', 'win_rate', 'avg_pnl'):
            diff_str = f'{(wv-fv)*100:+.1f}pp'
        elif key == 'sharpe':
            diff_str = f'{wv-fv:+.2f}'
        else:
            diff_str = f'{wv-fv:+}'
        print(f'{label:<15} {fv:>15{fmt}} {wv:>18{fmt}} {diff_str:>10}')

    # 判定
    full_sharpe = full_metrics.get('sharpe', 0)
    wf_sharpe = wf_metrics.get('sharpe', 0)
    ratio = wf_sharpe / full_sharpe if abs(full_sharpe) > 0.01 else 0

    print(f"\n{'='*70}")
    print(f"诊断:")
    print(f"  滚动/全量 Sharpe 比值: {ratio:.2f}")

    if ratio > 0.7:
        print(f"  [OK] 策略基本有效 -- 滚动窗口保留了 {ratio*100:.0f}% 的 Sharpe")
        print(f"     look-ahead bias 影响较小，策略可作为实盘参考")
    elif ratio > 0.4:
        print(f"  [WARN] 策略部分有效 -- 滚动窗口仅保留 {ratio*100:.0f}% 的 Sharpe")
        print(f"     存在中等 look-ahead bias，需注意信号延迟确认")
    else:
        print(f"  [FAIL] 策略严重失效 -- 滚动窗口仅保留 {ratio*100:.0f}% 的 Sharpe")
        print(f"     大部分 alpha 来自 look-ahead bias，策略不可直接实盘")

    # 保存
    output = {
        'test_config': {
            'stocks': len(data_30min),
            'period': f'{args.start} ~ {args.end}',
            'rebalance_days': args.rebalance,
        },
        'full_backtest': full_metrics,
        'walk_forward': wf_metrics,
        'sharpe_ratio': ratio,
        'diagnosis': 'valid' if ratio > 0.7 else ('partial' if ratio > 0.4 else 'invalid'),
    }

    out_dir = PROJECT_ROOT / 'backtest'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'v3a_walk_forward_result.json'
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


def _print_metrics(m):
    print(f'    Sharpe={m.get("sharpe",0):.2f}  收益={m.get("total_return",0)*100:.1f}%  '
          f'回撤={m.get("max_drawdown",0)*100:.1f}%  交易={m.get("trade_count",0)}笔  '
          f'胜率={m.get("win_rate",0)*100:.0f}%  '
          f'avg_pnl={m.get("avg_pnl",0)*100:.2f}%  '
          f'avg_win={m.get("avg_win",0)*100:.2f}%  avg_loss={m.get("avg_loss",0)*100:.2f}%')


if __name__ == '__main__':
    main()
