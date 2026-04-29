"""min_bars网格搜索 — 找出30min/daily最优笔参数

用法: python backtest_min_bars_grid.py [--random 50] [--seed 42]
"""
import os, sys
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()

import pandas as pd
import numpy as np
import argparse, glob, random, time

from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD

BUILTIN_CODES = [
    'sh600869', 'sz300870', 'sz002600', 'sz000559', 'sh600519',
    'sz300750', 'sz002230', 'sh603906', 'sz300953', 'sz002692',
    'sh603985', 'sz300626', 'sh688719', 'sz301222', 'sz002580',
]

INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.001
SLIPPAGE = 0.001
SL_MAX = 0.12
MAX_HOLD = 120
RECENT_BARS = 30


def sample_a_shares(n, seed=42):
    all_files = glob.glob('tdx_data/sh/lday/*.day') + glob.glob('tdx_data/sz/lday/*.day')
    codes = []
    for f in all_files:
        base = os.path.basename(f).replace('.day', '')
        prefix = base[:2]
        num = base[2:]
        if prefix == 'sh' and num[:1] in ('6',) and len(num) == 6:
            codes.append(base)
        elif prefix == 'sz' and num[:1] in ('0', '3') and len(num) == 6:
            codes.append(base)
    random.seed(seed)
    return random.sample(codes, min(n, len(codes)))


def check_macd_strict(macd, kline_idx):
    v = macd.get_value_at(kline_idx)
    if v is None:
        return False
    v_prev = macd.get_value_at(kline_idx - 1)
    score = 0
    has_green_shrink = False
    if v.macd > v.signal:
        score += 1
    if v.histogram <= 0 and v_prev is not None and v.histogram > v_prev.histogram:
        score += 1
        has_green_shrink = True
    if v_prev is not None and v.macd > v_prev.macd:
        score += 1
    return score >= 2 or (score == 1 and has_green_shrink)


def daily_bullish_at(df_daily, target_date):
    if df_daily is None or len(df_daily) < 60:
        return True
    mask = df_daily.index <= target_date
    sub = df_daily[mask].tail(65)
    if len(sub) < 60:
        return True
    c = sub['close']
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    if pd.notna(ma20) and pd.notna(ma60):
        return ma20 > ma60
    return True


def run_single(hs, codes, min_bars):
    """单个min_bars值的回测"""
    capital = INITIAL_CAPITAL
    trades = []
    wins = 0
    losses = 0
    bi_count_total = 0
    stock_count = 0

    for code in codes:
        try:
            df = hs.get_kline(code, period='30min')
            if df is None or len(df) < 200:
                continue
            df_daily = hs.get_kline(code, period='daily')

            kline = KLine.from_dataframe(df, strict_mode=False)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 6:
                continue
            strokes = StrokeGenerator(kline, fractals, min_bars=min_bars).get_strokes()
            if len(strokes) < 4:
                continue
            bi_count_total += len(strokes)
            stock_count += 1

            bi_buy = [s.end_index for s in strokes if s.end_value < s.start_value]
            bi_buy_low = {s.end_index: min(s.start_value, s.end_value)
                          for s in strokes if s.end_value < s.start_value}
            bi_sell_up = [s for s in strokes if s.end_value > s.start_value]
            if not bi_buy:
                continue

            up_bi_map = {}
            for s in strokes:
                if s.end_value > s.start_value:
                    up_bi_map[s.end_index] = s.start_value
            up_bi_areas = {}
            for s in strokes:
                if s.end_value > s.start_value:
                    area = MACD(pd.Series(df['close'].values)).compute_area(s.start_index, s.end_index, 'up')
                    if area > 0:
                        up_bi_areas[s.end_index] = area

            pivots = []
            try:
                pivots = PivotDetector(kline, strokes).get_pivots()
            except:
                pass
            pivot_zd = 0.0
            for p in reversed(pivots):
                if p.zg > 0:
                    pivot_zd = p.zd
                    break

            close_s = pd.Series(df['close'].values)
            low_s = pd.Series(df['low'].values)
            high_s = pd.Series(df['high'].values)
            vol_s = pd.Series(df['volume'].values) if 'volume' in df.columns else None
            macd = MACD(close_s)
            klen = len(close_s)

            position = None
            for i in range(120, klen):
                bar_close = float(close_s.iloc[i])
                bar_low = float(low_s.iloc[i])
                bar_high = float(high_s.iloc[i])

                if position is not None:
                    held = i - position['entry_idx']
                    position['highest'] = max(position['highest'], bar_high)

                    last_trail = position.get('last_trail_bi', -1)
                    for bi_end, bi_start_val in up_bi_map.items():
                        if bi_end > last_trail and bi_end <= i:
                            position['stop'] = max(position['stop'], bi_start_val)
                            position['last_trail_bi'] = bi_end

                    if bar_low <= position['stop']:
                        sell_price = max(position['stop'], bar_low)
                        pnl = (sell_price - position['entry_price']) / position['entry_price']
                        capital += position['shares'] * sell_price * (1 - COMMISSION)
                        trades.append(pnl)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        position = None
                        continue

                    if position['shares'] > 0 and held >= 6:
                        profit = (bar_close - position['entry_price']) / position['entry_price']
                        if profit > 0.03:
                            held_bis = sorted([e for e in up_bi_areas
                                               if position['entry_idx'] < e <= i])
                            if len(held_bis) >= 2:
                                last_area = up_bi_areas[held_bis[-1]]
                                prev_area = up_bi_areas[held_bis[-2]]
                                if prev_area > 0 and last_area < prev_area * 0.5:
                                    half = position['shares'] // 2
                                    if half >= 100:
                                        sell_price = bar_close * (1 - SLIPPAGE)
                                        pnl = (sell_price - position['entry_price']) / position['entry_price']
                                        capital += half * sell_price * (1 - COMMISSION)
                                        trades.append(pnl)
                                        if pnl > 0: wins += 1
                                        else: losses += 1
                                        position['shares'] -= half

                    if held >= MAX_HOLD:
                        sell_price = bar_close * (1 - SLIPPAGE)
                        pnl = (sell_price - position['entry_price']) / position['entry_price']
                        capital += position['shares'] * sell_price * (1 - COMMISSION)
                        trades.append(pnl)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        position = None
                        continue
                    continue

                bar_date = df.index[i]
                if not daily_bullish_at(df_daily, bar_date):
                    continue

                buy_idx = None
                for bi in reversed(bi_buy):
                    if i - RECENT_BARS <= bi <= i:
                        buy_idx = bi
                        break
                if buy_idx is None:
                    continue

                prev_buy_idx = None
                for bi in reversed(bi_buy):
                    if bi < buy_idx:
                        prev_buy_idx = bi
                        break
                if prev_buy_idx is not None:
                    if bi_buy_low.get(buy_idx, 0) <= bi_buy_low.get(prev_buy_idx, 0):
                        continue

                if not check_macd_strict(macd, i):
                    continue

                if vol_s is not None and buy_idx >= 5:
                    pullback_vol = float(vol_s.iloc[max(0, buy_idx - 10):buy_idx + 1].mean())
                    pre_vol_start = max(0, buy_idx - 30)
                    pre_vol_end = max(0, buy_idx - 10)
                    if pre_vol_end > pre_vol_start:
                        pre_vol = float(vol_s.iloc[pre_vol_start:pre_vol_end].mean())
                        if pre_vol > 0 and pullback_vol > pre_vol * 0.85:
                            continue

                near_support = False
                if pivot_zd > 0 and bar_close <= pivot_zd * 1.03:
                    near_support = True
                if not near_support:
                    for s in reversed(bi_sell_up):
                        if s.end_index <= buy_idx:
                            if bar_close <= s.start_value * 1.03:
                                near_support = True
                            break
                if not near_support:
                    continue

                lookback = min(30, buy_idx)
                recent_low = float(low_s.iloc[max(0, buy_idx - lookback):buy_idx + 1].min())
                stop_loss = max(recent_low, bar_close * (1 - SL_MAX))

                entry_price = bar_close * (1 + SLIPPAGE)
                per_share_risk = entry_price - stop_loss
                if per_share_risk <= 0:
                    continue
                shares = int(capital * 0.02 / per_share_risk / 100) * 100
                if shares <= 0:
                    continue
                max_val = capital * 0.15
                if shares * entry_price > max_val:
                    shares = int(max_val / entry_price / 100) * 100
                if shares <= 0:
                    continue

                capital -= shares * entry_price * (1 + COMMISSION)
                position = {
                    'entry_price': entry_price,
                    'entry_idx': i,
                    'highest': bar_high,
                    'stop': stop_loss,
                    'shares': shares,
                }
        except Exception:
            continue

    total = sum(trades) if trades else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg = total / len(trades) * 100 if trades else 0
    sharpe = 0
    if len(trades) > 1:
        arr = np.array(trades)
        sharpe = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    avg_bi = bi_count_total / stock_count if stock_count > 0 else 0

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total * 100,
        'avg_pnl': avg,
        'sharpe': sharpe,
        'avg_bi_per_stock': avg_bi,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', type=int, default=50, help='Random sample N stocks')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    hs = HybridSource()
    codes = sample_a_shares(args.random, args.seed)
    print(f"min_bars grid search on {len(codes)} stocks (seed={args.seed})")
    print("=" * 70)

    # 网格: min_bars 3~9
    results = []
    for mb in range(3, 10):
        t0 = time.time()
        r = run_single(hs, codes, min_bars=mb)
        elapsed = time.time() - t0
        results.append((mb, r))
        print(f"  min_bars={mb}: {r['trades']:>3}笔 WR{r['win_rate']:>5.1f}% "
              f"avg{r['avg_pnl']:>+6.2f}% Sharpe{r['sharpe']:>6.2f} "
              f"笔/股{r['avg_bi_per_stock']:>5.1f} ({elapsed:.0f}s)")

    print("\n" + "=" * 70)
    print(f"{'min_bars':>8} {'笔数':>5} {'胜率%':>7} {'均盈亏%':>8} {'Sharpe':>7} {'笔/股':>6}")
    print("-" * 50)
    for mb, r in results:
        print(f"{mb:>8} {r['trades']:>5} {r['win_rate']:>7.1f} {r['avg_pnl']:>+8.2f} "
              f"{r['sharpe']:>7.2f} {r['avg_bi_per_stock']:>6.1f}")

    # 找最优
    best = max(results, key=lambda x: x[1]['sharpe'])
    print(f"\n最优: min_bars={best[0]} Sharpe={best[1]['sharpe']:.2f} "
          f"WR={best[1]['win_rate']:.1f}% avg={best[1]['avg_pnl']:+.2f}%")
