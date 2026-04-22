"""v3a回测: 对比有无5分钟过滤的效果

用法: python backtest_v3a_5min_filter.py [--codes 20] [--random 50] [--seed 42]
  --codes N   : 使用内置的15只股票中取前N只
  --random N  : 从TDX全市场随机抽取N只A股
  --seed N    : 随机种子 (默认42)
"""
import os, sys, glob, random
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()

import pandas as pd
import numpy as np
import argparse

from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD

import importlib.util
_v3a_path = os.path.join('.', 'strategies', 'v3a_30min_strategy.py')
_v3a_spec = importlib.util.spec_from_file_location('v3a_30min_strategy', _v3a_path)
_v3a_mod = importlib.util.module_from_spec(_v3a_spec)
_v3a_spec.loader.exec_module(_v3a_mod)
V3a30MinStrategy = _v3a_mod.V3a30MinStrategy
V3aConfig = _v3a_mod.V3aConfig

# ============================================================================
BUILTIN_CODES = [
    'sh600869', 'sz300870', 'sz002600', 'sz000559', 'sh600519',
    'sz300750', 'sz002230', 'sh603906', 'sz300953', 'sz002692',
    'sh603985', 'sz300626', 'sh688719', 'sz301222', 'sz002580',
]


def sample_a_shares(n, seed=42):
    """从TDX数据中随机抽取A股 (排除指数/ETF/债券/北证)"""
    all_files = glob.glob('tdx_data/sh/lday/*.day') + glob.glob('tdx_data/sz/lday/*.day')
    codes = []
    for f in all_files:
        base = os.path.basename(f).replace('.day', '')
        # 只保留A股: sh600xxx/sh601xxx/sh603xxx/sh605xxx, sz000xxx/sz001xxx/sz002xxx/sz300xxx/sz301xxx
        prefix = base[:2]
        num = base[2:]
        if prefix == 'sh' and num[:1] in ('6',) and len(num) == 6:
            codes.append(base)
        elif prefix == 'sz' and num[:1] in ('0', '3') and len(num) == 6:
            codes.append(base)
    random.seed(seed)
    return random.sample(codes, min(n, len(codes)))
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.001
SLIPPAGE = 0.001
SL_MAX = 0.12
TRAIL_START = 0.05
TRAIL_DIST = 0.03
MAX_HOLD = 80
RECENT_BARS = 30


def get_chanlun_bi(df, min_bars=3):
    """从30min DataFrame获取bi买点索引"""
    kline = KLine.from_dataframe(df, strict_mode=False)
    fractals = FractalDetector(kline, confirm_required=False).get_fractals()
    if len(fractals) < 6:
        return []
    strokes = StrokeGenerator(kline, fractals, min_bars=min_bars).get_strokes()
    return [s.end_index for s in strokes if s.end_value < s.start_value]


def check_macd_strict(macd, kline_idx):
    """MACD确认 (收紧版): 至少满足2项, 或仅绿柱缩短"""
    v = macd.get_value_at(kline_idx)
    if v is None:
        return False
    v_prev = macd.get_value_at(kline_idx - 1)

    score = 0
    has_green_shrink = False

    # 条件1: DIF > DEA
    if v.macd > v.signal:
        score += 1

    # 条件2: 绿柱缩短
    if v.histogram <= 0 and v_prev is not None and v.histogram > v_prev.histogram:
        score += 1
        has_green_shrink = True

    # 条件3: DIF递增
    if v_prev is not None and v.macd > v_prev.macd:
        score += 1

    return score >= 2 or (score == 1 and has_green_shrink)


def daily_bullish_at(df_daily, target_date):
    """检查日线MA20>MA60"""
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


def run_backtest(hs, codes, enable_5min=True):
    capital = INITIAL_CAPITAL
    trades = []
    wins = 0
    losses = 0
    filtered_by_5min = 0

    for ci, code in enumerate(codes):
        try:
            print(f"  [{ci+1}/{len(codes)}] {code}...", end="", flush=True)
            df = hs.get_kline(code, period='30min')
            if df is None or len(df) < 200:
                print(" skip (30min too short)", flush=True)
                continue
            df_daily = hs.get_kline(code, period='daily')

            # 获取bi和中枢
            kline = KLine.from_dataframe(df, strict_mode=False)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 6:
                continue
            strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
            if len(strokes) < 4:
                continue
            bi_buy = [s.end_index for s in strokes if s.end_value < s.start_value]
            bi_buy_low = {s.end_index: min(s.start_value, s.end_value)
                          for s in strokes if s.end_value < s.start_value}
            bi_sell_up = [s for s in strokes if s.end_value > s.start_value]
            if not bi_buy:
                continue

            pivots = []
            try:
                from core.pivot import PivotDetector
                pivots = PivotDetector(kline, strokes).get_pivots()
            except:
                pass

            # 最近中枢ZD
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

            # 向上笔索引映射 (用于结构追踪止损)
            up_bi_map = {}  # {end_index: start_value}
            for s in strokes:
                if s.end_value > s.start_value:
                    up_bi_map[s.end_index] = s.start_value

            # 向上笔MACD面积 (用于背驰止盈)
            up_bi_areas = {}
            for s in strokes:
                if s.end_value > s.start_value:
                    area = macd.compute_area(s.start_index, s.end_index, 'up')
                    if area > 0:
                        up_bi_areas[s.end_index] = area

            v3a = None
            if enable_5min:
                v3a = V3a30MinStrategy(V3aConfig(mode='trend', enable_5min_filter=True), hs)

            position = None
            last_f5_check_i = -100  # 避免连续bar重复检测

            for i in range(120, klen):
                bar_close = float(close_s.iloc[i])
                bar_low = float(low_s.iloc[i])
                bar_high = float(high_s.iloc[i])

                # 持仓管理
                if position is not None:
                    held = i - position['entry_idx']
                    position['highest'] = max(position['highest'], bar_high)

                    # 结构追踪止损: 向上笔完成时, 止损移到该笔起点
                    last_trail = position.get('last_trail_bi', -1)
                    for bi_end, bi_start_val in up_bi_map.items():
                        if bi_end > last_trail and bi_end <= i:
                            struct_stop = bi_start_val
                            position['stop'] = max(position['stop'], struct_stop)
                            position['last_trail_bi'] = bi_end

                    # 固定止损
                    if bar_low <= position['stop']:
                        sell_price = max(position['stop'], bar_low)
                        pnl = (sell_price - position['entry_price']) / position['entry_price']
                        capital += position['shares'] * sell_price * (1 - COMMISSION)
                        trades.append(pnl)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        position = None
                        continue

                    # MACD面积背驰止盈 (减仓50%)
                    if position['shares'] > 0 and held >= 6:
                        profit = (bar_close - position['entry_price']) / position['entry_price']
                        if profit > 0.03:
                            # 找持仓期间完成的向上笔
                            held_bis = sorted([e for e in up_bi_areas
                                               if position['entry_idx'] < e <= i])
                            if len(held_bis) >= 2:
                                last_area = up_bi_areas[held_bis[-1]]
                                prev_area = up_bi_areas[held_bis[-2]]
                                if prev_area > 0 and last_area < prev_area * 0.5:
                                    # 动能衰竭, 减仓50%
                                    half = position['shares'] // 2
                                    if half >= 100:
                                        sell_price = bar_close * (1 - SLIPPAGE)
                                        pnl = (sell_price - position['entry_price']) / position['entry_price']
                                        capital += half * sell_price * (1 - COMMISSION)
                                        trades.append(pnl)
                                        if pnl > 0: wins += 1
                                        else: losses += 1
                                        position['shares'] -= half

                    # 时间止损 (放宽到120根)
                    if held >= 120:
                        sell_price = bar_close * (1 - SLIPPAGE)
                        pnl = (sell_price - position['entry_price']) / position['entry_price']
                        capital += position['shares'] * sell_price * (1 - COMMISSION)
                        trades.append(pnl)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        position = None
                        continue
                    continue

                # 入场检查
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

                # 2买结构确认: 当前笔低点 > 前一个笔低点 (higher low)
                prev_buy_idx = None
                for bi in reversed(bi_buy):
                    if bi < buy_idx:
                        prev_buy_idx = bi
                        break
                if prev_buy_idx is not None:
                    if bi_buy_low.get(buy_idx, 0) <= bi_buy_low.get(prev_buy_idx, 0):
                        continue  # 创新低, 不是2买

                # MACD确认 (收紧版)
                if not check_macd_strict(macd, i):
                    continue

                # 5分钟过滤 (每10根bar检测一次, 避免太慢)
                if enable_5min and v3a is not None and (i - last_f5_check_i) >= 10:
                    try:
                        bar_time = df.index[i]
                        f5 = v3a._check_5min_filter(code, cutoff_time=bar_time)
                        last_f5_check_i = i
                        if f5 is not None and f5['has_sell']:
                            filtered_by_5min += 1
                            continue
                    except:
                        pass

                # 回调缩量过滤
                if vol_s is not None and buy_idx >= 5:
                    pullback_vol = float(vol_s.iloc[max(0, buy_idx - 10):buy_idx + 1].mean())
                    pre_vol_start = max(0, buy_idx - 30)
                    pre_vol_end = max(0, buy_idx - 10)
                    if pre_vol_end > pre_vol_start:
                        pre_vol = float(vol_s.iloc[pre_vol_start:pre_vol_end].mean())
                        if pre_vol > 0 and pullback_vol > pre_vol * 0.85:
                            continue  # 回调未缩量

                # 结构支撑过滤
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

                # 止损
                lookback = min(30, buy_idx)
                recent_low = float(low_s.iloc[max(0, buy_idx - lookback):buy_idx + 1].min())
                stop_loss = max(recent_low, bar_close * (1 - SL_MAX))

                # 入场
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

        except Exception as e:
            print(f" error: {e}", flush=True)
            continue

        stock_trades = len([t for t in trades])  # rough count
        print(f" done", flush=True)

    total = sum(trades) if trades else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg = total / len(trades) * 100 if trades else 0

    # 计算Sharpe
    if len(trades) > 1:
        arr = np.array(trades)
        sharpe = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total * 100,
        'avg_pnl': avg,
        'sharpe': sharpe,
        'filtered_5min': filtered_by_5min,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', type=int, default=15, help='Number of builtin stocks')
    parser.add_argument('--random', type=int, default=0, help='Random sample N stocks from TDX')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    hs = HybridSource()

    if args.random > 0:
        codes = sample_a_shares(args.random, args.seed)
        print(f"Random sampled {len(codes)} A-shares (seed={args.seed})")
    else:
        codes = BUILTIN_CODES[:args.codes]

    print("=" * 60, flush=True)
    print("v3a 5min filter backtest", flush=True)
    print(f"Stocks: {len(codes)}", flush=True)
    print("=" * 60, flush=True)

    print("\n[1] No 5min filter", flush=True)
    r1 = run_backtest(hs, codes, enable_5min=False)
    print(f"  Trades: {r1['trades']}", flush=True)
    print(f"  Win rate: {r1['win_rate']:.1f}% ({r1['wins']}W/{r1['losses']}L)", flush=True)
    print(f"  Total PnL: {r1['total_pnl']:.2f}%", flush=True)
    print(f"  Avg PnL: {r1['avg_pnl']:.2f}%", flush=True)
    print(f"  Sharpe: {r1['sharpe']:.2f}", flush=True)

    print("\n[2] With 5min filter", flush=True)
    r2 = run_backtest(hs, codes, enable_5min=True)
    print(f"  Trades: {r2['trades']} (filtered {r2['filtered_5min']} by 5min)")
    print(f"  Win rate: {r2['win_rate']:.1f}% ({r2['wins']}W/{r2['losses']}L)")
    print(f"  Total PnL: {r2['total_pnl']:.2f}%")
    print(f"  Avg PnL: {r2['avg_pnl']:.2f}%")
    print(f"  Sharpe: {r2['sharpe']:.2f}")

    print("\n" + "=" * 60)
    if r1['trades'] > 0 and r2['trades'] > 0:
        print(f"Trade reduction: {r1['trades'] - r2['trades']} ({(1 - r2['trades']/r1['trades'])*100:.0f}%)")
        print(f"Win rate change: {r1['win_rate']:.1f}% -> {r2['win_rate']:.1f}% ({r2['win_rate']-r1['win_rate']:+.1f}%)")
        print(f"Avg PnL change: {r1['avg_pnl']:.2f}% -> {r2['avg_pnl']:.2f}% ({r2['avg_pnl']-r1['avg_pnl']:+.2f}%)")
        print(f"Sharpe change: {r1['sharpe']:.2f} -> {r2['sharpe']:.2f} ({r2['sharpe']-r1['sharpe']:+.2f})")
