#!/usr/bin/env python3
"""
30分钟T+0策略 - 对比回测：MACD背驰 vs 分型背驰

版本A: 现有逻辑 - MACD柱缩短触发背驰止盈
版本B: 新逻辑 - 30分钟顶分型(bi_sell) + MACD柱缩短触发背驰止盈
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "code"))

CODES = ["000426.SZ", "002600.SZ", "000559.SZ", "000629.SZ", "301128.SZ", "688613.SH"]
DATA_DIR = Path("/workspace/chanlun_system/artifacts")
START_DATE = "2024-06-01"
END_DATE = "2026-04-10"
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.0003
SLIPPAGE = 0.001


def load_data():
    data_map = {}
    for code in CODES:
        csv_path = DATA_DIR / f"min30_{code}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        df = df.rename(columns={'vol': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna().sort_index()
        mask = (df.index >= START_DATE) & (df.index <= END_DATE)
        df = df[mask]
        if len(df) >= 120:
            data_map[code] = df
    return data_map


def compute_bi(df):
    n = len(df)
    buy_s = pd.Series(False, index=df.index)
    sell_s = pd.Series(False, index=df.index)
    try:
        from czsc import CZSC, RawBar, Freq
        bars = []
        for i in range(n):
            vol = float(df['volume'].iloc[i]) if 'volume' in df.columns else 0
            amt = float(df['close'].iloc[i]) * vol if vol > 0 else 0
            bars.append(RawBar(
                symbol='A', id=i, dt=pd.Timestamp(df.index[i]), freq=Freq.F30,
                open=float(df['open'].iloc[i]), close=float(df['close'].iloc[i]),
                high=float(df['high'].iloc[i]), low=float(df['low'].iloc[i]),
                vol=vol, amount=amt,
            ))
        c = CZSC(bars)
        for bi in c.bi_list:
            if not bi.raw_bars:
                continue
            end_idx = bi.raw_bars[-1].id
            if end_idx is None or end_idx >= n:
                continue
            d = str(bi.direction)
            if '下' in d:
                buy_s.iloc[end_idx] = True
            elif '上' in d:
                sell_s.iloc[end_idx] = True
    except ImportError:
        h, l = df['high'].values, df['low'].values
        for i in range(1, n - 1):
            if l[i] < l[i-1] and l[i] < l[i+1]:
                buy_s.iloc[i] = True
            if h[i] > h[i-1] and h[i] > h[i+1]:
                sell_s.iloc[i] = True
    return buy_s, sell_s


def generate_signals(df, bi_buy, bi_sell, use_fractal=False):
    """生成信号, use_fractal=True时背驰需顶分型确认"""
    n = len(df)
    signals = pd.Series(0.0, index=df.index)
    if n < 120:
        return signals, []

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    hist = 2 * (dif - dea)

    position = 0.0
    entry_idx = -1
    entry_price = 0.0
    stop_loss = 0.0
    highest = 0.0
    last_sell_idx = -999
    has_diverged = False
    div_triggers = []

    for i in range(120, n):
        price = close[i]

        if position > 0:
            bars_held = i - entry_idx
            pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0

            if price > highest:
                highest = price

            # 止损
            if price <= stop_loss:
                signals.iloc[i] = 0.0
                position = 0.0
                last_sell_idx = i
                has_diverged = False
                continue

            # 背驰止盈半仓
            if (not has_diverged and pnl_pct >= 0.03 and bars_held >= 6):
                recent_high = np.max(high[max(0, i-5):i+1])
                macd_shrink = (i >= 2 and hist[i] > 0 and hist[i] < hist[i-1] and hist[i-1] < hist[i-2])

                if recent_high >= highest * 0.995 and macd_shrink:
                    if use_fractal:
                        # B: 顶分型 + MACD缩短
                        if bi_sell.iloc[i]:
                            signals.iloc[i] = position * 0.5
                            has_diverged = True
                            div_triggers.append({'idx': i, 'date': str(df.index[i]),
                                                 'type': 'fractal+macd', 'price': price,
                                                 'hist': round(float(hist[i]), 4)})
                            continue
                    else:
                        # A: 纯MACD缩短
                        signals.iloc[i] = position * 0.5
                        has_diverged = True
                        div_triggers.append({'idx': i, 'date': str(df.index[i]),
                                             'type': 'macd_only', 'price': price,
                                             'hist': round(float(hist[i]), 4)})
                        continue

            # 动态止盈
            if pnl_pct > 0.05:
                if price <= highest * 0.97:
                    signals.iloc[i] = 0.0
                    position = 0.0
                    last_sell_idx = i
                    has_diverged = False
                    continue

            # 时间止损
            if bars_held >= 80:
                signals.iloc[i] = 0.0
                position = 0.0
                last_sell_idx = i
                has_diverged = False
                continue

            signals.iloc[i] = position

        else:
            if i - last_sell_idx < 3:
                continue
            if not bi_buy.iloc[i]:
                continue
            macd_ok = dif[i] > dea[i] or (hist[i] > hist[i-1] and hist[i] <= 0) or (dif[i] > dif[i-1])
            if not macd_ok:
                continue

            lookback = min(30, i - 1)
            recent_low = np.min(low[i-lookback:i])
            sd = price - recent_low
            if sd <= 0:
                continue
            sp = sd / price
            stop_loss = recent_low if sp <= 0.12 else price * 0.88

            signals.iloc[i] = 0.5
            position = 0.5
            entry_idx = i
            entry_price = price
            highest = price
            has_diverged = False

    return signals, div_triggers


def backtest(data_map, signals_map):
    all_dates = sorted(set().union(*[set(df.index) for df in data_map.values()]))
    cash = INITIAL_CAPITAL
    positions = {}
    eq_curve = []
    trades = []

    for dt in all_dates:
        eq = cash
        prices = {}
        for code, df in data_map.items():
            if dt not in df.index:
                continue
            p = float(df.loc[dt, 'close'])
            prices[code] = p
            if code in positions:
                eq += positions[code]['shares'] * p
        eq_curve.append({'date': dt, 'equity': eq})

        for code, sig in signals_map.items():
            if code not in prices or dt not in sig.index:
                continue
            tp = float(sig.loc[dt])
            price = prices[code] * (1 + SLIPPAGE)

            if tp > 0:
                tv = eq * tp
                cv = positions[code]['shares'] * price if code in positions else 0
                nv = tv - cv
                if nv > 0 and cash >= nv:
                    shares = int(nv / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * (1 + COMMISSION)
                        if cost <= cash:
                            cash -= cost
                            if code in positions:
                                o = positions[code]
                                positions[code] = {'shares': o['shares']+shares,
                                                   'entry_price': (o['cost']+cost)/(o['shares']+shares),
                                                   'cost': o['cost']+cost}
                            else:
                                positions[code] = {'shares': shares, 'entry_price': price, 'cost': cost}
                            trades.append({'date': dt, 'code': code, 'action': 'buy',
                                           'price': price, 'shares': shares})

            elif tp == 0 and code in positions:
                pos = positions[code]
                sp = price * (1 - SLIPPAGE)
                rev = pos['shares'] * sp * (1 - COMMISSION)
                cash += rev
                pnl = rev - pos['cost']
                trades.append({'date': dt, 'code': code, 'action': 'sell',
                               'price': sp, 'shares': pos['shares'],
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl/pos['cost']*100, 2)})
                del positions[code]

    return eq_curve, trades


def stats(eq_curve, trades, label):
    eq = pd.DataFrame(eq_curve).set_index('date')
    final = eq['equity'].iloc[-1]
    ret = (final / INITIAL_CAPITAL - 1) * 100
    eq_d = eq.resample('D').last().dropna()
    dr = eq_d['equity'].pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0
    peak = eq['equity'].cummax()
    mdd = float(((peak - eq['equity']) / peak).max() * 100)
    sells = [t for t in trades if t['action'] == 'sell']
    wins = [t for t in sells if t['pnl'] > 0]
    losses = [t for t in sells if t['pnl'] <= 0]
    wr = len(wins)/len(sells)*100 if sells else 0
    aw = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    al = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    plr = abs(aw/al) if al != 0 else float('inf')
    return {'版本': label, '收益率': f'{ret:.1f}%', 'Sharpe': f'{sharpe:.2f}',
            '回撤': f'{mdd:.1f}%', '交易': len(sells), '胜率': f'{wr:.1f}%',
            '盈亏比': f'{plr:.2f}' if plr != float('inf') else '∞',
            '均盈': f'{aw:.2f}%', '均亏': f'{al:.2f}%'}


def main():
    print(f"\n{'='*70}")
    print("  背驰检测对比回测")
    print("  A: MACD柱缩短          B: 顶分型 + MACD柱缩短")
    print(f"{'='*70}\n")

    data_map = load_data()
    print(f"[数据] {len(data_map)}只\n")

    results = {}
    div_logs = {}

    for label, fractal in [("A-MACD背驰", False), ("B-分型背驰", True)]:
        sig_map = {}
        all_div = []
        t0 = time.time()

        for code, df in data_map.items():
            bi_buy, bi_sell = compute_bi(df)
            sig, divs = generate_signals(df, bi_buy, bi_sell, use_fractal=fractal)
            sig_map[code] = sig
            for d in divs:
                d['code'] = code
                all_div.append(d)

        div_logs[label] = all_div
        eq, trades = backtest(data_map, sig_map)
        results[label] = stats(eq, trades, label)
        results[label]['背驰次数'] = len(all_div)
        print(f"  {label} 完成 ({time.time()-t0:.1f}s)")

    # 对比表
    print(f"\n{'='*70}")
    print(f"{'指标':<12} {'A-MACD背驰':>14} {'B-分型背驰':>14} {'差异':>12}")
    print(f"{'-'*55}")

    for k in ['背驰次数', '交易', '胜率', '盈亏比', '均盈', '均亏', 'Sharpe', '回撤', '收益率']:
        a = str(results['A-MACD背驰'].get(k, '-'))
        b = str(results['B-分型背驰'].get(k, '-'))
        # 数字差异
        try:
            av = float(a.replace('%', ''))
            bv = float(b.replace('%', ''))
            diff = f"{bv - av:+.1f}" if '%' in a else f"{bv - av:+.2f}"
        except:
            diff = ''
        print(f"  {k:<10} {a:>14} {b:>14} {diff:>12}")

    # 背驰触发详情
    print(f"\n{'='*70}")
    for label in ['A-MACD背驰', 'B-分型背驰']:
        divs = div_logs[label]
        print(f"\n{label}: {len(divs)}次触发")
        for d in divs[:10]:
            print(f"  {d['date']}  {d['code']:<12} {d['price']:.2f}  hist={d.get('hist','?')}")
        if len(divs) > 10:
            print(f"  ... 还有{len(divs)-10}次")

    # 背驰后收益统计
    print(f"\n{'='*70}")
    print("背驰触发后, 剩余半仓的最终出场收益:")
    print(f"{'-'*55}")

    for label, fractal in [("A-MACD背驰", False), ("B-分型背驰", True)]:
        sig_map = {}
        for code, df in data_map.items():
            bi_buy, bi_sell = compute_bi(df)
            sig, _ = generate_signals(df, bi_buy, bi_sell, use_fractal=fractal)
            sig_map[code] = sig

        # 追踪背驰后的半仓出场
        eq, trades = backtest(data_map, sig_map)
        sells = [t for t in trades if t['action'] == 'sell']
        # 找背驰后的卖出（通过信号从半仓→0的transition）
        # 简化：统计所有卖出交易
        pos_trades = [t for t in sells if t['pnl'] > 0]
        neg_trades = [t for t in sells if t['pnl'] <= 0]
        print(f"\n  {label}:")
        print(f"    盈利卖出: {len(pos_trades)}笔, 均盈{np.mean([t['pnl_pct'] for t in pos_trades]):.2f}%" if pos_trades else f"    盈利卖出: 0笔")
        print(f"    亏损卖出: {len(neg_trades)}笔, 均亏{np.mean([t['pnl_pct'] for t in neg_trades]):.2f}%" if neg_trades else f"    亏损卖出: 0笔")

    print(f"\n{'='*70}")

    # 保存
    out_path = Path(__file__).parent / "compare_result.json"
    out_path.write_text(json.dumps({
        'results': results,
        'divergences': {k: v for k, v in div_logs.items()},
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


if __name__ == '__main__':
    main()
