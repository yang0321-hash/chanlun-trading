"""ML训练数据快速收集 — 使用daily_map缓存"""
import os, sys, time, pickle
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
from data.hybrid_source import HybridSource
from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from core.buy_sell_points import BuySellPointDetector
from indicator.macd import MACD
from indicator.market_environment import MarketEnvironment
from backtest.ml_signal_scorer import extract_features

OUTPUT = 'backtest/_ml_models/training_data.pkl'

def main():
    print('=== ML训练数据快速收集 (daily_map缓存) ===')
    t0 = time.time()

    # 1. 加载daily_map缓存
    hs = HybridSource()
    cache_file = '.claude/cache/daily_map_cache.pkl'
    if not os.path.exists(cache_file):
        print('缓存不存在，先加载全量数据...')
        from chanlun_unified.stock_pool import StockPoolManager
        spm = StockPoolManager()
        codes = spm.get_tdx_all_codes()
        daily_map = hs.load_all_daily(codes, min_price=2.0, max_price=2000.0, min_bars=200)
    else:
        cached = pickle.load(open(cache_file, 'rb'))
        daily_map = cached.get('data', {})
        print(f'缓存加载: {len(daily_map)}只')

    print(f'数据加载: {len(daily_map)}只 ({time.time()-t0:.1f}s)')

    # 2. 逐只提取信号
    all_signals = []
    codes = list(daily_map.keys())
    print(f'开始提取信号 ({len(codes)}只)...')

    for ri, code in enumerate(codes):
        df = daily_map[code]
        if df is None or len(df) < 200:
            continue
        try:
            cs = pd.Series(df['close'].values)
            mc = MACD(cs)
            kl = KLine.from_dataframe(df, strict_mode=False)
            closes_d = [k.close for k in kl.processed_data]
            highs_d = [k.high for k in kl.processed_data]
            lows_d = [k.low for k in kl.processed_data]

            fr = FractalDetector(kl, confirm_required=False).get_fractals()
            if len(fr) < 4:
                continue
            st = StrokeGenerator(kl, fr, min_bars=3).get_strokes()
            if len(st) < 3:
                continue
            pv = PivotDetector(kl, st).get_pivots()
            if not pv:
                continue

            # 建立 processed KLine index → DataFrame iloc 映射
            proc_dates = [k.datetime for k in kl.processed_data]
            df_iloc_map = {}
            for pi, dt in enumerate(proc_dates):
                if dt in df.index:
                    df_iloc_map[pi] = df.index.get_loc(dt)

            det = BuySellPointDetector(fr, st, [], pv, macd=mc,
                                        closes=closes_d, highs=highs_d, lows=lows_d)
            buys, _ = det.detect_all()

            seen = {}
            for b in buys:
                if b.index not in seen or b.confidence > seen[b.index].confidence:
                    seen[b.index] = b

            for b in seen.values():
                proc_idx = b.index
                if proc_idx not in df_iloc_map:
                    continue
                df_idx = df_iloc_map[proc_idx]
                if df_idx + 5 >= len(df):
                    continue
                if df_idx < 60:
                    continue

                all_signals.append({
                    'code': code,
                    'sig_idx': df_idx,
                    'signal_type': b.point_type,
                    'confidence': b.confidence,
                    'entry_price': b.price,
                    'stop_price': b.stop_loss,
                    'pivot_info': {
                        'zg': b.related_pivot.zg if b.related_pivot else 0,
                        'zd': b.related_pivot.zd if b.related_pivot else 0,
                    },
                    'weekly_trend': 'neutral',
                })
        except Exception:
            pass

        if (ri + 1) % 500 == 0:
            print(f'  [{ri+1}/{len(codes)}] signals={len(all_signals)} ({time.time()-t0:.0f}s)')

    print(f'信号提取完成: {len(all_signals)}个 ({time.time()-t0:.0f}s)')

    # 3. 提取特征
    print('提取特征...')
    market_env = MarketEnvironment()
    rows = []
    for i, sig in enumerate(all_signals):
        code = sig['code']
        sig_idx = sig['sig_idx']
        if code not in daily_map:
            continue
        df = daily_map[code]
        if sig_idx + 5 >= len(df):
            continue

        entry_price = sig.get('entry_price', df['close'].iloc[sig_idx])
        future_close = df['close'].iloc[sig_idx + 5]
        ret_5d = (future_close - entry_price) / entry_price * 100

        max_dd = 0
        for j in range(1, 6):
            if sig_idx + j < len(df):
                dd = (entry_price - df['low'].iloc[sig_idx + j]) / entry_price * 100
                max_dd = max(max_dd, dd)

        window = df.iloc[max(0, sig_idx - 100):sig_idx + 1]
        if len(window) < 30:
            continue

        features = extract_features(sig, window, sig.get('weekly_trend', 'neutral'), market_env)
        features['code'] = code
        features['date'] = str(df.index[sig_idx]) if sig_idx < len(df) else ''
        features['ret_5d'] = round(ret_5d, 2)
        features['max_dd_5d'] = round(max_dd, 2)
        features['label_triple'] = 0 if ret_5d < -3 else (2 if ret_5d > 3 else 1)
        rows.append(features)

        if (i + 1) % 10000 == 0:
            print(f'  [{i+1}/{len(all_signals)}] features={len(rows)} ({time.time()-t0:.0f}s)')

    dataset = pd.DataFrame(rows)
    print(f'\n数据集: {len(dataset)}行 x {len(dataset.columns)}列')

    labels = dataset['label_triple'].value_counts().sort_index()
    print(f'三分类: 大亏={labels.get(0,0)} 平={labels.get(1,0)} 大赢={labels.get(2,0)}')
    print(f'5日均收益: {dataset["ret_5d"].mean():+.2f}%')
    print(f'胜率: {(dataset["ret_5d"]>0).mean()*100:.1f}%')

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    dataset.to_pickle(OUTPUT)
    print(f'\n已保存至 {OUTPUT} (总耗时{time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
