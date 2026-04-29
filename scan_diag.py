#!/usr/bin/env python3
"""
诊断扫描: 看engine.generate()在全量数据上的信号分布
不经过动态池过滤，看实际有多少股票有信号
"""
import os, sys, json, time, struct
from pathlib import Path
from datetime import datetime, timedelta

for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

PROJECT_ROOT = Path('/workspace/chanlun_system')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from signal_engine import SignalEngine

TDX_ROOT = Path('/workspace/tdx_data')

def get_all_codes():
    codes = []
    for mdir in ['sh', 'sz']:
        mpath = TDX_ROOT / mdir / 'lday'
        if not mpath.exists(): continue
        for f in mpath.iterdir():
            if f.suffix == '.day':
                codes.append(f.stem)
    return codes

def read_tdx(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    records = []
    for i in range(n):
        r = struct.unpack('<IfffffII', data[i*32:(i+1)*32])
        records.append(r)
    df = pd.DataFrame(records, columns=['date','open','high','low','close','amount','vol','reserved'])
    first_open = df['open'].iloc[0]
    if not (50 < first_open < 10000000):
        for col in ['open','high','low','close']:
            df[col] = df[col] / 100.0
    df = df.rename(columns={'vol': 'volume'})
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date').sort_index()
    return df

def process_batch(batch_codes, engine):
    """批量处理，返回 (code, last_signal, last_date, n_signals, total_signal_sum)"""
    results = []
    for code in batch_codes:
        try:
            mdir = 'sh' if code.startswith('sh') else 'sz'
            mcode = code[2:]
            full_code = f'{mcode}.SH' if mdir == 'sh' else f'{mcode}.SZ'
            df = read_tdx(TDX_ROOT / mdir / 'lday' / f'{code}.day')
            if df.empty or len(df) < 120:
                continue
            last_date = df.index[-1]
            if last_date < pd.Timestamp('2026-04-20'):
                continue
            data_map = {full_code: df}
            sigs = engine.generate(data_map, live_mode=True)
            if full_code not in sigs:
                continue
            s = sigs[full_code]
            if len(s) == 0:
                continue
            last_val = s.iloc[-1]
            nonzero = s[s > 0]
            results.append({
                'code': full_code,
                'tdx_code': code,
                'last_date': last_date,
                'last_signal': last_val,
                'n_nonzero': len(nonzero),
                'max_signal': s.max() if len(s) > 0 else 0,
                'recent_signals': list(s.iloc[-5:]),
                'close': df['close'].iloc[-1],
                'industry': '',
            })
        except:
            pass
    return results

def main():
    t0 = time.time()
    codes = get_all_codes()
    print(f"TDX股票: {len(codes)}只")

    # engine初始化
    engine = SignalEngine()

    # 分批处理
    BATCH = 200
    all_results = []
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i+BATCH]
        results = process_batch(batch, engine)
        all_results.extend(results)
        if (i // BATCH) % 5 == 0:
            print(f"  {i}/{len(codes)} ({len(all_results)}个有数据)")
    print(f"\n处理完成: {len(all_results)}/{len(codes)}只有数据, {time.time()-t0:.0f}秒")

    if not all_results:
        print("无数据")
        return

    # 信号分布统计
    nonzero = [r for r in all_results if r['last_signal'] > 0]
    print(f"\n有信号股票: {len(nonzero)}只 / {len(all_results)}只")
    print(f"有信号股票信号值分布:")
    vals = [r['last_signal'] for r in nonzero]
    print(f"  min={min(vals):.4f} max={max(vals):.4f} mean={np.mean(vals):.4f} med={np.median(vals):.4f}")

    # 按信号值分段
    bins = [0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
    for i in range(len(bins)-1):
        cnt = sum(1 for v in vals if bins[i] <= v < bins[i+1])
        if cnt > 0:
            print(f"  {bins[i]:.2f}~{bins[i+1]:.2f}: {cnt}只")

    # 最后信号日期分布
    print(f"\n最后信号日期分布:")
    date_counts = {}
    for r in nonzero:
        d = r['last_date'].date()
        date_counts[d] = date_counts.get(d, 0) + 1
    for d in sorted(date_counts.keys()):
        print(f"  {d}: {date_counts[d]}只")

    # 显示今日(2026-04-13)有信号的股票
    today_signals = [r for r in nonzero if r['last_date'] >= pd.Timestamp('2026-04-10')]
    print(f"\n最近3天(4/10~)有信号: {len(today_signals)}只")
    today_signals.sort(key=lambda x: x['last_signal'], reverse=True)

    holdings = {'002445.SZ', '002580.SZ', '002980.SZ', '003036.SZ', '300205.SZ'}
    new_signals = [r for r in today_signals if r['code'] not in holdings]
    print(f"排除持仓后新信号: {len(new_signals)}只")

    print(f"\n信号值>0.05的股票 (前20):")
    high_sig = [r for r in new_signals if r['last_signal'] > 0.05]
    high_sig.sort(key=lambda x: x['last_signal'], reverse=True)
    print(f"{'代码':<12} {'信号值':>8} {'最近5信号':<40} {'最近日期'}")
    for r in high_sig[:20]:
        recent = ','.join([f'{v:.3f}' for v in r['recent_signals']])
        print(f"{r['code']:<12} {r['last_signal']:>8.4f}  [{recent}]  {r['last_date'].date()}")

if __name__ == '__main__':
    main()
