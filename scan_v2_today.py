#!/usr/bin/env python3
"""
v2.0框架 × 全A扫描
- 用scanner同款v14 engine, 扫全部TDX数据
- v2.0过滤: 置信度>=0.65, 2buy/3buy优先
- mod_strong(8分)仓位上限50%
"""
import os, sys, json, time, pickle
from pathlib import Path
from datetime import datetime, timedelta

for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

PROJECT_ROOT = Path('/workspace/chanlun_system')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import struct

# ============ 引擎 ============
from signal_engine import SignalEngine

# ============ TDX读取 ============
TDX_ROOT = Path('/workspace/tdx_data')

def get_all_codes():
    codes = []
    for mdir in ['sh', 'sz']:
        mpath = TDX_ROOT / mdir / 'lday'
        if not mpath.exists(): continue
        for f in mpath.iterdir():
            if f.suffix == '.day':
                codes.append(f.stem)  # e.g. sh000001
    return codes

def read_tdx(path):
    """读TDX .day文件，兼容int/float格式"""
    try:
        with open(path, 'rb') as f:
            records = []
            while True:
                data = f.read(32)
                if not data: break
                r = struct.unpack('<IfffffII', data)
                records.append(r)
        df = pd.DataFrame(records, columns=['date','open','high','low','close','amount','vol','reserved'])
        first_open = df['open'].iloc[0]
        if not (50 < first_open < 10000000):
            for col in ['open','high','low','close']:
                df[col] = df[col] / 100.0
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.set_index('date').sort_index()
        return df
    except:
        return pd.DataFrame()

# ============ v2.0 过滤 ============
def v2_filter(code, df, engine):
    """返回 (pass, stype, conf, info_dict)"""
    try:
        sig = engine.generate(code, df)
        
        # 置信度门槛
        conf = sig.get('confidence', 0)
        if conf < 0.65:
            return False, None, 0, 'conf<0.65'
        
        # 买点类型
        sig_type = sig.get('signal_type', 'none')
        if sig_type == 'none':
            return False, None, 0, 'no_signal'
        
        # 2buy优先,3buy次之
        if sig_type in ['2buy', '3buy', 'buy_2']:
            stype = sig_type
        elif sig_type == 'buy_1':
            stype = '1buy'
        else:
            return False, None, 0, f'not_buy:{sig_type}'
        
        # v2.0仓位上限 (mod_strong=50%, 单票上限25%)
        # 但3buy在mod_strong表现差(0.39), 只轻仓10%
        cap = 0.10 if stype == '3buy' else 0.25
        
        # SL
        sl = sig.get('stop_loss', 0)
        close = df['close'].iloc[-1]
        dist_sl = (close - sl) / close * 100 if sl > 0 else 999
        
        # 板块信息
        industry = sig.get('industry', 'unknown')
        
        info = {
            'stype': stype,
            'conf': conf,
            'close': close,
            'sl': sl,
            'dist_sl': dist_sl,
            'industry': industry,
            'signal_weight': sig.get('signal_weight', 0),
            'cap': cap,
            'pct_chng': sig.get('pct_chng', 0),
        }
        return True, stype, conf, info
        
    except Exception as e:
        return False, None, 0, str(e)

# ============ 实时行情 ============
def get_realtime_prices(codes):
    """腾讯行情批量获取现价"""
    qt_codes = ','.join(codes)
    try:
        resp = requests.get(
            f'http://qt.gtimg.cn/q={qt_codes}',
            headers={'User-Agent': 'Mozilla/5.0'},
            proxies={'http': None, 'https': None},
            timeout=10
        )
        prices = {}
        for line in resp.text.strip().split('\n'):
            if '~' not in line: continue
            f = line.split('~')
            if len(f) < 40: continue
            raw = f[2]
            if raw.startswith('6'):
                code = f'{raw}.SH'
            else:
                code = f'{raw}.SZ'
            prices[code] = {
                'name': f[1],
                'price': float(f[3]),
                'yclose': float(f[4]),
                'high': float(f[33]),
                'low': float(f[34]),
                'pct': (float(f[3]) - float(f[4])) / float(f[4]) * 100,
            }
        return prices
    except:
        return {}

# ============ 主程序 ============
def main():
    t0 = time.time()
    
    # 大盘
    df_sh = pickle.load(open('/workspace/sh000001_latest.pkl', 'rb'))
    latest_sh = df_sh.index[-1]
    sh_row = df_sh.iloc[-1]
    
    def v2_score(d):
        try:
            r = df_sh.loc[d]
        except: return None
        s = 0
        pct_ma5 = (r['close'] - r['MA5']) / r['MA5'] * 100
        s += 2 if pct_ma5 > 5 else (1 if pct_ma5 >= 0 else 0)
        spread = (r['MA5'] - r['MA10']) / r['MA10'] * 100
        s += 2 if spread > 1 else (1 if spread >= 0 else 0)
        pct_ma20 = (r['close'] - r['MA20']) / r['MA20'] * 100
        s += 2 if pct_ma20 > 5 else (1 if pct_ma20 >= 0 else 0)
        dif, dp = r['DIF'], r['DIF_prev']
        s += 2 if dif > 0 and dif > dp else (1 if dif > 0 else 0)
        vc = r['vol_change'] * 100
        s += 2 if vc > 20 else (1 if vc >= -20 else 0)
        md = r['MA5_diff']
        s += 2 if md > 0 else (1 if md == 0 else 0)
        return int(s)
    
    score = v2_score(latest_sh)
    bkt = 'strong' if score >= 9 else ('mod_strong' if score >= 6 else ('weak' if score >= 3 else 'very_weak'))
    pos_cap = {'strong': 0.75, 'mod_strong': 0.50, 'weak': 0.25, 'very_weak': 0.10}
    max_pos = pos_cap.get(bkt, 0.50)
    
    print("=" * 70)
    print(f"v2.0 全A扫描 | {latest_sh.date()} | 大盘: {bkt}({score}/12) 仓位上限{max_pos*100:.0f}%")
    print("=" * 70)
    
    codes = get_all_codes()
    print(f"[数据] TDX: {len(codes)}只")
    
    engine = SignalEngine()
    results = []
    
    def process(code):
        try:
            mdir = 'sh' if code.startswith('sh') else 'sz'
            mcode = code[2:]  # e.g. 000001
            full_code = f'{mcode}.SZ' if mdir == 'sz' else f'{mcode}.SH'
            
            df = read_tdx(TDX_ROOT / mdir / 'lday' / f'{code}.day')
            if df.empty or len(df) < 120:
                return None
            
            last_date = df.index[-1]
            if last_date < pd.Timestamp('2026-04-20'):
                return None
            
            ok, stype, conf, info = v2_filter(full_code, df, engine)
            if ok:
                return {
                    'code': full_code,
                    'tdx_code': code,
                    **info,
                }
            return None
        except:
            return None
    
    # 多线程扫描
    BATCH = 500
    done = 0
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i+BATCH]
        with ThreadPoolExecutor(max_workers=40) as ex:
            futures = {ex.submit(process, c): c for c in batch}
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)
        done = min(i + BATCH, len(codes))
        if (i // BATCH) % 2 == 0:
            print(f"  进度: {done}/{len(codes)} ({len(results)}个候选)")
    
    elapsed = time.time() - t0
    print(f"\n[完成] {len(codes)}只, {elapsed:.0f}秒, 候选{len(results)}个")
    
    if not results:
        print("\n今日无符合v2.0(conf>=0.65)的标的")
        return
    
    # 按置信度排序
    results.sort(key=lambda x: x['conf'], reverse=True)
    
    # 获取实时价格
    tdx_codes = [r['tdx_code'] for r in results[:50]]
    rt_prices = get_realtime_prices(tdx_codes)
    
    # 过滤掉持仓
    holdings = ['002445.SZ', '002580.SZ', '002980.SZ', '003036.SZ', '300205.SZ']
    new_results = [r for r in results if r['code'] not in holdings]
    
    print(f"\n{'='*70}")
    print(f"候选标的 (排除持仓, 共{len(new_results)}只, 按置信度排序)")
    print(f"{'='*70}")
    print(f"{'代码':<12} {'名称':<8} {'类型':<6} {'置信度':>7} {'现价':>8} {'距SL%':>6} {'行业':<12} {'仓位建议'}")
    print("-" * 80)
    
    for r in new_results[:20]:
        code = r['code']
        name = rt_prices.get(code, {}).get('name', code[-6:])
        price = rt_prices.get(code, {}).get('price', r['close'])
        pct = rt_prices.get(code, {}).get('pct', 0)
        dist_sl = r['dist_sl']
        
        # v2.0操作建议
        if dist_sl < 1.5:
            op = "⚠️SL临界"
        elif dist_sl < 3:
            op = "⚠️关注"
        else:
            op = f"模拟{int(r['cap']*100)}%仓"
        
        print(f"{code:<12} {name:<8} {r['stype']:<6} {r['conf']:>7.2f} "
              f"{price:>8.2f} {dist_sl:>5.1f}%  {r['industry']:<12} {op}")
    
    # 统计
    by_type = {}
    for r in new_results:
        t = r['stype']
        if t not in by_type: by_type[t] = []
        by_type[t].append(r)
    
    print(f"\n按类型:")
    for t, grp in sorted(by_type.items(), key=lambda x: -len(x[1])):
        avg_conf = np.mean([r['conf'] for r in grp])
        avg_dist = np.mean([r['dist_sl'] for r in grp])
        print(f"  {t}: {len(grp)}只  均置信{avg_conf:.2f}  均距SL{avg_dist:.1f}%")
    
    print(f"\n注: mod_strong(8分)仓位上限50%, 3buy降权至10%模拟仓, 2buy为25%模拟仓")
    print(f"    总候选{len(new_results)}只, 若全部等权配置将超出仓位上限")

if __name__ == '__main__':
    main()
