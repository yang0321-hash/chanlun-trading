#!/usr/bin/env python3
"""
TDX全A日线每日更新 (tushare Pro)
- 支持 --full-a 更新全部A股 (~8140只)
- 支持 --stock <code> 更新单只
- 限速: 150请求/61秒 (tushare 200次/分)
- 盘中快速增量: watchlist+持仓优先, 全量放收盘后
用法:
  python3 tdx_full_updater.py --full-a      # 全量更新 (收盘后跑, 约56分钟)
  python3 tdx_full_updater.py --quick      # 快速增量 (watchlist+持仓+持仓板块, 约3分钟)
  python3 tdx_full_updater.py --stock 002553  # 单只更新
"""
import struct, os, sys, time, json, argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.insert(0, '/workspace')
for k in list(os.environ.keys()):
    if 'proxy' in k.lower(): os.environ.pop(k, None)

import tushare as ts

TDX_BASE = Path('/workspace/tdx_data')
CHUNK_SIZE = 150   # 每批请求数
SLEEP_SEC = 61     # 批间休息秒数

# ── TDX文件读写 ───────────────────────────────────────────
def get_tdx_last_date(filepath):
    """快速读取TDX日线最后日期 (只读最后32字节)"""
    if not filepath.exists(): return None
    try:
        sz = os.path.getsize(filepath)
        if sz < 32: return None
        with open(filepath, 'rb') as f:
            f.seek(-32, 2)
            data = f.read(32)
            # 检测格式: date字段
            date_u4 = struct.unpack('<I', data[0:4])[0]
            date_f4 = struct.unpack('<f', data[0:4])[0]
            # date原始int是YYYYMMDD (如20260422); float bits则是异常大
            if 19900101 <= date_u4 <= 20991231:
                return str(date_u4)
            # 如果date是float bits但open是合理float值, 可能是float格式
            raw_open = struct.unpack('<f', data[4:8])[0]
            if 0.5 <= raw_open <= 100000.0:
                return None  # float格式文件, 需要特殊处理, 暂时跳过
            return None
    except: return None

def write_tdx_daily_float32(filepath, df):
    """用float32格式写TDX日线 (新格式, 与sh600519等一致)"""
    import numpy as np
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, 'wb') as f:
        for _, row in df.iterrows():
            try:
                td = int(row['trade_date'])
                o = float(row['open']); hi = float(row['high'])
                lo = float(row['low']); c = float(row['close'])
                amt = float(row.get('amount', 0) or 0)
                v = int(float(row['vol']) if pd.notna(row.get('vol', 0)) else 0)
                if np.isnan(o) or np.isnan(hi) or np.isnan(lo) or np.isnan(c): continue
                f.write(struct.pack('<I', td))
                f.write(struct.pack('<ffff', o, hi, lo, c))
                f.write(struct.pack('<fII', amt, v, 0))
            except: continue

def append_tdx_daily(filepath, new_df):
    """追加TDX日线 — 自动检测现有文件格式(float32或int×100)"""
    import numpy as np
    os.makedirs(filepath.parent, exist_ok=True)
    records = []
    
    if filepath.exists() and os.path.getsize(filepath) >= 32:
        try:
            raw = filepath.read_bytes()
            n_ex = len(raw) // 32
            # 检测格式: float32格式文件open字段作为float解释是合理价格
            raw_open_f = struct.unpack_from('<f', raw, 4)[0]
            is_float_existing = (0.5 <= raw_open_f <= 100000.0)
            
            arr_u4 = np.frombuffer(raw[:n_ex*32], dtype=np.uint32).reshape(n_ex, 8)
            arr_f = np.frombuffer(raw[:n_ex*32], dtype=np.float32).reshape(n_ex, 8)
            
            for i in range(n_ex):
                td = str(int(arr_u4[i, 0]))
                if is_float_existing:
                    o,h,l,c = float(arr_f[i,1]), float(arr_f[i,2]), float(arr_f[i,3]), float(arr_f[i,4])
                else:
                    o,h,l,c = arr_u4[i,1]/100.0, arr_u4[i,2]/100.0, arr_u4[i,3]/100.0, arr_u4[i,4]/100.0
                v = int(arr_u4[i, 6])
                records.append({'trade_date': td, 'open': o, 'high': h, 'low': l, 'close': c, 'vol': v})
        except Exception as e:
            print(f'    读原文件失败: {e}')
            records = []
    
    existing = pd.DataFrame(records)
    if not existing.empty:
        existing_dates = set(existing['trade_date'])
        new_filtered = new_df[~new_df['trade_date'].isin(existing_dates)]
        if len(new_filtered) == 0: return 0
        df_all = pd.concat([existing, new_filtered], ignore_index=True).sort_values('trade_date').reset_index(drop=True)
    else:
        df_all = new_df.sort_values('trade_date').reset_index(drop=True)
    
    # 统一用float32格式写回
    with open(filepath, 'wb') as f:
        for _, row in df_all.iterrows():
            try:
                td = int(row['trade_date'])
                o,h,l,c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
                v = int(float(row['vol']) if pd.notna(row['vol']) else 0)
                amt = float(row.get('amount', 0) or 0)
                if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c): continue
                f.write(struct.pack('<I', td))
                f.write(struct.pack('<ffff', o, h, l, c))
                f.write(struct.pack('<fII', amt, v, 0))
            except: continue
    return len(new_filtered) if not existing.empty else len(df_all)

# ── tushare更新 ────────────────────────────────────────────
def get_full_a_list(pro):
    """获取全A股列表"""
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry')
    df2 = pro.stock_basic(exchange='', list_status='D', fields='ts_code,symbol,name')
    df3 = pro.stock_basic(exchange='', list_status='P', fields='ts_code,symbol,name')
    all_df = pd.concat([df, df2, df3], ignore_index=True)
    codes = []
    for _, row in all_df.iterrows():
        tc = row['ts_code']
        if tc.endswith('.SH'): codes.append(('sh', tc[:-3]))
        elif tc.endswith('.SZ'): codes.append(('sz', tc[:-3]))
    return codes

def update_one(ts_code, market, code, pro, days=365):
    """更新单只股票的日线数据"""
    today = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    filepath = TDX_BASE / market / 'lday' / f'{market}{code}.day'
    last_date = get_tdx_last_date(filepath)

    # 检测现有文件格式
    existing_is_float = False
    if filepath.exists() and os.path.getsize(filepath) >= 32:
        with open(filepath, 'rb') as f:
            raw_open = struct.unpack('<f', f.read(8)[4:8])[0]
            if 0.5 <= raw_open <= 100000.0:
                existing_is_float = True

    # 决定起始日期
    if last_date and last_date >= today:
        return 'skip', 0
    start = str(int(last_date) + 1) if last_date else start_date
    if start > today: return 'skip', 0

    try:
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=today)
        if df is None or len(df) == 0: return 'skip', 0

        df = df[['trade_date','open','high','low','close','vol','amount']].copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

    if existing_is_float:
        write_tdx_daily_float32(filepath, df)
    else:
        n_added = append_tdx_daily(filepath, df)
        return 'updated', n_added
    return 'updated', len(df)
    except Exception as e:
        return f'error:{e}', 0

def run_updates(codes, pro, label=''):
    """批量更新, 带限速"""
    updated = skipped = errors = 0
    t0 = time.time()
    total = len(codes)

    for i, (market, code) in enumerate(codes):
        ts_code = f'{code}.{market.upper()}'
        result, n = update_one(ts_code, market, code, pro)

        if result == 'updated':
            updated += 1
            if updated <= 10 or updated % 500 == 0:
                print(f'  ✓ {ts_code} +{n}条', flush=True)
        elif result == 'skip':
            skipped += 1
        else:
            errors += 1
            if errors <= 3:
                print(f'  ✗ {ts_code}: {result[:80]}', flush=True)

        # 进度
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f'  [{i+1}/{total}] {label} 更新:{updated} 跳过:{skipped} 错误:{errors} ETA={eta/60:.0f}min', flush=True)

        # 限速
        if (i + 1) % CHUNK_SIZE == 0:
            print(f'  ...限速等待{SLEEP_SEC}s ({time.time()-t0:.0f}s)', flush=True)
            time.sleep(SLEEP_SEC)

    elapsed = time.time() - t0
    print(f'  {label}完成: 更新{updated} 跳过{skipped} 错误{errors} ({elapsed:.0f}s)', flush=True)
    return updated, skipped, errors

def get_quick_codes(pro):
    """快速模式: watchlist + 持仓 + 持仓板块龙头"""
    codes = set()
    # watchlist
    wl = Path('/workspace/watchlist.txt')
    if wl.exists():
        for line in wl.read_text().strip().split('\n'):
            c = line.strip().replace('.SZ','').replace('.SH','').replace('.sz','').replace('.sh','')
            if c.isdigit() and len(c) == 6:
                codes.add(('sh', c) if c.startswith(('6','5','9','11')) else ('sz', c))
    # 持仓
    positions = ['300936','002600','301062','301128','688613','002951','000826']
    for c in positions:
        codes.add(('sh', c) if c.startswith(('6','5','9','11')) else ('sz', c))
    # 行业龙头 (从同花顺获取)
    try:
        import requests
        url = 'http://push2.eastmoney.com/api/qt/clist/get?cb=jQuery&pn=1&pz=20&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:90+t:2&fields=f12,f14'
        r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5)
        data = json.loads(r.text.replace('jQuery(','').rstrip(')'))
        for item in data.get('data',{}).get('diff',[]):
            c = str(item.get('f12',''))
            if c.isdigit() and len(c) == 6:
                codes.add(('sh', c) if c.startswith(('6','5','9','11')) else ('sz', c))
    except: pass
    return list(codes)

# ── 主程序 ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-a', action='store_true', help='全量更新全部A股 (~56分钟)')
    parser.add_argument('--quick', action='store_true', help='快速增量 (~3分钟)')
    parser.add_argument('--stock', type=str, help='单只股票代码')
    args = parser.parse_args()

    token = os.getenv('TUSHARE_TOKEN', '')
    ts.set_token(token)
    pro = ts.pro_api()

    today = datetime.now().strftime('%Y%m%d')
    print(f'[{datetime.now().strftime('%H:%M:%S')}] TDX全A更新启动 ({today})', flush=True)

    if args.stock:
        c = args.stock
        market = 'sh' if c.startswith(('6','5','9','11')) else 'sz'
        result, n = update_one(f'{c}.{market.upper()}', market, c, pro)
        print(f'{c}: {result}')
        return

    if args.quick:
        print('快速增量模式...', flush=True)
        codes = get_quick_codes(pro)
        print(f'共{len(codes)}只', flush=True)
        run_updates(codes, pro, label='快速')
        return

    if args.full_a:
        print('全量模式: 获取全A股列表...', flush=True)
        all_codes = get_full_a_list(pro)
        print(f'全A: {len(all_codes)}只, 预计约{len(all_codes)/CHUNK_SIZE*SLEEP_SEC/60:.0f}分钟', flush=True)
        run_updates(all_codes, pro, label='全A')
        return

    # 默认: 快速
    print('默认快速模式...', flush=True)
    codes = get_quick_codes(pro)
    print(f'共{len(codes)}只', flush=True)
    run_updates(codes, pro, label='快速')

if __name__ == '__main__':
    main()
