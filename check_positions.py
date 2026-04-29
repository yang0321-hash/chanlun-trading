#!/usr/bin/env python3
"""持仓检查脚本"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, '/workspace')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(k, None)
from pathlib import Path

positions = {
    'SZ300936.SZ': 38.846,
    'SZ002600.SZ': 12.3,
    'SZ301062.SZ': 7.883,
    'SH688613.SH': 18.173,
    'SZ002951.SZ': 14.850,
    'SZ000826.SZ': 3.403,
    'SZ301128.SZ': 141.27,
}

def load_tdx(code):
    market = 'sz' if code.startswith('SZ') else 'sh'
    fname = code.lower().replace('.sz','').replace('.sh','')
    fpath = Path(f'/workspace/tdx_data/{market}/lday/{fname}.day')
    if not fpath.exists(): return None
    data = fpath.read_bytes()
    n = len(data) // 32
    arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
    dates = pd.to_datetime(arr[:, 0].astype(str), format='%Y%m%d')
    if fname == 'sz000826':
        prices = np.frombuffer(arr[:, 1:5].tobytes(), dtype=np.float32).reshape(n, 4)
    else:
        first_p = float(arr[0, 1])
        if first_p > 10_000_000:
            prices = np.frombuffer(arr[:, 1:5].tobytes(), dtype=np.float32).reshape(n, 4)
        else:
            prices = arr[:, 1:5] / 100.0
    volumes = arr[:, 6].astype(np.int64)
    return pd.DataFrame({
        'open': prices[:, 0], 'high': prices[:, 1], 'low': prices[:, 2],
        'close': prices[:, 3], 'volume': volumes
    }, index=dates).sort_index()

def load_index(idx_code):
    mkt = 'sh' if idx_code.startswith('sh') else 'sz'
    fpath = Path(f'/workspace/tdx_data/{mkt}/lday/{idx_code}.day')
    if not fpath.exists(): return None
    data = fpath.read_bytes()
    n = len(data) // 32
    arr = np.frombuffer(data[:n*32], dtype='<u4').reshape(n, 8)
    dates = pd.to_datetime(arr[:, 0].astype(str), format='%Y%m%d')
    first_p = float(arr[0, 1])
    if first_p > 10_000_000:
        prices = np.frombuffer(arr[:, 1:5].tobytes(), dtype=np.float32).reshape(n, 4)
    else:
        prices = arr[:, 1:5] / 100.0
    return pd.DataFrame({'close': prices[:, 3]}, index=dates).sort_index()

# 加载指数
sh_idx = load_index('sh000001')
sz_idx = load_index('sz399001')
sh_ma5 = sh_idx['close'].tail(5).mean()
sh_ma10 = sh_idx['close'].tail(10).mean()
sz_ma5 = sz_idx['close'].tail(5).mean()
sz_ma10 = sz_idx['close'].tail(10).mean()
sh_last = float(sh_idx['close'].iloc[-1])
sz_last = float(sz_idx['close'].iloc[-1])
print(f"大盘: 沪指={sh_last:.2f} MA5={sh_ma5:.2f} MA10={sh_ma10:.2f} [{('多头' if sh_ma5>sh_ma10 else '空头')}]")
print(f"     深成={sz_last:.2f} MA5={sz_ma5:.2f} MA10={sz_ma10:.2f} [{('多头' if sz_ma5>sz_ma10 else '空头')}]")

print()
print(f"{'代码':<14} {'成本':>8} {'当前':>8} {'盈亏%':>8} {'MA5':>8} {'MA10':>8} {'10日%':>8} {'诊断'}")
print("-" * 90)

results = []
for code, cost in positions.items():
    df = load_tdx(code)
    if df is None:
        print(f"{code:<14} {'无数据':>8}")
        continue
    cur = float(df['close'].iloc[-1])
    last_date = df.index[-1].date()
    pnl = (cur - cost) / cost * 100
    ma5 = df['close'].tail(5).mean()
    ma10 = df['close'].tail(10).mean()
    pct10 = (float(df['close'].iloc[-1]) / float(df['close'].iloc[-10]) - 1) * 100 if len(df) >= 10 else 0
    ma5_above = ma5 > ma10
    above_ma5 = cur > ma5
    above_ma10 = cur > ma10

    sl8 = cur * 0.92
    sl6 = cur * 0.94
    tp5 = cur * 1.05
    tp3 = cur * 1.03

    # 诊断
    if pnl <= -8:
        diag = "🛑 SL=8%止损"
    elif pnl <= -5:
        diag = "⚠️ 亏损5-8%"
    elif pnl >= 10:
        diag = "🔥 盈利>10%"
    elif pnl >= 5:
        diag = "📤 达TP5止盈"
    elif pnl >= 3:
        diag = "⏸ 达TP3"
    elif not ma5_above:
        diag = "❌ MA5<MA10"
    elif not above_ma10:
        diag = "⚠️ 价格<MA10"
    elif not above_ma5:
        diag = "⚠️ 价格<MA5"
    else:
        diag = "✅ 正常"

    print(f"{code:<14} {cost:>8.2f} {cur:>8.2f} {pnl:>+7.1f}% {ma5:>8.2f} {ma10:>8.2f} {pct10:>+7.1f}%  {diag}")

    results.append({
        'code': code, 'cost': cost, 'cur': cur, 'pnl': pnl,
        'ma5': ma5, 'ma10': ma10, 'ma5_above': ma5_above,
        'above_ma5': above_ma5, 'above_ma10': above_ma10,
        'sl8': sl8, 'sl6': sl6, 'tp5': tp5, 'tp3': tp3,
        'diag': diag, 'last_date': last_date
    })

print()
print(f"平均盈亏: {sum(r['pnl'] for r in results)/len(results):+.1f}%")
worst = min(results, key=lambda x: x['pnl'])
best = max(results, key=lambda x: x['pnl'])
print(f"最佳: {best['code']} ({best['pnl']:+.1f}%)  最差: {worst['code']} ({worst['pnl']:+.1f}%)")

print()
print("操作建议:")
for r in results:
    pnl = r['pnl']
    code = r['code']
    if pnl <= -8:
        print(f"  🛑 {code}: 止损，亏损{pnl:.1f}%")
    elif pnl >= 5 and r['diag'] == '📤 达TP5止盈':
        print(f"  📤 {code}: 已达TP5({r['tp5']:.2f})，建议分批止盈")
    elif not r['ma5_above']:
        print(f"  ⚠️ {code}: MA5<MA10空头排列，建议减仓观望")
    elif pnl < 0 and pnl > -5:
        print(f"  🔍 {code}: 亏损{pnl:.1f}%，未到止损位，暂持")
    else:
        print(f"  ✅ {code}: 正常持仓，盈亏{pnl:+.1f}%")
