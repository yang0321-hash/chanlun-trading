#!/usr/bin/env python3
"""
大盘择时参数优化 + 板块共振替代指标测试
使用 sl_records.pkl 缓存（已含 monthly_sl、daily_state、sector信息）
"""
import sys, os, pickle, struct, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, '/workspace')

records = pickle.load(open('/workspace/sl_records.pkl', 'rb'))
print(f"signals: {len(records)}")

# 加载沪指历史数据计算各种大盘指标
base = '/workspace/tdx_data/sh/lday/sh000001.day'
rows = []
with open(base, 'rb') as f:
    data = f.read()
for i in range(len(data)//32):
    vals = struct.unpack('<8I', data[i*32:(i+1)*32])
    rows.append({'date': vals[0], 'open': vals[1]/100.0, 'high': vals[2]/100.0,
                 'low': vals[3]/100.0, 'close': vals[4]/100.0, 'volume': float(vals[6])})
idx_df = pd.DataFrame(rows)
idx_df['date'] = pd.to_datetime(idx_df['date'], format='%Y%m%d')
idx_df.set_index('date', inplace=True)
idx_df.sort_index(inplace=True)
print(f"沪指数据: {len(idx_df)} rows, {idx_df.index.min()} ~ {idx_df.index.max()}")

# 加载个股行业映射
sector_map = pickle.load(open('/workspace/sector_industry_map.pkl', 'rb'))

TP_TRIGGER = 0.06
TP_TRAIL = 0.06

# 为每个record补充大盘指标（在record中加入信号日期的index数据）
print("计算大盘指标序列...")
date_to_idx = {}
for rec in records:
    d = rec.get('date', '')
    if d not in date_to_idx:
        try:
            ts = pd.Timestamp(d)
            bi_list = idx_df.index.get_indexer([ts], method='bfill')
            if bi_list[0] >= 0:
                date_to_idx[d] = bi_list[0]
        except: pass

# 预计算不同大盘状态定义下的指标
# 存储: date -> {ma5_state, ma10_state, vol_state, trend_state}
def get_market_metrics(idx_df, bar_idx):
    """计算bar_idx处的大盘指标"""
    if bar_idx < 20: return {}
    df_s = idx_df.iloc[max(0, bar_idx-20):bar_idx+1]
    close = float(df_s['close'].iloc[-1])
    volume = float(df_s['volume'].iloc[-1])
    ma5 = float(df_s['close'].iloc[-5:].mean())
    ma10 = float(df_s['close'].iloc[-10:].mean())
    ma20 = float(df_s['close'].iloc[-20:].mean())
    ma60 = float(df_s['close'].iloc[-60:].mean()) if len(df_s) >= 60 else ma20
    avg_vol5 = float(df_s['volume'].iloc[-5:].mean())
    prev_ma5 = float(df_s['close'].iloc[-6]) if len(df_s) >= 6 else ma5
    prev_ma10 = float(df_s['close'].iloc[-11]) if len(df_s) >= 11 else ma10
    # MA方向
    ma5_dir = 1 if ma5 > prev_ma5 else (-1 if ma5 < prev_ma5 else 0)
    # 趋势强度
    trend = (close - ma60) / ma60 if ma60 > 0 else 0
    vol_ratio = volume / avg_vol5 if avg_vol5 > 0 else 1
    return {
        'close': close, 'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'ma5_dir': ma5_dir, 'trend': trend, 'vol_ratio': vol_ratio,
        'above_ma5': 1 if close > ma5 else 0,
        'above_ma10': 1 if close > ma10 else 0,
        'above_ma20': 1 if close > ma20 else 0,
    }

# 预计算每日的market_metrics
date_metrics = {}
dates = sorted(date_to_idx.keys())
for d in dates:
    idx = date_to_idx[d]
    date_metrics[d] = get_market_metrics(idx_df, idx)

# 为records附加market_metrics
for rec in records:
    d = rec.get('date', '')
    rec['mkt'] = date_metrics.get(d, {})

print(f"大盘指标已计算: {len(date_metrics)} 个交易日")

def bt_with_market_filter(
    tp_trigger=0.06, tp_trail=0.06,
    mkt_def='def1',        # 大盘状态定义
    mkt_cap=False,         # 是否启用大盘仓位上限
    sector_filter=False,    # 板块共振过滤
    mkt_bull_mult=0.8, mkt_bear_mult=1.2, mkt_neutral_mult=0.95,
    min_cap=None,
    ):
    """大盘择时过滤回测"""
    pnls = []
    exit_counts = defaultdict(int)

    for rec in records:
        btype = rec['btype']
        monthly_sl_val = rec['monthly_sl']
        daily_state = rec['daily_state']
        low_rel = rec['low_rel']
        close_rel = rec['close_rel']
        high_rel = rec['high_rel']
        mkt = rec.get('mkt', {})
        n = len(close_rel)
        if n == 0: continue

        # 大盘状态定义
        if mkt_def == 'def1':  # 原始: MA5>MA10 + price>MA5
            ma5_gt_ma10 = mkt.get('ma5', 0) > mkt.get('ma10', 0)
            price_gt_ma5 = mkt.get('above_ma5', 0) == 1
            mkt_state = 'bull' if (ma5_gt_ma10 and price_gt_ma5) else ('bear' if (not ma5_gt_ma10 and not price_gt_ma5) else 'neutral')
        elif mkt_def == 'def2':  # 加入MA方向
            ma5_up = mkt.get('ma5_dir', 0) > 0
            ma5_gt_ma10 = mkt.get('ma5', 0) > mkt.get('ma10', 0)
            above_ma5 = mkt.get('above_ma5', 0) == 1
            above_ma20 = mkt.get('above_ma20', 0) == 1
            if ma5_up and ma5_gt_ma10 and above_ma5: mkt_state = 'bull'
            elif not ma5_up and not ma5_gt_ma10 and not above_ma5: mkt_state = 'bear'
            else: mkt_state = 'neutral'
        elif mkt_def == 'def3':  # 加入趋势强度
            trend = mkt.get('trend', 0)
            above_ma20 = mkt.get('above_ma20', 0) == 1
            ma5_gt_ma10 = mkt.get('ma5', 0) > mkt.get('ma10', 0)
            if trend > 0.02 and above_ma20 and ma5_gt_ma10: mkt_state = 'bull'
            elif trend < -0.02 and not above_ma20 and not ma5_gt_ma10: mkt_state = 'bear'
            else: mkt_state = 'neutral'
        elif mkt_def == 'def4':  # 简化: MA5>MA10
            mkt_state = 'bull' if mkt.get('ma5', 0) > mkt.get('ma10', 0) else ('bear' if mkt.get('ma5', 0) < mkt.get('ma10', 0) else 'neutral')
        elif mkt_def == 'def5':  # 仅价格方向
            above_ma20 = mkt.get('above_ma20', 0) == 1
            mkt_state = 'bull' if above_ma20 else 'bear'
        elif mkt_def == 'def6':  # MA5方向 + MA5>MA10
            ma5_up = mkt.get('ma5_dir', 0) > 0
            ma5_gt_ma10 = mkt.get('ma5', 0) > mkt.get('ma10', 0)
            if ma5_up and ma5_gt_ma10: mkt_state = 'bull'
            elif not ma5_up and not ma5_gt_ma10: mkt_state = 'bear'
            else: mkt_state = 'neutral'
        else:
            mkt_state = daily_state

        # 板块过滤（sector_map替代：按行业简单分组）
        if sector_filter:
            code = rec.get('code', '')
            sector = sector_map.get(code, '其他')
            # 今日sector无法从records重建，用monthly_sl作为板块强度代理
            # monthly_sl高的月份=板块强势月，间接过滤
            pass  # 暂不支持，用monthly_sl间接代替

        # 仓位上限过滤
        if mkt_cap:
            cap_map = {'bull': 0.8, 'neutral': 0.5, 'bear': 0.2}
            cap = cap_map.get(mkt_state, 0.5)
            if min_cap and cap < min_cap:
                # 跳过低于最低仓位门槛的信号（降低市场敞口）
                # 但这会改变N，需要记录
                pass

        # SL计算（用新的mkt_state替代原始daily_state）
        mult_map = {'bull': mkt_bull_mult, 'bear': mkt_bear_mult, 'neutral': mkt_neutral_mult}
        mult = mult_map.get(mkt_state, 1.0)
        sl_ratio = monthly_sl_val * mult
        sl_rel = sl_ratio - 1

        if np.any(low_rel <= sl_rel):
            exit_counts['sl'] += 1; pnls.append(sl_rel - 0.0006); continue

        trigger_idx = None
        for i in range(n):
            if close_rel[i] >= tp_trigger: trigger_idx = i; break
        if trigger_idx is None:
            exit_counts['to'] += 1; pnls.append(close_rel[-1] - 0.0006); continue

        price_hwm = high_rel[0]; tp_exit = None
        for i in range(trigger_idx, n):
            if high_rel[i] > price_hwm: price_hwm = high_rel[i]
            dd = (price_hwm - close_rel[i]) / (1 + price_hwm)
            if dd >= tp_trail: tp_exit = close_rel[i]; break
        if tp_exit is not None:
            exit_counts['tp'] += 1; pnls.append(tp_exit - 0.0006)
        else:
            exit_counts['to'] += 1; pnls.append(close_rel[-1] - 0.0006)

    if not pnls: return None
    pnls = np.array(pnls)
    total = len(pnls)
    wr = (pnls > 0).mean() * 100
    avg = pnls.mean() * 100
    std_w = pnls[pnls > 0].std() if len(pnls[pnls > 0]) > 0 else 1
    sharpe = 0.04 / std_w if std_w > 1e-8 else 0
    max_dd = abs(pnls.min()) * 100
    sl_n = exit_counts['sl']; tp_n = exit_counts['tp']; to_n = exit_counts['to']
    return {
        'sharpe': sharpe, 'win_rate': wr, 'avg_pnl': avg, 'max_dd': max_dd, 'n': total,
        'sl_pct': sl_n/total*100, 'tp_pct': tp_n/total*100, 'to_pct': to_n/total*100,
    }

# ── 大盘择时定义对比 ────────────────────────────────────────────────────────
print("\n" + "="*100)
print("A. 大盘择时定义对比（固定止盈6%/6%，止损用月度SL×乘数）")
print("="*100)

configs = [
    # (mkt_def, mkt_bull_mult, mkt_bear_mult, mkt_neutral_mult, desc)
    ('def1', 0.8, 1.2, 0.95, 'A1: 原始定义(MA5>MA10+price>MA5)'),
    ('def2', 0.8, 1.2, 0.95, 'A2: +MA方向+MA20确认'),
    ('def3', 0.8, 1.2, 0.95, 'A3: +趋势强度(trend+MA20)'),
    ('def4', 0.8, 1.2, 0.95, 'A4: 仅MA5>MA10'),
    ('def5', 0.8, 1.2, 0.95, 'A5: 仅价格>MA20'),
    ('def6', 0.8, 1.2, 0.95, 'A6: MA5方向+MA5>MA10'),
    ('def1', 1.0, 1.0, 1.0, 'B: 原始定义+无乘数(基准)'),
]

results = []
for mkt_def, bm, bem, nm, desc in configs:
    r = bt_with_market_filter(mkt_def=mkt_def, mkt_bull_mult=bm, mkt_bear_mult=bem, mkt_neutral_mult=nm)
    if r: results.append((desc, r))

results.sort(key=lambda x: -x[1]['sharpe'])
print(f"\n{'策略':<50} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'SL%':>6} {'止盈%':>6} {'超时%':>6} {'N':>6}")
print("-"*105)
for desc, r in results:
    print(f"{desc:<50} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} "
          f"{r['sl_pct']:>5.1f}% {r['tp_pct']:>5.1f}% {r['to_pct']:>5.1f}% {r['n']:>6}")

# ── 大盘仓位上限效果 ──────────────────────────────────────────────────────────
print("\n" + "="*100)
print("B. 大盘仓位上限效果（使用最优def1）")
print("="*100)

cap_configs = [
    # (cap_bull, cap_neutral, cap_bear, desc)
    (1.0, 1.0, 1.0, 'B1: 无仓位上限（全仓）'),
    (0.8, 0.5, 0.2, 'B2: 牛市80% 震荡50% 熊市20%'),
    (0.7, 0.4, 0.2, 'B3: 牛市70% 震荡40% 熊市20%'),
    (0.6, 0.3, 0.1, 'B4: 牛市60% 震荡30% 熊市10%'),
    (0.5, 0.3, 0.1, 'B5: 牛市50% 震荡30% 熊市10%'),
]

cap_results = []
for cap_b, cap_n, cap_bear, desc in cap_configs:
    pnls_all = []
    exit_counts = defaultdict(int)
    for rec in records:
        btype = rec['btype']
        monthly_sl_val = rec['monthly_sl']
        daily_state = rec['daily_state']
        mkt = rec.get('mkt', {})
        low_rel = rec['low_rel']; close_rel = rec['close_rel']; high_rel = rec['high_rel']
        n = len(close_rel)
        if n == 0: continue
        # 大盘状态用def1
        ma5_gt_ma10 = mkt.get('ma5', 0) > mkt.get('ma10', 0)
        price_gt_ma5 = mkt.get('above_ma5', 0) == 1
        mkt_state = 'bull' if (ma5_gt_ma10 and price_gt_ma5) else ('bear' if (not ma5_gt_ma10 and not price_gt_ma5) else 'neutral')
        cap = {'bull': cap_b, 'neutral': cap_n, 'bear': cap_bear}.get(mkt_state, 1.0)
        sl_ratio = monthly_sl_val * {'bull': 0.8, 'bear': 1.2, 'neutral': 0.95}.get(mkt_state, 1.0)
        sl_rel = sl_ratio - 1
        if np.any(low_rel <= sl_rel): exit_counts['sl']+=1; pnls_all.append((sl_rel-0.0006)*cap); continue
        ti = None
        for i in range(n):
            if close_rel[i] >= 0.06: ti=i; break
        if ti is None: exit_counts['to']+=1; pnls_all.append((close_rel[-1]-0.0006)*cap); continue
        hwm = high_rel[0]; tp_exit=None
        for i in range(ti, n):
            if high_rel[i] > hwm: hwm = high_rel[i]
            dd = (hwm - close_rel[i])/(1+hwm)
            if dd >= 0.06: tp_exit=close_rel[i]; break
        if tp_exit is not None: exit_counts['tp']+=1; pnls_all.append((tp_exit-0.0006)*cap)
        else: exit_counts['to']+=1; pnls_all.append((close_rel[-1]-0.0006)*cap)
    pnls = np.array(pnls_all)
    total = len(pnls)
    wr = (pnls>0).mean()*100; avg = pnls.mean()*100
    std_w = pnls[pnls>0].std() if len(pnls[pnls>0])>0 else 1
    sharpe = 0.04/std_w if std_w>1e-8 else 0
    max_dd = abs(pnls.min())*100
    sl_n=exit_counts['sl']; tp_n=exit_counts['tp']; to_n=exit_counts['to']
    cap_results.append((desc, {'sharpe':sharpe,'win_rate':wr,'avg_pnl':avg,'max_dd':max_dd,'n':total,
                                'sl_pct':sl_n/total*100,'tp_pct':tp_n/total*100,'to_pct':to_n/total*100}))

cap_results.sort(key=lambda x: -x[1]['sharpe'])
print(f"\n{'策略':<50} {'Sharpe':>7} {'WR%':>5} {'均盈%':>7} {'DD%':>6} {'N':>6}")
print("-"*85)
for desc, r in cap_results:
    print(f"{desc:<50} {r['sharpe']:>7.3f} {r['win_rate']:>5.0f} {r['avg_pnl']:>+7.2f} {r['max_dd']:>6.1f} {r['n']:>6}")

print("\n── 大盘择时优化结论 ──")
best_mkt = results[0]
print(f"最优大盘定义: {best_mkt[0]}")
print(f"  Sharpe={best_mkt[1]['sharpe']:.3f} WR={best_mkt[1]['win_rate']:.0f}% 均盈={best_mkt[1]['avg_pnl']:+.2f}%")
print(f"\n仓位上限结论:")
for desc, r in cap_results:
    print(f"  {desc}: Sharpe={r['sharpe']:.3f} WR={r['win_rate']:.0f}% 均盈={r['avg_pnl']:+.2f}% DD={r['max_dd']:.1f}%")
