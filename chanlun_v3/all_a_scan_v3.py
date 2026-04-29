# -*- coding: utf-8 -*-
"""
全A股缠论扫描 v3.0 — 四阶段筛选
──────────────────────────────────
Phase 1: 本地数据全量扫描6186只 → 找出30天内有信号的
Phase 2: 按信号质量筛选TOP N（约50-80只）
Phase 3: 对TOP N调API补充最新日线数据，重新扫描确认
Phase 4: 对确认后的候选批量取30分钟K线 → M30EntrySignal精确入场判断

用法: python all_a_scan_v3.py [--top 80] [--m30] [--m30top 20]
"""
import sys, os, glob, time, json, argparse
sys.stdout.reconfigure(encoding='utf-8')

from tdx_day_reader import load_stock_klines
from kline_supplement import load_latest, supplement_klines
from chanlun_engine import ChanLunEngine, KLine
from chanlun_batch_scan import scan_single_stock, score_market_env
from multi_level import (daily_to_weekly, weekly_direction_ggdd,
                         multi_level_analyze, M30EntrySignal)
from datetime import datetime

TDX_BASE = r"D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc"

# 信号级别优先级（数字越小越优先）
SIGNAL_PRIORITY = {
    '3B': 1, '2B3B': 2, '2B': 3, '1B': 4, 'pz1B': 5,
    'xzd1B': 6, 'sub1B': 7, 'quasi2B': 8, 'subQuasi2B': 9
}

# 高优先级信号
PRIORITY_SIGNALS = {'3B', '2B3B', '2B', '1B', 'pz1B', 'xzd1B', 'quasi2B'}


def get_all_stock_files():
    sh = glob.glob(os.path.join(TDX_BASE, 'sh', 'lday', '*.day'))
    sz = glob.glob(os.path.join(TDX_BASE, 'sz', 'lday', '*.day'))
    return sh + sz


def is_a_share(code, setcode):
    if setcode == '1':
        return code.startswith('6') or code.startswith('5')
    else:
        return code.startswith('0') or code.startswith('3')


def phase1_scan():
    """Phase 1: 全量扫描"""
    print("=" * 60)
    print("Phase 1: 全A股本地数据扫描")

    latest = load_latest()
    all_files = get_all_stock_files()
    print(f"共 {len(all_files)} 个.day文件")

    results = []
    scanned = 0
    t0 = time.time()

    for fp in all_files:
        fn = os.path.basename(fp)
        code = fn[2:8]
        sc = '1' if fn[:2] == 'sh' else '0'
        if not is_a_share(code, sc):
            continue

        raw = load_stock_klines(code, sc, 500)
        if not raw or len(raw) < 100:
            continue
        scanned += 1

        klines_raw = supplement_klines(code, sc, raw, latest)
        klines = [KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4]) for d in klines_raw]

        weekly = daily_to_weekly(klines)
        w_dir = weekly_direction_ggdd(weekly)

        try:
            engine = ChanLunEngine()
            result = engine.analyze(klines)
        except:
            continue

        buys = result.get('buy_sell_points', [])
        buy_points = [b for b in buys if 'B' in b['type']]

        if not buy_points:
            continue

        last_date = klines_raw[-1][0]
        try:
            last_dt = datetime.strptime(str(last_date), '%Y%m%d')
        except:
            continue

        recent_buys = []
        for b in buy_points:
            try:
                bdt = datetime.strptime(str(b['date']), '%Y%m%d')
                age = (last_dt - bdt).days
                if age <= 30:
                    recent_buys.append(b)
            except:
                pass

        if not recent_buys:
            continue

        recent_buys.sort(key=lambda b: (SIGNAL_PRIORITY.get(b['type'], 99), -b['confidence']))
        best = recent_buys[0]

        results.append({
            'code': code,
            'setcode': sc,
            'signal_type': best['type'],
            'signal_date': best['date'],
            'signal_price': best['price'],
            'confidence': best['confidence'],
            'reason': best.get('reason', ''),
            'stop_loss': best.get('stop_loss', 0),
            'close': klines_raw[-1][4],
            'last_date': last_date,
            'total_buys': len(recent_buys),
            'all_signals': [b['type'] for b in recent_buys],
            'weekly_dir': w_dir['direction'],
            'weekly_trend': w_dir['trend_type'],
            'weekly_conf': w_dir.get('confidence', 0),
        })

        if scanned % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  已扫描 {scanned} | 命中 {len(results)} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nPhase 1 完成: {scanned}只扫描 → {len(results)}只命中 ({elapsed:.1f}s)")
    return results


def phase2_filter(results, top_n=80):
    """Phase 2: 筛选TOP N"""
    print(f"\n{'='*60}")
    print(f"Phase 2: 筛选 TOP {top_n}")

    priority_hits = [r for r in results if r['signal_type'] in PRIORITY_SIGNALS]
    print(f"高优先级信号: {len(priority_hits)}只")
    sig_dist = {}
    for r in priority_hits:
        sig_dist[r['signal_type']] = sig_dist.get(r['signal_type'], 0) + 1
    for t in sorted(sig_dist, key=lambda x: SIGNAL_PRIORITY.get(x, 99)):
        print(f"  {t}: {sig_dist[t]}只")

    def sort_key(r):
        w_bonus = 0.1 if r.get('weekly_dir') == 'up' else (-0.2 if r.get('weekly_dir') == 'down' else 0)
        return (SIGNAL_PRIORITY.get(r['signal_type'], 99), -(r['confidence'] + w_bonus))

    priority_hits.sort(key=sort_key)
    top = priority_hits[:top_n]

    print(f"\nTOP {len(top)}只:")
    for r in top:
        print(f"  {r['code']}.{r['setcode']} | {r['signal_type']} {r['signal_date']} "
              f"@{r['signal_price']:.2f} | conf={r['confidence']:.2f} | close={r['close']:.2f}")

    return top


def phase3_confirm(top):
    """Phase 3: API补充最新日线 + 重新确认"""
    print(f"\n{'='*60}")
    print(f"Phase 3: API日线确认")

    latest = load_latest()
    confirmed = []

    for r in top:
        code, sc = r['code'], r['setcode']
        raw = load_stock_klines(code, sc, 500)
        if not raw:
            continue
        klines_raw = supplement_klines(code, sc, raw, latest)
        klines = [KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4]) for d in klines_raw]

        try:
            engine = ChanLunEngine()
            result = engine.analyze(klines)
        except:
            continue

        buys = [b for b in result.get('buy_sell_points', []) if 'B' in b['type']]
        last_date = klines_raw[-1][0]
        try:
            last_dt = datetime.strptime(str(last_date), '%Y%m%d')
        except:
            continue

        recent = [b for b in buys
                  if abs((last_dt - datetime.strptime(str(b['date']), '%Y%m%d')).days) <= 30]

        if recent:
            recent.sort(key=lambda b: (SIGNAL_PRIORITY.get(b['type'], 99), -b['confidence']))
            best = recent[0]
            r['signal_type'] = best['type']
            r['signal_date'] = best['date']
            r['signal_price'] = best['price']
            r['confidence'] = best['confidence']
            r['reason'] = best.get('reason', '')
            r['stop_loss'] = best.get('stop_loss', 0)
            r['close'] = klines_raw[-1][4]
            r['last_date'] = last_date
            r['confirmed'] = True
            confirmed.append(r)
        else:
            r['confirmed'] = False
            print(f"  ⚠️ {code} 信号未确认（可能已过期）")

    print(f"\nPhase 3 完成: {len(confirmed)}/{len(top)}只确认")
    return confirmed


def phase4_m30_analysis(confirmed, m30_top=20):
    """Phase 4: 批量取30分钟K线 + M30EntrySignal精确入场
    
    通过写入队列文件，由外部API批量获取30分钟数据。
    然后读取返回数据做分析。
    
    由于无法在Python中直接调用OpenClaw工具，
    这里生成fetch队列 + 提供 analyse 函数供外部调用。
    """
    print(f"\n{'='*60}")
    print(f"Phase 4: 30分钟精确入场分析 (TOP {m30_top})")

    # 只分析TOP候选（3B/2B/1B优先）
    candidates = [r for r in confirmed if r['signal_type'] in ('3B', '2B3B', '2B', '1B', 'sub1B')]
    if len(candidates) < m30_top:
        # 补充其他类型
        others = [r for r in confirmed if r not in candidates]
        candidates.extend(others[:m30_top - len(candidates)])
    candidates = candidates[:m30_top]

    # 生成30分钟API队列
    m30_queue = []
    for r in candidates:
        m30_queue.append({
            'code': r['code'],
            'setcode': r['setcode'],
            'period': '2',  # 30分钟
            'wantNum': '200',
        })

    queue_path = '_m30_fetch_queue.json'
    with open(queue_path, 'w', encoding='utf-8') as f:
        json.dump(m30_queue, f, ensure_ascii=False, indent=2)
    print(f"30分钟API队列已生成: {queue_path} ({len(m30_queue)}只)")
    print(f"  需要通过API批量获取30分钟K线数据")

    return candidates, queue_path


def analyse_m30_for_stock(code, setcode, daily_klines_raw, m30_klines_raw):
    """对单只股票做完整的三级别分析+30分钟入场判断
    
    Args:
        code: 股票代码
        setcode: 市场代码
        daily_klines_raw: 日线原始数据 [[date,open,high,low,close], ...]
        m30_klines_raw: 30分钟原始数据 [[date,open,high,low,close], ...] 或 KLine列表
    
    Returns:
        dict: 分析结果，包含 summary, m30_entry, daily_info
    """
    # 日线KLine
    daily_klines = [KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4])
                    for d in daily_klines_raw]

    # 30分钟KLine
    if m30_klines_raw and len(m30_klines_raw) > 0:
        if isinstance(m30_klines_raw[0], KLine):
            m30_klines = m30_klines_raw
        else:
            m30_klines = [KLine(date='m30_{:03d}'.format(i),
                                open=r[0], high=r[1], low=r[2], close=r[3])
                          for i, r in enumerate(m30_klines_raw)]
    else:
        m30_klines = None

    result = multi_level_analyze(daily_klines, m30_klines)

    # 提取关键信息
    info = {
        'code': code,
        'setcode': setcode,
        'summary': result.get('summary', ''),
        'daily_buys': [],
        'm30_entry': None,
    }

    # 日线买点
    for b in result.get('daily', {}).get('buys', []):
        info['daily_buys'].append({
            'type': b['type'],
            'date': b['date'],
            'price': b['price'],
            'confidence': b['confidence'],
        })

    # 30分钟入场信号
    if result.get('m30'):
        entry = result['m30'].get('entry', {})
        info['m30_entry'] = {
            'status': entry.get('status', 'none'),
            'detail': entry.get('detail', ''),
            'entry_price': entry.get('entry_price'),
            'stop_loss': entry.get('stop_loss'),
            'divergence_info': entry.get('divergence_info'),
            'fractal_info': entry.get('fractal_info'),
        }

    return info


def run_full_scan_with_m30(top_n=80, m30_top=20, do_m30=False):
    """完整四阶段扫描"""
    # 大盘评分
    idx_raw = load_stock_klines("000001", "1", 80)
    idx_klines = [[k[0], k[1], k[2], k[3], k[4], k[6] if len(k) > 6 else 0] for k in idx_raw]
    market_score = score_market_env(idx_klines)
    print(f"[大盘] {market_score.get('total', 0)}/12分 {market_score.get('level', '')}")

    # Phase 1
    results = phase1_scan()
    if not results:
        print("\n无符合条件的股票")
        return

    # Phase 2
    top = phase2_filter(results, top_n)
    if not top:
        print("\n筛选后无符合条件的股票")
        return

    # Phase 3
    confirmed = phase3_confirm(top)
    if not confirmed:
        print("\n确认后无有效候选")
        return

    # 保存确认结果
    with open('_all_a_top.json', 'w', encoding='utf-8') as f:
        json.dump(confirmed, f, ensure_ascii=False, indent=2, default=str)

    # Phase 4 (仅生成队列)
    if do_m30:
        candidates, queue_path = phase4_m30_analysis(confirmed, m30_top)
        return confirmed, candidates, queue_path

    # 报告
    _print_report(confirmed, market_score)

    return confirmed


def _print_report(top, market_score):
    """生成完整报告"""
    print(f"\n{'='*70}")
    print(f"全A股缠论扫描报告 v3.0 | {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    print(f"筛选后 {len(top)}只 | 大盘 {market_score.get('total', 0)}/12分 {market_score.get('level', '')}")

    sig_dist = {}
    for r in top:
        sig_dist[r['signal_type']] = sig_dist.get(r['signal_type'], 0) + 1

    print(f"\n[信号分布]")
    for t in sorted(sig_dist, key=lambda x: SIGNAL_PRIORITY.get(x, 99)):
        print(f"  {t}: {sig_dist[t]}只")

    w_dist = {}
    for r in top:
        wd = r.get('weekly_dir', 'unknown')
        w_dist[wd] = w_dist.get(wd, 0) + 1
    print(f"\n[周线方向]")
    for wd in ['up', 'consolidation', 'down', 'unknown']:
        if wd in w_dist:
            tag = {'up': '↑上涨', 'consolidation': '→盘整', 'down': '↓下跌', 'unknown': '?未知'}[wd]
            print(f"  {tag}: {w_dist[wd]}只")

    current_type = None
    for r in top:
        st = r['signal_type']
        if st != current_type:
            current_type = st
            print(f"\n  {'─'*50}")
            print(f"  {st} | {sig_dist.get(st, 0)}只")
            print(f"  {'─'*50}")

        code = r['code']
        close = r['close']
        sig_date = r['signal_date']
        sig_price = r['signal_price']
        conf = r['confidence']
        stop = r.get('stop_loss', 0)
        dist = (close - sig_price) / sig_price * 100 if sig_price else 0
        reason = r.get('reason', '')[:50]

        dist_str = f"+{dist:.1f}%" if dist >= 0 else f"{dist:.1f}%"
        w_dir = r.get('weekly_dir', '?')
        w_tag = {'up': '↑', 'down': '↓', 'consolidation': '→'}.get(w_dir, '?')
        conf_tag = '✅' if r.get('confirmed') else '⚠️'

        m30_tag = ''
        if r.get('m30_status'):
            m30_tag = {'strong_entry': '🎯', 'double_entry': '🎯',
                       'divergence': '⏳', 'no_entry': '❌', 'none': ''}.get(r['m30_status'], '')

        print(f"  {conf_tag} {code:8s} {close:>8.2f} | {st} {sig_date} @{sig_price:.2f} stop={stop:.2f} | "
              f"距{dist_str} | conf={conf:.2f} | 周{w_tag} {m30_tag} | {reason}")

    print(f"\n{'='*70}")
    print(f"[风险提示] 基于缠论结构识别，不构成投资建议")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='全A股缠论扫描 v3.0')
    parser.add_argument('--top', type=int, default=80, help='Phase 2 TOP N')
    parser.add_argument('--m30', action='store_true', help='启用Phase 4 30分钟分析')
    parser.add_argument('--m30top', type=int, default=20, help='Phase 4 取TOP N做30分钟分析')
    args = parser.parse_args()

    run_full_scan_with_m30(top_n=args.top, m30_top=args.m30top, do_m30=args.m30)


if __name__ == '__main__':
    main()
