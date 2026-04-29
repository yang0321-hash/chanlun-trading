# -*- coding: utf-8 -*-
"""
30分钟API批量获取 + 精确入场分析
──────────────────────────────────
从 _m30_fetch_queue.json 读取候选列表
逐只通过 tdx_kline 工具获取30分钟K线
然后调用 M30EntrySignal 分析精确入场信号

使用方式:
  1. 先跑 all_a_scan_v3.py --m30 生成队列
  2. 再由 OpenClaw 调用本脚本的 fetch_m30_for_stock() 获取数据
  3. 最后调用 analyse_batch() 批量分析

或者直接调 fetch_and_analyse_all() 一键完成（需要API访问权限）
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

from tdx_day_reader import load_stock_klines
from kline_supplement import load_latest, supplement_klines
from chanlun_engine import KLine
from multi_level import multi_level_analyze, M30EntrySignal

M30_CACHE_DIR = '_m30_cache'
QUEUE_FILE = '_m30_fetch_queue.json'
RESULT_FILE = '_m30_results.json'


def ensure_cache_dir():
    os.makedirs(M30_CACHE_DIR, exist_ok=True)


def load_queue():
    """加载30分钟API队列"""
    if not os.path.exists(QUEUE_FILE):
        print(f'队列文件不存在: {QUEUE_FILE}')
        return []
    with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def cache_m30_data(code, setcode, m30_items):
    """缓存30分钟数据到本地文件"""
    ensure_cache_dir()
    path = os.path.join(M30_CACHE_DIR, f'{code}_{setcode}_m30.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m30_items, f, ensure_ascii=False)
    return path


def load_cached_m30(code, setcode):
    """加载缓存的30分钟数据"""
    path = os.path.join(M30_CACHE_DIR, f'{code}_{setcode}_m30.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def analyse_single(code, setcode, m30_raw=None):
    """分析单只股票的三级别联立+30分钟入场
    
    Args:
        code: 股票代码
        setcode: 市场代码
        m30_raw: 30分钟K线数据 [[open,high,low,close], ...] 或 None
    
    Returns:
        dict: 分析结果
    """
    latest = load_latest()

    # 日线
    raw = load_stock_klines(code, setcode, 500)
    if not raw:
        return None
    klines_raw = supplement_klines(code, setcode, raw, latest)
    daily_klines = [KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4]) for d in klines_raw]

    # 30分钟
    m30_klines = None
    if m30_raw and len(m30_raw) >= 20:
        m30_klines = [KLine(date='m30_{:03d}'.format(i),
                            open=r[0], high=r[1], low=r[2], close=r[3])
                      for i, r in enumerate(m30_raw)]

    result = multi_level_analyze(daily_klines, m30_klines)

    # 提取30分钟入场信号
    m30_entry = None
    if result.get('m30'):
        entry = result['m30'].get('entry', {})
        m30_entry = {
            'status': entry.get('status', 'none'),
            'detail': entry.get('detail', ''),
            'entry_price': entry.get('entry_price'),
            'stop_loss': entry.get('stop_loss'),
            'divergence_info': entry.get('divergence_info'),
            'fractal_info': entry.get('fractal_info'),
            'zg_violation': entry.get('zg_violation', False),
            'sell_type': entry.get('sell_type'),
            'sell_price': entry.get('sell_price'),
        }

    # 日线买点
    daily_buys = []
    for b in result.get('daily', {}).get('buys', []):
        daily_buys.append({
            'type': b['type'],
            'date': b['date'],
            'price': b['price'],
            'confidence': b['confidence'],
        })

    return {
        'code': code,
        'setcode': setcode,
        'summary': result.get('summary', ''),
        'daily_buys': daily_buys,
        'm30_entry': m30_entry,
        'close': klines_raw[-1][4] if klines_raw else 0,
    }


def analyse_batch():
    """批量分析所有有缓存数据的候选"""
    queue = load_queue()
    if not queue:
        print('无候选')
        return []

    results = []
    for item in queue:
        code = item['code']
        setcode = item['setcode']
        m30_raw = load_cached_m30(code, setcode)

        if m30_raw is None:
            print(f'  ⏭️ {code} 无30分钟缓存，跳过')
            continue

        r = analyse_single(code, setcode, m30_raw)
        if r:
            results.append(r)
            entry = r.get('m30_entry', {})
            status = entry.get('status', 'none')
            tag = {'strong_entry': '🎯', 'double_entry': '🎯',
                   'divergence': '⏳', 'no_entry': '❌'}.get(status, '  ')
            print(f'  {tag} {code} | {status} | {entry.get("detail", "")[:50]}')

    # 保存结果
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n分析完成: {len(results)}只 → {RESULT_FILE}')

    # 分类统计
    entry_counts = {}
    for r in results:
        st = r.get('m30_entry', {}).get('status', 'none')
        entry_counts[st] = entry_counts.get(st, 0) + 1
    print('\n30分钟入场统计:')
    for st, cnt in sorted(entry_counts.items()):
        print(f'  {st}: {cnt}只')

    return results


def generate_fetch_commands():
    """生成API获取命令（供外部调用）"""
    queue = load_queue()
    if not queue:
        print('无候选')
        return

    print(f'\n需要获取 {len(queue)} 只股票的30分钟K线:')
    print('─────────────────────────────────')
    for item in queue:
        code = item['code']
        setcode = item['setcode']
        print(f'  tdx_kline code="{code}" setcode="{setcode}" period="2" wantNum="200"')
    print('─────────────────────────────────')
    print(f'共 {len(queue)} 只')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyse', action='store_true', help='批量分析缓存数据')
    parser.add_argument('--commands', action='store_true', help='生成API获取命令')
    args = parser.parse_args()

    if args.analyse:
        analyse_batch()
    elif args.commands:
        generate_fetch_commands()
    else:
        print('用法:')
        print('  python m30_batch.py --commands   # 生成API获取命令')
        print('  python m30_batch.py --analyse    # 批量分析缓存数据')
