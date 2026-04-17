"""
投资委员会回测 — 验证委员会决策的历史胜率

原理:
  1. 加载历史扫描结果（T日候选股）
  2. 用T日之前的K线数据运行投资委员会
  3. 对比T+1~T+5/T+10的实际涨跌
  4. 统计 buy/hold/reject 三组的平均收益
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 清除代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)


def fetch_sina_daily(code: str, datalen: int = 500) -> pd.DataFrame:
    """从新浪获取日线数据"""
    import requests
    session = requests.Session()
    session.trust_env = False

    prefix = 'sz' if code.startswith(('0', '3')) else 'sh'
    url = f'https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/CN_MarketDataService.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen={datalen}'
    resp = session.get(url, timeout=15)
    match = re.search(r'callback\((.*)\)', resp.text)
    if not match:
        return pd.DataFrame()

    data = json.loads(match.group(1))
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['day'])
    df = df.set_index('datetime')
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_all_daily(codes: List[str]) -> Dict[str, pd.DataFrame]:
    """批量获取日线"""
    result = {}
    for i, code in enumerate(codes):
        try:
            df = fetch_sina_daily(code)
            if len(df) > 60:
                result[code] = df
                print(f'  [{i+1}/{len(codes)}] {code}: {len(df)} bars OK')
            else:
                print(f'  [{i+1}/{len(codes)}] {code}: insufficient data')
        except Exception as e:
            print(f'  [{i+1}/{len(codes)}] {code}: error {e}')
    return result


def load_sector_map() -> Dict[str, str]:
    """加载行业映射"""
    for sp in ['chanlun_system/full_sector_map.json', 'chanlun_system/thshy_sector_map.json']:
        if os.path.exists(sp):
            with open(sp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'stock_to_sector' in data:
                    return data['stock_to_sector']
                elif isinstance(data, dict):
                    return data
    return {}


def build_candidate_from_scan(record: dict) -> dict:
    """从扫描记录构建候选股"""
    # 兼容多种扫描格式
    code = record.get('code', record.get('symbol', ''))
    # 去掉前缀 SZ/SH
    code = re.sub(r'^(SZ|SH|sz|sh)', '', code)

    return {
        'code': code,
        'name': record.get('name', code),
        'entry_price': float(record.get('entry_price', record.get('price', 0))),
        'stop_price': float(record.get('stop_price', record.get('stop_loss', 0))),
        'total_score': float(record.get('total_score', record.get('confidence', 0)) * 100
                           if record.get('confidence', 0) <= 1
                           else record.get('total_score', 50)),
        'risk_reward': float(record.get('risk_reward', 2.0)),
        'sector': record.get('sector', ''),
        '2buy_date': record.get('2buy_date', ''),
    }


def run_committee_on_date(candidates: List[dict], daily_map: Dict[str, pd.DataFrame],
                          sector_map: Dict[str, str], as_of_date: str) -> List[dict]:
    """在指定日期运行投资委员会"""
    from agents.committee_agents import (
        CommitteeContext, analyze_chanlun_structure,
        BullAnalyst, BearAnalyst, SentimentAnalyzer,
        SectorRotation, RiskManager, FundManager,
    )

    results = []
    as_of = pd.Timestamp(as_of_date)

    for cand in candidates:
        code = cand['code']
        if code not in daily_map:
            continue

        df_full = daily_map[code]
        # 截止到as_of_date（模拟T日决策）
        df = df_full[df_full.index <= as_of]

        if len(df) < 60:
            continue

        last_close = float(df['close'].iloc[-1])
        entry = cand.get('entry_price', last_close) or last_close
        stop = cand.get('stop_price', entry * 0.95) or entry * 0.95

        # 缠论分析
        chanlun = analyze_chanlun_structure(df, cand)

        ctx = CommitteeContext(
            symbol=code,
            name=cand.get('name', code),
            sector=sector_map.get(code, cand.get('sector', '')),
            df_daily=df,
            entry_price=entry,
            stop_price=stop,
            scanner_score=cand.get('total_score', 50),
            risk_reward=cand.get('risk_reward', 2.0),
            sector_momentum={},
            sector_map=sector_map,
            portfolio_state={'positions': [], 'capital': 1000000},
            chanlun=chanlun,
        )

        bull_arg = BullAnalyst().analyze(ctx)
        bear_arg = BearAnalyst().analyze(ctx)
        sent_arg = SentimentAnalyzer().analyze(ctx)
        sec_arg = SectorRotation(sector_map, {}).analyze(ctx)
        risk = RiskManager().evaluate(ctx, bull_arg, bear_arg, sent_arg, sec_arg)
        result = FundManager().decide(ctx, bull_arg, bear_arg, sent_arg, sec_arg, risk)

        result['code'] = code
        result['close_on_date'] = last_close
        results.append(result)

    return results


def calc_future_returns(code: str, daily_map: Dict[str, pd.DataFrame],
                        as_of_date: str, holding_days: List[int] = [1, 3, 5, 10]) -> Dict[int, float]:
    """计算T+N收益率"""
    df = daily_map.get(code)
    if df is None:
        return {}

    as_of = pd.Timestamp(as_of_date)
    df_after = df[df.index > as_of]

    if len(df_after) == 0:
        return {}

    entry_price = float(df[df.index <= as_of]['close'].iloc[-1])
    returns = {}

    for n in holding_days:
        if n <= len(df_after):
            future_price = float(df_after['close'].iloc[n - 1])
            returns[n] = (future_price - entry_price) / entry_price * 100

    return returns


def backtest_committee(scan_files: List[str] = None):
    """主回测流程"""
    if scan_files is None:
        # 自动找历史扫描
        scan_files = sorted([
            f for f in os.listdir('signals')
            if f.startswith('scan_') and f.endswith('.json') and '202604' in f
        ])

    print(f'找到 {len(scan_files)} 个扫描文件')
    sector_map = load_sector_map()
    print(f'行业映射: {len(sector_map)} 只')

    all_results = []

    for scan_file in scan_files:
        path = os.path.join('signals', scan_file)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 解析扫描结果
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get('top_n', data.get('all_signals', data.get('signals', [])))
            if not records and 'candidates' in data:
                records = data['candidates']
        else:
            continue

        if not records:
            continue

        # 确定日期
        scan_time = ''
        if isinstance(data, dict):
            scan_time = data.get('scan_time', '')
        if not scan_time and records:
            scan_time = records[0].get('timestamp', records[0].get('scan_time', ''))

        # 解析日期
        as_of_date = ''
        for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%d', '%Y%m%d_%H%M']:
            try:
                dt = datetime.strptime(scan_time[:16] if len(scan_time) >= 16 else scan_time, fmt)
                as_of_date = dt.strftime('%Y-%m-%d')
                break
            except (ValueError, TypeError):
                continue

        if not as_of_date:
            # 从文件名提取
            m = re.search(r'(\d{8})', scan_file)
            if m:
                d = m.group(1)
                as_of_date = f'{d[:4]}-{d[4:6]}-{d[6:8]}'

        if not as_of_date:
            continue

        print(f'\n{"="*60}')
        print(f'扫描文件: {scan_file} | 决策日期: {as_of_date} | 候选: {len(records)}')

        # 构建候选
        candidates = []
        for r in records[:30]:
            try:
                candidates.append(build_candidate_from_scan(r))
            except Exception:
                continue

        if not candidates:
            print('  无有效候选')
            continue

        codes = [c['code'] for c in candidates]
        print(f'  获取 {len(codes)} 只股票数据...')

        daily_map = fetch_all_daily(codes)

        print(f'  运行投资委员会 (as of {as_of_date})...')
        committee_results = run_committee_on_date(candidates, daily_map, sector_map, as_of_date)

        print(f'  评估完成: {len(committee_results)} 只')

        # 计算未来收益
        for r in committee_results:
            code = r['code']
            rets = calc_future_returns(code, daily_map, as_of_date)
            r['future_returns'] = rets
            r['as_of_date'] = as_of_date
            all_results.append(r)

    # === 汇总统计 ===
    if not all_results:
        print('\n无回测结果')
        return

    print(f'\n{"="*70}')
    print(f'投资委员会回测汇总')
    print(f'{"="*70}')
    print(f'总评估: {len(all_results)} 只')

    for decision in ['buy', 'hold', 'reject']:
        group = [r for r in all_results if r['decision'] == decision]
        if not group:
            continue

        n = len(group)
        avg_score = np.mean([r['composite_score'] for r in group])

        print(f'\n--- {decision.upper()} ({n}只) | 平均评分: {avg_score:.0f} ---')
        print(f'{"代码":<8} {"评分":>5} {"T+1":>7} {"T+3":>7} {"T+5":>7} {"T+10":>7} {"关键因素"}')
        print('-' * 70)

        for r in sorted(group, key=lambda x: x['composite_score'], reverse=True)[:15]:
            rets = r.get('future_returns', {})
            t1 = f'{rets[1]:+.1f}%' if 1 in rets else 'N/A'
            t3 = f'{rets[3]:+.1f}%' if 3 in rets else 'N/A'
            t5 = f'{rets[5]:+.1f}%' if 5 in rets else 'N/A'
            t10 = f'{rets[10]:+.1f}%' if 10 in rets else 'N/A'
            factors = ', '.join(r.get('key_factors', [])[:2])
            print(f'{r["code"]:<8} {r["composite_score"]:>5.0f} {t1:>7} {t3:>7} {t5:>7} {t10:>7} {factors}')

        # 统计
        for period in [1, 3, 5, 10]:
            rets = [r['future_returns'][period] for r in group if period in r.get('future_returns', {})]
            if rets:
                avg = np.mean(rets)
                win = sum(1 for x in rets if x > 0) / len(rets) * 100
                print(f'  T+{period}: avg={avg:+.1f}% | win_rate={win:.0f}% ({len(rets)}只)')

    # 保存详细结果
    output = []
    for r in all_results:
        row = {
            'code': r['code'],
            'as_of_date': r['as_of_date'],
            'decision': r['decision'],
            'score': round(r['composite_score'], 1),
            'bull': round(r['bull_confidence'], 2),
            'bear': round(r['bear_confidence'], 2),
            'risk_level': r['risk_level'],
            'close_on_date': r.get('close_on_date', 0),
            'key_factors': r.get('key_factors', []),
        }
        for period in [1, 3, 5, 10]:
            if period in r.get('future_returns', {}):
                row[f'T+{period}'] = round(r['future_returns'][period], 2)
        output.append(row)

    out_file = f'signals/backtest_committee_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n回测结果已保存: {out_file}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='投资委员会回测')
    parser.add_argument('--files', nargs='*', help='指定扫描文件')
    args = parser.parse_args()

    if args.files:
        backtest_committee(args.files)
    else:
        backtest_committee()
