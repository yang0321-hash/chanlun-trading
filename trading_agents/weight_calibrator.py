#!/usr/bin/env python3
"""评分权重回测标定 — 用历史委员会数据网格搜索最优权重

用法:
  python trading_agents/weight_calibrator.py
"""
import sys
import os
import json
import glob
import itertools
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
          'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from dotenv import load_dotenv
load_dotenv()

from data.hybrid_source import HybridSource
from agents.scoring import calc_composite_score, make_decision, DEFAULT_WEIGHTS


def load_committee_history(days=30):
    """加载历史委员会数据"""
    results = []
    files = sorted(glob.glob('signals/investment_committee_*.json'), reverse=True)
    cutoff = datetime.now() - timedelta(days=days)

    for f in files:
        try:
            basename = os.path.basename(f)
            date_str = basename.split('_')[2]
            file_date = datetime.strptime(date_str, '%Y%m%d')
            if file_date < cutoff:
                continue

            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)

            for d in data.get('decisions', []):
                d['scan_date'] = date_str
                results.append(d)
        except Exception:
            continue

    return results


def calc_t1_return(hs, code, date_str):
    """计算T+1收益 — 本地TDX → 30min重构日线"""
    # 方法1: 本地TDX日线数据
    try:
        df = hs.get_kline(code, period='daily')
        if df is not None and len(df) >= 2:
            df_dates = df.index.strftime('%Y%m%d')
            for i, d in enumerate(df_dates):
                if d >= date_str:
                    if i + 1 < len(df):
                        entry = float(df['close'].iloc[i])
                        exit_p = float(df['close'].iloc[i + 1])
                        return (exit_p - entry) / entry * 100
                    break
    except Exception:
        pass

    # 方法2: 用30min数据重构日线 (Sina在线)
    try:
        df30 = hs.get_kline(code, period='30min')
        if df30 is not None and len(df30) >= 8:
            # 重构日线: 按日期分组取OHLC
            df30_copy = df30.copy()
            df30_copy['date'] = df30_copy.index.date
            daily = df30_copy.groupby('date').agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last'),
            )
            daily_dates = [d.strftime('%Y%m%d') for d in daily.index]
            for i, d in enumerate(daily_dates):
                if d >= date_str:
                    if i + 1 < len(daily):
                        entry = float(daily['close'].iloc[i])
                        exit_p = float(daily['close'].iloc[i + 1])
                        return (exit_p - entry) / entry * 100
                    break
    except Exception:
        pass

    return None


def evaluate_weights(decisions_with_returns, weights):
    """评估一组权重在历史数据上的表现

    Returns: dict with metrics
    """
    buy_rets = []
    correct_buys = 0
    total_buys = 0
    correct_rejects = 0
    total_rejects = 0

    for d in decisions_with_returns:
        t1 = d.get('_t1')
        if t1 is None:
            continue

        # 用新权重重新计算评分和决策
        composite = calc_composite_score(
            bull_confidence=d.get('bull_confidence', d.get('bull_score', 0.5)),
            bear_confidence=d.get('bear_confidence', d.get('bear_score', 0.3)),
            sentiment_score=d.get('sentiment_score', 0),
            sector_score=d.get('sector_score', 0),
            scanner_score=d.get('scanner_score', d.get('total_score', 50)),
            risk_score=d.get('risk_score', 0.3),
            weights=weights,
        )

        decision, _ = make_decision(
            composite_score=composite,
            bull_confidence=d.get('bull_confidence', 0.5),
            bear_confidence=d.get('bear_confidence', 0.3),
            risk_score=d.get('risk_score', 0.3),
        )

        if decision == 'buy':
            buy_rets.append(t1)
            total_buys += 1
            if t1 > 0:
                correct_buys += 1
        elif decision == 'reject':
            total_rejects += 1
            if t1 <= 0:
                correct_rejects += 1

    if not buy_rets:
        return {'win_rate': 0, 'avg_ret': 0, 'sharpe': 0, 'total_buys': 0}

    avg_ret = np.mean(buy_rets)
    win_rate = correct_buys / total_buys if total_buys > 0 else 0
    sharpe = avg_ret / np.std(buy_rets) if np.std(buy_rets) > 0 else 0

    return {
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'sharpe': sharpe,
        'total_buys': total_buys,
        'correct_rejects': correct_rejects,
        'total_rejects': total_rejects,
    }


def grid_search(decisions_with_returns):
    """网格搜索最优权重"""
    print('=== 评分权重网格搜索 ===\n')

    # 定义搜索范围
    bull_range = [0.20, 0.25, 0.30, 0.35, 0.40]
    bear_range = [0.15, 0.20, 0.25, 0.30]
    sent_range = [0.10, 0.15, 0.20]
    sector_range = [0.10, 0.15, 0.20]

    best_sharpe = -999
    best_weights = None
    best_result = None

    total = len(bull_range) * len(bear_range) * len(sent_range) * len(sector_range)
    count = 0

    for bull in bull_range:
        for bear in bear_range:
            for sent in sent_range:
                for sector in sector_range:
                    count += 1
                    # scanner和risk自动补齐 = 1 - others
                    remaining = 1.0 - bull - bear - sent - sector
                    if remaining < 0.05:
                        continue
                    scanner = remaining * 0.5
                    risk = remaining * 0.5

                    weights = {
                        'technical_bull': bull,
                        'technical_bear': bear,
                        'sentiment': sent,
                        'sector_rotation': sector,
                        'scanner_base': scanner,
                        'risk_adjustment': risk,
                    }

                    result = evaluate_weights(decisions_with_returns, weights)
                    if result['total_buys'] < 3:
                        continue

                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_weights = weights
                        best_result = result

    return best_weights, best_result


def run_calibration():
    """执行权重标定"""
    hs = HybridSource()

    print('[1] 加载历史数据...')
    decisions = load_committee_history(days=30)
    print(f'   {len(decisions)} 条历史决策')

    if len(decisions) < 10:
        print('   数据不足，需要至少10条决策记录')
        return

    print('[2] 计算T+1收益...')
    import time
    for i, d in enumerate(decisions):
        code = d.get('symbol', '')
        date_str = d.get('scan_date', '')
        d['_t1'] = calc_t1_return(hs, code, date_str)
        if d['_t1'] is None and i % 5 == 4:
            time.sleep(0.5)  # 每5次Sina请求后暂停，避免限流

    valid = [d for d in decisions if d.get('_t1') is not None]
    print(f'   {len(valid)} 条有T+1数据')

    if len(valid) < 10:
        print('   T+1数据不足')
        return

    # 当前权重基准
    print('\n[3] 当前权重评估:')
    baseline = evaluate_weights(valid, DEFAULT_WEIGHTS)
    print(f'   胜率: {baseline["win_rate"]:.1%} | 平均收益: {baseline["avg_ret"]:+.2f}% | '
          f'Sharpe: {baseline["sharpe"]:.2f} | BUY: {baseline["total_buys"]}只')
    print(f'   当前权重: {DEFAULT_WEIGHTS}')

    # 网格搜索
    print('\n[4] 网格搜索最优权重...')
    best_weights, best_result = grid_search(valid)

    if best_weights:
        print(f'\n=== 最优权重 ===')
        print(f'  胜率: {best_result["win_rate"]:.1%} | 平均收益: {best_result["avg_ret"]:+.2f}% | '
              f'Sharpe: {best_result["sharpe"]:.2f} | BUY: {best_result["total_buys"]}只')
        print(f'  权重:')
        for k, v in best_weights.items():
            print(f'    {k}: {v:.2f}')

        # 对比
        print(f'\n=== 对比 ===')
        print(f'  {"":>15} {"当前":>8} {"最优":>8} {"变化":>8}')
        for k in DEFAULT_WEIGHTS:
            old = DEFAULT_WEIGHTS[k]
            new = best_weights.get(k, old)
            diff = new - old
            print(f'  {k:>15} {old:>8.2f} {new:>8.2f} {diff:>+8.2f}')

        # 保存结果
        output = {
            'calibration_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'baseline': {k: float(v) for k, v in baseline.items()},
            'optimal_weights': {k: round(v, 3) for k, v in best_weights.items()},
            'optimal_metrics': {k: float(v) for k, v in best_result.items()},
        }
        os.makedirs('signals', exist_ok=True)
        with open('signals/weight_calibration.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f'\n  结果已保存: signals/weight_calibration.json')
    else:
        print('  未找到更优权重')


if __name__ == '__main__':
    run_calibration()
