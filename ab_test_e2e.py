"""策略组合端到端A/B测试 — 验证全部改进的累积效果

对比4种配置:
  A(基线): 无过滤, 旧出场参数 (trailing 5%, 止盈5/10/15%, 锁利30%)
  B(+信号过滤): MIN_SCORE=80 + 估值过滤, 旧出场参数
  C(+新出场): 无过滤, 新出场参数 (trailing 3%, 止盈8/15/25%, 锁利50%, 阶梯保本)
  D(全组合): 信号过滤 + 新出场参数 (当前生产配置)

信号过滤模拟:
  - MIN_SCORE: 用confidence做代理, 过滤低置信度信号
  - 估值过滤: 价格>90%历史分位不买
  - 风险惩罚: 简化为低confidence信号额外扣分

用法:
  python ab_test_e2e.py
  python ab_test_e2e.py --stocks 50
  python ab_test_e2e.py --quick   # 10只
"""
import sys, os, json, time, argparse, random
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from loguru import logger
logger.remove()

INITIAL_CAPITAL = 1_000_000
COMMISSION_BUY = 0.0003
COMMISSION_SELL = 0.0013

# 旧出场参数
OLD_EXIT = {
    'trailing_activation': 0.05,
    'profit_targets': [(0.05, 0.3), (0.10, 0.3), (0.15, 0.4)],
    'dynamic_targets_strong': [(0.10, 0.10), (0.20, 0.15), (0.35, 0.25)],
    'dynamic_targets_normal': [(0.05, 0.3), (0.10, 0.3), (0.15, 0.4)],
    'dynamic_targets_weak': [(0.03, 0.4), (0.06, 0.3), (0.10, 0.3)],
    'lock_ratio': 0.30,
    'stepped_breakeven': False,
}

# 新出场参数
NEW_EXIT = {
    'trailing_activation': 0.03,
    'profit_targets': [(0.08, 0.2), (0.15, 0.2), (0.25, 0.3)],
    'dynamic_targets_strong': [(0.15, 0.15), (0.30, 0.20), (0.50, 0.15)],
    'dynamic_targets_normal': [(0.08, 0.2), (0.15, 0.2), (0.25, 0.3)],
    'dynamic_targets_weak': [(0.03, 0.4), (0.06, 0.3), (0.10, 0.3)],
    'lock_ratio': 0.50,
    'stepped_breakeven': True,
}


def load_stock_data(code):
    fp = f'test_output/{code}.day.json'
    if not os.path.exists(fp):
        return None
    with open(fp, 'r') as f:
        data = json.load(f)
    if len(data) < 200:
        return None
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df['volume'] = df['volume'].astype(float)
    if len(df) > 500:
        df = df.iloc[-500:]
    return df[['open', 'high', 'low', 'close', 'volume']]


def select_test_stocks(n=50):
    files = [f for f in os.listdir('test_output') if f.endswith('.day.json')]
    valid = []
    for f in files:
        code = f.replace('.day.json', '')
        if code.startswith(('sh6', 'sz0', 'sz3', 'sz002')):
            valid.append(code)
    random.seed(42)
    selected = random.sample(valid, min(n * 3, len(valid)))
    result = []
    for code in selected:
        df = load_stock_data(code)
        if df is not None and len(df) >= 300:
            result.append(code)
            if len(result) >= n:
                break
    return result


def generate_signals(stock_codes):
    from backtest.engine import BacktestEngine, BacktestConfig
    from strategies.chan_strategy import ChanLunStrategy

    all_signals = {}
    for i, code in enumerate(stock_codes):
        df = load_stock_data(code)
        if df is None:
            continue
        print(f'  [{i+1}/{len(stock_codes)}] {code} ({len(df)} bars)...', end='', flush=True)
        t0 = time.time()
        try:
            strategy = ChanLunStrategy(
                name=f'ChanLun_{code}',
                use_macd=True,
                use_volume_price=True,
                use_triple_barrier=True,
            )
            engine = BacktestEngine(BacktestConfig(
                initial_capital=INITIAL_CAPITAL,
                commission_buy=COMMISSION_BUY,
                commission_sell=COMMISSION_SELL,
            ))
            engine.add_data(code, df)
            engine.set_strategy(strategy)
            results = engine.run()

            signals = []
            for sig in engine.signals_generated:
                signals.append({
                    'date': str(sig.datetime.date()) if hasattr(sig.datetime, 'date') else str(sig.datetime)[:10],
                    'type': sig.signal_type.value,
                    'price': sig.price,
                    'reason': sig.reason,
                    'confidence': sig.confidence,
                })
            n_buy = sum(1 for s in signals if s['type'] == 'buy')
            n_sell = sum(1 for s in signals if s['type'] == 'sell')
            ret = results.get('total_return', 0)
            print(f' {n_buy}B/{n_sell}S, ret={ret:.1%} ({time.time()-t0:.1f}s)')
            if n_buy > 0:
                all_signals[code] = signals
        except Exception as e:
            print(f' ERROR: {e}')
    return all_signals


def apply_signal_filters(all_signals, stock_data, mode):
    """信号过滤: 根据mode决定过滤策略"""
    if mode in ('A', 'C'):
        # 无信号过滤
        return all_signals

    filtered = {}
    filtered_count = 0
    for code, signals in all_signals.items():
        new_sigs = []
        for sig in signals:
            if sig['type'] == 'buy':
                # 1. MIN_SCORE过滤: confidence < 0.4 视为低质量 (近似score<80)
                if sig['confidence'] < 0.4:
                    filtered_count += 1
                    continue

                # 2. 估值过滤: 价格>90%分位不买
                if code in stock_data:
                    df = stock_data[code]
                    closes = df['close'].values[-252:]
                    if len(closes) >= 60:
                        percentile = np.mean(closes < sig['price'])
                        if percentile > 0.90:
                            filtered_count += 1
                            continue

            new_sigs.append(sig)
        if any(s['type'] == 'buy' for s in new_sigs):
            filtered[code] = new_sigs

    print(f'    信号过滤: {sum(len(v) for v in all_signals.values())}→{sum(len(v) for v in filtered.values())} (过滤{filtered_count}条)')
    return filtered


def simulate(all_signals, stock_data, exit_config):
    all_dates = set()
    for code, df in stock_data.items():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    positions = {}
    cash = INITIAL_CAPITAL
    equity_curve = []
    trades = []

    stepped = exit_config['stepped_breakeven']
    lock_ratio = exit_config['lock_ratio']
    trail_act = exit_config['trailing_activation']

    for date in all_dates:
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)[:10]
        current_prices = {}
        for code, df in stock_data.items():
            if date in df.index:
                current_prices[code] = float(df.loc[date, 'close'])

        if not current_prices:
            continue

        for code, signals in all_signals.items():
            if code not in current_prices:
                continue
            price = current_prices[code]
            day_signals = [s for s in signals if s['date'] == date_str]

            for sig in day_signals:
                if sig['type'] == 'buy' and code not in positions:
                    n_total = len(stock_data)
                    alloc = min(INITIAL_CAPITAL / n_total, cash * 0.95)
                    shares = int(alloc / price / 100) * 100
                    if shares < 100:
                        continue
                    cost = price * shares * (1 + COMMISSION_BUY)
                    if cost > cash:
                        shares = int(cash / (price * (1 + COMMISSION_BUY)) / 100) * 100
                        if shares < 100:
                            continue
                        cost = price * shares * (1 + COMMISSION_BUY)
                    cash -= cost
                    positions[code] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date_str,
                        'highest': price,
                        'stop_loss': price * 0.95,
                        'partial_sold': 0.0,
                        'confidence': sig['confidence'],
                    }

                elif sig['type'] == 'sell' and code in positions:
                    pos = positions.pop(code)
                    revenue = price * pos['shares'] * (1 - COMMISSION_SELL)
                    profit = revenue - pos['entry_price'] * pos['shares'] * (1 + COMMISSION_BUY)
                    cash += revenue
                    trades.append({
                        'code': code, 'entry_date': pos['entry_date'],
                        'exit_date': date_str, 'entry_price': pos['entry_price'],
                        'exit_price': price, 'profit': profit,
                        'return': profit / (pos['entry_price'] * pos['shares']),
                        'reason': sig['reason'],
                        'confidence': pos.get('confidence', 0),
                    })

        # 出场检查
        for code in list(positions.keys()):
            if code not in current_prices:
                continue
            pos = positions[code]
            price = current_prices[code]
            entry = pos['entry_price']
            profit_pct = (price - entry) / entry

            if price > pos['highest']:
                pos['highest'] = price

            # 阶梯保本
            if stepped:
                if profit_pct >= 0.15:
                    pos['stop_loss'] = max(pos['stop_loss'], entry * 1.08)
                elif profit_pct >= 0.08:
                    pos['stop_loss'] = max(pos['stop_loss'], entry * 1.03)
                elif profit_pct >= 0.03:
                    pos['stop_loss'] = max(pos['stop_loss'], entry)

            # 固定止损
            if price <= pos['stop_loss']:
                revenue = price * pos['shares'] * (1 - COMMISSION_SELL)
                profit = revenue - entry * pos['shares'] * (1 + COMMISSION_BUY)
                cash += revenue
                trades.append({
                    'code': code, 'entry_date': pos['entry_date'],
                    'exit_date': date_str, 'entry_price': entry,
                    'exit_price': price, 'profit': profit,
                    'return': profit / (entry * pos['shares']),
                    'reason': f'止损@{pos["stop_loss"]:.2f}',
                    'confidence': pos.get('confidence', 0),
                })
                del positions[code]
                continue

            # ATR跟踪止损
            if profit_pct >= trail_act:
                trail_stop = pos['highest'] * (1 - 0.08)
                if trail_stop > pos['stop_loss']:
                    pos['stop_loss'] = trail_stop

            # 分批止盈
            targets = exit_config['profit_targets']
            remaining_pct = 1.0 - pos['partial_sold']
            for target_pct, sell_ratio in targets:
                if profit_pct >= target_pct and pos['partial_sold'] < target_pct:
                    sell_shares = int(pos['shares'] * sell_ratio * remaining_pct / 100) * 100
                    if sell_shares >= 100:
                        revenue = price * sell_shares * (1 - COMMISSION_SELL)
                        cash += revenue
                        pos['shares'] -= sell_shares
                        pos['partial_sold'] += sell_ratio * remaining_pct
                        new_stop = entry * (1 + profit_pct * lock_ratio)
                        pos['stop_loss'] = max(pos['stop_loss'], new_stop)

            if pos['shares'] < 100:
                revenue = price * pos['shares'] * (1 - COMMISSION_SELL)
                cash += revenue
                del positions[code]

        equity = cash + sum(
            current_prices.get(c, 0) * p['shares']
            for c, p in positions.items() if c in current_prices
        )
        equity_curve.append((date_str, equity))

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_equity': equity_curve[-1][1] if equity_curve else INITIAL_CAPITAL,
    }


def calc_metrics(result):
    trades = result['trades']
    equity = result['equity_curve']
    if not equity:
        return {}

    total_return = (equity[-1][1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = len(equity) / 252
    annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    peak = INITIAL_CAPITAL
    max_dd = 0
    for _, eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    if len(equity) > 20:
        daily_rets = [(equity[i][1] - equity[i-1][1]) / equity[i-1][1]
                       for i in range(1, len(equity)) if equity[i-1][1] > 0]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    else:
        sharpe = 0

    if trades:
        wins = [t for t in trades if t['profit'] > 0]
        losses = [t for t in trades if t['profit'] <= 0]
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t['return'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['return']) for t in losses]) if losses else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        avg_trade_ret = np.mean([t['return'] for t in trades])
        max_loss = min((t['return'] for t in trades), default=0)
        big_loss_rate = sum(1 for t in trades if t['return'] < -0.05) / len(trades)
    else:
        win_rate = profit_loss_ratio = avg_trade_ret = max_loss = big_loss_rate = 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': len(trades),
        'avg_trade_return': avg_trade_ret,
        'max_single_loss': max_loss,
        'big_loss_rate': big_loss_rate,
        'final_equity': equity[-1][1],
    }


def main():
    parser = argparse.ArgumentParser(description='策略组合端到端A/B测试')
    parser.add_argument('--quick', action='store_true', help='10只快速验证')
    parser.add_argument('--stocks', type=int, default=50, help='股票数量')
    args = parser.parse_args()
    n_stocks = 10 if args.quick else args.stocks

    configs = [
        ('A  基线',          'A', OLD_EXIT, '无过滤 + 旧出场'),
        ('B  +信号过滤',     'B', OLD_EXIT, 'MIN_SCORE80+估值过滤 + 旧出场'),
        ('C  +新出场',       'C', NEW_EXIT, '无过滤 + 新出场'),
        ('D  全组合',        'D', NEW_EXIT, '信号过滤 + 新出场 (生产配置)'),
    ]

    print('=' * 80)
    print('策略组合 端到端A/B测试')
    print('=' * 80)
    for label, mode, exit_cfg, desc in configs:
        print(f'  {label}: {desc}')
    print(f'  股票数: {n_stocks}')
    print()

    # 选股
    print('[1] 选股...')
    stocks = select_test_stocks(n_stocks)
    print(f'  已选 {len(stocks)} 只')

    # 生成信号
    print(f'\n[2] 生成买卖信号...')
    all_signals = generate_signals(stocks)
    total_buys = sum(1 for v in all_signals.values() for s in v if s['type'] == 'buy')
    print(f'  共 {total_buys} 笔买入信号, {len(all_signals)}只')

    if total_buys < 10:
        print('信号太少')
        return

    # 加载数据
    stock_data = {}
    for code in all_signals:
        df = load_stock_data(code)
        if df is not None:
            stock_data[code] = df
    print(f'  数据: {len(stock_data)}只')

    # A/B测试
    print(f'\n[3] A/B回测...')
    results = {}
    for label, mode, exit_cfg, desc in configs:
        print(f'\n  {label} ({desc}):')
        filtered = apply_signal_filters(all_signals, stock_data, mode)
        print(f'    回测中...', end='', flush=True)
        t0 = time.time()
        result = simulate(filtered, stock_data, exit_cfg)
        metrics = calc_metrics(result)
        results[label] = {'metrics': metrics, 'trades': result['trades']}
        print(f' ret={metrics.get("total_return",0):.1%}, '
              f'sharpe={metrics.get("sharpe",0):.2f}, '
              f'dd={metrics.get("max_drawdown",0):.1%}, '
              f'win={metrics.get("win_rate",0):.1%}, '
              f'trades={metrics.get("total_trades",0)} '
              f'({time.time()-t0:.1f}s)')

    # 对比表
    print('\n' + '=' * 100)
    print('端到端A/B测试结果')
    print('=' * 100)
    print(f'{"版本":<16s} {"年化":>7s} {"总收益":>7s} {"Sharpe":>7s} '
          f'{"最大回撤":>7s} {"胜率":>6s} {"盈亏比":>6s} {"均笔收益":>7s} '
          f'{"大亏率":>6s} {"交易数":>5s}')
    print('-' * 100)

    for label, _, _, _ in configs:
        m = results[label]['metrics']
        print(f'{label:<16s} '
              f'{m.get("annual_return",0):>6.1%} '
              f'{m.get("total_return",0):>6.1%} '
              f'{m.get("sharpe",0):>6.2f} '
              f'{m.get("max_drawdown",0):>6.1%} '
              f'{m.get("win_rate",0):>5.1%} '
              f'{m.get("profit_loss_ratio",0):>5.2f} '
              f'{m.get("avg_trade_return",0):>6.2%} '
              f'{m.get("big_loss_rate",0):>5.1%} '
              f'{m.get("total_trades",0):>4d}')

    # vs基线差异
    baseline = results[configs[0][0]]['metrics']
    print('\n' + '-' * 100)
    print('vs A(基线) 差异:')
    print(f'{"版本":<16s} {"收益差":>8s} {"Sharpe差":>9s} {"回撤差":>8s} {"胜率差":>7s} {"盈亏比差":>8s} {"大亏率差":>8s}')
    print('-' * 100)
    for label, _, _, _ in configs[1:]:
        m = results[label]['metrics']
        diffs = []
        for key in ['total_return', 'sharpe', 'max_drawdown', 'win_rate', 'profit_loss_ratio', 'big_loss_rate']:
            d = m.get(key, 0) - baseline.get(key, 0)
            diffs.append(d)
        print(f'{label:<16s} '
              f'{diffs[0]:>+7.2%} '
              f'{diffs[1]:>+8.2f} '
              f'{diffs[2]:>+7.2%} '
              f'{diffs[3]:>+6.1%} '
              f'{diffs[4]:>+7.2f} '
              f'{diffs[5]:>+7.1%}')

    # 增量贡献分析
    print('\n' + '=' * 100)
    print('增量贡献分析:')
    a = results[configs[0][0]]['metrics']
    b = results[configs[1][0]]['metrics']
    c = results[configs[2][0]]['metrics']
    d = results[configs[3][0]]['metrics']

    ret_signal = b['total_return'] - a['total_return']
    ret_exit = c['total_return'] - a['total_return']
    ret_synergy = d['total_return'] - a['total_return'] - ret_signal - ret_exit

    sh_signal = b['sharpe'] - a['sharpe']
    sh_exit = c['sharpe'] - a['sharpe']
    sh_synergy = d['sharpe'] - a['sharpe'] - sh_signal - sh_exit

    dd_signal = b['max_drawdown'] - a['max_drawdown']
    dd_exit = c['max_drawdown'] - a['max_drawdown']
    dd_synergy = d['max_drawdown'] - a['max_drawdown'] - dd_signal - dd_exit

    print(f'  信号过滤贡献: 收益{ret_signal:+.2%}, Sharpe{sh_signal:+.2f}, 回撤{dd_signal:+.2%}')
    print(f'  出场参数贡献: 收益{ret_exit:+.2%}, Sharpe{sh_exit:+.2f}, 回撤{dd_exit:+.2%}')
    print(f'  协同效应:     收益{ret_synergy:+.2%}, Sharpe{sh_synergy:+.2f}, 回撤{dd_synergy:+.2%}')
    print(f'  总改进(D-A):  收益{d["total_return"]-a["total_return"]:+.2%}, '
          f'Sharpe{d["sharpe"]-a["sharpe"]:+.2f}, 回撤{d["max_drawdown"]-a["max_drawdown"]:+.2%}')

    # 最终结论
    print('\n' + '=' * 100)
    total_ret_diff = d['total_return'] - a['total_return']
    total_sharpe_diff = d['sharpe'] - a['sharpe']
    total_dd_diff = d['max_drawdown'] - a['max_drawdown']
    total_bl_diff = d['big_loss_rate'] - a['big_loss_rate']

    if total_ret_diff > 0.05 and total_sharpe_diff > 0.2:
        verdict = '全组合显著优于基线'
    elif total_ret_diff > 0 or total_sharpe_diff > 0:
        verdict = '全组合略优于基线'
    elif total_ret_diff < -0.05:
        verdict = '基线更好，建议回滚'
    else:
        verdict = '无显著差异'

    print(f'结论: {verdict}')
    print(f'  总收益差: {total_ret_diff:+.2%} | Sharpe差: {total_sharpe_diff:+.2f} | '
          f'回撤差: {total_dd_diff:+.2%} | 大亏率差: {total_bl_diff:+.1%}')
    print()


if __name__ == '__main__':
    main()
