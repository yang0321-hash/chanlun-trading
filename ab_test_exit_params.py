"""出场参数A/B测试 — 验证新参数(阶梯保本+锁利+宽止盈)的实际改进

对比:
  A(旧参数): trailing 5%, 止盈 5%/10%/15%, 利润锁定30%
  B(新参数): trailing 3%, 止盈 8%/15%/25%, 利润锁定50%, 阶梯保本

用法:
  python ab_test_exit_params.py
  python ab_test_exit_params.py --stocks 20
  python ab_test_exit_params.py --quick   # 10只快速验证
"""
import sys, os, json, time, argparse, random
import numpy as np
import pandas as pd
from copy import deepcopy

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from loguru import logger
logger.remove()

INITIAL_CAPITAL = 1_000_000
COMMISSION_BUY = 0.0003
COMMISSION_SELL = 0.0013

# 旧出场参数 (HEAD~1)
OLD_EXIT_CONFIG = {
    'trailing_activation': 0.05,
    'profit_targets': [(0.05, 0.3), (0.10, 0.3), (0.15, 0.4)],
    'dynamic_targets_strong': [(0.10, 0.10), (0.20, 0.15), (0.35, 0.25)],
    'dynamic_targets_normal': [(0.05, 0.3), (0.10, 0.3), (0.15, 0.4)],
    'dynamic_targets_weak': [(0.03, 0.4), (0.06, 0.3), (0.10, 0.3)],
    'lock_ratio': 0.30,
    'stepped_breakeven': False,
}

# 新出场参数 (当前)
NEW_EXIT_CONFIG = {
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
    # 只取最近2年
    if len(df) > 500:
        df = df.iloc[-500:]
    return df[['open', 'high', 'low', 'close', 'volume']]


def select_test_stocks(n=30):
    """选择有足够数据的测试股票"""
    files = [f for f in os.listdir('test_output') if f.endswith('.day.json')]
    valid = []
    for f in files:
        code = f.replace('.day.json', '')
        if code.startswith(('sh6', 'sz0', 'sz3', 'sz002')):
            valid.append(code)

    random.seed(42)
    selected = random.sample(valid, min(n * 3, len(valid)))

    # 验证数据量
    result = []
    for code in selected:
        df = load_stock_data(code)
        if df is not None and len(df) >= 300:
            result.append(code)
            if len(result) >= n:
                break
    return result


def generate_signals(stock_codes):
    """Phase 1: 用ChanLunStrategy生成买卖信号"""
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


def simulate_with_exit_params(all_signals, stock_data, exit_config):
    """Phase 2: 用指定出场参数模拟交易"""
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

        # 处理信号
        for code, signals in all_signals.items():
            if code not in current_prices:
                continue
            price = current_prices[code]
            day_signals = [s for s in signals if s['date'] == date_str]

            for sig in day_signals:
                if sig['type'] == 'buy' and code not in positions:
                    # 计算仓位 (等权)
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
                        'stop_loss': price * 0.95,  # 初始5%止损
                        'partial_sold': 0.0,  # 已卖出比例
                    }

                elif sig['type'] == 'sell' and code in positions:
                    pos = positions.pop(code)
                    revenue = price * pos['shares'] * (1 - COMMISSION_SELL)
                    profit = revenue - pos['entry_price'] * pos['shares'] * (1 + COMMISSION_BUY)
                    cash += revenue
                    trades.append({
                        'code': code,
                        'entry_date': pos['entry_date'],
                        'exit_date': date_str,
                        'entry_price': pos['entry_price'],
                        'exit_price': price,
                        'profit': profit,
                        'return': profit / (pos['entry_price'] * pos['shares']),
                        'reason': sig['reason'],
                    })

        # 出场逻辑: 逐持仓检查
        for code in list(positions.keys()):
            if code not in current_prices:
                continue
            pos = positions[code]
            price = current_prices[code]
            entry = pos['entry_price']
            profit_pct = (price - entry) / entry

            # 更新最高价
            if price > pos['highest']:
                pos['highest'] = price

            # 阶梯保本 (新参数)
            if stepped:
                if profit_pct >= 0.15:
                    new_stop = entry * 1.08
                    pos['stop_loss'] = max(pos['stop_loss'], new_stop)
                elif profit_pct >= 0.08:
                    new_stop = entry * 1.03
                    pos['stop_loss'] = max(pos['stop_loss'], new_stop)
                elif profit_pct >= 0.03:
                    pos['stop_loss'] = max(pos['stop_loss'], entry)

            # 固定止损
            if price <= pos['stop_loss']:
                revenue = price * pos['shares'] * (1 - COMMISSION_SELL)
                profit = revenue - entry * pos['shares'] * (1 + COMMISSION_BUY)
                cash += revenue
                trades.append({
                    'code': code,
                    'entry_date': pos['entry_date'],
                    'exit_date': date_str,
                    'entry_price': entry,
                    'exit_price': price,
                    'profit': profit,
                    'return': profit / (entry * pos['shares']),
                    'reason': f'止损@{pos["stop_loss"]:.2f}',
                })
                del positions[code]
                continue

            # ATR跟踪止损 (简化: 用trailing_activation和offset)
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

                        # 利润锁定
                        new_stop = entry * (1 + profit_pct * lock_ratio)
                        pos['stop_loss'] = max(pos['stop_loss'], new_stop)

            # 清仓: 仓位过小
            if pos['shares'] < 100:
                revenue = price * pos['shares'] * (1 - COMMISSION_SELL)
                cash += revenue
                profit = revenue - entry * pos['shares'] * (1 + COMMISSION_BUY)
                trades.append({
                    'code': code,
                    'entry_date': pos['entry_date'],
                    'exit_date': date_str,
                    'entry_price': entry,
                    'exit_price': price,
                    'profit': profit,
                    'return': profit / max(entry * pos['shares'], 1),
                    'reason': '清仓(余额不足)',
                })
                del positions[code]

        # 权益曲线
        equity = cash + sum(
            current_prices.get(c, 0) * p['shares']
            for c, p in positions.items()
            if c in current_prices
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

    if len(equity) > 1:
        years = len(equity) / 252
        annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    else:
        annual_return = 0

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
    else:
        win_rate = profit_loss_ratio = avg_trade_ret = 0

    # 最大单笔亏损
    max_loss = min((t['return'] for t in trades), default=0)

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
        'final_equity': equity[-1][1],
    }


def main():
    parser = argparse.ArgumentParser(description='出场参数A/B测试')
    parser.add_argument('--quick', action='store_true', help='10只快速验证')
    parser.add_argument('--stocks', type=int, default=30, help='股票数量')
    args = parser.parse_args()

    n_stocks = 10 if args.quick else args.stocks

    print('=' * 70)
    print('出场参数 A/B测试')
    print('=' * 70)
    print(f'  A(旧): trailing 5%, 止盈 5/10/15%, 锁利30%')
    print(f'  B(新): trailing 3%, 止盈 8/15/25%, 锁利50%, 阶梯保本')
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
    total_sells = sum(1 for v in all_signals.values() for s in v if s['type'] == 'sell')
    print(f'  共 {total_buys}B / {total_sells}S, {len(all_signals)}只')

    if total_buys < 5:
        print('信号太少，退出')
        return

    # 加载数据
    stock_data = {}
    for code in all_signals:
        df = load_stock_data(code)
        if df is not None:
            stock_data[code] = df
    print(f'  数据加载: {len(stock_data)}只')

    # A/B测试
    print(f'\n[3] A/B回测...')
    configs = [
        ('A 旧参数', OLD_EXIT_CONFIG),
        ('B 新参数', NEW_EXIT_CONFIG),
    ]

    results = {}
    for label, config in configs:
        print(f'  {label}...', end='', flush=True)
        t0 = time.time()
        result = simulate_with_exit_params(all_signals, stock_data, config)
        metrics = calc_metrics(result)
        results[label] = {'metrics': metrics, 'trades': result['trades']}
        print(f' ret={metrics.get("total_return",0):.1%}, '
              f'sharpe={metrics.get("sharpe",0):.2f}, '
              f'dd={metrics.get("max_drawdown",0):.1%}, '
              f'win={metrics.get("win_rate",0):.1%}, '
              f'trades={metrics.get("total_trades",0)} '
              f'({time.time()-t0:.1f}s)')

    # 对比表
    print('\n' + '=' * 90)
    print('A/B测试结果')
    print('=' * 90)
    print(f'{"版本":<10s} {"年化":>8s} {"总收益":>8s} {"Sharpe":>8s} '
          f'{"最大回撤":>8s} {"胜率":>7s} {"盈亏比":>7s} {"均笔收益":>8s} {"最大单亏":>8s} {"交易数":>6s}')
    print('-' * 90)

    for label, _ in configs:
        m = results[label]['metrics']
        print(f'{label:<10s} '
              f'{m.get("annual_return",0):>7.1%} '
              f'{m.get("total_return",0):>7.1%} '
              f'{m.get("sharpe",0):>7.2f} '
              f'{m.get("max_drawdown",0):>7.1%} '
              f'{m.get("win_rate",0):>6.1%} '
              f'{m.get("profit_loss_ratio",0):>6.2f} '
              f'{m.get("avg_trade_return",0):>7.2%} '
              f'{m.get("max_single_loss",0):>7.2%} '
              f'{m.get("total_trades",0):>5d}')

    # 差异
    a = results['A 旧参数']['metrics']
    b = results['B 新参数']['metrics']
    print('\n' + '-' * 90)
    print('B vs A 差异:')
    for key, label in [
        ('total_return', '总收益'),
        ('sharpe', 'Sharpe'),
        ('max_drawdown', '最大回撤'),
        ('win_rate', '胜率'),
        ('avg_trade_return', '均笔收益'),
    ]:
        diff = b.get(key, 0) - a.get(key, 0)
        pct = diff / abs(a.get(key, 1)) * 100 if a.get(key, 0) != 0 else 0
        sign = '+' if diff >= 0 else ''
        print(f'  {label}: {sign}{diff:.4f} ({sign}{pct:.1f}%)')

    # 止损分析
    print('\n止损分析:')
    for label, _ in configs:
        trades = results[label]['trades']
        stops = [t for t in trades if '止损' in t.get('reason', '')]
        early = [t for t in trades if '清仓' in t.get('reason', '') or '余额' in t.get('reason', '')]
        print(f'  {label}: 止损{len(stops)}笔 (avg={np.mean([t["return"] for t in stops]):.2%}) '
              f'清仓{len(early)}笔')

    # 输出最终判定
    print('\n' + '=' * 90)
    ret_diff = b.get('total_return', 0) - a.get('total_return', 0)
    sharpe_diff = b.get('sharpe', 0) - a.get('sharpe', 0)
    dd_diff = b.get('max_drawdown', 0) - a.get('max_drawdown', 0)

    if ret_diff > 0.02 and sharpe_diff > 0.1:
        verdict = '新参数显著优于旧参数'
    elif ret_diff > 0 and sharpe_diff > 0:
        verdict = '新参数略优于旧参数'
    elif ret_diff < -0.02 or sharpe_diff < -0.1:
        verdict = '旧参数更好，建议回滚'
    else:
        verdict = '无显著差异'

    print(f'结论: {verdict}')
    print(f'  收益差: {ret_diff:+.2%} | Sharpe差: {sharpe_diff:+.2f} | 回撤差: {dd_diff:+.2%}')
    print()


if __name__ == '__main__':
    main()
