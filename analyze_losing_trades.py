"""
分析sz002600亏损交易，寻找改进点
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json
from loguru import logger
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.strategy import SignalType

# 导入原始策略
from strategies.chanlun_trading_system import ChanLunTradingSystem


def load_tdx_json(code: str, json_dir: str = '.claude/temp') -> pd.DataFrame:
    json_path = f"{json_dir}/{code}.day.json"
    if not os.path.exists(json_path):
        logger.error(f"未找到数据文件: {json_path}")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df['amount'] = df['volume'] * df['close']
    return df


class ChanLunWithTracking(ChanLunTradingSystem):
    """带交易跟踪的缠论策略"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed_trades = []  # 完成的交易对
        self._entry_trade = None  # 当前入场交易

    def on_order(self, signal, execution_price, quantity):
        """订单执行回调 - 跟踪交易对"""
        super().on_order(signal, execution_price, quantity)

        if signal.signal_type == SignalType.BUY:
            # 记录入场
            self._entry_trade = {
                'entry_date': signal.datetime,
                'entry_price': execution_price,
                'quantity': quantity,
                'entry_reason': signal.reason,
            }
        elif signal.signal_type == SignalType.SELL and self._entry_trade is not None:
            # 记录出场
            profit = (execution_price - self._entry_trade['entry_price']) * quantity
            profit_pct = (execution_price - self._entry_trade['entry_price']) / self._entry_trade['entry_price']

            self.completed_trades.append({
                'entry_date': self._entry_trade['entry_date'],
                'entry_price': self._entry_trade['entry_price'],
                'exit_date': signal.datetime,
                'exit_price': execution_price,
                'quantity': quantity,
                'entry_reason': self._entry_trade['entry_reason'],
                'exit_reason': signal.reason,
                'profit': profit,
                'profit_pct': profit_pct,
            })
            self._entry_trade = None


def analyze_losing_trades(code: str = 'sz002600', start_date: str = '2021-01-01'):
    """分析亏损交易"""
    df = load_tdx_json(code)
    if df is None:
        return
    df = df[df.index >= start_date]

    logger.info("=" * 80)
    logger.info(f"{code} 亏损交易分析")
    logger.info("=" * 80)

    config = BacktestConfig(
        initial_capital=500000,
        commission=0.0003,
        slippage=0.0001,
        min_unit=100,
        position_limit=0.95,
    )

    strategy = ChanLunWithTracking(
        name='最优配置',
        enable_buy1=False,
        enable_buy2=True,
        enable_buy3=False,
        min_confidence=0.60,
        enable_volume_confirm=True,
        min_volume_ratio=1.2,
        second_sell_partial_exit=True,
        second_sell_exit_ratio=0.3,
    )

    engine = BacktestEngine(config)
    engine.add_data(code, df)
    engine.set_strategy(strategy)
    result = engine.run()

    trades = strategy.completed_trades

    print("\n" + "=" * 100)
    print("交易详情分析".center(100))
    print("=" * 100 + "\n")

    if not trades:
        print("未找到完整交易对")
        return

    # 分类统计
    profitable = [t for t in trades if t['profit'] > 0]
    losing = [t for t in trades if t['profit'] <= 0]

    print(f"总交易次数: {len(trades)}")
    print(f"盈利交易: {len(profitable)}")
    print(f"亏损交易: {len(losing)}")
    print(f"胜率: {len(profitable)/len(trades)*100:.1f}%")
    print(f"总盈利: {sum(t['profit'] for t in profitable):,.0f}")
    print(f"总亏损: {sum(t['profit'] for t in losing):,.0f}")
    print(f"净盈亏: {sum(t['profit'] for t in trades):,.0f}")
    print()

    # 亏损交易详情
    print("=" * 100)
    print("亏损交易详情")
    print("=" * 100 + "\n")

    print(f"{'买入日期':<12}{'买入价':<8}{'卖出日期':<12}{'卖出价':<8}{'亏损%':<10}{'亏损额':<12}{'卖出原因'}")
    print("-" * 100)

    losing_sorted = sorted(losing, key=lambda x: x['profit_pct'])
    for trade in losing_sorted:
        buy_d = trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date'])[:10]
        sell_d = trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date'])[:10]
        print(f"{buy_d:<12}{trade['entry_price']:<8.2f}"
              f"{sell_d:<12}{trade['exit_price']:<8.2f}"
              f"{trade['profit_pct']:<10.2%}{trade['profit']:<12,.0f}"
              f"{trade['exit_reason'][:45]}")

    print()

    # 盈利交易详情
    print("=" * 100)
    print("盈利交易详情")
    print("=" * 100 + "\n")

    print(f"{'买入日期':<12}{'买入价':<8}{'卖出日期':<12}{'卖出价':<8}{'盈利%':<10}{'盈利额':<12}{'卖出原因'}")
    print("-" * 100)

    profitable_sorted = sorted(profitable, key=lambda x: x['profit_pct'], reverse=True)
    for trade in profitable_sorted:
        buy_d = trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date'])[:10]
        sell_d = trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date'])[:10]
        print(f"{buy_d:<12}{trade['entry_price']:<8.2f}"
              f"{sell_d:<12}{trade['exit_price']:<8.2f}"
              f"{trade['profit_pct']:<10.2%}{trade['profit']:<12,.0f}"
              f"{trade['exit_reason'][:45]}")

    print()

    # 亏损原因分析
    print("=" * 100)
    print("亏损原因分析")
    print("=" * 100 + "\n")

    # 按卖出原因分类
    sell_reasons = {}
    for trade in losing:
        reason = trade['exit_reason'][:35]
        if reason not in sell_reasons:
            sell_reasons[reason] = {'count': 0, 'total_loss': 0}
        sell_reasons[reason]['count'] += 1
        sell_reasons[reason]['total_loss'] += trade['profit']

    print(f"{'卖出原因':<40}{'次数':<8}{'总亏损':<12}")
    print("-" * 65)

    for reason, data in sorted(sell_reasons.items(), key=lambda x: x[1]['total_loss']):
        print(f"{reason:<40}{data['count']:<8}{data['total_loss']:<12,.0f}")

    print()

    # 入场时机分析
    print("=" * 100)
    print("入场时机分析 (按月份)")
    print("=" * 100 + "\n")

    # 按月份分析
    monthly_results = {}
    for trade in trades:
        month = trade['entry_date'].strftime('%Y-%m')
        if month not in monthly_results:
            monthly_results[month] = {'count': 0, 'profit': 0}
        monthly_results[month]['count'] += 1
        monthly_results[month]['profit'] += trade['profit_pct']

    print(f"{'月份':<10}{'交易次数':<10}{'平均收益率':<15}{'状态'}")
    print("-" * 50)

    for month in sorted(monthly_results.keys()):
        data = monthly_results[month]
        avg_ret = data['profit'] / data['count'] if data['count'] > 0 else 0
        status = "[OK]盈利" if avg_ret > 0 else "[X]亏损"
        print(f"{month:<10}{data['count']:<10}{avg_ret:<15.2%}{status}")

    print()

    # 改进建议
    print("=" * 100)
    print("改进建议")
    print("=" * 100 + "\n")

    suggestions = []

    # 分析1: 止损触发频率
    stop_loss_count = sum(1 for t in losing if '止损' in t['exit_reason'])
    if stop_loss_count > len(losing) * 0.5:
        suggestions.append("1. 超过50%亏损交易由止损触发")
        suggestions.append("   → 建议: 优化入场时机，等待更明确的回调确认信号")

    # 分析2: 跌停触发
    limit_down = sum(1 for t in losing if '跌停' in t['exit_reason'])
    if limit_down > 0:
        max_loss = max(t['profit_pct'] for t in losing if '跌停' in t['exit_reason'])
        suggestions.append(f"2. {limit_down}笔交易触发跌停风控 (单笔最大亏损{max_loss:.1%})")
        suggestions.append("   → 建议: 增加市场情绪过滤，在整体市场恐慌时降低仓位或空仓")

    # 分析3: 时间段分析
    losing_months = [m for m, d in monthly_results.items() if d['profit'] / d['count'] < 0]
    if len(losing_months) > len(monthly_results) * 0.6:
        suggestions.append(f"3. {len(losing_months)}/{len(monthly_results)}个月份亏损")
        suggestions.append("   → 建议: 策略可能不适合该股票，或需结合大盘择时")

    # 分析4: 连续亏损
    consecutive_losses = 0
    max_consecutive = 0
    for trade in trades:
        if trade['profit'] <= 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0

    if max_consecutive >= 3:
        suggestions.append(f"4. 最大连续亏损{max_consecutive}笔")
        suggestions.append("   → 建议: 添加连续亏损保护，连续亏损3笔后暂停交易")

    # 分析5: 盈亏比
    if profitable and losing:
        avg_profit = sum(t['profit_pct'] for t in profitable) / len(profitable)
        avg_loss = sum(t['profit_pct'] for t in losing) / len(losing)
        profit_loss_ratio = avg_profit / abs(avg_loss) if avg_loss != 0 else 0

        if profit_loss_ratio < 1.5:
            suggestions.append(f"5. 盈亏比失衡 (平均盈利{avg_profit:.1%} vs 平均亏损{avg_loss:.1%})")
            suggestions.append("   → 建议: 让盈利充分运行，延长止盈或使用移动止盈")

    for suggestion in suggestions:
        print(suggestion)

    if not suggestions:
        print("暂无明显改进点，策略表现符合预期。")

    print()

    # 策略适用性评估
    print("=" * 100)
    print("策略适用性评估")
    print("=" * 100 + "\n")

    if len(losing_months) > len(monthly_results) * 0.7:
        print("[!] 当前策略不太适合该股票")
        print("   原因: 多数月份表现不佳")
        print("   建议: 考虑换股或调整策略参数")
    elif result['total_return'] < 0 and abs(result['max_drawdown']) > 0.15:
        print("[!] 策略在该股票上风险较高")
        print("   原因: 回撤过大，收益为负")
        print("   建议: 降低仓位或增加过滤条件")
    else:
        print("[OK] 策略表现正常，可继续优化")

    print()

    return trades


if __name__ == '__main__':
    analyze_losing_trades()
