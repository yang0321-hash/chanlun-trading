"""
新架构演示 - 使用状态和配置管理
展示如何使用新的 state.py 和 config.py 模块
"""
import sys
sys.path.append('..')

from core.state import ChanLunState, SignalType, TrendDirection, FractalInfo
from core.config import (ChanLunConfig, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG,
                         StrategyConfig)
from datetime import datetime


def demo_state_management():
    """演示状态管理"""
    print("=" * 50)
    print("状态管理演示")
    print("=" * 50)

    # 创建状态
    state = ChanLunState(
        symbol="600519",
        current_datetime=datetime.now(),
        trend_direction=TrendDirection.UP,
        trend_strength=0.7
    )

    print(f"初始状态: {state}")

    # 添加一些识别结果
    state.fractals = [
        FractalInfo(index=10, fractal_type="bottom", high=100, low=95, confirmed=True),
        FractalInfo(index=20, fractal_type="top", high=115, low=110, confirmed=True),
    ]

    # 添加买入信号
    state.add_signal(
        signal_type=SignalType.BUY_2,
        price=100.0,
        confidence=0.75,
        reason="周线2买，回抽不破前低"
    )

    print(f"\n添加信号后: {state}")
    print(f"当前信号: {state.current_signal.signal_type.value}")
    print(f"买入历史: {[s.signal_type.value for s in state.get_latest_buys()]}")

    # 模拟持仓
    state.position = 100
    state.entry_price = 100.0
    state.stop_loss = 97.0
    print(f"\n持仓状态: {state.position}股, 成本{state.entry_price}, 止损{state.stop_loss}")


def demo_config_management():
    """演示配置管理"""
    print("\n" + "=" * 50)
    print("配置管理演示")
    print("=" * 50)

    # 使用默认配置
    config = ChanLunConfig()
    print(f"默认配置 - 入场信心: {config.strategy.entry_confidence}")

    # 使用保守配置
    conservative = ChanLunConfig.get_preset("conservative")
    print(f"保守配置 - 入场信心: {conservative.strategy.entry_confidence}")

    # 使用激进配置
    aggressive = ChanLunConfig.get_preset("aggressive")
    print(f"激进配置 - 入场信心: {aggressive.strategy.entry_confidence}")

    # 自定义配置
    custom_config = ChanLunConfig(
        strategy=StrategyConfig(
            entry_confidence=0.9,
            max_position_pct=0.3
        )
    )
    print(f"自定义配置 - 入场信心: {custom_config.strategy.entry_confidence}")

    # 导出为字典
    config_dict = config.to_dict()
    print(f"\n配置字典: {config_dict}")

    # 从字典恢复
    restored = ChanLunConfig.from_dict(config_dict)
    print(f"恢复配置 - 入场信心: {restored.strategy.entry_confidence}")


def demo_state_flow():
    """演示状态流转"""
    print("\n" + "=" * 50)
    print("状态流转演示 - 模拟交易流程")
    print("=" * 50)

    # 初始状态
    state = ChanLunState(symbol="sz002600")
    print(f"\n1. 初始: {state}")

    # 步骤1: 识别分型
    state.trend_direction = TrendDirection.DOWN
    print(f"\n2. 识别分型: 趋势向下")

    # 步骤2: 生成笔
    state.fractals = [
        FractalInfo(index=50, fractal_type="bottom", high=10, low=9, confirmed=True)
    ]
    print(f"\n3. 生成笔: 发现底分型")

    # 步骤3: 识别中枢
    state.pivots = [{"high": 12, "low": 8, "start": 30, "end": 60}]
    print(f"\n4. 识别中枢: [8, 12]")

    # 步骤4: 生成信号
    state.add_signal(
        signal_type=SignalType.BUY_1,
        price=9.5,
        confidence=0.8,
        reason="1买：下跌趋势中最后中枢下方底背驰"
    )
    print(f"\n5. 生成信号: {state.current_signal.signal_type.value}")

    # 步骤5: 执行交易
    state.position = 1000
    state.entry_price = 9.5
    state.stop_loss = 9.0
    print(f"\n6. 执行交易: 买入{state.position}股 @ {state.entry_price}")

    # 步骤6: 更新状态
    state.trend_direction = TrendDirection.UP
    state.add_signal(
        signal_type=SignalType.SELL_2,
        price=11.0,
        confidence=0.7,
        reason="2卖：上涨乏力，MACD顶背离"
    )
    print(f"\n7. 更新状态: {state.current_signal.signal_type.value}")


if __name__ == "__main__":
    demo_state_management()
    demo_config_management()
    demo_state_flow()
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
