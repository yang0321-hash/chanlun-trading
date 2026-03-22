"""简单测试 - 不闪退"""
import sys

try:
    print("开始测试...")
    print("Python版本:", sys.version)

    # 测试导入
    print("\n1. 导入pandas...")
    import pandas as pd
    print("   OK")

    print("\n2. 导入项目模块...")
    sys.path.insert(0, '.')
    from backtest import BacktestEngine, BacktestConfig
    print("   OK")

    print("\n3. 导入策略...")
    from strategies import ChanLunStrategy, OptimizedChanLunStrategy
    print("   OK")

    print("\n4. 生成数据...")
    import numpy as np
    from datetime import datetime

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    data = {
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    print("   OK")

    print("\n5. 运行回测...")
    config = BacktestConfig(initial_capital=100000)
    engine = BacktestEngine(config)
    engine.add_data("TEST", df)
    engine.set_strategy(ChanLunStrategy())
    results = engine.run()
    print(f"   OK - 收益率: {results['total_return']:.2%}")

    print("\n6. 运行优化策略...")
    engine2 = BacktestEngine(config)
    engine2.add_data("TEST", df)
    engine2.set_strategy(OptimizedChanLunStrategy())
    results2 = engine2.run()
    print(f"   OK - 收益率: {results2['total_return']:.2%}")

    print("\n" + "="*50)
    print("对比结果:")
    print(f"  原版: {results['total_return']:.2%}")
    print(f"  优化: {results2['total_return']:.2%}")
    print("="*50)

except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()

input("\n按回车键退出...")
