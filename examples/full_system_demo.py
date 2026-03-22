"""
完整系统演示
展示记忆系统、智能体层、图编排的完整使用
"""
import sys
sys.path.append('..')

import pandas as pd
from datetime import datetime

from graph import create_chanlun_graph, GraphTemplates
from memory import MemoryManager, PatternMemory, SignalType
from agents import PatternAgent, TrendAgent, SignalAgent, RiskAgent
from core.state import ChanLunState


def demo_memory_system():
    """演示记忆系统"""
    print("=" * 60)
    print("1. 记忆系统演示")
    print("=" * 60)

    # 创建记忆管理器
    memory = MemoryManager("./memory/demo")

    # 记录一些历史形态
    pattern_record, trade_record = memory.record_signal(
        pattern_id="BUY_2_600519",
        signal_type=SignalType.BUY_2,
        symbol="600519",
        price=1800.0,
        confidence=0.75,
        features={"fractal": "bottom", "in_pivot": False},
        market_condition="上涨"
    )

    print(f"记录形态: {pattern_record.signal_type.value} @ {pattern_record.price}")

    # 模拟完成交易
    memory.close_position(trade_record, exit_price=1900.0, exit_reason="止盈")

    # 获取信心度调整
    adjusted_conf = memory.get_adjusted_confidence("BUY_2_600519", 0.7)
    print(f"调整后信心度: {adjusted_conf:.2f}")

    # 生成报告
    print("\n" + memory.pattern_memory.generate_report())

    return memory


def demo_agent_chain():
    """演示智能体链"""
    print("\n" + "=" * 60)
    print("2. 智能体链演示")
    print("=" * 60)

    # 创建智能体
    pattern_agent = PatternAgent()
    trend_agent = TrendAgent()
    signal_agent = SignalAgent()

    # 使用模拟数据
    mock_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'open': range(100, 200),
        'high': range(102, 202),
        'low': range(98, 198),
        'close': range(101, 201),
        'volume': [10000] * 100
    })

    from agents.base_agent import AgentInput, AgentChain

    # 创建智能体链
    chain = AgentChain([pattern_agent, trend_agent, signal_agent])

    # 执行
    input_data = AgentInput(
        ohlcv_data=mock_df,
        current_index=99,
        symbol="600519"
    )

    results = chain.execute(input_data)

    print("\n智能体链执行结果:")
    for name, result in results.items():
        print(f"  {name}: {result.reasoning}")
        print(f"    信心度: {result.confidence:.2f}")

    return results


def demo_graph_execution():
    """演示图编排"""
    print("\n" + "=" * 60)
    print("3. 图编排演示")
    print("=" * 60)

    # 创建图
    graph = create_chanlun_graph({
        'signal': {'min_confidence': 0.5},
        'risk': {'max_position_pct': 0.8}
    })

    print("\n图结构:")
    print(graph.visualize())

    # 使用模拟数据执行
    mock_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'open': range(100, 200),
        'high': range(102, 202),
        'low': range(98, 198),
        'close': range(101, 201),
        'volume': [10000] * 100
    })

    # 执行图
    state = graph.execute(
        symbol="600519",
        df=mock_df,
        current_index=99
    )

    print("\n执行日志:")
    for log in state['execution_log']:
        print(f"  - {log}")

    print("\n最终决策:")
    decision = state['final_decision']
    print(f"  操作: {decision['action']}")
    print(f"  信号: {decision['signal_type']}")
    print(f"  理由: {decision['reasoning']}")
    print(f"  建议仓位: {decision['position_size']}股")

    return state


def demo_graph_templates():
    """演示图模板"""
    print("\n" + "=" * 60)
    print("4. 图模板演示")
    print("=" * 60)

    templates = [
        ("基础图", GraphTemplates.basic_graph),
        ("完整图", GraphTemplates.full_graph),
        ("周线日线图", GraphTemplates.weekly_daily_graph),
        ("保守策略图", GraphTemplates.conservative_graph),
        ("激进策略图", GraphTemplates.aggressive_graph),
    ]

    for name, template_func in templates:
        print(f"\n{name}:")
        graph = template_func()
        print(f"  节点数: {len(graph.nodes)}")
        print(f"  边数: {len(graph.edges)}")
        print(f"  起始: {graph.start_node}")
        print(f"  结束: {graph.end_node}")


def demo_integrated_system():
    """演示完整集成系统"""
    print("\n" + "=" * 60)
    print("5. 完整集成系统演示")
    print("=" * 60)

    # 1. 创建记忆管理器
    memory = MemoryManager("./memory/integrated_demo")

    # 2. 创建图（使用记忆系统）
    config = {
        'signal': {
            'min_confidence': 0.6,
            'use_memory': True
        },
        'risk': {
            'max_position_pct': 0.8,
            'initial_capital': 100000
        }
    }

    graph = GraphTemplates.full_graph(config)

    # 将记忆系统连接到智能体
    for agent in graph.agents.values():
        if hasattr(agent, 'set_memory_manager'):
            agent.set_memory_manager(memory)

    # 3. 使用模拟数据执行
    mock_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'open': [100 + i*0.5 for i in range(100)],
        'high': [102 + i*0.5 for i in range(100)],
        'low': [98 + i*0.5 for i in range(100)],
        'close': [101 + i*0.5 for i in range(100)],
        'volume': [10000] * 100
    })

    # 添加一些波动来产生信号
    for i in range(80, 90):
        mock_df.loc[i, 'close'] = 101 + i*0.5 + (5 if i % 2 == 0 else -5)

    state = graph.execute(
        symbol="600519",
        df=mock_df,
        current_index=99
    )

    print("\n执行结果:")
    print(f"  节点执行数: {len(state['execution_log'])}")
    print(f"  生成信号: {state['final_decision']['signal_type']}")
    print(f"  建议操作: {state['final_decision']['action']}")

    # 4. 导出记忆
    memory.export_to_json("./memory/integrated_demo/report.json")
    print("\n记忆已导出到: ./memory/integrated_demo/report.json")

    return state, memory


def main():
    """主函数"""
    print("\n" + "#" * 60)
    print("# 缠论交易系统 - 完整演示")
    print("#" * 60)

    try:
        # 1. 记忆系统
        memory = demo_memory_system()

        # 2. 智能体链
        agent_results = demo_agent_chain()

        # 3. 图编排
        graph_state = demo_graph_execution()

        # 4. 图模板
        demo_graph_templates()

        # 5. 完整集成
        final_state, final_memory = demo_integrated_system()

        print("\n" + "#" * 60)
        print("# 演示完成！")
        print("#" * 60)

        print("\n系统架构总结:")
        print("  1. 记忆系统 - 记录和学习历史形态")
        print("  2. 智能体层 - 分工协作的AI分析")
        print("  3. 图编排 - 灵活的流程编排")
        print("\n可以开始实际交易了！")

    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
