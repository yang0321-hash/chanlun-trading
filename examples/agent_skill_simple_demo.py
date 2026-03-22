"""
Agent & Skill 架构简化演示
展示新的 Agent + Skill 设计模式的核心功能
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from datetime import datetime

# 只导入止损相关的（这是最简单的）
from skills.risk.stop_loss_skill import StopLossSkill, calculate_stop_loss
from agents.base_agent import BaseAgent, AgentConfig, AgentState


def demo_stop_loss_skill():
    """演示止损 Skill"""
    print("=" * 60)
    print("1. 止损 Skill 演示")
    print("=" * 60)

    skill = StopLossSkill()

    # 多种止损方法对比
    print("\n1.1 不同止损方法对比:")
    print("  入场价: 100.0, 当前价: 105.0\n")

    methods = [
        ('fixed', {'entry_price': 100, 'current_price': 105, 'position_type': 'long'}),
        ('pivot', {'entry_price': 100, 'current_price': 105, 'position_type': 'long',
                   'pivot': {'low': 95, 'high': 110}}),
        ('fractal', {'entry_price': 100, 'current_price': 105, 'position_type': 'long',
                     'fractal': {'low': 98, 'high': 108}}),
        ('trailing', {'entry_price': 100, 'current_price': 110, 'position_type': 'long',
                      'highest_since_entry': 110}),
    ]

    for method_name, params in methods:
        result = skill.execute(**params, method=method_name)
        if result.success:
            sl = result.data
            print(f"  {method_name:10s}: 止损={sl.stop_loss_price:.2f} ({sl.stop_loss_pct:.1%}) - {sl.reason}")

    # 不同止损比例
    print("\n1.2 不同止损比例:")
    stop_pcts = [0.03, 0.05, 0.08, 0.12]

    for pct in stop_pcts:
        sl = calculate_stop_loss(entry_price=100, current_price=105, stop_pct=pct)
        print(f"  {pct*100:.0f}%: {sl.stop_loss_price:.2f}")

    return skill


def demo_agent_config():
    """演示 Agent 配置"""
    print("\n" + "=" * 60)
    print("2. Agent 配置演示")
    print("=" * 60)

    # 创建配置
    config = AgentConfig(
        name='测试Agent',
        description='这是一个测试Agent',
        skills=['fractal', 'buy_point', 'stop_loss'],
        skill_configs={
            'fractal': {'confirm_required': True},
            'buy_point': {'min_confidence': 0.7},
            'stop_loss': {'default_stop_pct': 0.08},
        },
        agent_config={
            'max_position': 0.3,
            'min_confidence': 0.6,
        }
    )

    print(f"\nAgent 配置:")
    print(f"  名称: {config.name}")
    print(f"  描述: {config.description}")
    print(f"  Skills: {config.skills}")
    print(f"  参数:")
    for key, value in config.agent_config.items():
        print(f"    {key}: {value}")

    print(f"\nSkill 配置:")
    for skill_name, skill_config in config.skill_configs.items():
        print(f"  {skill_name}:")
        for key, value in skill_config.items():
            print(f"    {key}: {value}")

    return config


def demo_agent_state():
    """演示 Agent 状态"""
    print("\n" + "=" * 60)
    print("3. Agent 状态演示")
    print("=" * 60)

    # 创建状态
    state = AgentState(
        agent_name='测试Agent',
        current_datetime=datetime.now()
    )

    # 添加一些分析结果
    state.update_analysis('600519', 'signal', 'buy')
    state.update_analysis('600519', 'confidence', 0.75)
    state.update_analysis('600519', 'entry_price', 100.0)

    # 添加信号
    state.add_signal({
        'symbol': '600519',
        'action': 'buy',
        'quantity': 100,
        'price': 100.0,
    })

    print(f"\nAgent 状态:")
    print(f"  名称: {state.agent_name}")
    print(f"  分析股票: {len(state.analysis_results)}")
    print(f"  生成信号: {len(state.signals_generated)}")

    print(f"\n600519 分析结果:")
    result = state.get_analysis('600519')
    for key, value in result.items():
        print(f"  {key}: {value}")

    print(f"\n所有信号:")
    for signal in state.get_signals():
        print(f"  {signal}")

    return state


def demo_skill_chain():
    """演示 Skill 链"""
    print("\n" + "=" * 60)
    print("4. Skill 链演示")
    print("=" * 60)

    from skills.base import SkillChain

    # 创建多个 Skill
    skill1 = StopLossSkill(config={'default_stop_pct': 0.05})
    skill2 = StopLossSkill(config={'default_stop_pct': 0.08})

    # 创建 Skill 链
    chain = SkillChain([skill1, skill2])

    # 执行链
    initial_params = {
        'entry_price': 100,
        'current_price': 105,
        'position_type': 'long',
    }

    print(f"\n输入参数: {initial_params}")
    print(f"\nSkill 链执行:")

    results = chain.execute(initial_params)

    print(f"  第1个 Skill (5%止损):")
    if results[0].success:
        sl = results[0].data
        print(f"    止损: {sl.stop_loss_price:.2f} ({sl.stop_loss_pct:.1%})")

    print(f"  第2个 Skill (8%止损):")
    # 链式执行时，前一个的输出不会自动传递到下一个
    # 这是设计上的选择，链式执行用于顺序执行不同逻辑

    return chain


def main():
    """主函数"""
    print("\n" + "#" * 60)
    print("# Agent & Skill 架构演示 (简化版)")
    print("#" * 60)

    try:
        # 1. 止损 Skill
        demo_stop_loss_skill()

        # 2. Agent 配置
        demo_agent_config()

        # 3. Agent 状态
        demo_agent_state()

        # 4. Skill 链
        demo_skill_chain()

        print("\n" + "#" * 60)
        print("# 演示完成！")
        print("#" * 60)

        print("\n架构核心概念:")
        print("  [Skill] 单一职责 - 每个Skill只做一件事")
        print("  [Skill] 无状态 - 不保存调用之间的状态")
        print("  [Skill] 可复用 - 可被多个Agent使用")
        print("  [Agent] 决策中心 - 管理业务逻辑和流程")
        print("  [Agent] 状态管理 - 维护分析历史和上下文")
        print("  [AgentConfig] 配置驱动 - 灵活配置Agent行为")

        print("\n目录结构:")
        print("  skills/")
        print("    ├── base.py           # Skill基类")
        print("    ├── pattern/          # 模式识别Skills")
        print("    ├── signal/           # 信号生成Skills")
        print("    └── risk/             # 风险管理Skills")
        print("  agents/")
        print("    ├── base_agent.py     # Agent基类")
        print("    └── chanlun_agent.py   # 缠论分析Agent")

    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
