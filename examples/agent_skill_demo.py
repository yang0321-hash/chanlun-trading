"""
Agent & Skill 架构演示
展示新的 Agent + Skill 设计模式
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from skills.pattern.fractal_skill import FractalSkill, find_recent_fractals
from skills.signal.buy_point_skill import BuyPointSkill, BuyPointType, find_buy_points
from skills.risk.stop_loss_skill import StopLossSkill, calculate_stop_loss
from agents.chanlun_agent import ChanLunAgent, MultiLevelAgent
from agents.base_agent import AgentConfig, AgentOrchestrator
from core.kline import KLine
from core.stroke import StrokeGenerator


def demo_skills():
    """演示 Skills 使用"""
    print("=" * 60)
    print("1. Skills 演示")
    print("=" * 60)

    # 创建模拟数据
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)

    # 生成有波动的价格数据
    close = np.cumsum(np.random.randn(100) * 2) + 100
    high = close + np.random.rand(100) * 3
    low = close - np.random.rand(100) * 3
    open_p = close + np.random.randn(100)
    volume = np.random.randint(10000, 100000, 100)

    df = pd.DataFrame({
        'date': dates,
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    # 转换为 KLine
    kline = KLine.from_dataframe(df)

    # 1. 分型识别 Skill
    print("\n1.1 分型识别 Skill:")
    fractal_skill = FractalSkill(config={'confirm_required': True})
    result = fractal_skill.execute(kline=kline, start_index=50)

    if result.success and result.data is not None:
        print(f"  识别到 {len(result.data)} 个分型")
        print(f"  顶分型: {result.metadata.get('top_count', 0)}")
        print(f"  底分型: {result.metadata.get('bottom_count', 0)}")
        print(f"  平均强度: {result.metadata.get('avg_strength', 0):.2f}")
        print(f"  置信度: {result.confidence:.2f}")

        # 显示最近的分型
        recent = find_recent_fractals(kline, n=3)
        if recent:
            print(f"\n  最近3个分型:")
            for f in recent:
                price = f.high if f.fractal_type == 'top' else f.low
                print(f"    {f.fractal_type} @ index={f.index}, price={price:.2f}")
    else:
        print(f"  分型识别失败: {result.error if result else '未知错误'}")

    # 2. 止损 Skill
    print("\n1.2 止损 Skill:")
    entry_price = 100.0
    current_price = 105.0

    stop_loss = calculate_stop_loss(
        entry_price=entry_price,
        current_price=current_price,
        position_type='long',
        method='fixed',
        stop_pct=0.08
    )

    print(f"  入场价: {entry_price}")
    print(f"  当前价: {current_price}")
    print(f"  止损价: {stop_loss.stop_loss_price:.2f}")
    print(f"  止损比例: {stop_loss.stop_loss_pct:.2%}")
    print(f"  止损方法: {stop_loss.method}")

    # 3. 多种止损方法对比
    print("\n1.3 多种止损方法:")
    methods = ['fixed', 'pivot', 'atr']

    for method in methods:
        skill = StopLossSkill()
        result = skill.execute(
            entry_price=100,
            current_price=105,
            position_type='long',
            method=method,
            pivot={'low': 95, 'high': 110},
            atr=2.5
        )
        if result.success:
            sl = result.data
            print(f"  {method:8s}: {sl.stop_loss_price:.2f} ({sl.stop_loss_pct:.2%}) - {sl.reason}")

    return df, kline


def demo_single_agent():
    """演示单个 Agent 使用"""
    print("\n" + "=" * 60)
    print("2. 单个 Agent 演示")
    print("=" * 60)

    # 创建模拟数据
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    close = np.cumsum(np.random.randn(100) * 2) + 100

    df = pd.DataFrame({
        'date': dates,
        'open': close + np.random.randn(100),
        'high': close + np.random.rand(100) * 3,
        'low': close - np.random.rand(100) * 3,
        'close': close,
        'volume': np.random.randint(10000, 100000, 100)
    })

    # 创建 Agent
    agent_config = AgentConfig(
        name='缠论Agent',
        description='演示用缠论分析Agent',
        skills=['fractal', 'buy_point', 'stop_loss'],
        skill_configs={
            'fractal': {'confirm_required': True},
            'buy_point': {'min_confidence': 0.5},  # 降低阈值以便演示
            'stop_loss': {'default_stop_pct': 0.08},
        }
    )

    agent = ChanLunAgent(config=agent_config)

    # 执行分析
    context = {'data': df}
    result = agent.analyze('600519', context)

    print(f"\n分析结果:")
    print(f"  股票: {result['symbol']}")
    print(f"  信号: {result['signal']}")
    print(f"  信号类型: {result.get('signal_type', 'N/A')}")
    print(f"  信号置信度: {result.get('signal_confidence', 0):.2f}")
    print(f"  当前价格: {result['current_price']:.2f}")
    print(f"  买点数量: {len(result['buy_points'])}")
    print(f"  笔数量: {result['strokes_count']}")
    print(f"  中枢数量: {result['pivots_count']}")

    # 显示买点详情
    if result['buy_points']:
        print(f"\n  买点详情:")
        for bp in result['buy_points']:
            print(f"    {bp.point_type.value}: {bp.reason} (置信度: {bp.confidence:.2f})")

    # 生成交易计划
    print(f"\n交易计划:")
    plan = agent.generate_trade_plan('600519', result, capital=100000)
    if plan:
        print(f"  操作: {plan['action']}")
        print(f"  数量: {plan['quantity']}股")
        print(f"  价格: {plan['price']:.2f}")
        print(f"  理由: {plan['reason']}")
        print(f"  止损: {plan.get('stop_loss', 'N/A')}")
        print(f"  目标: {plan.get('target', 'N/A')}")
    else:
        print(f"  无交易计划")

    # Agent 统计
    print(f"\nAgent 统计:")
    stats = agent.get_stats()
    print(f"  总信号数: {stats['total_signals']}")

    return agent, result


def demo_multi_level_agent():
    """演示多级别 Agent 使用"""
    print("\n" + "=" * 60)
    print("3. 多级别 Agent 演示")
    print("=" * 60)

    # 创建更长的数据以支持周线分析
    dates = pd.date_range('2023-01-01', periods=300)
    np.random.seed(42)
    close = np.cumsum(np.random.randn(300) * 2) + 100

    df = pd.DataFrame({
        'date': dates,
        'open': close + np.random.randn(300),
        'high': close + np.random.rand(300) * 3,
        'low': close - np.random.rand(300) * 3,
        'close': close,
        'volume': np.random.randint(10000, 100000, 300)
    })

    # 创建多级别 Agent
    multi_agent = MultiLevelAgent()

    # 执行分析
    context = {'data': df}
    result = multi_agent.analyze('600519', context)

    print(f"\n多级别分析结果:")
    print(f"  股票: {result['symbol']}")
    print(f"  周线信号: {result.get('weekly_signal', 'N/A')}")
    print(f"  日线信号: {result.get('daily_signal', 'N/A')}")
    print(f"  综合信号: {result.get('signal', 'N/A')}")
    print(f"  信号类型: {result.get('signal_type', 'N/A')}")
    print(f"  综合置信度: {result.get('confidence', 0):.2f}")
    print(f"  理由: {result.get('reason', 'N/A')}")

    return multi_agent, result


def demo_agent_orchestrator():
    """演示 Agent 编排器"""
    print("\n" + "=" * 60)
    print("4. Agent 编排器演示")
    print("=" * 60)

    # 创建多个 Agent
    agent1 = ChanLunAgent(AgentConfig(
        name='保守Agent',
        skill_configs={'buy_point': {'min_confidence': 0.8}}
    ))

    agent2 = ChanLunAgent(AgentConfig(
        name='激进Agent',
        skill_configs={'buy_point': {'min_confidence': 0.5}}
    ))

    # 创建编排器
    orchestrator = AgentOrchestrator([agent1, agent2])

    # 准备数据
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    close = np.cumsum(np.random.randn(100) * 2) + 100

    df = pd.DataFrame({
        'date': dates,
        'open': close + np.random.randn(100),
        'high': close + np.random.rand(100) * 3,
        'low': close - np.random.rand(100) * 3,
        'close': close,
        'volume': np.random.randint(10000, 100000, 100)
    })

    # 让所有 Agent 分析
    context = {'data': df}
    results = orchestrator.analyze_all('600519', context)

    print(f"\n各 Agent 分析结果:")
    for agent_name, result in results.items():
        if result.get('success'):
            print(f"  {agent_name}: {result.get('signal')} (置信度: {result.get('signal_confidence', 0):.2f})")

    # 获取共识
    print(f"\n共识:")
    consensus = orchestrator.get_consensus()
    print(f"  共识信号: {consensus['signal']}")
    print(f"  共识置信度: {consensus['confidence']:.2f}")
    print(f"  投票: {consensus['votes']}")

    return orchestrator


def main():
    """主函数"""
    print("\n" + "#" * 60)
    print("# Agent & Skill 架构演示")
    print("#" * 60)

    try:
        # 1. Skills 演示
        df, kline = demo_skills()

        # 2. 单个 Agent 演示
        agent, result = demo_single_agent()

        # 3. 多级别 Agent 演示
        multi_agent, multi_result = demo_multi_level_agent()

        # 4. Agent 编排器演示
        orchestrator = demo_agent_orchestrator()

        print("\n" + "#" * 60)
        print("# 演示完成！")
        print("#" * 60)

        print("\n架构优势总结:")
        print("  [Skill] 可复用 - 一个 Skill 可被多个 Agent 使用")
        print("  [Skill] 可测试 - 独立的输入输出，易于单元测试")
        print("  [Skill] 可组合 - 多个 Skills 可组合成复杂能力")
        print("  [Agent] 决策中心 - 集中管理业务逻辑和流程编排")
        print("  [Agent] 状态管理 - 维护分析历史和上下文")
        print("  [Agent] 灵活性 - 可动态替换或升级 Skills")

    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
