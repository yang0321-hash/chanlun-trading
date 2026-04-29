#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
周线+日线策略预设参数组合

提供三种优化好的参数配置供快速使用
"""

from typing import Dict, Any
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy


# 预设参数组合
PRESETS: Dict[str, Dict[str, Any]] = {
    'conservative': {
        'name': '周日线缠论-稳健型',
        'weekly_min_strokes': 4,
        'daily_min_strokes': 3,
        'stop_loss_pct': 0.06,
        'exit_ratio': 0.4,
        'description': '适合低波动大盘蓝筹，信号质量高，止损严格'
    },

    'balanced': {
        'name': '周日线缠论-平衡型',
        'weekly_min_strokes': 3,
        'daily_min_strokes': 3,
        'stop_loss_pct': 0.08,
        'exit_ratio': 0.5,
        'description': '默认参数，平衡收益与风险'
    },

    'aggressive': {
        'name': '周日线缠论-激进型',
        'weekly_min_strokes': 5,
        'daily_min_strokes': 4,
        'stop_loss_pct': 0.10,
        'exit_ratio': 0.6,
        'description': '适合高波动成长股，宽松止损捕捉大趋势'
    },

    # 针对不同市况的优化
    'trending': {
        'name': '周日线缠论-趋势型',
        'weekly_min_strokes': 5,
        'daily_min_strokes': 5,
        'stop_loss_pct': 0.12,
        'exit_ratio': 0.7,
        'description': '单边趋势行情，减少假信号'
    },

    'range_bound': {
        'name': '周日线缠论-震荡型',
        'weekly_min_strokes': 2,
        'daily_min_strokes': 2,
        'stop_loss_pct': 0.05,
        'exit_ratio': 0.3,
        'description': '震荡行情，快进快出'
    },
}


def create_strategy(preset: str = 'balanced') -> WeeklyDailyChanLunStrategy:
    """
    创建预设策略

    Args:
        preset: 预设名称 ('conservative', 'balanced', 'aggressive', 'trending', 'range_bound')

    Returns:
        WeeklyDailyChanLunStrategy 实例
    """
    if preset not in PRESETS:
        raise ValueError(f"未知预设: {preset}. 可选: {list(PRESETS.keys())}")

    params = PRESETS[preset]

    return WeeklyDailyChanLunStrategy(
        name=params['name'],
        weekly_min_strokes=params['weekly_min_strokes'],
        daily_min_strokes=params['daily_min_strokes'],
        stop_loss_pct=params['stop_loss_pct'],
        exit_ratio=params['exit_ratio']
    )


def list_presets():
    """列出所有预设参数"""
    print("\n" + "=" * 70)
    print("周线+日线策略预设参数")
    print("=" * 70)

    for key, params in PRESETS.items():
        print(f"\n[{key}] {params['name']}")
        print(f"  描述: {params['description']}")
        print(f"  参数:")
        print(f"    weekly_min_strokes: {params['weekly_min_strokes']}")
        print(f"    daily_min_strokes:  {params['daily_min_strokes']}")
        print(f"    stop_loss_pct:      {params['stop_loss_pct']:.0%}")
        print(f"    exit_ratio:         {params['exit_ratio']:.0%}")


def compare_presets():
    """对比各预设参数特点"""
    print("\n" + "=" * 70)
    print("预设参数对比")
    print("=" * 70)

    print(f"\n{'预设':<15} {'周线笔':<8} {'日线笔':<8} {'止损':<8} {'减仓':<8} {'适合场景'}")
    print("-" * 70)

    for key, params in PRESETS.items():
        print(f"{key:<15} {params['weekly_min_strokes']:<8} {params['daily_min_strokes']:<8} "
              f"{params['stop_loss_pct']:<8.0%} {params['exit_ratio']:<8.0%} {params['description']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'list':
            list_presets()
        elif sys.argv[1] == 'compare':
            compare_presets()
        else:
            print(f"创建策略: {sys.argv[1]}")
            strategy = create_strategy(sys.argv[1])
            print(f"策略名称: {strategy.name}")
    else:
        compare_presets()
