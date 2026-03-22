# 周线+日线策略参数优化指南

## 快速开始

### 1. 对比预设参数
```bash
python compare_presets.py --symbol sh600519
```
这会对比5种预设参数配置，找出最优组合。

### 2. 运行完整优化
```bash
# 快速优化 (3x3x3x3 = 81种组合)
python optimize_weekly_daily.py --symbol sh600519 --quick

# 完整优化 (4x4x5x5 = 400种组合)
python optimize_weekly_daily.py --symbol sh600519

# 敏感性分析
python optimize_weekly_daily.py --symbol sh600519 --sensitivity
```

## 预设参数说明

| 预设 | 周线笔 | 日线笔 | 止损 | 减仓 | 适用场景 |
|------|--------|--------|------|------|----------|
| `conservative` | 4 | 3 | 6% | 40% | 大盘蓝筹，稳健型 |
| `balanced` | 3 | 3 | 8% | 50% | 默认，平衡型 |
| `aggressive` | 5 | 4 | 10% | 60% | 成长股，激进型 |
| `trending` | 5 | 5 | 12% | 70% | 单边趋势 |
| `range_bound` | 2 | 2 | 5% | 30% | 震荡行情 |

## 代码示例

### 使用预设参数
```python
from strategies.weekly_daily_presets import create_strategy

# 创建稳健型策略
strategy = create_strategy('conservative')

# 创建平衡型策略
strategy = create_strategy('balanced')
```

### 自定义参数
```python
from strategies.weekly_daily_strategy import WeeklyDailyChanLunStrategy

strategy = WeeklyDailyChanLunStrategy(
    name='自定义策略',
    weekly_min_strokes=4,
    daily_min_strokes=3,
    stop_loss_pct=0.08,
    exit_ratio=0.5
)
```

## 参数详解

### weekly_min_strokes (周线最小笔数)
- **作用**: 控制周线笔的生成条件
- **值越大**: 信号越少但越可靠
- **推荐**:
  - 震荡市: 2-3
  - 趋势市: 4-5

### stop_loss_pct (止损百分比)
- **作用**: 跌破周线1买最低点后的额外止损幅度
- **推荐**:
  - 低波动股票: 5-6%
  - 中等波动: 8%
  - 高波动: 10-12%

### exit_ratio (顶背离减仓比例)
- **作用**: 日线MACD顶背离时减仓的比例
- **推荐**:
  - 保守: 30-40%
  - 平衡: 50%
  - 激进: 60-70%

## 优化结果解读

### 夏普比率
```
夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
```

- **> 1.5**: 优秀
- **1.0 - 1.5**: 良好
- **0.5 - 1.0**: 一般
- **< 0.5**: 需改进

### 其他重要指标
- **最大回撤**: < 20% 为佳
- **胜率**: > 40% 可接受 (缠论策略胜率可能较低，靠大赚)
- **盈亏比**: > 1.5 为佳
- **交易次数**: > 10笔才有统计意义

## 常见问题

### Q: 夏普比率为负数怎么办？
A: 意味着策略收益低于无风险利率或波动过大。尝试：
1. 增加 min_strokes 减少噪音交易
2. 收紧止损
3. 检查数据质量

### Q: 交易次数太少？
A: 尝试：
1. 降低 min_strokes
2. 检查策略是否过于保守

### Q: 回撤太大？
A: 尝试：
1. 收紧止损 (降低 stop_loss_pct)
2. 增加分批止盈 (降低 exit_ratio)
3. 增加 min_strokes 提高信号质量
