# 缠论交易系统 (ChanLun Trading System)

[![CI/CD](https://github.com/yang0321-hash/chanlun-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/yang0321-hash/chanlun-trading/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

基于缠论的Python量化交易系统，支持A股市场的数据分析、策略回测和可视化。

## 功能特性

### 核心算法
- **分型识别**: 自动识别顶分型和底分型
- **笔生成**: 根据分型生成笔
- **线段识别**: 根据笔生成线段
- **中枢识别**: 识别价格中枢结构
- **背驰判断**: 结合MACD判断背驰

### 数据支持
- AKShare免费数据源
- 支持A股日线、分钟线数据
- 前复权/后复权处理

### 回测系统
- 完整的回测引擎
- 绩效指标计算（收益率、夏普率、最大回撤等）
- 交易信号记录

### 可视化
- K线图绘制
- 分型、笔、线段、中枢标注
- 买卖点标注

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd chanlun-trading-system

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```python
from data import AKShareSource
from core import KLine, FractalDetector, StrokeGenerator
from backtest import BacktestEngine, ChanLunStrategy

# 1. 获取数据
source = AKShareSource()
df = source.get_kline('000001', period='daily', adjust='qfq')

# 2. 分析缠论要素
kline = KLine.from_dataframe(df)
detector = FractalDetector(kline)
fractals = detector.get_fractals()

generator = StrokeGenerator(kline)
strokes = generator.get_strokes()

# 3. 回测策略
engine = BacktestEngine()
engine.add_data('000001', df)
engine.set_strategy(ChanLunStrategy())
results = engine.run()

print(f"总收益率: {results['total_return']:.2%}")
```

## 项目结构

```
chanlun-trading-system/
├── core/              # 缠论核心算法
│   ├── kline.py      # K线处理
│   ├── fractal.py    # 分型识别
│   ├── stroke.py     # 笔生成
│   ├── segment.py    # 线段识别
│   ├── pivot.py      # 中枢识别
│   └── visualization.py # 可视化
├── data/             # 数据获取
│   ├── source.py     # 数据源接口
│   ├── akshare_source.py
│   └── storage.py    # 本地存储
├── indicator/        # 技术指标
│   └── macd.py       # MACD指标
├── backtest/         # 回测系统
│   ├── engine.py     # 回测引擎
│   ├── strategy.py   # 策略基类
│   ├── metrics.py    # 绩效计算
│   └── report.py     # 报告生成
├── strategies/       # 策略实现
│   └── chan_strategy.py  # 缠论策略
├── utils/            # 工具类
├── examples/         # 示例代码
└── config.yaml       # 配置文件
```

## 配置说明

编辑 `config.yaml` 文件配置系统参数：

```yaml
# 数据配置
data:
  source: akshare
  storage_path: ./data/storage
  default_period: daily
  fq: qfq

# 缠论参数
chanlun:
  fractal_confirm: true
  min_stroke_bars: 5
  min_segment_strokes: 3
  strict_mode: true

# 回测配置
backtest:
  initial_capital: 100000
  commission: 0.0003
  slippage: 0.0001
  min_unit: 100
```

## 策略说明

### 缠论买卖点

- **第一类买点**: 下跌趋势中，最后中枢下方的底背驰点
- **第一类卖点**: 上涨趋势中，最后中枢上方的顶背驰点
- **第二类买点**: 第一类买点后，回抽不破前低的点
- **第二类卖点**: 第一类卖点后，反弹不破前高的点
- **第三类买点**: 突破中枢后回踩不破中枢上沿
- **第三类卖点**: 跌破中枢后反弹不破中枢下沿

## 注意事项

1. 本系统仅供学习研究使用，不构成任何投资建议
2. 回测结果不代表实盘表现，实盘交易需谨慎
3. 免费数据源可能有延迟或错误，建议使用付费数据源
4. 策略参数需要根据市场环境调整

## 许可证

MIT License
