# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python quantitative trading system based on ChanLun (缠论) theory for A-share markets. The system implements ChanLun's pattern recognition algorithms (分型 → 笔 → 线段 → 中枢) and provides backtesting capabilities with multiple strategy implementations.

## Running the System

```bash
# Quick demo with online data (AKShare)
python run_demo.py

# Basic usage example
python examples/basic_usage.py

# Backtest with TDX local data
python backtest_002600_tdx.py

# Compare multiple strategies
python three_strategy_compare.py

# Run backtest and generate report
python generate_optimization_report.py
```

## Architecture

### Core Data Flow

```
Raw OHLCV Data
      ↓
KLine (with inclusion merging)
      ↓
Fractal (顶分型/底分型)
      ↓
Stroke (笔)
      ↓
Segment (线段)
      ↓
Pivot/Center (中枢)
      ↓
Buy/Sell Signals (1买/2买/3买, 1卖/2卖/3卖)
```

### Key Modules

- **core/kline.py**: K-line processing with ChanLun's inclusion relationship merging (包含关系). Use `KLine.from_dataframe(df)` to convert OHLCV data.
- **core/fractal.py**: Identifies 顶分型 and 底分型 patterns.
- **core/stroke.py**: Generates 笔 from fractals.
- **core/segment.py**: Generates 线段 from strokes.
- **core/pivot.py**: Identifies 中枢 (price centers/pivots).
- **indicator/macd.py**: MACD for 背驰 detection.

### Strategy Pattern

All strategies inherit from `backtest.strategy.Strategy`:
- Implement `on_bar(bar, symbol, index, context) → Signal`
- Access position via `self.position[symbol]`
- Access cash via `self.cash`
- Return `Signal(signal_type, symbol, datetime, price, quantity, reason)` for trades, `None` for no action

### Available Strategies

- `strategies/chan_strategy.py`: Basic ChanLun strategy with 6 buy/sell point types
- `strategies/weekly_daily_strategy.py`: Multi-timeframe (周线+日线) strategy
  - Entry: 周线2买
  - Stop Loss: 跌破周线1买最低点
  - Partial Exit: 日线MACD顶背离 (sell 50%)
  - Full Exit: 日线2卖
- `strategies/multilevel_chan_strategy.py`: Multi-level ChanLun with trend confirmation
- `strategies/optimized_chan_strategy.py`: Optimized version with filters
- `strategies/advanced_chan_strategy.py`: Advanced with re-entry rules

### Data Sources

- **data/akshare_source.py**: Free A-share data via AKShare (default)
- **data/tdx_source.py**: Local TongDaXin (通达信) data files

### 通达信数据配置

**本地通达信路径**:
```
D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc
```

**数据目录结构**:
```
vipdoc/
├── sh/lday/    # 上海日线数据 (sh600000.day, sh600519.day, ...)
├── sz/lday/    # 深圳日线数据 (sz000001.day, sz002600.day, ...)
└── bj/lday/    # 北京日线数据
```

**自动解析流程**:
```bash
# 1. 运行回测时自动检查JSON数据
python backtest_weekly_daily.py sh600000

# 2. 如果JSON不存在，自动调用tdx-parser解析.day文件
node .claude/skills/tdx-parser/scripts/parse_tdx.js \
    --input "D:/大侠神器2.0/.../vipdoc" \
    --code sh600000 \
    --output test_output \
    --format json

# 3. 输出: test_output/sh600000.day.json
# 4. 继续执行回测
```

**手动解析数据**:
```bash
# 单股解析
node .claude/skills/tdx-parser/scripts/parse_tdx.js \
    --code sh600519 --format json

# 批量解析所有
node .claude/skills/tdx-parser/scripts/parse_tdx.js --all --format json
```

### Backtest Engine

`backtest/engine.BacktestEngine`:
- `add_data(symbol, df)`: Add OHLCV DataFrame
- `set_strategy(strategy)`: Set strategy instance
- `run()`: Execute backtest, returns metrics dict

Config via `BacktestConfig(initial_capital, commission, slippage, min_unit)`

## Important ChanLun Concepts

- **包含关系 (Inclusion)**: When one K-line's high/low fully contains another's. Direction of merge depends on trend (up: take max high/max low, down: take min high/min low)
- **分型**: 顶分型 = middle K highest, 底分型 = middle K lowest (3 K-line pattern)
- **笔**: Requires alternating top/bottom fractals with minimum bar separation
- **线段**: Generated from strokes, has its own break/confirmation rules
- **中枢**: 3+ overlapping strokes forming a price range
- **买卖点**:
  - 1买: Last pivot bottom with divergence in downtrend
  - 2买: Pullback after 1买 that doesn't break the low
  - 3买: Breakout above pivot that doesn't re-enter
  - Mirror for sell points

## Python Requirements

```bash
pip install akshare pandas numpy plotly loguru python-dotenv scipy matplotlib
```

For TDX data parsing, use the `/tdx-parser` skill to convert `.day` files to JSON.

## Stock Code to Name Matching

**CRITICAL**: Always verify stock codes against company names using the built-in skill.

### Common Mistake - Code Confusion

| Incorrect | Correct |
|-----------|---------|
| sz002600 = 驰宏锌锗 ❌ | sz002600 = **领益智造** ✅ |
| (600497 is 驰宏锌锗) | (消费电子行业) |

### How to Match Stock Names

When referencing stocks in output, reports, or documentation:

1. **Use the stock-name-matcher skill**:
   ```bash
   /stock-name-matcher sz002600
   ```

2. **Reference the data file**: `.claude/skills/stock-name-matcher/stock_data.json`

3. **Code format rules**:
   - `sh`/`600xxx` = Shanghai Exchange (上海证券交易所)
   - `sz`/`000xxx` = Shenzhen Mainboard (深圳主板)
   - `sz`/`002xxx` = Shenzhen SME (深圳中小板)
   - `sz`/`300xxx` = Shenzhen ChiNext (深圳创业板)
   - `bj`/`8xxxxx` = Beijing Exchange (北京证券交易所)

### Verification Before Output

Before stating a stock name in any response:
1. Cross-check with `stock_data.json`
2. Or use the skill to verify
3. Format as: `sz002600 (领益智造)`
