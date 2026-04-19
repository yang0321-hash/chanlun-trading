# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python quantitative trading system based on ChanLun (缠论) theory for A-share markets. The system implements ChanLun's pattern recognition algorithms (分型 → 笔 → 线段 → 中枢) and provides backtesting capabilities with multiple strategy implementations.

## Running the System

### Quick Start
```bash
# 快捷启动 (桌面 gogogo 快捷方式)
gogogo.bat

# 环境配置
python setup_config.py              # Initialize configuration
python test_env_config.py           # Verify environment
```

### 四层Agent交易系统 (主力系统)

```bash
# 单独运行各Agent
python trading_agents/orchestrator.py --agent pre_market    # 盘前 (07:00)
python trading_agents/orchestrator.py --agent intraday       # 盘中 (09:30)
python trading_agents/orchestrator.py --agent daily_scan     # 每日扫描 (14:30)
python trading_agents/orchestrator.py --agent post_market    # 复盘 (21:30)
python trading_agents/orchestrator.py --agent all            # 全部执行
python trading_agents/orchestrator.py --status               # 查看状态

# 也可通过入口
python run_trading_system.py --status

# 工具
python trading_agents/weight_calibrator.py   # 评分权重标定
```

**Windows定时任务** (已注册，工作日自动运行):

| 任务 | 时间 | Agent | bat入口 |
|------|------|-------|---------|
| ChanLun_PreMarket | 07:00 | 盘前系统检测+竞价分析 | run_pre_market.bat |
| ChanLun_Intraday | 09:30 | 盘中实时监控+止损 | run_intraday.bat |
| ChanLunDaily | 14:30 | 全市场扫描+委员会 | run_daily.bat |
| ChanLun_PostMarket | 21:30 | 复盘+策略分析+明日规划 | run_post_market.bat |

### Backtesting
```bash
# Single stock backtest
python backtest_002600_tdx.py       # TDX local data
python backtest_weekly_daily.py sh600519  # Multi-timeframe

# AAA strategy (latest)
python aaa_backtest_v4_stable.py    # Stable version
python quick_aaa_test.py            # Quick test

# Batch backtest
python batch_backtest_watchlist.py  # Batch from watchlist.txt
python three_strategy_compare.py    # Compare strategies
```

### Real-time Monitoring
```bash
python chanlun_30min_monitor.py     # Intraday monitoring (standalone)
python chanlun_30min_monitor.py --scope all  # 全市场30min监控
```

### Data Management
```bash
python update_tdx_data.py           # Update TDX from AKShare
python quick_update_tdx.py          # Quick update
node .claude/skills/tdx-parser/scripts/parse_tdx.js --code sh600519 --format json
```

### Testing & Notifications
```bash
python test_notification.py         # Test notification system
python test_wechat_notify.py        # Test WeChat notify
```

## Output Directories

```
backtest_charts/     # Backtest visualization charts (PNG)
signals/             # Generated trade signals (JSON)
test_output/         # Test output files
.claude/temp/        # TDX converted JSON files
data/storage/        # AKShare data cache
```

## Configuration Files

```
config.yaml          # Main config (data source, notifications)
.env                 # Environment variables (API keys, tokens)
watchlist.txt        # Watchlist for batch backtest
```

## Architecture

### 交易系统全流程

```
全市场5253只A股 (tdx_all, 排除北证, 价格2.0-2000.0)
        ↓
   扫描器 scan_enhanced_v3
   (CC15引擎 + 1买/2买/3买识别 + 行业动量 + 30min确认)
        ↓ Top N 候选股
   投资委员会 6-Agent评估 (标定权重)
   (牛0.30/熊0.15/情绪0.10/行业0.20/扫描0.125/风控0.125)
        ↓ buy/hold/reject + 行业去相关
   持仓管理 PositionManager (自动记录/止损追踪)
        ↓
   出场管理 UnifiedExitManager (7层退出)
   (周线反转→缠论止损→固定止损→ATR跟踪→分批止盈→时间止损→信号反转)
        ↓
   复盘Agent (策略有效性+风险评估+明日规划)
```

### 四层Agent架构

| 层 | Agent | 文件 | 时间 | 职责 |
|---|-------|------|------|------|
| L0 | 主Agent | `trading_agents/orchestrator.py` | 全天 | 总调度+生命周期管理 |
| L1 | 盘前Agent | `trading_agents/pre_market.py` | 07:00 | 系统检测+盘前简报+竞价异动 |
| L2 | 盘中Agent | `trading_agents/intraday.py` | 09:30 | 实时监控+止损检查+午盘/尾盘总结 |
| L3 | 复盘Agent | `trading_agents/post_market.py` | 21:30 | 策略分析+风险评估+明日规划 |

**关键组件**:
- `trading_agents/position_manager.py`: 持仓管理器 — 自动跟踪买卖/止损上移/行业分布/最高价
- `trading_agents/weight_calibrator.py`: 权重标定 — 网格搜索最优6-Agent权重
- `daily_workflow.py`: 每日扫描+委员会+行业去相关+飞书推送
- `signals/agent_log.json`: 去重日志，防止重复推送
- `signals/positions.json`: 持仓数据 (由PositionManager自动维护)

### 委员会评分权重 (已标定)

```
technical_bull: 0.30    # 牛分析师 (缠论买点+趋势+量价)
technical_bear: 0.15    # 熊分析师 (风险+背离+高位警告)
sentiment:      0.10    # 市场情绪 (动量+量比+分位)
sector_rotation:0.20    # 行业轮动 (动量+成长性) ← 最被低估
scanner_base:   0.125   # 扫描器评分
risk_adjustment:0.125   # 风控惩罚
```

**决策阈值**: buy≥70+risk≤0.6 (强买) | buy≥55+risk≤0.4 (谨慎买) | reject<30

### 出场管理 (7层优先级)

1. 周线趋势反转 → 全部清仓
2. 缠论结构止损 (中枢ZD/笔低点)
3. 固定止损 (-5%)
4. ATR跟踪止损 (3×ATR)
5. 分批止盈 (强趋势10%/20%/35%, 正常5%/10%/15%, 弱势3%/6%/10%)
5.5. 结构加速出场 (缠论结构恶化卖50%)
6. 时间止损 (>60根K线无盈利)
7. 信号反转 (出现反向卖点)

### 去重和去相关

- **Agent去重**: `signals/agent_log.json` 记录每个Agent每日运行状态，`--force` 强制重跑
- **行业去相关**: `daily_workflow._diversify_results()` 同行业最多2只BUY推荐，第3只降为HOLD
- **信号去重**: 盘中Agent按 `symbol_type_price_hour` 去重，1小时内不重复推送

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
- **core/buy_sell_points.py**: BuySellPointDetector — 1买/2买/3买/类2买/类3买 + 置信度
- **core/trend_track.py**: TrendTrackDetector — 趋势追踪
- **indicator/macd.py**: MACD for 背驰 detection.
- **data/hybrid_source.py**: HybridSource — TDX本地日线 + Sina在线分钟线/实时行情
- **scan_enhanced_v3.py**: 全市场扫描器 — 1买/2买/3买 + 行业动量 + 30min确认
- **agents/investment_committee.py**: 6-Agent投资委员会入口
- **agents/committee_agents.py**: Bull/Bear/Sentiment/Sector/Risk/Fund 6个Agent
- **agents/scoring.py**: 评分权重 + 决策阈值 + 仓位计算 (权重已标定)
- **strategies/unified_exit_manager.py**: 7层出场管理
- **strategies/unified_config.py**: 统一策略配置 (保守/激进/多周期预设)
- **chanlun_unified/stock_pool.py**: 股票池管理 (tdx_all=5253只, 排除北证)

### Strategy Pattern

All strategies inherit from `backtest.strategy.Strategy`:
- Implement `on_bar(bar, symbol, index, context) → Signal`
- Access position via `self.position[symbol]`
- Access cash via `self.cash`
- Return `Signal(signal_type, symbol, datetime, price, quantity, reason)` for trades, `None` for no action

### Available Strategies

- `strategies/chan_strategy.py`: Basic ChanLun strategy with 6 buy/sell point types
- `strategies/aaa_strategy.py`: AAA scoring system (Trend + Momentum + Volume)
  - Entry: AAA score > threshold
  - Exit: AAA score < threshold or stop loss
- `strategies/weekly_daily_strategy.py`: Multi-timeframe (周线+日线) strategy
  - Entry: 周线2买
  - Stop Loss: 跌破周线1买最低点
  - Partial Exit: 日线MACD顶背离 (sell 50%)
  - Full Exit: 日线2卖
- `strategies/hot_money_limit_up_strategy.py`: Hot money tracking with limit-up detection
- `strategies/intelligent_chanlun_strategy.py`: Adaptive ChanLun with dynamic parameters
- `strategies/multilevel_chan_strategy.py`: Multi-level ChanLun with trend confirmation
- `strategies/optimized_chan_strategy.py`: Optimized version with filters
- `strategies/advanced_chan_strategy.py`: Advanced with re-entry rules

### Data Sources

- **data/hybrid_source.py**: HybridSource (主力数据源)
  - TDX本地: 日线(vipdoc/sh/lday + sz/lday), 1min/5min分钟线
  - Sina在线: 30min/5min/1min K线 + 实时报价
  - 本地路径: `tdx_data/sh/lday/`, `tdx_data/sz/lday/`
- **data/akshare_source.py**: AKShare (指数日线、竞价数据)
- **data/tdx_source.py**: TDX数据源基类

**代理问题**: Clash代理会拦截eastmoney/Sina请求，所有模块在import前执行:
```python
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
```
Sina请求需 `session.trust_env = False`。

### 通达信数据配置

**配置路径** (在 `config.yaml` 中设置):
```yaml
tdx:
  data_path: "D:/your_tdx_path/vipdoc"
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

### 买卖点加分项 (信号质量评估)

评估买点信号质量时，以下因素提升置信度：

**1买加分项**:
| 加分项 | 条件 | 说明 |
|--------|------|------|
| 趋势背驰 | 下跌趋势末段，MACD面积递减 | **注意**: 强势行情中1买趋势背驰不安全，仅弱势行情中趋势背驰才可靠 |
| 离开段放量 | 离开中枢的那一笔成交量放大 | 资金进场的信号 |
| 底分型放量 | 1买位置的底分型K线放量 | 底部有资金承接 |

**1买市场环境要求**:
- ✅ 弱势行情(大盘下跌趋势) → 1买+趋势背驰 = 高概率底部反转
- ⚠️ 强势行情(大盘上升趋势) → 1买可能是中继回调，趋势背驰信号失效风险高
- → 强势行情中应优先寻找2买和3买，而非1买

**2买加分项**:

二买 = 一买后第一次空头反抽不创新低。本质是市场对一买的**确认过程**。

| 加分项 | 条件 | 说明 |
|--------|------|------|
| 回踩深度浅 | 回调不破一买低点，回踩越浅越强 | 浅回踩=多头越强，深回踩=多头弱 |
| 回踩缩量 | 回调笔成交量明显萎缩 | 抛压衰竭，卖盘耗尽 |
| 底分型放量确认 | 二买底分型第三根K线放量收高 | 资金承接有力 |
| 中枢内企稳 | 价格在中枢ZD附近止跌 | 结构支撑明确 |
| 一买后上涨温和放量 | 一买→二买之间上涨段放量 | 资金持续流入 |
| MACD回抽零轴 | DIF/DEA回抽零轴后再次金叉 | 趋势动能恢复 |

**2买三种强度分类**:
| 强度 | 名称 | 回踩位置 | 成交量 | 后续预期 |
|------|------|---------|--------|----------|
| 最强 | **2买3买重叠** | 几乎不回踩，直接突破中枢 | 明显缩量 | 直接趋势向上，空间最大 |
| 中等 | **类2买(中枢内)** | 回踩至中枢中部以上 | 温和缩量 | 震荡后向上 |
| 较弱 | **中枢下2买** | 回踩接近一买低点 | 缩量不明显 | 可能继续中枢震荡 |

**3买加分项**:

三买 = 离开上涨中枢后，第一次空头反拉不进入中枢。

| 加分项 | 条件 | 说明 |
|--------|------|------|
| 回踩不进中枢 | 回调低点 > ZG | 标准三买，确认突破有效 |
| 突破放量 | 离开中枢时放量上攻 | 资金力度强，突破真实 |
| 回踩缩量 | 回调笔成交量萎缩 | 筹码稳定，抛压轻 |
| 再次放量确认 | 回踩结束后再次放量上涨 | 三买最终确认 |
| 次级别背驰确认 | 30min级别回踩出现背驰 | 精确入场时机 |

**3买三种强度分类**:
| 强度 | 名称 | 回踩位置 | 说明 |
|------|------|---------|------|
| 最强 | **强三买** | 回踩在GG之上(不破中枢最高点) | 回调最小，多头最强 |
| 标准 | **标准三买** | 回踩在ZG~GG之间 | 标准形态，不破ZG |
| 偏弱 | **弱三买** | 回踩在ZD~ZG之间(进入中枢) | 回调较深，需谨慎 |

**3买黄金分割加分**: 离开中枢的上涨段从高点回撤时，回踩幅度未跌破0.618位置 → 额外加分。说明多头强势，回调幅度健康，趋势延续概率高。

**3买量价节奏**: 放量突破 → 缩量回踩 → 放量确认 = 健康节奏

**2买3买重叠 (最强形态)**: 二买回踩不进入中枢 = 同时满足二买和三买条件，这是最强的买点形态。

## 核心交易原则

**周线定方向，日线30分钟找买点。**

这是缠论多级别联立的核心方法论。三个级别各有分工，不可混淆：

### 周线 — 定方向 (战略层)

周线决定大趋势方向。只有周线方向明确时，才在日线级别寻找顺势买点。

| 周线状态 | 含义 | 操作策略 |
|----------|------|----------|
| **周线上升趋势** (MA5>MA20, 中枢上移) | 多头 | 只做多，找日线/30min买点 |
| **周线下降趋势** (MA5<MA20, 中枢下移) | 空头 | 不做多，或找周线1买抄底 |
| **周线盘整** (中枢震荡) | 方向不明 | 小仓位做日线中枢震荡，等突破 |
| **周线2买确认** | 趋势转折确认 | 重点加仓信号 |
| **周线1买+底背驰** | 底部反转信号 | 开始建仓 |

**周线止损**: 跌破周线1买最低点 → 立即清仓（趋势判断错误）

**周线关键判断**:
- 周线笔的方向 → 当前大趋势
- 周线中枢位置(ZG/ZD) → 支撑/压力位
- 周线MACD → 趋势强度和背驰信号
- 周线与日线方向共振 → 高概率行情

### 日线 — 找买点区域 (战术层)

在周线方向确定后，日线级别识别具体的买卖点信号。

| 买点 | 定义 | 周线配合要求 | 可靠性 |
|------|------|-------------|--------|
| **日线1买** | 下跌趋势最后一个中枢底背驰 | 周线接近底部/已出1买 | 中 (抄底，风险较大) |
| **日线2买** | 1买后回踩不破前低 | 周线趋势向上或盘整 | **高 (最佳入场)** |
| **日线3买** | 突破中枢后回踩不进入 | 周线多头趋势 | 高 (顺势) |

**日线止损**:
- 缠论结构止损: 中枢ZD 或 最近笔低点
- 固定止损: -5%
- 3买止损: 中枢上沿(ZG)

### 30分钟 — 精确入场 (执行层)

30分钟级别用于精确把握日线买点的具体入场时机。

- 日线出现2买信号 → 切到30分钟确认
- 30分钟也出现买点 → **共振入场**，置信度最高
- 30分钟无买点 → 降低仓位(50%)或等确认
- 30分钟出现卖点 → 即使日线买点未失效也应警惕

### 多级别联立规则

```
周线多头 + 日线2买 + 30min确认 → 满仓入场 (最高置信度)
周线多头 + 日线2买 + 30min无确认 → 半仓入场
周线盘整 + 日线2买 + 30min确认 → 半仓入场
周线空头 + 日线任何买点 → 不入场 (逆势)
周线多头 + 日线3买 + 30min确认 → 满仓入场
周线1买 + 日线1买 → 试探仓 (抄底)
```

### 实际应用流程

1. **每周一** (盘前Agent): 扫描所有候选股的周线状态，标记周线多头/盘整/空头
2. **每日** (扫描器): 仅对周线多头/盘整的股票，寻找日线级别买点
3. **盘中** (盘中Agent): 对日线出买点的股票，检查30min级别确认
4. **持仓管理**: 周线趋势反转 → 全部清仓；日线止损 → 卖出个股

## Dependencies

### Python Requirements
```bash
pip install akshare pandas numpy plotly loguru python-dotenv scipy matplotlib
```

**Common Issues:**
- scipy compatibility: If encountering errors, use `pip install scipy==1.11.4`
- AKShare `stock_zh_a_spot_em()` 会因Clash代理失败 → 使用Sina作为fallback
- `mootdx` 可选安装，用于盘中Agent直连通达信TCP获取30min数据

### Node.js (for TDX parsing)
```bash
node --version  # Requires Node.js 14+
```

### 环境变量 (.env)
```
FEISHU_WEBHOOK_URL=...           # Hermes飞书机器人
CHANLUN_FEISHU_WEBHOOK_URL=...   # 缠论专用飞书机器人
TUSHARE_TOKEN=...                # Tushare Pro API
TRADINGAGENTS_TOKEN=...          # TradingAgents云端API
DEEPSEEK_API_KEY=...             # DeepSeek API
```

## Available Skills

Use these skills via `/skill-name` command:

```bash
/tdx-parser          # Parse TongDaXin .day files to JSON
/tdx-updater         # Update TDX data from AKShare
/backtest            # Run strategy backtest
/backtest-viz        # Generate backtest charts
/chanlun             # ChanLun analysis with visualization
/stock-name-matcher  # Match stock codes to company names
/投委会              # 投资委员会评估（6Agent+缠论买卖点）
```

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

## Coding Conventions

### 多CC实例协调 (3个CC + Hermes同时运行)

**文件认领机制**: 编辑核心文件前必须先认领，防止多实例冲突。

```bash
# 认领文件 (开始编辑前)
python .claude/coordination.py claim scan_enhanced_v3.py "优化3买逻辑"

# 查看当前所有认领
python .claude/coordination.py status

# 释放文件 (编辑完成后)
python .claude/coordination.py release scan_enhanced_v3.py

# 释放当前实例所有认领
python .claude/coordination.py release-all
```

**规则**:
1. 编辑 `.py` 文件前先 `claim`，编辑完 `release`
2. 锁1小时自动过期，不怕忘记释放
3. 如果看到 LOCKED 警告，先和对方确认再继续
4. `signals/` 目录不需要认领（时间戳命名，天然不冲突）
5. Hermès只读signals，不修改代码，无需认领

**分工建议** (3个CC实例):
| CC实例 | 建议负责模块 | 核心文件 |
|--------|-------------|---------|
| CC1 | 扫描器+信号 | scan_enhanced_v3.py, core/buy_sell_points.py |
| CC2 | 策略+出场 | strategies/*, agents/scoring.py |
| CC3 | 数据+Agent | data/*, trading_agents/* |

**冲突处理**:
- PreToolUse hook 自动检测锁，给出警告
- 如果需要紧急修改被锁文件: `python .claude/coordination.py release <file>` (强释)
- Git冲突: 各自commit不同文件，避免merge冲突

### Agent开发规范

所有 `trading_agents/` 下的Agent必须遵循:

1. **模块顶部清除代理**:
```python
import sys, os
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
from dotenv import load_dotenv; load_dotenv()
```

2. **使用PositionManager管理持仓** (不要直接读写positions.json):
```python
from trading_agents.position_manager import PositionManager
pm = PositionManager()
pm.buy(code, name, price, shares, stop_price, buy_point_type='2buy')
pm.sell(code)
pm.get_all_positions()
```

3. **去重机制**: 运行前检查 `check_today_done(agent_name)`, 完成后 `mark_done(agent_name)`

4. **通知**: 使用 `CHANLUN_FEISHU_WEBHOOK_URL` (缠论专用)，不与Hermes共用

5. **出场管理**: 盘中止损检查应通过 `UnifiedExitManager`，不要只做简单 `price <= stop`

### 文件组织

- `trading_agents/`: 四层Agent (orchestrator, pre_market, intraday, post_market)
- `agents/`: 委员会Agent (investment_committee, committee_agents, scoring)
- `core/`: 缠论核心引擎
- `strategies/`: 策略实现 + 过滤器 + 出场管理
- `signals/`: 运行时输出 (JSON/TXT)，自动生成不手动编辑
- `data/`: 数据源
- `backtest/`: 回测引擎
