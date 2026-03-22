# 缠论选股器

识别A股市场日线级别的1买、2买、3买信号。

## 安装依赖

```bash
pip install pandas numpy loguru akshare scipy
```

## 使用方法

### 1. 扫描本地TDX数据

```bash
# 扫描test_output目录中的所有JSON文件
python run_screener.py --local test_output

# 扫描并保存结果到文件
python run_screener.py --local test_output --output signals.txt
```

### 2. 扫描指定股票（在线数据）

```bash
# 扫描指定股票列表
python run_screener.py --symbols sh600519,sz000001,sz000002
```

### 3. Python代码使用

```python
from scanner.chanlun_screener import ChanLunScreener, print_scan_result
import pandas as pd

# 初始化选股器
screener = ChanLunScreener(use_macd=True, min_klines=60)

# 扫描本地数据
result = screener.scan_local_files('test_output', '*.json')

# 或扫描指定股票
result = screener.scan_multiple(['sh600519', 'sz000001'])

# 打印结果
print_scan_result(result)
```

## 信号类型

| 类型 | 说明 | 条件 |
|------|------|------|
| 1买 | 第一类买点 | 价格跌破中枢下沿后开始反弹 |
| 2买 | 第二类买点 | 1买后回抽不破前低 |
| 3买 | 第三类买点 | 突破中枢后回踩确认 |

## 输出示例

```
============================================================
[缠论选股扫描结果]
扫描时间: 2026-03-21 21:30:00
扫描数量: 100 只
发现信号: 5 个
============================================================

[第一类买点 (1买)] - 3 个:
------------------------------------------------------------
  [sh600000] 浦发银行
  价格: ¥10.28 | 强度: 85%
  描述: 跌破中枢后开始反弹 + MACD背驰
  中枢: [13.47, 13.53]

[第三类买点 (3买)] - 2 个:
------------------------------------------------------------
  [sz000001] 平安银行
  价格: ¥11.72 | 强度: 65%
  描述: 突破中枢后回踩，中枢上沿支撑
============================================================
```
