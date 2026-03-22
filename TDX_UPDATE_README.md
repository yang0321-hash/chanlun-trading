# 通达信数据自动更新工具

自动从 AKShare 获取最新行情数据并更新本地通达信 .day 文件。

## 文件说明

| 文件 | 说明 |
|-----|------|
| `update_tdx_data.py` | 完整版更新工具，支持更多选项 |
| `quick_update_tdx.py` | 简化版，快速更新常用股票 |
| `update_tdx.bat` | Windows批处理文件，可用于定时任务 |

## 安装依赖

```bash
pip install akshare pandas
```

## 使用方法

### 方式1: 直接运行Python脚本

```bash
# 快速更新前100只股票
python quick_update_tdx.py --limit 100

# 更新指定股票
python quick_update_tdx.py --code sh600519

# 只更新主要指数
python quick_update_tdx.py --index
```

### 方式2: 运行批处理文件

双击 `update_tdx.bat` 即可执行更新。

## 设置Windows定时任务

### 方法1: 任务计划程序（推荐）

1. 打开「任务计划程序」
   - Win+R 输入 `taskschd.msc` 回车

2. 创建基本任务
   - 点击「创建基本任务」
   - 名称: `通达信数据更新`
   - 触发器: 选择「每天」
   - 时间: 设置为 15:30 (交易结束后)
   - 操作: 选择「启动程序」
   - 程序: `D:\新建文件夹\claude\update_tdx.bat`

3. 高级设置
   - 勾选「不管用户是否登录都要运行」
   - 勾选「使用最高权限运行」

### 方法2: 任务计划程序（PowerShell）

```powershell
$action = New-ScheduledTaskAction -Execute "D:\新建文件夹\claude\update_tdx.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At 15:30
Register-ScheduledTask -TaskName "通达信数据更新" -Action $action -Trigger $trigger -Description "每天15:30自动更新通达信数据"
```

### 方法3: 使用Python定时

```python
# 创建 scheduler.py
import schedule
import time
import subprocess

def update_job():
    subprocess.run(["python", "quick_update_tdx.py"])

# 每天下午3点30分执行
schedule.every().day.at("15:30").do(update_job)

print("定时任务已启动...")
while True:
    schedule.run_pending()
    time.sleep(60)
```

## 数据更新策略

### 增量更新
- 只获取本地缺失的交易日数据
- 保留历史数据完整
- 自动去重

### 优先级更新
主要指数优先更新：
- sh000001 (上证指数)
- sh000300 (沪深300)
- sz399001 (深证成指)
- sz399006 (创业板指)

### 备份机制
每次更新前自动备份原文件为 `.bak`

## 注意事项

1. **数据源限制**: AKShare 有请求频率限制，大量更新时注意间隔
2. **交易时间**: 建议在交易日 15:00 之后执行更新
3. **网络连接**: 需要稳定的网络连接获取数据
4. **磁盘空间**: 确保有足够空间存储备份文件

## 故障排除

### 问题1: 模块未找到
```bash
pip install akshare pandas
```

### 问题2: 更新失败
- 检查网络连接
- 确认 TDX 路径正确
- 查看 .bak 备份文件恢复

### 问题3: 数据不完整
- AKShare 数据可能缺失历史数据
- 使用通达信软件手动下载补充

## 配置路径

如需修改 TDX 路径，编辑脚本中的变量：

```python
TDX_PATH = r"你的通达信路径"
```
