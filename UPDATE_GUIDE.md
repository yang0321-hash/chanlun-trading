# TDX数据更新指南

## 问题说明

当前TDX数据已过期（最新日期：2025-09-12），需要更新到最新数据。

## 更新方案

### 方案1：手动更新通达信数据（推荐）

1. 打开通达信软件
2. 等待自动下载完成数据
3. 数据会保存到通达信安装目录的 `vipdoc` 文件夹
4. 运行以下命令解析数据：

```bash
# 使用tdx-parser解析所有数据
node .claude/skills/tdx-parser/scripts/parse_tdx.js --all --format json
```

### 方案2：使用AKShare在线数据（需要网络）

安装依赖后运行：

```bash
pip install akshare
python update_tdx_akshare.py --limit 50
```

### 方案3：使用pytdx连接通达信接口

```bash
pip install pytdx
python update_tdx_data.py --market sh --limit 100
```

## 快速检查数据日期

```bash
# 检查本地数据日期
python -c "
import json, glob, os
files = glob.glob('test_output/*.json')
for f in sorted(files)[:5]:
    with open(f) as fp:
        data = json.load(fp)
        print(f'{os.path.basename(f)}: {data[-1][\"date\"]}')
"
```

## 验证更新效果

更新后运行选股器：

```bash
python final_screener.py --limit 50
```

查看结果中的日期列是否为最新日期（2026-03-20左右）。
