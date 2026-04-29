"""czsc桥接模块 — 通过子进程调用系统Python的czsc

当hermes venv没有czsc时，通过subprocess调用系统Python来获取bi信号。
支持单只和批量模式。
"""
import json
import os
import shutil
import subprocess
import tempfile
from typing import List, Dict, Optional

SYSTEM_PYTHON = r'C:\Users\nick0\AppData\Local\Programs\Python\Python312\python.exe'

# 单只股票脚本
_SCRIPT_SINGLE = '''
import os, sys, json
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

import pandas as pd
from czsc import CZSC, RawBar, Freq

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file, index_col=0, parse_dates=True)
n = len(df)

bars = []
for i in range(n):
    vol = float(df['volume'].iloc[i]) if 'volume' in df.columns else 0
    amt = float(df['close'].iloc[i]) * vol if vol > 0 else 0
    bars.append(RawBar(
        symbol='X', id=i,
        dt=pd.Timestamp(df.index[i]),
        freq=Freq.F30,
        open=float(df['open'].iloc[i]),
        close=float(df['close'].iloc[i]),
        high=float(df['high'].iloc[i]),
        low=float(df['low'].iloc[i]),
        vol=vol, amount=amt,
    ))

c = CZSC(bars)
bi_list = []
for bi in c.bi_list:
    if not bi.raw_bars or len(bi.raw_bars) < 2:
        continue
    direction = str(bi.direction)
    is_down = 1 if ('下' in direction) else 0
    bi_list.append({
        'start_idx': bi.raw_bars[0].id,
        'end_idx': bi.raw_bars[-1].id,
        'is_down': is_down,
    })

result = {'bi_count': len(c.bi_list), 'bi_list': bi_list, 'n': n}
with open(output_file, 'w') as f:
    json.dump(result, f)
'''

# 批量脚本: 从目录读取多个CSV文件，输出JSON
_SCRIPT_BATCH = '''
import os, sys, json
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)

import pandas as pd
from czsc import CZSC, RawBar, Freq

input_dir = sys.argv[1]
output_file = sys.argv[2]

results = {}
for fname in os.listdir(input_dir):
    if not fname.endswith('.csv'):
        continue
    code = fname[:-4]  # remove .csv
    try:
        df = pd.read_csv(os.path.join(input_dir, fname), index_col=0, parse_dates=True)
        n = len(df)
        if n < 120:
            continue

        bars = []
        for i in range(n):
            vol = float(df['volume'].iloc[i]) if 'volume' in df.columns else 0
            amt = float(df['close'].iloc[i]) * vol if vol > 0 else 0
            bars.append(RawBar(
                symbol=code, id=i,
                dt=pd.Timestamp(df.index[i]),
                freq=Freq.F30,
                open=float(df['open'].iloc[i]),
                close=float(df['close'].iloc[i]),
                high=float(df['high'].iloc[i]),
                low=float(df['low'].iloc[i]),
                vol=vol, amount=amt,
            ))

        c = CZSC(bars)
        bi_list = []
        for bi in c.bi_list:
            if not bi.raw_bars or len(bi.raw_bars) < 2:
                continue
            direction = str(bi.direction)
            is_down = 1 if ('下' in direction) else 0
            bi_list.append({
                'start_idx': bi.raw_bars[0].id,
                'end_idx': bi.raw_bars[-1].id,
                'is_down': is_down,
            })
        results[code] = bi_list
    except Exception:
        pass

with open(output_file, 'w') as f:
    json.dump(results, f)
'''


def get_czsc_bi(df, timeout=30) -> Optional[List[dict]]:
    """调用系统Python的czsc获取bi列表

    Args:
        df: OHLCV DataFrame with datetime index
        timeout: subprocess timeout in seconds

    Returns:
        [{'start_idx': int, 'end_idx': int, 'is_down': int}, ...] or None
    """
    if not os.path.exists(SYSTEM_PYTHON):
        return None

    tmp_in = None
    tmp_out = None
    try:
        tmp_in = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        tmp_out = tempfile.NamedTemporaryFile(suffix='.json', delete=False)

        df.to_csv(tmp_in.name)
        tmp_in.close()
        tmp_out.close()

        result = subprocess.run(
            [SYSTEM_PYTHON, '-c', _SCRIPT_SINGLE, tmp_in.name, tmp_out.name],
            capture_output=True, text=True, timeout=timeout,
        )

        if result.returncode != 0:
            return None

        with open(tmp_out.name, 'r') as f:
            data = json.load(f)

        return data.get('bi_list', [])

    except Exception:
        return None
    finally:
        for f in [tmp_in, tmp_out]:
            if f and os.path.exists(f.name):
                try:
                    os.unlink(f.name)
                except OSError:
                    pass


def get_czsc_bi_batch(data_map: Dict[str, 'pd.DataFrame'],
                      timeout=300) -> Dict[str, List[dict]]:
    """批量调用czsc — 一次subprocess处理所有股票

    Args:
        data_map: {code: OHLCV DataFrame}
        timeout: subprocess timeout in seconds

    Returns:
        {code: [{'start_idx', 'end_idx', 'is_down'}, ...]}
    """
    if not os.path.exists(SYSTEM_PYTHON):
        return {}

    tmp_dir = None
    tmp_out = None
    try:
        # 创建临时目录，保存所有CSV
        tmp_dir = tempfile.mkdtemp(prefix='czsc_batch_')
        tmp_out = tempfile.NamedTemporaryFile(suffix='.json', delete=False)

        for code, df in data_map.items():
            # 文件名用code (去掉/等特殊字符)
            safe_name = code.replace('/', '_').replace('\\', '_')
            df.to_csv(os.path.join(tmp_dir, f'{safe_name}.csv'))

        tmp_out.close()

        result = subprocess.run(
            [SYSTEM_PYTHON, '-c', _SCRIPT_BATCH, tmp_dir, tmp_out.name],
            capture_output=True, text=True, timeout=timeout,
        )

        if result.returncode != 0:
            return {}

        with open(tmp_out.name, 'r') as f:
            return json.load(f)

    except Exception:
        return {}
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if tmp_out and os.path.exists(tmp_out.name):
            try:
                os.unlink(tmp_out.name)
            except OSError:
                pass
