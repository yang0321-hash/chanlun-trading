"""TDX数据前复权工具 — 用Tushare复权因子修正TDX本地数据

问题: TDX .day 文件存储的是未复权原始价格
      除权除息产生的价格跳空 → 缠论分型/笔全部被干扰

解决: 从Tushare获取 adj_factor，乘以原始价格得到前复权价格
      缓存复权因子到本地，避免重复API调用

用法:
  from data.tdx_adj_factor import adjust_tdx_daily

  # 修正单只股票
  df_adjusted = adjust_tdx_daily(df_raw, '000001.SZ')

  # 批量预缓存
  cache_adj_factors(['300001.SZ', '300003.SZ', ...])
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# 加载系统Python的tushare
SYSTEM_SITE = r'C:\Users\nick0\AppData\Local\Programs\Python\Python312\Lib\site-packages'
if SYSTEM_SITE not in sys.path:
    sys.path.insert(0, SYSTEM_SITE)

# 清除代理
for k in list(os.environ.keys()):
    if k.lower() in ('http_proxy', 'https_proxy'):
        del os.environ[k]
os.environ['NO_PROXY'] = '*'

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'adj_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_tushare_pro():
    """获取Tushare pro API"""
    import tushare as ts

    token = os.environ.get('TUSHARE_TOKEN', '')
    if not token:
        env_file = PROJECT_ROOT / '.env'
        if env_file.exists():
            for line in env_file.read_text(encoding='utf-8').splitlines():
                if 'TUSHARE' in line.upper() and '=' in line:
                    _, v = line.split('=', 1)
                    token = v.strip()
                    break
    if not token:
        return None

    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = "http://111.170.34.57:8010/"
    return pro


def fetch_adj_factor(ts_code: str, start_date: str = None) -> pd.DataFrame:
    """从Tushare获取复权因子

    Args:
        ts_code: '000001.SZ'
        start_date: '20200101'

    Returns:
        DataFrame: trade_date, adj_factor
    """
    pro = _get_tushare_pro()
    if pro is None:
        return pd.DataFrame()

    try:
        df = pro.adj_factor(ts_code=ts_code, start_date=start_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
        return df
    except Exception:
        return pd.DataFrame()


def load_cached_adj(ts_code: str) -> Optional[pd.Series]:
    """从本地缓存加载复权因子"""
    cache_file = CACHE_DIR / f'{ts_code}_adj.json'
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding='utf-8'))
        dates = data.get('dates', [])
        factors = data.get('factors', [])
        if len(dates) != len(factors):
            return None
        s = pd.Series(factors, index=pd.to_datetime(dates))
        s.index.name = 'trade_date'
        return s
    except Exception:
        return None


def save_cached_adj(ts_code: str, adj_series: pd.Series):
    """保存复权因子到本地缓存"""
    cache_file = CACHE_DIR / f'{ts_code}_adj.json'
    data = {
        'ts_code': ts_code,
        'updated': pd.Timestamp.now().isoformat(),
        'dates': [str(d.date()) for d in adj_series.index],
        'factors': [float(f) for f in adj_series.values],
    }
    cache_file.write_text(json.dumps(data), encoding='utf-8')


def get_adj_factor(ts_code: str, force_refresh: bool = False) -> Optional[pd.Series]:
    """获取复权因子(缓存优先)

    Args:
        ts_code: '000001.SZ'
        force_refresh: 强制刷新
    """
    if not force_refresh:
        cached = load_cached_adj(ts_code)
        if cached is not None and len(cached) > 0:
            # 检查是否需要更新(最新数据距今天超过3天)
            latest = cached.index[-1]
            if (pd.Timestamp.now() - latest).days < 3:
                return cached

    # 从Tushare获取
    adj_df = fetch_adj_factor(ts_code)
    if adj_df.empty:
        return load_cached_adj(ts_code)  # 回退到旧缓存

    adj_series = adj_df['adj_factor']
    save_cached_adj(ts_code, adj_series)
    return adj_series


def adjust_tdx_daily(df_raw: pd.DataFrame, ts_code: str,
                     adj_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """对TDX原始日线数据做前复权

    前复权: 最新一天的价格不变，历史价格向下调整
    公式: 复权价 = 原始价 × (当日因子 / 最新因子)

    Args:
        df_raw: TDX原始日线(DatetimeIndex, OHLCV)
        ts_code: '000001.SZ'
        adj_series: 可选，预加载的复权因子

    Returns:
        DataFrame: 前复权后的OHLCV
    """
    if df_raw.empty:
        return df_raw

    if adj_series is None:
        adj_series = get_adj_factor(ts_code)
        if adj_series is None or adj_series.empty:
            return df_raw  # 无法获取复权因子，返回原始数据

    # 按日期对齐
    df = df_raw.copy()

    # 找到每个交易日对应的复权因子
    # 用最近的因子填充(处理非交易日)
    common_idx = df.index.intersection(adj_series.index)
    if common_idx.empty:
        return df_raw

    # reindex到df的日期
    adj_aligned = adj_series.reindex(df.index, method='ffill')
    adj_aligned = adj_aligned.fillna(method='bfill')

    if adj_aligned.isna().all():
        return df_raw

    # 前复权: 以最新因子为基准
    latest_factor = adj_aligned.iloc[-1]
    if latest_factor <= 0:
        return df_raw

    ratio = adj_aligned / latest_factor

    # 对OHLC做复权(保持volume不变)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df[col] * ratio

    return df


def cache_adj_factors(ts_codes: list, verbose: bool = True):
    """批量预缓存复权因子"""
    pro = _get_tushare_pro()
    if pro is None:
        print("无法连接Tushare")
        return

    success = 0
    for i, code in enumerate(ts_codes):
        adj = get_adj_factor(code, force_refresh=True)
        if adj is not None and len(adj) > 0:
            success += 1
        if verbose and (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(ts_codes)}] 已缓存 {success} 只')
        time.sleep(0.35)  # Tushare限频

    if verbose:
        print(f'  完成: {success}/{len(ts_codes)} 只缓存成功')


def to_ts_code(code: str) -> str:
    """各种格式 → '000001.SZ'"""
    code = code.strip().replace('sz', '').replace('sh', '').replace('SZ', '').replace('SH', '')
    if '.' in code:
        return code
    if code.startswith('6'):
        return f'{code}.SH'
    elif code.startswith(('0', '3')):
        return f'{code}.SZ'
    elif code.startswith(('8', '4')):
        return f'{code}.BJ'
    return f'{code}.SZ'
