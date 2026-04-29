"""
Kronos 模型封装

封装 Kronos 基础模型的加载、预测、缓存和 GPU 管理。
所有 torch/kronos 依赖在内部懒加载，确保无依赖时系统照常运行。

用法:
    predictor = KronosPredictor(KronosConfig())
    if predictor.is_available():
        pred_df = predictor.predict(df, timestamps, pred_len=5)
"""

import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from .kronos_config import KronosConfig


class KronosPredictor:
    """Kronos 模型封装，提供懒加载、缓存、单股/批量预测"""

    def __init__(self, config: KronosConfig):
        self._config = config
        self._model = None           # Kronos model
        self._tokenizer = None       # KronosTokenizer
        self._predictor = None       # KronosPredictor (kronos 库内部)
        self._device = None          # 'cuda' 或 'cpu'
        self._loaded = False
        self._available = None       # 缓存依赖检查结果

        # LRU 缓存: key -> (prediction_df, timestamp)
        self._cache: OrderedDict = OrderedDict()

    # ---------- 公共接口 ----------

    def is_available(self) -> bool:
        """检查 torch 和 kronos 依赖是否可用"""
        if self._available is not None:
            return self._available
        try:
            import torch  # noqa: F401
            # kronos 库可能叫不同名字，尝试多种导入
            self._available = self._try_import_kronos()
        except ImportError:
            self._available = False
            logger.warning("Kronos 依赖不可用 (需要 torch 和 kronos)，AI 预测功能已禁用")
        return self._available

    def predict(
        self,
        df: pd.DataFrame,
        timestamps: pd.Series,
        pred_len: int = None,
        symbol: str = '',
    ) -> Optional[pd.DataFrame]:
        """
        单股预测

        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume, [amount])
            timestamps: 对应的时间戳 Series
            pred_len: 预测K线根数 (默认用配置值)
            symbol: 股票代码 (用于缓存 key)

        Returns:
            预测 DataFrame 或 None (不可用时)
        """
        pred_len = pred_len or self._config.pred_len

        # 检查缓存
        cache_key = self._cache_key(symbol, timestamps)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # 确保模型已加载
        if not self._ensure_loaded():
            return None

        # 准备输入
        df_input, ts_input = self._prepare_input(df, timestamps)

        try:
            pred_df = self._predictor.predict(
                df=df_input,
                x_timestamp=ts_input,
                y_timestamp=self._build_future_timestamps(ts_input, pred_len),
                pred_len=pred_len,
                T=self._config.temperature,
                top_p=self._config.top_p,
                sample_count=self._config.sample_count,
            )
            self._set_cached(cache_key, pred_df)
            return pred_df
        except Exception as e:
            logger.error(f"Kronos 预测失败 [{symbol}]: {e}")
            return None

    def predict_batch(
        self,
        df_list: List[pd.DataFrame],
        timestamps_list: List[pd.Series],
        pred_len: int = None,
        symbols: List[str] = None,
    ) -> List[Optional[pd.DataFrame]]:
        """
        批量预测 (并行 GPU 推理)

        Args:
            df_list: 多个 OHLCV DataFrame
            timestamps_list: 对应的时间戳列表
            pred_len: 预测K线根数
            symbols: 股票代码列表 (用于缓存)

        Returns:
            预测结果列表，不可用的返回 None
        """
        pred_len = pred_len or self._config.pred_len
        symbols = symbols or [''] * len(df_list)
        results = [None] * len(df_list)

        if not self._ensure_loaded():
            return results

        # 分离已缓存和需预测的
        to_predict_indices = []
        to_predict_dfs = []
        to_predict_ts = []

        for i, (df, ts, sym) in enumerate(zip(df_list, timestamps_list, symbols)):
            cache_key = self._cache_key(sym, ts)
            cached = self._get_cached(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                to_predict_indices.append(i)
                df_input, ts_input = self._prepare_input(df, ts)
                to_predict_dfs.append(df_input)
                to_predict_ts.append(ts_input)

        if not to_predict_dfs:
            return results

        # 批量预测需要统一长度和 pred_len
        try:
            # 检查是否所有序列等长
            lengths = [len(df) for df in to_predict_dfs]
            if len(set(lengths)) == 1:
                # 等长 → 可用 predict_batch
                future_ts = self._build_future_timestamps(to_predict_ts[0], pred_len)
                pred_list = self._predictor.predict_batch(
                    df_list=to_predict_dfs,
                    x_timestamp_list=to_predict_ts,
                    y_timestamp_list=[future_ts] * len(to_predict_dfs),
                    pred_len=pred_len,
                    T=self._config.temperature,
                    top_p=self._config.top_p,
                    sample_count=self._config.sample_count,
                )
                for idx, pred_df in zip(to_predict_indices, pred_list):
                    results[idx] = pred_df
                    sym = symbols[idx]
                    cache_key = self._cache_key(sym, timestamps_list[idx])
                    self._set_cached(cache_key, pred_df)
            else:
                # 不等长 → 逐个预测
                for idx, df_input, ts_input in zip(to_predict_indices, to_predict_dfs, to_predict_ts):
                    future_ts = self._build_future_timestamps(ts_input, pred_len)
                    pred_df = self._predictor.predict(
                        df=df_input,
                        x_timestamp=ts_input,
                        y_timestamp=future_ts,
                        pred_len=pred_len,
                        T=self._config.temperature,
                        top_p=self._config.top_p,
                        sample_count=self._config.sample_count,
                    )
                    results[idx] = pred_df
                    sym = symbols[idx]
                    cache_key = self._cache_key(sym, timestamps_list[idx])
                    self._set_cached(cache_key, pred_df)
        except Exception as e:
            logger.error(f"Kronos 批量预测失败: {e}")

        return results

    def cleanup(self):
        """释放 GPU 内存"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            del self._predictor
            self._model = None
            self._tokenizer = None
            self._predictor = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Kronos GPU 内存已释放")
            except ImportError:
                pass

    # ---------- 内部方法 ----------

    def _try_import_kronos(self) -> bool:
        """尝试导入 Kronos 模型类"""
        import sys
        import os

        # 方式1: 从环境变量指定的路径导入
        kronos_repo = os.environ.get('KRONOS_REPO_PATH', '')
        if kronos_repo and os.path.exists(kronos_repo) and kronos_repo not in sys.path:
            sys.path.insert(0, kronos_repo)

        # 方式2: 自动检测项目内的 kronos_repo 目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_repo = os.path.join(project_root, 'kronos_repo')
        if os.path.isdir(local_repo) and local_repo not in sys.path:
            sys.path.insert(0, local_repo)

        try:
            from model import Kronos, KronosTokenizer  # noqa: F401
            return True
        except ImportError:
            pass

        # 方式3: pip install kronos-predict
        try:
            from kronos import Kronos, KronosTokenizer  # noqa: F401
            return True
        except ImportError:
            pass

        logger.debug("未找到 Kronos 库，请设置 KRONOS_REPO_PATH 或克隆 kronos_repo 到项目目录")
        return False

    def _ensure_loaded(self) -> bool:
        """懒加载模型，首次预测时调用"""
        if self._loaded:
            return True
        if not self.is_available():
            return False

        try:
            import torch
            from model import Kronos, KronosTokenizer
            from model import KronosPredictor as _KronosPredictor

            # 设备选择
            if self._config.device == 'auto':
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self._device = self._config.device

            logger.info(f"加载 Kronos 模型 [{self._config.model_name}] 到 {self._device}...")

            self._tokenizer = KronosTokenizer.from_pretrained(
                self._config.tokenizer_hub_path
            )
            self._model = Kronos.from_pretrained(
                self._config.model_hub_path
            )
            self._model.to(self._device)
            self._model.eval()

            self._predictor = _KronosPredictor(
                self._model,
                self._tokenizer,
                max_context=self._config.max_context,
            )

            self._loaded = True
            logger.info(f"Kronos 模型加载完成 (context={self._config.max_context})")
            return True
        except Exception as e:
            logger.error(f"Kronos 模型加载失败: {e}")
            return False

    def _prepare_input(
        self, df: pd.DataFrame, timestamps: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """验证并准备模型输入数据"""
        max_ctx = self._config.max_context

        # 截取最后 max_ctx 根K线
        if len(df) > max_ctx:
            df = df.iloc[-max_ctx:]
            timestamps = timestamps.iloc[-max_ctx:]

        # 确保必要列存在
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"DataFrame 缺少必要列: {col}")

        # 填充可选列
        df = df.copy()
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        if 'amount' not in df.columns:
            df['amount'] = 0.0

        # 只保留模型需要的列
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        df = df[[c for c in keep_cols if c in df.columns]]

        return df.reset_index(drop=True), timestamps.reset_index(drop=True)

    def _build_future_timestamps(self, last_ts: pd.Series, pred_len: int) -> pd.Series:
        """构建未来时间戳 (用于 Kronos y_timestamp 参数)"""
        last = last_ts.iloc[-1]
        if hasattr(last, 'freq') and last.freq:
            freq = last.freq
        else:
            # 根据时间间隔推断频率
            if len(last_ts) >= 2:
                delta = last_ts.iloc[-1] - last_ts.iloc[-2]
                freq = delta
            else:
                freq = pd.Timedelta(days=1)

        return pd.Series(pd.date_range(start=last + freq, periods=pred_len, freq=freq))

    def _cache_key(self, symbol: str, timestamps: pd.Series) -> str:
        """生成缓存 key: symbol + 最后一根K线时间"""
        last_ts = str(timestamps.iloc[-1]) if len(timestamps) > 0 else ''
        return f"{symbol}:{last_ts}"

    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存，过期返回 None"""
        if key in self._cache:
            pred_df, ts = self._cache[key]
            if time.time() - ts < self._config.cache_ttl_seconds:
                # 移到末尾 (LRU)
                self._cache.move_to_end(key)
                return pred_df
            else:
                del self._cache[key]
        return None

    def _set_cached(self, key: str, pred_df: pd.DataFrame):
        """写入缓存，超限时逐出最旧的"""
        self._cache[key] = (pred_df, time.time())
        while len(self._cache) > self._config.cache_size:
            self._cache.popitem(last=False)  # FIFO 逐出
