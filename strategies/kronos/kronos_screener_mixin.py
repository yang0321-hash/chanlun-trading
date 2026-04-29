"""
Kronos 选股增强 Mixin

为批量扫描脚本提供基于 Kronos AI 预测的股票排序和筛选。
在缠论日线筛选后、30分钟分析前，用 Kronos 批量预测对候选股排序，
优先处理 Kronos 看好的标的。

用法 (在扫描脚本中):
    mixin = KronosScreenerMixin(KronosConfig())
    ranked = mixin.rank_with_kronos(candidates, data_map)
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .kronos_config import KronosConfig
from .kronos_predictor import KronosPredictor


class KronosScreenerMixin:
    """Kronos 批量选股增强"""

    def __init__(
        self,
        config: KronosConfig = None,
        predictor: KronosPredictor = None,
    ):
        self._config = config or KronosConfig()
        self._predictor = predictor or KronosPredictor(self._config)

    def rank_with_kronos(
        self,
        candidates: List[dict],
        data_map: Dict[str, pd.DataFrame],
        pred_len: int = None,
        top_n: int = None,
    ) -> List[dict]:
        """
        使用 Kronos 预测对候选股重新排序

        Args:
            candidates: 候选股列表, 每个包含 'code' 字段
            data_map: code -> OHLCV DataFrame
            pred_len: 预测K线根数 (默认用 screener_pred_len)
            top_n: 只预测前 N 只 (默认用 screener_top_n)

        Returns:
            排序后的候选列表，每项增加 kronos_score, kronos_predicted_return, kronos_max_dd 字段
        """
        pred_len = pred_len or self._config.screener_pred_len
        top_n = top_n or self._config.screener_top_n

        if not self._predictor.is_available():
            logger.info("Kronos 不可用，跳过选股增强")
            return candidates

        # 取 top_n 候选
        to_predict = candidates[:top_n]
        remaining = candidates[top_n:]

        # 收集数据和时间戳
        df_list = []
        ts_list = []
        symbols = []
        indices = []  # 在 to_predict 中的索引

        for i, cand in enumerate(to_predict):
            code = cand.get('code', '')
            if code not in data_map or data_map[code].empty:
                # 无数据，添加默认分数
                cand['kronos_score'] = 0.0
                cand['kronos_predicted_return'] = 0.0
                cand['kronos_max_dd'] = 0.0
                continue

            df = data_map[code]
            if hasattr(df.index, 'to_series'):
                timestamps = df.index.to_series().reset_index(drop=True)
            else:
                timestamps = pd.Series(df.index)

            df_list.append(df)
            ts_list.append(timestamps)
            symbols.append(code)
            indices.append(i)

        if not df_list:
            return candidates

        # 批量预测
        logger.info(f"Kronos 批量预测 {len(df_list)} 只候选股...")
        predictions = self._predictor.predict_batch(
            df_list=df_list,
            timestamps_list=ts_list,
            pred_len=pred_len,
            symbols=symbols,
        )

        # 计算分数
        for idx, pred_df, code in zip(indices, predictions, symbols):
            cand = to_predict[idx]
            if pred_df is None or pred_df.empty:
                cand['kronos_score'] = 0.0
                cand['kronos_predicted_return'] = 0.0
                cand['kronos_max_dd'] = 0.0
                continue

            current_close = data_map[code]['close'].iloc[-1]
            pred_close_last = pred_df['close'].iloc[-1]
            pred_low_min = pred_df['low'].min()

            predicted_return = (pred_close_last - current_close) / current_close
            predicted_max_dd = (pred_low_min - current_close) / current_close

            # 综合评分: 预期收益 × (1 + 最小回撤保护)
            # predicted_max_dd 通常是负数，所以 (1 + max_dd) 惩罚大回撤
            kronos_score = predicted_return * (1 + predicted_max_dd)

            cand['kronos_score'] = round(kronos_score, 4)
            cand['kronos_predicted_return'] = round(predicted_return, 4)
            cand['kronos_max_dd'] = round(predicted_max_dd, 4)

        # 按 kronos_score 降序排列 (有分数的优先)
        scored = [c for c in to_predict if 'kronos_score' in c and c['kronos_score'] > 0]
        unscored = [c for c in to_predict if c not in scored]

        scored.sort(key=lambda x: x.get('kronos_score', 0), reverse=True)

        # 剩余未预测的保持原序
        for cand in remaining:
            cand['kronos_score'] = 0.0
            cand['kronos_predicted_return'] = 0.0
            cand['kronos_max_dd'] = 0.0

        return scored + unscored + remaining

    def predict_single(
        self, df: pd.DataFrame, code: str = '', pred_len: int = None
    ) -> Optional[dict]:
        """
        单股预测，返回预测摘要

        Returns:
            {'predicted_return': float, 'max_drawdown': float, 'score': float}
            或 None
        """
        pred_len = pred_len or self._config.screener_pred_len

        if not self._predictor.is_available():
            return None

        if hasattr(df.index, 'to_series'):
            timestamps = df.index.to_series().reset_index(drop=True)
        else:
            timestamps = pd.Series(df.index)

        pred_df = self._predictor.predict(
            df=df, timestamps=timestamps,
            pred_len=pred_len, symbol=code,
        )
        if pred_df is None or pred_df.empty:
            return None

        current_close = df['close'].iloc[-1]
        pred_close_last = pred_df['close'].iloc[-1]
        pred_low_min = pred_df['low'].min()

        predicted_return = (pred_close_last - current_close) / current_close
        predicted_max_dd = (pred_low_min - current_close) / current_close
        score = predicted_return * (1 + predicted_max_dd)

        return {
            'predicted_return': round(predicted_return, 4),
            'max_drawdown': round(predicted_max_dd, 4),
            'score': round(score, 4),
            'pred_close': round(pred_close_last, 2),
            'pred_low': round(pred_low_min, 2),
        }
