"""
本地数据存储模块
"""

import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd


class DataStorage:
    """
    本地数据存储类

    用于缓存K线数据，减少重复请求
    """

    def __init__(self, storage_path: str = './data/storage'):
        """
        初始化存储

        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, symbol: str, period: str) -> Path:
        """获取数据文件路径"""
        filename = f"{symbol}_{period}.pkl"
        return self.storage_path / filename

    def save(self, symbol: str, period: str, df: pd.DataFrame) -> None:
        """
        保存数据

        Args:
            symbol: 股票代码
            period: 周期
            df: K线数据
        """
        file_path = self.get_file_path(symbol, period)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'data': df,
                'saved_at': datetime.now()
            }, f)

    def load(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        加载数据

        Args:
            symbol: 股票代码
            period: 周期

        Returns:
            K线数据，如果不存在则返回None
        """
        file_path = self.get_file_path(symbol, period)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data['data']
        except Exception:
            return None

    def exists(self, symbol: str, period: str) -> bool:
        """
        检查数据是否存在

        Args:
            symbol: 股票代码
            period: 周期

        Returns:
            是否存在
        """
        return self.get_file_path(symbol, period).exists()

    def delete(self, symbol: str, period: str) -> bool:
        """
        删除数据

        Args:
            symbol: 股票代码
            period: 周期

        Returns:
            是否删除成功
        """
        file_path = self.get_file_path(symbol, period)

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def clear(self) -> None:
        """清空所有缓存数据"""
        for file in self.storage_path.glob('*.pkl'):
            file.unlink()
