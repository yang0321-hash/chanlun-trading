"""
日志管理模块
"""

import sys
from loguru import logger
from pathlib import Path
from typing import Optional


def get_logger(name: Optional[str] = None) -> 'logger':
    """
    获取logger实例

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger


def setup_logger(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    rotation: str = '100 MB',
    retention: str = '30 days'
) -> None:
    """
    配置日志系统

    Args:
        level: 日志级别
        log_file: 日志文件路径
        rotation: 日志轮转设置
        retention: 日志保留时间
    """
    # 移除默认handler
    logger.remove()

    # 控制台输出 - 带颜色
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding='utf-8'
        )


# 初始化时自动配置
def init_logger(config):
    """根据配置初始化日志"""
    setup_logger(
        level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.file'),
        rotation=config.get('logging.rotation', '100 MB')
    )
