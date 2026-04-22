"""Tushare API 统一初始化工具。

所有使用 tushare 的代码都应通过 get_pro() 获取 pro 实例，
确保统一使用自定义 API 地址。
"""

import os
import tushare as ts

# 自定义 API 地址（内网加速）
_CUSTOM_URL = "http://111.170.34.57:8010/"


def get_pro(token: str | None = None) -> ts.pro_api:
    """返回配置好自定义 API 地址的 tushare pro 实例。

    Args:
        token: Tushare token，默认从环境变量 TUSHARE_TOKEN 读取。

    Returns:
        配置完成的 ts.pro_api 实例。
    """
    token = token or os.getenv("TUSHARE_TOKEN", "")
    pro = ts.pro_api(token)
    pro._DataApi__http_url = _CUSTOM_URL
    return pro
