"""
MiniMax M2.7 LLM客户端 — Anthropic兼容接口

用于委员会分析和复盘报告的智能文本生成。
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def _get_client():
    if not ANTHROPIC_AVAILABLE:
        return None
    api_key = os.getenv('MINIMAX_API_KEY', '')
    base_url = os.getenv('MINIMAX_BASE_URL', '')
    if not api_key or not base_url:
        return None
    return anthropic.Anthropic(base_url=base_url, api_key=api_key)


def ask_minimax(
    prompt: str,
    system: str = '',
    max_tokens: int = 2000,
) -> Optional[str]:
    """
    调用MiniMax M2.7生成分析文本

    Args:
        prompt: 用户提示（含数据）
        system: 系统提示（角色设定）
        max_tokens: 最大输出token

    Returns:
        生成的文本，失败返回None
    """
    client = _get_client()
    if client is None:
        return None

    model = os.getenv('MINIMAX_MODEL', 'MiniMax-M2.7')

    try:
        kwargs = {
            'model': model,
            'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if system:
            kwargs['system'] = system

        msg = client.messages.create(**kwargs)

        texts = []
        for block in msg.content:
            if hasattr(block, 'text') and block.text:
                texts.append(block.text)
        return '\n'.join(texts) if texts else None

    except Exception as e:
        print(f'[MiniMax] API error: {e}')
        return None


def analyze_with_minimax(data_summary: str, task: str = 'committee') -> Optional[str]:
    """
    用MiniMax对交易数据做智能分析

    Args:
        data_summary: 结构化数据摘要
        task: 'committee' | 'post_market' | 'risk'

    Returns:
        分析文本
    """
    prompts = {
        'committee': (
            '你是一位资深A股投资顾问，擅长缠论分析。'
            '请根据以下数据，给出简明扼要的操作建议（买入/观望/放弃），'
            '包含：1)关键因素(2-3条) 2)风险提示 3)建议仓位。'
            '控制在150字以内。'
        ),
        'post_market': (
            '你是一位A股量化策略分析师。'
            '请根据以下今日复盘数据，给出：'
            '1)市场整体判断(一句话) 2)持仓操作建议 3)明日关注要点。'
            '控制在200字以内。'
        ),
        'risk': (
            '你是一位风控经理。'
            '请根据以下数据，评估当前持仓风险，'
            '给出：1)风险等级(低/中/高) 2)主要风险点 3)应对建议。'
            '控制在100字以内。'
        ),
    }

    system = prompts.get(task, prompts['committee'])
    return ask_minimax(data_summary, system=system, max_tokens=1000)
