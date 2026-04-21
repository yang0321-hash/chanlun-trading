"""
LLM新闻分析Agent — DeepSeek + AKShare

3层降级:
  Tier 1: DeepSeek + 新闻 → LLM情绪分析
  Tier 2: 有新闻无DeepSeek → 关键词规则分析
  Tier 3: 无新闻 → 中性AgentArgument (confidence=0)
"""

import os
import json
from typing import Optional, List, Dict
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from agents.committee_agents import AgentArgument, CommitteeContext


@dataclass
class NewsData:
    symbol: str
    company_name: str
    items: List[Dict]  # [{'title': ..., 'content': ..., 'date': ...}]
    source: str = 'none'


# 关键词情绪分析词库
POSITIVE_KEYWORDS = ['利好', '上涨', '突破', '增长', '超预期', '盈利', '订单', '新高',
                     '涨停', '获批', '中标', '签约', '合作', '创新高', '业绩大增']
NEGATIVE_KEYWORDS = ['利空', '下跌', '风险', '处罚', '亏损', '减持', '退市', '暴雷',
                     '违规', '诉讼', '预警', '停产', '事故', '商誉减值']

NEWS_PROMPT = """你是一位A股新闻情绪分析师。请分析以下与{company}({code})相关的近期新闻:

{news_text}

请严格用以下JSON格式回复(不要包含其他内容):
{{"sentiment_score": <float -1.0到1.0>, "confidence": <float 0到1>, "key_themes": [<2-3个主题>], "risk_flags": [<风险点,可为空>], "reasoning": "<一句话总结>"}}"""


def _fetch_cctv_news() -> List[Dict]:
    """获取CCTV财经新闻"""
    try:
        import akshare as ak
        df = ak.news_cctv()
        if df is not None and not df.empty:
            items = []
            for _, row in df.head(15).iterrows():
                items.append({
                    'title': str(row.get('title', '')),
                    'content': str(row.get('content', ''))[:300],
                    'date': str(row.get('date', '')),
                })
            return items
    except Exception:
        pass
    return []


def _fetch_stock_news(name: str) -> List[Dict]:
    """获取个股相关新闻 (从CCTV新闻中过滤)"""
    all_news = _fetch_cctv_news()
    if not all_news:
        return []

    matched = []
    for item in all_news:
        title = item.get('title', '')
        content = item.get('content', '')
        if name and (name in title or name in content):
            matched.append(item)

    return matched if matched else all_news[:5]


class NewsAnalyst:
    """新闻情绪分析Agent"""

    def __init__(self):
        self._client = None
        self._init_deepseek()

    def _init_deepseek(self):
        """初始化DeepSeek客户端"""
        api_key = os.getenv('DEEPSEEK_API_KEY', '')
        if not api_key:
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_key,
                base_url='https://api.deepseek.com',
                timeout=15,
            )
        except ImportError:
            pass

    def analyze(self, ctx: CommitteeContext) -> AgentArgument:
        """分析新闻情绪"""
        news = _fetch_stock_news(ctx.name)
        if not news:
            return self._no_news_result(ctx)

        if self._client:
            return self._llm_analysis(ctx, news)
        else:
            return self._rule_analysis(ctx, news)

    def _llm_analysis(self, ctx: CommitteeContext, news: NewsData) -> AgentArgument:
        """DeepSeek LLM分析"""
        news_text = '\n'.join(
            f'{i+1}. [{item["date"]}] {item["title"]}'
            for i, item in enumerate(news[:10])
        )
        prompt = NEWS_PROMPT.format(company=ctx.name, code=ctx.symbol, news_text=news_text)

        try:
            resp = self._client.chat.completions.create(
                model='deepseek-chat',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()

            # 解析JSON
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            result = json.loads(text)

            score = float(result.get('sentiment_score', 0))
            score = max(-1.0, min(1.0, score))
            confidence = float(result.get('confidence', 0.5))
            themes = result.get('key_themes', [])
            flags = result.get('risk_flags', [])
            reasoning = result.get('reasoning', '')

            key_points = themes[:3]
            if flags:
                key_points.append(f'风险: {", ".join(flags[:2])}')

            return AgentArgument(
                agent_name='NewsAnalyst',
                stance='bull' if score > 0.1 else ('bear' if score < -0.1 else 'neutral'),
                reasoning=reasoning,
                confidence=confidence,
                key_points=key_points,
                data_references={
                    'news_sentiment_raw': score,
                    'news_count': len(news),
                    'source': 'deepseek',
                },
            )
        except Exception as e:
            return self._rule_analysis(ctx, news)

    def _rule_analysis(self, ctx: CommitteeContext, news: List[Dict]) -> AgentArgument:
        """关键词规则分析 (无LLM降级)"""
        pos_count = 0
        neg_count = 0
        total = len(news)
        key_points = []

        for item in news[:10]:
            text = item.get('title', '') + item.get('content', '')
            for kw in POSITIVE_KEYWORDS:
                if kw in text:
                    pos_count += 1
                    if kw not in str(key_points):
                        key_points.append(kw)
                    break
            for kw in NEGATIVE_KEYWORDS:
                if kw in text:
                    neg_count += 1
                    break

        if total == 0:
            return self._no_news_result(ctx)

        score = (pos_count - neg_count) / total  # [-1, +1]
        score = max(-1.0, min(1.0, score))
        confidence = min(0.6, (pos_count + neg_count) / max(total, 1))

        return AgentArgument(
            agent_name='NewsAnalyst',
            stance='bull' if score > 0.1 else ('bear' if score < -0.1 else 'neutral'),
            reasoning=f'关键词分析: 利好{pos_count}条, 利空{neg_count}条, 共{total}条新闻',
            confidence=confidence,
            key_points=key_points[:5],
            data_references={
                'news_sentiment_raw': score,
                'news_count': total,
                'source': 'rule',
            },
        )

    def _no_news_result(self, ctx: CommitteeContext) -> AgentArgument:
        """无新闻时返回中性结果"""
        return AgentArgument(
            agent_name='NewsAnalyst',
            stance='neutral',
            reasoning='未获取到相关新闻数据',
            confidence=0,
            key_points=['无新闻数据'],
            data_references={
                'news_sentiment_raw': 0.0,
                'news_count': 0,
                'source': 'none',
            },
        )
