"""
投资委员会 — 主入口

整合6个Agent（牛分析师、熊分析师、情绪分析、行业轮动、风控、基金经理）
对扫描器候选股进行多维度评估，输出结构化买卖决策。

使用方法:
    from agents.investment_committee import InvestmentCommittee
    committee = InvestmentCommittee(hs, sector_map, portfolio_state, sector_momentum)
    results = committee.evaluate_batch(candidates)
    committee.save_results(results)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

from data.hybrid_source import HybridSource
from agents.committee_agents import (
    CommitteeContext, ChanlunInfo, RiskAssessment,
    BullAnalyst, BearAnalyst, SentimentAnalyzer,
    SectorRotation, RiskManager, FundManager,
    analyze_chanlun_structure,
)

try:
    from utils.minimax_client import analyze_with_minimax
    MINIMAX_AVAILABLE = True
except ImportError:
    MINIMAX_AVAILABLE = False

try:
    from agents.market_analyzer import MarketAnalyzer, MarketContext
    MARKET_AVAILABLE = True
except ImportError:
    MARKET_AVAILABLE = False

try:
    from agents.debate_mechanism import CommitteeDebateRound, DebateAdjustment
    DEBATE_AVAILABLE = True
except ImportError:
    DEBATE_AVAILABLE = False

try:
    from agents.news_analyst import NewsAnalyst
    NEWS_ANALYST_AVAILABLE = True
except ImportError:
    NEWS_ANALYST_AVAILABLE = False


class InvestmentCommittee:
    """
    投资委员会 — 多Agent协作决策

    数据流:
      Phase 0: MarketAnalyzer (大盘环境)
      Phase A: Bull + Bear + Sentiment + Sector (并行)
      Phase B: RiskManager (综合Phase A)
      Phase C: FundManager (最终决策 + 大盘调整)
          ↓
      InvestmentCommitteeResult (JSON)
    """

    def __init__(
        self,
        hs: HybridSource,
        sector_map: Dict[str, str],
        portfolio_state: Dict[str, Any],
        sector_momentum: Dict[str, float] = None,
        config: Dict[str, Any] = None,
    ):
        self.hs = hs
        self.sector_map = sector_map
        self.portfolio_state = portfolio_state
        self.sector_momentum = sector_momentum or {}
        self.config = config or {}
        self.market_ctx: Optional['MarketContext'] = None

        # 初始化Agent
        self.bull = BullAnalyst()
        self.bear = BearAnalyst()
        self.sentiment = SentimentAnalyzer()
        self.sector_rotation = SectorRotation(sector_map, sector_momentum)
        self.risk_mgr = RiskManager()
        self.fund_mgr = FundManager()

        # 新增Agent (优雅降级)
        self.debate = CommitteeDebateRound() if DEBATE_AVAILABLE else None
        self.news_analyst = NewsAnalyst() if NEWS_ANALYST_AVAILABLE else None

    def _init_market(self):
        """Phase 0: 大盘多周期缠论分析"""
        if not MARKET_AVAILABLE:
            return
        try:
            ma = MarketAnalyzer()
            self.market_ctx = ma.analyze()
            print(f'[Market] regime={self.market_ctx.regime} '
                  f'phase={self.market_ctx.index_phase} '
                  f'risk={self.market_ctx.risk_premium:+.2f} '
                  f'pos={self.market_ctx.position_adjust:.1f}')
        except Exception as e:
            print(f'[Market] analysis failed: {e}')

    def evaluate_batch(self, candidates: List[Dict]) -> List[Dict]:
        """批量评估候选股"""
        # Phase 0: 大盘环境分析
        self._init_market()

        results = []
        n = len(candidates)
        for i, candidate in enumerate(candidates):
            print(f'  [{i+1}/{n}] 评估 {candidate.get("code", "?")} {candidate.get("name", "")}...', end='')

            try:
                result = self.evaluate_one(candidate)
                results.append(result)
                print(f' {result["decision"]} ({result["composite_score"]:.0f}分)')
            except Exception as e:
                print(f' 评估失败: {e}')
                results.append(self._error_result(candidate, str(e)))

        # 同行业去重: 每行业最多保留2只buy(按评分降序)
        results = self._sector_dedup(results, max_per_sector=2)
        return results

    def _sector_dedup(self, results: List[Dict], max_per_sector: int = 2) -> List[Dict]:
        """同行业去重 — 每行业最多保留max_per_sector只buy(按评分降序)"""
        sector_buy_count = {}
        # 按评分降序遍历，确保高分优先保留
        for r in sorted(results, key=lambda x: -x.get('composite_score', 0)):
            if r.get('decision') == 'buy':
                sector = r.get('sector', '未知')
                sector_buy_count[sector] = sector_buy_count.get(sector, 0) + 1
                if sector_buy_count[sector] > max_per_sector:
                    r['decision'] = 'hold'
                    r['position_pct'] = 0
                    r['shares'] = 0
                    r['warnings'].append(f'行业去重: {sector}已有{max_per_sector}只buy, 降为hold')

        return results

    def evaluate_one(self, candidate: Dict) -> Dict:
        """评估单只股票"""
        # 如果还没做大盘分析，做一次
        if self.market_ctx is None and MARKET_AVAILABLE:
            self._init_market()

        # Step 1: 构建上下文
        ctx = self._build_context(candidate)
        ctx.market = self.market_ctx  # 注入大盘环境

        # Step 2: Phase A — 分析Agent
        bull_arg = self.bull.analyze(ctx)
        bear_arg = self.bear.analyze(ctx)
        sentiment_arg = self.sentiment.analyze(ctx)
        sector_arg = self.sector_rotation.analyze(ctx)

        # Phase A+: 新闻分析 (新)
        news_arg = None
        if self.news_analyst:
            try:
                news_arg = self.news_analyst.analyze(ctx)
            except Exception as e:
                print(f' [新闻分析失败: {e}]')

        # Phase A+: 牛熊辩论 (新)
        debate_result = None
        if self.debate:
            try:
                debate_result = self.debate.run(ctx, bull_arg, bear_arg)
            except Exception as e:
                print(f' [辩论失败: {e}]')

        # Step 3: Phase B — 风控
        risk = self.risk_mgr.evaluate(ctx, bull_arg, bear_arg, sentiment_arg, sector_arg)

        # Step 4: Phase C — 基金经理决策
        result = self.fund_mgr.decide(
            ctx, bull_arg, bear_arg, sentiment_arg, sector_arg, risk,
            weights=self.config.get('weights'),
            news_arg=news_arg,
            debate_adjustment=debate_result,
        )

        return result

    def _build_context(self, candidate: Dict) -> CommitteeContext:
        """构建委员会上下文"""
        code = candidate.get('code', '')

        # 获取日线数据
        try:
            df_daily = self.hs.get_kline(code, period='daily')
        except Exception:
            df_daily = pd.DataFrame()

        # 行业信息: 优先用完整映射，避免"未知"
        mapped_sector = self.sector_map.get(code, '')
        sector = mapped_sector or candidate.get('sector', '') or '未知'

        # 缠论结构分析
        chanlun = None
        if len(df_daily) >= 60:
            chanlun = analyze_chanlun_structure(df_daily, candidate)

        # 如果缠论分析失败，从扫描器数据中提取部分信息
        if not chanlun and candidate:
            pivot_info = candidate.get('pivot_info', '')
            zg, zd = 0.0, 0.0
            if pivot_info and 'ZG=' in pivot_info:
                try:
                    parts = pivot_info.split()
                    for p in parts:
                        if p.startswith('ZG='):
                            zg = float(p.split('=')[1])
                        elif p.startswith('ZD='):
                            zd = float(p.split('=')[1])
                except (ValueError, IndexError):
                    pass

            buy_date = candidate.get('2buy_date', '')
            signal_type = candidate.get('signal_type', '')
            chanlun = ChanlunInfo(
                pivot_zg=zg,
                pivot_zd=zd,
                buy_type=signal_type if signal_type in ('1buy', '2buy', '3buy') else ('2buy' if buy_date else ''),
                buy_price=candidate.get('entry_price', 0),
                buy_date=buy_date,
                buy_strength=candidate.get('buy_strength', ''),
                golden_ratio_pass=candidate.get('golden_ratio_pass', False),
                weekly_trend='bear' if candidate.get('weekly_trend', '').startswith('空头') else
                             ('bull' if candidate.get('weekly_trend', '').startswith('多头') else
                              ('range' if candidate.get('weekly_trend', '').startswith('盘整') else '')),
            )
            # 判断价格位置
            if len(df_daily) > 0:
                last_close = float(df_daily['close'].iloc[-1])
                if zg > 0 and zd > 0:
                    if last_close > zg:
                        chanlun.price_vs_pivot = 'above'
                    elif last_close < zd:
                        chanlun.price_vs_pivot = 'below'
                    else:
                        chanlun.price_vs_pivot = 'inside'
                    chanlun.stop_by_structure = zd
        elif chanlun and candidate:
            # 缠论分析成功, 但扫描器有额外信息(强度/黄金分割/周线)需要补充
            chanlun.buy_strength = candidate.get('buy_strength', '') or chanlun.buy_strength
            chanlun.golden_ratio_pass = candidate.get('golden_ratio_pass', False) or chanlun.golden_ratio_pass
            wt = candidate.get('weekly_trend', '')
            if wt and not chanlun.weekly_trend:
                chanlun.weekly_trend = ('bear' if '空头' in wt else
                                        ('bull' if '多头' in wt else
                                         ('range' if '盘整' in wt else '')))
            # 也补充signal_type
            signal_type = candidate.get('signal_type', '')
            if signal_type and not chanlun.buy_type:
                chanlun.buy_type = signal_type

        return CommitteeContext(
            symbol=code,
            name=candidate.get('name', code),
            sector=sector,
            df_daily=df_daily,
            entry_price=candidate.get('entry_price', 0),
            stop_price=candidate.get('stop_price', 0),
            scanner_score=candidate.get('total_score', 0),
            risk_reward=candidate.get('risk_reward', 0),
            sector_momentum=self.sector_momentum,
            sector_map=self.sector_map,
            portfolio_state=self.portfolio_state,
            chanlun=chanlun,
        )

    def save_results(self, results: List[Dict], output_file: str = None):
        """保存评估结果"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_file = f'signals/investment_committee_{timestamp}.json'

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        # 统计
        buy_count = sum(1 for r in results if r.get('decision') == 'buy')
        hold_count = sum(1 for r in results if r.get('decision') == 'hold')
        reject_count = sum(1 for r in results if r.get('decision') == 'reject')

        # 组合风险摘要
        positions = self.portfolio_state.get('positions', [])
        total_exposure = sum(r.get('position_pct', 0) for r in results if r.get('decision') == 'buy')

        # 行业分布
        sector_exp = {}
        for r in results:
            if r.get('decision') == 'buy':
                s = r.get('sector', '未知')
                sector_exp[s] = sector_exp.get(s, 0) + r.get('position_pct', 0)

        output = {
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'market': {
                'regime': self.market_ctx.regime if self.market_ctx else 'unknown',
                'phase': self.market_ctx.index_phase if self.market_ctx else 'unknown',
                'risk_premium': self.market_ctx.risk_premium if self.market_ctx else 0,
                'position_adjust': self.market_ctx.position_adjust if self.market_ctx else 1.0,
                'warnings': self.market_ctx.warnings if self.market_ctx else [],
                'summary': self.market_ctx.stroke_summary if self.market_ctx else '',
            } if self.market_ctx else {},
            'total_evaluated': len(results),
            'buy_count': buy_count,
            'hold_count': hold_count,
            'reject_count': reject_count,
            'portfolio_risk_summary': {
                'total_new_exposure_pct': round(total_exposure, 4),
                'sector_concentration': {k: round(v, 4) for k, v in sector_exp.items()},
                'current_positions': len(positions),
            },
            'decisions': results,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        print(f'\n投资委员会结果已保存: {output_file}')
        return output_file

    def print_report(self, results: List[Dict]):
        """打印评估报告"""
        print(f'\n{"="*80}')
        print(f'投资委员会评估报告 — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print(f'{"="*80}')

        # 大盘环境概览
        if self.market_ctx:
            mc = self.market_ctx
            print(f'\n[大盘环境] {mc.index_phase} | regime={mc.regime} | '
                  f'风险溢价={mc.risk_premium:+.2f} | 仓位系数={mc.position_adjust:.1f}')
            if mc.stroke_summary:
                print(f'  {mc.stroke_summary}')
            for w in mc.warnings:
                print(f'  ! {w}')

        buy_list = [r for r in results if r['decision'] == 'buy']
        hold_list = [r for r in results if r['decision'] == 'hold']
        reject_list = [r for r in results if r['decision'] == 'reject']

        if buy_list:
            print(f'\n>>> 买入推荐 ({len(buy_list)}只):')
            print(f'{"代码":<8} {"名称":<8} {"行业":<10} {"评分":>5} {"仓位":>6} '
                  f'{"止损":>8} {"关键因素"}')
            print('-' * 80)
            for r in buy_list:
                factors = ', '.join(r.get('key_factors', [])[:3])
                print(f'{r["symbol"]:<8} {r["name"]:<8} {r["sector"]:<10} '
                      f'{r["composite_score"]:>5.0f} {r["position_pct"]:>5.1%} '
                      f'{r["stop_loss"]:>8.2f} {factors}')

        if hold_list:
            print(f'\n>>> 观望 ({len(hold_list)}只):')
            for r in hold_list:
                print(f'  {r["symbol"]} {r["name"]} — {r["composite_score"]:.0f}分 '
                      f'(风险:{r["risk_level"]}) {r.get("debate_summary", "")}')

        if reject_list:
            print(f'\n>>> 否决 ({len(reject_list)}只):')
            for r in reject_list:
                warnings = ', '.join(r.get('warnings', []))
                print(f'  {r["symbol"]} {r["name"]} — {r["composite_score"]:.0f}分 '
                      f'({warnings})')

        # MiniMax AI总结
        if MINIMAX_AVAILABLE and buy_list:
            self._ai_summary(buy_list)

    def _ai_summary(self, buy_list: List[Dict]):
        """用MiniMax生成买入推荐的AI总结"""
        data_lines = []
        for r in buy_list[:5]:
            factors = ', '.join(r.get('key_factors', [])[:3])
            data_lines.append(
                f"{r['symbol']}({r['name']}): 评分{r['composite_score']:.0f}, "
                f"仓位{r['position_pct']:.0%}, 止损{r['stop_loss']:.2f}, "
                f"风险{r['risk_level']}, {factors}"
            )
        prompt = '今日买入推荐:\n' + '\n'.join(data_lines)
        ai_text = analyze_with_minimax(prompt, task='committee')
        if ai_text:
            print(f'\n>>> AI分析 (MiniMax):')
            print(f'  {ai_text}')

    def _error_result(self, candidate: Dict, error: str) -> Dict:
        code = candidate.get('code', '')
        return {
            'symbol': code,
            'name': candidate.get('name', code),
            'sector': candidate.get('sector', '未知'),
            'decision': 'reject',
            'confidence': 0.0,
            'composite_score': 0.0,
            'position_pct': 0.0,
            'shares': 0,
            'entry_price': candidate.get('entry_price', 0),
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'key_factors': [f'评估失败: {error}'],
            'bull_confidence': 0.0,
            'bear_confidence': 0.0,
            'sentiment_score': 0.0,
            'sector_score': 0.0,
            'risk_score': 1.0,
            'risk_level': 'EXTREME',
            'debate_summary': f'评估异常: {error}',
            'warnings': [error],
        }


# ============================================================
# CLI入口
# ============================================================

def main():
    """命令行运行投资委员会"""
    import argparse
    parser = argparse.ArgumentParser(description='投资委员会评估')
    parser.add_argument('--scan-file', type=str, help='扫描结果JSON文件')
    parser.add_argument('--top', type=int, default=10, help='评估前N只候选股')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, '.')
    for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        os.environ.pop(k, None)

    hs = HybridSource()

    # 加载行业映射
    sector_map = {}
    for sp in ['chanlun_system/full_sector_map.json', 'chanlun_system/thshy_sector_map.json']:
        if os.path.exists(sp):
            with open(sp, 'r', encoding='utf-8') as f:
                sector_map = json.load(f).get('stock_to_sector', {})
            if sector_map:
                break

    # 加载扫描结果
    if args.scan_file:
        with open(args.scan_file, 'r', encoding='utf-8') as f:
            scan_data = json.load(f)
        candidates = scan_data.get('top_n', scan_data.get('all_signals', []))
    else:
        # 运行扫描
        print('[1] 运行增强版扫描...')
        from scan_enhanced_v3 import scan_enhanced, load_sector_map, calc_sector_momentum
        candidates = scan_enhanced(pool='tdx_all', top_n=args.top)

    if not candidates:
        print('无候选股')
        return

    # 计算行业动量（如果扫描结果没有自带）
    print(f'\n[2] 投资委员会评估 ({len(candidates)}只候选股)...')

    # 加载持仓
    positions = {'positions': [], 'capital': 1000000}
    pos_path = 'signals/positions.json'
    if os.path.exists(pos_path):
        with open(pos_path, 'r', encoding='utf-8') as f:
            positions = json.load(f)

    # 计算行业动量
    sector_momentum = {}
    try:
        from scan_enhanced_v3 import load_sector_map, calc_sector_momentum
        s_map = load_sector_map()
        # 加载日线数据用于计算动量
        daily_map = {}
        for c in candidates[:50]:  # 只为候选股加载
            code = c.get('code', '')
            try:
                df = hs.get_kline(code, period='daily')
                if len(df) >= 10:
                    daily_map[code] = df
            except Exception:
                pass
        sector_momentum = calc_sector_momentum(daily_map, s_map)
    except Exception:
        pass

    committee = InvestmentCommittee(
        hs=hs,
        sector_map=sector_map,
        portfolio_state=positions,
        sector_momentum=sector_momentum,
    )

    results = committee.evaluate_batch(candidates[:args.top])
    committee.print_report(results)
    committee.save_results(results)


if __name__ == '__main__':
    main()
