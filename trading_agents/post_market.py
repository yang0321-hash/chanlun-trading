#!/usr/bin/env python3
"""复盘Agent — 冷静分析师

21:30 全天数据汇总
21:35 策略有效性分析
21:40 风险控制评估
21:45 明日策略规划
"""
import sys
import os
import json
import traceback
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
          'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from dotenv import load_dotenv
load_dotenv()

from data.hybrid_source import HybridSource

try:
    from utils.minimax_client import analyze_with_minimax
    MINIMAX_AVAILABLE = True
except ImportError:
    MINIMAX_AVAILABLE = False


# ==================== 工具函数 ====================

def code_to_prefix(code: str) -> str:
    code = code.upper()
    if '.' in code:
        code = code.split('.')[0]
    if code.startswith(('0', '3')):
        return f'sz{code}'
    if code.startswith('6'):
        return f'sh{code}'
    if code.startswith(('SZ', 'SH')):
        return code.lower()
    return code.lower()


def load_sector_map() -> Dict[str, str]:
    for sp in ['chanlun_system/full_sector_map.json',
               'chanlun_system/thshy_sector_map.json']:
        if os.path.exists(sp):
            try:
                with open(sp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'stock_to_sector' in data:
                    return data['stock_to_sector']
                elif isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {}


def load_positions() -> dict:
    path = 'signals/positions.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'positions': [], 'capital': 1000000}


def load_committee_results(days: int = 10, exclude_today: bool = True) -> List[dict]:
    """加载最近N天的委员会结果（排除今天，因为今天无T+1数据）"""
    results = []
    files = sorted(glob.glob('signals/investment_committee_*.json'), reverse=True)

    cutoff = datetime.now() - timedelta(days=days)
    today_str = datetime.now().strftime('%Y%m%d')

    for f in files:
        try:
            basename = os.path.basename(f)
            date_str = basename.split('_')[2]  # 20260417
            file_date = datetime.strptime(date_str, '%Y%m%d')

            if file_date < cutoff:
                continue
            if exclude_today and date_str == today_str:
                continue

            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)

            for d in data.get('decisions', []):
                d['scan_date'] = date_str
                d['source_file'] = basename
            results.append(data)
        except Exception:
            continue

    return results


def load_agent_log() -> dict:
    path = 'signals/agent_log.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_agent_log(log: dict):
    os.makedirs('signals', exist_ok=True)
    with open('signals/agent_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def check_today_done(agent_name: str) -> bool:
    log = load_agent_log()
    today = datetime.now().strftime('%Y-%m-%d')
    entry = log.get(today, {})
    return entry.get(agent_name, {}).get('status') == 'done'


def mark_done(agent_name: str):
    log = load_agent_log()
    today = datetime.now().strftime('%Y-%m-%d')
    if today not in log:
        log[today] = {}
    log[today][agent_name] = {
        'status': 'done',
        'time': datetime.now().strftime('%H:%M'),
    }
    save_agent_log(log)


def send_notification(title: str, text: str):
    """发送通知到缠论专用飞书机器人"""
    chanlun_webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL')
    if chanlun_webhook:
        try:
            from utils.notification import FeishuConfig, FeishuNotifier
            notifier = FeishuNotifier(FeishuConfig(webhook_url=chanlun_webhook))
            notifier.send_post(title, text)
            return True
        except Exception as e:
            print(f'飞书推送失败: {e}')
    print('  (未配置 CHANLUN_FEISHU_WEBHOOK_URL，跳过推送)')
    return False


def send_card(title: str, elements: list):
    """发送飞书卡片"""
    chanlun_webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL')
    if chanlun_webhook:
        try:
            from utils.notification import FeishuConfig, FeishuNotifier
            notifier = FeishuNotifier(FeishuConfig(webhook_url=chanlun_webhook))
            notifier.send_card(title, elements)
            return True
        except Exception as e:
            print(f'飞书卡片推送失败: {e}')
    return False


# ==================== 复盘Agent ====================

class PostMarketAgent:
    """复盘Agent — 冷静分析师"""

    def __init__(self):
        self.hs = HybridSource()
        self.sector_map = load_sector_map()
        self.positions_data = load_positions()
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.report_parts = []

    def run(self):
        """执行完整复盘流程"""
        print(f'=== 复盘Agent {self.today} ===')
        print()

        # 1. 全天汇总
        print('[1] 全天数据汇总...')
        summary = self.daily_summary()
        self.report_parts.append(summary)

        # 2. 策略有效性分析
        print('[2] 策略有效性分析...')
        strategy = self.strategy_analysis()
        self.report_parts.append(strategy)

        # 3. 风险控制评估
        print('[3] 风险控制评估...')
        risk = self.risk_assessment()
        self.report_parts.append(risk)

        # 4. 热点板块复盘
        print('[4] 热点板块复盘...')
        hot_sector_report = self.hot_sector_review()
        self.report_parts.append(hot_sector_report)

        # 5. 明日策略规划
        print('[5] 明日策略规划...')
        plan = self.next_day_plan()
        self.report_parts.append(plan)

        # 6. AI总结 (MiniMax)
        if MINIMAX_AVAILABLE:
            print('[6] AI分析总结...')
            ai = self._ai_summary()
            if ai:
                self.report_parts.append(ai)

        # 7. 生成报告 + 推送
        print('[7] 生成报告...')
        report = '\n\n'.join(self.report_parts)
        self.save_report(report)
        self.push_report()

        # 8. 标记完成
        mark_done('post_market')
        print('\n复盘完成')

    # ---------- 1. 全天汇总 ----------

    def daily_summary(self) -> str:
        """全天数据汇总"""
        lines = [f'=== 复盘报告 {self.today} ===', '']

        # 大盘缠论分析
        try:
            from agents.market_analyzer import MarketAnalyzer
            ma = MarketAnalyzer()
            mc = ma.analyze()
            lines.append(f'【大盘缠论】{mc.index_phase} | regime={mc.regime} | '
                        f'风险溢价={mc.risk_premium:+.2f} | 仓位系数={mc.position_adjust:.1f}')
            if mc.stroke_summary:
                lines.append(f'  {mc.stroke_summary}')
            for w in mc.warnings:
                lines.append(f'  ! {w}')
            lines.append('')
        except Exception as e:
            lines.append(f'【大盘缠论】分析失败: {e}')

        # 大盘概况 — 用Sina获取指数数据
        try:
            sh_close, sh_chg = self._get_index_change('sh000001')
            sz_close, sz_chg = self._get_index_change('sz399001')

            if sh_close and sz_close:
                lines.append(f'【大盘】上证 {sh_close:,.0f} ({sh_chg:+.2f}%) | '
                             f'深证 {sz_close:,.0f} ({sz_chg:+.2f}%)')
            else:
                lines.append('【大盘】指数数据获取失败')
        except Exception:
            lines.append('【大盘】数据获取失败')

        # 持仓盈亏
        positions = self.positions_data.get('positions', [])
        if positions:
            total_pnl = 0
            pos_lines = []
            for p in positions:
                code = p.get('code', '')
                name = p.get('name', '')
                try:
                    df = self.hs.get_kline(code, period='daily')
                    if df is not None and len(df) >= 1:
                        last_close = float(df['close'].iloc[-1])
                        entry = p.get('entry_price', 0)
                        pnl = (last_close - entry) / entry * 100 if entry > 0 else 0
                        total_pnl += pnl
                        pos_lines.append(
                            f'  {code_to_prefix(code)} ({name}) '
                            f'现价:{last_close:.2f} 盈亏:{pnl:+.2f}%')
                except Exception:
                    pos_lines.append(f'  {code_to_prefix(code)} ({name}) 数据缺失')

            avg_pnl = total_pnl / len(positions) if positions else 0
            lines.append(f'\n【持仓】{len(positions)}只 平均盈亏:{avg_pnl:+.2f}%')
            lines.extend(pos_lines)
        else:
            lines.append('\n【持仓】当前无持仓')

        # 今日委员会结果
        today_files = glob.glob(f'signals/investment_committee_{self.today.replace("-", "")}*.json')
        if not today_files:
            # 也检查昨天的
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            today_files = glob.glob(f'signals/investment_committee_{yesterday}*.json')

        if today_files:
            latest_file = sorted(today_files)[-1]
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            buy_count = data.get('buy_count', 0)
            hold_count = data.get('hold_count', 0)
            reject_count = data.get('reject_count', 0)

            # 计算实际涨跌
            buy_rets = self._calc_decisions_returns(
                data.get('decisions', []),
                fallback_source=os.path.basename(latest_file)
            )

            lines.append(f'\n【今日委员会】buy:{buy_count} hold:{hold_count} reject:{reject_count}')
            if buy_rets:
                avg_ret = np.mean(buy_rets)
                win = sum(1 for r in buy_rets if r > 0) / len(buy_rets) * 100
                lines.append(f'  BUY组 T+1: avg {avg_ret:+.2f}% 胜率 {win:.0f}%')

                for d in data.get('decisions', []):
                    if d.get('decision') == 'buy':
                        code = d.get('symbol', '')
                        name = d.get('name', '')
                        score = d.get('composite_score', 0)
                        lines.append(f'    {code_to_prefix(code)} ({name}) 评分:{score:.0f}')
        else:
            lines.append('\n【今日委员会】无数据')

        return '\n'.join(lines)

    # ---------- 2. 策略有效性分析 ----------

    def strategy_analysis(self) -> str:
        """策略有效性分析 — 回测最近N天的委员会决策"""
        lines = ['\n=== 策略有效性分析 ===']

        committee_data = load_committee_results(days=10)
        if not committee_data:
            lines.append('  近10日无历史委员会数据（需要至少运行2天后才有T+1数据）')
            return '\n'.join(lines)

        # 收集所有决策 + T+1收益
        all_buy = []
        all_hold = []
        all_reject = []

        for data in committee_data:
            decisions = data.get('decisions', [])
            scan_date = data.get('scan_time', '')[:10]

            for d in decisions:
                code = d.get('symbol', '')
                dec = d.get('decision', '')
                score = d.get('composite_score', 0)

                # 获取T+1收益
                t1 = self._calc_single_return(code, scan_date)
                if t1 is None:
                    continue

                entry = {
                    'code': code, 'name': d.get('name', ''),
                    'date': scan_date, 'score': score,
                    't1': t1, 'decision': dec,
                }

                if dec == 'buy':
                    all_buy.append(entry)
                elif dec == 'hold':
                    all_hold.append(entry)
                else:
                    all_reject.append(entry)

        # 统计
        total = len(all_buy) + len(all_hold)
        if total == 0:
            lines.append('  无可回测数据')
            return '\n'.join(lines)

        lines.append(f'\n【近10日统计】')

        for label, group in [('BUY', all_buy), ('HOLD', all_hold), ('REJECT', all_reject)]:
            if not group:
                continue
            rets = [r['t1'] for r in group]
            avg = np.mean(rets)
            win = sum(1 for r in rets if r > 0) / len(rets) * 100
            lines.append(f'  {label} ({len(group)}只): avg {avg:+.2f}% | 胜率 {win:.0f}%')

        # 评分区分度
        if len(all_buy) >= 5:
            scores = [r['score'] for r in all_buy]
            rets = [r['t1'] for r in all_buy]
            corr = np.corrcoef(scores, rets)[0, 1] if len(rets) >= 3 else 0

            high = [r for r in all_buy if r['score'] >= 80]
            low = [r for r in all_buy if r['score'] < 80]

            lines.append(f'\n【评分区分度】')
            lines.append(f'  评分-收益相关性: {corr:.2f}')
            if high:
                h_avg = np.mean([r['t1'] for r in high])
                h_win = sum(1 for r in high if r['t1'] > 0) / len(high) * 100
                lines.append(f'  >=80分 ({len(high)}只): avg {h_avg:+.2f}% 胜率 {h_win:.0f}%')
            if low:
                l_avg = np.mean([r['t1'] for r in low])
                l_win = sum(1 for r in low if r['t1'] > 0) / len(low) * 100
                lines.append(f'  <80分 ({len(low)}只): avg {l_avg:+.2f}% 胜率 {l_win:.0f}%')

        # 最佳/最差推荐
        if all_buy:
            best = max(all_buy, key=lambda x: x['t1'])
            worst = min(all_buy, key=lambda x: x['t1'])
            lines.append(f'\n【最佳推荐】{best["date"]} {code_to_prefix(best["code"])} '
                         f'({best["name"]}) 评分:{best["score"]:.0f} T+1:{best["t1"]:+.2f}%')
            lines.append(f'【最差推荐】{worst["date"]} {code_to_prefix(worst["code"])} '
                         f'({worst["name"]}) 评分:{worst["score"]:.0f} T+1:{worst["t1"]:+.2f}%')

        return '\n'.join(lines)

    # ---------- 3. 风险控制评估 ----------

    def risk_assessment(self) -> str:
        """风险控制评估"""
        lines = ['\n=== 风险控制评估 ===']

        positions = self.positions_data.get('positions', [])

        if not positions:
            lines.append('  当前无持仓，风险可控')
            return '\n'.join(lines)

        # 持仓回撤
        max_drawdown = 0
        near_stop = []
        sectors = {}

        for p in positions:
            code = p.get('code', '')
            entry = p.get('entry_price', 0)
            highest = p.get('highest_since_entry', entry)
            stop = p.get('current_stop', p.get('stop_price', 0))

            try:
                df = self.hs.get_kline(code, period='daily')
                if df is not None and len(df) >= 1:
                    last_close = float(df['close'].iloc[-1])
                    # 从最高点回撤
                    dd = (highest - last_close) / highest * 100 if highest > 0 else 0
                    max_drawdown = max(max_drawdown, dd)

                    # 接近止损检查
                    if stop > 0:
                        distance = (last_close - stop) / last_close * 100
                        if distance < 3:  # 距止损3%以内
                            near_stop.append(f'{code_to_prefix(code)} '
                                           f'距止损仅{distance:.1f}%')

                    # 行业集中度
                    sector = self.sector_map.get(code, '未知')
                    sectors[sector] = sectors.get(sector, 0) + 1
            except Exception:
                continue

        lines.append(f'  持仓数: {len(positions)}只')
        lines.append(f'  最大回撤: {max_drawdown:.1f}%')

        if near_stop:
            lines.append(f'\n  【止损警告】{len(near_stop)}只接近止损:')
            for ns in near_stop:
                lines.append(f'    {ns}')

        # 行业集中度
        if sectors:
            lines.append(f'\n  【行业分布】')
            for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
                pct = count / len(positions) * 100
                if pct >= 30:
                    lines.append(f'    {sector}: {count}只 ({pct:.0f}%) ← 集中!')
                else:
                    lines.append(f'    {sector}: {count}只 ({pct:.0f}%)')

        return '\n'.join(lines)

    # ---------- 4. 热点板块复盘 ----------

    def hot_sector_review(self) -> str:
        """热点板块复盘 — 当日板块表现 + 明日关注"""
        lines = ['\n=== 热点板块复盘 ===', '']
        try:
            from data.hot_sector_analyzer import HotSectorAnalyzer
            hsa = HotSectorAnalyzer()
            sectors = hsa.identify_hot_sectors(top_n=10)
            hsa.save_results(sectors)

            if not sectors:
                lines.append('无热点板块数据')
                return '\n'.join(lines)

            # 当日板块表现
            lines.append('【今日板块表现TOP10】')
            for i, s in enumerate(sectors, 1):
                phase_icon = {'启动': '🟢', '加速': '🟡', '高潮': '🔴', '退潮': '⚪', '震荡': '🔵'}.get(s.phase, '⚪')
                lines.append(f'  {i:>2}. {s.name:<10} {phase_icon}{s.phase} 评分{s.score:.0f} '
                             f'1日{s.return_1d:+.2f}% 5日{s.return_5d:+.2f}% '
                             f'上涨{s.up_ratio_1d:.0%} 涨停{s.limit_up_count}只')

            # 板块阶段统计
            phase_count = {}
            for s in sectors:
                phase_count[s.phase] = phase_count.get(s.phase, 0) + 1
            lines.append(f'\n【阶段分布】' + ' | '.join(f'{k}{v}个' for k, v in sorted(phase_count.items())))

            # 明日关注
            lines.append(f'\n【明日关注】')
            watch = [s for s in sectors if s.phase in ('启动', '加速')]
            if watch:
                for s in watch:
                    lines.append(f'  > {s.name} ({s.phase}, 评分{s.score:.0f}, 涨停{s.limit_up_count}只, '
                                 f'龙一连板{s.dragon_boards}板)')
                    # 板块内缠论候选
                    results = hsa.rank_stocks_in_sector(s.name, top_n=3)
                    if results:
                        for r in results:
                            lines.append(f'    候选: {r["code"]} ¥{r["price"]} {r["type"]} conf={r["confidence"]:.2f}')
                    else:
                        lines.append(f'    等回踩出买点再入场')
            else:
                lines.append('  当前无启动/加速板块，关注明日新热点')

            # 退潮风险提醒
            declining = [s for s in sectors if s.phase == '退潮']
            if declining:
                lines.append(f'\n【退潮风险】')
                for s in declining:
                    lines.append(f'  ⚠ {s.name} ({s.return_1d:+.2f}%) 可能退潮，注意止盈')

        except Exception as e:
            lines.append(f'热点板块复盘失败: {e}')

        return '\n'.join(lines)

    # ---------- 5. 明日策略规划 ----------

    def next_day_plan(self) -> str:
        """明日策略规划"""
        lines = ['\n=== 明日策略规划 ===']

        # 加载最新的委员会推荐
        committee_files = sorted(glob.glob('signals/investment_committee_*.json'),
                                 reverse=True)
        scan_files = sorted(glob.glob('signals/scan_enhanced_*.json'),
                            reverse=True)

        focus_stocks = []

        if committee_files:
            with open(committee_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)

            buy_decisions = [d for d in data.get('decisions', [])
                            if d.get('decision') == 'buy']
            buy_decisions.sort(key=lambda x: x.get('composite_score', 0), reverse=True)

            if buy_decisions:
                lines.append(f'\n【重点关注】(基于最新委员会推荐)')
                for i, d in enumerate(buy_decisions[:5], 1):
                    code = d.get('symbol', '')
                    name = d.get('name', '')
                    score = d.get('composite_score', 0)
                    stop = d.get('stop_loss', 0)
                    entry = d.get('entry_price', 0)
                    factors = ', '.join(d.get('key_factors', [])[:2])
                    risk = d.get('risk_level', '')

                    lines.append(
                        f'  {i}. {code_to_prefix(code)} ({name}) '
                        f'评分:{score:.0f} | 止损:{stop:.2f} | {risk}')
                    lines.append(f'     {factors}')

                    focus_stocks.append(code)

        # 风险预案
        lines.append(f'\n【风险预案】')
        positions = self.positions_data.get('positions', [])
        if positions:
            for p in positions:
                code = p.get('code', '')
                name = p.get('name', '')
                stop = p.get('current_stop', p.get('stop_price', 0))
                if stop > 0:
                    lines.append(f'  - {code_to_prefix(code)} ({name}) '
                               f'破止损{stop:.2f}则卖出')
        else:
            lines.append('  - 当前无持仓，关注明日买入机会')

        # 大盘预判 — 使用MarketAnalyzer
        try:
            from agents.market_analyzer import MarketAnalyzer
            ma = MarketAnalyzer()
            mc = ma.analyze()
            lines.append(f'\n【大盘预判】{mc.index_phase} (regime={mc.regime})')
            lines.append(f'  风险溢价={mc.risk_premium:+.2f} 仓位系数={mc.position_adjust:.1f}')
            if mc.warnings:
                lines.append(f'  风险: {" | ".join(mc.warnings[:3])}')

            # 根据regime给出建议
            advice_map = {
                'strong': '大盘强势，可积极寻找买点，仓位可放大至{:.0%}'.format(mc.position_adjust * 0.8),
                'normal': '大盘正常，维持标准仓位操作',
                'weak': '大盘弱势，降低仓位，仅做强势板块龙头',
                'danger': '大盘危险，建议空仓或极低仓位',
            }
            lines.append(f'  建议: {advice_map.get(mc.regime, "观望")}')
        except Exception as e:
            lines.append(f'\n【大盘预判】分析失败: {e}')

        return '\n'.join(lines)

    # ---------- 辅助方法 ----------

    def _get_index_change(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """获取指数的收盘价和涨跌幅 (通过AKShare)"""
        try:
            import akshare as ak
            # symbol: sh000001 → 000001, sz399001 → 399001
            code = symbol[2:]  # 去掉 sh/sz 前缀
            df = ak.stock_zh_index_daily(symbol=f'sh{code}' if code.startswith('0000') else f'sz{code}')
            if df is not None and len(df) >= 2:
                last = float(df['close'].iloc[-1])
                prev = float(df['close'].iloc[-2])
                chg = (last - prev) / prev * 100
                return last, chg
        except Exception:
            pass

        # Fallback: 用个股近似 (上证用601398工商银行, 深证用000001平安银行)
        try:
            if '000001' in symbol:
                df = self.hs.get_kline('601398', period='daily')
            else:
                df = self.hs.get_kline('000001', period='daily')
            if df is not None and len(df) >= 2:
                last = float(df['close'].iloc[-1])
                prev = float(df['close'].iloc[-2])
                chg = (last - prev) / prev * 100
                return last, chg
        except Exception:
            pass

        return None, None

    def _calc_decisions_returns(self, decisions: List[dict],
                                 fallback_source: str = '') -> List[float]:
        """计算一组决策的T+1收益"""
        rets = []
        for d in decisions:
            code = d.get('symbol', '')
            scan_date = d.get('source_file', '') or fallback_source
            # 从文件名提取日期
            if not scan_date:
                continue
            try:
                date_str = scan_date.split('_')[2]  # 20260417
                r = self._calc_single_return(code, None, date_str)
                if r is not None:
                    rets.append(r)
            except Exception:
                continue
        return rets

    def _calc_single_return(self, code: str, scan_date: str = None,
                            date_str: str = None) -> Optional[float]:
        """计算单只股票从scan_date到下一个交易日的收益

        先用本地数据，如果不够则用Sina实时数据。
        """
        if date_str is None and scan_date:
            try:
                date_str = scan_date.replace('-', '')[:8]
            except Exception:
                return None

        if not date_str:
            return None

        # 方法1: 本地数据
        try:
            df = self.hs.get_kline(code, period='daily')
            if df is not None and len(df) >= 2:
                df_index = df.index.strftime('%Y%m%d')

                target_idx = None
                for i, d in enumerate(df_index):
                    if d >= date_str:
                        target_idx = i
                        break

                if target_idx is not None and target_idx + 1 < len(df):
                    entry = float(df['close'].iloc[target_idx])
                    exit_close = float(df['close'].iloc[target_idx + 1])
                    return (exit_close - entry) / entry * 100

                # 本地数据包含scan_date但无T+1，用最新close vs scan_date close
                if target_idx is not None:
                    entry = float(df['close'].iloc[target_idx])
                    latest = float(df['close'].iloc[-1])
                    if df_index[-1] > date_str:
                        return (latest - entry) / entry * 100
        except Exception:
            pass

        # 方法2: Sina实时
        try:
            import requests, re
            session = requests.Session()
            session.trust_env = False
            prefix = 'sz' if code.startswith(('0', '3')) else 'sh'
            url = (f'https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/'
                   f'CN_MarketDataService.getKLineData?symbol={prefix}{code}'
                   f'&scale=240&ma=no&datalen=10')
            resp = session.get(url, timeout=10)
            match = re.search(r'callback\((.*)\)', resp.text)
            if not match:
                return None
            klines = json.loads(match.group(1))
            if len(klines) < 2:
                return None

            # 找scan_date对应的K线
            entry_price = None
            for i, k in enumerate(klines):
                kdate = k['day'][:10].replace('-', '')
                if kdate >= date_str:
                    entry_price = float(k['close'])
                    # 下一根K线是T+1
                    if i + 1 < len(klines):
                        exit_price = float(klines[i + 1]['close'])
                        return (exit_price - entry_price) / entry_price * 100
                    break

            # 没找到精确日期，用最后两根K线
            if entry_price is None and len(klines) >= 2:
                # scan_date可能比Sina数据更早，取倒数第二根和最后一根
                prev_close = float(klines[-2]['close'])
                last_close = float(klines[-1]['close'])
                # 但这不太准确，跳过
                return None

        except Exception:
            pass

        return None

    def save_report(self, report: str):
        """保存报告"""
        os.makedirs('signals', exist_ok=True)
        filename = f'signals/post_market_{self.today.replace("-", "")}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'  报告已保存: {filename}')

    def _ai_summary(self) -> str:
        """用MiniMax生成AI复盘总结"""
        # 取前4部分的核心内容（去掉过长部分）
        data = '\n'.join(self.report_parts[:2000])
        ai_text = analyze_with_minimax(data, task='post_market')
        if ai_text:
            return f'=== AI复盘总结 ===\n{ai_text}'
        return ''

    def push_report(self):
        """推送报告"""
        title = f'缠论复盘 {datetime.now().strftime("%m-%d")}'

        # 尝试飞书卡片
        elements = []
        for part in self.report_parts:
            # 截取前500字符避免卡片过长
            text = part[:500]
            elements.append({
                'tag': 'div',
                'text': {'tag': 'lark_md', 'content': text}
            })

        if not send_card(title, elements):
            # 回退纯文本
            full_text = '\n\n'.join(self.report_parts)
            send_notification(title, full_text)


# ==================== 入口 ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='复盘Agent')
    parser.add_argument('--force', action='store_true', help='强制运行')
    args = parser.parse_args()

    if not args.force and check_today_done('post_market'):
        print('今日复盘已完成，跳过（用 --force 强制）')
    else:
        agent = PostMarketAgent()
        agent.run()
