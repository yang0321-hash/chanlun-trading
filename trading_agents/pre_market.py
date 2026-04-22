#!/usr/bin/env python3
"""盘前Agent — 严谨分析师

07:00 健康检测
08:25 系统自检
08:30 盘前简报
09:15-09:25 竞价数据获取
09:28 竞价推送
"""
import sys
import os
import json
import time
import traceback
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


def load_positions() -> dict:
    path = 'signals/positions.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'positions': [], 'capital': 1000000}


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


# ==================== 盘前Agent ====================

class PreMarketAgent:
    """盘前Agent — 严谨分析师"""

    def __init__(self):
        self.hs = HybridSource()
        self.positions_data = load_positions()
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.report_parts = []

    def run(self):
        """执行完整盘前流程"""
        print(f'=== 盘前Agent {self.today} ===')
        print()

        # 1. 健康检测
        print('[1] 系统健康检测...')
        health = self.health_check()
        self.report_parts.append(health)

        # 2. 盘前简报
        print('[2] 生成盘前简报...')
        brief = self.pre_market_brief()
        self.report_parts.append(brief)

        # 3. 热点板块识别
        print('[3] 热点板块识别...')
        hot_sector_report = self.hot_sector_scan()
        self.report_parts.append(hot_sector_report)

        # 4. 竞价数据 (09:15后才有)
        now = datetime.now()
        if now.hour >= 9 and now.minute >= 15:
            print('[4] 获取竞价数据...')
            auction = self.fetch_auction_data()
            self.report_parts.append(auction)
        else:
            print('[4] 跳过竞价数据（09:15后才可获取）')
            self.report_parts.append('\n【竞价异动】尚未到竞价时间，盘前无数据')

        # 5. 生成报告 + 推送
        print('[5] 生成报告...')
        report = '\n\n'.join(self.report_parts)
        self.save_report(report)
        self.push_report()

        # 5. 标记完成
        mark_done('pre_market')
        print('\n盘前分析完成')

    # ---------- 1. 健康检测 ----------

    def health_check(self) -> str:
        """系统健康检测"""
        lines = [f'=== 盘前系统检测 {self.today} ===', '']

        checks = {'OK': 0, 'WARN': 0, 'FAIL': 0}

        # 数据源检测
        data_status = self._check_data_source()
        checks[data_status[0]] += 1
        lines.append(f'【数据源】{data_status[1]}')

        # 通知渠道检测
        notify_status = self._check_notification()
        checks[notify_status[0]] += 1
        lines.append(f'【通知渠道】{notify_status[1]}')

        # 持仓数据完整性
        pos_status = self._check_positions()
        checks[pos_status[0]] += 1
        lines.append(f'【持仓数据】{pos_status[1]}')

        # 昨日信号文件
        signal_status = self._check_yesterday_signals()
        checks[signal_status[0]] += 1
        lines.append(f'【昨日信号】{signal_status[1]}')

        # 缠论引擎
        engine_status = self._check_engine()
        checks[engine_status[0]] += 1
        lines.append(f'【缠论引擎】{engine_status[1]}')

        ok = checks['OK']
        warn = checks['WARN']
        fail = checks['FAIL']
        overall = 'OK' if fail == 0 else ('WARN' if warn > 0 else 'FAIL')
        lines.insert(1, f'系统状态: {overall} (OK:{ok} WARN:{warn} FAIL:{fail})')

        return '\n'.join(lines)

    def _check_data_source(self) -> Tuple[str, str]:
        """检测数据源"""
        try:
            # 检测TDX
            tdx_info = self.hs.tdx_status

            # 检测Sina — 获取一只股票实时行情
            price = self.hs.get_realtime_price('000001')
            if price and price > 0:
                return ('OK', f'{tdx_info} | Sina实时:OK')
            else:
                return ('WARN', f'{tdx_info} | Sina实时:无响应(非交易时间可能正常)')
        except Exception as e:
            return ('WARN', f'检测异常: {e}')

    def _check_notification(self) -> Tuple[str, str]:
        """检测通知渠道"""
        webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL')
        if webhook:
            # 不实际发送，只验证URL格式
            if 'open.feishu.cn' in webhook:
                return ('OK', f'飞书机器人已配置')
            return ('WARN', '飞书Webhook格式异常')
        return ('WARN', 'CHANLUN_FEISHU_WEBHOOK_URL 未配置')

    def _check_positions(self) -> Tuple[str, str]:
        """检测持仓数据"""
        positions = self.positions_data.get('positions', [])
        capital = self.positions_data.get('capital', 0)

        if not positions:
            return ('OK', f'无持仓 | 资金:{capital:,.0f}')

        issues = []
        for p in positions:
            if not p.get('code'):
                issues.append('缺少code字段')
            if not p.get('entry_price'):
                issues.append(f'{p.get("code","?")} 缺少entry_price')

        if issues:
            return ('WARN', f'{len(positions)}只持仓 | 问题: {", ".join(issues[:3])}')
        return ('OK', f'{len(positions)}只持仓 | 资金:{capital:,.0f}')

    def _check_yesterday_signals(self) -> Tuple[str, str]:
        """检测昨日信号"""
        import glob
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

        committee = glob.glob(f'signals/investment_committee_{yesterday}*.json')
        scan = glob.glob(f'signals/scan_enhanced_{yesterday}*.json')

        parts = []
        if committee:
            parts.append(f'委员会:{len(committee)}份')
        if scan:
            parts.append(f'扫描:{len(scan)}份')

        if parts:
            return ('OK', f'昨日({yesterday}) ' + ' | '.join(parts))

        # 检查更早的
        all_committee = glob.glob('signals/investment_committee_*.json')
        if all_committee:
            latest = os.path.basename(sorted(all_committee)[-1])
            return ('WARN', f'昨日无信号文件 | 最新: {latest}')

        return ('WARN', '无历史信号文件')

    def _check_engine(self) -> Tuple[str, str]:
        """检测缠论引擎可导入性"""
        try:
            from core.kline import KLine
            from core.fractal import FractalDetector
            from core.stroke import StrokeGenerator
            from core.pivot import PivotDetector
            from core.buy_sell_points import BuySellPointDetector
            return ('OK', '所有模块加载正常')
        except ImportError as e:
            return ('FAIL', f'模块加载失败: {e}')

    # ---------- 2. 盘前简报 ----------

    def pre_market_brief(self) -> str:
        """盘前简报: 大盘概况 + 持仓盈亏 + 昨日信号回顾"""
        lines = [f'\n=== 盘前简报 {self.today} ===', '']

        # 大盘概况
        self._append_market_overview(lines)

        # 持仓概况
        self._append_position_overview(lines)

        # 昨日信号回顾
        self._append_signal_review(lines)

        return '\n'.join(lines)

    # ---------- 3. 热点板块 ----------

    def hot_sector_scan(self) -> str:
        """热点板块识别 + 板块内选股"""
        lines = ['\n=== 热点板块扫描 ===', '']
        try:
            from data.hot_sector_analyzer import HotSectorAnalyzer
            hsa = HotSectorAnalyzer()
            sectors = hsa.identify_hot_sectors(top_n=10)
            hsa.save_results(sectors)

            if not sectors:
                lines.append('无热点板块数据')
                return '\n'.join(lines)

            lines.append(f'TOP10热点板块 (基于TDX本地数据):')
            lines.append(f'{"板块":<10} {"评分":>5} {"阶段":<5} {"5日%":>7} {"上涨率":>6} {"涨停":>4} {"龙一连板":>6}')
            lines.append('-' * 55)
            for s in sectors:
                lines.append(f'{s.name:<10} {s.score:>5.1f} {s.phase:<5} '
                             f'{s.return_5d:>+6.2f}% {s.up_ratio_1d:>5.0%} '
                             f'{s.limit_up_count:>4} {s.dragon_boards:>6}')

            # 启动/加速板块的缠论选股
            actionable = [s for s in sectors if s.phase in ('启动', '加速') and s.stock_count >= 10]
            if actionable:
                lines.append(f'\n【重点关注】启动/加速板块 ({len(actionable)}个):')
                for s in actionable[:3]:
                    lines.append(f'\n  > {s.name} (评分{s.score}, {s.phase}, 涨停{s.limit_up_count}只)')
                    results = hsa.rank_stocks_in_sector(s.name, top_n=3)
                    if results:
                        for r in results:
                            lines.append(f'    {r["code"]} ¥{r["price"]} {r["type"]} conf={r["confidence"]:.2f}')
                    else:
                        lines.append(f'    当前无缠论买点，等回踩确认')

        except Exception as e:
            lines.append(f'热点板块扫描失败: {e}')

        return '\n'.join(lines)

    def _append_market_overview(self, lines: list):
        """大盘概况"""
        lines.append('【大盘概况】')

        try:
            import akshare as ak
            # 上证指数
            df_sh = ak.stock_zh_index_daily(symbol='sh000001')
            df_sz = ak.stock_zh_index_daily(symbol='sz399001')

            if df_sh is not None and len(df_sh) >= 2:
                sh_close = float(df_sh['close'].iloc[-1])
                sh_prev = float(df_sh['close'].iloc[-2])
                sh_chg = (sh_close - sh_prev) / sh_prev * 100
            else:
                sh_close, sh_chg = 0, 0

            if df_sz is not None and len(df_sz) >= 2:
                sz_close = float(df_sz['close'].iloc[-1])
                sz_prev = float(df_sz['close'].iloc[-2])
                sz_chg = (sz_close - sz_prev) / sz_prev * 100
            else:
                sz_close, sz_chg = 0, 0

            if sh_close > 0:
                lines.append(f'  上证: {sh_close:,.0f} ({sh_chg:+.2f}%) | '
                             f'深证: {sz_close:,.0f} ({sz_chg:+.2f}%)')
            else:
                lines.append('  指数数据获取失败')
        except Exception as e:
            lines.append(f'  大盘数据获取失败: {e}')

        # 简单趋势判断 — 用Sina
        try:
            import requests, re
            session = requests.Session()
            session.trust_env = False
            url = (f'https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/'
                   f'CN_MarketDataService.getKLineData?symbol=sh000001'
                   f'&scale=240&ma=no&datalen=30')
            resp = session.get(url, timeout=10)
            match = re.search(r'callback\((.*)\)', resp.text)
            if match:
                klines = json.loads(match.group(1))
                if len(klines) >= 20:
                    closes = np.array([float(k['close']) for k in klines])
                    ma5 = np.mean(closes[-5:])
                    ma20 = np.mean(closes[-20:])
                    last = closes[-1]

                    if last > ma5 > ma20:
                        trend = '多头排列'
                    elif last < ma5 < ma20:
                        trend = '空头排列'
                    else:
                        trend = '震荡'

                    lines.append(f'  趋势: {trend} | MA5:{ma5:,.0f} MA20:{ma20:,.0f}')
        except Exception:
            pass

    def _append_position_overview(self, lines: list):
        """持仓概况"""
        positions = self.positions_data.get('positions', [])

        if not positions:
            lines.append('\n【持仓概况】空仓')
            return

        lines.append(f'\n【持仓概况】{len(positions)}只')

        for p in positions:
            code = p.get('code', '')
            name = p.get('name', '')
            entry = p.get('entry_price', 0)
            stop = p.get('current_stop', p.get('stop_price', 0))

            try:
                price = self.hs.get_realtime_price(code)
                if price and price > 0:
                    pnl = (price - entry) / entry * 100 if entry > 0 else 0

                    # 距止损
                    stop_info = ''
                    if stop > 0:
                        dist = (price - stop) / price * 100
                        if dist < 3:
                            stop_info = f' ← 接近止损!'
                        else:
                            stop_info = f' 距止损{dist:.1f}%'

                    lines.append(f'  {code_to_prefix(code)} ({name}) '
                               f'现价:{price:.2f} 盈亏:{pnl:+.2f}%{stop_info}')
                else:
                    # 非交易时间获取不到实时价，用最新日线收盘价
                    df = self.hs.get_kline(code, period='daily')
                    if df is not None and len(df) >= 1:
                        last_close = float(df['close'].iloc[-1])
                        pnl = (last_close - entry) / entry * 100 if entry > 0 else 0
                        lines.append(f'  {code_to_prefix(code)} ({name}) '
                                   f'昨收:{last_close:.2f} 盈亏:{pnl:+.2f}%')
                    else:
                        lines.append(f'  {code_to_prefix(code)} ({name}) 数据不可用')
            except Exception:
                lines.append(f'  {code_to_prefix(code)} ({name}) 获取价格失败')

    def _append_signal_review(self, lines: list):
        """昨日信号回顾"""
        import glob

        lines.append('\n【昨日信号回顾】')

        # 找最新的委员会结果
        committee_files = sorted(glob.glob('signals/investment_committee_*.json'),
                                 reverse=True)
        if not committee_files:
            lines.append('  无历史委员会数据')
            return

        # 取最近一个交易日的
        with open(committee_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)

        scan_date = os.path.basename(committee_files[0]).split('_')[2]
        buy_count = data.get('buy_count', 0)
        hold_count = data.get('hold_count', 0)
        reject_count = data.get('reject_count', 0)

        lines.append(f'  {scan_date} 委员会: buy:{buy_count} hold:{hold_count} reject:{reject_count}')

        # 计算BUY推荐的T+1表现
        buy_decisions = [d for d in data.get('decisions', [])
                        if d.get('decision') == 'buy']

        if buy_decisions:
            rets = []
            for d in buy_decisions:
                code = d.get('symbol', '')
                try:
                    df = self.hs.get_kline(code, period='daily')
                    if df is not None and len(df) >= 2:
                        # scan_date日的close vs 次日close
                        df_dates = df.index.strftime('%Y%m%d')
                        idx = None
                        for i, dt in enumerate(df_dates):
                            if dt >= scan_date:
                                idx = i
                                break
                        if idx is not None and idx + 1 < len(df):
                            entry_p = float(df['close'].iloc[idx])
                            exit_p = float(df['close'].iloc[idx + 1])
                            ret = (exit_p - entry_p) / entry_p * 100
                            rets.append(ret)
                            d['_t1'] = ret
                except Exception:
                    continue

            if rets:
                avg_ret = np.mean(rets)
                win = sum(1 for r in rets if r > 0) / len(rets) * 100
                lines.append(f'  BUY组 T+1: avg {avg_ret:+.2f}% 胜率 {win:.0f}%')

                # 涨幅前3
                sorted_buy = sorted(buy_decisions, key=lambda x: x.get('_t1', -999), reverse=True)
                for d in sorted_buy[:3]:
                    if '_t1' in d:
                        lines.append(f'    {code_to_prefix(d["symbol"])} ({d.get("name","")}) '
                                   f'{d["_t1"]:+.2f}%')

    # ---------- 3. 竞价数据 ----------

    def fetch_auction_data(self) -> str:
        """获取集合竞价数据（容错，失败不影响其他功能）"""
        lines = ['\n【竞价异动】']

        # 方法1: AKShare实时行情（包含竞价信息）
        auction_stocks = self._fetch_auction_akshare()

        if auction_stocks:
            lines.append(f'  检测到 {len(auction_stocks)} 只异动股:')
            for i, s in enumerate(auction_stocks[:10], 1):
                lines.append(f'    {i}. {code_to_prefix(s["code"])} ({s["name"]}) '
                           f'竞价:{s.get("pct_chg", 0):+.2f}% 量比:{s.get("volume_ratio", 0):.1f}')
            return '\n'.join(lines)

        # 方法2: Sina实时报价（非交易时间可能无数据）
        auction_stocks = self._fetch_auction_sina()

        if auction_stocks:
            lines.append(f'  检测到 {len(auction_stocks)} 只异动股:')
            for i, s in enumerate(auction_stocks[:10], 1):
                lines.append(f'    {i}. {code_to_prefix(s["code"])} ({s["name"]}) '
                           f'{s.get("pct_chg", 0):+.2f}%')
            return '\n'.join(lines)

        lines.append('  竞价数据获取失败（非交易时间或API不可用）')
        return '\n'.join(lines)

    def _fetch_auction_akshare(self) -> List[dict]:
        """通过AKShare获取竞价数据"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()

            if df is None or len(df) == 0:
                return []

            # 筛选高开异动（竞价涨幅>2%，且排除ST/退市）
            results = []
            for _, row in df.iterrows():
                code = str(row.get('代码', ''))
                name = str(row.get('名称', ''))

                # 排除ST、退市、北交所
                if 'ST' in name or '退' in name:
                    continue
                if code.startswith(('4', '8')):
                    continue

                # 检查是否在候选池或委员会推荐中
                if not self._is_in_watchlist(code):
                    continue

                pct = float(row.get('涨跌幅', 0)) if row.get('涨跌幅') else 0
                vol_ratio = float(row.get('量比', 0)) if row.get('量比') else 0

                # 竞价异动: 涨幅>2% 或 量比>2
                if pct > 2 or (vol_ratio > 2 and pct > 0.5):
                    results.append({
                        'code': code,
                        'name': name,
                        'pct_chg': pct,
                        'volume_ratio': vol_ratio,
                    })

            results.sort(key=lambda x: x.get('pct_chg', 0), reverse=True)
            return results

        except Exception as e:
            print(f'  AKShare竞价数据获取失败: {e}')
            return []

    def _fetch_auction_sina(self) -> List[dict]:
        """通过Sina获取实时行情作为竞价替代"""
        try:
            # 获取watchlist中的股票实时报价
            watchlist = self._get_watchlist_codes()
            if not watchlist:
                return []

            # 分批获取（Sina一次最多约30只）
            results = []
            for i in range(0, len(watchlist), 30):
                batch = watchlist[i:i+30]
                df = self.hs.get_realtime_quote(batch)
                if df is None or len(df) == 0:
                    continue

                for _, row in df.iterrows():
                    pct = float(row.get('pct_chg', 0))
                    if pct > 2:  # 高开超过2%
                        results.append({
                            'code': str(row.get('code', '')),
                            'name': str(row.get('name', '')),
                            'pct_chg': pct,
                        })

            results.sort(key=lambda x: x.get('pct_chg', 0), reverse=True)
            return results

        except Exception as e:
            print(f'  Sina行情获取失败: {e}')
            return []

    def _is_in_watchlist(self, code: str) -> bool:
        """检查股票是否在候选池中"""
        watchlist = self._get_watchlist_codes()
        # 匹配时去掉前缀
        code_clean = code.lstrip('shsz')
        for w in watchlist:
            w_clean = w.lstrip('shsz')
            if code_clean == w_clean:
                return True
        return False

    def _get_watchlist_codes(self) -> List[str]:
        """获取候选池代码"""
        import glob
        codes = []

        # 从持仓中获取
        for p in self.positions_data.get('positions', []):
            c = p.get('code', '')
            if c:
                codes.append(c)

        # 从最新委员会推荐中获取
        committee_files = sorted(glob.glob('signals/investment_committee_*.json'),
                                 reverse=True)
        if committee_files:
            try:
                with open(committee_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for d in data.get('decisions', []):
                    if d.get('decision') in ('buy', 'hold'):
                        c = d.get('symbol', '')
                        if c and c not in codes:
                            codes.append(c)
            except Exception:
                pass

        # 从扫描结果中获取
        scan_files = sorted(glob.glob('signals/scan_enhanced_*.json'), reverse=True)
        if scan_files:
            try:
                with open(scan_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for s in data.get('signals', data.get('results', [])):
                    c = s.get('symbol', s.get('code', ''))
                    if c and c not in codes:
                        codes.append(c)
            except Exception:
                pass

        return codes

    # ---------- 报告保存和推送 ----------

    def save_report(self, report: str):
        """保存报告"""
        os.makedirs('signals', exist_ok=True)
        filename = f'signals/pre_market_{self.today.replace("-", "")}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'  报告已保存: {filename}')

    def push_report(self):
        """推送报告"""
        title = f'缠论盘前 {datetime.now().strftime("%m-%d")}'

        # 尝试飞书卡片
        elements = []
        for part in self.report_parts:
            text = part[:500]
            elements.append({
                'tag': 'div',
                'text': {'tag': 'lark_md', 'content': text}
            })

        if not send_card(title, elements):
            full_text = '\n\n'.join(self.report_parts)
            send_notification(title, full_text)


# ==================== 入口 ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='盘前Agent')
    parser.add_argument('--force', action='store_true', help='强制运行')
    args = parser.parse_args()

    if not args.force and check_today_done('pre_market'):
        print('今日盘前分析已完成，跳过（用 --force 强制）')
    else:
        agent = PreMarketAgent()
        agent.run()
