#!/usr/bin/env python3
"""每日自动化工作流 — 收盘后扫描 + 委员会评估 + 推送通知

使用方法:
  python daily_workflow.py              # 手动运行（不推送）
  python daily_workflow.py --notify     # 运行并发送通知
  python daily_workflow.py --auto       # 自动模式（仅交易日收盘后运行）
  python daily_workflow.py --register   # 注册Windows定时任务（每交易日15:05）
"""
import sys
import os
import traceback

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
          'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import json
import numpy as np

# 加载 .env 环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from data.hybrid_source import HybridSource


# ==================== 配置 ====================

POOL_FILE = 'chanlun_system/csi500_pool_all_a.json'
SECTOR_FILE = 'chanlun_system/full_sector_map.json'
POSITIONS_FILE = 'signals/positions.json'
DAILY_LOG_FILE = 'signals/daily_log.json'
TOP_N = 15


# ==================== 工具函数 ====================

def load_sector_map() -> Dict[str, str]:
    """加载行业映射"""
    for sp in [SECTOR_FILE, 'chanlun_system/thshy_sector_map.json']:
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


def load_pool_codes() -> List[str]:
    """加载候选池股票代码（转为纯数字格式）"""
    if os.path.exists(POOL_FILE):
        with open(POOL_FILE, 'r') as f:
            data = json.load(f)
        codes = []
        for c in data:
            # 000001.SZ → 000001
            code = c.split('.')[0] if '.' in c else c
            codes.append(code)
        return codes
    return []


def load_positions() -> dict:
    """加载当前持仓"""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'positions': [], 'capital': 1000000, 'update_time': ''}


def save_positions(data):
    """保存持仓"""
    os.makedirs('signals', exist_ok=True)
    data['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_daily_log() -> List[dict]:
    """加载运行日志"""
    if os.path.exists(DAILY_LOG_FILE):
        with open(DAILY_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_daily_log(log: List[dict]):
    """保存运行日志"""
    os.makedirs('signals', exist_ok=True)
    with open(DAILY_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log[-90:], f, ensure_ascii=False, indent=2)  # 保留90天


def check_today_done() -> bool:
    """检查今天是否已运行过"""
    log = load_daily_log()
    today = datetime.now().strftime('%Y-%m-%d')
    return any(entry.get('date') == today for entry in log)


def is_trading_day() -> bool:
    """检查当前是否为交易日收盘后"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    if now.hour < 14 or (now.hour == 14 and now.minute < 30):
        return False
    return True


def code_to_prefix(code: str) -> str:
    """纯数字代码带前缀 sz/sh"""
    if code.startswith(('sz', 'sh', 'SZ', 'SH')):
        return code.lower()
    if code.startswith(('0', '3')):
        return f'sz{code}'
    return f'sh{code}'


# ==================== 扫描 ====================

def run_scan_enhanced(hs: HybridSource, top_n: int = TOP_N) -> List[dict]:
    """主模式：使用 scan_enhanced_v3 扫描"""
    try:
        from scan_enhanced_v3 import scan_enhanced
        results = scan_enhanced(pool='tdx_all', lookback_days=30, min_price=2.0, max_price=2000.0, top_n=top_n)
        return results or []
    except Exception as e:
        print(f'  scan_enhanced_v3 失败: {e}')
        return []


def run_scan_fallback(hs: HybridSource, codes: List[str],
                      top_n: int = TOP_N) -> List[dict]:
    """回退模式：纯缠论日线级别扫描

    对候选池运行缠论分析，检测近期买点，按置信度+R/R排序。
    """
    from core.kline import KLine
    from core.fractal import detect_fractals
    from core.stroke import generate_strokes
    from core.segment import SegmentGenerator
    from core.pivot import detect_pivots, PivotLevel
    from core.buy_sell_points import BuySellPointDetector
    from indicator.macd import MACD

    sector_map = load_sector_map()
    candidates = []

    total = min(len(codes), 200)  # 限制扫描数量避免太慢
    print(f'  回退扫描: {total}/{len(codes)} 只')

    for i, code in enumerate(codes[:total]):
        if (i + 1) % 50 == 0:
            print(f'    [{i+1}/{total}]...')

        try:
            df = hs.get_kline(code, period='daily')
            if df is None or len(df) < 120:
                continue

            kline = KLine.from_dataframe(df)
            fractals = detect_fractals(kline)
            strokes = generate_strokes(kline, fractals)
            segments = SegmentGenerator(kline, strokes).get_segments()
            pivots = detect_pivots(kline, strokes, level=PivotLevel.DAY)
            macd = MACD(df['close'])

            detector = BuySellPointDetector(
                fractals=fractals, strokes=strokes, segments=segments,
                pivots=pivots, macd=macd, fuzzy_tolerance=0.005,
            )
            buys, _ = detector.detect_all()

            if not buys:
                continue

            last_close = float(df['close'].iloc[-1])

            # 只取最近10根K线内的买点
            recent_buys = [b for b in buys if b.index >= len(df) - 10]
            if not recent_buys:
                continue

            best = max(recent_buys, key=lambda b: b.confidence)

            # 计算止损和R/R
            stop = best.stop_price if hasattr(best, 'stop_price') and best.stop_price else 0
            if stop <= 0:
                # 用最近笔低点或中枢ZD
                for p in reversed(pivots):
                    if p.end_index < best.index:
                        stop = p.zd * 0.99 if p.zd > 0 else p.low * 0.99
                        break
            if stop <= 0:
                stop = last_close * 0.93

            risk = last_close - stop
            reward = risk * 2.0  # 简单2:1目标
            rr = reward / risk if risk > 0 else 0

            # 综合评分
            score = best.confidence * 60 + min(rr, 3) * 10
            if best.point_type in ('1buy', '2buy', '3buy'):
                score += 15
            elif 'quasi' in best.point_type:
                score += 5

            candidates.append({
                'code': code,
                'name': sector_map.get(code, ''),
                'sector': sector_map.get(code, ''),
                'price': last_close,
                'entry_price': last_close,
                'stop_price': round(stop, 2),
                'risk_reward': round(rr, 1),
                'total_score': round(score, 1),
                'confidence': round(best.confidence, 2),
                'buy_type': best.point_type,
                'buy_reason': best.reason if hasattr(best, 'reason') else '',
                '2buy_date': df.index[best.index].strftime('%Y-%m-%d')
                    if best.index < len(df) else '',
            })

        except Exception:
            continue

    candidates.sort(key=lambda x: x['total_score'], reverse=True)
    return candidates[:top_n]


# ==================== 持仓管理 ====================

def update_stops(positions: list, hs: HybridSource) -> list:
    """更新持仓股票的止损价和移动止损"""
    updated = []
    for pos in positions:
        code = pos['code']
        try:
            df = hs.get_kline(code, period='daily')
            if df is None or len(df) < 20:
                continue

            last_close = float(df['close'].iloc[-1])
            pos['highest_since_entry'] = max(
                pos.get('highest_since_entry', pos['entry_price']), last_close)

            # 移动止损: 从最高点回撤15%
            trail_stop = pos['highest_since_entry'] * 0.85
            current_stop = pos.get('stop_price', 0)
            pos['current_stop'] = max(trail_stop, current_stop)

            # 盈利保护: 盈利>20%时止损不低于成本
            if pos['highest_since_entry'] > pos['entry_price'] * 1.20:
                pos['current_stop'] = max(pos['current_stop'], pos['entry_price'])

            pos['current_price'] = last_close
            pos['pnl_pct'] = round(
                (last_close - pos['entry_price']) / pos['entry_price'] * 100, 2)
            pos['status'] = '正常'

            if last_close <= pos['current_stop']:
                pos['status'] = '触发止损!'

        except Exception as e:
            pos['status'] = f'更新失败: {e}'

        updated.append(pos)
    return updated


# ==================== 委员会评估 ====================

def run_committee(candidates: List[dict], hs: HybridSource,
                  sector_map: Dict[str, str]) -> List[dict]:
    """运行投资委员会评估"""
    try:
        from agents.investment_committee import InvestmentCommittee
        from trading_agents.position_manager import PositionManager

        # 使用 PositionManager 获取持仓状态（含行业信息）
        pm = PositionManager()
        positions_data = {
            'capital': pm.capital,
            'max_positions': 10,
            'positions': [
                {
                    'code': p.code,
                    'name': p.name,
                    'sector': p.sector,
                    'entry_price': p.entry_price,
                    'shares': p.shares,
                    'stop_price': p.current_stop,
                }
                for p in pm.get_all_positions()
            ],
        }

        committee = InvestmentCommittee(
            hs=hs, sector_map=sector_map,
            portfolio_state=positions_data,
        )
        results = committee.evaluate_batch(candidates[:TOP_N])

        # 行业去相关: 标记同行业过多推荐的警告
        results = _diversify_results(results, sector_map)

        # 每日BUY上限: 最多保留评分最高的3只BUY推荐
        buy_results = [r for r in results if r.get('decision') == 'buy']
        if len(buy_results) > 3:
            buy_results.sort(key=lambda r: r.get('composite_score', 0), reverse=True)
            top_symbols = {r.get('symbol', '') or r.get('code', '') for r in buy_results[:3]}
            for r in results:
                sym = r.get('symbol', '') or r.get('code', '')
                if r.get('decision') == 'buy' and sym not in top_symbols:
                    r['decision'] = 'hold'
                    r['downgrade_reason'] = '每日BUY上限(3只)'

        # 保存结果
        committee.save_results(results)
        return results
    except Exception as e:
        print(f'  委员会评估失败: {e}')
        traceback.print_exc()
        return []


def _diversify_results(results: List[dict], sector_map: Dict[str, str]) -> List[dict]:
    """行业去相关: 同行业推荐过多时降低优先级"""
    # 统计BUY推荐的行业分布
    sector_buy_count = {}
    for r in results:
        if r.get('decision') == 'buy':
            sector = r.get('sector', '未知')
            sector_buy_count[sector] = sector_buy_count.get(sector, 0) + 1

    # 对同行业第3+只的BUY降为HOLD
    sector_seen = {}
    for r in results:
        if r.get('decision') != 'buy':
            continue
        sector = r.get('sector', '未知')
        sector_seen[sector] = sector_seen.get(sector, 0) + 1
        if sector_seen[sector] > 2:  # 同行业最多推荐2只
            r['decision'] = 'hold'
            r['diversify_downgrade'] = True
            r.setdefault('key_factors', []).append(
                f'行业去相关: {sector}已有{sector_seen[sector]-1}只BUY推荐'
            )

    return results


# ==================== 报告格式化 ====================

def format_report_text(committee_results: List[dict],
                       positions_data: dict,
                       scan_count: int = 0) -> str:
    """格式化纯文本报告（微信/钉钉/通用）"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = [f'缠论每日报告 {now}', '']

    # 委员会推荐
    buy = [r for r in committee_results if r.get('decision') == 'buy']
    hold = [r for r in committee_results if r.get('decision') == 'hold']
    reject = [r for r in committee_results if r.get('decision') == 'reject']

    if buy:
        tier_labels = {1: '[主线]', 2: '[活跃]', 3: '[发现]'}
        lines.append(f'【委员会推荐 BUY】({len(buy)}只)')
        for i, r in enumerate(buy[:5], 1):
            code = r.get('symbol', '') or r.get('code', '')
            name = r.get('name', '')
            score = r.get('composite_score', 0)
            stop = r.get('stop_loss', 0)
            rr = r.get('risk_reward', 0)
            factors = ', '.join(r.get('key_factors', [])[:2])
            tier = r.get('sector_tier', 2)
            tl = tier_labels.get(tier, '')
            lines.append(
                f'  {i}. {tl}{code_to_prefix(code)} ({name}) '
                f'评分:{score:.0f} 止损:{stop:.2f} R/R:{rr:.1f}')
            if factors:
                lines.append(f'     {factors}')
        lines.append('')

    if hold:
        lines.append(f'【观望 HOLD】({len(hold)}只)')
        for r in hold[:3]:
            code = r.get('symbol', '') or r.get('code', '')
            score = r.get('composite_score', 0)
            lines.append(f'  {code_to_prefix(code)} 评分:{score:.0f}')
        lines.append('')

    # 持仓跟踪
    positions = positions_data.get('positions', [])
    if positions:
        total_pnl = np.mean([p.get('pnl_pct', 0) for p in positions])
        lines.append(f'【持仓跟踪】{len(positions)}只 平均盈亏:{total_pnl:+.2f}%')
        for p in positions:
            mark = '!!' if '触发' in p.get('status', '') else '  '
            code = p.get('code', '')
            name = p.get('name', '')
            price = p.get('current_price', 0)
            pnl = p.get('pnl_pct', 0)
            stop = p.get('current_stop', 0)
            lines.append(
                f'  {mark} {code_to_prefix(code)} ({name}) '
                f'现价:{price:.2f} 盈亏:{pnl:+.2f}% 止损:{stop:.2f}')
        lines.append('')

        alerts = [p for p in positions if '触发' in p.get('status', '')]
        if alerts:
            lines.append(f'【止损警告】{len(alerts)}只触发止损!')
    else:
        lines.append('【持仓跟踪】当前无持仓')

    lines.append('')
    lines.append(
        f'【市场概况】扫描:{scan_count}只 | '
        f'评估:{len(committee_results)}只 | '
        f'buy:{len(buy)} hold:{len(hold)} reject:{len(reject)}')

    return '\n'.join(lines)


def build_feishu_card(committee_results: List[dict],
                      positions_data: dict) -> List[dict]:
    """构建飞书卡片元素"""
    elements = []

    buy = [r for r in committee_results if r.get('decision') == 'buy']
    hold = [r for r in committee_results if r.get('decision') == 'hold']
    reject = [r for r in committee_results if r.get('decision') == 'reject']

    # 委员会推荐
    if buy:
        lines = []
        for i, r in enumerate(buy[:5], 1):
            code = r.get('symbol', '') or r.get('code', '')
            name = r.get('name', '')
            score = r.get('composite_score', 0)
            stop = r.get('stop_loss', 0)
            factors = ', '.join(r.get('key_factors', [])[:2])
            lines.append(
                f'{i}. {code_to_prefix(code)} ({name}) '
                f'评分:{score:.0f} 止损:{stop:.2f}')
            if factors:
                lines.append(f'   {factors}')
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md",
                     "content": f"**推荐 BUY ({len(buy)}只)**\n" + '\n'.join(lines)}
        })

    # 持仓
    positions = positions_data.get('positions', [])
    if positions:
        pos_lines = []
        for p in positions:
            mark = '[!]' if '触发' in p.get('status', '') else '[OK]'
            pnl = p.get('pnl_pct', 0)
            pos_lines.append(
                f'{mark} {code_to_prefix(p.get("code", ""))} '
                f'{p.get("pnl_pct", 0):+.2f}%')
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md",
                     "content": f"**持仓 ({len(positions)}只)**\n" + '\n'.join(pos_lines)}
        })

    # 统计
    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md",
                 "content": f"评估:{len(committee_results)}只 | "
                            f"buy:{len(buy)} hold:{len(hold)} reject:{len(reject)}"}
    })

    return elements


# ==================== 通知 ====================

def send_report(report_text: str, committee_results: List[dict] = None,
                positions_data: dict = None):
    """发送通知到缠论专用飞书机器人"""
    title = f'缠论日报 {datetime.now().strftime("%m-%d")}'
    sent = False

    # 优先使用缠论专用飞书机器人
    chanlun_webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL')
    if chanlun_webhook:
        try:
            from utils.notification import FeishuConfig, FeishuNotifier
            notifier = FeishuNotifier(FeishuConfig(webhook_url=chanlun_webhook))

            if committee_results:
                elements = build_feishu_card(
                    committee_results, positions_data or {'positions': []})
                notifier.send_card(title, elements)
            else:
                notifier.send_text(report_text)
            print('  缠论飞书机器人推送成功')
            sent = True
        except Exception as e:
            print(f'  缠论飞书机器人推送失败: {e}')

    # 回退到通用通知渠道
    if not sent:
        try:
            from utils.notification import load_notification_config
            nm = load_notification_config()
            nm.send_all(title, report_text)
            print('  通用通知渠道推送成功')
            sent = True
        except Exception as e:
            print(f'  通用通知渠道推送失败: {e}')

    if not sent:
        print('  无可用通知渠道（请在 .env 配置 CHANLUN_FEISHU_WEBHOOK_URL）')


def send_error_notification(error_msg: str):
    """发送错误通知到缠论专用飞书机器人"""
    chanlun_webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL')
    if chanlun_webhook:
        try:
            from utils.notification import FeishuConfig, FeishuNotifier
            notifier = FeishuNotifier(FeishuConfig(webhook_url=chanlun_webhook))
            notifier.send_text(f'缠论日报错误 {datetime.now().strftime("%m-%d")}\n今日扫描失败:\n{error_msg}')
        except Exception:
            pass


# ==================== 定时任务 ====================

def register_scheduled_task():
    """注册Windows定时任务 — 每个交易日14:30运行"""
    # 使用 bat 文件包装（避免中文路径问题）
    bat_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'run_daily.bat')
    if not os.path.exists(bat_file):
        print(f'错误: 找不到 {bat_file}')
        return

    # 先删除旧任务
    os.system('schtasks /delete /tn "ChanLunDaily" /f 2>nul')

    cmd = (
        f'schtasks /create /tn "ChanLunDaily" '
        f'/tr "\\"{bat_file}\\"" '
        f'/sc weekly /d MON,TUE,WED,THU,FRI /st 14:30 /f'
    )
    result = os.system(cmd)

    if result == 0:
        print('定时任务注册成功: ChanLunDaily')
        print(f'  时间: 周一至周五 14:30')
        print(f'  脚本: {bat_file}')
        print(f'\n管理命令:')
        print(f'  查看: schtasks /query /tn "ChanLunDaily"')
        print(f'  删除: schtasks /delete /tn "ChanLunDaily" /f')
        print(f'  手动: python daily_workflow.py --notify')
    else:
        print(f'定时任务注册失败 (错误码: {result})')
        print('  请以管理员身份运行')


# ==================== 主流程 ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='缠论每日自动化工作流')
    parser.add_argument('--notify', action='store_true', help='发送通知')
    parser.add_argument('--auto', action='store_true', help='自动模式（仅交易日运行）')
    parser.add_argument('--register', action='store_true', help='注册Windows定时任务')
    parser.add_argument('--force', action='store_true', help='强制运行（忽略去重）')
    parser.add_argument('--top', type=int, default=TOP_N, help=f'Top N候选（默认{TOP_N}）')
    args = parser.parse_args()

    # 注册定时任务
    if args.register:
        register_scheduled_task()
        return

    # 自动模式检查
    if args.auto:
        if not is_trading_day():
            print('当前非交易日或未收盘，跳过')
            return
        if check_today_done() and not args.force:
            print('今日已完成扫描，跳过（用 --force 强制重跑）')
            return

    print(f'=== 缠论每日工作流 {datetime.now().strftime("%Y-%m-%d %H:%M")} ===')
    print()

    hs = HybridSource()
    sector_map = load_sector_map()
    scan_count = 0

    # ---- Phase 0: 板块候选池扫描 ----
    print('[0] 板块候选池扫描...')
    try:
        from trading_agents.sector_scanner import SectorScanner
        ss = SectorScanner()
        sector_report = ss.run()
        main_sectors = ss.get_main_theme_sectors()
        if main_sectors:
            print(f'  今日主线: {", ".join(main_sectors[:5])}')
    except Exception as e:
        print(f'  板块扫描跳过: {e}')
        main_sectors = []

    # ---- Phase 1: 扫描 ----
    print('[1] 扫描买点信号...')
    candidates = run_scan_enhanced(hs, top_n=args.top)

    if not candidates:
        print('  增强扫描无结果，切换回退模式...')
        codes = load_pool_codes()
        if codes:
            candidates = run_scan_fallback(hs, codes, top_n=args.top)
            scan_count = min(len(codes), 200)
        else:
            print('  无候选池，跳过扫描')

    # 构建候选lookup (Phase 2.5需要日线中枢数据)
    candidate_lookup = {}
    if candidates:
        scan_count = scan_count or len(candidates)
        for c in candidates:
            c_code = c.get('code', '')
            if c_code:
                candidate_lookup[c_code] = c
        print(f'  扫描完成: {len(candidates)} 只候选股')
        for i, c in enumerate(candidates[:5], 1):
            code = c.get('code', '')
            name = c.get('name', '')
            score = c.get('total_score', 0)
            buy_type = c.get('buy_type', '')
            print(f'    {i}. {code_to_prefix(code)} ({name}) '
                  f'评分:{score} 买点:{buy_type}')
    else:
        print('  未发现买点信号')

    # ---- Phase 2: 投资委员会评估 ----
    print()
    print('[2] 投资委员会评估...')
    committee_results = []
    if candidates:
        committee_results = run_committee(candidates, hs, sector_map)
        if committee_results:
            buy_count = sum(1 for r in committee_results if r.get('decision') == 'buy')
            hold_count = sum(1 for r in committee_results if r.get('decision') == 'hold')
            reject_count = sum(1 for r in committee_results if r.get('decision') == 'reject')
            print(f'  评估完成: buy:{buy_count} hold:{hold_count} reject:{reject_count}')

            # 打印buy推荐
            for r in committee_results:
                if r.get('decision') == 'buy':
                    code = r.get('symbol', '') or r.get('code', '')
                    name = r.get('name', '')
                    score = r.get('composite_score', 0)
                    stop = r.get('stop_loss', 0)
                    position = r.get('position_pct', 0)
                    print(f'    >> BUY {code_to_prefix(code)} ({name}) '
                          f'评分:{score:.0f} 仓位:{position:.0%} 止损:{stop:.2f}')
        else:
            print('  委员会评估无结果')
    else:
        print('  无候选股，跳过委员会评估')

    # ---- Phase 2.5: 30min二次确认 (日线信号感知) ----
    print()
    print('[2.5] 30分钟二次确认...')
    v3a_confirmed = []
    if committee_results:
        buy_results = [r for r in committee_results if r.get('decision') == 'buy']
        if buy_results:
            try:
                from strategies.daily_30min_confirm import Daily30minConfirmer
                confirmer = Daily30minConfirmer(hs)

                for r in buy_results:
                    code = r.get('symbol', '') or r.get('code', '')
                    name = r.get('name', '')
                    score = r.get('composite_score', 0)

                    # 从原始候选数据获取日线中枢信息
                    orig = candidate_lookup.get(code, {})
                    daily_context = {
                        'signal_type': orig.get('signal_type', '2buy'),
                        'pivot_zg': orig.get('pivot_zg', 0),
                        'pivot_zd': orig.get('pivot_zd', 0),
                        'pivot_gg': orig.get('pivot_gg', orig.get('pivot_zg', 0)),
                        'entry_price': r.get('entry_price', orig.get('entry_price', 0)),
                        'stop_price': r.get('stop_loss', orig.get('stop_price', 0)),
                    }

                    result = confirmer.confirm_daily_signal(code, daily_context)
                    if result and result.passed and result.confidence >= 0.45:
                        r['v3a_confirmed'] = True
                        r['v3a_confidence'] = result.confidence
                        r['v3a_signal_type'] = daily_context['signal_type']
                        r['v3a_stop'] = result.stop_loss
                        committee_stop = r.get('stop_loss', 0)
                        final_stop = max(committee_stop, result.stop_loss) if committee_stop > 0 else result.stop_loss
                        r['final_stop'] = final_stop
                        v3a_confirmed.append(r)
                        print(f'  [OK] {code_to_prefix(code)} ({name}) '
                              f'确认通过 conf={result.confidence:.2f} '
                              f'类型={daily_context["signal_type"]} '
                              f'止损={final_stop:.2f} ({result.reason})')
                    else:
                        r['decision'] = 'hold'
                        r['v3a_confirmed'] = False
                        conf_str = f'{result.confidence:.2f}' if result else '无数据'
                        reason_str = f' ({result.reason})' if result else ''
                        print(f'  [X] {code_to_prefix(code)} ({name}) '
                              f'未确认 ({conf_str}){reason_str} -> 降为HOLD')

                if v3a_confirmed:
                    print(f'\n  30min确认: {len(v3a_confirmed)}/{len(buy_results)}只')

                    chanlun_webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL')
                    if chanlun_webhook and args.notify:
                        try:
                            from utils.notification import FeishuConfig, FeishuNotifier
                            notifier = FeishuNotifier(FeishuConfig(webhook_url=chanlun_webhook))
                            lines = []
                            for r in v3a_confirmed:
                                code = r.get('symbol', '') or r.get('code', '')
                                lines.append(
                                    f"{code_to_prefix(code)} ({r.get('name', '')}) "
                                    f"评分:{r.get('composite_score', 0):.0f} "
                                    f"确认={r.get('v3a_confidence', 0):.2f} "
                                    f"止损:{r.get('final_stop', 0):.2f}")
                            notifier.send_text(
                                f'30min确认买入 ({len(v3a_confirmed)}只)\n' +
                                '\n'.join(lines) +
                                '\n\n[!] 请人工确认后操作')
                            print('  飞书通知已推送')
                        except Exception as e:
                            print(f'  通知推送失败: {e}')
                else:
                    print('  无确认通过')
            except Exception as e:
                print(f'  确认模块加载失败: {e}')
                import traceback; traceback.print_exc()
        else:
            print('  无BUY推荐，跳过')
    else:
        print('  无委员会结果，跳过')

    # ---- Phase 2.8: v2.0规则引擎 — 仓位调整 ----
    print()
    print('[2.8] v2.0规则引擎 — 仓位调整...')
    try:
        from strategies.trading_rules import TradingRules
        # 获取大盘评分: 优先读缓存，回退到实时计算
        market_score = 6  # 默认偏强
        score_loaded = False

        # 1) 读今日盘前/板块扫描缓存的评分
        today_str = datetime.now().strftime('%Y-%m-%d')
        for cache_file in [f'signals/market_score_{today_str}.json',
                           f'signals/sector_pool_{today_str}.json']:
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                    if 'market_score' in cached:
                        market_score = cached['market_score']
                        score_loaded = True
                        print(f'  大盘评分: {market_score}/12 (来自{os.path.basename(cache_file)})')
                        break
                except Exception:
                    pass

        # 2) 回退/验证: 用TDX本地数据实时计算
        try:
            idx_df = hs.get_kline('sh000001', period='daily')
            if idx_df is not None and len(idx_df) >= 25:
                m_closes = idx_df['close'].values.astype(float)
                m_volumes = idx_df.get('volume')
                if m_volumes is not None:
                    m_volumes = m_volumes.values.astype(float)
                mr = TradingRules.calc_market_score(m_closes, m_volumes)
                if not score_loaded:
                    market_score = mr.score
                    score_loaded = True
                    print(f'  大盘评分: {mr.score}/12 ({mr.state}, 本地计算)')
                elif mr.score != market_score:
                    print(f'  大盘评分: 缓存={market_score} vs 实时={mr.score} → 使用实时值')
                    market_score = mr.score
                # 缓存供后续使用
                try:
                    os.makedirs('signals', exist_ok=True)
                    with open(f'signals/market_score_{today_str}.json', 'w',
                              encoding='utf-8') as f:
                        json.dump({'market_score': market_score,
                                   'state': mr.state,
                                   'date': today_str},
                                  f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception:
            if not score_loaded:
                print(f'  大盘评分: 使用默认值 {market_score}')

        # 对BUY推荐应用仓位限制
        for r in committee_results:
            if r.get('decision') != 'buy':
                continue
            code = r.get('symbol', '') or r.get('code', '')
            # 优先从候选数据获取signal_type，委员会结果可能丢失此字段
            orig_candidate = candidate_lookup.get(code, {})
            signal_type = (r.get('buy_type') or r.get('signal_type')
                           or orig_candidate.get('signal_type', '2buy'))
            sector_tier = orig_candidate.get('sector_tier', 2)

            # 买点限制检查
            allowed, reason = TradingRules.is_buy_allowed(
                signal_type, market_score, sector_tier)
            if not allowed:
                r['decision'] = 'hold'
                r['v2_veto'] = reason
                print(f'  [VETO] {code_to_prefix(code)} '
                      f'{signal_type} → HOLD ({reason})')
                continue

            # 仓位矩阵计算
            max_pos = TradingRules.get_max_position(market_score, sector_tier)
            if max_pos == 0:
                r['decision'] = 'hold'
                r['v2_veto'] = '板块禁止'
                continue

            # 调整仓位不超过规则上限
            orig_pos = r.get('position_pct', 0.3)
            adj_pos = min(orig_pos, max_pos / 100)
            r['position_pct'] = adj_pos
            r['market_score'] = market_score
            r['v2_adjusted'] = True

            # 弱势市特殊规则
            if TradingRules.is_weak_market(market_score):
                stop_pct = TradingRules.get_stop_loss_pct(signal_type, market_score)
                entry = r.get('entry_price', r.get('stop_loss', 0) / (1 + stop_pct))
                if entry > 0:
                    r['stop_loss'] = entry * (1 + stop_pct)
                r['weak_market'] = True
                r['time_stop_days'] = 3
                print(f'  [弱市] {code_to_prefix(code)} '
                      f'仓位:{adj_pos:.0%} 止损:{stop_pct:.0%} 限时3日')
            else:
                print(f'  [OK] {code_to_prefix(code)} '
                      f'Tier{sector_tier} 仓位:{adj_pos:.0%}')

        vetoed = sum(1 for r in committee_results if r.get('v2_veto'))
        adjusted = sum(1 for r in committee_results if r.get('v2_adjusted'))
        print(f'  规则引擎: 调整{adjusted}只 拦截{vetoed}只')
    except Exception as e:
        print(f'  规则引擎异常: {e}')

    # ---- Phase 3: 更新持仓 ----
    print()
    print('[3] 更新持仓...')
    positions_data = load_positions()
    if positions_data.get('positions'):
        positions_data['positions'] = update_stops(
            positions_data['positions'], hs)
        save_positions(positions_data)

        alerts = [p for p in positions_data['positions']
                  if '触发' in p.get('status', '')]
        print(f'  已更新 {len(positions_data["positions"])} 只持仓')
        if alerts:
            for p in alerts:
                print(f'  !! {code_to_prefix(p["code"])} 触发止损: '
                      f'现价:{p.get("current_price", 0):.2f} < 止损:{p.get("current_stop", 0):.2f}')
    else:
        print('  当前无持仓')

    # ---- Phase 4: 生成报告 ----
    print()
    print('[4] 生成报告...')
    report = format_report_text(committee_results, positions_data, scan_count)
    print()
    print(report)

    # 保存报告
    report_file = f'signals/daily_report_{datetime.now().strftime("%Y%m%d")}.txt'
    os.makedirs('signals', exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n报告已保存: {report_file}')

    # ---- Phase 5: 发送通知 ----
    if args.notify:
        print()
        print('[5] 发送通知...')
        send_report(report, committee_results, positions_data)
    else:
        print('\n(使用 --notify 参数发送通知)')

    # ---- 记录日志 ----
    log = load_daily_log()
    log.append({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M'),
        'scan_count': scan_count,
        'candidates': len(candidates),
        'committee_evaluated': len(committee_results),
        'buy_count': sum(1 for r in committee_results if r.get('decision') == 'buy'),
        'positions': len(positions_data.get('positions', [])),
        'notified': args.notify,
    })
    save_daily_log(log)

    print('\n=== 工作流完成 ===')


if __name__ == '__main__':
    main()
