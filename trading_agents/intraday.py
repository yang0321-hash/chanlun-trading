#!/usr/bin/env python3
"""盘中Agent — 实时监控者

09:30 启动监控
每5分钟 周期扫描 (30min缠论买点 + 止损检查)
11:28 午盘总结
13:00 恢复监控
14:40 尾盘分析
15:00 结束
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
          'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

from dotenv import load_dotenv
load_dotenv()

from data.hybrid_source import HybridSource
from trading_agents.position_manager import PositionManager


# ==================== 工具函数 ====================

def code_to_prefix(code: str) -> str:
    code = code.upper()
    # 处理 000001.SZ 格式
    if '.' in code:
        code = code.split('.')[0]
    # 处理纯数字 000001 格式
    if code.startswith(('0', '3')):
        return f'sz{code}'
    if code.startswith('6'):
        return f'sh{code}'
    # 已有前缀
    if code.startswith(('SZ', 'SH')):
        return code.lower()
    return code.lower()


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


# ==================== 盘中Agent ====================

class IntradayAgent:
    """盘中Agent — 实时监控者"""

    SIGNAL_TYPE_MAP = {
        'bi_buy': '笔买', '2buy': '2买', 'quasi2buy': '类2买',
        '3buy': '3买', 'quasi3buy': '类3买', 'zg_breakout': 'ZG突破',
        '2plus3buy': '2+3买',
    }

    SCAN_INTERVAL = 300   # 扫描间隔（秒）
    MIDDAY_HOUR = 11
    MIDDAY_MIN = 28
    LATE_HOUR = 14
    LATE_MIN = 40
    MARKET_CLOSE_HOUR = 15

    def __init__(self, once=False):
        self.hs = HybridSource()
        self.pm = PositionManager()
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.once = once  # 单次扫描模式
        self.watchlist = []
        self.watchlist_names = {}
        self.watchlist_sectors = {}  # code -> sector name
        self.events = []          # 盘中事件
        self.new_signals = []     # 新信号
        self._signal_dedup = {}   # {(code, type): last_push_time} 1h去重
        self.scan_count = 0
        self.midday_done = False
        self.late_done = False

        # 尝试加载 UnifiedExitManager (使用30分钟优化参数)
        try:
            from strategies.unified_exit_manager import UnifiedExitManager
            from strategies.unified_config import UnifiedStrategyConfig
            min30_config = UnifiedStrategyConfig.min30_optimized()
            self.exit_mgr = UnifiedExitManager(min30_config.exit)
            self.min30_config = min30_config
            # 同步持仓到 exit_mgr
            for pos in self.pm.get_all_positions():
                self.exit_mgr.on_buy(
                    pos.code, pos.entry_price, 0,
                    chan_stop=pos.chan_stop,
                    buy_point_type=pos.buy_point_type,
                )
        except Exception:
            self.exit_mgr = None
            self.min30_config = None

        # 尝试加载 v3a 策略
        self.v3a_strategy = None
        if os.getenv('V3A_ENABLED', 'true').lower() == 'true':
            try:
                from strategies.v3a_30min_strategy import V3a30MinStrategy, V3aConfig
                v3a_mode = os.getenv('V3A_MODE', 'trend')
                self.v3a_strategy = V3a30MinStrategy(V3aConfig(mode=v3a_mode), self.hs)
                print(f'  v3a策略已加载 (模式: {v3a_mode})')
            except Exception as e:
                print(f'  v3a策略加载失败: {e}')
                self.v3a_strategy = None

    def run(self):
        """执行盘中监控"""
        print(f'=== 盘中Agent {self.today} ===')
        print()

        # 1. 构建监控列表
        print('[1] 构建监控列表...')
        self._build_watchlist()

        # 2. 单次扫描 or 持续监控
        if self.once:
            print('[2] 单次扫描模式...')
            self.scan_cycle()
            self._generate_report()
        else:
            print('[2] 持续监控模式...')
            self._monitoring_loop()

        # 3. 保存报告
        self.save_report()
        self.push_report()

        # 4. 标记完成
        mark_done('intraday')
        print('\n盘中监控完成')

    def _build_watchlist(self):
        """构建监控列表: 持仓 + 委员会BUY + 最新扫描候选"""
        import glob

        codes = set()

        # 持仓
        for pos in self.pm.get_all_positions():
            c = pos.code
            if c:
                codes.add(c)
                if pos.name:
                    self.watchlist_names[c] = pos.name
                    if '.' in c:
                        pure = c.split('.')[0]
                        self.watchlist_names[f'sz{pure}'] = pos.name
                        self.watchlist_names[f'sh{pure}'] = pos.name

        # 最新委员会BUY/_HOLD
        committee_files = sorted(glob.glob('signals/investment_committee_*.json'),
                                 reverse=True)
        if committee_files:
            try:
                with open(committee_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for d in data.get('decisions', []):
                    if d.get('decision') in ('buy', 'hold'):
                        c = d.get('symbol', '')
                        if c:
                            codes.add(c)
                            # 同时存两种格式的映射
                            name = d.get('name', '')
                            self.watchlist_names[c] = name
                            if '.' in c:
                                pure = c.split('.')[0]
                                self.watchlist_names[f'sz{pure}'] = name
                                self.watchlist_names[f'sh{pure}'] = name
            except Exception:
                pass

        # 最新扫描结果 (取评分前20)
        scan_files = sorted(glob.glob('signals/scan_enhanced_*.json'), reverse=True)
        if scan_files:
            try:
                with open(scan_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                signals = data.get('signals', data.get('results', []))
                signals_sorted = sorted(signals,
                                       key=lambda x: x.get('score', x.get('composite_score', 0)),
                                       reverse=True)
                for s in signals_sorted[:20]:
                    c = s.get('symbol', s.get('code', ''))
                    if c:
                        codes.add(c)
                        name = s.get('name', '')
                        if name:
                            self.watchlist_names[c] = name
                            if '.' in c:
                                pure = c.split('.')[0]
                                self.watchlist_names[f'sz{pure}'] = name
                                self.watchlist_names[f'sh{pure}'] = name
            except Exception:
                pass

        # 观察池 (委员会BUY候选等回踩)
        obs_file = 'signals/observation_pool.json'
        if os.path.exists(obs_file):
            try:
                with open(obs_file, 'r', encoding='utf-8') as f:
                    obs = json.load(f)
                for item in obs.get('stocks', []):
                    if item.get('status') == 'watching':
                        c = item.get('code', '')
                        if c:
                            codes.add(c)
                            self.watchlist_names[c] = item.get('name', '')
                            # Also store entry zone for pullback check
                            if not hasattr(self, '_obs_entries'):
                                self._obs_entries = {}
                            self._obs_entries[c] = {
                                'entry_zone': item.get('entry_zone', ''),
                                'stop_loss': item.get('stop_loss', 0),
                                'score': item.get('score', 0),
                                'name': item.get('name', ''),
                            }
            except Exception:
                pass

        # 额外监控列表 (手动添加的候选股)
        extra_file = 'signals/extra_watchlist.json'
        if os.path.exists(extra_file):
            try:
                with open(extra_file, 'r', encoding='utf-8') as f:
                    extra = json.load(f)
                for item in extra:
                    c = item if isinstance(item, str) else item.get('code', '')
                    if c:
                        codes.add(c)
                        if isinstance(item, dict) and item.get('name'):
                            self.watchlist_names[c] = item['name']
            except Exception:
                pass

        # 热点板块: 加载启动/加速/高潮阶段的板块成分股(每板块最多20只)
        hot_file = 'signals/hot_sectors_latest.json'
        if os.path.exists(hot_file):
            try:
                with open(hot_file, 'r', encoding='utf-8') as f:
                    hot = json.load(f)
                for sector in hot.get('sectors', []):
                    if sector.get('phase', '') in ('启动', '加速', '高潮'):
                        for code in sector.get('stocks', [])[:20]:
                            codes.add(code)
                            sname = sector.get('name', '')
                            self.watchlist_names[code] = sname
                            self.watchlist_sectors[code] = sname
            except Exception:
                pass

        # 统一代码格式 (去掉.SZ/.SH后缀，统一为纯数字或sz/sh前缀)
        normalized = set()
        for c in codes:
            if '.' in c:
                # 000001.SZ → sz000001
                pure, exchange = c.split('.')
                if exchange.upper() == 'SZ':
                    normalized.add(f'sz{pure}')
                elif exchange.upper() == 'SH':
                    normalized.add(f'sh{pure}')
                else:
                    normalized.add(pure)
            else:
                normalized.add(c)
        self.watchlist = list(normalized)
        print(f'  监控列表: {len(self.watchlist)}只')

    def _monitoring_loop(self):
        """持续监控循环"""
        while True:
            now = datetime.now()

            # 检查是否收盘
            if now.hour >= self.MARKET_CLOSE_HOUR:
                print('  收盘，结束监控')
                break

            # 午休 11:30-13:00 跳过扫描
            if now.hour == 12 or (now.hour == 11 and now.minute >= 30):
                if now.hour == 11 and now.minute >= 30 and not self.midday_done:
                    self.midday_summary()
                    self.midday_done = True
                time.sleep(60)
                continue

            # 检查午盘总结时间
            if (now.hour == self.MIDDAY_HOUR and now.minute >= self.MIDDAY_MIN
                    and not self.midday_done):
                self.midday_summary()
                self.midday_done = True

            # 检查尾盘分析时间
            if (now.hour == self.LATE_HOUR and now.minute >= self.LATE_MIN
                    and not self.late_done):
                self.late_analysis()
                self.late_done = True

            # 执行扫描
            self.scan_cycle()

            # 等待下一轮
            print(f'  等待{self.SCAN_INTERVAL}秒...')
            time.sleep(self.SCAN_INTERVAL)

    def scan_cycle(self):
        """一轮扫描: 获取实时报价 + 止损检查 + 买点检测"""
        self.scan_count += 1
        now_str = datetime.now().strftime('%H:%M')
        print(f'\n  --- 扫描 #{self.scan_count} ({now_str}) ---')

        if not self.watchlist:
            print('  监控列表为空，跳过')
            return

        # 实时报价 (Sina)
        self._check_realtime_quotes()

        # 更新持仓最高价
        self._update_position_prices()

        # 止损检查
        self._check_stop_losses()

        # 买点检测 (每3轮一次，减少API压力)
        if self.scan_count % 3 == 1:
            self._detect_buy_signals()

        # 减仓点检测 (每5轮一次)
        if self.scan_count % 5 == 2:
            self._detect_sell_signals()

        # 观察池回踩检测 (每3轮一次)
        if self.scan_count % 3 == 0:
            self._check_observation_pool()

    def _check_realtime_quotes(self):
        """获取实时报价，记录涨跌"""
        try:
            # Sina批量获取 (最多30只/批)
            for i in range(0, len(self.watchlist), 30):
                batch = self.watchlist[i:i+30]
                df = self.hs.get_realtime_quote(batch)
                if df is None or len(df) == 0:
                    continue

                for _, row in df.iterrows():
                    code = str(row.get('code', ''))
                    name = str(row.get('name', ''))
                    pct = float(row.get('pct_chg', 0))
                    price = float(row.get('price', 0))

                    if name:
                        self.watchlist_names[code] = name

                    # 记录异动 (去重: 同code同type只保留最新)
                    if abs(pct) >= 5:
                        etype = 'surge' if pct > 0 else 'drop'
                        # 更新已有事件而非追加
                        found = False
                        for ev in self.events:
                            if ev['code'] == code and ev['type'] == etype:
                                ev['time'] = datetime.now().strftime('%H:%M')
                                ev['pct'] = pct
                                ev['price'] = price
                                found = True
                                break
                        if not found:
                            self.events.append({
                                'time': datetime.now().strftime('%H:%M'),
                                'type': etype,
                                'code': code,
                                'name': name,
                                'pct': pct,
                                'price': price,
                            })

                time.sleep(0.1)
        except Exception as e:
            print(f'  实时行情获取异常: {e}')

    def _update_position_prices(self):
        """更新持仓最高价 — 用events中的实时价格"""
        positions = self.pm.get_all_positions()
        if not positions:
            return
        # 从events收集最新价格
        price_map = {}
        for ev in self.events:
            if ev.get('price', 0) > 0:
                price_map[ev['code']] = ev['price']
        # 对没有event价格的持仓，单独获取
        for pos in positions:
            if pos.code not in price_map:
                try:
                    p = self.hs.get_realtime_price(pos.code)
                    if p and p > 0:
                        price_map[pos.code] = p
                except Exception:
                    pass
        if price_map:
            self.pm.update_prices(price_map)

    def _check_stop_losses(self):
        """检查持仓止损 — 使用 UnifiedExitManager 7层退出"""
        positions = self.pm.get_all_positions()
        if not positions:
            return

        for pos in positions:
            code = pos.code
            try:
                price = self.hs.get_realtime_price(code)
                if not price or price <= 0:
                    continue

                # v3a策略: 额外出场检查 + ZG突破加仓 + T+0清仓
                if self.v3a_strategy and pos.strategy_mode.startswith('v3a'):
                    # v3a出场检查
                    exit_reason = self.v3a_strategy.check_exit(
                        code, pos.entry_price,
                        pos.highest_since_entry,
                        bars_held=self.scan_count,
                    )
                    if exit_reason:
                        self.pm.sell(code)
                        self.events.append({
                            'time': datetime.now().strftime('%H:%M'),
                            'type': 'v3a_exit',
                            'code': code,
                            'name': pos.name,
                            'price': price,
                            'reason': exit_reason,
                        })
                        print(f'  [V3A EXIT] {code} ({pos.name}) {exit_reason}')
                        send_notification(f'v3a卖出 {pos.name}',
                            f'{code} ({pos.name}) 价格:{price:.2f} 原因:{exit_reason}')
                        continue

                    # ZG突破加仓 (仅限initial仓位, 非t0模式)
                    if (pos.entry_type == 'initial'
                        and self.v3a_strategy.config.mode != 't0'):
                        zg_signal = self.v3a_strategy.check_zg_breakout(code, {'price': pos.entry_price})
                        if zg_signal:
                            shares_add = int(self.pm.capital * self.v3a_strategy.config.add_pct / price / 100) * 100
                            if shares_add >= 100:
                                self.pm.buy(code, pos.name, price, shares_add,
                                           stop_price=zg_signal.stop_loss,
                                           buy_point_type='zg_breakout',
                                           chan_stop=zg_signal.chan_stop,
                                           entry_type='zg_breakout')
                                print(f'  [V3A ADD] {code} ({pos.name}) ZG突破加仓{shares_add}股 @ {price:.2f}')
                                send_notification(f'v3a加仓 {pos.name}',
                                    f'{code} ZG突破加仓{shares_add}股 @ {price:.2f}')
                    continue  # v3a持仓已处理, 跳过下面的通用逻辑

                # 通用出场逻辑 (非v3a持仓)
                # 优先用 UnifiedExitManager
                if self.exit_mgr and self.exit_mgr.has_position(code):
                    # 检测30min卖点 (供1卖减仓使用)
                    sell_30min = self._get_30min_sell_point(code)
                    signal = self.exit_mgr.check_exit(
                        symbol=code,
                        price=price,
                        current_qty=pos.shares,
                        bar_index=self.scan_count,
                        sell_point_30min=sell_30min,
                    )
                    if signal:
                        self._handle_exit(code, pos.name, price, signal,
                                         current_qty=pos.shares)
                        continue

                # Fallback: 简单止损检查
                stop = pos.current_stop
                if stop > 0 and price <= stop:
                    self._handle_simple_stop(code, pos.name, price, stop)

            except Exception:
                pass

    def _handle_exit(self, code: str, name: str, price: float, signal,
                     current_qty: int = 0):
        """处理 UnifiedExitManager 的退出信号"""
        # 判断是否部分减仓: 1卖减仓通过exit_type和quantity判断
        is_partial = signal.exit_type == 'sell_1sell_reduce' or signal.quantity < current_qty

        self.events.append({
            'time': datetime.now().strftime('%H:%M'),
            'type': 'exit_signal',
            'code': code,
            'name': name,
            'price': price,
            'action': signal.action,
            'reason': signal.reason,
            'partial': is_partial,
            'msg': f'{code_to_prefix(code)} ({name}) {signal.action}: {signal.reason}',
        })
        print(f'  [EXIT] {code_to_prefix(code)} ({name}) {signal.action}: {signal.reason}'
              f'{" (部分减仓)" if is_partial else ""}')

        # 实际卖出
        if is_partial and signal.quantity < current_qty:
            sell_qty = min(signal.quantity, current_qty)
            self.pm.partial_sell(code, quantity=sell_qty)
        else:
            self.pm.sell(code)
            if self.exit_mgr:
                self.exit_mgr.on_sell(code)

        action_cn = '30min1卖减仓' if signal.exit_type == 'sell_1sell_reduce' else (
            '部分减仓' if is_partial else (
            '紧急清仓' if signal.action == 'force_exit' else '卖出信号'))
        send_notification(
            f'{action_cn} {name}',
            f'{code_to_prefix(code)} ({name})\n'
            f'价格:{price:.2f}\n'
            f'原因:{signal.reason}'
        )

    def _handle_simple_stop(self, code: str, name: str, price: float, stop: float):
        """简单止损处理"""
        self.events.append({
            'time': datetime.now().strftime('%H:%M'),
            'type': 'stop_triggered',
            'code': code,
            'name': name,
            'price': price,
            'stop': stop,
            'msg': f'{code_to_prefix(code)} ({name}) '
                  f'触发止损! 现价:{price:.2f} 止损:{stop:.2f}',
        })
        print(f'  [STOP] {code_to_prefix(code)} 触发止损 {price:.2f} <= {stop:.2f}')

        # 实际卖出
        self.pm.sell(code)

        send_notification('止损告警',
                         f'{code_to_prefix(code)} ({name}) '
                         f'触发止损! 现价:{price:.2f} 止损:{stop:.2f}')

    def _detect_buy_signals(self):
        """检测30min买点（从监控列表中选10只轮检）"""
        # 每次检测不同的10只，轮流覆盖
        start = (self.scan_count // 3 * 10) % max(len(self.watchlist), 1)
        batch = self.watchlist[start:start+10]

        for code in batch:
            try:
                signal = self._analyze_30min(code)
                if signal:
                    # 去重: 同code+type 1小时内不重复推送
                    dedup_key = (code, signal['type'])
                    now = datetime.now()
                    last_time = self._signal_dedup.get(dedup_key)
                    if last_time and (now - last_time).total_seconds() < 3600:
                        continue

                    # 用实时价格替换K线收盘价
                    try:
                        rt_price = self.hs.get_realtime_price(code)
                        if rt_price and rt_price > 0:
                            signal['kline_price'] = signal['price']
                            signal['price'] = rt_price
                    except Exception:
                        pass

                    self._signal_dedup[dedup_key] = now
                    self.new_signals.append(signal)
                    name = self.watchlist_names.get(code, '')
                    print(f'  [SIGNAL] {code_to_prefix(code)} ({name}) '
                          f'{signal["type"]} @ {signal["price"]:.2f} '
                          f'置信度:{signal["confidence"]:.2f}')

                    # 推送新信号
                    sector_tag = self.watchlist_sectors.get(code, '')
                    tag_str = f' [{sector_tag}]' if sector_tag else ''
                    send_notification(
                        f'盘中信号{tag_str} {code_to_prefix(code)} ({name})',
                        f'{signal["type"]} 价格:{signal["price"]:.2f} '
                        f'止损:{signal["stop"]:.2f} '
                        f'置信度:{signal["confidence"]:.2f}\n'
                        f'{signal.get("reason", "")}'
                    )
            except Exception:
                pass

            time.sleep(0.2)

    def _analyze_30min(self, code: str) -> Optional[dict]:
        """分析单只股票的30min买点"""
        # v3a策略优先
        if self.v3a_strategy:
            return self._analyze_30min_v3a(code)
        return self._analyze_30min_original(code)

    def _analyze_30min_v3a(self, code: str) -> Optional[dict]:
        """v3a策略: 日线过滤 + 30分钟笔买 + MACD确认 + 动态置信度"""
        try:
            signal = self.v3a_strategy.scan_entry(code)
            if not signal:
                return None
            # 过滤低置信度信号
            if signal.confidence < 0.55:
                return None
            return {
                'code': code,
                'type': self.SIGNAL_TYPE_MAP.get(signal.signal_type, signal.signal_type),
                'price': signal.price,
                'stop': signal.stop_loss,
                'confidence': signal.confidence,
                'reason': signal.reason,
                'time': datetime.now().strftime('%H:%M'),
                'pivot_zg': signal.pivot_zg,
                'pivot_zd': signal.pivot_zd,
                'strategy_mode': f'v3a_{self.v3a_strategy.config.mode}',
                'chan_stop': signal.chan_stop,
            }
        except Exception:
            return None

    def _analyze_30min_original(self, code: str) -> Optional[dict]:
        """30min分析逻辑 — 使用优化参数 (4%启动/3%回撤跟踪止盈)"""
        try:
            from core.kline import KLine
            from core.fractal import FractalDetector
            from core.stroke import StrokeGenerator
            from core.segment import SegmentGenerator
            from core.pivot import PivotDetector
            from core.buy_sell_points import BuySellPointDetector
            from core.trend_track import TrendTrackDetector
            from indicator.macd import MACD

            # 获取30min数据
            df = self.hs.get_kline(code, period='30min')
            if df is None or len(df) < 120:
                return None

            kline = KLine.from_dataframe(df, strict_mode=False)
            fractal_detector = FractalDetector(kline, confirm_required=False)
            fractals = fractal_detector.get_fractals()

            if len(fractals) < 6:
                return None

            stroke_gen = StrokeGenerator(kline, fractals, min_bars=3)
            strokes = stroke_gen.get_strokes()

            if len(strokes) < 4:
                return None

            segments = SegmentGenerator(kline, strokes).get_segments()
            pivot_detector = PivotDetector(kline, strokes)
            pivots = pivot_detector.get_pivots()

            if not pivots:
                return None

            close_s = pd.Series([k.close for k in kline])
            macd = MACD(close_s)

            td = TrendTrackDetector(strokes, pivots)
            td.detect()

            det = BuySellPointDetector(fractals, strokes, segments, pivots,
                                       macd, trend_tracks=td._tracks)
            buys, _ = det.detect_all()

            if not buys:
                return None

            current_price = close_s.iloc[-1]
            klen = len(kline)

            # 筛选最近的买点（最近30根K线内）
            recent_buys = [
                b for b in buys
                if b.index >= klen - 30
                and current_price >= b.price * 0.97
                and b.stop_loss < current_price
                and b.confidence >= 0.5
            ]

            if not recent_buys:
                return None

            best = max(recent_buys, key=lambda b: b.confidence)

            # === 2买/3买强度分类 ===
            buy_strength = ''
            strength_label = ''
            if best.point_type in ('2buy', 'quasi2buy') and pivots:
                last_pivot = pivots[-1]
                zg = last_pivot.zg if hasattr(last_pivot, 'zg') else 0
                zd = last_pivot.zd if hasattr(last_pivot, 'zd') else 0
                if zg > 0 and zd > 0:
                    if best.price >= zg:
                        buy_strength = 'strong'      # 2买3买重叠
                        strength_label = '强2买'
                    elif best.price >= zd:
                        buy_strength = 'standard'    # 类2买(中枢内)
                        strength_label = '类2买'
                    else:
                        buy_strength = 'normal'      # 中枢下2买
                        strength_label = '2买'
            elif best.point_type in ('3buy', 'quasi3buy') and pivots:
                last_pivot = pivots[-1]
                zg = last_pivot.zg if hasattr(last_pivot, 'zg') else 0
                zd = last_pivot.zd if hasattr(last_pivot, 'zd') else 0
                gg = last_pivot.high if hasattr(last_pivot, 'high') else zg
                if zg > 0:
                    if best.price > gg:
                        buy_strength = 'strong'      # 强3买
                        strength_label = '强3买'
                    elif best.price > zg:
                        buy_strength = 'standard'    # 标准3买
                        strength_label = '标准3买'
                    else:
                        buy_strength = 'normal'      # 弱3买
                        strength_label = '3买'

            # 使用优化止盈参数 (4%启动, 3%回撤)
            from agents.scoring import MIN30_TRAIL_START, MIN30_TRAIL_DIST
            trail_start = MIN30_TRAIL_START
            trail_dist = MIN30_TRAIL_DIST

            type_map = {
                '1buy': '1买', '2buy': strength_label or '2买', '3buy': strength_label or '3买',
                'quasi2buy': strength_label or '类2买', 'quasi3buy': strength_label or '类3买',
            }

            signal_type = type_map.get(best.point_type, best.point_type)
            reason_parts = []
            if best.reason:
                reason_parts.append(best.reason[:40])
            if buy_strength:
                reason_parts.append(f'强度:{buy_strength}')
            reason_parts.append(f'止盈:{trail_start*100:.0f}%/{trail_dist*100:.0f}%')

            return {
                'code': code,
                'type': signal_type,
                'price': float(current_price),
                'stop': float(max(best.stop_loss, current_price * 0.95)),
                'confidence': float(best.confidence),
                'reason': ' | '.join(reason_parts),
                'time': datetime.now().strftime('%H:%M'),
                'buy_strength': buy_strength,
                'trail_start': trail_start,
                'trail_dist': trail_dist,
            }

        except Exception:
            return None

    # ---------- 午盘总结 ----------

    def midday_summary(self):
        """11:28 午盘总结"""
        print('\n  === 午盘总结 ===')
        lines = [f'=== 午盘总结 {self.today} ===', '']

        # 大盘
        self._append_market_snapshot(lines)

        # 持仓动态
        self._append_position_snapshot(lines)

        # 新信号
        if self.new_signals:
            lines.append(f'\n【新信号】{len(self.new_signals)}个')
            for s in self.new_signals[-5:]:
                name = self.watchlist_names.get(s['code'], '')
                lines.append(f'  {code_to_prefix(s["code"])} ({name}) '
                           f'{s["type"]} 置信度:{s["confidence"]:.2f}')

        # 事件
        surge_events = [e for e in self.events if e.get('type') == 'surge']
        drop_events = [e for e in self.events if e.get('type') == 'drop']
        stop_events = [e for e in self.events if e.get('type') == 'stop_triggered']

        if stop_events:
            lines.append(f'\n【止损触发】')
            for e in stop_events:
                lines.append(f'  {e["msg"]}')

        if surge_events:
            lines.append(f'\n【异动上涨】{len(surge_events)}只')
            for e in surge_events[:5]:
                lines.append(f'  {code_to_prefix(e["code"])} ({e["name"]}) '
                           f'{e["pct"]:+.2f}%')

        if drop_events:
            lines.append(f'\n【异动下跌】{len(drop_events)}只')
            for e in drop_events[:5]:
                lines.append(f'  {code_to_prefix(e["code"])} ({e["name"]}) '
                           f'{e["pct"]:+.2f}%')

        lines.append(f'\n【下午关注】')
        positions = self.pm.get_all_positions()
        if positions:
            for pos in positions:
                if pos.current_stop > 0:
                    lines.append(f'  - {code_to_prefix(pos.code)} ({pos.name}) '
                               f'止损{pos.current_stop:.2f}')
        else:
            lines.append('  - 空仓，关注买入机会')

        # 推送
        report = '\n'.join(lines)
        send_notification(f'午盘总结 {datetime.now().strftime("%m-%d")}', report)

    # ---------- 尾盘分析 ----------

    def late_analysis(self):
        """14:40 尾盘分析"""
        print('\n  === 尾盘分析 ===')
        lines = [f'=== 尾盘分析 {self.today} ===', '']

        # 大盘
        self._append_market_snapshot(lines)

        # 今日表现
        up = sum(1 for e in self.events if e.get('type') == 'surge')
        down = sum(1 for e in self.events if e.get('type') == 'drop')

        lines.append(f'\n【今日表现】')
        lines.append(f'  异动上涨: {up}只 | 异动下跌: {down}只')
        lines.append(f'  新信号: {len(self.new_signals)}个')
        lines.append(f'  扫描轮次: {self.scan_count}')

        # 涨停板（异动 >= 9.8%）
        limit_ups = [e for e in self.events
                     if e.get('pct', 0) >= 9.8]
        if limit_ups:
            lines.append(f'\n【涨停板】')
            for e in limit_ups:
                lines.append(f'  {code_to_prefix(e["code"])} ({e["name"]}) '
                           f'{e["pct"]:+.2f}%')

        # 尾盘异动（最近30分钟的事件）
        cutoff = datetime.now() - timedelta(minutes=30)
        recent_events = [e for e in self.events
                        if e.get('type') in ('surge', 'drop')]
        if recent_events:
            lines.append(f'\n【尾盘异动】')
            for e in recent_events[-5:]:
                lines.append(f'  {code_to_prefix(e["code"])} ({e["name"]}) '
                           f'{e["pct"]:+.2f}%')

        # 明日预判
        lines.append(f'\n【明日预判】')
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
                        lines.append(f'  大盘: 多头排列, 偏多')
                    elif last < ma5 < ma20:
                        lines.append(f'  大盘: 空头排列, 偏空')
                    else:
                        lines.append(f'  大盘: 震荡, 观望')

                    # 关注个股
                    if self.new_signals:
                        lines.append(f'  重点关注:')
                        for s in self.new_signals[-3:]:
                            name = self.watchlist_names.get(s['code'], '')
                            lines.append(f'    {code_to_prefix(s["code"])} ({name}) '
                                       f'{s["type"]}')
                    else:
                        lines.append(f'  今日无新信号')
        except Exception:
            lines.append('  大盘数据不可用')

        # 推送
        report = '\n'.join(lines)
        send_notification(f'尾盘分析 {datetime.now().strftime("%m-%d")}', report)

    def _check_observation_pool(self):
        """检查观察池股票是否回踩到入场区并出现缠论买点确认"""
        if not hasattr(self, '_obs_entries') or not self._obs_entries:
            return

        for code, info in self._obs_entries.items():
            if info.get('confirmed'):
                continue

            # Get current price from events
            price = 0
            for ev in self.events:
                if ev.get('code') == code or ev.get('code') == code.upper():
                    price = ev.get('price', 0)
                    break
            if price <= 0:
                continue

            # Parse entry zone
            entry_zone = info.get('entry_zone', '')
            try:
                zone_part = entry_zone.split('(')[0].strip()
                lo, hi = zone_part.split('-')
                lo, hi = float(lo), float(hi)
            except Exception:
                lo = price * 0.95
                hi = price * 1.0

            if not (lo <= price <= hi or price < lo):
                continue

            # Price in entry zone — run ChanLun confirmation
            try:
                from core.kline import KLine
                from core.fractal import FractalDetector
                from core.stroke import StrokeGenerator
                from core.pivot import PivotDetector
                from indicator.macd import MACD
                from core.buy_sell_points import BuySellPointDetector

                df = self.hs.get_kline(code, period='daily')
                if df is None or len(df) < 60:
                    continue

                close_s = pd.Series(df['close'].values)
                macd_obj = MACD(close_s)
                kline = KLine.from_dataframe(df, strict_mode=False)
                fractals = FractalDetector(kline, confirm_required=False).get_fractals()
                if len(fractals) < 4:
                    continue
                strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
                if len(strokes) < 3:
                    continue
                pivots = PivotDetector(kline, strokes).get_pivots()
                det = BuySellPointDetector(fractals, strokes, [], pivots, macd=macd_obj)
                buys, _ = det.detect_all()

                n = len(df)
                recent_buys = [b for b in buys if b.index >= n - 5]
                if not recent_buys:
                    continue

                best = max(recent_buys, key=lambda b: b.confidence)
                types = list(set(b.point_type for b in recent_buys))

                # Mark confirmed
                info['confirmed'] = True
                msg = (f"★ 观察池回踩确认! {info.get('name',code)} "
                       f"价格={price:.2f} 入场区=[{lo:.1f}-{hi:.1f}] "
                       f"买点={','.join(types)} conf={best.confidence:.2f} "
                       f"止损={info.get('stop_loss',0):.2f}")

                print(f'  {msg}')
                self.events.append({
                    'time': datetime.now().strftime('%H:%M'),
                    'type': 'pullback_confirm',
                    'code': code,
                    'name': info.get('name', ''),
                    'price': price,
                    'signal': ','.join(types),
                    'confidence': best.confidence,
                })

                # Save alert
                alert_file = 'signals/observation_alerts.json'
                all_alerts = []
                if os.path.exists(alert_file):
                    try:
                        with open(alert_file, 'r', encoding='utf-8') as f:
                            all_alerts = json.load(f)
                    except Exception:
                        pass
                all_alerts.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'code': code,
                    'name': info.get('name', ''),
                    'price': price,
                    'stop_loss': info.get('stop_loss', 0),
                    'signal_type': ','.join(types),
                    'confidence': best.confidence,
                    'message': msg,
                })
                with open(alert_file, 'w', encoding='utf-8') as f:
                    json.dump(all_alerts, f, ensure_ascii=False, indent=2)

                # Update pool status
                pool_file = 'signals/observation_pool.json'
                if os.path.exists(pool_file):
                    with open(pool_file, 'r', encoding='utf-8') as f:
                        pool = json.load(f)
                    for s in pool.get('stocks', []):
                        if s.get('code') == code:
                            s['status'] = 'confirmed'
                            s['confirm_price'] = price
                            s['confirm_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                    with open(pool_file, 'w', encoding='utf-8') as f:
                        json.dump(pool, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f'  观察池检查 {code} 异常: {e}')

    def _detect_sell_signals(self):
        """检测30min减仓点 — 持仓股 + 今日出买点的个股"""
        position_codes = {pos.code for pos in self.pm.get_all_positions()}
        # 也监控今日出买点的股票 (可能快速反转需要减仓)
        signal_codes = {s['code'] for s in self.new_signals if 'code' in s}
        check_codes = position_codes | signal_codes
        for code in check_codes:
            try:
                sell_info = self._check_30min_sell(code)
                if sell_info:
                    name = self.watchlist_names.get(code, '')
                    dedup_key = (code, 'sell')
                    now = datetime.now()
                    last_time = self._signal_dedup.get(dedup_key)
                    if last_time and (now - last_time).total_seconds() < 3600:
                        continue

                    self._signal_dedup[dedup_key] = now
                    print(f'  [SELL] {code_to_prefix(code)} ({name}) '
                          f'{sell_info["type"]} @ {sell_info["price"]:.2f}')

                    send_notification(
                        f'减仓提醒 {code_to_prefix(code)} ({name})',
                        f'{sell_info["type"]} 价格:{sell_info["price"]:.2f}\n'
                        f'{sell_info["reason"]}'
                    )
            except Exception:
                pass
            time.sleep(0.2)

    def _get_30min_sell_point(self, code: str) -> Optional[object]:
        """获取30min最新卖点对象 (供UnifiedExitManager 1卖减仓使用)"""
        try:
            from core.kline import KLine
            from core.fractal import FractalDetector
            from core.stroke import StrokeGenerator
            from core.pivot import PivotDetector
            from core.buy_sell_points import BuySellPointDetector
            from indicator.macd import MACD

            df = self.hs.get_kline(code, period='30min')
            if df is None or len(df) < 80:
                return None

            close_s = pd.Series(df['close'].values)
            kline = KLine.from_dataframe(df, strict_mode=False)
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 4:
                return None
            strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
            if len(strokes) < 3:
                return None
            pivots = PivotDetector(kline, strokes).get_pivots()
            if not pivots:
                return None
            macd = MACD(close_s)
            det = BuySellPointDetector(fractals, strokes, [], pivots, macd)
            _, sells = det.detect_all()

            if not sells:
                return None

            klen = len(close_s)
            # 最近5根K线内的1卖
            recent_1sells = [
                s for s in sells
                if s.point_type in ('1sell',)
                and s.index >= klen - 5
            ]
            if recent_1sells:
                return max(recent_1sells, key=lambda s: s.confidence)
            return None
        except Exception:
            return None

    def _check_30min_sell(self, code: str) -> Optional[dict]:
        """检测单只股票30min卖点"""
        try:
            from core.kline import KLine
            from core.fractal import FractalDetector
            from core.stroke import StrokeGenerator
            from core.segment import SegmentGenerator
            from core.pivot import PivotDetector
            from core.buy_sell_points import BuySellPointDetector
            from indicator.macd import MACD

            df = self.hs.get_kline(code, period='30min')
            if df is None or len(df) < 120:
                return None

            kline = KLine.from_dataframe(df, strict_mode=False)
            close_s = pd.Series([k.close for k in kline])
            fractals = FractalDetector(kline, confirm_required=False).get_fractals()
            if len(fractals) < 6:
                return None
            strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
            if len(strokes) < 4:
                return None
            segments = SegmentGenerator(kline, strokes).get_segments()
            pivots = PivotDetector(kline, strokes).get_pivots()
            if not pivots:
                return None
            macd = MACD(close_s)
            det = BuySellPointDetector(fractals, strokes, segments, pivots, macd)
            _, sells = det.detect_all()

            if not sells:
                return None

            current_price = close_s.iloc[-1]
            klen = len(kline)

            # 最近10根K线内的高置信度卖点
            recent_sells = [
                s for s in sells
                if s.index >= klen - 10
                and s.confidence >= 0.6
            ]
            if not recent_sells:
                return None

            best = max(recent_sells, key=lambda s: s.confidence)
            type_cn = {
                '1sell': '1卖(趋势终点)', '2sell': '2卖(反弹高点)',
                '3sell': '3卖(跌破中枢)', 'quasi2sell': '类2卖',
            }
            return {
                'type': type_cn.get(best.point_type, best.point_type),
                'price': current_price,
                'confidence': best.confidence,
                'reason': f'{best.point_type} 置信度:{best.confidence:.2f} | {best.reason[:60]}',
            }
        except Exception:
            return None

    # ---------- 辅助方法 ----------

    def _append_market_snapshot(self, lines: list):
        """大盘快照"""
        lines.append('【大盘】')
        try:
            import requests, re
            session = requests.Session()
            session.trust_env = False

            for label, symbol in [('上证', 'sh000001'), ('深证', 'sz399001')]:
                url = (f'https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/'
                       f'CN_MarketDataService.getKLineData?symbol={symbol}'
                       f'&scale=240&ma=no&datalen=5')
                resp = session.get(url, timeout=10)
                match = re.search(r'callback\((.*)\)', resp.text)
                if match:
                    klines = json.loads(match.group(1))
                    if len(klines) >= 2:
                        last = float(klines[-1]['close'])
                        prev = float(klines[-2]['close'])
                        chg = (last - prev) / prev * 100
                        lines.append(f'  {label} {last:,.0f} ({chg:+.2f}%)')
        except Exception:
            lines.append('  数据不可用')

    def _append_position_snapshot(self, lines: list):
        """持仓快照"""
        positions = self.pm.get_all_positions()
        if not positions:
            return

        lines.append(f'\n【持仓动态】')
        for pos in positions:
            try:
                price = self.hs.get_realtime_price(pos.code)
                if price and price > 0:
                    pnl = (price - pos.entry_price) / pos.entry_price * 100
                    stop_dist = (price - pos.current_stop) / price * 100 if pos.current_stop > 0 else 999
                    stop_mark = ' ← 接近止损!' if stop_dist < 3 else ''
                    lines.append(f'  {code_to_prefix(pos.code)} ({pos.name}) '
                               f'{pnl:+.2f}%{stop_mark}')
                else:
                    lines.append(f'  {code_to_prefix(pos.code)} ({pos.name}) 数据不可用')
            except Exception:
                lines.append(f'  {code_to_prefix(pos.code)} ({pos.name}) 获取失败')

    def _generate_report(self):
        """单次扫描模式生成报告"""
        lines = [f'=== 盘中扫描 {self.today} ===', '']

        lines.append(f'监控列表: {len(self.watchlist)}只 | 扫描轮次: {self.scan_count}')

        if self.new_signals:
            lines.append(f'\n【检测到信号】{len(self.new_signals)}个')
            for s in self.new_signals:
                name = self.watchlist_names.get(s['code'], '')
                lines.append(f'  {code_to_prefix(s["code"])} ({name}) '
                           f'{s["type"]} @ {s["price"]:.2f} '
                           f'止损:{s["stop"]:.2f} 置信度:{s["confidence"]:.2f}')

        if self.events:
            stop_events = [e for e in self.events if e.get('type') == 'stop_triggered']
            if stop_events:
                lines.append(f'\n【止损触发】')
                for e in stop_events:
                    lines.append(f'  {e["msg"]}')

        self.report_text = '\n'.join(lines)

    def save_report(self):
        """保存报告"""
        os.makedirs('signals', exist_ok=True)

        # 汇总所有事件和信号
        lines = [f'=== 盘中报告 {self.today} ===', '']
        lines.append(f'扫描轮次: {self.scan_count}')
        lines.append(f'监控股票: {len(self.watchlist)}只')
        lines.append(f'盘中事件: {len(self.events)}个')
        lines.append(f'新信号: {len(self.new_signals)}个')

        if self.new_signals:
            lines.append(f'\n【信号列表】')
            for s in self.new_signals:
                name = self.watchlist_names.get(s['code'], '')
                lines.append(f'  {s["time"]} {code_to_prefix(s["code"])} ({name}) '
                           f'{s["type"]} @ {s["price"]:.2f} 置信度:{s["confidence"]:.2f}')

        if self.events:
            lines.append(f'\n【事件列表】')
            for e in self.events:
                if e.get('type') == 'stop_triggered':
                    lines.append(f'  {e["time"]} [止损] {e["msg"]}')
                elif e.get('type') == 'surge':
                    lines.append(f'  {e["time"]} [涨] {code_to_prefix(e["code"])} '
                               f'({e["name"]}) {e["pct"]:+.2f}%')
                elif e.get('type') == 'drop':
                    lines.append(f'  {e["time"]} [跌] {code_to_prefix(e["code"])} '
                               f'({e["name"]}) {e["pct"]:+.2f}%')
                elif e.get('type') == 'pullback_confirm':
                    lines.append(f'  {e["time"]} [回踩确认] {code_to_prefix(e["code"])} '
                               f'({e["name"]}) 价格={e["price"]:.2f} '
                               f'{e["signal"]} conf={e["confidence"]:.2f}')

        report = '\n'.join(lines)
        filename = f'signals/intraday_{self.today.replace("-", "")}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'  报告已保存: {filename}')

    def push_report(self):
        """推送报告"""
        title = f'缠论盘中 {datetime.now().strftime("%m-%d")}'
        report = f'扫描{self.scan_count}轮 | 监控{len(self.watchlist)}只 | ' \
                 f'信号{len(self.new_signals)}个 | 事件{len(self.events)}个'
        send_notification(title, report)


# ==================== 入口 ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='盘中Agent')
    parser.add_argument('--force', action='store_true', help='强制运行')
    parser.add_argument('--once', action='store_true', help='单次扫描模式')
    args = parser.parse_args()

    if not args.force and check_today_done('intraday'):
        print('今日盘中监控已完成，跳过（用 --force 强制）')
    else:
        agent = IntradayAgent(once=args.once)
        agent.run()
