#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论30分钟2买监控 - 关注列表（带微信提醒）

监控47只关注个股，当出现30分钟级别2买时提醒
"""
import os
# 绕过Clash系统代理：monkey-patch urllib让所有HTTP请求直连
import urllib.request
urllib.request.getproxies = lambda: {}
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'utils'))

import akshare as ak

from core.kline import KLine
from core.fractal import FractalDetector
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector
from indicator.macd import MACD

# 导入通知模块
try:
    from notification import NotificationManager, load_notification_config, format_2buy_alert, format_summary_alert, format_2buy_html
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False

# 导入新闻工具
try:
    sys.path.insert(0, str(project_root / 'skills' / 'alphaear-news'))
    from scripts.news_tools import NewsNowTools
    from scripts.database_manager import DatabaseManager
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False


# 关注列表
WATCHLIST_CORE = [
    'SZ000938', 'SZ300166', 'SH603290', 'SZ300661', 'SZ300666',
    'SZ300782', 'SH600110', 'SH603026', 'SH603659', 'SZ002759',
    'SZ300014', 'SZ300207', 'SZ300390', 'SZ301150', 'SZ301327',
    'SZ301358', 'SZ002176', 'SH600167', 'SH600483', 'SH600744',
    'SZ000537', 'SZ000690', 'SZ000899', 'SZ002218', 'SZ003816',
    'SZ301179', 'SZ300054', 'SH603806', 'SZ300827', 'SZ002407',
    'SZ002422', 'SH600316', 'SZ002240', 'SZ002460', 'SZ002782',
    'SZ002773', 'SZ002263', 'SZ002361', 'SZ301196', 'SZ002467',
    'SZ300383', 'SH600487', 'SZ002281', 'SZ002468', 'SZ002475',
    'SZ300866', 'SZ002149',
]

# 股票名称映射（核心关注）
STOCK_NAMES = {
    'SZ000938': '紫光股份', 'SZ300166': '东方国信', 'SH603290': '斯达半导',
    'SZ300661': '圣邦股份', 'SZ300666': '江特电机', 'SZ300782': '卓胜微',
    'SH600110': '诺德股份', 'SH603026': '石大胜华', 'SH603659': '新城控股',
    'SZ002759': '天际股份', 'SZ300014': '亿纬锂能', 'SZ300207': '欣旺达',
    'SZ300390': '天华超净', 'SZ301150': '中一科技', 'SZ301327': '华宝新能',
    'SZ301358': '百胜智能', 'SZ002176': '江特电机', 'SH600167': '联美控股',
    'SH600483': '福能股份', 'SH600744': '国电南瑞', 'SZ000537': '广宇发展',
    'SZ000690': '宝新能源', 'SZ000899': '赣能股份', 'SZ002218': '日出东方',
    'SZ003816': '中国广核', 'SZ301179': '泽宇智能', 'SZ300054': '火炬电子',
    'SH603806': '福斯特', 'SZ300827': '上能电气', 'SZ002407': '多氟多',
    'SZ002422': '科伦药业', 'SH600316': '洪都航空', 'SZ002240': '盛新锂能',
    'SZ002460': '赣锋锂业', 'SZ002782': '可立克', 'SZ002773': '康欣新材',
    'SZ002263': '大东南', 'SZ002361': '神剑股份', 'SZ301196': '唯科科技',
    'SZ002467': '二六三', 'SZ300383': '光环新网', 'SH600487': '亨通光电',
    'SZ002281': '光迅科技', 'SZ002468': '申通快递', 'SZ002475': '立讯精密',
    'SZ300866': '安克创新', 'SZ002149': '西部材料',
}


def load_csi1000_stocks():
    """从AKShare获取中证1000成分股列表"""
    try:
        df = ak.index_stock_cons_csindex(symbol='000852')
        code_col = df.columns[4]   # 成分券代码
        name_col = df.columns[5]   # 成分券名称

        stocks = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if 'ST' in name or '退' in name:
                continue
            prefix = 'SH' if code.startswith('6') else 'SZ'
            symbol = f'{prefix}{code}'
            stocks[symbol] = name
        return stocks
    except Exception as e:
        print(f'[WARN] 获取中证1000列表失败: {e}')
        return {}


def build_watchlist(scope='core'):
    """
    构建扫描列表
    scope: 'core' = 47只关注, 'csi1000' = 中证1000, 'all' = 关注+中证1000
    """
    if scope == 'core':
        return WATCHLIST_CORE, dict(STOCK_NAMES)

    csi1000 = load_csi1000_stocks()
    names = dict(STOCK_NAMES)
    names.update(csi1000)

    if scope == 'csi1000':
        # 中证1000 + 核心关注（去重）
        all_symbols = list(dict.fromkeys(WATCHLIST_CORE + list(csi1000.keys())))
        return all_symbols, names
    else:  # 'all'
        all_symbols = list(dict.fromkeys(WATCHLIST_CORE + list(csi1000.keys())))
        return all_symbols, names


@dataclass
class BuySignal:
    """买入信号"""
    symbol: str
    name: str
    signal_type: str  # '2买', '3买', '2+3买', 'ZG突破'
    timeframe: str  # '30min'
    price: float
    confidence: float
    reason: str
    stop_loss: float
    target: float
    timestamp: datetime
    zg: float = 0          # 中枢ZG（上沿）
    pivot_zd: float = 0    # 中枢ZD（下沿）
    signal_score: str = '' # 信号评级: A(2+3买), B(3买), C(2买)


class ChanLun30MinMonitor:
    """缠论30分钟监控器 — 日线定方向 + 30min找买点"""

    def __init__(self, enable_notification: bool = True):
        self.last_signals = {}  # 记录上次信号，避免重复
        self._daily_trend_cache = {}  # symbol -> (date_str, bool)
        self._daily_df_cache = {}  # symbol -> df
        self.notifier = None

        if enable_notification and NOTIFICATION_AVAILABLE:
            self.notifier = load_notification_config()

            # 检查启用的通知方式
            enabled = []
            if self.notifier.wechat:
                enabled.append("企业微信")
            if self.notifier.dingtalk:
                enabled.append("钉钉")
            if self.notifier.feishu:
                enabled.append("飞书")
            if self.notifier.email:
                enabled.append("邮件")

            if enabled:
                print(f"[INFO] 通知已启用: {', '.join(enabled)}")
            else:
                print("[WARN] 未找到任何通知配置")
                print("[HINT] 参考 config/README_NOTIFICATION.md 配置通知")

        # 新闻工具
        self.news_tool = None
        if NEWS_AVAILABLE:
            try:
                db = DatabaseManager('data/signal_flux.db')
                self.news_tool = NewsNowTools(db)
            except Exception:
                pass

    def _fetch_stock_news(self, name: str, code: str) -> str:
        """获取个股相关新闻（用于信号通知叠加）"""
        if not self.news_tool or not name:
            return ''

        try:
            # 从财联社和雪球抓取金融新闻标题
            related = []
            for src in ['cls', 'xueqiu']:
                items = self.news_tool.fetch_hot_news(src, count=15)
                for item in items:
                    title = item.get('title', '')
                    # 匹配：标题含股票名称或相关关键词
                    if name[:2] in title or name in title:
                        related.append(title)
                time.sleep(0.2)

            if related:
                return '\n     '.join(related[:3])  # 最多3条
        except Exception:
            pass

        return ''

    def get_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取日线数据（mootdx，带缓存）"""
        if symbol in self._daily_df_cache:
            return self._daily_df_cache[symbol]

        code = symbol.replace('SZ', '').replace('SH', '')
        market_tag = symbol[:2].lower()
        market_id = 0 if market_tag == 'sz' else 1

        try:
            from mootdx.quotes import Quotes
            api = Quotes.factory(market=market_tag).client
            data = api.get_security_bars(4, market_id, code, 0, 800)
            if not data:
                return None
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={'datetime': 'date'})
            df = df[['date', 'open', 'close', 'high', 'low', 'vol', 'amount']].rename(columns={'vol': 'volume'})
            df = df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
            if len(df) >= 100:
                self._daily_df_cache[symbol] = df
                return df
        except Exception:
            pass
        return None

    def check_daily_trend(self, symbol: str) -> bool:
        """日线趋势过滤：非强下跌才允许30min入场（带缓存）"""
        today = datetime.now().strftime('%Y-%m-%d')
        cached = self._daily_trend_cache.get(symbol)
        if cached and cached[0] == today:
            return cached[1]

        df_day = self.get_daily_data(symbol)
        if df_day is None or len(df_day) < 100:
            self._daily_trend_cache[symbol] = (today, False)
            return False

        try:
            r = self._chanlun_full(df_day.tail(300).copy())
            if r is None:
                self._daily_trend_cache[symbol] = (today, False)
                return False
            ok = r['trend'].value != 'strong_down'
            self._daily_trend_cache[symbol] = (today, ok)
            return ok
        except Exception:
            self._daily_trend_cache[symbol] = (today, False)
            return False

    @staticmethod
    def _chanlun_full(df):
        """完整缠论分析"""
        from core.segment import SegmentGenerator
        from core.buy_sell_points import BuySellPointDetector
        from core.trend_track import TrendTrackDetector

        kline = KLine.from_dataframe(df, strict_mode=True)
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
        if len(strokes) < 3:
            return None
        segments = SegmentGenerator(kline, strokes).get_segments()
        pivots = PivotDetector(kline, strokes).get_pivots()
        close_s = pd.Series([k.close for k in kline])
        macd = MACD(close_s)
        td = TrendTrackDetector(strokes, pivots)
        td.detect()
        buys, sells = [], []
        if pivots:
            det = BuySellPointDetector(fractals, strokes, segments, pivots, macd, trend_tracks=td._tracks)
            buys, sells = det.detect_all()
        return {
            'klen': len(kline), 'kline': kline, 'fractals': fractals,
            'strokes': strokes, 'pivots': pivots, 'buys': buys, 'sells': sells,
            'trend': td.get_trend_status(), 'latest': close_s.iloc[-1],
        }

    @staticmethod
    def classify_2plus3_buy(buys):
        """
        识别2+3买：2买价格在中枢ZG之上（同时满足2买和3买）
        返回 [(classified_type, buy_point), ...]
        """
        results = []
        for b in buys:
            if b.point_type == '2buy':
                if b.related_pivot and b.related_pivot.zg > 0:
                    if b.price > b.related_pivot.zg:
                        results.append(('2+3buy', b))
                        continue
                results.append(('2buy', b))
            elif b.point_type == '3buy':
                results.append(('3buy', b))
        return results

    def get_30min_data(self, symbol: str, days: int = 10) -> Optional[pd.DataFrame]:
        """获取30分钟数据（优先mootdx直连，避免代理问题）"""
        code = symbol.replace('SZ', '').replace('SH', '')
        market_tag = symbol[:2].lower()
        market_id = 0 if market_tag == 'sz' else 1

        # 方法1: mootdx直连通达信（TCP连接，不走HTTP代理）
        try:
            from mootdx.quotes import Quotes
            # 每次新建客户端，避免sz/sh连接冲突
            api = Quotes.factory(market=market_tag).client
            # 分段获取，确保足够数据
            all_dfs = []
            for start in range(0, 4000, 800):
                data = api.get_security_bars(2, market_id, code, start, 800)
                if not data:
                    break
                batch = pd.DataFrame(data)
                if len(batch) == 0:
                    break
                all_dfs.append(batch)
                if len(batch) < 800:
                    break

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined['datetime'] = pd.to_datetime(combined['datetime'])
                combined = combined.rename(columns={'datetime': 'date'})
                combined = combined[['date', 'open', 'close', 'high', 'low', 'vol', 'amount']].rename(columns={'vol': 'volume'})
                combined = combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
                # 只保留最近N天的数据
                cutoff = datetime.now() - timedelta(days=days)
                combined = combined[combined['date'] >= cutoff]
                if len(combined) >= 60:
                    return combined.rename(columns={'date': 'datetime'})
        except Exception:
            pass

        # 方法2: akshare（可能被代理拦截）
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            df = ak.stock_zh_a_hist_min_em(
                symbol=code,
                start_date=start_date.strftime('%Y-%m-%d 09:30:00'),
                end_date=end_date.strftime('%Y-%m-%d 15:00:00'),
                period='30',
                adjust='qfq'
            )

            if df is None or df.empty:
                return None

            # 标准化列名
            column_map = {
                '时间': 'datetime',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
            }

            for col in df.columns:
                if 'date' in col.lower() or '时间' in col or 'datetime' in col.lower():
                    column_map[col] = 'datetime'
                elif 'open' in col.lower() or '开盘' in col:
                    column_map[col] = 'open'
                elif 'high' in col.lower() or '最高' in col:
                    column_map[col] = 'high'
                elif 'low' in col.lower() or '最低' in col:
                    column_map[col] = 'low'
                elif 'close' in col.lower() or '收盘' in col:
                    column_map[col] = 'close'
                elif 'volume' in col.lower() or '成交量' in col:
                    column_map[col] = 'volume'

            df = df.rename(columns=column_map)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)

            return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"[ERR] {symbol} 获取数据失败: {e}")
            return None

    def analyze_30min_buy(self, symbol: str) -> List[BuySignal]:
        """
        分析30分钟级别买点（2买/3买/2+3买）+ ZG突破加仓
        需日线确认非强下跌趋势才入场

        返回信号列表（可能有多个：买点+ZG突破）
        """
        signals = []

        # === 0) 日线趋势过滤 ===
        if not self.check_daily_trend(symbol):
            return signals

        # === 1) 获取30min数据 ===
        df = self.get_30min_data(symbol, days=15)
        if df is None or len(df) < 60:
            return signals

        try:
            r = self._chanlun_full(df)
            if r is None or not r['buys']:
                return signals

            current_price = r['latest']
            klen = r['klen']
            buys = r['buys']
            pivots = r['pivots']

            # === 2) 筛选最近有效买点 ===
            recent_buys = [
                b for b in buys
                if b.index >= klen - 30
                and current_price >= b.price * 0.97
                and b.stop_loss < current_price
                and b.confidence >= 0.5
            ]
            if not recent_buys:
                return signals

            # === 3) 2+3买分类 ===
            classified = self.classify_2plus3_buy(recent_buys)
            if not classified:
                return signals

            # 取最优信号
            score_order = {'2+3buy': 0, '3buy': 1, '2buy': 2}
            classified.sort(key=lambda x: (score_order.get(x[0], 9), -x[1].confidence))
            best_type, best = classified[0]

            # 信号评级
            score_map = {'2+3buy': 'A', '3buy': 'B', '2buy': 'C'}
            signal_score = score_map.get(best_type, 'C')

            type_map = {'2+3buy': '2+3买', '2buy': '2买', '3buy': '3买'}
            signal_type = type_map.get(best_type, best_type)

            # 中枢ZG/ZD
            zg = best.related_pivot.zg if best.related_pivot and best.related_pivot.zg > 0 else 0
            zd = best.related_pivot.zd if best.related_pivot and best.related_pivot.zd > 0 else 0

            stop_loss = max(best.stop_loss, current_price * 0.96)
            target = current_price * 1.10

            # 去重
            signal_key = f"{symbol}_{best_type}_{int(current_price)}_{datetime.now().hour}"
            if signal_key not in self.last_signals or datetime.now() - self.last_signals[signal_key] >= timedelta(hours=1):
                self.last_signals[signal_key] = datetime.now()
                signals.append(BuySignal(
                    symbol=symbol, name='', signal_type=signal_type,
                    timeframe='30min', price=current_price,
                    confidence=best.confidence,
                    reason=best.reason[:60] if best.reason else signal_type,
                    stop_loss=stop_loss, target=target,
                    timestamp=datetime.now(), zg=zg, pivot_zd=zd,
                    signal_score=signal_score,
                ))

            # === 4) ZG突破加仓信号 ===
            if pivots and len(pivots) >= 2:
                prev_pivot = pivots[-2]  # 前一个中枢
                if prev_pivot.zg > 0 and current_price > prev_pivot.zg:
                    zg_key = f"{symbol}_zg_break_{int(prev_pivot.zg)}_{datetime.now().hour}"
                    if zg_key not in self.last_signals or datetime.now() - self.last_signals[zg_key] >= timedelta(hours=1):
                        self.last_signals[zg_key] = datetime.now()
                        zg_stop = prev_pivot.zg * 0.97
                        signals.append(BuySignal(
                            symbol=symbol, name='', signal_type='ZG突破',
                            timeframe='30min', price=current_price,
                            confidence=0.7,
                            reason=f'突破前中枢ZG={prev_pivot.zg:.2f}',
                            stop_loss=max(zg_stop, current_price * 0.96),
                            target=current_price * 1.08,
                            timestamp=datetime.now(),
                            zg=prev_pivot.zg, pivot_zd=prev_pivot.zd,
                            signal_score='加',
                        ))

            return signals

        except Exception:
            return signals

    def _check_macd(self, macd: MACD) -> bool:
        """检查MACD条件"""
        if len(macd) < 3:
            return False

        if macd.histogram[-2] <= 0 and macd.histogram[-1] > 0:
            return True

        try:
            has_div, _ = macd.check_divergence(
                len(macd) - 30,
                len(macd) - 1,
                'down'
            )
            if has_div:
                return True
        except:
            pass

        if macd.dif[-1] > macd.dea[-1]:
            return True

        return False

    def scan_watchlist(self, watchlist=None, stock_names=None) -> List[BuySignal]:
        """扫描关注列表"""
        if watchlist is None:
            watchlist = WATCHLIST_CORE
        if stock_names is None:
            stock_names = STOCK_NAMES

        total = len(watchlist)
        print(f"\n{'='*90}")
        print(f"  缠论30分钟扫描器 — 日线定方向 + 30min找买点")
        print(f"{'='*90}")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  股票数量: {total}")
        print(f"{'-'*90}\n")

        all_signals = []
        checked = 0
        errors = 0
        progress_step = max(10, total // 20)

        for symbol in watchlist:
            checked += 1
            if checked % progress_step == 0:
                print(f"  已扫描 {checked}/{total} ({checked*100//total}%)...", flush=True)

            try:
                sigs = self.analyze_30min_buy(symbol)
                for signal in sigs:
                    if not signal.name:
                        signal.name = stock_names.get(symbol, '')
                    all_signals.append(signal)
                    score = signal.signal_score
                    name_str = signal.name[:6] if signal.name else ''
                    zg_str = f" ZG={signal.zg:.2f}" if signal.zg > 0 else ""
                    zd_str = f" ZD={signal.pivot_zd:.2f}" if signal.pivot_zd > 0 else ""
                    print(f"  [ALERT] {symbol} ({name_str}): "
                          f"{signal.signal_type}({score}级) @ {signal.price:.2f} "
                          f"conf={signal.confidence:.0%}{zg_str}{zd_str} "
                          f"SL={signal.stop_loss:.2f}")
            except Exception:
                errors += 1
                continue

            time.sleep(0.15)

        print(f"\n  扫描完成: {checked}只, 错误: {errors}只, 信号: {len(all_signals)}个")

        # 按评级排序 A > B > C > 加
        score_order = {'A': 0, 'B': 1, 'C': 2, '加': 3}
        all_signals.sort(key=lambda s: (score_order.get(s.signal_score, 9), -s.confidence))

        return all_signals

    def send_notification(self, signals: List[BuySignal]):
        """发送通知（到所有已配置的渠道）"""
        if not self.notifier:
            return

        if not signals:
            return

        # 发送汇总
        summary = format_summary_alert(signals)
        title = "缠论30分钟监控 - 信号提醒"

        # 准备HTML内容（邮件用）——叠加新闻
        html_content = None
        if self.notifier.email:
            html_parts = [f"<h2>{title}</h2><ul>"]
            for s in signals:
                news_line = ''
                news = self._fetch_stock_news(s.name, s.symbol)
                if news:
                    news_line = f'<br><em>相关新闻:</em><br><small>{news}</small>'

                html_parts.append(f"""
                <li>
                    <strong>{s.symbol} {s.name}</strong><br>
                    价格: {s.price:.2f}, 止损: {s.stop_loss:.2f}, 目标: {s.target:.2f}<br>
                    {s.reason}{news_line}
                </li>
                """)
            html_parts.append("</ul>")
            html_content = "".join(html_parts)

        self.notifier.send_all(title, summary, html_content)

        # 单独发送每个信号（企业微信/飞书卡片）——叠加新闻
        if self.notifier.feishu and len(signals) <= 3:
            for signal in signals:
                alert = format_2buy_alert(signal)
                # 叠加个股相关新闻
                news = self._fetch_stock_news(signal.name, signal.symbol)
                if news:
                    alert += f'\n\n📰 相关新闻:\n{news}'
                self.notifier.feishu.send_post(
                    f"{signal.symbol} {signal.name} - 2买信号",
                    alert
                )

    def format_signals(self, signals: List[BuySignal]) -> str:
        """格式化信号"""
        if not signals:
            return "未发现买点信号"

        lines = [
            f"\n{'='*100}",
            f"发现 {len(signals)} 个30分钟买点信号",
            f"{'='*100}",
            f"{'评级':<5}{'时间':<18}{'代码':<10}{'名称':<8}{'信号':<8}{'价格':<10}"
            f"{'置信度':<8}{'ZG':<10}{'ZD':<10}{'止损':<10}{'理由':<20}",
            f"{'-'*100}"
        ]

        for s in signals:
            zg_str = f"{s.zg:.2f}" if s.zg > 0 else "-"
            zd_str = f"{s.pivot_zd:.2f}" if s.pivot_zd > 0 else "-"
            lines.append(
                f"{s.signal_score:<5}"
                f"{s.timestamp.strftime('%m-%d %H:%M'):<18}"
                f"{s.symbol:<10}{s.name:<8}"
                f"{s.signal_type:<8}{s.price:<10.2f}"
                f"{s.confidence*100:>5.0f}%"
                f"{zg_str:<10}{zd_str:<10}{s.stop_loss:<10.2f}"
                f"{s.reason[:20]:<20}"
            )

        lines.append(f"{'='*100}")
        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='缠论30分钟2买监控')
    parser.add_argument('--once', action='store_true', help='只扫描一次')
    parser.add_argument('--interval', type=int, default=30, help='扫描间隔（分钟）')
    parser.add_argument('--symbol', type=str, help='指定单只股票')
    parser.add_argument('--no-notify', action='store_true', help='禁用所有通知')
    parser.add_argument('--scope', type=str, default='csi1000',
                        choices=['core', 'csi1000', 'all'],
                        help='扫描范围: core=47只关注, csi1000=中证1000, all=全部')

    args = parser.parse_args()

    enable_notification = not args.no_notify
    monitor = ChanLun30MinMonitor(enable_notification=enable_notification)

    if args.symbol:
        sigs = monitor.analyze_30min_buy(args.symbol)
        if sigs:
            for s in sigs:
                s.name = STOCK_NAMES.get(args.symbol.upper(), '')
            print(monitor.format_signals(sigs))
            monitor.send_notification(sigs)
        else:
            print(f"{args.symbol} 未发现买点信号（日线趋势下跌或无30min信号）")
        return

    # 构建扫描列表
    watchlist, stock_names = build_watchlist(args.scope)
    print(f"\n扫描范围: {args.scope}  股票数量: {len(watchlist)}")

    if args.once:
        signals = monitor.scan_watchlist(watchlist, stock_names)
        print(monitor.format_signals(signals))
        monitor.send_notification(signals)
    else:
        print(f"持续监控模式，扫描间隔: {args.interval}分钟")
        print("按 Ctrl+C 退出\n")

        try:
            while True:
                # 每轮重新加载列表（成分股可能调整）
                if args.scope != 'core':
                    watchlist, stock_names = build_watchlist(args.scope)

                signals = monitor.scan_watchlist(watchlist, stock_names)
                print(monitor.format_signals(signals))
                monitor.send_notification(signals)

                print(f"\n等待{args.interval}分钟后进行下次扫描...")
                print(f"下次扫描时间: {(datetime.now() + timedelta(minutes=args.interval)).strftime('%Y-%m-%d %H:%M:%S')}\n")

                time.sleep(args.interval * 60)

        except KeyboardInterrupt:
            print("\n监控已停止")


if __name__ == '__main__':
    main()
