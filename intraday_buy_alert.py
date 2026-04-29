#!/usr/bin/env python3
"""盘中实时买点预警

读取买点预警池 (signals/buy_alert_pool.json)，
盘中每30分钟获取30min K线数据，运行缠论分析检测买点信号，
发现买点立即推送到飞书。

使用方式：
    python intraday_buy_alert.py              # 单次扫描
    python intraday_buy_alert.py --loop       # 持续监控（盘中自动运行）
"""
import sys, os
sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
from dotenv import load_dotenv
load_dotenv()

import json, time, argparse
from datetime import datetime, timedelta
from loguru import logger

import numpy as np
import pandas as pd
import requests
from data.sina_source import SinaSource
from data.hybrid_source import HybridSource
from data.hot_sector_analyzer import HotSectorAnalyzer

sys.path.insert(0, 'chanlun_unified')
from signal_engine_cc15 import SignalEngine
from backtest_cc15_mtf import _build_pivots_from_strokes

ALERT_POOL_PATH = 'signals/buy_alert_pool.json'
PUSH_LOG = 'signals/intraday_push_log.json'


def _session():
    s = requests.Session()
    s.trust_env = False
    return s


def send_feishu(title, text):
    webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL', '')
    if not webhook:
        return False
    try:
        s = _session()
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": title}},
                "elements": [{"tag": "markdown", "content": text}]
            }
        }
        r = s.post(webhook, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"飞书推送失败: {e}")
        return False


def load_alert_pool():
    if not os.path.exists(ALERT_POOL_PATH):
        return None
    with open(ALERT_POOL_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_push_log():
    if not os.path.exists(PUSH_LOG):
        return {}
    with open(PUSH_LOG, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_push_log(log):
    os.makedirs('signals', exist_ok=True)
    with open(PUSH_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def _normalize_code(code):
    """标准化为 SinaSource 格式: sh600519 / sz000001"""
    c = code.strip().lower()
    for suffix in ['.sh', '.sz', '.bj']:
        c = c.replace(suffix, '')
    if c.startswith('sh') or c.startswith('sz'):
        return c
    if c.startswith('6'):
        return f'sh{c}'
    return f'sz{c}'


def _to_display_code(code):
    """标准化为显示格式: sh600519"""
    return _normalize_code(code)


def detect_30min_buy_signals(code, sina):
    """获取30min K线并运行缠论分析，返回买点信号列表"""
    sina_code = _normalize_code(code)
    df = sina.get_kline(sina_code, period='30min')
    if df is None or len(df) < 60:
        return []

    engine = SignalEngine()
    try:
        _, _, _, strokes = engine._detect_bi_deterministic(df)
    except Exception:
        return []

    if len(strokes) < 6:
        return []

    pivots = _build_pivots_from_strokes(strokes)

    # MACD
    close = df['close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    n = len(df)
    signals = []

    # 只检测最后一根K线形成的信号（盘中实时，不是历史）
    last_idx = n - 1

    # --- 1买检测: 最后一笔底背驰 ---
    down_strokes = [s for s in strokes if s['start_type'] == 'top' and s['end_type'] == 'bottom']
    for k in range(1, len(down_strokes)):
        curr = down_strokes[k]
        # 底背驰的终点必须在最后一根K线
        if curr['end_idx'] != last_idx:
            continue
        curr_area = abs(sum(hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
        for lb in range(1, min(4, k+1)):
            prev = down_strokes[k - lb]
            prev_area = abs(sum(hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            if curr['end_val'] < prev['end_val'] and curr_area < prev_area:
                signals.append({
                    'type': '1buy',
                    'idx': last_idx,
                    'price': round(float(close.iloc[last_idx]), 2),
                    'time': str(df.index[last_idx])[:16],
                    'area_ratio': round(float(curr_area / prev_area), 2) if prev_area > 0 else 0,
                })
                break

    # --- 2买检测: 1买后回踩，距离极近 ---
    buy_div_set = set()
    for k in range(1, len(down_strokes)):
        curr = down_strokes[k]
        curr_area = abs(sum(hist.iloc[curr['start_idx']:curr['end_idx']+1].values))
        for lb in range(1, min(4, k+1)):
            prev = down_strokes[k - lb]
            prev_area = abs(sum(hist.iloc[prev['start_idx']:prev['end_idx']+1].values))
            if curr['end_val'] < prev['end_val'] and curr_area < prev_area:
                buy_div_set.add(curr['end_idx'])
                break

    if buy_div_set:
        last_1buy = max(buy_div_set)
        low = df['low']
        last_close = close.iloc[-1]
        buy_low = low.iloc[last_1buy]
        bars_since = n - 1 - last_1buy

        # 2买：1买后3-10根K线，回踩距离<0.5%
        if 3 <= bars_since <= 10:
            dist_pct = (last_close - buy_low) / buy_low * 100
            if 0 < dist_pct <= 0.5:
                signals.append({
                    'type': '2buy',
                    'idx': last_idx,
                    'price': round(float(last_close), 2),
                    'time': str(df.index[-1])[:16],
                    'distance_pct': round(float(dist_pct), 2),
                    'bars_since_1buy': bars_since,
                })

    # --- 3买检测: 突破中枢后回踩在ZG附近 ---
    if pivots:
        last_pivot = pivots[-1]
        zg = last_pivot['zg']
        zd = last_pivot['zd']
        last_close = close.iloc[-1]

        up_strokes = [s for s in strokes if s['start_type'] == 'bottom' and s['end_type'] == 'top']
        if up_strokes:
            last_up = up_strokes[-1]
            # 突破ZG必须在最近3根K线内
            if last_up['end_val'] > zg and last_up['end_idx'] >= last_idx - 2:
                recent_low = df['low'].iloc[last_up['end_idx']:].min()
                if zd <= recent_low <= zg and last_close >= zg * 0.99:
                    signals.append({
                        'type': '3buy',
                        'idx': last_up['end_idx'],
                        'price': round(float(last_close), 2),
                        'time': str(df.index[-1])[:16],
                        'pivot_zg': round(float(zg), 2),
                        'pivot_zd': round(float(zd), 2),
                    })

    return signals


def _get_stock_name(code, hs):
    """获取股票名称"""
    try:
        pure = _normalize_code(code)
        if not pure.endswith(('.SZ', '.SH')):
            pure = pure.upper()
            pure = pure.replace('SH', '') if pure.startswith('SH') else pure
            pure = pure.replace('SZ', '') if pure.startswith('SZ') else pure
            pure = (pure + '.SH') if code.startswith('6') else (pure + '.SZ')
        q = hs.get_realtime_quote([pure])
        if len(q) > 0:
            return q.iloc[0].get('name', code)
    except Exception:
        pass
    return code


def _load_sector_map():
    """加载股票→板块映射"""
    path = 'chanlun_system/full_sector_map.json'
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('stock_to_sector', data) if isinstance(data, dict) else {}


def _get_hot_sector_stocks():
    """获取今天热点板块内的所有股票集合"""
    try:
        analyzer = HotSectorAnalyzer()
        hot_sectors = analyzer.identify_hot_sectors(top_n=15)
        if not hot_sectors:
            return set(), []

        # 评分>15的板块 = 热点（启动/加速阶段资金流入）
        hot_names = set()
        active_names = set()
        for s in hot_sectors:
            if s.score >= 25:
                hot_names.add(s.name)
            elif s.score >= 15:
                active_names.add(s.name)

        # 热点板块内所有股票
        sector_map = _load_sector_map()
        hot_stocks = set()
        active_stocks = set()
        for code_num, sector in sector_map.items():
            full = (f'sh{code_num}' if code_num.startswith('6') else f'sz{code_num}')
            if sector in hot_names:
                hot_stocks.add(full)
            elif sector in active_names:
                active_stocks.add(full)

        return hot_stocks | active_stocks, hot_sectors

    except Exception as e:
        logger.debug(f'热点板块加载失败: {e}')
        return set(), []


def scan_once(dry_run=False):
    """单次扫描预警池中所有股票"""
    pool = load_alert_pool()
    if not pool:
        print('预警池不存在，请先运行: python generate_buy_alert_pool.py')
        return []

    # 优先使用top_candidates（最紧迫的100只），fallback到全部alerts
    alerts = pool.get('top_candidates', pool.get('alerts', []))
    if not alerts:
        print('预警池为空')
        return []

    print(f'=== 盘中买点预警扫描 ===')
    print(f'预警池: {len(alerts)} 只股票, 生成时间: {pool.get("generate_time", "?")}')

    # 加载热点板块过滤
    hot_stocks, hot_sectors = _get_hot_sector_stocks()
    if hot_sectors:
        print(f'热点板块: {", ".join(s.name for s in hot_sectors[:5])} (共{len(hot_stocks)}只)')
    else:
        print('(热点板块加载失败，不过滤)')
    print()

    sina = SinaSource()
    hs = HybridSource()
    push_log = load_push_log()
    today = datetime.now().strftime('%Y-%m-%d')

    results = []

    for i, alert in enumerate(alerts):
        code = alert['code']
        alert_type = alert.get('alert_type', '?')
        display_code = _to_display_code(code)

        try:
            signals = detect_30min_buy_signals(code, sina)

            if signals:
                name = _get_stock_name(code, hs)

                # 板块过滤：热点板块推送，非热点仅记录
                in_hot_sector = bool(hot_stocks) and display_code in hot_stocks

                for sig in signals:
                    sig['code'] = display_code
                    sig['name'] = name
                    sig['alert_type'] = alert_type
                    sig['in_hot_sector'] = in_hot_sector

                    # 去重: 同一天同一股票同一信号类型只推一次
                    dedup_key = f"{today}_{display_code}_{sig['type']}"
                    if dedup_key in push_log:
                        continue

                    results.append(sig)

                    if in_hot_sector:
                        push_log[dedup_key] = {
                            'time': datetime.now().strftime('%H:%M'),
                            'price': sig['price'],
                        }
                        tag = '[hot]'
                    else:
                        tag = '[other]'

                    print(f'  {tag} [{sig["type"].upper()}] {display_code} {name} | {sig["price"]} | {sig["time"]}')

                    # 只有热点板块才推送飞书
                    if in_hot_sector and not dry_run:
                        title = f'[30min {sig["type"]}] {name}'
                        text = (
                            f'**{name}** ({display_code})\n'
                            f'信号类型: {sig["type"]}\n'
                            f'当前价格: {sig["price"]}\n'
                            f'时间: {sig["time"]}\n'
                            f'预警来源: {alert_type}\n'
                        )
                        if 'area_ratio' in sig:
                            text += f'MACD面积比: {sig["area_ratio"]}\n'
                        if 'distance_pct' in sig:
                            text += f'距1买低点: {sig["distance_pct"]}%\n'
                        if 'pivot_zg' in sig:
                            text += f'中枢: {sig["pivot_zd"]}-{sig["pivot_zg"]}\n'

                        ok = send_feishu(title, text)
                        if ok:
                            print(f'    -> pushed to Feishu')

        except Exception as e:
            logger.debug(f'{display_code} scan error: {e}')

        # 每10只打印进度
        if (i + 1) % 10 == 0:
            print(f'  ... scanned {i+1}/{len(alerts)}')

    # 保存推送日志
    save_push_log(push_log)

    print(f'\n扫描完成: {len(results)} 个买点信号')
    return results


def is_trading_time():
    """判断当前是否在交易时间"""
    now = datetime.now()
    # 工作日
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    # 9:30 - 11:30, 13:00 - 15:00
    return (930 <= t <= 1130) or (1300 <= t <= 1500)


def run_loop(interval_minutes=30):
    """持续监控模式：每隔 interval_minutes 扫描一次"""
    print(f'=== 盘中买点监控启动 ===')
    print(f'扫描间隔: {interval_minutes} 分钟')
    print(f'仅在交易时间运行 (9:30-11:30, 13:00-15:00)')
    print()

    while True:
        if is_trading_time():
            try:
                scan_once()
            except Exception as e:
                logger.error(f'扫描异常: {e}')
        else:
            now_str = datetime.now().strftime('%H:%M')
            print(f'[{now_str}] 非交易时间，等待...')

        time.sleep(interval_minutes * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='盘中实时买点预警')
    parser.add_argument('--loop', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=30, help='扫描间隔(分钟)')
    parser.add_argument('--dry-run', action='store_true', help='不推送飞书')
    args = parser.parse_args()

    if args.loop:
        run_loop(args.interval)
    else:
        scan_once(dry_run=args.dry_run)
