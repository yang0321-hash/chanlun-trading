"""
尾盘高频监控 — 14:45~15:00 每3分钟扫描持仓+ETF
复用stop_monitor的逻辑，但对尾盘异动敏感度更高
"""
import sys, os, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)
from dotenv import load_dotenv; load_dotenv()

from datetime import datetime
import requests

CONFIG_FILE = 'signals/stop_monitor_config.json'
SNAP_FILE = 'signals/tail_snapshot.json'

# 尾盘异动阈值 (比盘中更敏感)
TAIL_STOP_WARN = 5.0      # 距止损<5%就预警 (盘中是3%)
TAIL_PRICE_SWING = 2.0    # 3分钟内价格变动超2%即异动
TAIL_SECTOR_THRESHOLD = -1.5  # ETF跌幅超1.5%即预警 (盘中是2%)

ETF_LIST = {
    'sh512000': '券商ETF', 'sh510300': '沪深300ETF', 'sh510050': '上证50ETF',
    'sh512100': '中证500ETF', 'sz159915': '创业板ETF', 'sh512660': '军工ETF',
    'sh512170': '医药ETF', 'sh512690': '白酒ETF', 'sh515790': '光伏ETF',
    'sh516160': '新能源车ETF', 'sh515880': '通信ETF', 'sh512400': '有色金属ETF',
    'sh512800': '银行ETF', 'sh159928': '消费ETF',
}

SECTOR_ETF_MAP = {
    'sz002600': ('sh515880', '通信ETF'),
    'sz300936': ('sz159915', '创业板ETF'),
    'sh688613': ('sh512170', '医药ETF'),
    'sz301062': ('sh512100', '中证500ETF'),
    'sz002951': ('sh512100', '中证500ETF'),
    'sz000826': ('sh512100', '中证500ETF'),
    'sz301128': ('sz159915', '创业板ETF'),
}


def _session():
    s = requests.Session()
    s.trust_env = False
    return s


def fetch_sina(session, codes):
    sina_codes = []
    code_map = {}
    for c in codes:
        pre = 'sh' if c.startswith('sh') else 'sz'
        sc = f'{pre}{c[2:]}'
        sina_codes.append(sc)
        code_map[sc] = c

    url = 'http://hq.sinajs.cn/list=' + ','.join(sina_codes)
    headers = {'Referer': 'http://finance.sina.com.cn'}
    try:
        resp = session.get(url, headers=headers, timeout=10)
    except requests.exceptions.RequestException:
        return {}

    result = {}
    for line in resp.text.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('="')
        if len(parts) < 2:
            continue
        raw = parts[0].split('_')[-1]
        data = parts[1].rstrip('"').split(',')
        if len(data) < 32:
            continue
        price = float(data[3]) if float(data[3]) > 0 else float(data[2])
        prev = float(data[2])
        code = code_map.get(raw, raw)
        result[code] = {'price': price, 'prev': prev, 'chg': (price - prev) / prev * 100 if prev > 0 else 0}
    return result


def send_feishu(title, text):
    webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL', '')
    if not webhook:
        return
    try:
        s = _session()
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": title}},
                "elements": [{"tag": "markdown", "content": text}]
            }
        }
        s.post(webhook, json=payload, timeout=10)
    except Exception:
        pass


def run_tail_monitor():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

    session = _session()
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    alerts = []

    print(f'\n===== 尾盘高频监控 {now} =====')

    # 持仓实时价
    pos_codes = [p['code'] for p in config]
    pos_quotes = fetch_sina(session, pos_codes)

    # 上次快照 (对比3分钟变化)
    prev_snap = {}
    if os.path.exists(SNAP_FILE):
        with open(SNAP_FILE, 'r', encoding='utf-8') as f:
            prev_snap = json.load(f)

    # 检查持仓
    for pos in config:
        code = pos['code']
        name = pos['name']
        entry = pos['entry']
        stop = pos['stop']
        tp = pos['tp']
        q = pos_quotes.get(code)
        if not q:
            continue

        price = q['price']
        chg = q['chg']
        pnl = (price - entry) / entry * 100
        stop_dist = (price - stop) / price * 100
        tp_dist = (tp - price) / price * 100

        # 尾盘放宽预警距离
        status = ''
        if price <= stop:
            status = '*** 尾盘止损触发 ***'
            alerts.append(f'**尾盘止损** {name}({code}) {price:.2f} <= 止损{stop:.2f}')
        elif price >= tp:
            status = '*** 尾盘止盈触发 ***'
            alerts.append(f'**尾盘止盈** {name}({code}) {price:.2f} >= 止盈{tp:.2f}')
        elif stop_dist < TAIL_STOP_WARN:
            status = f'! 尾盘接近止损 ({stop_dist:.1f}%)'

        # 3分钟价格急变检测
        prev_price = prev_snap.get(code, {}).get('price', price)
        price_delta = (price - prev_price) / prev_price * 100 if prev_price > 0 else 0
        if abs(price_delta) >= TAIL_PRICE_SWING:
            direction = '急拉' if price_delta > 0 else '急跌'
            status += f' [{direction}{price_delta:+.2f}%]'
            alerts.append(f'**尾盘{direction}** {name}({code}) {price:.2f} 3分钟变动{price_delta:+.2f}%')

        print(f'  {name}({code}): {price:.2f} ({chg:+.2f}%) 盈亏{pnl:+.1f}% 距止损{stop_dist:.1f}% {status}')

    # ETF板块联动 (尾盘更敏感)
    etf_codes = list(ETF_LIST.keys())
    etf_quotes = fetch_sina(session, etf_codes)

    print(f'\n  [ETF板块]')
    for etf_code, etf_name in ETF_LIST.items():
        eq = etf_quotes.get(etf_code)
        if not eq:
            continue
        etf_chg = eq['chg']
        flag = ''
        if etf_chg < TAIL_SECTOR_THRESHOLD:
            flag = '!!'
            # 检查关联持仓
            for pos in config:
                mapping = SECTOR_ETF_MAP.get(pos['code'])
                if mapping and mapping[0] == etf_code:
                    pq = pos_quotes.get(pos['code'], {})
                    stop = pos['stop']
                    sd = (pq.get('price', 0) - stop) / pq.get('price', 1) * 100 if pq.get('price', 0) > 0 else 999
                    alerts.append(
                        f'**尾盘板块风险** {pos["name"]}({pos["code"]}) '
                        f'所属{etf_name}跌{etf_chg:+.2f}%，距止损{sd:.1f}%'
                    )
        print(f'    {etf_name}: {etf_chg:+.2f}% {flag}')

    # 发送飞书
    if alerts:
        title = f'尾盘警报 ({len(alerts)}条) {now}'
        text = '\n'.join(alerts)
        print(f'\n  >>> 飞书: {title}')
        send_feishu(title, text)

    # 保存快照
    snap = {}
    for code, q in pos_quotes.items():
        snap[code] = {'price': q['price']}
    with open(SNAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(snap, f)

    return alerts


if __name__ == '__main__':
    run_tail_monitor()
