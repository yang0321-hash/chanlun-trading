"""
持仓止损止盈实时监控 — Sina实时行情，每5分钟检查一次，触发时飞书通知+控制台提醒
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
ALERT_LOG = 'signals/stop_alerts.json'

# 持仓 → 所属行业ETF映射 (用于板块联动预警)
SECTOR_ETF_MAP = {
    'sz002600': ('sh515880', '通信ETF'),   # 领益智造-消费电子/通信
    'sz300936': ('sz159915', '创业板ETF'), # 中英科技-通信设备
    'sh688613': ('sh512170', '医药ETF'),   # 奥精医疗-医疗器械
    'sz301062': ('sh512100', '中证500ETF'),# 铭科精技-汽车零部件(中小盘)
    'sz002951': ('sh512100', '中证500ETF'),# 金时科技-印刷(中小盘)
    'sz000826': ('sh512100', '中证500ETF'),# 启迪环境-环保(中小盘)
    'sz301128': ('sz159915', '创业板ETF'), # 强瑞技术-通信设备
}

# 板块联动阈值
SECTOR_ALERT_THRESHOLD = -2.0   # 行业ETF跌幅超2%
CSI500_ALERT_THRESHOLD = -1.5   # 中证500跌幅超1.5%


def load_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_alert_log():
    if os.path.exists(ALERT_LOG):
        with open(ALERT_LOG, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"alerts": [], "last_check": ""}


def save_alert_log(log):
    with open(ALERT_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def _sina_session():
    s = requests.Session()
    s.trust_env = False
    return s


def fetch_realtime(session, codes: list) -> dict:
    """从Sina获取实时行情，返回 {code: {price, prev, chg}}"""
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
    except requests.exceptions.RequestException as e:
        print(f'  [Sina] 请求失败: {e}')
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
        result[code] = {
            'price': price,
            'prev': prev,
            'chg': (price - prev) / prev * 100 if prev > 0 else 0,
        }
    return result


def send_feishu(title, text):
    """发送飞书通知"""
    webhook = os.getenv('CHANLUN_FEISHU_WEBHOOK_URL', '')
    if not webhook:
        return False
    try:
        session = _sina_session()
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": title}},
                "elements": [{"tag": "markdown", "content": text}]
            }
        }
        resp = session.post(webhook, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def check_positions():
    """检查所有持仓的止损止盈 (Sina实时行情)"""
    config = load_config()
    log = load_alert_log()
    session = _sina_session()

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    alerts = []

    print(f'\n===== 止损止盈监控 {now} =====')

    # 批量获取所有持仓实时价
    codes = [pos['code'] for pos in config]
    quotes = fetch_realtime(session, codes)

    for pos in config:
        code = pos['code']
        name = pos['name']
        entry = pos['entry']
        stop = pos['stop']
        tp = pos['tp']

        q = quotes.get(code)
        if not q:
            print(f'  {name}({code}): 获取失败')
            continue

        price = q['price']
        chg = q['chg']
        pnl = (price - entry) / entry * 100
        stop_dist = (price - stop) / price * 100
        tp_dist = (tp - price) / price * 100

        status = ''
        alert_type = ''

        if price <= stop:
            status = '*** 止损触发 ***'
            alert_type = 'STOP_LOSS'
        elif price >= tp:
            status = '*** 止盈触发 ***'
            alert_type = 'TAKE_PROFIT'
        elif stop_dist < 3:
            status = f'! 接近止损 (仅{stop_dist:.1f}%)'
            alert_type = 'NEAR_STOP'
        elif tp_dist < 5:
            status = f'! 接近止盈 (仅{tp_dist:.1f}%)'
            alert_type = 'NEAR_TP'

        print(f'  {name}({code}): {price:.2f} ({chg:+.2f}%) 盈亏{pnl:+.1f}% | '
              f'距止损{stop_dist:.1f}% 距止盈{tp_dist:.1f}% {status}')

        if alert_type and alert_type not in ('NEAR_STOP', 'NEAR_TP'):
            alert_msg = f'**{alert_type}** {name}({code}) 现价{price:.2f}'
            if alert_type == 'STOP_LOSS':
                alert_msg += f' <= 止损{stop:.2f}'
            else:
                alert_msg += f' >= 止盈{tp:.2f}'
            alerts.append(alert_msg)

    # 板块联动预警
    sector_alerts = check_sector_correlation(session, config, quotes)
    alerts.extend(sector_alerts)

    # 发送警报
    if alerts:
        title = f'持仓警报 ({len(alerts)}条)'
        text = '\n'.join(alerts)
        print(f'\n  >>> 发送飞书警报: {title}')
        send_feishu(title, text)

    log['last_check'] = now
    if alerts:
        log['alerts'].extend(alerts)
        log['alerts'] = log['alerts'][-50:]
    save_alert_log(log)

    return alerts


def check_sector_correlation(session, config, quotes) -> list:
    """板块联动预警: 行业ETF大跌时对关联持仓发出预警"""
    alerts = []

    # 收集需要查询的ETF (去重)
    etf_codes_needed = set()
    for pos in config:
        mapping = SECTOR_ETF_MAP.get(pos['code'])
        if mapping:
            etf_codes_needed.add(mapping[0])
    # 加上中证500
    etf_codes_needed.add('sh512100')

    if not etf_codes_needed:
        return alerts

    etf_quotes = fetch_realtime(session, list(etf_codes_needed))

    # 检查中证500整体风险
    csi500 = etf_quotes.get('sh512100', {})
    csi500_chg = csi500.get('chg', 0)
    csi500_alert = csi500_chg < CSI500_ALERT_THRESHOLD

    if csi500_alert:
        print(f'  [板块联动] 中证500ETF {csi500_chg:+.2f}% 跌超{CSI500_ALERT_THRESHOLD}%，中小盘整体承压')

    for pos in config:
        code = pos['code']
        name = pos['name']
        q = quotes.get(code, {})
        price = q.get('price', 0)
        stop = pos['stop']
        stop_dist = (price - stop) / price * 100 if price > 0 else 999

        mapping = SECTOR_ETF_MAP.get(code)
        if not mapping:
            continue

        etf_code, etf_name = mapping
        etf_q = etf_quotes.get(etf_code, {})
        etf_chg = etf_q.get('chg', 0)

        # 行业ETF大跌
        if etf_chg < SECTOR_ALERT_THRESHOLD:
            msg = f'**板块预警** {name}({code}) 所属{etf_name}跌{etf_chg:+.2f}%，个股距止损{stop_dist:.1f}%'
            alerts.append(msg)
            print(f'  [板块联动] {name} 所属{etf_name} {etf_chg:+.2f}% → 预警')

        # 中证500大跌 + 个股本身偏近止损
        if csi500_alert and stop_dist < 5:
            msg = f'**中小盘风险** {name}({code}) 中证500跌{csi500_chg:+.2f}%，距止损仅{stop_dist:.1f}%'
            alerts.append(msg)
            print(f'  [板块联动] {name} 中证500弱势 + 距止损{stop_dist:.1f}% → 预警')

    return alerts


if __name__ == '__main__':
    check_positions()
