#!/usr/bin/env python3
"""
监控 002553/300244/002307 的买点状态
- 每30分钟检查一次实时价格
- 当价格接近信号价（误差<5%）且未超信号价1%以上时推送飞书
- 逻辑模拟持仓，非真实持仓
"""
import os, sys, json, time, requests
for k in list(os.environ.keys()):
    if 'proxy' in k.lower(): os.environ.pop(k, None)
sys.path.insert(0, '/workspace')

# 飞书
webhook_url = os.environ.get('FEISHU_WEBHOOK_URL')

# 信号基准 (来自04-14缠论扫描)
SIGNALS = {
    'sz002553': {'name': '南方精工', 'sig_price': 26.15, 'sig_date': '2026-04-14', 'type': '2buy'},
    'sz300244': {'name': '迪安诊断', 'sig_price': 18.89, 'sig_date': '2026-04-14', 'type': '2buy'},
    'sz002307': {'name': '北新路桥', 'sig_price': 4.61,  'sig_date': '2026-04-14', 'type': '2buy'},
}

# 今日实时价格缓存 (避免重复推送)
CACHE_FILE = '/workspace/scripts/.price_cache.json'

def get_price(code_tq):
    """腾讯行情接口"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = 'http://qt.gtimg.cn/q=%s' % code_tq
        r = requests.get(url, headers=headers, timeout=5)
        parts = r.text.split('~')
        if len(parts) > 10:
            return {'price': float(parts[3]), 'chg_pct': float(parts[5]), 'chg_amt': float(parts[4])}
    except:
        pass
    return None

def load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except:
        return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass

def send_feishu(sig_info, price_data):
    if not webhook_url:
        print('无飞书webhook配置')
        return False
    code_tq = sig_info['code_tq']
    code = code_tq.replace('sz', '').upper() + '.SZ'
    name = sig_info['name']
    sig_price = sig_info['sig_price']
    cur_price = price_data['price']
    chg_pct = price_data['chg_pct']
    dist = (cur_price - sig_price) / sig_price * 100

    card = {
        'msg_type': 'interactive',
        'card': {
            'header': {'title': {'tag': 'plain_text', 'content': '🟢 买点提醒 %s' % name}, 'template': 'green'},
            'elements': [
                {'tag': 'div', 'text': {'tag': 'lark_md', 'content': '**【策略模拟持仓-非真实持仓】**'}},
                {'tag': 'hr'},
                {'tag': 'div', 'text': {'tag': 'lark_md', 'content': '**%s %s**\n\n类型: %s\n信号日期: %s\n信号价: %.2f\n现价: %.2f (%+.2f%%)\n偏离信号价: %+.2f%%\n\n状态: ✅ **买点区间内**' % (
                    code, name, sig_info['type'], sig_info['sig_date'],
                    sig_price, cur_price, chg_pct, dist
                )}},
                {'tag': 'hr'},
                {'tag': 'div', 'text': {'tag': 'lark_md', 'content': '**操作建议**\n• 止损: %.2f (SL=6%%)' % (sig_price * 0.94)}},
                {'tag': 'div', 'text': {'tag': 'lark_md', 'content': '• 目标: %.2f (中枢上沿参考)' % (sig_price * 1.10)}},
                {'tag': 'div', 'text': {'tag': 'lark_md', 'content': '• 仓位: 单票≤30% | 每30分钟检查'}},
            ]
        }
    }
    r = requests.post(webhook_url, headers={'Content-Type': 'application/json'},
                      data=json.dumps(card, ensure_ascii=False).encode('utf-8'), timeout=10)
    if r.status_code == 200 and 'success' in r.text:
        return True
    print('飞书失败:', r.text[:100])
    return False

def main():
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] 检查买点状态...' % now)

    cache = load_cache()
    pushed = []

    for code_tq, info in SIGNALS.items():
        pd_data = get_price(code_tq)
        if pd_data is None:
            print('  %s: 获取失败' % code_tq)
            continue

        name = info['name']
        sig_price = info['sig_price']
        cur_price = pd_data['price']
        chg_pct = pd_data['chg_pct']
        dist = (cur_price - sig_price) / sig_price * 100

        print('  %s %s: 现价=%.2f 涨跌=%+.2f%% 偏离=%+.2f%%' % (
            code_tq.replace('sz', '').upper(), name, cur_price, chg_pct, dist))

        # 买点区间: 信号价的 -5% ~ +1% (即回踩到信号价附近或小涨)
        in_zone = -5.0 <= dist <= 1.0

        last = cache.get(code_tq, {})
        already_pushed = last.get('pushed', False)

        if in_zone and not already_pushed:
            info2 = dict(info)
            info2['code_tq'] = code_tq
            ok = send_feishu(info2, pd_data)
            if ok:
                cache[code_tq] = {'pushed': True, 'price': cur_price, 'dist': dist}
                pushed.append(code_tq.replace('sz', '').upper())
                print('  → 🟢 飞书推送成功!')
        elif in_zone:
            print('  → 买点区间内 (今日已推送过)')
        else:
            if dist < -5:
                print('  → 跌破买点，重置标志')
                cache[code_tq] = {'pushed': False, 'price': cur_price, 'dist': dist}
            else:
                print('  → 高于信号价1%%+，等待回踩')

    save_cache(cache)

    if pushed:
        print('\n✅ 已推送买点: %s' % ', '.join(pushed))
    else:
        print('\n今日买点状态: 无新推送 (已推送过的不会重复)')

    # 简单状态汇总
    print('\n=== 买点监控状态 ===')
    for code_tq, info in SIGNALS.items():
        pd_data = get_price(code_tq)
        if pd_data:
            dist = (pd_data['price'] - info['sig_price']) / info['sig_price'] * 100
            if dist < -5:
                tag = '❌跌破买点'
            elif dist <= 1.0:
                tag = '✅买点区间'
            else:
                tag = '⏳等待回踩'
            print('  %s: %.2f (偏离%+.1f%%) %s' % (info['name'], pd_data['price'], dist, tag))

if __name__ == '__main__':
    main()
