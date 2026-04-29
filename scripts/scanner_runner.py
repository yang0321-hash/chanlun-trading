#!/usr/bin/env python3
"""
盘后扫描器 runner — 每日收盘后自动运行
1. 快速更新TDX数据 (watchlist+持仓+龙头)
2. 运行scanner_v3_mp.py全A扫描
3. 解析信号并推送飞书
"""
import os, sys, time, json, subprocess, pickle
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspace')

WEBHOOK_URL = os.environ.get('FEISHU_WEBHOOK_URL')
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
SCANNER_SCRIPT = '/workspace/chanlun_system/scanner_v3_mp.py'
UPDATER_SCRIPT = '/workspace/scripts/tdx_full_updater.py'
SIGNALS_CACHE = '/workspace/scanner_v3_signals.pkl'
PREV_SIGNALS = '/workspace/scanner_prev_signals.pkl'

def send_feishu(card):
    import requests
    r = requests.post(WEBHOOK_URL,
        headers={'Content-Type':'application/json'},
        data=json.dumps(card, ensure_ascii=False).encode('utf-8'), timeout=10)
    if r.status_code == 200 and 'success' in r.text:
        return True
    print('飞书失败:', r.text[:100])
    return False

def send_scan_results(sig_df):
    """格式化扫描结果并推送飞书"""
    import pandas as pd

    # 读取历史信号做对比 (新增信号)
    prev = set()
    if os.path.exists(PREV_SIGNALS):
        try:
            prev_df = pd.read_pickle(PREV_SIGNALS)
            prev = set(zip(prev_df['code'], prev_df['date']))
        except: pass

    # 近7天信号
    cutoff = (datetime.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    sigs = sig_df[sig_df['date'] >= cutoff].copy()

    # 按日期分组
    by_date = sigs.groupby('date').apply(
        lambda g: g.sort_values('code').to_dict('records')
    ).to_dict()

    # 构建飞书消息
    date_blocks = []
    for date in sorted(by_date.keys(), reverse=True):
        items = by_date[date]
        # 过滤只剩2buy/2plus3buy
        items = [x for x in items if x['type'] in ('2buy', '2plus3buy')]
        if not items: continue
        new_items = [x for x in items if (x['code'], x['date']) not in prev]
        label = f"🆕 新信号({len(new_items)})" if new_items else "信号"
        rows = []
        for x in items[:8]:  # 每天最多8只
            code = x['code']
            name_map = {
                'SZ300413': '完美世界', 'SZ002553': '南方精工', 'SZ300244': '迪安诊断',
                'SZ002307': '北新路桥', 'SZ000826': '启迪环境', 'SZ300936': '强瑞技术',
                'SZ002600': '领益智造', 'SZ301062': '上海艾录', 'SZ688613': '奥精医疗',
                'SZ002951': '金时科技', 'SZ301128': '卓锦股份',
            }
            name = name_map.get(code, code.replace('SZ','深').replace('SH','沪'))
            dist = (x['price'] - x['sl_price']) / x['sl_price'] * 100
            type_icon = '🔵' if x['type'] == '2buy' else '🟢'
            new_tag = ' 🆕' if (code, date) in prev else ''
            rows.append(f"{type_icon} **{name}** ({code})\n   {x['date']} @ {x['price']:.2f} SL={x['sl_price']:.2f}{new_tag}")
        date_blocks.append(f"**📅 {date}** {label}\n" + '\n'.join(rows))
        if len(date_blocks) >= 5: break

    if not date_blocks:
        text = '📊 近7日无新2买/2+3买信号'
        blocks = [{'tag':'div', 'text':{'tag':'lark_md', 'content':text}}]
    else:
        blocks = [{'tag':'div', 'text':{'tag':'lark_md', 'content':b}} for b in date_blocks]

    total = len(sigs[sigs['date'] >= cutoff])
    card = {
        'msg_type': 'interactive',
        'card': {
            'header': {'title': {'tag':'plain_text', 'content':'📊 缠论盘后扫描报告'}, 'template':'blue'},
            'elements': [
                {'tag':'div', 'text':{'tag':'lark_md', 'content':f"**【策略模拟持仓-非真实持仓】**\n全A扫描 | 近7日信号 | 共**{total}**个"}},
                {'tag':'hr'},
            ] + blocks + [
                {'tag':'hr'},
                {'tag':'div', 'text':{'tag':'lark_md', 'content':f"🕐 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n参数: SL=6% | TP=3%/5% | 仓位≤30%"}}
            ]
        }
    }
    return send_feishu(card)

def run():
    t0 = time.time()
    today_str = datetime.now().strftime('%Y-%m-%d')

    # Step 1: 快速更新TDX
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Step1: TDX快速更新...')
    r = subprocess.run([sys.executable, UPDATER_SCRIPT, '--quick'],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout[-300:] if r.stdout else '无输出')
    if r.returncode != 0: print('更新器错误:', r.stderr[-200:])

    # Step 2: 扫描
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Step2: 全A扫描...')
    scanner_env = dict(os.environ)
    r = subprocess.run([sys.executable, SCANNER_SCRIPT],
                       capture_output=True, text=True, timeout=600, env=scanner_env)
    print(r.stdout[-300:] if r.stdout else '无输出')
    if r.returncode != 0:
        print('扫描器错误:', r.stderr[-200:])
        return

    # Step 3: 读取结果并推送
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Step3: 飞书推送...')
    if not os.path.exists(SIGNALS_CACHE):
        print('无信号文件')
        return

    import pandas as pd
    sig_df = pd.read_pickle(SIGNALS_CACHE)
    print(f'扫描结果: {len(sig_df)} 个近30天信号')

    ok = send_scan_results(sig_df)
    print(f'飞书推送: {"✅ 成功" if ok else "❌ 失败"}')

    # 保存历史
    try:
        sig_df.to_pickle(PREV_SIGNALS)
    except: pass

    print(f'总耗时: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    run()
