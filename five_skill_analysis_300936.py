# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from mootdx.quotes import Quotes
import json
import os

# 设置路径和UTF-8输出
sys.path.insert(0, '.claude/skills/chanlun')
sys.path.insert(0, 'D:/新建文件夹/trump-code/skills')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入缠论分析器
from chanlun import ChanLunAnalyzer

# 初始化mootdx客户端
client = Quotes.factory(market='std')

code = '300936'
market = 0  # 深圳

print('='*100)
print(' sz300936 (中英科技) - 五技能联合分析报告 '.center(100, '='))
print('='*100)
print(f'分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'技能组合: mootdx-astock + stock-analysis + a-share-short-decision + chanlun + stock-monitor-skill')
print('='*100)

# ========== 技能1: mootdx-astock - 实时行情 ==========
quote = client.quotes(symbol=code, market=market)
row = quote.iloc[0]

print('\n' + '【一、实时行情】mootdx-astock'.ljust(100, '-'))
print(f'  股票代码: sz300936')
print(f'  股票名称: 中英科技 (通信设备)')
print(f'  最新价: {row["price"]:.2f}  今开: {row["open"]:.2f}  昨收: {row.get("last_close", 0):.2f}')
print(f'  最高: {row["high"]:.2f}  最低: {row["low"]:.2f}')
change_pct = (row["price"]-row["open"])/row["open"]*100 if row["open"]>0 else 0
print(f'  涨跌幅: {change_pct:+.2f}%')
print(f'  成交量: {row["volume"]:,.0f} 手  成交额: {row["amount"]/100000000:.2f} 亿')

# ========== 技能2: stock-analysis - 技术指标 ==========
print('\n' + '【二、技术指标分析】stock-analysis'.ljust(100, '-'))

bars = client.bars(symbol=code, market=market, frequency=9, start=0, offset=100)

df = pd.DataFrame({
    'close': bars['close'].values,
    'high': bars['high'].values,
    'low': bars['low'].values,
    'volume': bars['volume'].values
})

# MA均线
df['ma5'] = df['close'].rolling(5).mean()
df['ma10'] = df['close'].rolling(10).mean()
df['ma20'] = df['close'].rolling(20).mean()
df['ma60'] = df['close'].rolling(60).mean()

# MACD
df['ema12'] = df['close'].ewm(span=12).mean()
df['ema26'] = df['close'].ewm(span=26).mean()
df['dif'] = df['ema12'] - df['ema26']
df['dea'] = df['dif'].ewm(span=9).mean()
df['macd'] = (df['dif'] - df['dea']) * 2

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# BOLL
df['mid'] = df['close'].rolling(20).mean()
df['std'] = df['close'].rolling(20).std()
df['upper'] = df['mid'] + 2 * df['std']
df['lower'] = df['mid'] - 2 * df['std']

# KDJ
low_list = df['low'].rolling(9).min()
high_list = df['high'].rolling(9).max()
rsv = (df['close'] - low_list) / (high_list - low_list) * 100
df['k'] = rsv.ewm(com=2).mean()
df['d'] = df['k'].ewm(com=2).mean()
df['j'] = 3 * df['k'] - 2 * df['d']

latest = df.iloc[-1]
prev = df.iloc[-2]

print(f'  【均线系统】')
print(f'    MA5:   {latest["ma5"]:.2f}')
print(f'    MA10:  {latest["ma10"]:.2f}')
print(f'    MA20:  {latest["ma20"]:.2f}')
print(f'    MA60:  {latest["ma60"]:.2f}')
trend_status = "多头排列" if latest["ma5"]>latest["ma10"]>latest["ma20"] else "空头排列" if latest["ma5"]<latest["ma10"]<latest["ma20"] else "震荡整理"
print(f'    趋势:  {trend_status}')

print(f'  【MACD指标】')
print(f'    DIF:   {latest["dif"]:.2f}')
print(f'    DEA:   {latest["dea"]:.2f}')
print(f'    MACD:  {latest["macd"]:.2f}')
macd_signal = "金叉买入" if latest["dif"]>latest["dea"] and prev["dif"]<=prev["dea"] else "死叉卖出" if latest["dif"]<latest["dea"] and prev["dif"]>=prev["dea"] else "持仓观望"
print(f'    信号:  {macd_signal}')

print(f'  【RSI指标】')
print(f'    RSI:   {latest["rsi"]:.1f}')
rsi_status = "超买" if latest["rsi"]>70 else "超卖" if latest["rsi"]<30 else "中性"
print(f'    状态:  {rsi_status}')

print(f'  【KDJ指标】')
print(f'    K:     {latest["k"]:.1f}')
print(f'    D:     {latest["d"]:.1f}')
print(f'    J:     {latest["j"]:.1f}')

print(f'  【BOLL布林】')
print(f'    上轨:  {latest["upper"]:.2f}')
print(f'    中轨:  {latest["mid"]:.2f}')
print(f'    下轨:  {latest["lower"]:.2f}')
boll_pos = (latest['close'] - latest['lower']) / (latest['upper'] - latest['lower']) * 100
print(f'    位置:  {boll_pos:.1f}%')

# ========== 技能3: a-share-short-decision - 短期决策 ==========
print('\n' + '【三、短期决策信号】a-share-short-decision'.ljust(100, '-'))

# 各维度评分
if latest['ma5'] > latest['ma10'] > latest['ma20']:
    trend_score = 20
    trend_msg = '强势多头'
elif latest['ma5'] > latest['ma10']:
    trend_score = 10
    trend_msg = '短期向上'
elif latest['ma5'] < latest['ma10'] < latest['ma20']:
    trend_score = 0
    trend_msg = '强势空头'
else:
    trend_score = 5
    trend_msg = '震荡整理'

if latest['dif'] > latest['dea'] and latest['macd'] > 0:
    momentum_score = 20
    momentum_msg = '强势动能'
elif latest['dif'] > latest['dea']:
    momentum_score = 10
    momentum_msg = '金叉初期'
elif latest['dif'] < latest['dea'] and latest['macd'] < 0:
    momentum_score = 0
    momentum_msg = '弱势动能'
else:
    momentum_score = 5
    momentum_msg = '死叉初期'

if latest['rsi'] < 30:
    overbought_score = 15
    overbought_msg = '超卖反弹'
elif latest['rsi'] > 70:
    overbought_score = 0
    overbought_msg = '超买风险'
elif 40 <= latest['rsi'] <= 60:
    overbought_score = 10
    overbought_msg = '中性区域'
else:
    overbought_score = 5
    overbought_msg = '偏区边缘'

avg_vol = df['volume'].iloc[-10:].mean()
vol_ratio = latest['volume'] / avg_vol if avg_vol > 0 else 1
if vol_ratio > 2:
    volume_score = 15
    volume_msg = '巨量活跃'
elif vol_ratio > 1.5:
    volume_score = 10
    volume_msg = '放量'
elif vol_ratio > 0.8:
    volume_score = 5
    volume_msg = '平量'
else:
    volume_score = 0
    volume_msg = '缩量'

if boll_pos < 20:
    position_score = 15
    position_msg = '低位区'
elif boll_pos > 80:
    position_score = 0
    position_msg = '高位区'
elif 40 <= boll_pos <= 60:
    position_score = 10
    position_msg = '中位区'
else:
    position_score = 5
    position_msg = '偏位区'

if latest['k'] > latest['d'] and latest['k'] < 20:
    kdj_score = 15
    kdj_msg = '金叉超卖'
elif latest['k'] > latest['d'] and prev['k'] <= prev['d']:
    kdj_score = 10
    kdj_msg = '金叉信号'
elif latest['k'] < latest['d'] and latest['k'] > 80:
    kdj_score = 0
    kdj_msg = '死叉超买'
else:
    kdj_score = 5
    kdj_msg = '观望'

decision_score = trend_score + momentum_score + overbought_score + volume_score + position_score + kdj_score

print(f'  【六维评分】(满分100)')
print(f'    趋势:   {trend_score:2d}/20  {trend_msg}')
print(f'    动能:   {momentum_score:2d}/20  {momentum_msg}')
print(f'    超买超卖: {overbought_score:2d}/15  {overbought_msg}')
print(f'    量能:   {volume_score:2d}/15  {volume_msg}')
print(f'    位置:   {position_score:2d}/15  {position_msg}')
print(f'    KDJ:    {kdj_score:2d}/15  {kdj_msg}')
print(f'  ───────────────────────────────────')
print(f'    总分:   {decision_score:2d}/100')

if decision_score >= 75:
    decision_action = '强力买入'
    decision_emoji = '🟢🟢'
elif decision_score >= 60:
    decision_action = '买入'
    decision_emoji = '🟢'
elif decision_score >= 45:
    decision_action = '观望'
    decision_emoji = '🟡'
elif decision_score >= 30:
    decision_action = '谨慎'
    decision_emoji = '🟠'
else:
    decision_action = '规避'
    decision_emoji = '🔴'

print(f'  【决策建议】{decision_emoji} {decision_action}')

# ========== 技能4: chanlun - 缠论分析 ==========
print('\n' + '【四、缠论结构分析】chanlun'.ljust(100, '-'))

chanlun_data = []
for _, bar in bars.iterrows():
    chanlun_data.append({
        'date': str(bar['datetime']).split(' ')[0].replace('-', ''),
        'open': float(bar['open']),
        'high': float(bar['high']),
        'low': float(bar['low']),
        'close': float(bar['close']),
        'volume': int(bar['volume'])
    })

df_chanlun = pd.DataFrame(chanlun_data)
df_chanlun['datetime'] = pd.to_datetime(df_chanlun['date'], format='%Y%m%d')

analyzer = ChanLunAnalyzer(df_chanlun)
analyzer.analyze()

top_count = sum(1 for p in analyzer.points if p.f_type=='top')
bottom_count = sum(1 for p in analyzer.points if p.f_type=='bottom')

print(f'  【统计】')
print(f'    K线数量: {len(df_chanlun)}条')
print(f'    顶分型: {top_count}个')
print(f'    底分型: {bottom_count}个')
print(f'    笔: {len(analyzer.strokes)}笔')
print(f'    中枢: {len(analyzer.centers)}个')

if analyzer.centers:
    print(f'  【中枢分析】')
    for i, c in enumerate(analyzer.centers[-3:], 1):
        print(f'    中枢{i}: [{c.lower:.2f}, {c.upper:.2f}] @ {c.start_date.strftime("%Y-%m-%d")} ~ {c.end_date.strftime("%m-%d")}')

    latest_center = analyzer.centers[-1]
    current_price = latest['close']

    print(f'  【当前位置】')
    print(f'    最新中枢: [{latest_center.lower:.2f}, {latest_center.upper:.2f}]')
    print(f'    当前价格: {current_price:.2f}')

    if current_price > latest_center.upper:
        chanlun_pos = '中枢上方'
        chanlun_signal = '强势区域'
        chanlun_action = '可回调介入'
    elif current_price < latest_center.lower:
        chanlun_pos = '中枢下方'
        chanlun_signal = '弱势区域'
        chanlun_action = '等待确认'
    else:
        chanlun_pos = '中枢内部'
        chanlun_signal = '震荡区域'
        chanlun_action = '高抛低吸'

    print(f'    位置判断: {chanlun_pos} - {chanlun_signal}')
    print(f'    操作策略: {chanlun_action}')

if analyzer.points:
    latest_fractal = analyzer.points[-1]
    print(f'  【最新分型】')
    if latest_fractal.f_type == 'top':
        print(f'    顶分型 @ {latest_fractal.date.strftime("%Y-%m-%d")} 价格: {latest_fractal.price:.2f}')
    else:
        print(f'    底分型 @ {latest_fractal.date.strftime("%Y-%m-%d")} 价格: {latest_fractal.price:.2f}')

# ========== 技能5: stock-monitor-skill - 监控预警 ==========
print('\n' + '【五、监控预警分析】stock-monitor-skill'.ljust(100, '-'))

# 模拟持仓成本（假设）
mock_cost = 50.00
pnl_pct = (latest['close'] - mock_cost) / mock_cost * 100

print(f'  【模拟持仓分析】')
print(f'    假设成本: ¥{mock_cost:.2f}')
print(f'    当前价格: ¥{latest["close"]:.2f}')
print(f'    浮动盈亏: {pnl_pct:+.2f}%')

# 七大预警规则检查
alerts_triggered = []

# 1. 成本百分比
if pnl_pct >= 15:
    alerts_triggered.append(('🎯', '盈利15%+', '紧急'))
elif pnl_pct <= -12:
    alerts_triggered.append(('🎯', '亏损12%+', '紧急'))

# 2. 日内涨跌幅
if change_pct >= 4:
    alerts_triggered.append(('📈', '日内涨幅>4%', '警告'))
elif change_pct <= -4:
    alerts_triggered.append(('📉', '日内跌幅>4%', '警告'))

# 3. 成交量异动
if vol_ratio >= 2:
    alerts_triggered.append(('📊', f'放量{vol_ratio:.1f}倍', '提醒'))
elif vol_ratio <= 0.5:
    alerts_triggered.append(('📉', f'缩量{vol_ratio:.1f}倍', '提醒'))

# 4. 均线金叉死叉
if latest['ma5'] > latest['ma10'] and prev['ma5'] <= prev['ma10']:
    alerts_triggered.append(('🌟', 'MA5金叉MA10', '重要'))
elif latest['ma5'] < latest['ma10'] and prev['ma5'] >= prev['ma10']:
    alerts_triggered.append(('💀', 'MA5死叉MA10', '重要'))

# 5. RSI超买超卖
if latest['rsi'] >= 70:
    alerts_triggered.append(('🔥', 'RSI超买', '警告'))
elif latest['rsi'] <= 30:
    alerts_triggered.append(('❄️', 'RSI超卖', '警告'))

# 6. 跳空缺口（简化判断）
gap = bars['close'].iloc[-1] - bars['high'].iloc[-2]
gap_pct = gap / bars['high'].iloc[-2] * 100 if bars['high'].iloc[-2] > 0 else 0
if gap_pct >= 1:
    alerts_triggered.append(('⬆️', f'向上跳空{gap_pct:.1f}%', '提醒'))
elif gap_pct <= -1:
    alerts_triggered.append(('⬇️', f'向下跳空{abs(gap_pct):.1f}%', '提醒'))

print(f'  【预警规则检查】')
if alerts_triggered:
    for icon, alert, level in alerts_triggered:
        print(f'    {icon} {alert:<15} [{level}]')
else:
    print(f'    ✅ 无预警触发')

# 风险等级计算
risk_level = 0
if latest['rsi'] > 70:
    risk_level += 20
if boll_pos > 80:
    risk_level += 20
if change_pct > 5:
    risk_level += 15
if vol_ratio > 2 and latest['dif'] < latest['dea']:
    risk_level += 25  # 放量下跌

if risk_level >= 50:
    risk_status = '🔴 高风险'
elif risk_level >= 30:
    risk_status = '🟠 中风险'
elif risk_level >= 10:
    risk_status = '🟡 低风险'
else:
    risk_status = '🟢 安全'

print(f'  【风险评级】{risk_status} (得分: {risk_level}/100)')

# ========== 综合决策 ==========
print('\n' + '【六、五技能综合决策】'.ljust(100, '-'))

mootdx_score = 50 + int(change_pct) if change_pct > 0 else 50 + int(change_pct)
stock_score = decision_score
short_score = decision_score

if analyzer.centers:
    current_price = latest['close']
    latest_center = analyzer.centers[-1]
    if current_price > latest_center.upper:
        chanlun_score = 70
    elif current_price < latest_center.lower:
        chanlun_score = 30
    else:
        chanlun_score = 50
else:
    chanlun_score = 50

# monitor评分（根据风险等级反向）
monitor_score = max(0, 100 - risk_level)

final_scores = {
    'mootdx实时': mootdx_score,
    'stock技术': stock_score,
    'short决策': short_score,
    'chanlun': chanlun_score,
    'monitor': monitor_score
}

avg_score = sum(final_scores.values()) / len(final_scores)

print(f'  【各技能评分】')
for skill, score in final_scores.items():
    bar = '█' * int(score / 10)
    print(f'    {skill:<12}: {bar:<10} {score:.0f}/100')

print(f'  ──────────────────────────────────────────────────')
print(f'    综合评分: {"█" * int(avg_score / 10):<10} {avg_score:.0f}/100')

# 最终建议
if avg_score >= 70:
    final_action = '买入'
    final_emoji = '🟢🟢🟢'
    final_reason = '五技能共振向上'
elif avg_score >= 55:
    final_action = '偏多'
    final_emoji = '🟢🟢'
    final_reason = '多数指标支持'
elif avg_score >= 45:
    final_action = '观望'
    final_emoji = '🟡'
    final_reason = '多空分歧，等待确认'
elif avg_score >= 30:
    final_action = '偏空'
    final_emoji = '🟠'
    final_reason = '多数指标偏弱'
else:
    final_action = '规避'
    final_emoji = '🔴'
    final_reason = '五技能共振向下'

print(f'  【最终操作建议】')
print(f'    {final_emoji} {final_action}')
print(f'    理由: {final_reason}')

# 关键价位
print(f'  【关键价位】')
print(f'    压力位: {latest["upper"]:.2f} (BOLL上轨) | {latest["ma10"]:.2f} (MA10)')
print(f'    当前价: {latest["close"]:.2f}')
print(f'    支撑位: {latest["mid"]:.2f} (BOLL中轨) | {latest["lower"]:.2f} (BOLL下轨)')
if analyzer.centers:
    print(f'    中枢: [{latest_center.lower:.2f}, {latest_center.upper:.2f}]')

# 监控建议
print(f'  【监控建议】')
print(f'    建议设置预警:')
print(f'    • 价格预警: >{latest["upper"]:.2f} 或 <{latest["lower"]:.2f}')
print(f'    • RSI预警: >70 (超买) 或 <30 (超卖)')
print(f'    • 成交量: 放量>2倍注意')

print('\n' + '='*100)
