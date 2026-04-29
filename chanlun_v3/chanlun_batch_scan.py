# -*- coding: utf-8 -*-
"""
缠论批量扫描器 v2.0
- 全市场批量扫描（50-100只）
- V2.0仓位系统联动
- 6因子12分大盘评分
- 板块三档分类过滤
- 买点置信度 + 仓位计算

使用方式：
  AI通过tdx_kline获取数据 → 写入JSON → python chanlun_batch_scan.py input.json
"""
import sys
import json
import os
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chanlun_engine import ChanLunEngine, KLine


# ============================================================
# V2.0 大盘环境评分（6因子12分制）
# ============================================================

def calc_ma(prices, n):
    """计算简单均线"""
    if len(prices) < n:
        return None
    return sum(prices[-n:]) / n


def calc_ema(prices, n):
    """计算EMA"""
    if len(prices) < n:
        return None
    k = 2 / (n + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def calc_macd(closes, fast=12, slow=26, signal=9):
    """计算MACD"""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    dif = ema_fast - ema_slow
    # 简化：用DIF的EMA近似DEA
    return dif, None, None


def score_market_env(index_klines):
    """
    大盘环境评分（基于上证指数K线）
    
    6因子，每项0-2分，满分12分：
    1. 价格 vs MA5      （站上+2, 接近+1, 下方+0）
    2. 价格 vs MA10     （站上+2, 接近+1, 下方+0）
    3. 价格 vs MA20     （站上+2, 接近+1, 下方+0）
    4. MACD DIF 方向    （上穿+2, 走平+1, 下穿+0）
    5. 成交量 vs 5日均量（放量+2, 持平+1, 缩量+0）
    6. MA5 趋势方向     （向上+2, 走平+1, 向下+0）
    """
    if not index_klines or len(index_klines) < 25:
        return {"total": 6, "max": 12, "level": "中性", "details": ["数据不足，默认中性"], "factors": {}}
    
    closes = [k[4] for k in index_klines]
    volumes = [k[5] for k in index_klines]  # 用成交额
    price = closes[-1]
    
    score = 0
    details = []
    factors = {}
    
    # Factor 1-3: 价格 vs MA
    for ma_n, label in [(5, "MA5"), (10, "MA10"), (20, "MA20")]:
        ma_val = calc_ma(closes, ma_n)
        if ma_val is None:
            s, desc = 1, f"{label} 数据不足"
        elif price > ma_val * 1.01:
            s, desc = 2, f"站上{label}({ma_val:.2f})"
        elif price > ma_val * 0.99:
            s, desc = 1, f"接近{label}({ma_val:.2f})"
        else:
            s, desc = 0, f"低于{label}({ma_val:.2f})"
        score += s
        details.append(f"  [{label}] {desc} → {s}分")
        factors[label] = s
    
    # Factor 4: MACD DIF方向
    if len(closes) >= 30:
        dif_now, _, _ = calc_macd(closes[-30:])
        dif_prev, _, _ = calc_macd(closes[-31:-1])
        if dif_now is not None and dif_prev is not None:
            if dif_now > dif_prev + 0.5:
                s, desc = 2, "DIF上行"
            elif dif_now > dif_prev - 0.5:
                s, desc = 1, "DIF走平"
            else:
                s, desc = 0, "DIF下行"
        else:
            s, desc = 1, "DIF无法计算"
    else:
        s, desc = 1, "数据不足"
    score += s
    details.append(f"  [MACD] {desc} → {s}分")
    factors["MACD"] = s
    
    # Factor 5: 成交量
    if len(volumes) >= 6:
        avg_vol = sum(volumes[-6:-1]) / 5
        cur_vol = volumes[-1]
        if cur_vol > avg_vol * 1.2:
            s, desc = 2, "放量"
        elif cur_vol > avg_vol * 0.8:
            s, desc = 1, "持平"
        else:
            s, desc = 0, "缩量"
    else:
        s, desc = 1, "数据不足"
    score += s
    details.append(f"  [量能] {desc} → {s}分")
    factors["VOL"] = s
    
    # Factor 6: MA5趋势
    if len(closes) >= 10:
        ma5_now = calc_ma(closes[-5:], 5)
        ma5_5d = calc_ma(closes[-10:-5], 5)
        if ma5_now and ma5_5d:
            if ma5_now > ma5_5d * 1.005:
                s, desc = 2, "MA5向上"
            elif ma5_now > ma5_5d * 0.995:
                s, desc = 1, "MA5走平"
            else:
                s, desc = 0, "MA5向下"
        else:
            s, desc = 1, "无法计算"
    else:
        s, desc = 1, "数据不足"
    score += s
    details.append(f"  [趋势] {desc} → {s}分")
    factors["TREND"] = s
    
    level = "强势" if score >= 9 else "中性" if score >= 6 else "弱势"
    
    return {
        "total": score,
        "max": 12,
        "level": level,
        "details": details,
        "factors": factors
    }


# ============================================================
# V2.0 仓位计算系统
# ============================================================

def calc_position(buy_type, confidence, market_score, sector_tier="辅线"):
    """
    V2.0 仓位计算
    
    基础仓位：
      1买 = 20% (试探仓)
      2买 = 30% (确认仓) 
      3买 = 30% (效率仓)
      quasi2B = 15%
    
    调整因子：
      大盘评分 ≥9: ×1.0
      大盘评分 6-8: ×0.8
      大盘评分 0-5: ×0.3 且只允许1买
    
      置信度 ≥0.80: ×1.0
      置信度 0.65-0.79: ×0.8
      置信度 0.55-0.64: ×0.6
    
      主线板块: ×1.2
      辅线板块: ×1.0
      重灾区: ×0.5
    
    总仓位上限：
      强势: 80%
      中性: 60%
      弱势: 30%
    """
    # 基础仓位
    base = {"1B": 0.20, "sub1B": 0.12, "2B": 0.30, "3B": 0.30, "quasi2B": 0.15, "subQuasi2B": 0.08,
            "pz1B": 0.15, "2B3B": 0.35, "q2B": 0.18, "xzd1B": 0.18}.get(buy_type, 0.10)
    
    # 大盘调整
    mkt_level = market_score.get("level", "中性")
    mkt_factor = {"强势": 1.0, "中性": 0.8, "弱势": 0.3}.get(mkt_level, 0.8)
    
    # 弱势市只允许1买
    if mkt_level == "弱势" and buy_type != "1B":
        return 0.0, "弱势市不允许此类买点"
    
    # 置信度调整
    if confidence >= 0.80:
        conf_factor = 1.0
    elif confidence >= 0.65:
        conf_factor = 0.8
    elif confidence >= 0.55:
        conf_factor = 0.6
    else:
        return 0.0, "置信度不足"
    
    # 板块调整
    sector_factor = {"主线": 1.2, "辅线": 1.0, "重灾区": 0.5}.get(sector_tier, 1.0)
    
    # 最终仓位
    position = base * mkt_factor * conf_factor * sector_factor
    position = min(position, 0.40)  # 单只上限40%
    position = round(position, 3)
    
    # 总仓位上限
    max_total = {"强势": 0.80, "中性": 0.60, "弱势": 0.30}.get(mkt_level, 0.60)
    
    reason = f"基础{base:.0%} × 大盘{mkt_factor} × 置信{conf_factor} × 板块{sector_factor}"
    
    return position, reason


# ============================================================
# 单股扫描
# ============================================================

def scan_single_stock(code, name, klines_raw, market_score, sector_tier):
    """
    扫描单只股票
    
    klines_raw: [[date, open, high, low, close], ...]
    返回: 扫描结果dict 或 None
    """
    if len(klines_raw) < 30:
        return None
    
    klines = [KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4]) for d in klines_raw]
    
    engine = ChanLunEngine()
    result = engine.analyze(klines)
    
    pivots = result.get('pivots', [])
    buy_points = result.get('buy_sell_points', [])
    strokes = result.get('strokes', [])
    multi_level = result.get('multi_level', None)  # P1: 区间套分析结果
    
    # 找活跃中枢（最后一个非upgraded）
    active_pivot = None
    for p in reversed(pivots):
        if p['evolution'] != 'upgraded':
            active_pivot = p
            break
    
    # 筛选高质量买点
    quality_buys = [bp for bp in buy_points 
                    if bp['type'] in ('1B', '2B', '3B', 'quasi2B', 'sub1B', 'subQuasi2B', 'pz1B', '2B3B', 'q2B', 'xzd1B') 
                    and bp['confidence'] >= 0.50]
    
    # 时效过滤：只保留最近20个交易日内的买点信号
    from datetime import datetime as _dt
    
    def parse_to_date(d):
        """将各种日期格式转为datetime"""
        s = str(d).replace('-', '').replace('/', '')
        return _dt.strptime(s, '%Y%m%d')
    
    try:
        last_dt = parse_to_date(klines[-1].date)
        max_age_days = 30  # 30个自然日 ≈ 20个交易日
    except:
        last_dt = None
        max_age_days = 999
    
    if last_dt:
        quality_buys_fresh = []
        for bp in quality_buys:
            try:
                buy_dt = parse_to_date(bp['date'])
                age = (last_dt - buy_dt).days
                if age <= max_age_days:
                    quality_buys_fresh.append(bp)
            except:
                quality_buys_fresh.append(bp)  # 解析失败的保留
        
        # 直接使用时效过滤后的结果，不做fallback
        quality_buys = quality_buys_fresh
    
    if not quality_buys:
        return None
    
    # 优先选本级别信号，其次选次级别
    level_priority = {'3B': 0, '2B3B': 1, '2B': 2, '1B': 3, 'pz1B': 4, 'xzd1B': 5, 'sub1B': 6, 'q2B': 7, 'quasi2B': 8, 'subQuasi2B': 9}
    quality_buys.sort(key=lambda x: (level_priority.get(x['type'], 9), -x['confidence']))
    
    # 取最优买点
    best_buy = quality_buys[0]
    
    # 当前价格
    last_close = klines[-1].close
    last_date = klines[-1].date
    
    # 趋势判断
    last_stroke = strokes[-1] if strokes else None
    if last_stroke:
        trend = "↑上升笔" if last_stroke['type'] == 'up' else "↓下降笔"
        trend_amp = last_stroke['amplitude']
    else:
        trend = "无"
        trend_amp = 0
    
    # 计算价位距离
    price_distance = (last_close - best_buy['price']) / best_buy['price'] * 100
    
    # 计算仓位
    position, pos_reason = calc_position(
        best_buy['type'], best_buy['confidence'], market_score, sector_tier
    )
    
    # P1: 区间套修正 — 用 multi_level.final_modifier 调整仓位
    ml_info = ""
    if multi_level:
        ml_mod = multi_level['final_modifier']
        if ml_mod != 1.0:
            old_pos = position
            position = round(position * ml_mod, 2)
            ml_info = f" × 区间套{ml_mod:.2f}"
            pos_reason += ml_info
        # 记录区间套详细信息
        ml_info = (f"日线位置={multi_level['daily_modifier']:.2f} "
                   f"趋势={multi_level['trend_modifier']:.2f} "
                   f"综合={multi_level['final_modifier']:.2f}")
    
    if position <= 0:
        return None
    
    return {
        "code": code,
        "name": name,
        "close": round(last_close, 2),
        "date": last_date,
        "trend": trend,
        "trend_amp": round(trend_amp, 2),
        "buy_type": best_buy['type'],
        "buy_date": best_buy['date'],
        "buy_price": round(best_buy['price'], 2),
        "confidence": round(best_buy['confidence'], 2),
        "stop_loss": round(best_buy['stop_loss'], 2),
        "buy_reason": best_buy['reason'],
        "price_distance": round(price_distance, 2),
        "position": position,
        "pos_reason": pos_reason,
        "pivot_info": f"ZG={active_pivot['zg']} ZD={active_pivot['zd']} N={active_pivot['segment_count']}" if active_pivot else "无活跃中枢",
        "pivot_evolution": active_pivot['evolution'] if active_pivot else "",
        "all_buys_count": len(quality_buys),
        "strokes_count": len(strokes),
        "pivots_count": len(pivots),
        "multi_level_info": ml_info,
    }


# ============================================================
# 批量扫描主函数
# ============================================================

def batch_scan(input_data):
    """
    批量扫描入口
    
    input_data格式:
    {
        "index_klines": [[date, open, high, low, close, amount], ...],  // 上证指数
        "stocks": {
            "600519": {
                "name": "贵州茅台",
                "sector_tier": "主线",  // 主线/辅线/重灾区
                "klines": [[date, open, high, low, close], ...]
            },
            ...
        }
    }
    """
    # 1. 计算大盘评分
    market_score = score_market_env(input_data.get("index_klines", []))
    
    # 2. 批量扫描
    results = []
    for code, info in input_data.get("stocks", {}).items():
        try:
            r = scan_single_stock(
                code, 
                info.get("name", code),
                info.get("klines", []),
                market_score,
                info.get("sector_tier", "辅线")
            )
            if r:
                results.append(r)
        except Exception as e:
            pass  # 跳过分析失败的
    
    # 3. 排序
    # 优先级：3买 > 2买 > 1买 > quasi2B，同类型按置信度降序
    type_priority = {'3B': 0, '2B3B': 1, '2B': 2, '1B': 3, 'pz1B': 4, 'xzd1B': 5, 'xzd1B': '小转大(xzd1B)', 'sub1B': 6, 'q2B': 7, 'quasi2B': 8, 'subQuasi2B': 9}
    results.sort(key=lambda x: (type_priority.get(x['buy_type'], 9), -x['confidence']))
    
    # 4. 计算总仓位
    max_total = {"强势": 0.80, "中性": 0.60, "弱势": 0.30}.get(market_score['level'], 0.60)
    
    # 按优先级分配总仓位
    allocated = 0
    for r in results:
        if allocated + r['position'] <= max_total:
            allocated += r['position']
            r['allocated'] = True
        else:
            remaining = max_total - allocated
            if remaining > 0.05:
                r['position'] = round(remaining, 3)
                r['allocated'] = True
                allocated = max_total
            else:
                r['position'] = 0
                r['allocated'] = False
    
    return {
        "market_score": market_score,
        "max_total_position": round(max_total, 2),
        "allocated_position": round(allocated, 3),
        "candidates": results,
        "total_scanned": len(input_data.get("stocks", {})),
        "total_hits": len(results),
    }


# ============================================================
# 报告生成
# ============================================================

def generate_report(scan_result):
    """生成文本报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    
    ms = scan_result['market_score']
    
    lines.append("=" * 62)
    lines.append(f"缠论批量扫描报告 | {now}")
    lines.append("=" * 62)
    
    # 大盘环境
    lines.append(f"\n[大盘环境] {ms['total']}/{ms['max']}分 | {ms['level']}")
    for d in ms['details']:
        lines.append(d)
    
    # 仓位上限
    lines.append(f"\n[仓位上限] 总仓位上限 {scan_result['max_total_position']:.0%} | 已分配 {scan_result['allocated_position']:.1%}")
    
    # 扫描概况
    lines.append(f"\n[扫描概况] 共扫 {scan_result['total_scanned']} 只 | 命中 {scan_result['total_hits']} 只")
    
    if not scan_result['candidates']:
        lines.append("\n  无符合条件的标的")
    else:
        # 按买点类型分组
        groups = {}
        for c in scan_result['candidates']:
            groups.setdefault(c['buy_type'], []).append(c)
        
        type_names = {'3B': '三买(3B)', '2B3B': '二买三买重合(2B3B)', '2B': '二买(2B)', '1B': '一买(1B)', 'pz1B': '盘整背驰(pz1B)', 'xzd1B': '小转大(xzd1B)', 'sub1B': '次级别一买(sub1B)', 'q2B': '类二买确认(q2B)', 'quasi2B': '类二买(quasi2B)', 'subQuasi2B': '次级别类二买(subQuasi2B)'}
        
        for bt in ['3B', '2B3B', '2B', '1B', 'pz1B', 'xzd1B', 'sub1B', 'q2B', 'quasi2B', 'subQuasi2B']:
            if bt not in groups:
                continue
            group = groups[bt]
            lines.append(f"\n{'─' * 62}")
            lines.append(f"  {type_names.get(bt, bt)} | {len(group)}只")
            lines.append(f"{'─' * 62}")
            
            for i, c in enumerate(group, 1):
                # 标记是否已分配仓位
                alloc_tag = "" if c.get('allocated', False) else " [超出总仓位]"
                dist_tag = f"距买点{'+'if c['price_distance']>=0 else ''}{c['price_distance']:.1f}%"
                
                lines.append(f"\n  {i}. {c['name']}({c['code']}) | {c['close']} | {c['trend']}({c['trend_amp']})")
                lines.append(f"     {type_names.get(bt,bt)} {c['buy_date']} @ {c['buy_price']} | 置信度 {c['confidence']}")
                lines.append(f"     止损 {c['stop_loss']} | {dist_tag}{alloc_tag}")
                lines.append(f"     建议仓位 {c['position']:.1%} | {c['pos_reason']}")
                lines.append(f"     中枢 {c['pivot_info']} ({c['pivot_evolution']})")
                lines.append(f"     买点原因: {c['buy_reason']}")
    
    # 风险提示
    lines.append(f"\n{'=' * 62}")
    lines.append("[风险提示]")
    lines.append("  1. 以上分析基于缠论结构识别，不构成投资建议")
    lines.append("  2. 仓位为建议值，请根据个人风险承受能力调整")
    lines.append(f"  3. 当前大盘环境: {ms['level']}，总仓位上限 {scan_result['max_total_position']:.0%}")
    if ms['level'] == '弱势':
        lines.append("  4. 弱势环境仅允许一买(1B)，仓位严格控制30%以内")
    lines.append("=" * 62)
    
    return "\n".join(lines)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python chanlun_batch_scan.py <input.json> [output.json]")
        print()
        print("input.json格式:")
        print(json.dumps({
            "index_klines": [["日期","O","H","L","C","成交额"], ...],
            "stocks": {
                "600519": {"name":"贵州茅台","sector_tier":"主线","klines":[["日期","O","H","L","C"],...]},
            }
        }, ensure_ascii=False, indent=2))
        sys.exit(1)
    
    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    result = batch_scan(input_data)
    report = generate_report(result)
    print(report)
    
    # 可选保存JSON结果
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nJSON结果已保存: {output_file}")
