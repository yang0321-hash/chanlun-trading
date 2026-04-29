"""
多级别联立分析器（缠论区间套）
周线判断方向 → 日线找结构 → 30分钟找买点

数据流：
1. 日K → 合成周K → 周线缠论分析（方向）
2. 日K → 日线缠论分析（中枢、三买候选）
3. API取30分钟K → 30分钟缠论分析（精确买点）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from typing import List, Dict, Optional, Tuple
from chanlun_engine import ChanLunEngine, KLine, TrendDivergenceDetector, MACDDivergence


def _engine_with_divergence(klines):
    """对K线做缠论分析+趋势背驰检测，返回(engine, result, trend)"""
    engine = ChanLunEngine()
    result = engine.analyze(klines)
    trend = TrendDivergenceDetector.detect(engine)
    return engine, result, trend


class M30EntrySignal:
    """30分钟精确入场信号
    
    规则：
    1. 30分钟下跌笔出现背驰（MACD面积衰减）
    2. 入场确认（二选一）：
       A. 强分型确认：下跌笔终点是强底分型 → 直接入场
       B. 二次分型确认：标准/弱分型 → 等待反弹UP笔+回调DN笔不破前低 → 确认入场
    """
    
    # 入场状态
    NONE = "none"               # 无信号
    DIVERGENCE_DETECTED = "divergence"  # 背驰已确认，等分型确认
    STRONG_ENTRY = "strong_entry"      # 强分型入场 ✅
    DOUBLE_FRACTAL = "double_entry"    # 二次分型入场 ✅
    NO_ENTRY = "no_entry"              # 条件不满足
    
    @staticmethod
    def detect(m30_engine, m30_klines, m30_buy_points=None):
        """检测30分钟精确入场信号
        
        缠论原文：大级别的买点必然是次级别的第一类买点。
        因此优先检查30分钟引擎是否检测到1B/2B买点。
        
        优先级：
        1. 30分钟有1B/2B买点 → 最强入场（原文要求）
        2. 背驰+强分型 → 强入场
        3. 背驰+二次分型确认 → 确认入场
        4. 无背驰+强分型 → 谨慎入场
        5. 其他 → 不入场
        
        Args:
            m30_engine: 30分钟缠论引擎实例
            m30_klines: 30分钟K线数据
            m30_buy_points: 30分钟引擎检测到的买点列表（可选）
        
        Returns:
            {
                'status': 状态,
                'entry_price': 入场价格（如有）,
                'stop_loss': 止损价格（如有）,
                'detail': 文字描述,
                'divergence_info': 背驰详情,
                'fractal_info': 分型详情,
                'sub_level_signal': 次级别买点类型（如有）,
            }
        """
        result = {
            'status': M30EntrySignal.NONE,
            'entry_price': None,
            'stop_loss': None,
            'detail': '',
            'divergence_info': None,
            'fractal_info': None,
            'sub_level_signal': None,
        }
        
        strokes = m30_engine.strokes
        if len(strokes) < 3:
            result['detail'] = '30分钟笔数不足({})'.format(len(strokes))
            return result
        
        # ========== Step 0: 检查30分钟引擎是否有1B/2B买点 ==========
        # 缠论原文：大级别买点 = 次级别1B
        # 这是最高优先级的入场确认
        if m30_buy_points:
            # 筛选最近的（最后一笔范围内）1B/2B买点
            recent_buys = []
            last_stroke_end = strokes[-1].end_date
            for b in m30_buy_points:
                if b['type'] in ('1B', '2B', 'sub1B'):
                    # 买点在最后3笔范围内 = 近期信号
                    if len(strokes) >= 3:
                        start_date = strokes[-3].start_date
                    else:
                        start_date = strokes[0].start_date
                    if b['date'] >= start_date:
                        recent_buys.append(b)
            
            if recent_buys:
                # 取最优的（优先1B>2B>sub1B，置信度最高的）
                type_priority = {'1B': 3, '2B': 2, 'sub1B': 1}
                best_sub = max(recent_buys, 
                              key=lambda x: (type_priority.get(x['type'], 0), x.get('confidence', 0)))
                
                entry_price = best_sub['price']
                stop_loss = best_sub.get('stop_loss', entry_price * 0.99)
                sig_type = best_sub['type']
                
                # 如果是1B（趋势背驰）= 最强确认
                if sig_type == '1B':
                    detail = '🎯 次级别1B确认入场 @{:.2f} (趋势背驰, 原文要求)'.format(entry_price)
                    status = 'strong_entry'
                elif sig_type == '2B':
                    detail = '🎯 次级别2B确认入场 @{:.2f} (回调不破前低)'.format(entry_price)
                    status = 'strong_entry'
                else:  # sub1B
                    detail = '✅ 次级别sub1B入场 @{:.2f} (盘整背驰)'.format(entry_price)
                    status = 'double_entry'  # 略低于1B/2B
                
                result.update({
                    'status': status,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'detail': detail,
                    'sub_level_signal': sig_type,
                })
                return result
        
        # ========== Step 1: 找最后下跌笔，检查背驰 ==========
        last_down = None
        prev_down = None
        last_up = None
        
        # 从后往前找
        for s in reversed(strokes):
            if last_down is None and s.stype.name == 'DOWN':
                last_down = s
            elif last_down is not None and prev_down is None and s.stype.name == 'DOWN':
                prev_down = s
                break
        
        # 找最后的UP笔
        for s in reversed(strokes):
            if s.stype.name == 'UP':
                last_up = s
                break
        
        if last_down is None:
            result['detail'] = '30分钟无下跌笔'
            return result
        
        # 背驰检测
        divergence_found = False
        div_info = {}
        
        if prev_down is not None:
            div = MACDDivergence.check_divergence(last_down, prev_down, m30_klines)
            area_ratio = div.get('area_ratio', 1.0)
            divergence_found = div.get('divergence', False)
            div_info = {
                'area_ratio': area_ratio,
                'diverged': divergence_found,
                'last_amp': last_down.amplitude,
                'prev_amp': prev_down.amplitude,
            }
        else:
            # 只有一笔下跌，无法比较
            div_info = {'area_ratio': None, 'diverged': False, 'note': '仅一笔下跌'}
        
        result['divergence_info'] = div_info
        
        # ========== Step 2: 检查当前是否在下跌笔中（最后一笔是DOWN） ==========
        is_in_down = (strokes[-1].stype.name == 'DOWN')
        is_after_down = (last_up is not None and last_down is not None and 
                        last_up.index < last_down.index)
        
        if not is_in_down and not is_after_down:
            # 最后一笔是UP，且没有最近的下跌笔可确认
            result['detail'] = '30分钟最后一笔↑，下跌确认尚未出现'
            result['status'] = M30EntrySignal.NO_ENTRY
            return result
        
        # 下跌笔的端点分型强度
        end_strength = last_down.end_strength
        end_score = last_down.strength_score
        
        result['fractal_info'] = {
            'strength': end_strength,
            'score': end_score,
            'end_price': last_down.end_value,
            'end_date': last_down.end_date,
        }
        
        # ========== Step 3A: 强分型入场（无次级别1B/2B时的备选） ==========
        # 注意：到这里说明30分钟引擎未检测到1B/2B买点
        # 缠论原文：大级别买点=次级别1B，所以这里只是"近似入场"
        if end_strength == 'strong':
            entry_price = last_down.end_value
            stop_loss = entry_price * 0.99  # 跌破买入价1%止损
            
            if divergence_found:
                detail = '✅ 背驰+强分型 @{:.2f} (无次级别1B/2B, 非原文严格入场)'.format(entry_price)
                detail += ' MACD衰减{:.2f}'.format(div_info.get('area_ratio', 0))
            else:
                detail = '⚠️ 仅强分型 @{:.2f} (无背驰无次级别1B/2B)'.format(entry_price)
            
            result.update({
                'status': M30EntrySignal.STRONG_ENTRY,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'detail': detail,
            })
            return result
        
        # ========== Step 3B: 二次分型确认（无次级别1B/2B的备选） ==========
        # 标准/弱分型：等待反弹UP笔+回调DN笔
        # 条件：
        #   1. 最后下跌笔之后有反弹UP笔
        #   2. 反弹UP笔之后有回调DN笔
        #   3. 回调DN笔低点 >= 最后下跌笔低点（不破前低）
        
        # 检查最后一笔是否是UP（反弹中）
        last_stroke = strokes[-1]
        second_last = strokes[-2] if len(strokes) >= 2 else None
        third_last = strokes[-3] if len(strokes) >= 3 else None
        
        if last_stroke.stype.name == 'UP':
            # 反弹UP笔进行中
            # 找到这个UP笔之前的DN笔（就是last_down或更近的）
            down_before_up = second_last if second_last and second_last.stype.name == 'DOWN' else None
            up_start_price = down_before_up.end_value if down_before_up else last_down.end_value
            
            rebound_high = last_stroke.high
            detail = '等待二次确认: 反弹UP进行中 高={:.2f} 起点低={:.2f}'.format(
                rebound_high, up_start_price)
            
            if divergence_found:
                detail += ' (背驰已确认)'
            
            result.update({
                'status': M30EntrySignal.DIVERGENCE_DETECTED,
                'detail': detail,
            })
            return result
        
        if last_stroke.stype.name == 'DOWN' and second_last and second_last.stype.name == 'UP':
            # 已有反弹UP + 回调DN
            rebound_up = second_last
            pullback_dn = last_stroke
            
            first_low = last_down.end_value if last_down.index < rebound_up.index else pullback_dn.end_value
            pullback_low = pullback_dn.end_value
            
            # 二次确认：回调不破前低
            if pullback_low >= first_low * 0.998:  # 允许0.2%的误差
                entry_price = pullback_low
                stop_loss = first_low * 0.99  # 止损=前低
                
                detail = '✅ 二次分型确认入场 @{:.2f}'.format(entry_price)
                detail += ' 前低={:.2f} 回调低={:.2f}'.format(first_low, pullback_low)
                if divergence_found:
                    detail += ' (背驰确认)'
                
                result.update({
                    'status': M30EntrySignal.DOUBLE_FRACTAL,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'detail': detail,
                })
                return result
            else:
                detail = '❌ 二次确认失败: 回调破前低 ({:.2f} < {:.2f})'.format(
                    pullback_low, first_low)
                result.update({
                    'status': M30EntrySignal.NO_ENTRY,
                    'detail': detail,
                })
                return result
        
        # 其他情况：下跌中，等反弹
        if last_stroke.stype.name == 'DOWN':
            detail = '等待反弹: 下跌笔进行中 低={:.2f}'.format(last_stroke.end_value)
            if end_strength == 'strong':
                detail += ' 强分型!'
            if divergence_found:
                detail += ' 背驰确认'
            detail += ' → 等反弹UP确认'
            result.update({
                'status': M30EntrySignal.DIVERGENCE_DETECTED,
                'detail': detail,
            })
            return result
        
        result['detail'] = '30分钟状态不明确'
        return result


def daily_to_weekly(daily_klines: List[KLine]) -> List[KLine]:
    """日线K线合成周线
    
    合并规则：同一周的日K合并为一根周K
    周的定义：周一到周五（自然周）
    """
    if not daily_klines:
        return []
    
    weekly = []
    current_week = None
    week_open = week_high = week_low = week_close = None
    
    for k in daily_klines:
        # 解析日期获取周几
        parts = k.date.split('-') if '-' in k.date else [k.date[:4], k.date[4:6], k.date[6:8]]
        if '-' in k.date:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        
        import datetime
        try:
            dt = datetime.date(y, m, d)
        except ValueError:
            continue
        week_num = dt.isocalendar()[1]  # ISO周数
        year_week = (y, week_num)
        
        if year_week != current_week:
            # 新的一周，保存上一周
            if current_week is not None:
                weekly.append(KLine(
                    date=f"{week_start}",
                    open=week_open,
                    high=week_high,
                    low=week_low,
                    close=week_close
                ))
            current_week = year_week
            week_start = k.date
            week_open = k.open
            week_high = k.high
            week_low = k.low
            week_close = k.close
        else:
            week_high = max(week_high, k.high)
            week_low = min(week_low, k.low)
            week_close = k.close
    
    # 最后一周
    if current_week is not None:
        weekly.append(KLine(
            date=f"{week_start}",
            open=week_open,
            high=week_high,
            low=week_low,
            close=week_close
        ))
    
    return weekly


def weekly_direction(weekly_klines: List[KLine]) -> Dict:
    """周线方向判断
    
    返回：
    - direction: 'up' / 'down' / 'consolidation'
    - trend_type: 'trend_up' / 'trend_down' / 'consolidation'
    - pivots: 周线中枢列表
    - last_pivot: 最后一个中枢
    - position: 当前价格相对中枢的位置描述
    """
    if len(weekly_klines) < 10:
        return {'direction': 'unknown', 'trend_type': 'unknown', 'confidence': 0}
    
    engine = ChanLunEngine()
    result = engine.analyze(weekly_klines)
    pivots = result['pivots']
    strokes = result['strokes']
    buys = result.get('buy_sell_points', [])
    sells = result.get('buy_sell_points', [])
    
    if not pivots:
        # 无中枢 → 单边走势
        if strokes and len(strokes) >= 2:
            last = strokes[-1]
            if last.get('type') == 'up':
                return {'direction': 'up', 'trend_type': 'no_pivot_up', 'confidence': 0.5,
                        'pivots': [], 'last_pivot': None, 'detail': '周线无中枢，单边上涨'}
            else:
                return {'direction': 'down', 'trend_type': 'no_pivot_down', 'confidence': 0.5,
                        'pivots': [], 'last_pivot': None, 'detail': '周线无中枢，单边下跌'}
        return {'direction': 'unknown', 'trend_type': 'unknown', 'confidence': 0}
    
    # 统计有效中枢（非sub_level）
    valid_pivots = [p for p in pivots if not p.get('sub_level', False)]
    
    if len(valid_pivots) < 2:
        # 只有1个有效中枢 → 盘整
        p = valid_pivots[-1] if valid_pivots else pivots[-1]
        last_close = weekly_klines[-1].close
        
        if last_close > p.get('zg', 0):
            pos = '中枢上方'
        elif last_close < p.get('zd', 0):
            pos = '中枢下方'
        else:
            pos = '中枢内部'
        
        return {
            'direction': 'consolidation',
            'trend_type': 'consolidation',
            'confidence': 0.6,
            'pivots': valid_pivots,
            'last_pivot': p,
            'position': pos,
            'detail': f'周线盘整(1个中枢) ZG={p.get("zg",0):.2f} ZD={p.get("zd",0):.2f} 当前{pos}'
        }
    
    # 2个以上有效中枢 → 趋势
    p1 = valid_pivots[-2]
    p2 = valid_pivots[-1]
    
    if p2.get('zd', 0) > p1.get('zg', 0):
        # 中枢上移 → 上涨趋势
        return {
            'direction': 'up',
            'trend_type': 'trend_up',
            'confidence': 0.9,
            'pivots': valid_pivots,
            'last_pivot': p2,
            'detail': f'周线上涨趋势 P1.ZG={p1.get("zg",0):.2f} < P2.ZD={p2.get("zd",0):.2f}'
        }
    elif p2.get('zg', 0) < p1.get('zd', 0):
        # 中枢下移 → 下跌趋势
        return {
            'direction': 'down',
            'trend_type': 'trend_down',
            'confidence': 0.9,
            'pivots': valid_pivots,
            'last_pivot': p2,
            'detail': f'周线下跌趋势 P1.ZD={p1.get("zd",0):.2f} > P2.ZG={p2.get("zg",0):.2f}'
        }
    else:
        # 中枢有重叠 → 盘整（大级别）
        return {
            'direction': 'consolidation',
            'trend_type': 'large_consolidation',
            'confidence': 0.7,
            'pivots': valid_pivots,
            'last_pivot': p2,
            'detail': f'周线大级别盘整 P1和P2有重叠'
        }


def weekly_direction_ggdd(weekly_klines: List[KLine]) -> Dict:
    """方法A：用GG/DD判断周线方向（AB测试验证更优）
    用中枢的GG/DD代替ZG/ZD，更贴近手动画法的震荡区间
    """
    if len(weekly_klines) < 10:
        return {'direction': 'unknown', 'trend_type': 'unknown', 'confidence': 0}
    
    engine = ChanLunEngine()
    result = engine.analyze(weekly_klines)
    pivots = result['pivots']
    strokes = result['strokes']
    
    if not pivots:
        if strokes and len(strokes) >= 2:
            last = strokes[-1]
            d = 'up' if last.get('type') == 'up' else 'down'
            return {'direction': d, 'trend_type': f'no_pivot_{d}', 'confidence': 0.5,
                    'pivots': [], 'last_pivot': None, 'detail': f'周线无中枢，单边{d}'}
        return {'direction': 'unknown', 'trend_type': 'unknown', 'confidence': 0}
    
    valid_pivots = [p for p in pivots if not p.get('sub_level', False)]
    last_close = weekly_klines[-1].close
    p = valid_pivots[-1] if valid_pivots else pivots[-1]
    gg = p.get('gg', 0)
    dd = p.get('dd', 0)
    
    if len(valid_pivots) >= 2:
        p1 = valid_pivots[-2]
        p2 = valid_pivots[-1]
        if p2.get('dd', 0) > p1.get('gg', 0):
            return {
                'direction': 'up', 'trend_type': 'trend_up', 'confidence': 0.9,
                'pivots': valid_pivots, 'last_pivot': p2,
                'detail': f'周线上涨趋势(DD上移) P1.GG={p1.get("gg",0):.2f} < P2.DD={p2.get("dd",0):.2f}'
            }
        elif p2.get('gg', 0) < p1.get('dd', 0):
            return {
                'direction': 'down', 'trend_type': 'trend_down', 'confidence': 0.9,
                'pivots': valid_pivots, 'last_pivot': p2,
                'detail': f'周线下跌趋势(GG下移) P1.DD={p1.get("dd",0):.2f} > P2.GG={p2.get("gg",0):.2f}'
            }
    
    # 单中枢或中枢重叠 → 看价格在GG/DD的位置
    if last_close > gg:
        pos = 'GG上方'
    elif last_close < dd:
        pos = 'DD下方'
    else:
        pos = 'GG-DD区间内'
    
    direction = 'consolidation'
    if last_close > gg:
        direction = 'up'  # 价格在GG上方偏多
    elif last_close < dd:
        direction = 'down'
    
    return {
        'direction': direction, 'trend_type': 'consolidation', 'confidence': 0.6,
        'pivots': valid_pivots, 'last_pivot': p, 'position': pos,
        'detail': f'周线盘整 GG={gg:.2f} DD={dd:.2f} 当前{pos}'
    }


def weekly_position_analysis(daily_klines: List[KLine]) -> Dict:
    """周线位置全景分析：看更多K线，判断股价在周线级别所处的位置
    
    返回：
    - weekly_klines: 周线K线数据
    - weekly_pivots: 周线中枢列表
    - weekly_buy_sell: 周线买卖点
    - position_desc: 位置描述文字
    - trend: 周线趋势
    """
    weekly = daily_to_weekly(daily_klines)
    if len(weekly) < 10:
        return {'position_desc': '周线数据不足', 'trend': 'unknown'}
    
    engine = ChanLunEngine()
    result = engine.analyze(weekly)
    pivots = result['pivots']
    strokes = result['strokes']
    buys = result.get('buy_sell_points', [])
    
    last_close = weekly[-1].close
    last_date = weekly[-1].date
    
    # 构建位置描述
    lines = []
    lines.append('周线K线数: {}'.format(len(weekly)))
    lines.append('周线笔数: {} 中枢数: {}'.format(len(strokes), len(pivots)))
    lines.append('')
    
    # 周线中枢详情
    if pivots:
        lines.append('周线中枢:')
        for pv in pivots:
            sub = '次' if pv.get('sub_level', True) else '本'
            seg = pv.get('segment_count', len(pv.get('strokes', [])))
            evo = pv.get('evolution', '')[:4]
            lines.append('  P#{} ZG={:.2f} ZD={:.2f} GG={:.2f} DD={:.2f} {}seg {} {}'.format(
                pv.get('index',0), pv.get('zg',0), pv.get('zd',0),
                pv.get('gg',0), pv.get('dd',0), seg, sub, evo))
    
    # 当前价格位置
    lines.append('')
    lines.append('当前价格: {:.2f} ({})'.format(last_close, last_date))
    
    if pivots:
        last_p = pivots[-1]
        zg = last_p.get('zg', 0)
        zd = last_p.get('zd', 0)
        gg = last_p.get('gg', 0)
        dd = last_p.get('dd', 0)
        
        if last_close > gg:
            pos = 'GG上方(强势)'
        elif last_close > zg:
            pos = 'ZG~GG区间(中枢上方偏强)'
        elif last_close > zd:
            pos = 'ZD~ZG区间(中枢内部)'
        elif last_close > dd:
            pos = 'DD~ZD区间(中枢下方偏弱)'
        else:
            pos = 'DD下方(弱势)'
        
        lines.append('相对最后中枢: {}'.format(pos))
        lines.append('  ZG={:.2f} ZD={:.2f} GG={:.2f} DD={:.2f}'.format(zg, zd, gg, dd))
        lines.append('  距ZG: {:+.1f}%  距GG: {:+.1f}%'.format(
            (last_close-zg)/zg*100, (last_close-gg)/gg*100))
    
    # 周线买卖点
    buy_types = [b for b in buys if 'B' in b.get('type','') and 'S' not in b.get('type','')]
    sell_types = [b for b in buys if 'S' in b.get('type','')]
    
    if buy_types:
        lines.append('')
        lines.append('周线买点:')
        for b in buy_types:
            lines.append('  {} {} @{:.2f} conf={:.2f}'.format(
                b['type'], b['date'], b['price'], b['confidence']))
    
    if sell_types:
        lines.append('')
        lines.append('周线卖点:')
        for b in sell_types:
            lines.append('  {} {} @{:.2f} conf={:.2f}'.format(
                b['type'], b['date'], b['price'], b['confidence']))
    
    # 最后几笔方向
    if strokes:
        last_stroke = strokes[-1]
        d = '↑' if last_stroke.get('type') == 'up' else '↓'
        lines.append('')
        lines.append('周线最后笔: {} {} '.format(d, last_stroke.get('end','')))
        if len(strokes) >= 2:
            prev = strokes[-2]
            pd = '↑' if prev.get('type') == 'up' else '↓'
            lines.append('周线倒数第2笔: {} {}'.format(pd, prev.get('end','')))
    
    # 趋势判断
    valid_pivots = [p for p in pivots if not p.get('sub_level', True)]
    trend = '盘整'
    if len(valid_pivots) >= 2:
        p1, p2 = valid_pivots[-2], valid_pivots[-1]
        if p2.get('dd', 0) > p1.get('gg', 0):
            trend = '上涨趋势'
        elif p2.get('gg', 0) < p1.get('dd', 0):
            trend = '下跌趋势'
    
    return {
        'weekly_klines': weekly,
        'weekly_pivots': pivots,
        'weekly_strokes': strokes,
        'weekly_buy_sell': buys,
        'position_desc': '\n'.join(lines),
        'trend': trend,
        'direction': 'up' if '上涨' in trend else ('down' if '下跌' in trend else 'consolidation'),
        'last_close': last_close,
    }


def analyze_30min(klines_30min: List[KLine]) -> Dict:
    """30分钟缠论分析 → 找精确买点
    
    返回：
    - buy_points: 30分钟级别的买点列表
    - last_stroke: 最后一笔（判断当前位置）
    """
    if len(klines_30min) < 20:
        return {'buy_points': [], 'detail': '30分钟数据不足'}
    
    engine = ChanLunEngine()
    result = engine.analyze(klines_30min)
    buys = result.get('buy_sell_points', [])
    pivots = result['pivots']
    strokes = result['strokes']
    
    return {
        'buy_points': buys,
        'pivots': pivots,
        'strokes': strokes,
        'detail': f'30分钟: {len(pivots)}个中枢, {len(buys)}个信号'
    }


def multi_level_analyze(daily_klines: List[KLine], klines_30min: List[KLine] = None) -> Dict:
    """三级别联立分析（区间套）
    
    周线：方向 + 背驰预警
    日线：中枢位置 + 123买点
    30分钟：背驰精确买点
    """
    # ========== 第一层：周线 ==========
    weekly = daily_to_weekly(daily_klines)
    w_engine, w_result, w_trend = _engine_with_divergence(weekly)
    
    # 周线方向
    w_dir = weekly_direction_ggdd(weekly)
    
    # 周线位置
    w_pos = weekly_position_analysis(daily_klines)
    
    # 周线背驰状态
    w_div = {
        'stage': w_trend.stage.value,
        'pivot_count': w_trend.pivot_count,
        'direction': w_trend.trend_direction,
        'macd_shrinking': w_trend.macd_shrinking,
        'exhaustion_risk': w_trend.exhaustion_risk,
        'note': w_trend.note,
    }
    
    # 周线买卖点
    w_buys = [b for b in w_result.get('buy_sell_points', []) 
              if 'B' in b['type'] and 'S' not in b['type']]
    w_sells = [b for b in w_result.get('buy_sell_points', []) 
               if 'S' in b['type']]
    
    # ========== 第二层：日线 ==========
    d_engine, d_result, d_trend = _engine_with_divergence(daily_klines)
    
    # 日线买点
    daily_buys = d_result.get('buy_sell_points', [])
    
    # ========== 周线→日线联立过滤 ==========
    filtered_buys = []
    for b in daily_buys:
        is_buy = 'B' in b['type'] and 'S' not in b['type']
        if not is_buy:
            filtered_buys.append(b)
            continue
        
        # 周线背驰预警时，日线买点降级
        if w_trend.exhaustion_risk >= 0.6:
            b = dict(b)
            risk_discount = 1.0 - w_trend.exhaustion_risk * 0.3
            b['confidence'] *= risk_discount
            b['reason'] = "[周线背驰预警] " + b.get('reason', '')
        
        # 周线方向不利的降级
        if w_dir['direction'] == 'down' and w_dir.get('confidence', 0) >= 0.7:
            b = dict(b)
            b['confidence'] *= 0.7
            b['reason'] = "[周线下跌] " + b.get('reason', '')
        
        # 周线方向有利的小幅加分
        elif w_dir['direction'] == 'up' and w_dir.get('confidence', 0) >= 0.7:
            b = dict(b)
            b['confidence'] = min(b['confidence'] * 1.05, 1.0)
        
        filtered_buys.append(b)
    
    # ========== 第三层：30分钟 ==========
    m30_info = None
    if klines_30min and len(klines_30min) >= 20:
        m30_engine, m30_result, m30_trend = _engine_with_divergence(klines_30min)
        
        # 30分钟买点
        m30_buys = [b for b in m30_result.get('buy_sell_points', [])
                     if 'B' in b['type'] and 'S' not in b['type']]
        
        # 30分钟背驰检测（最后两笔对比）
        m30_divergence = None
        if len(m30_engine.strokes) >= 2:
            from chanlun_engine import MACDDivergence
            curr = m30_engine.strokes[-1]
            prev = m30_engine.strokes[-2]
            # 找同方向的上一笔做背驰对比
            for s in reversed(m30_engine.strokes[:-1]):
                if s.stype == curr.stype:
                    prev = s
                    break
            div = MACDDivergence.check_divergence(curr, prev, klines_30min)
            m30_divergence = {
                'diverged': div.get('divergence', False),
                'area_ratio': div.get('area_ratio', 1.0),
            }
        
        # 30分钟趋势背驰
        m30_trend_info = {
            'stage': m30_trend.stage.value,
            'direction': m30_trend.trend_direction,
            'macd_shrinking': m30_trend.macd_shrinking,
            'note': m30_trend.note,
        }
        
        # 30分钟最后笔信息
        m30_last_stroke = None
        if m30_engine.strokes:
            ls = m30_engine.strokes[-1]
            m30_last_stroke = {
                'type': ls.stype.name,
                'end': ls.end_date,
                'strength': ls.end_strength,
                'score': ls.strength_score,
            }
        
        m30_info = {
            'buys': m30_buys,
            'divergence': m30_divergence,
            'trend': m30_trend_info,
            'last_stroke': m30_last_stroke,
            'pivots_count': len(m30_result.get('pivots', [])),
            'detail': '30分钟: {}个中枢, {}个买点{}'.format(
                len(m30_result.get('pivots', [])),
                len(m30_buys),
                ', 背驰!' if m30_divergence and m30_divergence.get('diverged') else ''
            ),
        }
        
        # 30分钟精确入场信号（传入30分钟买卖点用于次级别确认）
        m30_entry = M30EntrySignal.detect(m30_engine, klines_30min, m30_buys)
        m30_info['entry'] = m30_entry
    
    # ========== 综合评级 ==========
    summary_lines = []
    
    # 周线方向
    wd = w_dir.get('direction', 'unknown')
    wt = w_pos.get('trend', 'unknown')
    summary_lines.append('周线: {} {}'.format(wd, wt))
    
    # 周线背驰
    if w_trend.exhaustion_risk >= 0.6:
        summary_lines.append('⚠️ 周线背驰预警: {}'.format(w_trend.note))
    elif w_trend.pivot_count >= 2:
        summary_lines.append('周线趋势确认: {}'.format(w_trend.note))
    
    # 日线买点
    daily_buy_b = [b for b in filtered_buys if 'B' in b['type'] and 'S' not in b['type']]
    if daily_buy_b:
        best = max(daily_buy_b, key=lambda x: x.get('confidence', 0))
        summary_lines.append('日线最强买点: {} {} @{:.2f} conf={:.2f}'.format(
            best['type'], best['date'], best['price'], best['confidence']))
    
    # 日线卖点（带操作建议）
    daily_sells = [b for b in filtered_buys if 'S' in b['type']]
    if daily_sells:
        # 卖点操作分级
        SELL_ACTION = {
            '1S': '减仓50%', 'sub1S': '减仓30%',
            '2S': '清仓', '3S': '清仓',
        }
        best_sell = max(daily_sells, key=lambda x: x.get('confidence', 0))
        action = best_sell.get('action', '') or SELL_ACTION.get(best_sell['type'], '')
        action_tag = ' → {}'.format(action) if action else ''
        summary_lines.append('日线最强卖点: {} {} @{:.2f} conf={:.2f}{}'.format(
            best_sell['type'], best_sell['date'], best_sell['price'], best_sell['confidence'], action_tag))
    
    # 多级别共振：30分钟回调 vs 日线中枢ZG
    # 进入中枢后回抽不过前高 → 2卖信号
    if m30_info and daily_buy_b and klines_30min is not None and len(klines_30min) > 0:
        m30_entry = m30_info.get('entry', {})
        daily_pivots = d_result.get('pivots', [])
        if daily_pivots:
            last_pivot = daily_pivots[-1]
            daily_zg = last_pivot.get('zg', 0)
            daily_gg = last_pivot.get('gg', 0)
            
            if daily_zg > 0:
                # 兼容dict和KLine对象
                def _kl_low(b): return b['low'] if isinstance(b, dict) else b.low
                def _kl_high(b): return b['high'] if isinstance(b, dict) else b.high
                
                recent_bars = klines_30min[-20:] if len(klines_30min) >= 20 else klines_30min
                m30_recent_low = min(_kl_low(b) for b in recent_bars)
                m30_recent_high = max(_kl_high(b) for b in recent_bars)
                
                if m30_recent_low < daily_zg:
                    # 30分钟回调已进入日线中枢 → 结构破坏
                    m30_entry['zg_violation'] = True
                    
                    if daily_gg > 0 and m30_recent_high < daily_gg * 0.99:
                        # 回抽不能突破前高(GG) → 2卖确认
                        m30_entry['status'] = 'sell_signal'
                        m30_entry['sell_type'] = '2S'
                        m30_entry['detail'] = '🔴 2卖确认(清仓): 30分钟回调入中枢(ZG={:.2f})后回抽{:.2f}不过GG{:.2f}'.format(
                            daily_zg, m30_recent_high, daily_gg)
                        m30_entry['sell_price'] = m30_recent_high
                        m30_entry['stop_loss'] = m30_recent_high * 1.01
                        m30_entry['entry_price'] = None
                    else:
                        # 已入中枢但还未确认回抽失败 → 2卖预警
                        m30_entry['status'] = 'sell_warning'
                        m30_entry['sell_type'] = 'potential_2S'
                        m30_entry['detail'] = '⚠️ 2卖预警: 回调{:.2f}已入中枢(ZG={:.2f}), 等回抽不过GG{:.2f}'.format(
                            m30_recent_low, daily_zg, daily_gg)
                        m30_entry['entry_price'] = None
                        m30_entry['stop_loss'] = None
                    
                    m30_info['entry'] = m30_entry
    
    # 30分钟确认
    if m30_info:
        entry = m30_info.get('entry', {})
        if entry.get('status') in ('strong_entry', 'double_entry'):
            sub_sig = entry.get('sub_level_signal', '')
            sub_tag = ' [{}]'.format(sub_sig) if sub_sig else ''
            summary_lines.append('🎯 30分钟入场{}: {} @{:.2f}'.format(
                sub_tag, entry.get('detail', '')[:40], entry.get('entry_price', 0)))
        elif entry.get('status') == 'divergence':
            summary_lines.append('⏳ 30分钟: {}'.format(entry.get('detail', '')[:50]))
        elif entry.get('status') == 'sell_signal':
            summary_lines.append('🔴 30分钟2卖: {}'.format(entry.get('detail', '')[:60]))
        elif entry.get('status') == 'sell_warning':
            summary_lines.append('⚠️ 30分钟2卖预警: {}'.format(entry.get('detail', '')[:60]))
        elif entry.get('status') == 'no_entry':
            summary_lines.append('❌ 30分钟: {}'.format(entry.get('detail', '')[:50]))
    
    return {
        'weekly': {
            **w_dir,
            'position': w_pos,
            'divergence': w_div,
            'buys': w_buys,
            'sells': w_sells,
        },
        'daily': {
            'pivots': d_result['pivots'],
            'strokes': d_result['strokes'],
            'buy_sell_points': filtered_buys,
            'trend': {
                'stage': d_trend.stage.value,
                'direction': d_trend.trend_direction,
                'note': d_trend.note,
            },
        },
        'm30': m30_info,
        'summary': '\n'.join(summary_lines),
    }


if __name__ == '__main__':
    from tdx_day_reader import load_stock_klines
    from kline_supplement import load_latest, supplement_klines
    
    test_stocks = [
        ('002828', '0', '002828'),
        ('600233', '1', '600233'),
        ('000608', '0', '000608'),
        ('600396', '1', '600396'),
    ]
    
    for code, sc, name in test_stocks:
        raw = load_stock_klines(code, sc, 500)
        kl = supplement_klines(code, sc, raw, load_latest())
        klines = [KLine(date=d[0], open=d[1], high=d[2], low=d[3], close=d[4]) for d in kl]
        
        result = multi_level_analyze(klines)
        
        print('\n' + '=' * 60)
        print('{} ({})'.format(code, name))
        print('=' * 60)
        print(result['summary'])
        
        # 周线详情
        w = result['weekly']
        print('\n周线详情:')
        print('  方向={} 趋势={}'.format(w['direction'], w.get('position', {}).get('trend', '')))
        div = w.get('divergence', {})
        print('  背驰: stage={} risk={:.1f} {}'.format(
            div.get('stage',''), div.get('exhaustion_risk',0), div.get('note','')))
        
        # 日线买点
        buys = [b for b in result['daily']['buy_sell_points'] 
                if 'B' in b['type'] and 'S' not in b['type']]
        print('\n日线买点({}):'.format(len(buys)))
        for b in buys[:5]:
            print('  {} {} @{:.2f} conf={:.2f} | {}'.format(
                b['type'], b['date'], b['price'], b['confidence'], 
                b.get('reason','')[:60]))
