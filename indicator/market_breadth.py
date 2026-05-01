"""
市场宽度指标 — 全A股MA/涨跌/量能全景

从TDX本地日线数据计算:
- MA20/MA60/MA250上方占比
- 涨跌比 + 涨跌停数
- 量能趋势（放量/缩量）
- 综合健康度评分 0~100

用法:
  from indicator.market_breadth import calc_market_breadth
  result = calc_market_breadth()  # 使用TDX本地数据
"""

import os
import struct
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MarketBreadthResult:
    date: str
    total: int
    above_ma20: int
    pct_ma20: float
    above_ma60: int
    pct_ma60: float
    above_ma250: int
    pct_ma250: float
    up_count: int
    down_count: int
    advance_decline: float
    limit_up: int
    limit_down: int
    vol_expanding: int
    vol_shrinking: int
    health_score: float
    grade: str

    def summary(self) -> str:
        lines = [
            f'市场宽度 | {self.date} | {self.total}只',
            f'MA20上方: {self.pct_ma20:.1f}% | MA60上方: {self.pct_ma60:.1f}% | MA250上方: {self.pct_ma250:.1f}%',
            f'涨跌比: {self.advance_decline:.2f} (涨{self.up_count}/跌{self.down_count})',
            f'涨停{self.limit_up}/跌停{self.limit_down} | 放量{self.vol_expanding}/缩量{self.vol_shrinking}',
            f'健康度: {self.health_score:.0f}/100 {self.grade}',
        ]
        return '\n'.join(lines)


def _read_last_n(filepath: str, n: int = 300):
    """快速读取TDX .day文件最后n根日K"""
    if not os.path.exists(filepath):
        return None
    sz = os.path.getsize(filepath)
    if sz < 32:
        return None
    cnt = sz // 32
    start = max(0, cnt - n)
    with open(filepath, 'rb') as f:
        f.seek(start * 32)
        data = f.read()
    out = []
    for i in range(len(data) // 32):
        o = data[i * 32:i * 32 + 32]
        if len(o) < 32:
            break
        d, op, h, l, c, amt, vol, _ = struct.unpack('<IIIIIfII', o)
        ds = str(d)
        if len(ds) == 8 and d > 20200101:
            out.append({
                'date': ds, 'close': c / 100, 'high': h / 100,
                'low': l / 100, 'open': op / 100, 'volume': vol,
            })
    return out


def calc_market_breadth(tdx_vipdoc: Optional[str] = None) -> MarketBreadthResult:
    """
    计算全市场宽度指标

    Args:
        tdx_vipdoc: TDX vipdoc路径，不传则从config.yaml读取
    """
    if tdx_vipdoc is None:
        # 复用HybridSource的TDX路径检测
        _tdx_candidates = [
            r'D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc',
            r'D:\new_tdx\vipdoc',
            r'D:\新建文件夹\claude\tdx_data',
            r'D:\tdx_data',
        ]
        for p in _tdx_candidates:
            if os.path.exists(p):
                tdx_vipdoc = p
                break

    if not tdx_vipdoc or not os.path.exists(tdx_vipdoc):
        return MarketBreadthResult(
            date='', total=0, above_ma20=0, pct_ma20=0,
            above_ma60=0, pct_ma60=0, above_ma250=0, pct_ma250=0,
            up_count=0, down_count=0, advance_decline=0,
            limit_up=0, limit_down=0, vol_expanding=0, vol_shrinking=0,
            health_score=0, grade='无数据',
        )

    total = 0
    above_ma20 = above_ma60 = above_ma250 = 0
    up_count = down_count = limit_up = limit_down = 0
    vol_expanding = vol_shrinking = 0
    last_date = ''

    for mkt in ['sh', 'sz']:
        folder = os.path.join(tdx_vipdoc, mkt, 'lday')
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if not (f.startswith(mkt) and f.endswith('.day')):
                continue
            code = f[2:8]
            if not code.startswith(('0', '3', '6')):
                continue
            fp = os.path.join(folder, f)
            recs = _read_last_n(fp, 300)
            if not recs or len(recs) < 250:
                continue

            total += 1
            last_price = recs[-1]['close']
            last_date = max(last_date, recs[-1]['date'])

            prev_close = recs[-2]['close'] if len(recs) >= 2 else last_price
            chg = (last_price - prev_close) / prev_close * 100 if prev_close > 0 else 0
            if chg > 0:
                up_count += 1
            elif chg < 0:
                down_count += 1
            if chg > 9.8:
                limit_up += 1
            elif chg < -9.8:
                limit_down += 1

            # MA
            for period, counter in [(20, 'ma20'), (60, 'ma60'), (250, 'ma250')]:
                if len(recs) >= period:
                    ma = sum(r['close'] for r in recs[-period:]) / period
                    if last_price > ma:
                        if counter == 'ma20':
                            above_ma20 += 1
                        elif counter == 'ma60':
                            above_ma60 += 1
                        else:
                            above_ma250 += 1

            # 量能
            if len(recs) >= 20:
                vol_ma = sum(r['volume'] for r in recs[-20:]) / 20
                if vol_ma > 0:
                    today_vol = recs[-1]['volume']
                    if today_vol > vol_ma * 1.3:
                        vol_expanding += 1
                    elif today_vol < vol_ma * 0.7:
                        vol_shrinking += 1

    # 计算指标
    pct_ma20 = above_ma20 / total * 100 if total else 0
    pct_ma60 = above_ma60 / total * 100 if total else 0
    pct_ma250 = above_ma250 / total * 100 if total else 0
    ad = up_count / down_count if down_count > 0 else 99.0

    # 健康度评分 (0~100)
    score = 0.0
    score += min(30, pct_ma20 / 100 * 60)
    score += min(20, pct_ma60 / 100 * 40)
    if ad > 2:
        score += 20
    elif ad > 1.5:
        score += 15
    elif ad > 1:
        score += 10
    elif ad > 0.5:
        score += 5
    if limit_up > 50:
        score += 15
    elif limit_up > 20:
        score += 10
    elif limit_up > 5:
        score += 5
    score += min(15, pct_ma250 / 100 * 30)
    score = round(min(100, score))

    if score >= 80:
        grade = '强势'
    elif score >= 60:
        grade = '偏强'
    elif score >= 40:
        grade = '中性'
    elif score >= 20:
        grade = '偏弱'
    else:
        grade = '弱势'

    return MarketBreadthResult(
        date=last_date, total=total,
        above_ma20=above_ma20, pct_ma20=round(pct_ma20, 1),
        above_ma60=above_ma60, pct_ma60=round(pct_ma60, 1),
        above_ma250=above_ma250, pct_ma250=round(pct_ma250, 1),
        up_count=up_count, down_count=down_count,
        advance_decline=round(ad, 2),
        limit_up=limit_up, limit_down=limit_down,
        vol_expanding=vol_expanding, vol_shrinking=vol_shrinking,
        health_score=score, grade=grade,
    )
