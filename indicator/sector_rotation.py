"""
板块轮动监控 — 全A股8分类 + 多空比 + 量价背离

从TDX本地日线数据:
  1. 按涨跌幅/量能/动量自动聚类8种状态
  2. 多空比判断市场整体方向
  3. 量价背离检测 (复用 vol_price_divergence)
  4. 缠论买点交叉验证

8分类:
  强势领涨 / 放量突破 / 温和上涨 / 异动放量 / 盘整 / 温和下跌 / 放量破位 / 强势下跌

用法:
  from indicator.sector_rotation import SectorRotationMonitor
  monitor = SectorRotationMonitor()
  report = monitor.analyze()
  print(report['summary'])
"""

import os
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from indicator.vol_price_divergence import (
    detect_volume_price_divergence, DivergenceType,
)
from indicator.market_breadth import calc_market_breadth


@dataclass
class StockState:
    code: str
    market: str
    close: float
    chg_1d: float
    chg_5d: float
    chg_10d: float
    vol_ratio: float
    vol_trend: float
    amplitude: float
    strength: float
    category: str = ''
    divergence: str = ''


@dataclass
class SectorReport:
    date: str
    total: int
    categories: Dict[str, int]
    bull_count: int
    bear_count: int
    bull_bear_ratio: float
    flow_in_top: List[StockState]
    flow_out_top: List[StockState]
    divergence_signals: List[StockState]
    summary: str
    advice: str


def _read_last_n(filepath: str, n: int = 60):
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
            out.append({'date': ds, 'close': c / 100, 'high': h / 100,
                        'low': l / 100, 'open': op / 100, 'volume': vol})
    return out


def _classify_stock(strength: float, chg_5d: float, vol_ratio: float) -> str:
    if strength > 5 and chg_5d > 5:
        return '强势领涨'
    elif strength > 2 and vol_ratio > 1.5:
        return '放量突破'
    elif strength > 0 and chg_5d > 0:
        return '温和上涨'
    elif strength < -5 and chg_5d < -5:
        return '强势下跌'
    elif strength < -2 and vol_ratio > 1.5:
        return '放量破位'
    elif strength < 0 and chg_5d < 0:
        return '温和下跌'
    elif vol_ratio > 2.0:
        return '异动放量'
    else:
        return '盘整'


class SectorRotationMonitor:
    """板块轮动监控"""

    TDX_CANDIDATES = [
        r'D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc',
        r'D:\new_tdx\vipdoc',
        r'D:\新建文件夹\claude\tdx_data',
        r'D:\tdx_data',
    ]

    def __init__(self, tdx_vipdoc: Optional[str] = None):
        if tdx_vipdoc is None:
            for p in self.TDX_CANDIDATES:
                if os.path.exists(p):
                    tdx_vipdoc = p
                    break
        self._tdx = tdx_vipdoc

    def analyze(self, top_n: int = 20) -> SectorReport:
        all_stocks: List[StockState] = []
        categories = defaultdict(list)
        last_date = ''
        total = 0

        for mkt in ['sh', 'sz']:
            folder = os.path.join(self._tdx, mkt, 'lday') if self._tdx else ''
            if not folder or not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if not (f.startswith(mkt) and f.endswith('.day')):
                    continue
                code = f[2:8]
                if not code.startswith(('0', '3', '6')):
                    continue
                fp = os.path.join(folder, f)
                recs = _read_last_n(fp, 60)
                if not recs or len(recs) < 20:
                    continue

                total += 1
                last = recs[-1]
                prev = recs[-2]
                last_date = max(last_date, last['date'])

                chg_1d = (last['close'] - prev['close']) / prev['close'] * 100 if prev['close'] > 0 else 0
                chg_5d = (last['close'] - recs[-6]['close']) / recs[-6]['close'] * 100 if len(recs) >= 6 else 0
                chg_10d = (last['close'] - recs[-11]['close']) / recs[-11]['close'] * 100 if len(recs) >= 11 else 0

                vol_5 = sum(r['volume'] for r in recs[-5:]) / 5 if len(recs) >= 5 else last['volume']
                vol_20 = sum(r['volume'] for r in recs[-20:]) / 20 if len(recs) >= 20 else vol_5
                vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
                vol_trend = (vol_5 - vol_20) / vol_20 * 100 if vol_20 > 0 else 0

                amplitude = (last['high'] - last['low']) / last['close'] * 100 if last['close'] > 0 else 0
                strength = (
                    chg_5d * 0.3 +
                    chg_10d * 0.2 +
                    vol_ratio * 10 * 0.2 +
                    (1 if chg_1d > 0 else -1) * amplitude * 0.3
                )

                # 量价背离
                div_label = ''
                closes = [r['close'] for r in recs[-25:]]
                volumes = [r['volume'] for r in recs[-25:]]
                vpd = detect_volume_price_divergence(closes, volumes)
                if vpd:
                    if vpd.divergence_type == DivergenceType.BEARISH_DISTRIBUTION:
                        div_label = '价涨量缩'
                    elif vpd.divergence_type == DivergenceType.BULLISH_ACCUMULATION:
                        div_label = '价跌量增'
                    elif vpd.divergence_type == DivergenceType.STRONG_BULL:
                        div_label = '放量上涨'
                    elif vpd.divergence_type == DivergenceType.WEAK_BEAR:
                        div_label = '缩量下跌'

                cat = _classify_stock(strength, chg_5d, vol_ratio)
                ss = StockState(
                    code=code, market=mkt, close=last['close'],
                    chg_1d=round(chg_1d, 2), chg_5d=round(chg_5d, 2),
                    chg_10d=round(chg_10d, 2), vol_ratio=round(vol_ratio, 2),
                    vol_trend=round(vol_trend, 1), amplitude=round(amplitude, 2),
                    strength=round(strength, 2), category=cat, divergence=div_label,
                )
                all_stocks.append(ss)
                categories[cat].append(ss)

        # 多空比
        bull = (len(categories.get('强势领涨', [])) +
                len(categories.get('放量突破', [])) +
                len(categories.get('温和上涨', [])))
        bear = (len(categories.get('强势下跌', [])) +
                len(categories.get('放量破位', [])) +
                len(categories.get('温和下跌', [])))
        ratio = bull / bear if bear > 0 else float('inf')

        # 资金流入TOP
        flow_in = sorted(
            [s for s in all_stocks if s.chg_5d > 0 and s.vol_ratio > 1.2],
            key=lambda x: x.strength, reverse=True,
        )[:top_n]

        # 资金流出TOP
        flow_out = sorted(
            [s for s in all_stocks if s.chg_5d < -2 and s.vol_ratio > 1.2],
            key=lambda x: x.strength,
        )[:10]

        # 背离信号
        divs = sorted(
            [s for s in all_stocks if s.divergence],
            key=lambda x: abs(x.strength), reverse=True,
        )[:15]

        # 建议
        if ratio > 2.0:
            advice = '多头碾压，适合进攻'
        elif ratio > 1.3:
            advice = '多头占优，关注强势板块回调买点'
        elif ratio > 0.7:
            advice = '分化市场，选股>择时'
        else:
            advice = '空头主导，防御为主'

        cat_counts = {cat: len(stocks) for cat, stocks in categories.items()}

        summary_lines = [
            f'板块轮动 | {last_date} | {total}只 | 多空比{ratio:.2f}',
            f'  多头: {bull} ({bull/total*100:.1f}%) 空头: {bear} ({bear/total*100:.1f}%)',
        ]
        for cat in ['强势领涨', '放量突破', '温和上涨', '异动放量', '盘整', '温和下跌', '放量破位', '强势下跌']:
            n = cat_counts.get(cat, 0)
            if n:
                summary_lines.append(f'  {cat}: {n}')

        return SectorReport(
            date=last_date, total=total, categories=cat_counts,
            bull_count=bull, bear_count=bear, bull_bear_ratio=round(ratio, 2),
            flow_in_top=flow_in, flow_out_top=flow_out,
            divergence_signals=divs,
            summary='\n'.join(summary_lines), advice=advice,
        )
