"""
热点板块识别 — 基于本地TDX日线数据

纯本地实现，不依赖AKShare/Sina，避免代理问题。
复用 sector_map + TDX .day 二进制文件。

功能:
1. identify_hot_sectors() — 识别TOP N热点板块
2. rank_stocks_in_sector() — 板块内缠论选股
3. detect_limit_ups_from_tdx() — 涨停板检测
"""

import os
import json
import struct
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HotSector:
    name: str
    score: float = 0.0
    phase: str = ''
    return_1d: float = 0.0
    return_5d: float = 0.0
    up_ratio_1d: float = 0.0
    limit_up_count: int = 0
    stock_count: int = 0
    amount_ratio: float = 0.0
    dragon: str = ''
    dragon_name: str = ''
    dragon_boards: int = 0
    top_stocks: List[Dict] = field(default_factory=list)
    all_codes: List[str] = field(default_factory=list)


class HotSectorAnalyzer:
    """热点板块识别器 — 纯本地TDX数据"""

    def __init__(
        self,
        tdx_path: str = 'tdx_data',
        sector_map_file: str = 'chanlun_system/full_sector_map.json',
    ):
        self.tdx_path = tdx_path
        self.sector_map = self._load_sector_map(sector_map_file)
        self._stock_cache: Dict[str, np.ndarray] = {}

    def _load_sector_map(self, path: str) -> Dict[str, str]:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('stock_to_sector', data) if isinstance(data, dict) else {}

    def _read_day_tail(self, filepath: str, n: int = 10) -> Optional[np.ndarray]:
        """读取.day文件最后N条记录，返回numpy数组 [date, open, high, low, close, amount, vol]"""
        try:
            sz = os.path.getsize(filepath)
            total = sz // 32
            if total < 2:
                return None
            start = max(0, total - n)
            records = []
            with open(filepath, 'rb') as f:
                f.seek(start * 32)
                data = f.read((total - start) * 32)
            for i in range(len(data) // 32):
                off = i * 32
                dv, op, hp, lp, cp, amt, vol, _ = struct.unpack('IIIIIfII', data[off:off + 32])
                if dv == 0:
                    continue
                records.append([dv, op / 100.0, hp / 100.0, lp / 100.0, cp / 100.0, amt, vol])
            if not records:
                return None
            return np.array(records)
        except Exception:
            return None

    def _load_all_stocks_tail(self, n: int = 10) -> Dict[str, np.ndarray]:
        """批量加载所有股票最后N天数据"""
        if self._stock_cache and self._cache_n == n:
            return self._stock_cache

        result = {}
        for subdir in ['sh/lday', 'sz/lday']:
            day_dir = Path(self.tdx_path) / subdir
            if not day_dir.exists():
                continue
            for f in day_dir.glob('*.day'):
                code_num = f.stem[2:]  # sh600000 -> 600000
                exchange = f.stem[:2]
                full_code = f'{exchange}{code_num}'
                # Skip indices, BJ
                if code_num.startswith(('0000', '8', '9')):
                    continue
                arr = self._read_day_tail(str(f), n)
                if arr is not None and len(arr) >= 2:
                    result[full_code] = arr

        self._stock_cache = result
        self._cache_n = n
        return result

    def _get_sector_for_code(self, code: str) -> str:
        num = code[2:] if code[:2] in ('sh', 'sz') else code
        return self.sector_map.get(num, '')

    def detect_limit_ups_from_tdx(self) -> List[Dict]:
        """从TDX日线检测最近一天涨停板"""
        stocks = self._load_all_stocks_tail(10)
        limit_ups = []

        for code, arr in stocks.items():
            if len(arr) < 2:
                continue
            last = arr[-1]
            prev = arr[-2]
            prev_close = prev[4]
            if prev_close <= 0:
                continue
            change = (last[4] - prev_close) / prev_close * 100
            # 涨停阈值: 科创板/创业板20%, 其他10%
            num = code[2:]
            threshold = 19.5 if (num.startswith('688') or num.startswith('3')) else 9.5
            if change >= threshold:
                # 计算连续涨停
                consecutive = 1
                for j in range(len(arr) - 2, -1, -1):
                    if j == 0:
                        break
                    c_prev = arr[j - 1][4]
                    if c_prev <= 0:
                        break
                    c_change = (arr[j][4] - c_prev) / c_prev * 100
                    if c_change >= threshold:
                        consecutive += 1
                    else:
                        break

                limit_ups.append({
                    'code': code,
                    'price': round(last[4], 2),
                    'change': round(change, 2),
                    'amount': round(last[5] / 1e8, 2),
                    'volume': int(last[6]),
                    'consecutive': consecutive,
                    'sector': self._get_sector_for_code(code),
                })

        limit_ups.sort(key=lambda x: x['consecutive'] * 1000 + x['amount'], reverse=True)
        return limit_ups

    def classify_sector_phase(self, returns: List[float]) -> str:
        """判断板块阶段: 启动/加速/高潮/退潮"""
        if len(returns) < 5:
            return 'unknown'

        recent_2 = returns[-2:]
        early_3 = returns[-5:-2]
        avg_early = np.mean(early_3)
        avg_recent = np.mean(recent_2)

        # 高潮: 近2天平均涨幅 > 3%
        if avg_recent > 3:
            return '高潮'
        # 启动: 前3天弱/跌 + 近2天突然转正
        if avg_early <= 0 and avg_recent > 1:
            return '启动'
        # 加速: 连续上涨且加速
        if all(r > 0 for r in recent_2) and avg_recent > avg_early:
            return '加速'
        # 退潮: 涨幅收窄或转负
        if avg_recent < avg_early and avg_recent < 1:
            return '退潮'
        return '震荡'

    def identify_hot_sectors(self, lookback: int = 5, top_n: int = 10) -> List[HotSector]:
        """识别热点板块 — 综合评分"""
        stocks = self._load_all_stocks_tail(lookback + 2)
        limit_ups = self.detect_limit_ups_from_tdx()

        # 涨停按板块统计
        lu_by_sector = defaultdict(list)
        for lu in limit_ups:
            if lu['sector']:
                lu_by_sector[lu['sector']].append(lu)

        # 按板块聚合所有股票
        sector_data = defaultdict(lambda: {
            'returns': [], '1d_returns': [], 'daily_rets_agg': [], 'codes': [], 'amounts': []
        })
        for code, arr in stocks.items():
            sector = self._get_sector_for_code(code)
            if not sector or len(arr) < lookback + 1:
                continue
            ret = (arr[-1][4] - arr[-lookback - 1][4]) / arr[-lookback - 1][4] * 100
            ret_1d = (arr[-1][4] - arr[-2][4]) / arr[-2][4] * 100 if len(arr) >= 2 else 0
            daily_rets = []
            for j in range(max(0, len(arr) - lookback), len(arr)):
                if j > 0 and arr[j - 1][4] > 0:
                    daily_rets.append((arr[j][4] - arr[j - 1][4]) / arr[j - 1][4] * 100)

            sector_data[sector]['returns'].append(ret)
            sector_data[sector]['1d_returns'].append(ret_1d)
            sector_data[sector]['daily_rets_agg'].append(daily_rets)
            sector_data[sector]['codes'].append(code)
            sector_data[sector]['amounts'].append(arr[-1][5])

        # 计算每个板块指标并评分
        sectors = []
        for name, data in sector_data.items():
            if len(data['codes']) < 5:
                continue

            rets = data['returns']
            r1d = data['1d_returns']
            median_ret = float(np.median(rets))
            avg_r1d = float(np.mean(r1d))
            up_ratio = sum(1 for r in r1d if r > 0) / len(rets)

            # 板块阶段
            # 取板块平均日收益序列
            max_len = max(len(dr) for dr in data.get('daily_rets_agg', [[]]))
            avg_daily = []
            for day_i in range(max_len):
                day_vals = [dr[day_i] for dr in data['daily_rets_agg'] if day_i < len(dr)]
                if day_vals:
                    avg_daily.append(float(np.mean(day_vals)))
            phase = self.classify_sector_phase(avg_daily) if len(avg_daily) >= 3 else 'unknown'

            # 涨停统计
            lu_list = lu_by_sector.get(name, [])
            lu_count = len(lu_list)

            # 龙一
            dragon_code = ''
            dragon_boards = 0
            dragon_name = ''
            if lu_list:
                dragon_code = lu_list[0]['code']
                dragon_boards = lu_list[0]['consecutive']

            # 综合评分 (0-100)
            score = 0.0
            # 涨幅 (0-30)
            score += min(max(median_ret, 0), 15) * 2
            # 上涨比例 (0-25)
            score += up_ratio * 25
            # 涨停数 (0-20)
            score += min(lu_count * 4, 20)
            # 阶段加分 (0-15)
            phase_bonus = {'启动': 15, '加速': 12, '高潮': 5, '退潮': 0, '震荡': 3, 'unknown': 0}
            score += phase_bonus.get(phase, 0)
            # 连板高度 (0-10)
            score += min(dragon_boards * 3, 10)

            all_codes = data['codes']
            sectors.append(HotSector(
                name=name,
                score=round(score, 1),
                phase=phase,
                return_1d=round(avg_r1d, 2),
                return_5d=round(median_ret, 2),
                up_ratio_1d=round(up_ratio, 3),
                limit_up_count=lu_count,
                stock_count=len(all_codes),
                dragon=dragon_code,
                dragon_boards=dragon_boards,
                all_codes=all_codes,
            ))

        sectors.sort(key=lambda x: x.score, reverse=True)
        return sectors[:top_n]

    def find_stocks_in_sector(self, sector_name: str) -> List[str]:
        """获取板块内所有股票代码"""
        codes = []
        for code_num, sector in self.sector_map.items():
            if sector == sector_name:
                # Try both sh and sz
                if code_num.startswith('6'):
                    codes.append(f'sh{code_num}')
                else:
                    codes.append(f'sz{code_num}')
        return codes

    def rank_stocks_in_sector(self, sector_name: str, top_n: int = 5) -> List[Dict]:
        """板块内缠论选股 — 按买点置信度排序"""
        from core.kline import KLine
        from core.fractal import FractalDetector
        from core.stroke import StrokeGenerator
        from core.segment import SegmentGenerator
        from core.pivot import PivotDetector
        from core.buy_sell_points import BuySellPointDetector
        from indicator.macd import MACD

        codes = self.find_stocks_in_sector(sector_name)
        results = []

        for code in codes:
            # Load daily data from TDX
            exchange = code[:2]
            num = code[2:]
            day_path = Path(self.tdx_path) / exchange / 'lday' / f'{code}.day'
            if not day_path.exists():
                continue

            arr = self._read_day_tail(str(day_path), 300)
            if arr is None or len(arr) < 60:
                continue

            # Build DataFrame
            records = []
            for row in arr:
                dv = int(row[0])
                dt = f'{str(dv)[:4]}-{str(dv)[4:6]}-{str(dv)[6:8]}'
                records.append({
                    'datetime': dt, 'open': row[1], 'high': row[2],
                    'low': row[3], 'close': row[4], 'volume': int(row[6]),
                })
            df = pd.DataFrame(records)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()

            try:
                kline = KLine.from_dataframe(df)
                fractals = FractalDetector(kline, confirm_required=False).get_fractals()
                strokes = StrokeGenerator(kline, fractals, min_bars=3).get_strokes()
                if len(strokes) < 5:
                    continue
                segments = SegmentGenerator(kline, strokes).get_segments()
                pivots = PivotDetector(kline, strokes).get_pivots()
                if not pivots:
                    continue
                close_s = pd.Series([k.close for k in kline])
                macd = MACD(close_s)
                det = BuySellPointDetector(fractals, strokes, segments, pivots, macd)
                buys, sells = det.detect_all()

                # 只取最近30根K线的买点
                recent = [b for b in buys if b.index >= len(df) - 30]
                for b in recent:
                    results.append({
                        'code': code,
                        'price': round(b.price, 2),
                        'type': b.point_type,
                        'confidence': round(b.confidence, 2),
                        'reason': b.reason[:60],
                        'bars_ago': len(df) - b.index,
                    })
            except Exception:
                continue

        # 排序: confidence > type priority > recency
        type_priority = {'1buy': 3, '2buy': 2, '3buy': 2, 'quasi2buy': 1, 'quasi3buy': 1}
        results.sort(key=lambda x: (
            x['confidence'],
            type_priority.get(x['type'], 0),
            -x['bars_ago'],
        ), reverse=True)

        return results[:top_n]

    def save_results(self, sectors: List[HotSector], output_dir: str = 'signals'):
        """保存结果到JSON"""
        from datetime import datetime
        os.makedirs(output_dir, exist_ok=True)
        today = datetime.now().strftime('%Y%m%d')

        output = {
            'date': today,
            'total_sectors': len(sectors),
            'sectors': [],
        }
        for s in sectors:
            d = asdict(s)
            d.pop('all_codes', None)  # Too large for JSON
            output['sectors'].append(d)

        # Save dated version
        path1 = os.path.join(output_dir, f'hot_sectors_{today}.json')
        with open(path1, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # Save latest version (for intraday to load)
        path2 = os.path.join(output_dir, 'hot_sectors_latest.json')
        # Include all_codes for watchlist building
        output_latest = {
            'date': today,
            'sectors': [],
        }
        for s in sectors[:5]:
            output_latest['sectors'].append({
                'name': s.name,
                'score': s.score,
                'phase': s.phase,
                'return_5d': s.return_5d,
                'limit_up_count': s.limit_up_count,
                'dragon': s.dragon,
                'stocks': s.all_codes,
            })
        with open(path2, 'w', encoding='utf-8') as f:
            json.dump(output_latest, f, ensure_ascii=False, indent=2)

        return path1, path2

    def print_report(self, sectors: List[HotSector]):
        """打印热点板块报告"""
        print(f'\n{"="*90}')
        print(f'=== 热点板块识别报告 (TDX本地数据) ===')
        print(f'{"="*90}')
        print(f'{"#":>3} {"板块":<12} {"评分":>5} {"阶段":<6} {"1日%":>7} {"5日%":>7} '
              f'{"上涨率":>6} {"涨停":>4} {"股票数":>5} {"龙一连板":>6}')
        print('-' * 90)

        for i, s in enumerate(sectors, 1):
            print(f'{i:>3} {s.name:<12} {s.score:>5.1f} {s.phase:<6} '
                  f'{s.return_1d:>+6.2f}% {s.return_5d:>+6.2f}% '
                  f'{s.up_ratio_1d:>5.1%} {s.limit_up_count:>4} {s.stock_count:>5} '
                  f'{s.dragon_boards:>6}')
