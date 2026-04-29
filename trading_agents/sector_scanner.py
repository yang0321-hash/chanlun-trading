#!/usr/bin/env python3
"""每日板块候选池扫描

收盘后运行，识别当日最强主线/辅线/重灾区板块，生成次日操作参考。

步骤:
  1. 获取今日涨幅前50 / 跌幅前50
  2. 提取行业信息 (full_sector_map + AKShare)
  3. 板块聚合统计
  4. 重灾区识别
  5. 板块指数MA状态
  6. 生成候选池 + 明日建议
"""
import sys
import os

sys.path.insert(0, '.')
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
          'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

import json
import traceback
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from data.hybrid_source import HybridSource


# ==================== 行业映射 ====================

def load_sector_map() -> Dict[str, str]:
    """加载行业映射 (纯数字code → 行业名)"""
    for path in ['chanlun_system/full_sector_map.json',
                 'chanlun_system/thshy_sector_map.json']:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('stock_to_sector', {})
            except Exception:
                pass
    return {}


def code_to_pure(code: str) -> str:
    """统一转换为纯数字代码"""
    code = code.upper()
    for prefix in ('SH', 'SZ', 'BJ'):
        code = code.replace(prefix, '')
    code = code.replace('.', '')
    while len(code) < 6:
        code = '0' + code
    return code


# ==================== 数据获取 ====================

def get_all_stocks_today() -> Optional[pd.DataFrame]:
    """获取全市场今日行情 (AKShare)"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={
            '代码': 'code', '名称': 'name',
            '涨跌幅': 'pct_chg', '涨跌额': 'change',
            '最新价': 'price', '今开': 'open',
            '最高': 'high', '最低': 'low',
            '成交量': 'volume', '成交额': 'amount',
            '换手率': 'turnover',
        })
        df['code'] = df['code'].apply(code_to_pure)
        return df
    except Exception as e:
        print(f'  AKShare全市场行情获取失败: {e}')
        return None


def get_all_stocks_tdx(hs: HybridSource) -> Optional[pd.DataFrame]:
    """备选: 从TDX本地数据计算今日涨跌幅"""
    try:
        sector_map = load_sector_map()
        rows = []
        for code in sector_map:
            try:
                df = hs.get_kline(code, period='daily')
                if df is None or len(df) < 2:
                    continue
                last = df.iloc[-1]
                prev = df.iloc[-2]
                pct = (last['close'] - prev['close']) / prev['close'] * 100
                rows.append({
                    'code': code,
                    'name': '',
                    'price': last['close'],
                    'pct_chg': pct,
                    'volume': last.get('volume', 0),
                    'amount': last.get('amount', 0),
                })
            except Exception:
                continue
        if rows:
            return pd.DataFrame(rows)
        return None
    except Exception:
        return None


# ==================== 板块指数 ====================

SECTOR_INDEX_MAP = {
    '电力': '880928', '风电设备': '885521', '光伏设备': '885522',
    '煤炭': '880302', '有色金属': '880630', '钢铁': '880318',
    '银行': '880831', '证券': '880832', '保险': '880833',
    '白酒': '880834', '房地产': '880835',
    '半导体': '885856', '消费电子': '885850',
    '化学制药': '885540', '中药': '885541', '生物制药': '885542',
    '汽车整车': '885841', '汽车零部件': '885842',
    '通信设备': '885860', '计算机': '885861',
    '军工': '885871', '航空装备': '885872',
    '传媒': '885862', '教育': '885863',
    '医疗器械': '885545', '医疗服务': '885546',
    '通用设备': '885820', '专用设备': '885821',
    '物流': '885880', '零售': '885881',
    '环保': '885885', '建筑材料': '885830',
    '港口航运': '885825', '机场航运': '885826',
    '旅游': '885882', '酒店餐饮': '885883',
    '储能': '885530', '电网设备': '885531',
    '稀土': '885816', '贵金属': '885817',
}


def get_sector_index_status(sector_name: str) -> dict:
    """查询板块指数MA状态"""
    idx_code = SECTOR_INDEX_MAP.get(sector_name)
    if not idx_code:
        return {'status': '无指数代码', 'above_ma20': None}

    try:
        import requests
        session = requests.Session()
        session.trust_env = False
        url = (f'https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/'
               f'CN_MarketDataService.getKLineData?symbol={idx_code}'
               f'&scale=240&ma=no&datalen=25')
        resp = session.get(url, timeout=8)
        import re
        match = re.search(r'callback\((.*)\)', resp.text)
        if match:
            klines = json.loads(match.group(1))
            if len(klines) >= 20:
                closes = np.array([float(k['close']) for k in klines])
                ma5 = np.mean(closes[-5:])
                ma20 = np.mean(closes[-20:])
                last = closes[-1]
                return {
                    'status': 'OK',
                    'close': last,
                    'ma5': ma5,
                    'ma20': ma20,
                    'above_ma20': last > ma20,
                    'ma5_above_ma20': ma5 > ma20,
                }
    except Exception:
        pass
    return {'status': '获取失败', 'above_ma20': None}


# ==================== 核心扫描 ====================

class SectorScanner:
    """板块候选池扫描器"""

    def __init__(self):
        self.hs = HybridSource()
        self.sector_map = load_sector_map()
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.all_df = None
        self.top50 = []
        self.bottom50 = []
        self.sector_count = Counter()
        self.sector_stocks = defaultdict(list)
        self.disaster_sectors = {}
        self.sector_index_cache = {}

    def _calc_market_score(self) -> int:
        """计算并缓存大盘评分"""
        try:
            from strategies.trading_rules import TradingRules
            idx_df = self.hs.get_kline('000001', period='daily')
            if idx_df is not None and len(idx_df) >= 25:
                closes = idx_df['close'].values.astype(float)
                vols = idx_df.get('volume')
                volumes = vols.values.astype(float) if vols is not None else None
                result = TradingRules.calc_market_score(closes, volumes)
                print(f'  大盘评分: {result.score}/12 ({result.state})')
                return result.score
        except Exception as e:
            print(f'  大盘评分计算失败: {e}')
        return 6

    def run(self) -> str:
        """执行完整扫描流程"""
        lines = [f'【板块候选池 · {self.today}】', '']

        # Step 0: 大盘评分
        self.market_score = self._calc_market_score()

        # Step 1+2: 获取全市场行情
        print('[1] 获取全市场行情...')
        self.all_df = get_all_stocks_today()
        if self.all_df is None:
            print('  AKShare失败, 尝试TDX本地...')
            self.all_df = get_all_stocks_tdx(self.hs)
        if self.all_df is None:
            lines.append('ERROR: 无法获取行情数据')
            return '\n'.join(lines)
        print(f'  获取到 {len(self.all_df)} 只股票')

        # 过滤: 去掉ST、退市、北交所
        df = self.all_df[~self.all_df['name'].str.contains('ST|退', na=False)].copy()
        self.all_df = df
        print(f'  过滤后 {len(df)} 只')

        # Step 2: 行业标注
        print('[2] 行业标注...')
        self._attach_sectors()
        coverage = len(df[df['sector'] != '未知']) / max(len(df), 1) * 100
        print(f'  行业覆盖率: {coverage:.0f}%')

        # Step 3: 涨幅前50聚合
        print('[3] 涨幅前50聚合...')
        self._aggregate_top_gainers()

        # Step 4: 跌幅重灾区
        print('[4] 重灾区识别...')
        self._identify_disaster_zones()

        # Step 5: 板块指数状态
        print('[5] 板块指数状态...')
        self._check_sector_indices()

        # Step 6: 生成候选池
        print('[6] 生成候选池...')
        report = self._generate_report(coverage)
        lines.append(report)

        result = '\n'.join(lines)

        # 保存
        self._save_report(result)
        self._save_log()

        return result

    def _attach_sectors(self):
        """为所有股票标注行业"""
        self.all_df['sector'] = self.all_df['code'].apply(
            lambda c: self.sector_map.get(c, '未知')
        )

    def _aggregate_top_gainers(self):
        """涨幅前50聚合"""
        df = self.all_df.sort_values('pct_chg', ascending=False)
        self.top50 = df.head(50).to_dict('records')

        for stock in self.top50:
            sector = stock.get('sector', '未知')
            if sector and sector != '未知':
                self.sector_count[sector] += 1
                self.sector_stocks[sector].append({
                    'code': stock['code'],
                    'name': stock.get('name', ''),
                    'pct_chg': stock.get('pct_chg', 0),
                    'price': stock.get('price', 0),
                })

    def _identify_disaster_zones(self):
        """重灾区识别: 跌幅前50"""
        df = self.all_df.sort_values('pct_chg', ascending=True)
        self.bottom50 = df.head(50).to_dict('records')

        sector_big_drop = Counter()
        sector_drop_info = defaultdict(list)

        for stock in self.bottom50:
            sector = stock.get('sector', '未知')
            pct = stock.get('pct_chg', 0)
            if sector and sector != '未知':
                sector_big_drop[sector] += 1
                if pct < -5:
                    sector_drop_info[sector].append({
                        'code': stock['code'],
                        'name': stock.get('name', ''),
                        'pct_chg': pct,
                    })

        for sector, count in sector_big_drop.items():
            if count >= 3 or sector_drop_info.get(sector):
                avg_pct = np.mean([s.get('pct_chg', 0) for s in
                                   self.bottom50
                                   if s.get('sector') == sector])
                self.disaster_sectors[sector] = {
                    'count': count,
                    'big_drop': sector_drop_info.get(sector, []),
                    'avg_pct': avg_pct,
                }

    def _check_sector_indices(self):
        """查询TOP5板块指数MA状态"""
        top_sectors = [s for s, _ in self.sector_count.most_common(5)]
        for sector in top_sectors:
            status = get_sector_index_status(sector)
            self.sector_index_cache[sector] = status

    def _generate_report(self, coverage: float) -> str:
        """生成完整报告"""
        lines = []

        # ━━━ 数据概况 ━━━
        lines.append('━━━ 数据概况 ━━━')
        lines.append(f'涨幅前50样本: {len(self.top50)}只')
        lines.append(f'行业数据来源: full_sector_map.json')
        lines.append(f'行业覆盖率: {coverage:.0f}%')

        idx_available = sum(1 for v in self.sector_index_cache.values()
                            if v['status'] == 'OK')
        idx_total = len(self.sector_index_cache)
        if idx_total > 0:
            lines.append(f'板块指数状态: {idx_available}/{idx_total}可查')
        else:
            lines.append('板块指数状态: 未查询')

        # ━━━ 板块强度排名 ━━━
        lines.append('')
        lines.append('━━━ 板块强度排名（涨幅前50聚合） ━━━')
        top15 = self.sector_count.most_common(15)
        for i, (sector, count) in enumerate(top15):
            stocks = self.sector_stocks[sector][:3]
            rep = '、'.join(s['name'] for s in stocks if s.get('name'))
            lines.append(f'  {i+1:2d}. {sector:<10s} — {count}次/50 — 代表: {rep}')

        # ━━━ 重灾区 ━━━
        lines.append('')
        lines.append('━━━ 重灾区板块（跌幅聚合） ━━━')
        if self.disaster_sectors:
            sorted_disaster = sorted(
                self.disaster_sectors.items(),
                key=lambda x: x[1]['avg_pct'])
            for i, (sector, info) in enumerate(sorted_disaster):
                big = info['big_drop']
                if big:
                    names = '、'.join(s['name'] for s in big[:3] if s.get('name'))
                    lines.append(f'  {i+1}. {sector} — '
                                 f'板块内{len(big)}只跌幅>5% ({names})')
                else:
                    lines.append(f'  {i+1}. {sector} — '
                                 f'跌幅前50中{info["count"]}只 '
                                 f'(均跌{info["avg_pct"]:.1f}%)')
        else:
            lines.append('  今日无明显重灾区')

        # ━━━ 候选池分类 ━━━
        lines.append('')
        lines.append('━━━ 板块候选池 ━━━')

        main_sectors = []
        aux_sectors = []

        for sector, count in top15:
            is_disaster = sector in self.disaster_sectors
            idx_info = self.sector_index_cache.get(sector, {})
            big_drop = is_disaster

            if is_disaster:
                continue

            rank = self.sector_count[sector]
            if rank <= 5:
                idx_status = idx_info.get('status', '未知')
                above_ma20 = idx_info.get('above_ma20')
                if idx_status == 'OK' and above_ma20 is False:
                    aux_sectors.append((sector, count, idx_info))
                else:
                    main_sectors.append((sector, count, idx_info))
            else:
                aux_sectors.append((sector, count, idx_info))

        # 主线
        lines.append('✅ 主线板块（标准仓位）:')
        if main_sectors:
            for i, (sector, count, idx_info) in enumerate(main_sectors):
                ma_note = ''
                if idx_info.get('status') == 'OK':
                    above = 'MA20上方' if idx_info.get('above_ma20') else 'MA20下方'
                    ma5_dir = 'MA5>MA20' if idx_info.get('ma5_above_ma20') else 'MA5<MA20'
                    ma_note = f' ({above}, {ma5_dir})'
                else:
                    ma_note = f' (板块均线状态待确认)'
                lines.append(f'  {i+1}. {sector}{ma_note}')
        else:
            lines.append('  无')

        # 辅线
        lines.append('')
        lines.append('⚠️ 辅线板块（7折仓位）:')
        if aux_sectors:
            for i, (sector, count, idx_info) in enumerate(aux_sectors):
                lines.append(f'  {i+1}. {sector} ({count}次/50)')
        else:
            lines.append('  无')

        # 重灾区
        lines.append('')
        lines.append('❌ 重灾区板块（禁止开仓）:')
        if self.disaster_sectors:
            for sector, info in sorted(self.disaster_sectors.items(),
                                       key=lambda x: x[1]['avg_pct']):
                lines.append(f'  · {sector} — 均跌{info["avg_pct"]:.1f}%')
        else:
            lines.append('  无')

        # ━━━ 选股范围 ━━━
        all_pool = main_sectors + aux_sectors
        pool_names = set(s[0] for s in all_pool)
        disaster_names = set(self.disaster_sectors.keys())

        total_stocks = len(self.all_df)
        pool_stocks = len(self.all_df[
            self.all_df['sector'].isin(pool_names)])
        disaster_stocks = len(self.all_df[
            self.all_df['sector'].isin(disaster_names)])

        lines.append('')
        lines.append('━━━ 选股范围 ━━━')
        lines.append(f'操作范围: 主线+辅线 ({len(pool_names)}个板块, '
                     f'{pool_stocks}只候选)')
        lines.append(f'排除范围: 重灾区 ({len(disaster_names)}个板块, '
                     f'{disaster_stocks}只)')

        # ━━━ 明日建议 ━━━
        lines.append('')
        lines.append('━━━ 明日操作建议 ━━━')
        suggestions = self._generate_suggestions(main_sectors, aux_sectors)
        for s in suggestions:
            lines.append(f'  · {s}')

        # ━━━ 数据完整性 ━━━
        lines.append('')
        lines.append('━━━ 数据完整性 ━━━')
        lines.append(f'板块指数代码: '
                     f'{idx_available}/{idx_total}可查' if idx_total else
                     '板块指数: 未查询')
        lines.append(f'行业信息完整性: {coverage:.0f}%')

        return '\n'.join(lines)

    def _generate_suggestions(self, main_sectors, aux_sectors) -> List[str]:
        """生成明日操作建议"""
        suggestions = []

        for sector, count, idx_info in main_sectors[:3]:
            stocks = self.sector_stocks.get(sector, [])
            if stocks:
                top_stock = stocks[0]
                name = top_stock.get('name', top_stock['code'])
                pct = top_stock.get('pct_chg', 0)
                if pct > 9:
                    suggestions.append(
                        f'{sector}: {name} 今日涨停(+{pct:.1f}%), '
                        f'关注次日接力/回踩机会')
                else:
                    suggestions.append(
                        f'{sector}: {name} 今日+{pct:.1f}%, '
                        f'关注MA5支撑回踩买点')

        for sector, count, idx_info in aux_sectors[:2]:
            suggestions.append(
                f'{sector}: 持续性待观察, 次日不追高, 等回调确认')

        for sector in list(self.disaster_sectors.keys())[:2]:
            suggestions.append(f'回避: {sector} (今日重灾区)')

        if not suggestions:
            suggestions.append('今日无明显主线, 建议观望')

        return suggestions

    def _save_report(self, report: str):
        """保存完整报告"""
        os.makedirs('signals', exist_ok=True)
        path = f'signals/sector_pool_{self.today}.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'\n报告已保存: {path}')

        # 同时保存JSON
        json_path = f'signals/sector_pool_{self.today}.json'
        data = {
            'date': self.today,
            'market_score': getattr(self, 'market_score', 6),
            'top15': [
                {'sector': s, 'count': c}
                for s, c in self.sector_count.most_common(15)
            ],
            'main_sectors': [s[0] for s in
                             self._classify_main(self.sector_count.most_common(15))],
            'aux_sectors': [s[0] for s in
                            self._classify_aux(self.sector_count.most_common(15))],
            'disaster_sectors': {
                s: {'count': info['count'], 'avg_pct': round(info['avg_pct'], 2)}
                for s, info in self.disaster_sectors.items()
            },
            'sector_stocks': {
                s: stocks for s, stocks in self.sector_stocks.items()
            },
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f'数据已保存: {json_path}')

    def _classify_main(self, top15):
        """分类主线"""
        result = []
        for sector, count in top15:
            if sector in self.disaster_sectors:
                continue
            if count <= 5:
                result.append((sector, count))
        return result

    def _classify_aux(self, top15):
        """分类辅线"""
        result = []
        main = set(s[0] for s in self._classify_main(top15))
        for sector, count in top15:
            if sector in self.disaster_sectors:
                continue
            if sector not in main:
                result.append((sector, count))
        return result

    def _save_log(self):
        """追加到板块候选池日志"""
        log_dir = 'workspace/memory'
        log_path = f'{log_dir}/板块候选池日志.md'
        os.makedirs(log_dir, exist_ok=True)

        main_names = '、'.join(s[0] for s in
                               self._classify_main(
                                   self.sector_count.most_common(15))[:3])
        aux_names = '、'.join(s[0] for s in
                              self._classify_aux(
                                  self.sector_count.most_common(15))[:3])
        disaster_names = '、'.join(list(self.disaster_sectors.keys())[:3])
        main_count = len(self._classify_main(self.sector_count.most_common(15)))

        row = (f'| {self.today} | {main_names or "无"} | '
               f'{aux_names or "无"} | {disaster_names or "无"} | '
               f'{main_count}只主线 | - |\n')

        header = ('| 日期 | 主线板块 | 辅线板块 | 重灾区板块 | 仓位建议 | 备注 |\n'
                  '|------|---------|---------|-----------|---------|------|\n')

        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if '日期' not in content:
                content = header + content
            content += row
        else:
            content = header + row

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'日志已追加: {log_path}')

    def get_main_theme_sectors(self) -> List[str]:
        """供scan_enhanced_v3调用的接口: 返回今日主线板块名列表"""
        if not self.sector_count:
            self.run()
        main = self._classify_main(self.sector_count.most_common(15))
        return [s[0] for s in main]


# ==================== CLI ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='每日板块候选池扫描')
    parser.add_argument('--push', action='store_true',
                        help='推送结果到飞书')
    args = parser.parse_args()

    scanner = SectorScanner()
    report = scanner.run()
    print('\n' + report)

    if args.push:
        from trading_agents.pre_market import send_notification
        title = f'板块候选池 · {scanner.today}'
        send_notification(title, report)
