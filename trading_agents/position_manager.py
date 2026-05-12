#!/usr/bin/env python3
"""持仓管理器 — 自动跟踪买卖、止损、最高价、仓位占比

职责:
  - 买入时自动记录: 入场价、止损、买点类型
  - 每日更新: 最高价、浮动盈亏
  - 卖出时自动清除
  - 持久化到 signals/positions.json
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

POSITIONS_FILE = 'signals/positions.json'


@dataclass
class Position:
    code: str
    name: str
    entry_price: float
    entry_date: str
    shares: int
    stop_price: float
    chan_stop: float          # 缠论结构止损
    buy_point_type: str      # 1buy/2buy/3buy
    highest_since_entry: float = 0.0
    current_stop: float = 0.0
    last_update: str = ''
    sector: str = ''
    strategy_mode: str = ''   # 'v3a_t0', 'v3a_trend' 等
    entry_type: str = ''      # 'initial', 'zg_breakout' 等

    def __post_init__(self):
        if self.highest_since_entry == 0:
            self.highest_since_entry = self.entry_price
        if self.current_stop == 0:
            self.current_stop = self.stop_price
        if not self.last_update:
            self.last_update = datetime.now().strftime('%Y-%m-%d')


class PositionManager:

    def __init__(self, capital: float = 1000000):
        self.capital = capital
        self.positions: Dict[str, Position] = {}
        self._load()

    def _load(self):
        if not os.path.exists(POSITIONS_FILE):
            return
        try:
            with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.capital = data.get('capital', 1000000)
            for p in data.get('positions', []):
                pos = Position(**p)
                self.positions[pos.code] = pos
        except Exception:
            pass

    def _save(self):
        os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)
        data = {
            'capital': self.capital,
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'positions': [asdict(p) for p in self.positions.values()],
        }
        with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def buy(self, code: str, name: str, price: float, shares: int,
            stop_price: float, buy_point_type: str = '',
            chan_stop: float = 0.0, sector: str = '',
            strategy_mode: str = '', entry_type: str = ''):
        """记录买入"""
        if code in self.positions:
            # 加仓: 更新均价和数量
            pos = self.positions[code]
            total_cost = pos.entry_price * pos.shares + price * shares
            total_shares = pos.shares + shares
            pos.entry_price = total_cost / total_shares
            pos.shares = total_shares
            pos.stop_price = min(pos.stop_price, stop_price)
            if chan_stop > 0:
                pos.chan_stop = min(pos.chan_stop, chan_stop) if pos.chan_stop > 0 else chan_stop
            pos.current_stop = pos.stop_price
            if entry_type:
                pos.entry_type = entry_type
        else:
            self.positions[code] = Position(
                code=code, name=name,
                entry_price=price,
                entry_date=datetime.now().strftime('%Y-%m-%d'),
                shares=shares,
                stop_price=stop_price,
                chan_stop=chan_stop,
                buy_point_type=buy_point_type,
                sector=sector,
                strategy_mode=strategy_mode,
                entry_type=entry_type or 'initial',
            )
        self._save()

    def sell(self, code: str) -> Optional[Position]:
        """记录卖出, 返回已平仓的Position"""
        pos = self.positions.pop(code, None)
        if pos:
            self._save()
        return pos

    def partial_sell(self, code: str, ratio: float = 0.5,
                     quantity: int = 0) -> Optional[Position]:
        """部分减仓, 返回修改后的Position

        Args:
            ratio: 卖出比例 (如0.5=减半), 当quantity>0时忽略
            quantity: 卖出股数, 0则用ratio计算

        若减完后剩余为0则完全清仓。
        """
        pos = self.positions.get(code)
        if not pos:
            return None

        sell_shares = quantity if quantity > 0 else int(pos.shares * ratio)
        if sell_shares <= 0:
            return pos

        remaining = pos.shares - sell_shares
        if remaining <= 0:
            return self.sell(code)

        pos.shares = remaining
        self._save()
        return pos

    def update_prices(self, price_map: Dict[str, float]):
        """更新所有持仓的最高价和浮动盈亏

        Args:
            price_map: {code: current_price}
        """
        for code, price in price_map.items():
            if code not in self.positions:
                continue
            pos = self.positions[code]
            if price > pos.highest_since_entry:
                pos.highest_since_entry = price
            pos.last_update = datetime.now().strftime('%Y-%m-%d')
        self._save()

    def update_stops(self, stop_map: Dict[str, float]):
        """更新止损价 (上移止损时用)

        Args:
            stop_map: {code: new_stop_price}
        """
        for code, new_stop in stop_map.items():
            if code not in self.positions:
                continue
            pos = self.positions[code]
            # 止损只能上移，不能下移
            if new_stop > pos.current_stop:
                pos.current_stop = new_stop
        self._save()

    def get_position(self, code: str) -> Optional[Position]:
        return self.positions.get(code)

    def get_all_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_total_value(self, price_map: Dict[str, float]) -> float:
        """总持仓市值"""
        total = 0
        for code, pos in self.positions.items():
            price = price_map.get(code, pos.entry_price)
            total += price * pos.shares
        return total

    def get_total_pnl(self, price_map: Dict[str, float]) -> float:
        """总盈亏金额"""
        pnl = 0
        for code, pos in self.positions.items():
            price = price_map.get(code, pos.entry_price)
            pnl += (price - pos.entry_price) * pos.shares
        return pnl

    def get_sector_concentration(self) -> Dict[str, int]:
        """行业集中度统计"""
        sectors = {}
        for pos in self.positions.values():
            s = pos.sector or '未知'
            sectors[s] = sectors.get(s, 0) + 1
        return sectors

    def check_stops(self, price_map: Dict[str, float]) -> List[dict]:
        """检查哪些持仓触发止损

        Returns:
            list of {code, name, price, stop, distance_pct}
        """
        triggered = []
        for code, pos in self.positions.items():
            price = price_map.get(code, 0)
            if price <= 0:
                continue
            stop = pos.current_stop
            if stop > 0 and price <= stop:
                triggered.append({
                    'code': code,
                    'name': pos.name,
                    'price': price,
                    'stop': stop,
                    'pnl_pct': (price - pos.entry_price) / pos.entry_price * 100,
                })
        return triggered

    def check_near_stops(self, price_map: Dict[str, float],
                         threshold_pct: float = 3.0) -> List[dict]:
        """检查接近止损的持仓"""
        warnings = []
        for code, pos in self.positions.items():
            price = price_map.get(code, 0)
            if price <= 0 or pos.current_stop <= 0:
                continue
            distance = (price - pos.current_stop) / price * 100
            if distance <= threshold_pct:
                warnings.append({
                    'code': code,
                    'name': pos.name,
                    'price': price,
                    'stop': pos.current_stop,
                    'distance_pct': distance,
                    'pnl_pct': (price - pos.entry_price) / pos.entry_price * 100,
                })
        return warnings

    # ==================== 回撤控制 (回测验证参数同步) ====================

    MAX_POSITIONS = 5
    DD_REDUCE_THRESHOLD = 0.15
    DD_STOP_THRESHOLD = 0.25
    SECTOR_LIMIT = 1

    def _get_or_init_peak_capital(self) -> float:
        meta_path = POSITIONS_FILE.replace('positions.json', 'portfolio_meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                return meta.get('peak_capital', self.capital)
            except Exception:
                pass
        return self.capital

    def _save_peak_capital(self, peak: float):
        meta_path = POSITIONS_FILE.replace('positions.json', 'portfolio_meta.json')
        os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({'peak_capital': peak,
                           'updated': datetime.now().strftime('%Y-%m-%d')},
                          f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update_drawdown(self, price_map=None):
        if price_map:
            position_value = self.get_total_value(price_map)
        else:
            position_value = sum(
                p.entry_price * p.shares for p in self.positions.values())
        cash = max(0, self.capital - position_value)
        current_capital = cash + position_value
        peak = self._get_or_init_peak_capital()
        if current_capital > peak:
            peak = current_capital
            self._save_peak_capital(peak)
        self._current_capital = current_capital
        self._peak_capital = peak

    def get_drawdown(self, price_map=None) -> float:
        if not hasattr(self, '_peak_capital'):
            self.update_drawdown(price_map)
        if self._peak_capital <= 0:
            return 0.0
        return (self._current_capital - self._peak_capital) / self._peak_capital

    def get_position_count(self) -> int:
        return len(self.positions)

    def get_dd_scale(self, price_map=None) -> float:
        dd = self.get_drawdown(price_map)
        if dd < -self.DD_STOP_THRESHOLD:
            return 0.0
        elif dd < -self.DD_REDUCE_THRESHOLD:
            return 0.5
        return 1.0

    def can_open_position(self, sector: str = '',
                          price_map=None) -> Tuple[bool, str]:
        if len(self.positions) >= self.MAX_POSITIONS:
            return False, f'持仓已满({len(self.positions)}/{self.MAX_POSITIONS})'
        dd = self.get_drawdown(price_map)
        if dd < -self.DD_STOP_THRESHOLD:
            return False, f'回撤过深({dd:.1%}), 停止开仓'
        if sector:
            if sector in self.get_sectors_in_use():
                return False, f'行业{sector}已有持仓(限{self.SECTOR_LIMIT}只)'
        return True, ''

    def is_sector_full(self, sector: str) -> bool:
        if not sector:
            return False
        return any(p.sector == sector for p in self.positions.values() if p.sector)

    def get_sectors_in_use(self) -> set:
        return {p.sector for p in self.positions.values()
                if p.sector and p.sector != '未知'}

    def get_portfolio_status(self, price_map=None) -> dict:
        dd = self.get_drawdown(price_map)
        return {
            'position_count': len(self.positions),
            'max_positions': self.MAX_POSITIONS,
            'drawdown': round(dd, 4),
            'dd_scale': self.get_dd_scale(price_map),
            'sectors_in_use': list(self.get_sectors_in_use()),
            'can_open': dd >= -self.DD_STOP_THRESHOLD,
        }

    # ==================== T+1卖出队列 (A股T+1约束) ====================

    @staticmethod
    def _t1_file():
        return POSITIONS_FILE.replace('positions.json', 't1_sell_queue.json')

    def queue_t1_sell(self, code, sell_type, signal_price, reason=''):
        """将卖出信号加入T+1队列 (当日记录, 次日开盘执行)"""
        queue = self._t1_load()
        queue[code] = {
            'sell_type': sell_type,
            'signal_price': round(signal_price, 3),
            'signal_date': datetime.now().strftime('%Y-%m-%d'),
            'reason': reason,
            'status': 'pending',
        }
        self._t1_save(queue)

    def get_t1_pending(self):
        """获取今天需要执行的T+1卖出 (昨日或更早的pending)"""
        today = datetime.now().strftime('%Y-%m-%d')
        queue = self._t1_load()
        return [{'code': c, **info} for c, info in queue.items()
                if info.get('status') == 'pending' and info.get('signal_date') != today]

    def mark_t1_executed(self, code):
        """标记T+1卖出已执行"""
        queue = self._t1_load()
        if code in queue:
            queue[code]['status'] = 'executed'
            queue[code]['executed_date'] = datetime.now().strftime('%Y-%m-%d')
            self._t1_save(queue)

    def _t1_load(self):
        path = self._t1_file()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _t1_save(self, queue):
        path = self._t1_file()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)
