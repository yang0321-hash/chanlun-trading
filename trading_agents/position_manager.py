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

    def partial_sell(self, code: str, ratio: float = 0.5) -> Optional[Position]:
        """部分减仓, 返回修改后的Position (ratio=卖出比例, 如0.5=减半)

        若减完后剩余为0则完全清仓。
        """
        pos = self.positions.get(code)
        if not pos:
            return None

        sell_shares = int(pos.shares * ratio)
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
