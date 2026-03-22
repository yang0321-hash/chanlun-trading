"""
交易记忆系统
记录每次交易的详细信息，支持复盘和分析
"""
import json
import pickle
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from enum import Enum


class TradeStatus(Enum):
    """交易状态"""
    PENDING = "待成交"
    OPEN = "持仓中"
    CLOSED = "已平仓"
    CANCELLED = "已取消"


class TradeSide(Enum):
    """交易方向"""
    LONG = "做多"
    SHORT = "做空"


@dataclass
class TradeRecord:
    """
    单笔交易记录
    """
    # 基本信息
    trade_id: str
    symbol: str
    side: TradeSide
    status: TradeStatus

    # 开仓信息
    entry_datetime: datetime
    entry_price: float
    quantity: int
    entry_reason: str = ""

    # 平仓信息
    exit_datetime: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""

    # 盈亏信息
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0

    # 风险管理
    stop_loss: float = 0.0
    take_profit: float = 0.0
    max_adverse_excursion: float = 0.0  # 最大不利偏移
    max_favorable_excursion: float = 0.0  # 最大有利偏移

    # 关联信息
    pattern_id: str = ""  # 关联的形态ID
    signal_type: str = ""  # 信号类型

    # 标签
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def is_long(self) -> bool:
        return self.side == TradeSide.LONG

    def is_short(self) -> bool:
        return self.side == TradeSide.SHORT

    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    def is_closed(self) -> bool:
        return self.status == TradeStatus.CLOSED

    def holding_days(self) -> int:
        """持仓天数"""
        if not self.exit_datetime:
            return (date.today() - self.entry_datetime.date()).days
        return (self.exit_datetime.date() - self.entry_datetime.date()).days

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['side'] = self.side.value
        data['status'] = self.status.value
        data['entry_datetime'] = self.entry_datetime.isoformat()
        if self.exit_datetime:
            data['exit_datetime'] = self.exit_datetime.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        """从字典创建"""
        data['side'] = TradeSide(data['side'])
        data['status'] = TradeStatus(data['status'])
        data['entry_datetime'] = datetime.fromisoformat(data['entry_datetime'])
        if data.get('exit_datetime'):
            data['exit_datetime'] = datetime.fromisoformat(data['exit_datetime'])
        return cls(**data)


class TradeMemory:
    """
    交易记忆系统
    记录所有交易历史，支持查询和分析
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./memory/trades")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.trades: List[TradeRecord] = []
        self._trade_counter = 0

        self._load()

    def create_trade(
        self,
        symbol: str,
        side: TradeSide,
        entry_price: float,
        quantity: int,
        entry_reason: str = "",
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        pattern_id: str = "",
        signal_type: str = ""
    ) -> TradeRecord:
        """创建新交易"""
        self._trade_counter += 1
        trade = TradeRecord(
            trade_id=f"T{datetime.now().strftime('%Y%m%d%H%M%S')}{self._trade_counter:03d}",
            symbol=symbol,
            side=side,
            status=TradeStatus.OPEN,
            entry_datetime=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            entry_reason=entry_reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_id=pattern_id,
            signal_type=signal_type
        )
        self.trades.append(trade)
        self._save()
        return trade

    def close_trade(
        self,
        trade: TradeRecord,
        exit_price: float,
        exit_reason: str = "",
        commission: float = 0.0
    ):
        """平仓"""
        trade.exit_price = exit_price
        trade.exit_datetime = datetime.now()
        trade.exit_reason = exit_reason
        trade.status = TradeStatus.CLOSED
        trade.commission = commission

        # 计算盈亏
        if trade.quantity > 0:
            if trade.is_long():
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity - commission
            else:
                trade.pnl = (trade.entry_price - exit_price) * trade.quantity - commission

            trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
        else:
            # 如果没有设置数量，只计算百分比盈亏
            if trade.is_long():
                trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
            trade.pnl = trade.pnl_pct * trade.entry_price  # 假设1股

        self._save()

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """获取交易"""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def get_open_trades(self) -> List[TradeRecord]:
        """获取所有持仓"""
        return [t for t in self.trades if t.is_open()]

    def get_closed_trades(self) -> List[TradeRecord]:
        """获取已平仓交易"""
        return [t for t in self.trades if t.is_closed()]

    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """获取某股票的所有交易"""
        return [t for t in self.trades if t.symbol == symbol]

    def get_trades_by_pattern(self, pattern_id: str) -> List[TradeRecord]:
        """获取某形态的所有交易"""
        return [t for t in self.trades if t.pattern_id == pattern_id]

    def get_trades_by_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[TradeRecord]:
        """获取日期范围内的交易"""
        return [
            t for t in self.trades
            if start_date <= t.entry_datetime.date() <= end_date
        ]

    def get_profitable_trades(self) -> List[TradeRecord]:
        """获取盈利交易"""
        return [t for t in self.trades if t.status == TradeStatus.CLOSED and t.pnl > 0]

    def get_losing_trades(self) -> List[TradeRecord]:
        """获取亏损交易"""
        return [t for t in self.trades if t.status == TradeStatus.CLOSED and t.pnl < 0]

    def get_statistics(self) -> Dict[str, Any]:
        """获取交易统计"""
        closed = self.get_closed_trades()
        if not closed:
            return {
                "total_trades": len(self.trades),
                "closed_trades": 0,
                "open_trades": len(self.get_open_trades())
            }

        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl < 0]

        total_pnl = sum(t.pnl for t in closed)
        avg_pnl = total_pnl / len(closed)

        win_rate = len(winners) / len(closed)

        avg_win = sum(t.pnl for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t.pnl for t in losers) / len(losers) if losers else 0

        profit_factor = abs(sum(t.pnl for t in winners) / sum(t.pnl for t in losers)) if losers else 0

        max_profit = max(t.pnl for t in winners) if winners else 0
        max_loss = min(t.pnl for t in losers) if losers else 0

        avg_hold_days = sum(t.holding_days() for t in closed) / len(closed)

        return {
            "total_trades": len(self.trades),
            "closed_trades": len(closed),
            "open_trades": len(self.get_open_trades()),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "avg_hold_days": avg_hold_days,
        }

    def _save(self):
        """保存到文件"""
        trades_file = self.storage_path / "trades.pkl"
        with open(trades_file, 'wb') as f:
            pickle.dump((self.trades, self._trade_counter), f)

        # 导出JSON
        json_file = self.storage_path / "trades.json"
        trades_data = [t.to_dict() for t in self.trades]
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(trades_data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从文件加载"""
        trades_file = self.storage_path / "trades.pkl"
        if trades_file.exists():
            with open(trades_file, 'rb') as f:
                self.trades, self._trade_counter = pickle.load(f)

    def generate_report(self) -> str:
        """生成交易报告"""
        stats = self.get_statistics()
        lines = ["=" * 60, "交易记忆报告", "=" * 60]

        lines.append(f"\n总交易数: {stats['total_trades']}")
        lines.append(f"已平仓: {stats['closed_trades']}")
        lines.append(f"持仓中: {stats['open_trades']}")

        if stats['closed_trades'] > 0:
            lines.append("\n" + "-" * 40)
            lines.append("盈亏统计:")
            lines.append(f"  盈利次数: {stats['winners']}")
            lines.append(f"  亏损次数: {stats['losers']}")
            lines.append(f"  胜率: {stats['win_rate']:.2%}")
            lines.append(f"  总盈亏: {stats['total_pnl']:.2f}")
            lines.append(f"  平均盈亏: {stats['avg_pnl']:.2f}")
            lines.append(f"  平均盈利: {stats['avg_win']:.2f}")
            lines.append(f"  平均亏损: {stats['avg_loss']:.2f}")
            lines.append(f"  盈亏比: {stats['profit_factor']:.2f}")
            lines.append(f"  最大盈利: {stats['max_profit']:.2f}")
            lines.append(f"  最大亏损: {stats['max_loss']:.2f}")
            lines.append(f"  平均持仓天数: {stats['avg_hold_days']:.1f}")

        # 最近交易
        lines.append("\n" + "-" * 40)
        lines.append("最近5笔交易:")
        recent = sorted(self.trades, key=lambda x: x.entry_datetime, reverse=True)[:5]
        for trade in recent:
            status_emoji = "🟢" if trade.status == TradeStatus.OPEN else ("✅" if trade.pnl > 0 else "❌")
            pnl_str = f"PnL:{trade.pnl_pct:.2%}" if trade.is_closed() else trade.status.value
            lines.append(f"  {status_emoji} {trade.trade_id} {trade.symbol} "
                        f"{trade.side.value} @ {trade.entry_price:.2f} {pnl_str}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
