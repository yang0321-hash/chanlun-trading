"""
记忆管理器
统一管理形态记忆和交易记忆
"""
from typing import Optional, Dict, Any
from pathlib import Path

from .pattern_memory import PatternMemory, PatternRecord, SignalType
from .trade_memory import TradeMemory, TradeRecord, TradeSide


class MemoryManager:
    """
    记忆管理器
    作为记忆系统的统一入口，协调形态记忆和交易记忆
    """

    def __init__(self, storage_path: Optional[str] = None):
        base_path = Path(storage_path) if storage_path else Path("./memory")
        base_path.mkdir(parents=True, exist_ok=True)

        self.pattern_memory = PatternMemory(base_path / "patterns")
        self.trade_memory = TradeMemory(base_path / "trades")

    def record_signal(
        self,
        pattern_id: str,
        signal_type: SignalType,
        symbol: str,
        price: float,
        confidence: float,
        features: Optional[Dict[str, Any]] = None,
        market_condition: str = ""
    ) -> tuple[PatternRecord, TradeRecord]:
        """
        记录信号并创建交易
        同时更新形态记忆和交易记忆
        """
        from datetime import datetime

        # 1. 添加到形态记忆
        pattern_record = self.pattern_memory.add_pattern(
            pattern_id=pattern_id,
            signal_type=signal_type,
            symbol=symbol,
            datetime=datetime.now(),
            price=price,
            confidence=confidence,
            features=features,
            market_condition=market_condition
        )

        # 2. 确定交易方向
        side = TradeSide.LONG if signal_type.value.startswith("买") else TradeSide.SHORT

        # 3. 创建交易记录
        trade_record = self.trade_memory.create_trade(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=0,  # 后续设置
            entry_reason=f"{signal_type.value}: {pattern_id}",
            pattern_id=pattern_id,
            signal_type=signal_type.value
        )

        return pattern_record, trade_record

    def close_position(
        self,
        trade: TradeRecord,
        exit_price: float,
        exit_reason: str = ""
    ):
        """平仓并更新记忆"""
        # 更新交易记忆
        self.trade_memory.close_trade(trade, exit_price, exit_reason)

        # 更新形态记忆
        if trade.pattern_id:
            pattern_record = None
            for pr in self.pattern_memory.records:
                if pr.pattern_id == trade.pattern_id and not pr.is_completed:
                    pattern_record = pr
                    break

            if pattern_record:
                self.pattern_memory.complete_trade(
                    record=pattern_record,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    hold_days=trade.holding_days()
                )

    def get_adjusted_confidence(self, pattern_id: str, base_confidence: float) -> float:
        """获取根据历史调整后的信心度"""
        return self.pattern_memory.adjust_confidence(pattern_id, base_confidence)

    def get_full_report(self) -> str:
        """获取完整报告"""
        lines = ["#" * 60, "# 缠论系统完整记忆报告", "#" * 60]
        lines.append("\n")
        lines.append(self.pattern_memory.generate_report())
        lines.append("\n")
        lines.append(self.trade_memory.generate_report())
        return "\n".join(lines)

    def export_to_json(self, filepath: str):
        """导出为JSON"""
        import json

        data = {
            "pattern_records": [r.to_dict() for r in self.pattern_memory.records],
            "pattern_stats": {
                pid: {
                    'pattern_id': stat.pattern_id,
                    'signal_type': stat.signal_type.value,
                    'total_count': stat.total_count,
                    'win_count': stat.win_count,
                    'lose_count': stat.lose_count,
                    'avg_pnl_pct': stat.avg_pnl_pct,
                    'win_rate': stat.win_rate,
                    'confidence_score': stat.confidence_score,
                }
                for pid, stat in self.pattern_memory.stats.items()
            },
            "trades": [t.to_dict() for t in self.trade_memory.trades],
            "trade_stats": self.trade_memory.get_statistics(),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
