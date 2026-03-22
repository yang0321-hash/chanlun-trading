"""
形态记忆系统
记录每种缠论形态的历史表现，学习哪些形态更可靠
"""
import json
import pickle
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY_1 = "1买"
    BUY_2 = "2买"
    BUY_3 = "3买"
    SELL_1 = "1卖"
    SELL_2 = "2卖"
    SELL_3 = "3卖"


@dataclass
class PatternRecord:
    """
    单个形态记录
    记录一次具体的形态识别和交易结果
    """
    # 形态标识
    pattern_id: str  # 形态唯一标识
    signal_type: SignalType
    symbol: str

    # 形态特征
    datetime: datetime
    price: float
    confidence: float

    # 形态细节
    features: Dict[str, Any] = field(default_factory=dict)  # 形态特征
    market_condition: str = ""  # 市场状态：上涨/下跌/震荡

    # 交易结果
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_days: int = 0
    exit_reason: str = ""

    # 状态
    is_completed: bool = False  # 是否已平仓
    is_winner: bool = False  # 是否盈利

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['signal_type'] = self.signal_type.value
        data['datetime'] = self.datetime.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternRecord":
        """从字典创建"""
        data['signal_type'] = SignalType(data['signal_type'])
        data['datetime'] = datetime.fromisoformat(data['datetime'])
        return cls(**data)


@dataclass
class PatternStats:
    """形态统计信息"""
    pattern_id: str
    signal_type: SignalType
    total_count: int = 0
    win_count: int = 0
    lose_count: int = 0
    avg_pnl_pct: float = 0.0
    max_profit_pct: float = 0.0
    max_loss_pct: float = 0.0
    avg_hold_days: float = 0.0
    win_rate: float = 0.0
    confidence_score: float = 0.5  # 综合信心度 0-1

    def update(self, record: PatternRecord):
        """更新统计"""
        self.total_count += 1

        if record.is_completed:
            if record.is_winner:
                self.win_count += 1
            else:
                self.lose_count += 1

            # 更新平均盈亏
            n = self.total_count
            self.avg_pnl_pct = (self.avg_pnl_pct * (n - 1) + record.pnl_pct) / n

            # 更新最大盈亏
            self.max_profit_pct = max(self.max_profit_pct, record.pnl_pct)
            self.max_loss_pct = min(self.max_loss_pct, record.pnl_pct)

            # 更新平均持仓天数
            self.avg_hold_days = (self.avg_hold_days * (n - 1) + record.hold_days) / n

            # 更新胜率
            self.win_rate = self.win_count / n if n > 0 else 0

            # 更新综合信心度 (胜率 * 盈利比)
            profit_ratio = (1 + self.avg_pnl_pct) if self.avg_pnl_pct > 0 else (1 + self.avg_pnl_pct)
            self.confidence_score = self.win_rate * min(profit_ratio, 2) / 2


class PatternMemory:
    """
    形态记忆系统
    记录和学习历史形态表现
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./memory/patterns")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.records: List[PatternRecord] = []
        self.stats: Dict[str, PatternStats] = {}

        self._load()

    def add_pattern(
        self,
        pattern_id: str,
        signal_type: SignalType,
        symbol: str,
        datetime: datetime,
        price: float,
        confidence: float,
        features: Optional[Dict[str, Any]] = None,
        market_condition: str = ""
    ) -> PatternRecord:
        """添加新形态记录"""
        record = PatternRecord(
            pattern_id=pattern_id,
            signal_type=signal_type,
            symbol=symbol,
            datetime=datetime,
            price=price,
            confidence=confidence,
            features=features or {},
            market_condition=market_condition
        )
        self.records.append(record)
        self._update_stats(record)
        self._save()
        return record

    def complete_trade(
        self,
        record: PatternRecord,
        exit_price: float,
        exit_reason: str,
        hold_days: int
    ):
        """完成交易，记录结果"""
        record.exit_price = exit_price
        record.exit_reason = exit_reason
        record.hold_days = hold_days
        record.is_completed = True

        # 计算盈亏
        if record.entry_price > 0:
            record.pnl_pct = (exit_price - record.entry_price) / record.entry_price
            record.pnl = record.pnl_pct * record.entry_price
            record.is_winner = record.pnl_pct > 0

        self._update_stats(record)
        self._save()

    def get_pattern_stats(self, pattern_id: str) -> Optional[PatternStats]:
        """获取特定形态的统计"""
        return self.stats.get(pattern_id)

    def get_signal_type_stats(self, signal_type: SignalType) -> Dict[str, PatternStats]:
        """获取特定信号类型的所有统计"""
        return {
            pid: stat for pid, stat in self.stats.items()
            if stat.signal_type == signal_type
        }

    def get_best_patterns(self, signal_type: Optional[SignalType] = None,
                          min_count: int = 5) -> List[PatternStats]:
        """获取表现最好的形态"""
        all_stats = list(self.stats.values())

        if signal_type:
            all_stats = [s for s in all_stats if s.signal_type == signal_type]

        # 过滤样本数足够的
        all_stats = [s for s in all_stats if s.total_count >= min_count]

        # 按信心度排序
        return sorted(all_stats, key=lambda x: x.confidence_score, reverse=True)

    def get_worst_patterns(self, signal_type: Optional[SignalType] = None,
                           min_count: int = 5) -> List[PatternStats]:
        """获取表现最差的形态"""
        all_stats = list(self.stats.values())

        if signal_type:
            all_stats = [s for s in all_stats if s.signal_type == signal_type]

        all_stats = [s for s in all_stats if s.total_count >= min_count]

        return sorted(all_stats, key=lambda x: x.confidence_score)

    def get_pending_records(self) -> List[PatternRecord]:
        """获取未完成的记录"""
        return [r for r in self.records if not r.is_completed]

    def get_recent_records(self, n: int = 10) -> List[PatternRecord]:
        """获取最近的记录"""
        return sorted(self.records, key=lambda x: x.datetime, reverse=True)[:n]

    def get_confidence(self, pattern_id: str) -> float:
        """获取形态的信心度"""
        stat = self.stats.get(pattern_id)
        return stat.confidence_score if stat else 0.5

    def adjust_confidence(self, pattern_id: str, base_confidence: float) -> float:
        """
        根据历史表现调整信心度
        如果历史表现好，提高信心度；表现差，降低信心度
        """
        stat = self.stats.get(pattern_id)
        if not stat or stat.total_count < 3:
            return base_confidence

        # 历史信心度与当前信心度的加权平均
        # 样本越多，历史权重越大
        history_weight = min(stat.total_count / 20, 0.7)
        adjusted = (base_confidence * (1 - history_weight) +
                   stat.confidence_score * history_weight)

        return max(0.1, min(0.95, adjusted))

    def _update_stats(self, record: PatternRecord):
        """更新统计信息"""
        if record.pattern_id not in self.stats:
            self.stats[record.pattern_id] = PatternStats(
                pattern_id=record.pattern_id,
                signal_type=record.signal_type
            )
        self.stats[record.pattern_id].update(record)

    def _save(self):
        """保存到文件"""
        # 保存记录
        records_file = self.storage_path / "records.pkl"
        with open(records_file, 'wb') as f:
            pickle.dump(self.records, f)

        # 保存统计
        stats_file = self.storage_path / "stats.json"
        stats_data = {}
        for pid, stat in self.stats.items():
            stats_data[pid] = {
                'pattern_id': stat.pattern_id,
                'signal_type': stat.signal_type.value,
                'total_count': stat.total_count,
                'win_count': stat.win_count,
                'lose_count': stat.lose_count,
                'avg_pnl_pct': stat.avg_pnl_pct,
                'max_profit_pct': stat.max_profit_pct,
                'max_loss_pct': stat.max_loss_pct,
                'avg_hold_days': stat.avg_hold_days,
                'win_rate': stat.win_rate,
                'confidence_score': stat.confidence_score,
            }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从文件加载"""
        # 加载记录
        records_file = self.storage_path / "records.pkl"
        if records_file.exists():
            with open(records_file, 'rb') as f:
                self.records = pickle.load(f)

        # 加载统计
        stats_file = self.storage_path / "stats.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)

            for pid, data in stats_data.items():
                self.stats[pid] = PatternStats(
                    pattern_id=data['pattern_id'],
                    signal_type=SignalType(data['signal_type']),
                    total_count=data['total_count'],
                    win_count=data['win_count'],
                    lose_count=data['lose_count'],
                    avg_pnl_pct=data['avg_pnl_pct'],
                    max_profit_pct=data['max_profit_pct'],
                    max_loss_pct=data['max_loss_pct'],
                    avg_hold_days=data['avg_hold_days'],
                    win_rate=data['win_rate'],
                    confidence_score=data['confidence_score'],
                )

    def generate_report(self) -> str:
        """生成记忆报告"""
        lines = ["=" * 60, "缠论形态记忆报告", "=" * 60]

        # 总体统计
        total = len(self.records)
        completed = len([r for r in self.records if r.is_completed])
        pending = total - completed

        lines.append(f"\n总记录数: {total}")
        lines.append(f"已完成: {completed}")
        lines.append(f"未完成: {pending}")

        # 按信号类型统计
        lines.append("\n" + "-" * 40)
        lines.append("按信号类型统计:")
        for signal_type in SignalType:
            type_stats = self.get_signal_type_stats(signal_type)
            if type_stats:
                total_count = sum(s.total_count for s in type_stats.values())
                win_count = sum(s.win_count for s in type_stats.values())
                win_rate = win_count / total_count if total_count > 0 else 0
                avg_pnl = sum(s.avg_pnl_pct for s in type_stats.values()) / len(type_stats)
                lines.append(f"  {signal_type.value}: {total_count}次, 胜率{win_rate:.1%}, 平均盈亏{avg_pnl:.2%}")

        # 最佳形态
        lines.append("\n" + "-" * 40)
        lines.append("表现最佳形态 (至少5次):")
        best = self.get_best_patterns(min_count=5)[:5]
        for stat in best:
            lines.append(f"  {stat.pattern_id}: {stat.total_count}次, "
                        f"胜率{stat.win_rate:.1%}, "
                        f"平均盈亏{stat.avg_pnl_pct:.2%}, "
                        f"信心度{stat.confidence_score:.2f}")

        # 最差形态
        lines.append("\n" + "-" * 40)
        lines.append("表现最差形态 (至少5次):")
        worst = self.get_worst_patterns(min_count=5)[:5]
        for stat in worst:
            lines.append(f"  {stat.pattern_id}: {stat.total_count}次, "
                        f"胜率{stat.win_rate:.1%}, "
                        f"平均盈亏{stat.avg_pnl_pct:.2%}, "
                        f"信心度{stat.confidence_score:.2f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
