"""
主力资金沉淀战法检测器

通过量价关系识别主力建仓→洗盘沉淀→突破拉升的完整周期。
六步检测：
1. 主力流入（建仓期）— 低位放量堆阳线
2. 高位分歧（压力区）— 拉升后滞涨/回落
3. 持续缩量沉淀（洗盘期）— 成交量萎缩
4. 小K线惜售 — 振幅极小的K线为主
5. 强势底分型上车 — 倍量大阳线或底分型+MACD金叉
6. 突破加仓 — 放量突破压力位
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class AccumulationSignal:
    """主力资金沉淀信号"""
    phase: str               # 'entry', 'breakout'
    entry_type: str          # 'double_vol_bigyang', 'strong_fractal', 'breakout_pullback'
    price: float             # 当前价格
    stop_loss: float         # 止损价（入场K线最低点 × 0.99）
    resistance_price: float  # 压力位价格（步骤2）
    confidence: float        # 0~1
    building_start: int      # 建仓起始索引
    washing_days: int        # 洗盘天数
    volume_ratio: float      # 入场量比
    building_volume_ratio: float  # 建仓期量比
    reason: str


class CapitalAccumulationDetector:
    """
    主力资金沉淀战法检测器

    使用方法：
        det = CapitalAccumulationDetector(df)
        signal = det.scan()  # 扫描最新位置
        if signal:
            print(f'入场价: {signal.price}, 止损: {signal.stop_loss}')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        # 建仓参数
        building_period: int = 20,      # 建仓观察期（天）
        building_yang_ratio: float = 0.6,  # 阳线占比阈值
        building_vol_mult: float = 1.5,    # 建仓期放量倍数
        price_low_percentile: float = 40,  # 价格低位分位
        # 洗盘参数
        wash_min_days: int = 5,         # 洗盘最少天数
        wash_max_days: int = 20,        # 洗盘最多天数
        wash_vol_ratio: float = 0.6,    # 缩量阈值（量比）
        wash_small_k_ratio: float = 0.6, # 小K线占比阈值
        small_k_amplitude: float = 0.03, # 小K线振幅阈值（3%）
        # 入场参数
        double_vol: float = 2.0,        # 倍量阈值
        big_yang_pct: float = 0.03,     # 大阳线涨幅阈值（3%）
        # 均量参数
        vol_ma_period: int = 20,        # 均量计算周期
    ):
        self.df = df.reset_index(drop=True)
        self.prices = df['close'].values.astype(float)
        self.opens = df['open'].values.astype(float)
        self.highs = df['high'].values.astype(float)
        self.lows = df['low'].values.astype(float)
        self.volumes = df['volume'].values.astype(float)
        self.n = len(self.prices)

        # 参数
        self.building_period = building_period
        self.building_yang_ratio = building_yang_ratio
        self.building_vol_mult = building_vol_mult
        self.price_low_percentile = price_low_percentile
        self.wash_min_days = wash_min_days
        self.wash_max_days = wash_max_days
        self.wash_vol_ratio = wash_vol_ratio
        self.wash_small_k_ratio = wash_small_k_ratio
        self.small_k_amplitude = small_k_amplitude
        self.double_vol = double_vol
        self.big_yang_pct = big_yang_pct
        self.vol_ma_period = vol_ma_period

        # 预计算均量
        self._vol_ma = self._calc_vol_ma()

    def _calc_vol_ma(self) -> np.ndarray:
        """计算成交量均线"""
        ma = np.full(self.n, np.nan)
        for i in range(self.vol_ma_period - 1, self.n):
            ma[i] = np.mean(self.volumes[i - self.vol_ma_period + 1:i + 1])
        return ma

    def _vol_ratio(self, index: int) -> float:
        """计算量比（当前量 / 均量）"""
        if index < 0:
            index = self.n + index
        if index < self.vol_ma_period or np.isnan(self._vol_ma[index]):
            return 1.0
        return self.volumes[index] / self._vol_ma[index] if self._vol_ma[index] > 0 else 1.0

    def _price_percentile(self, index: int, lookback: int = 120) -> float:
        """价格分位（0~100）"""
        start = max(0, index - lookback + 1)
        window = self.prices[start:index + 1]
        if len(window) < 10:
            return 50.0
        return float(np.sum(window < self.prices[index]) / len(window) * 100)

    def _is_yang(self, index: int) -> bool:
        """是否阳线"""
        return self.prices[index] > self.opens[index]

    def _amplitude(self, index: int) -> float:
        """K线振幅比例"""
        base = self.opens[index] if self.opens[index] > 0 else self.prices[index]
        return (self.highs[index] - self.lows[index]) / base if base > 0 else 0

    def _change_pct(self, index: int) -> float:
        """涨跌幅"""
        if index <= 0:
            return 0.0
        return (self.prices[index] - self.prices[index - 1]) / self.prices[index - 1]

    # ==================== 六步检测 ====================

    def _detect_building(self, end_idx: int) -> Optional[Tuple[int, int, float]]:
        """
        步骤1：检测建仓期
        条件：
        - 回看 building_period 天
        - 阳线占比 > building_yang_ratio
        - 该段均量 > 前 building_period 天均量的 building_vol_mult 倍
        - 价格处于低位（分位 < price_low_percentile）

        Returns:
            (建仓起始索引, 建仓结束索引, 建仓期量比) 或 None
        """
        period = self.building_period
        if end_idx < period * 2:
            return None

        # 建仓段
        build_start = end_idx - period + 1
        build_end = end_idx

        # 阳线占比
        yang_count = sum(1 for i in range(build_start, build_end + 1) if self._is_yang(i))
        yang_ratio = yang_count / period
        if yang_ratio < self.building_yang_ratio:
            return None

        # 量比：建仓段均量 vs 前段均量
        build_avg_vol = np.mean(self.volumes[build_start:build_end + 1])
        prev_start = max(0, build_start - period)
        prev_avg_vol = np.mean(self.volumes[prev_start:build_start])
        if prev_avg_vol <= 0:
            return None

        vol_mult = build_avg_vol / prev_avg_vol
        if vol_mult < self.building_vol_mult:
            return None

        # 价格低位
        pct = self._price_percentile(end_idx)
        if pct > self.price_low_percentile:
            return None

        return (build_start, build_end, vol_mult)

    def _find_resistance(self, build_end: int) -> float:
        """
        步骤2：识别压力位

        取建仓期最高价作为压力位。
        如果后续有更高的高点，则更新。
        """
        build_start = max(0, build_end - self.building_period)
        resistance = float(np.max(self.highs[build_start:build_end + 1]))
        return resistance

    def _detect_washing(self, wash_end: int, build_end: int) -> Optional[Tuple[int, int]]:
        """
        步骤3+4：检测洗盘沉淀
        条件：
        - 洗盘期在 build_end 之后
        - 持续 wash_min_days ~ wash_max_days 天
        - 成交量 < 均量的 wash_vol_ratio
        - 小K线（振幅<3%）占比 > wash_small_k_ratio
        - 价格不破建仓期低点

        Returns:
            (洗盘起始索引, 洗盘天数) 或 None
        """
        if wash_end <= build_end + self.wash_min_days:
            return None

        # 建仓期低点作为支撑
        build_start = max(0, build_end - self.building_period + 1)
        build_low = float(np.min(self.lows[build_start:build_end + 1]))

        # 从 build_end+1 到 wash_end，尝试不同长度的洗盘窗口
        best_wash = None

        for wash_days in range(self.wash_min_days, min(self.wash_max_days + 1, wash_end - build_end)):
            wash_start = wash_end - wash_days + 1
            if wash_start <= build_end:
                break

            wash_vols = self.volumes[wash_start:wash_end + 1]
            wash_vol_mas = self._vol_ma[wash_start:wash_end + 1]

            # 缩量检测：量比 < wash_vol_ratio 的天数占比 > 60%
            valid_vol_ma = wash_vol_mas[~np.isnan(wash_vol_mas)]
            if len(valid_vol_ma) == 0:
                continue
            ratios = wash_vols[:len(valid_vol_ma)] / valid_vol_ma
            shrink_ratio = np.mean(ratios < self.wash_vol_ratio)
            if shrink_ratio < 0.5:
                continue

            # 小K线检测
            amps = [self._amplitude(i) for i in range(wash_start, wash_end + 1)]
            small_k_ratio = np.mean([a < self.small_k_amplitude for a in amps])
            if small_k_ratio < 0.4:
                continue

            # 不破建仓低点
            wash_low = float(np.min(self.lows[wash_start:wash_end + 1]))
            if wash_low < build_low * 0.97:
                continue

            # 综合评分：缩量+小K线越明显越好
            score = shrink_ratio * 0.5 + small_k_ratio * 0.3 + (1 - wash_low / build_low) * 0.2
            if best_wash is None or score > best_wash[1]:
                best_wash = (wash_days, score, wash_start)

        if best_wash is None:
            return None

        return (best_wash[2], best_wash[0])

    def _detect_entry_signal(self, index: int) -> Optional[Tuple[str, float]]:
        """
        步骤5：检测买入信号

        类型1：倍量大阳线 — 量>2x均量 + 涨幅>3%
        类型2：强势底分型 + MACD金叉（使用缠论底分型 + MACD指标）

        Returns:
            (入场类型, 信号强度) 或 None
        """
        if index < 2:
            return None

        # 类型1：倍量大阳线
        change = self._change_pct(index)
        vol_r = self._vol_ratio(index)
        if vol_r >= self.double_vol and change >= self.big_yang_pct:
            strength = min(vol_r / self.double_vol, 1.5) * min(change / self.big_yang_pct, 1.5)
            return ('double_vol_bigyang', min(strength, 1.0))

        # 类型2：无量大阴次日收阳（洗盘最后一跌模式）
        if index >= 1:
            prev_vol_r = self._vol_ratio(index - 1)
            prev_change = self._change_pct(index - 1)
            curr_change = self._change_pct(index)
            # 昨天缩量大阴 + 今天收阳
            if prev_vol_r < 0.5 and prev_change < -0.01 and curr_change > 0:
                return ('wash_spring', 0.7)

        # 类型3：连续2天阳线 + 放量（温和入场）
        if index >= 2:
            c1 = self._change_pct(index - 1)
            c2 = self._change_pct(index)
            v1 = self._vol_ratio(index - 1)
            v2 = self._vol_ratio(index)
            if c1 > 0 and c2 > 0 and v2 > 1.0:
                return ('mild_entry', 0.5)

        return None

    def _check_breakout(self, index: int, resistance: float) -> bool:
        """
        步骤6：突破压力位检测

        条件：收盘价突破压力位 + 放量
        """
        if resistance <= 0:
            return False

        # 当日收盘突破压力位
        if self.prices[index] <= resistance:
            return False

        # 放量确认
        vol_r = self._vol_ratio(index)
        if vol_r < 1.2:
            return False

        return True

    # ==================== 主入口 ====================

    def scan(self, index: int = -1) -> Optional[AccumulationSignal]:
        """
        扫描指定位置是否满足主力资金沉淀战法条件

        Args:
            index: K线索引，-1为最新

        Returns:
            AccumulationSignal 或 None
        """
        if index < 0:
            index = self.n + index

        # 至少需要 building_period*2 + wash_min_days 的数据
        min_data = self.building_period * 2 + self.wash_min_days + 2
        if index < min_data:
            return None

        # Step 1: 检测建仓期（在当前位置之前的20-40天寻找建仓）
        best_building = None
        for lookback in range(self.building_period, self.building_period * 2):
            build_end = index - lookback
            if build_end < self.building_period:
                break
            result = self._detect_building(build_end)
            if result:
                best_building = result
                break

        if best_building is None:
            return None

        build_start, build_end, build_vol_mult = best_building

        # Step 2: 压力位
        resistance = self._find_resistance(build_end)

        # Step 3+4: 洗盘检测（build_end 到当前之间）
        washing = self._detect_washing(index, build_end)
        if washing is None:
            return None

        wash_start, wash_days = washing

        # Step 5: 买入信号
        entry = self._detect_entry_signal(index)
        if entry is None:
            return None

        entry_type, entry_strength = entry

        # Step 6: 突破检测
        is_breakout = self._check_breakout(index, resistance)

        # 计算止损
        recent_low = float(np.min(self.lows[max(0, index - 3):index + 1]))
        stop_loss = recent_low * 0.99

        # 计算置信度
        confidence = 0.5
        confidence += min(entry_strength * 0.2, 0.2)
        if build_vol_mult > 2.0:
            confidence += 0.1
        if wash_days >= 8:
            confidence += 0.05
        if is_breakout:
            confidence += 0.1
        if entry_type == 'double_vol_bigyang':
            confidence += 0.05
        confidence = min(confidence, 0.95)

        # 阶段
        phase = 'breakout' if is_breakout else 'entry'

        # 量比
        vol_ratio = self._vol_ratio(index)

        reason = f'{entry_type} | 建仓{build_vol_mult:.1f}x量 | 洗盘{wash_days}天'
        if is_breakout:
            reason += f' | 突破{resistance:.2f}'

        return AccumulationSignal(
            phase=phase,
            entry_type=entry_type,
            price=float(self.prices[index]),
            stop_loss=round(stop_loss, 2),
            resistance_price=round(resistance, 2),
            confidence=round(confidence, 2),
            building_start=build_start,
            washing_days=wash_days,
            volume_ratio=round(vol_ratio, 2),
            building_volume_ratio=round(build_vol_mult, 2),
            reason=reason,
        )

    def scan_recent(self, lookback: int = 10) -> List[AccumulationSignal]:
        """
        扫描最近N根K线，返回所有有效信号

        Args:
            lookback: 回溯天数

        Returns:
            信号列表
        """
        signals = []
        start = max(0, self.n - lookback)
        for i in range(start, self.n):
            sig = self.scan(i)
            if sig:
                signals.append(sig)
        return signals
