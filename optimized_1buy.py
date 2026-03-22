"""
缠论1买信号 - 优化版

优化策略：
1. 成交量确认：反弹时放量
2. 趋势过滤：避免强烈下跌趋势
3. 中枢质量：中枢震荡幅度合理
4. 回调幅度：下跌不过度
5. 多重确认：MACD背驰 + K线形态
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json
import numpy as np

# 确保输出编码正确
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core.kline import KLine
from core.fractal import FractalDetector, Fractal
from core.stroke import StrokeGenerator
from core.pivot import PivotDetector, Pivot
from indicator.macd import MACD


class EnhancedBuyPointDetector:
    """增强版买点检测器"""

    def __init__(self):
        self.min_klines = 60
        self.use_macd = True

    def detect_1buy_enhanced(
        self,
        df: pd.DataFrame,
        kline: KLine,
        symbol: str
    ) -> Optional[Dict]:
        """
        检测增强版1买信号

        Returns:
            Dict or None: 信号信息
        """
        if len(df) < self.min_klines:
            return None

        # 基础缠论要素
        fractals = FractalDetector(kline, confirm_required=False).get_fractals()
        if len(fractals) < 3:
            return None

        strokes = StrokeGenerator(kline, fractals).get_strokes()
        if len(strokes) < 3:
            return None

        pivots = PivotDetector(kline, strokes).get_pivots()
        if not pivots:
            return None

        current_price = df['close'].iloc[-1]
        last_pivot = pivots[-1]
        last_stroke = strokes[-1]

        # 基础条件：价格在中枢下方
        if current_price >= last_pivot.low:
            return None

        # 基础条件：正在反弹
        if not last_stroke.is_up:
            return None

        # 开始增强检测
        score = 0
        reasons = []

        # 1. 成交量确认 (0-20分)
        vol_score, vol_reason = self._check_volume(df, strokes)
        score += vol_score
        if vol_reason:
            reasons.append(vol_reason)

        # 2. 趋势过滤 (0-20分)
        trend_score, trend_reason = self._check_trend(df, strokes)
        score += trend_score
        if trend_reason:
            reasons.append(trend_reason)

        # 3. 中枢质量 (0-15分)
        pivot_score, pivot_reason = self._check_pivot_quality(last_pivot, current_price)
        score += pivot_score
        if pivot_reason:
            reasons.append(pivot_reason)

        # 4. 回调幅度 (0-15分)
        drop_score, drop_reason = self._check_drop_amount(last_pivot, current_price)
        score += drop_score
        if drop_reason:
            reasons.append(drop_reason)

        # 5. MACD背驰 (0-20分)
        if self.use_macd:
            macd_score, macd_reason = self._check_macd_divergence(df)
            score += macd_score
            if macd_reason:
                reasons.append(macd_reason)

        # 6. K线形态 (0-10分)
        pattern_score, pattern_reason = self._check_candle_pattern(df, fractals)
        score += pattern_score
        if pattern_reason:
            reasons.append(pattern_reason)

        # 总分阈值：至少50分才发出信号
        if score < 50:
            return None

        # 计算信号强度
        confidence = min(score / 100, 0.95)

        # 计算止损位
        stop_loss = last_pivot.low * 0.97  # 中枢下沿下方3%

        # 计算目标位
        target1 = last_pivot.high  # 第一目标：中枢上沿
        target2 = last_pivot.high + (last_pivot.high - last_pivot.low)  # 第二目标

        return {
            'symbol': symbol,
            'type': '1buy_enhanced',
            'price': current_price,
            'score': score,
            'confidence': confidence,
            'reasons': reasons,
            'pivot_high': last_pivot.high,
            'pivot_low': last_pivot.low,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'reward_ratio': (target1 - current_price) / (current_price - stop_loss) if current_price > stop_loss else 0
        }

    def _check_volume(self, df: pd.DataFrame, strokes: List) -> Tuple[int, str]:
        """检查成交量确认"""
        score = 0
        reason = None

        if len(df) < 10:
            return 0, None

        # 最近5天平均成交量
        recent_vol = df['volume'].iloc[-5:].mean()
        # 前20天平均成交量
        base_vol = df['volume'].iloc[-25:-5].mean()

        if base_vol > 0:
            vol_ratio = recent_vol / base_vol

            if vol_ratio >= 1.5:
                score = 20
                reason = f"放量反弹(vol={vol_ratio:.1f}x)"
            elif vol_ratio >= 1.2:
                score = 15
                reason = f"温和放量(vol={vol_ratio:.1f}x)"
            elif vol_ratio >= 0.8:
                score = 10
                reason = f"平量反弹(vol={vol_ratio:.1f}x)"
            else:
                score = 5
                reason = f"缩量反弹(vol={vol_ratio:.1f}x)"

        return score, reason

    def _check_trend(self, df: pd.DataFrame, strokes: List) -> Tuple[int, str]:
        """检查趋势，避免强烈下跌趋势"""
        score = 0
        reason = None

        if len(df) < 60:
            return 10, "数据不足，默认中性"

        # 计算长期趋势（60日均线）
        ma60 = df['close'].iloc[-60:].mean()
        current_price = df['close'].iloc[-1]

        # 计算短期趋势（20日均线）
        ma20 = df['close'].iloc[-20:].mean()

        # 价格相对MA60的位置
        ma60_ratio = current_price / ma60

        if ma60_ratio >= 0.95:
            # 价格在MA60附近或上方，趋势较好
            score = 20
            reason = f"趋势良好(MA60={ma60_ratio:.1%})"
        elif ma60_ratio >= 0.85:
            # 价格低于MA60但幅度不大
            score = 15
            reason = f"趋势偏弱(MA60={ma60_ratio:.1%})"
        elif ma60_ratio >= 0.75:
            # 价格明显低于MA60
            score = 8
            reason = f"下降趋势(MA60={ma60_ratio:.1%})"
        else:
            # 价格严重低于MA60，强烈下跌
            score = 3
            reason = f"强烈下跌(MA60={ma60_ratio:.1%})"

        # 检查最近是否创新低
        recent_low = df['low'].iloc[-20:].min()
        if current_price < recent_low * 1.02:
            score = max(score - 5, 0)
            if reason:
                reason += ",近期创新低"

        return score, reason

    def _check_pivot_quality(self, pivot: Pivot, current_price: float) -> Tuple[int, str]:
        """检查中枢质量"""
        score = 0
        reason = None

        # 中枢宽度
        pivot_range = pivot.range_value / pivot.low

        # 中枢笔数
        stroke_count = pivot.strokes_count

        # 评分
        if 0.05 <= pivot_range <= 0.20:
            # 中枢宽度适中（5%-20%）
            range_score = 10
        elif pivot_range < 0.05:
            range_score = 5  # 太窄
        else:
            range_score = 8  # 稍宽

        if stroke_count >= 5:
            stroke_score = 5
        elif stroke_count >= 3:
            stroke_score = 3
        else:
            stroke_score = 1

        score = range_score + stroke_score

        if stroke_count >= 5 and 0.05 <= pivot_range <= 0.20:
            reason = f"优质中枢(宽{pivot_range:.1%},{stroke_count}笔)"
        else:
            reason = f"中枢(宽{pivot_range:.1%},{stroke_count}笔)"

        return score, reason

    def _check_drop_amount(self, pivot: Pivot, current_price: float) -> Tuple[int, str]:
        """检查回调幅度"""
        score = 0
        reason = None

        # 计算距离中枢下沿的跌幅
        drop_ratio = (pivot.low - current_price) / pivot.low

        if drop_ratio <= 0.05:
            # 轻微跌破
            score = 15
            reason = f"浅跌破({drop_ratio:.1%})"
        elif drop_ratio <= 0.15:
            # 中度跌破
            score = 12
            reason = f"中度跌破({drop_ratio:.1%})"
        elif drop_ratio <= 0.30:
            # 深度跌破
            score = 6
            reason = f"深度跌破({drop_ratio:.1%})"
        else:
            # 严重跌破
            score = 2
            reason = f"严重超跌({drop_ratio:.1%})"

        return score, reason

    def _check_macd_divergence(self, df: pd.DataFrame) -> Tuple[int, str]:
        """检查MACD背驰"""
        score = 0
        reason = None

        try:
            macd = MACD(df['close'])
            if len(macd) > 20:
                has_div, _ = macd.check_divergence(len(macd) - 20, len(macd) - 1, 'down')

                if has_div:
                    score = 20
                    reason = "MACD底背驰"
                else:
                    score = 5
                    reason = "无MACD背驰"
        except:
            score = 0

        return score, reason

    def _check_candle_pattern(self, df: pd.DataFrame, fractals: List) -> Tuple[int, str]:
        """检查K线形态"""
        score = 0
        reason = None

        if len(fractals) < 2:
            return 5, None

        # 检查最后两个分型
        last_fractal = fractals[-1]
        prev_fractal = fractals[-2]

        # 底分型 + 底分型 = 双底
        if not last_fractal.is_bottom and not prev_fractal.is_bottom:
            score = 10
            reason = "双底形态"
        # 最后是底分型
        elif not last_fractal.is_bottom:
            score = 7
            reason = "底分型确认"
        else:
            score = 3

        return score, reason


def scan_enhanced_1buy(limit=100):
    """扫描增强版1买信号"""
    data_dir = "test_output"

    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在 - {data_dir}")
        return

    import glob
    all_files = glob.glob(os.path.join(data_dir, "*.json"))

    # 只扫描个股
    def is_stock(symbol):
        if symbol.startswith('sh000') or symbol.startswith('sz399'):
            return False
        if symbol.startswith('sh6'):
            return True
        if symbol.startswith('sz') and symbol[2:5] in ('000', '001', '002', '300'):
            return True
        return False

    files = [f for f in all_files if is_stock(os.path.basename(f).replace('.day.json', '').replace('.json', ''))]

    if limit:
        files = files[:limit]

    print("=" * 60)
    print(f"增强版1买信号扫描")
    print(f"扫描 {len(files)} 只个股")
    print("=" * 60)

    detector = EnhancedBuyPointDetector()
    signals = []

    for i, filepath in enumerate(files):
        try:
            symbol = os.path.basename(filepath).replace('.day.json', '').replace('.json', '')

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])

            if len(df) < 60:
                continue

            kline = KLine.from_dataframe(df, strict_mode=True)

            signal = detector.detect_1buy_enhanced(df, kline, symbol)

            if signal:
                signals.append(signal)

        except Exception as e:
            pass

    # 打印结果
    print(f"\n发现 {len(signals)} 个高质量1买信号\n")

    if signals:
        # 按分数排序
        signals.sort(key=lambda x: x['score'], reverse=True)

        print(f"{'代码':<10} {'价格':>8} {'分数':>4} {'胜率':>6} {'盈亏比':>6} {'止损':>8} {'目标1':>8} {'目标2':>8}")
        print("-" * 80)

        for sig in signals[:30]:  # 只显示前30个
            print(f"{sig['symbol']:<10} "
                  f"¥{sig['price']:>6.2f} "
                  f"{sig['score']:>3} "
                  f"{sig['confidence']:>5.0%} "
                  f"{sig['reward_ratio']:>5.1f} "
                  f"¥{sig['stop_loss']:>6.2f} "
                  f"¥{sig['target1']:>6.2f} "
                  f"¥{sig['target2']:>6.2f}")

            # 显示原因
            reasons = " | ".join(sig['reasons'])
            print(f"  └─ {reasons}")

        print("\n" + "=" * 60)
        print("说明：")
        print("- 分数: 综合评分(0-100)，越高越好")
        print("- 胜率: 预期成功概率")
        print("- 盈亏比: 目标收益/止损风险的比值")
        print("- 止损: 建议止损位")
        print("- 目标1/2: 第一/第二目标价位")

    return signals


if __name__ == '__main__':
    signals = scan_enhanced_1buy(limit=200)
