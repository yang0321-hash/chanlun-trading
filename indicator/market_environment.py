"""
大盘环境判定 — BULL/NEUTRAL/BEAR三级

基于上证指数:
  - MA250位置 + MA250斜率 → 长期趋势
  - 20日动量 → 短期动能
  - 三级环境: BULL/NEUTRAL/BEAR

按环境给不同买点类型设权重:
  BULL:    全部有效 (consolidationB×1.5, 3B×1.0, quasi2B×1.0)
  NEUTRAL: 仅1B/2B有效 (3B×0.5, quasi2B×0.3, weak3B禁用)
  BEAR:    仅1B有效 (其余全禁用)

用法:
  from indicator.market_environment import MarketEnvironment
  env = MarketEnvironment()         # 自动检测TDX路径
  state = env.get_state()           # 'BULL'/'NEUTRAL'/'BEAR'
  weights = env.get_signal_weights()  # {'1buy': 1.0, '2buy': 0.5, ...}
  print(env.get_summary())
"""

import os
import struct
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketState:
    state: str          # 'BULL' / 'NEUTRAL' / 'BEAR'
    ma250: float        # MA250值
    ma250_slope: float  # MA250斜率(%)
    momentum_20d: float # 20日动量(%)
    dist_ma250: float   # 距MA250(%)
    description: str    # 中文描述

    def __str__(self):
        tags = {'BULL': 'BULL', 'NEUTRAL': 'NEUTRAL', 'BEAR': 'BEAR'}
        return (f"{tags.get(self.state, '?')} "
                f"MA250={self.ma250:.0f} 斜率={self.ma250_slope:+.2f}% "
                f"动量20d={self.momentum_20d:+.1f}% "
                f"距MA250={self.dist_ma250:+.1f}%")


def _read_idx_records(filepath: str, n: int = 600):
    """读取上证指数日线数据"""
    if not os.path.exists(filepath):
        return []
    sz = os.path.getsize(filepath)
    if sz < 32:
        return []
    cnt = sz // 32
    start = max(0, cnt - n)
    out = []
    with open(filepath, 'rb') as f:
        f.seek(start * 32)
        data = f.read()
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


class MarketEnvironment:
    """大盘环境判定"""

    TDX_CANDIDATES = [
        r'D:\大侠神器2.0\直接使用_大侠神器2.0.1.251231(ODM250901)\直接使用_大侠神器2.0.10B1206(260930)\new_tdx(V770)\vipdoc',
        r'D:\new_tdx\vipdoc',
        r'D:\新建文件夹\claude\tdx_data',
        r'D:\tdx_data',
    ]

    # 买点类型权重映射
    SIGNAL_WEIGHTS = {
        'BULL': {
            '1buy': 1.0, '2buy': 1.2, '3buy': 1.0,
            'quasi2buy': 1.0, 'quasi3buy': 0.5,
            '2b3bbuy': 1.2, 'sub1buy': 1.5,
        },
        'NEUTRAL': {
            '1buy': 1.0, '2buy': 1.0, '3buy': 0.5,
            'quasi2buy': 0.3, 'quasi3buy': 0.0,
            '2b3bbuy': 0.8, 'sub1buy': 0.8,
        },
        'BEAR': {
            '1buy': 1.0, '2buy': 0.5, '3buy': 0.0,
            'quasi2buy': 0.0, 'quasi3buy': 0.0,
            '2b3bbuy': 0.0, 'sub1buy': 0.0,
        },
    }

    def __init__(self, tdx_vipdoc: Optional[str] = None):
        self._records = []
        self._state: Optional[MarketState] = None

        if tdx_vipdoc is None:
            for p in self.TDX_CANDIDATES:
                if os.path.exists(p):
                    tdx_vipdoc = p
                    break

        if tdx_vipdoc:
            # 尝试多种指数文件名
            for idx_file in ['sh000001.day', 'sh1a0001.day', 'sh999999.day']:
                fp = os.path.join(tdx_vipdoc, 'sh', 'lday', idx_file)
                if os.path.exists(fp):
                    self._records = _read_idx_records(fp)
                    break

    @property
    def records(self):
        return self._records

    def _calc_ma250(self) -> Tuple[Optional[float], Optional[float]]:
        """计算MA250及其20日斜率"""
        closes = [r['close'] for r in self._records]
        n = len(closes)
        if n < 270:
            return None, None
        ma250_now = sum(closes[-250:]) / 250
        ma250_20ago = sum(closes[-270:-20]) / 250
        slope = (ma250_now - ma250_20ago) / ma250_20ago * 100
        return ma250_now, slope

    def _calc_momentum(self, period: int = 20) -> Optional[float]:
        """计算N日动量"""
        closes = [r['close'] for r in self._records]
        if len(closes) < period + 1:
            return None
        return (closes[-1] - closes[-period - 1]) / closes[-period - 1] * 100

    def get_state(self) -> str:
        """返回当前市场状态: BULL/NEUTRAL/BEAR"""
        ms = self.get_market_state()
        return ms.state

    def get_market_state(self) -> MarketState:
        """返回完整的市场状态"""
        if self._state:
            return self._state

        if not self._records or len(self._records) < 270:
            self._state = MarketState(
                state='NEUTRAL', ma250=0, ma250_slope=0,
                momentum_20d=0, dist_ma250=0,
                description='数据不足，默认中性'
            )
            return self._state

        close = self._records[-1]['close']
        ma250, slope = self._calc_ma250()
        mom = self._calc_momentum(20)

        if ma250 is None:
            state = 'NEUTRAL'
        elif close >= ma250 and slope is not None and slope > 0:
            state = 'BULL'
        elif (slope is not None and slope > 0) or (mom is not None and mom > 0):
            state = 'NEUTRAL'
        else:
            state = 'BEAR'

        dist = ((close / ma250) - 1) * 100 if ma250 and ma250 > 0 else 0
        desc = {
            'BULL': '站上MA250 + MA250上升 → 全面做多',
            'NEUTRAL': 'MA250上升或动能偏正 → 谨慎做多',
            'BEAR': 'MA250下降 + 动能负 → 仅1买抄底',
        }

        self._state = MarketState(
            state=state,
            ma250=ma250 or 0,
            ma250_slope=slope or 0,
            momentum_20d=mom or 0,
            dist_ma250=round(dist, 1),
            description=desc.get(state, ''),
        )
        return self._state

    def get_signal_weights(self) -> Dict[str, float]:
        """返回当前环境下各买点类型的权重"""
        state = self.get_state()
        return dict(self.SIGNAL_WEIGHTS.get(state, self.SIGNAL_WEIGHTS['NEUTRAL']))

    def get_active_types(self) -> Dict[str, float]:
        """返回权重>0的买点类型 (即当前环境允许的买点)"""
        weights = self.get_signal_weights()
        return {k: v for k, v in weights.items() if v > 0}

    def is_signal_allowed(self, signal_type: str) -> bool:
        """判断某买点类型在当前环境下是否允许"""
        weights = self.get_signal_weights()
        return weights.get(signal_type, 0) > 0

    def get_signal_weight(self, signal_type: str) -> float:
        """获取某买点类型的权重"""
        weights = self.get_signal_weights()
        return weights.get(signal_type, 0)

    def get_summary(self) -> str:
        """返回可读摘要"""
        ms = self.get_market_state()
        active = self.get_active_types()
        tags = {'BULL': 'BULL', 'NEUTRAL': 'NEUTRAL', 'BEAR': 'BEAR'}
        lines = [
            f"大盘环境 | {tags.get(ms.state, '?')}",
            f"  MA250={ms.ma250:.0f} 斜率={ms.ma250_slope:+.2f}% "
            f"动量20d={ms.momentum_20d:+.1f}% 距MA250={ms.dist_ma250:+.1f}%",
            f"  {ms.description}",
            f"  有效信号: {', '.join(f'{k}×{v:.1f}' for k, v in active.items())}",
        ]
        return '\n'.join(lines)
