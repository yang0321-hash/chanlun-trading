"""
涨停板游资打板策略 v2.0

策略逻辑：
1. 选股阶段 - 首板 + 连板(2-4板)
2. 做T持有 - 底仓持有，日内做T降低成本
3. 固定30%分仓 - 每只股票30%仓位
4. 破板即走 - 涨停打开立即止损

支持板型：
- 首板：低位首个涨停，挖掘新题材
- 2板：确认阶段，题材发酵
- 3板：加速阶段，游资主攻
- 4板：分化阶段，谨慎参与
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from loguru import logger
from dataclasses import dataclass

from backtest.strategy import Strategy, Signal, SignalType


class BoardType(Enum):
    """涨停板类型"""
    FIRST_BOARD = 'first_board'     # 首板
    SECOND_BOARD = 'second_board'   # 二板
    THIRD_BOARD = 'third_board'     # 三板
    FOURTH_BOARD = 'fourth_board'   # 四板
    REVERSAL_BOARD = 'reversal_board'  # 反包板
    UNKNOWN = 'unknown'


@dataclass
class DragonLeader:
    """龙一信息"""
    code: str
    name: str
    consecutive_limit_up: int  # 连续涨停板数
    price: float
    sector: str
    board_type: BoardType = BoardType.UNKNOWN
    amount: float = 0  # 成交额（亿）
    turnover: float = 0  # 换手率
    seal_amount: float = 0  # 封单额（万）
    first_limit_time: str = ''  # 首次涨停时间


@dataclass
class PositionInfo:
    """持仓信息"""
    base_qty: int = 0         # 底仓数量
    base_cost: float = 0      # 底仓成本
    t_qty: int = 0            # 做T仓位
    t_cost: float = 0         # 做T成本
    entry_date: datetime = None
    last_board_date: datetime = None  # 最后涨停日期
    total_profit: float = 0   # 累计做T收益


class HotMoneyLimitUpStrategy(Strategy):
    """
    涨停板游资打板策略 v2.0 - 做T持有版

    核心逻辑：
    - 首板 + 连板(2-4板)选股
    - 固定30%分仓
    - 做T持有：底仓不动，日内做T
    - 破板即走止损
    """

    def __init__(
        self,
        name: str = '游资打板策略v2',
        position_size: float = 0.3,  # 固定30%分仓
        max_positions: int = 3,      # 最多同时持有3只
        target_boards: List[int] = None,  # 目标板数 [1,2,3,4]
        t_profit_threshold: float = 2.0,  # 做T止盈点数(%)
        t_loss_threshold: float = -1.0,   # 做T止损点数(%)
        base_hold_days: int = 5,     # 底仓默认持有天数
    ):
        """
        初始化策略

        Args:
            name: 策略名称
            position_size: 单次仓位比例 (默认0.3=30%)
            max_positions: 最大持仓数量
            target_boards: 目标板数列表 (默认[1,2,3,4])
            t_profit_threshold: 做T止盈阈值(%)
            t_loss_threshold: 做T止损阈值(%)
            base_hold_days: 底仓默认持有天数
        """
        super().__init__(name)
        self.position_size = position_size
        self.max_positions = max_positions
        self.target_boards = target_boards or [1, 2, 3, 4]
        self.t_profit_threshold = t_profit_threshold
        self.t_loss_threshold = t_loss_threshold
        self.base_hold_days = base_hold_days

        # 持仓信息 {code: PositionInfo}
        self._positions: Dict[str, PositionInfo] = {}

        # 选中的股票
        self._selected_stocks: Dict[str, DragonLeader] = {}

        # 昨日涨停价格 {code: limit_price}
        self._yesterday_limit_price: Dict[str, float] = {}

        logger.info(f"初始化{self.name}: 仓位{position_size*100}%, "
                   f"目标板数{target_boards}, 最多{max_positions}只")

    def reset(self) -> None:
        """重置策略状态"""
        super().reset()
        self._positions.clear()
        self._selected_stocks.clear()
        self._yesterday_limit_price.clear()

    def get_position_info(self, symbol: str) -> PositionInfo:
        """获取持仓信息"""
        if symbol not in self._positions:
            self._positions[symbol] = PositionInfo()
        return self._positions[symbol]

    def is_limit_up(self, bar: pd.Series, code: str) -> bool:
        """判断是否涨停"""
        if 'open' not in bar or 'close' not in bar:
            return False

        open_p = bar['open']
        close_p = bar['close']

        if open_p <= 0:
            return False

        change_pct = (close_p - open_p) / open_p * 100

        # 科创板/创业板20%，普通股10%，ST股5%
        if code.startswith('sh688') or code.startswith('sz3'):
            threshold = 19.5
        elif 'ST' in code:
            threshold = 4.5
        else:
            threshold = 9.5

        return change_pct >= threshold

    def calc_limit_price(self, base_price: float, code: str) -> float:
        """计算涨停价"""
        if code.startswith('sh688') or code.startswith('sz3'):
            return base_price * 1.2
        elif 'ST' in code:
            return base_price * 1.05
        else:
            return base_price * 1.1

    def detect_board_type(
        self,
        consecutive: int,
        previous_drop: bool = False
    ) -> BoardType:
        """检测板型"""
        if consecutive == 1:
            return BoardType.FIRST_BOARD
        elif consecutive == 2:
            return BoardType.SECOND_BOARD
        elif consecutive == 3:
            return BoardType.THIRD_BOARD
        elif consecutive == 4:
            return BoardType.FOURTH_BOARD
        elif previous_drop and consecutive == 1:
            return BoardType.REVERSAL_BOARD
        else:
            return BoardType.UNKNOWN

    def set_selected_stocks(self, dragons: List[DragonLeader]) -> None:
        """设置选中的股票列表"""
        self._selected_stocks = {d.code: d for d in dragons}
        logger.info(f"设置选中股票: {[d.name for d in dragons]}")

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
        index: int,
        context: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        处理K线

        交易逻辑：
        1. 有持仓时：检查是否破板（止损）或可做T
        2. 无持仓时：检查是否打板买入
        """
        current_price = bar['close']
        pos_info = self.get_position_info(symbol)

        # 更新今日涨停状态
        is_limit = self.is_limit_up(bar, symbol)

        # ========== 有持仓逻辑 ==========
        if pos_info.base_qty > 0:
            return self._handle_position_with_stock(bar, symbol, current_price, pos_info, is_limit)

        # ========== 空仓逻辑 ==========
        # 检查是否在选中列表
        selected = context.get('selected_stocks', {})
        if symbol in selected:
            dragon = selected[symbol]
            return self._check_entry_signal(bar, symbol, current_price, dragon, is_limit)

        return None

    def _handle_position_with_stock(
        self,
        bar: pd.Series,
        symbol: str,
        current_price: float,
        pos_info: PositionInfo,
        is_limit: bool
    ) -> Optional[Signal]:
        """
        处理有持仓的情况

        规则：
        1. 破板即走 - 止损
        2. 做T机会 - 冲高回落做T
        3. 底仓持有天数到期 - 卖出底仓
        """
        holding_days = 0
        if pos_info.entry_date:
            holding_days = (bar['datetime'] - pos_info.entry_date).days

        # 1. 检查破板止损
        if self._is_board_broken(bar, symbol, current_price, pos_info):
            # 全部卖出
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar['datetime'],
                price=current_price,
                quantity=pos_info.base_qty + pos_info.t_qty,
                reason=f'破板止损(持有{holding_days}天,做T收益{pos_info.total_profit:.2f}%)',
                confidence=1.0,
                metadata={'exit_reason': 'board_broken', 'total_profit': pos_info.total_profit}
            )

        # 2. 检查做T机会（冲高回落）
        if pos_info.base_qty > 0 and not is_limit:
            t_signal = self._check_t_trading_signal(bar, symbol, current_price, pos_info)
            if t_signal:
                return t_signal

        # 3. 检查底仓持有天数
        if holding_days >= self.base_hold_days and not is_limit:
            # 卖出底仓
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                datetime=bar['datetime'],
                price=current_price,
                quantity=pos_info.base_qty,
                reason=f'持有到期({holding_days}天), 做T收益{pos_info.total_profit:.2f}%',
                confidence=0.8,
                metadata={'exit_reason': 'hold_days_exceeded', 'total_profit': pos_info.total_profit}
            )

        # 4. 继续涨停，持有
        if is_limit:
            pos_info.last_board_date = bar['datetime']
            logger.info(f"  {symbol}继续涨停,持有")

        return None

    def _is_board_broken(
        self,
        bar: pd.Series,
        symbol: str,
        current_price: float,
        pos_info: PositionInfo
    ) -> bool:
        """
        判断是否破板

        破板定义：
        1. 昨日涨停，今日收盘价 < 昨日收盘价 * 0.97
        2. 盘中最高价触及涨停但无法封住（高开低走）
        3. 换手率异常放大且下跌
        """
        if pos_info.last_board_date is None:
            return False

        # 昨日涨停价
        yesterday_limit = self._yesterday_limit_price.get(symbol, 0)
        if yesterday_limit == 0:
            return False

        # 今日跌幅超过3%视为破板
        change_pct = (current_price - bar['open']) / bar['open'] * 100

        # 高开低走，收盘价低于开盘价3%
        if change_pct < -3:
            return True

        # 无法维持涨停价
        if current_price < yesterday_limit * 0.97:
            return True

        return False

    def _check_t_trading_signal(
        self,
        bar: pd.Series,
        symbol: str,
        current_price: float,
        pos_info: PositionInfo
    ) -> Optional[Signal]:
        """
        检查做T信号

        做T策略：
        1. 冲高2%以上回落 - 卖出做T仓位
        2. 回调-1%以上企稳 - 买回做T仓位
        """
        intraday_high = bar.get('high', bar['close'])
        intraday_low = bar.get('low', bar['close'])

        # 计算日内振幅
        from_open_high = (intraday_high - bar['open']) / bar['open'] * 100
        current_change = (current_price - bar['open']) / bar['open'] * 100

        # 有做T仓位，检查是否买入平仓
        if pos_info.t_qty < 0:  # 做T卖出状态（负数表示卖出）
            # 回调到-1%以内，买回
            if current_change <= self.t_loss_threshold:
                buy_qty = abs(pos_info.t_qty)
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    datetime=bar['datetime'],
                    price=current_price,
                    quantity=buy_qty,
                    reason=f'做T买回: 回调{current_change:.2f}%',
                    confidence=0.9,
                    metadata={'trade_type': 't_close'}
                )

        # 无做T仓位，检查是否卖出做T
        elif pos_info.t_qty == 0 and from_open_high >= self.t_profit_threshold:
            # 冲高到2%以上，卖出部分做T
            t_qty = int(pos_info.base_qty * 0.3)  # 卖出底仓的30%
            if t_qty > 0:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    datetime=bar['datetime'],
                    price=current_price,
                    quantity=t_qty,
                    reason=f'做T卖出: 冲高{from_open_high:.2f}%',
                    confidence=0.8,
                    metadata={'trade_type': 't_open'}
                )

        return None

    def _check_entry_signal(
        self,
        bar: pd.Series,
        symbol: str,
        current_price: float,
        dragon: DragonLeader,
        is_limit: bool
    ) -> Optional[Signal]:
        """
        检查买入信号

        规则：
        1. 目标板数 (首板/2板/3板/4板)
        2. 打板买入 (接近涨停)
        3. 固定30%仓位
        """
        # 检查板数是否在目标范围
        if dragon.consecutive_limit_up not in self.target_boards:
            return None

        # 检查是否涨停
        if not is_limit:
            return None

        # 检查当前持仓数量
        current_positions = sum(1 for p in self._positions.values() if p.base_qty > 0)
        if current_positions >= self.max_positions:
            logger.info(f"  {symbol}: 已满仓({current_positions}/{self.max_positions})")
            return None

        # 计算买入数量
        buy_amount = self.initial_capital * self.position_size
        qty = int(buy_amount / current_price / 100) * 100  # 整手

        if qty < 100:
            return None

        # 记录买入信息
        pos_info = self.get_position_info(symbol)
        pos_info.entry_date = bar['datetime']
        pos_info.last_board_date = bar['datetime']

        self._yesterday_limit_price[symbol] = current_price

        board_type_cn = {
            BoardType.FIRST_BOARD: '首板',
            BoardType.SECOND_BOARD: '二板',
            BoardType.THIRD_BOARD: '三板',
            BoardType.FOURTH_BOARD: '四板',
        }.get(dragon.board_type, f'{dragon.consecutive_limit_up}板')

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            datetime=bar['datetime'],
            price=current_price,
            quantity=qty,
            reason=f'{board_type_cn}打板: {dragon.sector}龙一',
            confidence=0.8,
            metadata={
                'sector': dragon.sector,
                'consecutive_boards': dragon.consecutive_limit_up,
                'board_type': dragon.board_type.value,
                'is_dragon_leader': True
            }
        )

    def on_order(
        self,
        signal: Signal,
        executed_price: float,
        executed_quantity: int
    ) -> None:
        """订单成交回调"""
        symbol = signal.symbol
        pos_info = self.get_position_info(symbol)
        trade_type = signal.metadata.get('trade_type', 'normal')

        if signal.is_buy():
            if trade_type == 't_close':
                # 做T买回
                pos_info.t_qty += executed_quantity
                # 计算做T收益
                if pos_info.t_cost > 0:
                    t_profit = (pos_info.t_cost - executed_price) / pos_info.t_cost * 100
                    pos_info.total_profit += t_profit
                logger.info(f"  {symbol} 做T买回 {executed_quantity}股@{executed_price:.2f}")
            else:
                # 正常买入（底仓）
                self.position[symbol] = self.position.get(symbol, 0) + executed_quantity
                pos_info.base_qty = self.position[symbol]
                pos_info.base_cost = executed_price
                self.cash -= executed_price * executed_quantity
                logger.info(f"  {symbol} 买入底仓 {executed_quantity}股@{executed_price:.2f}")

        elif signal.is_sell():
            if trade_type == 't_open':
                # 做T卖出
                pos_info.t_qty -= executed_quantity
                pos_info.t_cost = executed_price
                logger.info(f"  {symbol} 做T卖出 {executed_quantity}股@{executed_price:.2f}")
            else:
                # 正常卖出
                self.position[symbol] = self.position.get(symbol, 0) - executed_quantity
                pos_info.base_qty = self.position[symbol]
                self.cash += executed_price * executed_quantity

                # 如果全部卖出，清理持仓信息
                if pos_info.base_qty <= 0:
                    logger.info(f"  {symbol} 清仓, 累计做T收益: {pos_info.total_profit:.2f}%")
                    self._positions[symbol] = PositionInfo()
                    self._yesterday_limit_price.pop(symbol, None)

    def get_positions_summary(self) -> List[Dict]:
        """获取持仓摘要"""
        summary = []
        for symbol, pos_info in self._positions.items():
            if pos_info.base_qty > 0:
                summary.append({
                    'symbol': symbol,
                    'base_qty': pos_info.base_qty,
                    'base_cost': pos_info.base_cost,
                    't_qty': pos_info.t_qty,
                    'entry_date': pos_info.entry_date,
                    'last_board_date': pos_info.last_board_date,
                    'total_profit': pos_info.total_profit
                })
        return summary


class HotMoneyLimitUpSelector:
    """
    游资打板选股器 - 独立选股逻辑

    配合TDXSectorAnalyzer使用，扫描涨停板股票并选出龙一
    """

    def __init__(
        self,
        target_boards: List[int] = None,
        min_amount: float = 0.5,  # 最小成交额（亿）
        require_clear_leader: bool = True
    ):
        """
        初始化选股器

        Args:
            target_boards: 目标板数列表
            min_amount: 最小成交额（亿）
            require_clear_leader: 是否要求明确龙一
        """
        self.target_boards = target_boards or [1, 2, 3, 4]
        self.min_amount = min_amount
        self.require_clear_leader = require_clear_leader

    def select_from_analysis(
        self,
        analysis_result: Dict
    ) -> List[DragonLeader]:
        """
        从分析结果中选择龙一

        Args:
            analysis_result: TDXSectorAnalyzer.run_daily_analysis()的返回结果

        Returns:
            龙一列表
        """
        dragons = []

        limit_up_stocks = analysis_result.get('limit_up_stocks', [])

        for stock in limit_up_stocks:
            consecutive = stock.get('consecutive_boards', 0)
            amount = stock.get('amount', 0)

            # 过滤条件
            if consecutive not in self.target_boards:
                continue
            if amount < self.min_amount:
                continue

            # 创建DragonLeader
            board_type = self._get_board_type(consecutive)

            dragon = DragonLeader(
                code=stock['code'],
                name=stock.get('name', stock['code']),
                consecutive_limit_up=consecutive,
                price=stock['price'],
                sector=stock.get('sector', '未知'),
                board_type=board_type,
                amount=amount,
                turnover=stock.get('turnover', 0)
            )

            dragons.append(dragon)

        # 按成交额和板数排序
        dragons.sort(key=lambda x: (x.consecutive_limit_up, x.amount), reverse=True)

        logger.info(f"选股器选出{len(dragons)}只龙一: "
                   f"{[f'{d.name}({d.consecutive_limit_up}板)' for d in dragons]}")

        return dragons

    def _get_board_type(self, consecutive: int) -> BoardType:
        """获取板型"""
        mapping = {
            1: BoardType.FIRST_BOARD,
            2: BoardType.SECOND_BOARD,
            3: BoardType.THIRD_BOARD,
            4: BoardType.FOURTH_BOARD,
        }
        return mapping.get(consecutive, BoardType.UNKNOWN)
