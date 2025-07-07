import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy


class SMABollingStrategy(Strategy):
    """
    策略概述：
    1. 定义超买和超卖区间，超卖30，超买70
    2. 利用rsi计算rsi based SMA，RSI长度为14
    3. 计算bolling上下轨

    出入场时机：
    1. 当价格突破bolling下轨，且rsi based ma上穿rsi或者在上方，做多
    2. 当价格突破bolling上轨，且rsi based ma下穿rsi或者在rsi下方，平多
    3. 横盘不交易
    """

    def __init__(self):
        super().__init__()
        # 策略参数
        self.rsi_period = 12
        self.rsi_sma_period = 14
        self.bb_period = 24
        self.bb_std = 2.0
        self.rsi_oversold = 28
        self.rsi_overbought = 68
        self.adx_period = 12
        self.adx_threshold = 22
        self.bb_width_threshold = 0.01
        self.sma_trend_period = 14

    @property
    def rsi(self):
        """RSI指标"""
        return ta.rsi(self.candles, period=self.rsi_period, sequential=True)

    @property
    def rsi_sma(self):
        """RSI基于的SMA"""
        return ta.sma(self.rsi, period=self.rsi_sma_period, sequential=True)

    @property
    def bollinger_bands(self):
        """布林带"""
        return ta.bollinger_bands(self.candles, period=self.bb_period, devfactor=self.bb_std, sequential=True)

    @property
    def bb_upper(self):
        """布林带上轨"""
        return self.bollinger_bands.upperband

    @property
    def bb_lower(self):
        """布林带下轨"""
        return self.bollinger_bands.lowerband

    @property
    def bb_middle(self):
        """布林带中轨"""
        return self.bollinger_bands.middleband

    @property
    def adx(self):
        """ADX指标"""
        return ta.adx(self.candles, period=self.adx_period, sequential=True)

    @property
    def bb_width(self):
        """布林带宽度"""
        return (self.bb_upper - self.bb_lower) / self.bb_middle

    @property
    def sma_trend(self):
        """趋势判断用的SMA"""
        return ta.sma(self.candles, period=self.sma_trend_period, sequential=True)

    def is_sideways_market(self):
        """判断是否为横盘市场"""
        # 检查数据是否足够
        if len(self.adx) < 2 or len(self.bb_width) < 2:
            return True  # 数据不足时，暂时认为是横盘，避免交易

        current_adx = self.adx[-1]
        current_bb_width = self.bb_width[-1]

        # 横盘条件：ADX较低（趋势弱）或布林带宽度较小（波动小）
        is_sideways = (current_adx < self.adx_threshold or
                       current_bb_width < self.bb_width_threshold)

        return is_sideways

    def is_uptrend(self):
        """判断是否为上行趋势"""
        # 检查数据是否足够
        if (len(self.candles) < 2 or len(self.adx) < 2 or
                len(self.sma_trend) < 2 or len(self.bb_middle) < 2):
            return False

        current_price = self.candles[-1][4]  # close price
        previous_price = self.candles[-2][4]
        current_adx = self.adx[-1]
        current_sma = self.sma_trend[-1]
        bb_mid = self.bb_middle[-1]

        # 上行趋势条件：
        # 1. ADX显示趋势强度足够（大于阈值）
        # 2. 价格在SMA上方且SMA呈上升趋势
        # 3. 价格高于布林带中轨
        # 4. 价格上涨
        uptrend_conditions = [
            current_adx >= self.adx_threshold,  # 趋势强度足够
            current_price > current_sma,  # 价格在趋势线上方
            current_price > bb_mid,  # 价格在布林带中轨上方
            current_price > previous_price,  # 价格上涨
        ]

        # 至少满足2个条件才认为是上行趋势
        return sum(uptrend_conditions) >= 2

    def is_downtrend(self):
        """判断是否为下行趋势"""
        # 检查数据是否足够
        if (len(self.candles) < 2 or len(self.adx) < 2 or
                len(self.sma_trend) < 2 or len(self.bb_middle) < 2):
            return False

        current_price = self.candles[-1][4]  # close price
        previous_price = self.candles[-2][4]
        current_adx = self.adx[-1]
        current_sma = self.sma_trend[-1]
        bb_mid = self.bb_middle[-1]

        # 下行趋势条件：
        # 1. ADX显示趋势强度足够（大于阈值）
        # 2. 价格在SMA下方且SMA呈下降趋势
        # 3. 价格低于布林带中轨
        # 4. 价格下跌
        downtrend_conditions = [
            current_adx >= self.adx_threshold,  # 趋势强度足够
            current_price < current_sma,  # 价格在趋势线下方
            current_price < bb_mid,  # 价格在布林带中轨下方
            current_price < previous_price,  # 价格下跌
        ]

        # 至少满足2个条件才认为是下行趋势
        return sum(downtrend_conditions) >= 2

    def should_long(self) -> bool:
        """做多条件"""
        # 检查数据是否足够
        if (len(self.rsi) < 2 or len(self.rsi_sma) < 2 or
                len(self.bb_lower) < 2 or len(self.bb_middle) < 2):
            return False

        # 检查是否为横盘市场
        if self.is_sideways_market():
            return False

        current_price = self.candles[-1][4]  # close price
        current_rsi = self.rsi[-1]
        current_rsi_sma = self.rsi_sma[-1]
        bb_lower = self.bb_lower[-1]
        bb_middle = self.bb_middle[-1]

        # 做多信号：价格突破布林带下轨 且 RSI SMA上穿RSI或在RSI上方
        if self.is_uptrend():
            # 上行趋势中，使用布林带中轨作为支撑
            long_signal = (current_price < bb_middle and
                           current_rsi_sma > current_rsi)
        else:
            # 非上行趋势，使用布林带下轨
            long_signal = (current_price < bb_lower and
                           current_rsi_sma > current_rsi)

        return long_signal

    def should_short(self) -> bool:
        """做空条件 - 现货交易不做空"""
        return False

    def should_cancel_entry(self) -> bool:
        """取消入场条件"""
        return False

    def go_long(self):
        """开多仓"""
        # 使用25%的可用资金
        cash_pct = 0.25
        available_balance = self.available_margin
        trade_amount = available_balance * cash_pct

        # 最小交易金额检查
        min_trade_amount = 100
        if trade_amount < min_trade_amount:
            return

        current_price = self.candles[-1][4]
        qty = utils.size_to_qty(trade_amount, current_price, precision=6)

        # 记录开仓信息
        self.log(f'开多仓: 价格={current_price:.2f}, 数量={qty:.6f}, '
                 f'RSI={self.rsi[-1]:.2f}, RSI_SMA={self.rsi_sma[-1]:.2f}, '
                 f'BB_下轨={self.bb_lower[-1]:.2f}, ADX={self.adx[-1]:.2f}')

        self.buy = qty, current_price

    def go_short(self):
        """开空仓 - 现货交易不使用"""
        pass

    def update_position(self):
        """更新持仓逻辑"""
        # 如果持有多仓，检查平仓条件
        if self.position.pnl_percentage > 0:  # 有多仓
            current_price = self.candles[-1][4]
            current_rsi = self.rsi[-1]
            current_rsi_sma = self.rsi_sma[-1]
            bb_middle = self.bb_middle[-1]

            # 平多信号：价格突破布林带中轨 且 RSI SMA下穿RSI或在RSI下方
            close_long_signal = (current_price > bb_middle and
                                 current_rsi_sma < current_rsi)

            if close_long_signal:
                self.log(f'平多仓: 价格={current_price:.2f}, '
                         f'RSI={current_rsi:.2f}, RSI_SMA={current_rsi_sma:.2f}, '
                         f'BB_中轨={bb_middle:.2f}, 收益率={self.position.pnl_percentage:.2f}%')
                self.liquidate()

    def on_open_position(self, order):
        """开仓时的回调"""
        pass

    def on_close_position(self, order):
        """平仓时的回调"""
        pass

    def terminate(self):
        """策略结束时的统计"""
        pass

    def log(self, msg, log_type='info'):
        """日志记录"""
        if log_type == 'info':
            self.logger.info(msg)
        elif log_type == 'error':
            self.logger.error(msg)
        elif log_type == 'warning':
            self.logger.warning(msg)