import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy


class YuanbaoSMABollingStrategyWithShort(Strategy):
    """
    优化目标：提升夏普率（支持多空双向交易）
    核心改进：
    - 新增强下降趋势判断（支持做空）
    - 多空双向信号过滤（结合多时间框架趋势）
    - 动态空头仓位管理（基于波动率）
    - 智能空头止损止盈（反向波动率控制）
    """

    def __init__(self):
        super().__init__()
        # 基础参数（可优化范围）
        self.rsi_period = 12  # RSI计算周期
        self.rsi_sma_period = 14  # RSI的SMA周期
        self.bb_period = 20  # 布林带周期
        self.adx_period = 14  # ADX周期
        self.atr_period = 14  # ATR周期（用于波动率）
        self.min_trend_period = 3  # 最小趋势确认周期

        # 策略阈值（可优化）
        self.adx_threshold = 22  # ADX趋势强度阈值
        self.bb_width_threshold = 0.015  # 布林带宽度阈值（横向市场）
        self.volume_spike_factor = 1.5  # 成交量放大倍数（过滤假突破）
        self.trend_confirmation = 2  # 趋势确认所需连续条件数

        # 风险参数（多空统一）
        self.max_risk_per_trade = 0.01  # 单笔最大风险（账户1%）
        self.stop_loss_atr_multiplier = 2.0  # 止损ATR倍数
        self.take_profit_atr_multiplier = 3.0  # 止盈ATR倍数

        self.entry_price = None

    # ------------------------------
    # 指标计算（新增多时间框架支持）
    # ------------------------------
    @property
    def rsi(self):
        return ta.rsi(self.candles, period=self.rsi_period, sequential=True)

    @property
    def rsi_sma(self):
        return ta.sma(self.rsi, period=self.rsi_sma_period, sequential=True)

    @property
    def bollinger_bands(self):
        return ta.bollinger_bands(self.candles, period=self.bb_period, sequential=True)

    @property
    def bb_upper(self):
        return self.bollinger_bands.upperband

    @property
    def bb_lower(self):
        return self.bollinger_bands.lowerband

    @property
    def bb_middle(self):
        return self.bollinger_bands.middleband

    @property
    def adx(self):
        return ta.adx(self.candles, period=self.adx_period, sequential=True)

    @property
    def bb_width(self):
        return (self.bb_upper - self.bb_lower) / self.bb_middle

    @property
    def sma_trend(self):
        return ta.sma(self.candles, period=50, sequential=True)  # 50周期SMA（长期趋势）

    @property
    def hourly_sma(self):
        """1小时图50周期SMA（多时间框架趋势确认）"""
        hourly_candles = self.get_candles(self.symbol, '1h')
        return ta.sma(hourly_candles, period=50, sequential=False)[-1] if len(hourly_candles) >= 50 else None

    @property
    def atr(self):
        return ta.atr(self.candles, period=self.atr_period, sequential=True)

    @property
    def close(self):
        """简化收盘价获取"""
        return [candle[4] for candle in self.candles]

    # ------------------------------
    # 市场状态判断（多空通用）
    # ------------------------------
    def is_sideways_market(self):
        """结合布林带宽度+ADX+RSI波动判断横向市场"""
        if len(self.adx) < 2 or len(self.bb_width) < 2:
            return True

        current_adx = self.adx[-1]
        current_bb_width = self.bb_width[-1]
        current_rsi = self.rsi[-1]
        prev_rsi = self.rsi[-2]

        # 横向市场条件：ADX弱+布林带窄+RSI波动小
        rsi_volatility = abs(current_rsi - prev_rsi)
        return (current_adx < self.adx_threshold and
                current_bb_width < self.bb_width_threshold and
                rsi_volatility < 5)  # RSI波动小于5%

    def is_strong_uptrend(self):
        """强上升趋势确认（多单开仓条件）"""
        if len(self.candles) < self.min_trend_period * 2:
            return False

        # 条件1：价格>长期SMA（50周期）
        above_sma = self.close[-1] > self.sma_trend[-1]
        # 条件2：ADX>阈值且上升（趋势强化）
        adx_rising = self.adx[-1] > self.adx_threshold and self.adx[-1] > self.adx[-2]
        # 条件3：1小时图趋势同向（多时间框架确认）
        hourly_trend_up = self.hourly_sma and self.close[-1] > self.hourly_sma
        # 条件4：成交量放大（突破有效性）
        avg_volume = utils.ema(self.candles[:, 5], period=20)
        volume_spike = self.candles[-1][5] > avg_volume * self.volume_spike_factor

        return sum([above_sma, adx_rising, hourly_trend_up, volume_spike]) >= 3

    def is_strong_downtrend(self):
        """强下降趋势确认（空单开仓条件）"""
        if len(self.candles) < self.min_trend_period * 2:
            return False

        # 条件1：价格<长期SMA（50周期）
        below_sma = self.close[-1] < self.sma_trend[-1]
        # 条件2：ADX>阈值且上升（趋势强化）
        adx_rising = self.adx[-1] > self.adx_threshold and self.adx[-1] > self.adx[-2]
        # 条件3：1小时图趋势同向（多时间框架确认）
        hourly_trend_down = self.hourly_sma and self.close[-1] < self.hourly_sma
        # 条件4：成交量放大（恐慌性抛售）
        avg_volume = utils.ema(self.candles[:, 5], period=20)
        volume_spike = self.candles[-1][5] > avg_volume * self.volume_spike_factor

        return sum([below_sma, adx_rising, hourly_trend_down, volume_spike]) >= 3

    # ------------------------------
    # 交易信号（多空双向）
    # ------------------------------
    def should_long(self) -> bool:
        """做多条件：强上升趋势+价格跌破中轨+RSI SMA上穿"""
        if self.is_sideways_market():
            return False

        # 基础信号：价格跌破布林带中轨+RSI SMA上穿
        price_below_mid = self.close[-1] < self.bb_middle[-1]
        rsi_sma_cross_above = self.rsi_sma[-1] > self.rsi[-1] and self.rsi_sma[-2] <= self.rsi[-2]

        # 强趋势过滤：仅在强上升趋势中交易
        in_strong_uptrend = self.is_strong_uptrend()

        return price_below_mid and rsi_sma_cross_above and in_strong_uptrend

    def should_short(self) -> bool:
        """做空条件：强下降趋势+价格突破上轨+RSI SMA下穿"""
        if self.is_sideways_market():
            return False

        # 基础信号：价格突破布林带上轨+RSI SMA下穿
        price_above_upper = self.close[-1] > self.bb_upper[-1]
        rsi_sma_cross_below = self.rsi_sma[-1] < self.rsi[-1] and self.rsi_sma[-2] >= self.rsi[-2]

        # 强趋势过滤：仅在强下降趋势中交易
        in_strong_downtrend = self.is_strong_downtrend()

        return price_above_upper and rsi_sma_cross_below and in_strong_downtrend

    # ------------------------------
    # 动态仓位管理（多空统一）
    # ------------------------------
    def calculate_position_size(self, is_long: bool) -> float:
        """基于ATR的动态仓位计算（控制单笔风险）"""
        if self.available_margin < 10:  # 最小可用资金限制
            return 0

        current_price = self.close[-1]
        atr_value = self.atr[-1] if len(self.atr) >= self.atr_period else 0.1  # 防止除零

        # 单笔最大风险=账户余额*1%
        max_risk = self.available_margin * self.max_risk_per_trade
        # 仓位大小=最大风险/(止损幅度*价格)（做空时价格用当前价）
        position_size = max_risk / (self.stop_loss_atr_multiplier * atr_value)

        # 最小交易量限制（避免滑点过大）
        min_qty = 10  # 根据具体交易对调整
        qty = utils.size_to_qty(position_size, current_price, precision=6)
        return max(qty, min_qty) if is_long else max(qty, min_qty)  # 空头最小数量需符合交易所要求

    # ------------------------------
    # 多空仓位管理（开仓/平仓）
    # ------------------------------
    def go_long(self):
        qty = self.calculate_position_size(is_long=True)
        if qty > 0:
            self.buy = qty, self.close[-1]

    def go_short(self):
        qty = self.calculate_position_size(is_long=False)
        if qty > 0:
            self.sell = qty, self.close[-1]  # 现货做空使用sell操作

    # ------------------------------
    # 多空止损止盈（动态调整）
    # ------------------------------
    def update_position(self):
        if self.position.is_long:
            # 多头止损：入场价 - 止损ATR倍数*ATR
            # 多头止盈：入场价 + 止盈ATR倍数*ATR
            if len(self.candles) >= 20:
                recent_atr = ta.atr(self.candles[-20:], period=self.atr_period, sequential=False)[-1]
                stop_loss_price = self.entry_price - self.stop_loss_atr_multiplier * recent_atr
                take_profit_price = self.entry_price + self.take_profit_atr_multiplier * self.atr[-1]

                # 更新止损止盈
                self.stop_loss = stop_loss_price
                self.take_profit = take_profit_price

        elif self.position.is_short:
            # 空头止损：入场价 + 止损ATR倍数*ATR（防止价格上涨）
            # 空头止盈：入场价 - 止盈ATR倍数*ATR（防止价格下跌）
            if len(self.candles) >= 20:
                recent_atr = ta.atr(self.candles[-20:], period=self.atr_period, sequential=False)[-1]
                stop_loss_price = self.entry_price + self.stop_loss_atr_multiplier * recent_atr  # 空头止损向上
                take_profit_price = self.entry_price - self.take_profit_atr_multiplier * self.atr[-1]  # 空头止盈向下

                # 更新止损止盈
                self.stop_loss = stop_loss_price
                self.take_profit = take_profit_price

        # 触发止盈/止损（Jesse自动处理，当价格触及时平仓）
        # 注意：需确保策略参数中启用了自动止损止盈（默认开启）

    # ------------------------------
    # 其他回调函数
    # ------------------------------
    def on_open_position(self, order):
        """开仓后记录入场价"""
        self.entry_price = self.close[-1]  # 记录当前收盘价作为入场价

    def on_close_position(self, order):
        """平仓后重置止损止盈"""
        self.stop_loss = None
        self.take_profit = None
        self.entry_price = None  # 清空入场价记录

    def hyperparameters(self):
        # 扩展参数范围以适应多空场景（部分参数需对称调整）
        return [
            {'name': 'rsi_period', 'type': int, 'min': 8, 'max': 16, 'default': 12},
            {'name': 'rsi_sma_period', 'type': int, 'min': 10, 'max': 18, 'default': 14},
            {'name': 'bb_period', 'type': int, 'min': 16, 'max': 24, 'default': 20},
            {'name': 'adx_threshold', 'type': int, 'min': 18, 'max': 26, 'default': 22},
            {'name': 'bb_width_threshold', 'type': float, 'min': 0.01, 'max': 0.02, 'step': 0.002, 'default': 0.015},
            {'name': 'volume_spike_factor', 'type': float, 'min': 1.3, 'max': 1.8, 'step': 0.1, 'default': 1.5},
            {'name': 'stop_loss_atr_multiplier', 'type': float, 'min': 1.5, 'max': 2.5, 'step': 0.2, 'default': 2.0},
            {'name': 'take_profit_atr_multiplier', 'type': float, 'min': 2.5, 'max': 3.5, 'step': 0.2, 'default': 3.0},
        ]
