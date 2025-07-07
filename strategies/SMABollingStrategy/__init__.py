import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy


class SMABollingStrategy(Strategy):
    """
    Strategy Overview:
    1. Define overbought and oversold zones: oversold 30, overbought 70
    2. Use RSI to calculate an RSI-based SMA, RSI length 14
    3. Calculate Bollinger Bands

    Entry/Exit Conditions:
    1. When price breaks below the Bollinger lower band and RSI-based SMA crosses above RSI or stays above, go long
    2. When price breaks above the Bollinger upper band and RSI-based SMA crosses below RSI or stays below, exit long
    3. No trading in sideways market
    """

    def __init__(self):
        super().__init__()
        # Strategy parameters
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
        """RSI indicator"""
        return ta.rsi(self.candles, period=self.rsi_period, sequential=True)

    @property
    def rsi_sma(self):
        """SMA based on RSI"""
        return ta.sma(self.rsi, period=self.rsi_sma_period, sequential=True)

    @property
    def bollinger_bands(self):
        """Bollinger Bands"""
        return ta.bollinger_bands(self.candles, period=self.bb_period, devfactor=self.bb_std, sequential=True)

    @property
    def bb_upper(self):
        """Bollinger upper band"""
        return self.bollinger_bands.upperband

    @property
    def bb_lower(self):
        """Bollinger lower band"""
        return self.bollinger_bands.lowerband

    @property
    def bb_middle(self):
        """Bollinger middle band"""
        return self.bollinger_bands.middleband

    @property
    def adx(self):
        """ADX indicator"""
        return ta.adx(self.candles, period=self.adx_period, sequential=True)

    @property
    def bb_width(self):
        """Bollinger Band width"""
        return (self.bb_upper - self.bb_lower) / self.bb_middle

    @property
    def sma_trend(self):
        """SMA for trend determination"""
        return ta.sma(self.candles, period=self.sma_trend_period, sequential=True)

    def is_sideways_market(self):
        """Determine if market is sideways"""
        # Check if data is sufficient
        if len(self.adx) < 2 or len(self.bb_width) < 2:
            return True  # If insufficient data, treat as sideways to avoid trading

        current_adx = self.adx[-1]
        current_bb_width = self.bb_width[-1]

        # Sideways conditions: ADX low (weak trend) or Bollinger width small (low volatility)
        is_sideways = (current_adx < self.adx_threshold or
                       current_bb_width < self.bb_width_threshold)

        return is_sideways

    def is_uptrend(self):
        """Determine if uptrend"""
        # Check if data is sufficient
        if (len(self.candles) < 2 or len(self.adx) < 2 or
                len(self.sma_trend) < 2 or len(self.bb_middle) < 2):
            return False

        current_price = self.candles[-1][4]  # close price
        previous_price = self.candles[-2][4]
        current_adx = self.adx[-1]
        current_sma = self.sma_trend[-1]
        bb_mid = self.bb_middle[-1]

        # Uptrend conditions:
        # 1. ADX shows enough trend strength (above threshold)
        # 2. Price above SMA and SMA trending up
        # 3. Price above Bollinger middle band
        # 4. Price is rising
        uptrend_conditions = [
            current_adx >= self.adx_threshold,  # Trend strength
            current_price > current_sma,  # Price above trend line
            current_price > bb_mid,  # Price above mid band
            current_price > previous_price,  # Price is rising
        ]

        # At least two conditions
        return sum(uptrend_conditions) >= 2

    def is_downtrend(self):
        """Determine if downtrend"""
        # Check if data is sufficient
        if (len(self.candles) < 2 or len(self.adx) < 2 or
                len(self.sma_trend) < 2 or len(self.bb_middle) < 2):
            return False

        current_price = self.candles[-1][4]  # close price
        previous_price = self.candles[-2][4]
        current_adx = self.adx[-1]
        current_sma = self.sma_trend[-1]
        bb_mid = self.bb_middle[-1]

        # Downtrend conditions:
        # 1. ADX shows enough trend strength (above threshold)
        # 2. Price below SMA and SMA trending down
        # 3. Price below Bollinger middle band
        # 4. Price is falling
        downtrend_conditions = [
            current_adx >= self.adx_threshold,  # Trend strength
            current_price < current_sma,  # Price below trend line
            current_price < bb_mid,  # Price below mid band
            current_price < previous_price,  # Price is falling
        ]

        # At least two conditions
        return sum(downtrend_conditions) >= 2

    def should_long(self) -> bool:
        """Long conditions"""
        # Check if data is sufficient
        if (len(self.rsi) < 2 or len(self.rsi_sma) < 2 or
                len(self.bb_lower) < 2 or len(self.bb_middle) < 2):
            return False

        # Check if market is sideways
        if self.is_sideways_market():
            return False

        current_price = self.candles[-1][4]  # close price
        current_rsi = self.rsi[-1]
        current_rsi_sma = self.rsi_sma[-1]
        bb_lower = self.bb_lower[-1]
        bb_middle = self.bb_middle[-1]

        # Long signal: Price breaks below Bollinger lower band AND RSI SMA crosses above RSI or stays above
        if self.is_uptrend():
            # In uptrend, use mid band as support
            long_signal = (current_price < bb_middle and
                           current_rsi_sma > current_rsi)
        else:
            # Not uptrend, use lower band
            long_signal = (current_price < bb_lower and
                           current_rsi_sma > current_rsi)

        return long_signal

    def should_short(self) -> bool:
        """Short conditions - not used in spot trading"""
        return False

    def should_cancel_entry(self) -> bool:
        """Cancel entry conditions"""
        return False

    def go_long(self):
        """Open long position"""
        # Use 25% of available balance
        cash_pct = 0.25
        available_balance = self.available_margin
        trade_amount = available_balance * cash_pct

        # Minimum trade amount
        min_trade_amount = 100
        if trade_amount < min_trade_amount:
            return

        current_price = self.candles[-1][4]
        qty = utils.size_to_qty(trade_amount, current_price, precision=6)

        # Log entry info
        self.log(f'Go long: Price={current_price:.2f}, Quantity={qty:.6f}, '
                 f'RSI={self.rsi[-1]:.2f}, RSI_SMA={self.rsi_sma[-1]:.2f}, '
                 f'BB_Lower={self.bb_lower[-1]:.2f}, ADX={self.adx[-1]:.2f}')

        self.buy = qty, current_price

    def go_short(self):
        """Open short position - not used in spot trading"""
        pass

    def update_position(self):
        """Update open position logic"""
        # If holding a long position, check exit condition
        if self.position.pnl_percentage > 0:  # Has long position
            current_price = self.candles[-1][4]
            current_rsi = self.rsi[-1]
            current_rsi_sma = self.rsi_sma[-1]
            bb_middle = self.bb_middle[-1]

            # Exit signal: Price breaks above Bollinger middle band AND RSI SMA crosses below RSI or stays below
            close_long_signal = (current_price > bb_middle and
                                 current_rsi_sma < current_rsi)

            if close_long_signal:
                self.log(f'Exit long: Price={current_price:.2f}, '
                         f'RSI={current_rsi:.2f}, RSI_SMA={current_rsi_sma:.2f}, '
                         f'BB_Middle={bb_middle:.2f}, Return={self.position.pnl_percentage:.2f}%')
                self.liquidate()

    def on_open_position(self, order):
        """Callback when a position is opened"""
        pass

    def on_close_position(self, order):
        """Callback when a position is closed"""
        pass

    def terminate(self):
        """Strategy end statistics"""
        pass

    def log(self, msg, log_type='info'):
        """Logging"""
        if log_type == 'info':
            self.logger.info(msg)
        elif log_type == 'error':
            self.logger.error(msg)
        elif log_type == 'warning':
            self.logger.warning(msg)


