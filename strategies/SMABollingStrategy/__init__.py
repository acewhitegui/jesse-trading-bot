import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy


class SMABollingStrategy(Strategy):
    """
    Strategy Overview:
    1. Define overbought and oversold zones, oversold 30, overbought 70
    2. Use RSI to calculate RSI based SMA, RSI length is 14
    3. Calculate Bollinger upper and lower bands

    Entry and Exit Timing:
    1. When price breaks below Bollinger lower band and RSI based MA crosses above RSI or is above it, go long
    2. When price breaks above Bollinger upper band and RSI based MA crosses below RSI or is below RSI, close long
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
        """RSI based SMA"""
        return ta.sma(self.rsi, period=self.rsi_sma_period, sequential=True)

    @property
    def bollinger_bands(self):
        """Bollinger Bands"""
        return ta.bollinger_bands(self.candles, period=self.bb_period, sequential=True)

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
        """Bollinger band width"""
        return (self.bb_upper - self.bb_lower) / self.bb_middle

    @property
    def sma_trend(self):
        """SMA for trend determination"""
        return ta.sma(self.candles, period=self.sma_trend_period, sequential=True)

    def is_sideways_market(self):
        """Check if market is sideways"""
        # Check if data is sufficient
        if len(self.adx) < 2 or len(self.bb_width) < 2:
            return True  # When data is insufficient, assume sideways to avoid trading

        current_adx = self.adx[-1]
        current_bb_width = self.bb_width[-1]

        # Sideways conditions: Low ADX (weak trend) or small Bollinger band width (low volatility)
        is_sideways = (current_adx < self.adx_threshold or
                       current_bb_width < self.bb_width_threshold)

        self.log(
            f"{self.symbol}, sideways market checking, is sideways: {is_sideways}, "
            f"current_adx: {current_adx} with adx_threshold: {self.adx_threshold}, "
            f"current_bb_width: {current_bb_width} with bb_width_threshold: {self.bb_width_threshold}")
        return is_sideways

    def is_uptrend(self):
        """Check if market is in uptrend"""
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
        # 1. ADX shows sufficient trend strength (above threshold)
        # 2. Price is above SMA and SMA is trending upward
        # 3. Price is above Bollinger middle band
        # 4. Price is rising
        uptrend_conditions = [
            current_adx >= self.adx_threshold,  # Sufficient trend strength
            current_price > current_sma,  # Price above trend line
            current_price > bb_mid,  # Price above Bollinger middle band
            current_price > previous_price,  # Price rising
        ]

        # At least 2 conditions must be met to consider uptrend
        return sum(uptrend_conditions) >= 2

    def is_downtrend(self):
        """Check if market is in downtrend"""
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
        # 1. ADX shows sufficient trend strength (above threshold)
        # 2. Price is below SMA and SMA is trending downward
        # 3. Price is below Bollinger middle band
        # 4. Price is falling
        downtrend_conditions = [
            current_adx >= self.adx_threshold,  # Sufficient trend strength
            current_price < current_sma,  # Price below trend line
            current_price < bb_mid,  # Price below Bollinger middle band
            current_price < previous_price,  # Price falling
        ]

        # At least 2 conditions must be met to consider downtrend
        return sum(downtrend_conditions) >= 2

    def should_long(self) -> bool:
        """Long entry conditions"""
        # Check if market is sideways
        if self.is_sideways_market():
            return False

        current_price = self.candles[-1][4]  # close price
        current_rsi = self.rsi[-1]
        current_rsi_sma = self.rsi_sma[-1]
        bb_lower = self.bb_lower[-1]
        bb_middle = self.bb_middle[-1]

        # Long signal: Price breaks below Bollinger lower band and RSI SMA crosses above RSI or is above RSI
        if self.is_uptrend():
            # In uptrend, use Bollinger middle band as support
            long_signal = (current_price < bb_middle and
                           current_rsi_sma > current_rsi)
            self.log(f"{self.symbol}, long: {long_signal}, uptrend: True, "
                     f"price({current_price:.4f}) < bb_middle({bb_middle:.4f}): {current_price < bb_middle}, "
                     f"rsi_sma({current_rsi_sma:.2f}) > rsi({current_rsi:.2f}): {current_rsi_sma > current_rsi}")
        else:
            # In non-uptrend, use Bollinger lower band
            long_signal = (current_price < bb_lower and
                           current_rsi_sma > current_rsi and current_rsi_sma > self.rsi_oversold)
            self.log(f"{self.symbol}, long: {long_signal}, uptrend: False, "
                     f"price({current_price:.4f}) < bb_lower({bb_lower:.4f}): {current_price < bb_lower}, "
                     f"rsi_sma({current_rsi_sma:.2f}) > rsi({current_rsi:.2f}): {current_rsi_sma > current_rsi}, "
                     f"rsi_sma({current_rsi_sma:.2f}) > oversold({self.rsi_oversold}): {current_rsi_sma > self.rsi_oversold}")

        return long_signal

    def should_short(self) -> bool:
        """Short entry conditions - No shorting in spot trading"""
        return False

    def should_cancel_entry(self) -> bool:
        """Cancel entry conditions"""
        return True

    def go_long(self):
        """Open long position"""
        # Use 25% of available funds
        cash_pct = 0.25
        available_balance = self.available_margin
        trade_amount = available_balance * cash_pct

        # Minimum trade amount check
        min_trade_amount = 100
        if trade_amount < min_trade_amount:
            return

        current_price = self.candles[-1][4]
        qty = utils.size_to_qty(trade_amount, current_price, precision=6)

        # Log opening position info
        self.log(f'Open long: Price={current_price:.2f}, Qty={qty:.6f}, '
                 f'RSI={self.rsi[-1]:.2f}, RSI_SMA={self.rsi_sma[-1]:.2f}, '
                 f'BB_Lower={self.bb_lower[-1]:.2f}, ADX={self.adx[-1]:.2f}')

        self.buy = qty, current_price

    def go_short(self):
        """Open short position - Not used in spot trading"""
        pass

    def update_position(self):
        """Update position logic"""
        self.log(f'{self.symbol} position updated, quantity: {self.position.qty}, position info: {self.position.to_dict}')
        # If holding long position, check closing conditions
        if self.position.qty > 0:  # Has long position
            current_price = self.candles[-1][4]
            current_rsi = self.rsi[-1]
            current_rsi_sma = self.rsi_sma[-1]
            bb_middle = self.bb_middle[-1]

            # Close long signal: Price breaks above Bollinger middle band and RSI SMA crosses below RSI or is below RSI
            close_long_signal = (current_price > bb_middle and
                                 current_rsi_sma < current_rsi)

            if close_long_signal:
                self.log(f'Close long: Price={current_price:.2f}, '
                         f'RSI={current_rsi:.2f}, RSI_SMA={current_rsi_sma:.2f}, '
                         f'BB_Middle={bb_middle:.2f}, Return={self.position.pnl_percentage:.2f}%')
                self.liquidate()

    def on_open_position(self, order):
        """Callback when opening position"""
        pass

    def on_close_position(self, order):
        """Callback when closing position"""
        pass

    def terminate(self):
        """Statistics when strategy ends"""
        pass
