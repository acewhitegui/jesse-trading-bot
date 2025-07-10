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
        self.rsi_oversold = 28
        self.adx_period = 12
        self.adx_threshold = 22
        self.bb_width_threshold = 0.01
        self.stop_loss_factor = 4
        # high time framework
        self.short_tema_short_period = 10
        self.short_tema_long_period = 80
        self.long_tema_short_period = 20
        self.long_tema_long_period = 70

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
    def atr(self):
        return ta.atr(self.candles)

    @property
    def bb_width(self):
        """Bollinger band width"""
        return (self.bb_upper - self.bb_lower) / self.bb_middle

    @property
    def short_term_trend(self):
        # Get short-term trend using TEMA crossover
        short_tema_short = ta.tema(self.candles, self.short_tema_short_period)
        short_tema_long = ta.tema(self.candles, self.short_tema_long_period)

        if short_tema_short > short_tema_long:
            return 1  # Uptrend
        else:
            return -1  # Downtrend

    @property
    def long_term_trend(self):
        # Get long-term trend using TEMA crossover on 4h timeframe
        candles_4h = self.get_candles(self.exchange, self.symbol, '4h')
        long_tema_short = ta.tema(candles_4h, self.long_tema_short_period)
        long_tema_long = ta.tema(candles_4h, self.long_tema_long_period)

        if long_tema_short > long_tema_long:
            return 1  # Uptrend
        else:
            return -1  # Downtrend

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
        return self.short_term_trend == 1 and self.long_term_trend == 1

    def is_downtrend(self):
        """Check if market is in downtrend"""
        # Check if data is sufficient
        return self.short_term_trend == -1 and self.long_term_trend == -1

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
                           current_rsi < current_rsi_sma)
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
        cash_pct = 0.50
        available_balance = self.available_margin
        trade_amount = available_balance * cash_pct

        # Minimum trade amount check
        min_trade_amount = 25
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
        self.log(
            f'{self.symbol} position updated, quantity: {self.position.qty}, position info: {self.position.to_dict}')
        # If holding long position, check closing conditions
        if self.position.qty > 0:  # Has long position
            current_price = self.candles[-1][4]
            current_rsi = self.rsi[-1]
            current_rsi_sma = self.rsi_sma[-1]
            bb_middle = self.bb_middle[-1]
            bb_upper = self.bb_upper[-1]

            # Close long signal: Price breaks above Bollinger middle band and RSI SMA crosses below RSI or is below RSI
            if self.is_uptrend():
                close_long_signal = (current_price > bb_upper and
                                     current_rsi_sma < current_rsi)
            else:
                # 下跌趋势，需要止损
                close_long_signal = (current_price > bb_middle and
                                     current_rsi_sma < current_rsi)
                self.stop_loss = self.position.qty, self.position.entry_price - (self.atr * self.stop_loss_factor)

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

    def hyperparameters(self):
        """
        Returns a list of dicts describing hyperparameters for optimization.
        Each dict contains 'name', 'type', 'min', 'max', and 'default' keys.
        """
        return [
            {'name': 'rsi_period', 'type': int, 'min': 8, 'max': 20, 'default': 14},
            {'name': 'rsi_sma_period', 'type': int, 'min': 8, 'max': 24, 'default': 14},
            {'name': 'bb_period', 'type': int, 'min': 10, 'max': 40, 'default': 20},
            {'name': 'adx_period', 'type': int, 'min': 8, 'max': 24, 'default': 14},
            {'name': 'adx_threshold', 'type': int, 'min': 10, 'max': 40, 'default': 25},
            {'name': 'bb_width_threshold', 'type': float, 'min': 0.005, 'max': 0.05, 'default': 0.02},
            {'name': 'rsi_oversold', 'type': int, 'min': 20, 'max': 40, 'default': 30},
            {'name': 'sma_trend_period', 'type': int, 'min': 8, 'max': 30, 'default': 20},
            {'name': 'stop_loss_factor', 'type': int, 'min': 2, 'max': 7, 'default': 4},
        ]

    def dna(self) -> str:
        symbol = self.symbol
        dna_dict = {
            "BTC-USDT": "eyJhZHhfcGVyaW9kIjogOSwgImFkeF90aHJlc2hvbGQiOiAzMCwgImJiX3BlcmlvZCI6IDMyLCAiYmJfd2lkdGhfdGhy"
                        "ZXNob2xkIjogMC4wNDcsICJyc2lfcGVyaW9kIjogMjEsICJyc2lfc21hX3BlcmlvZCI6IDE4LCAic21hX3RyZW5kX3BlcmlvZCI6IDEyfQ==",
            "ETH-USDT": "eyJhZHhfcGVyaW9kIjogMTAsICJhZHhfdGhyZXNob2xkIjogMTAsICJiYl9wZXJpb2QiOiAxNSwgImJiX3dpZHRoX3RocmVz"
                        "aG9sZCI6IDAuMDQxLCAicnNpX3BlcmlvZCI6IDE2LCAicnNpX3NtYV9wZXJpb2QiOiAxNCwgInNtYV90cmVuZF9wZXJpb2QiOiAyMn0=",
            "XRP-USDT": "eyJhZHhfcGVyaW9kIjogMTcsICJhZHhfdGhyZXNob2xkIjogMzUsICJiYl9wZXJpb2QiOiAxOCwgImJiX3dpZHRoX3RocmV"
                        "zaG9sZCI6IDAuMDA4LCAicnNpX3BlcmlvZCI6IDgsICJyc2lfc21hX3BlcmlvZCI6IDEzLCAic21hX3RyZW5kX3BlcmlvZCI6IDI1fQ=="
        }
        return dna_dict.get(symbol, "")
