import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy


class YuanbaoSMABollingStrategy(Strategy):
    """
    Objective: Improve Sharpe ratio (supports both long and short trades)
    Core Enhancements:
    - Enhanced downtrend detection (enable shorts)
    - Dual-side signal filtering (multi-timeframe trend check)
    - Dynamic position sizing (volatility-based)
    - Intelligent SL/TP (volatility adaptive)
    - More robust entry/exit rules for trending/sideways
    """

    def __init__(self):
        super().__init__()
        # Base parameters (can be optimized)
        self.rsi_period = 12  # RSI lookback
        self.rsi_sma_period = 14  # RSI SMA lookback
        self.bb_period = 20  # Bollinger Bands period
        self.adx_period = 14  # ADX period
        self.atr_period = 14  # ATR period (volatility)
        self.min_trend_period = 3  # Minimum bars for trend confirmation

        # Strategy thresholds (can be optimized)
        self.adx_threshold = 22
        self.bb_width_threshold = 0.015  # Flat market filter
        self.volume_spike_factor = 1.5  # Volume breakout filter
        self.trend_confirmation = 2  # Minimum signals to confirm trend

        # Risk settings (same for both sides)
        self.max_risk_per_trade = 0.01  # 1% risk per trade
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 3.0

        self.entry_price = None

    # ------------------------------
    # Indicator Calculations (multi-timeframe)
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
        # Prevent division by zero
        base = self.bb_middle if min(self.bb_middle) > 0 else [v if v != 0 else 1e-6 for v in self.bb_middle]
        return (self.bb_upper - self.bb_lower) / base

    @property
    def sma_trend(self):
        return ta.sma(self.candles, period=50, sequential=True)

    @property
    def hourly_sma(self):
        """1h SMA50 for confirmation."""
        hourly_candles = self.get_candles(self.exchange, self.symbol, '1h')
        if hourly_candles is not None and len(hourly_candles) >= 50:
            return ta.sma(hourly_candles, period=50, sequential=False)[-1]
        return None

    @property
    def atr(self):
        return ta.atr(self.candles, period=self.atr_period, sequential=True)

    @property
    def close(self):
        return [candle[4] for candle in self.candles]

    @property
    def volume(self):
        return [candle[5] for candle in self.candles]

    # ------------------------------
    # Market State Detection
    # ------------------------------
    def is_sideways_market(self):
        """Filter for consolidating or flat markets using BB width, ADX, and RSI volatility."""
        if len(self.adx) < 2 or len(self.bb_width) < 2:
            return True

        current_adx = self.adx[-1]
        current_bb_width = self.bb_width[-1]
        current_rsi = self.rsi[-1]
        prev_rsi = self.rsi[-2]
        rsi_volatility = abs(current_rsi - prev_rsi)

        # Sideways if weak ADX, narrow BB, and low RSI movement
        return (current_adx < self.adx_threshold and
                current_bb_width < self.bb_width_threshold and
                rsi_volatility < 5)

    def is_strong_uptrend(self):
        """Confirm strong uptrend for longs: price > SMA, ADX strong, multi-timeframe align, volume spike."""
        if len(self.candles) < self.min_trend_period * 2:
            return False

        above_sma = self.close[-1] > self.sma_trend[-1]
        adx_rising = self.adx[-1] > self.adx_threshold and self.adx[-1] > self.adx[-2]
        hourly_trend_up = self.hourly_sma is not None and self.close[-1] > self.hourly_sma
        avg_volume = utils.ema(self.volume, period=20)
        volume_spike = self.volume[-1] > avg_volume * self.volume_spike_factor

        # Require at least 3/4 conditions
        return sum([above_sma, adx_rising, hourly_trend_up, volume_spike]) >= 3

    def is_strong_downtrend(self):
        """Confirm strong downtrend for shorts: price < SMA, ADX strong, multi-timeframe align, volume spike."""
        if len(self.candles) < self.min_trend_period * 2:
            return False

        below_sma = self.close[-1] < self.sma_trend[-1]
        adx_rising = self.adx[-1] > self.adx_threshold and self.adx[-1] > self.adx[-2]
        hourly_trend_down = self.hourly_sma is not None and self.close[-1] < self.hourly_sma
        avg_volume = utils.ema(self.volume, period=20)
        volume_spike = self.volume[-1] > avg_volume * self.volume_spike_factor

        return sum([below_sma, adx_rising, hourly_trend_down, volume_spike]) >= 3

    # ------------------------------
    # Trading Signals
    # ------------------------------
    def should_long(self) -> bool:
        """
        Long entry:
            - Not sideways
            - Price pulls back below BB mid
            - RSI crosses up above its SMA (momentum shift)
            - Confirm strong uptrend
        """
        if self.is_sideways_market():
            return False

        price_below_mid = self.close[-1] < self.bb_middle[-1]
        rsi_cross_above = self.rsi[-1] > self.rsi_sma[-1] and self.rsi[-2] <= self.rsi_sma[-2]
        strong_uptrend = self.is_strong_uptrend()

        return price_below_mid and rsi_cross_above and strong_uptrend

    def should_short(self) -> bool:
        """
        Short entry:
            - Not sideways
            - Price pushes above BB upper
            - RSI crosses below its SMA (momentum down)
            - Confirm strong downtrend
        """
        if self.is_sideways_market():
            return False

        price_above_upper = self.close[-1] > self.bb_upper[-1]
        rsi_cross_below = self.rsi[-1] < self.rsi_sma[-1] and self.rsi[-2] >= self.rsi_sma[-2]
        strong_downtrend = self.is_strong_downtrend()

        return price_above_upper and rsi_cross_below and strong_downtrend

    # ------------------------------
    # Dynamic Position Sizing
    # ------------------------------
    def calculate_position_size(self, is_long: bool) -> float:
        """ATR-based position sizing, capped per trade risk."""
        if self.available_margin < 10:
            return 0

        current_price = self.close[-1]
        atr_value = self.atr[-1] if len(self.atr) >= self.atr_period else max(0.1, abs(self.close[-1] * 0.01))
        max_risk = self.available_margin * self.max_risk_per_trade

        position_size = max_risk / (self.stop_loss_atr_multiplier * atr_value)
        min_qty = 10
        qty = utils.size_to_qty(position_size, current_price, precision=6)
        return max(qty, min_qty)

    # ------------------------------
    # Position Management (Entry/Exit)
    # ------------------------------
    def go_long(self):
        qty = self.calculate_position_size(is_long=True)
        if qty > 0:
            self.buy = qty, self.close[-1]

    def go_short(self):
        qty = self.calculate_position_size(is_long=False)
        if qty > 0:
            self.sell = qty, self.close[-1]

    # ------------------------------
    # Adaptive Stop Loss & Take Profit
    # ------------------------------
    def update_position(self):
        # Use the last 20 bars ATR for recency
        if self.position.is_long:
            if len(self.candles) >= 20:
                recent_atr = ta.atr(self.candles[-20:], period=self.atr_period, sequential=False)[-1]
                stop_loss_price = self.entry_price - self.stop_loss_atr_multiplier * recent_atr
                take_profit_price = self.entry_price + self.take_profit_atr_multiplier * recent_atr

                self.stop_loss = stop_loss_price
                self.take_profit = take_profit_price

        elif self.position.is_short:
            if len(self.candles) < 20:
                return

            recent_atr = ta.atr(self.candles[-20:], period=self.atr_period, sequential=False)[-1]
            stop_loss_price = self.entry_price + self.stop_loss_atr_multiplier * recent_atr
            take_profit_price = self.entry_price - self.take_profit_atr_multiplier * recent_atr

            self.stop_loss = stop_loss_price
            self.take_profit = take_profit_price

    # ------------------------------
    # Other Callbacks
    # ------------------------------
    def on_open_position(self, order):
        """Record entry price when position is opened."""
        self.entry_price = self.close[-1]

    def on_close_position(self, order):
        """Clear all stops and entry price when position is closed."""
        self.stop_loss = None
        self.take_profit = None
        self.entry_price = None

    def hyperparameters(self):
        # Extended ranges for long & short symmetry
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
