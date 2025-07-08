import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy
import numpy as np


class YuanbaoSMABollingStrategy(Strategy):
    def hyperparameters(self) -> list:
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

    # ------------------------------
    # Indicator Calculations (multi-timeframe)
    # ------------------------------
    @property
    def rsi(self):
        return ta.rsi(self.candles, period=self.hp['rsi_period'], sequential=True)

    @property
    def rsi_sma(self):
        return ta.sma(self.rsi, period=self.hp['rsi_sma_period'], sequential=True)

    @property
    def bollinger_bands(self):
        return ta.bollinger_bands(self.candles, period=self.hp['bb_period'], sequential=True)

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
        return ta.adx(self.candles, sequential=True)  # default period

    @property
    def bb_width(self):
        if len(self.bb_middle) == 0:
            return np.array([])

        middle = np.array(self.bb_middle)
        upper = np.array(self.bb_upper)
        lower = np.array(self.bb_lower)

        middle_safe = np.where(middle == 0, 1e-8, middle)
        return (upper - lower) / middle_safe

    @property
    def sma_trend(self):
        return ta.sma(self.candles, period=50, sequential=True)

    @property
    def hourly_sma(self):
        try:
            hourly_candles = self.get_candles(self.exchange, self.symbol, '1h')
            if hourly_candles is not None and len(hourly_candles) >= 50:
                return ta.sma(hourly_candles, period=50, sequential=False)
            return None
        except:
            return None

    @property
    def atr(self):
        return ta.atr(self.candles, sequential=True)  # default period

    @property
    def close_prices(self):
        return self.candles[:, 2].astype(float)  # column 2 is close price

    @property
    def current_close(self):
        """Get the current close price as a scalar"""
        return float(self.candles[-1, 2])

    @property
    def volume(self):
        return self.candles[:, 5].astype(float)  # column 5 is volume

    @property
    def volume_ema(self):
        return ta.ema(self.volume, period=20, sequential=True)

    # ------------------------------
    # Market State Detection
    # ------------------------------
    def is_sideways_market(self):
        if len(self.adx) < 2 or len(self.bb_width) < 2 or len(self.rsi) < 2:
            return True

        current_adx = self.adx[-1]
        current_bb_width = self.bb_width[-1]
        current_rsi = self.rsi[-1]
        prev_rsi = self.rsi[-2]
        rsi_volatility = abs(current_rsi - prev_rsi)

        return (current_adx < self.hp['adx_threshold'] and
                current_bb_width < self.hp['bb_width_threshold'] and
                rsi_volatility < 5)

    def is_strong_uptrend(self):
        if len(self.candles) < 40:
            return False

        current_close = self.current_close
        above_sma = current_close > self.sma_trend[-1]
        adx_rising = self.adx[-1] > self.hp['adx_threshold'] and self.adx[-1] > self.adx[-2]
        hourly_trend_up = self.hourly_sma is not None and current_close > self.hourly_sma

        volume_spike = self.volume[-1] > self.volume_ema[-1] * self.hp['volume_spike_factor']

        return sum([above_sma, adx_rising, hourly_trend_up, volume_spike]) >= 2

    def is_strong_downtrend(self):
        if len(self.candles) < 40:
            return False

        current_close = self.current_close
        below_sma = current_close < self.sma_trend[-1]
        adx_rising = self.adx[-1] > self.hp['adx_threshold'] and self.adx[-1] > self.adx[-2]
        hourly_trend_down = self.hourly_sma is not None and current_close < self.hourly_sma

        volume_spike = self.volume[-1] > self.volume_ema[-1] * self.hp['volume_spike_factor']

        return sum([below_sma, adx_rising, hourly_trend_down, volume_spike]) >= 2

    # ------------------------------
    # Trading Signals
    # ------------------------------
    def should_long(self) -> bool:
        if (len(self.rsi) < 2 or len(self.rsi_sma) < 2 or
                len(self.bb_middle) < 1 or len(self.candles) < 1):
            return False

        if self.is_sideways_market():
            return False

        current_close = self.current_close
        price_below_mid = current_close < self.bb_middle[-1]
        rsi_cross_above = self.rsi[-1] > self.rsi_sma[-1] and self.rsi[-2] <= self.rsi_sma[-2]
        strong_uptrend = self.is_strong_uptrend()

        return price_below_mid and rsi_cross_above and strong_uptrend

    def should_short(self) -> bool:
        if (len(self.rsi) < 2 or len(self.rsi_sma) < 2 or
                len(self.bb_upper) < 1 or len(self.candles) < 1):
            return False

        if self.is_sideways_market():
            return False

        current_close = self.current_close
        price_above_upper = current_close > self.bb_upper[-1]
        rsi_cross_below = self.rsi[-1] < self.rsi_sma[-1] and self.rsi[-2] >= self.rsi_sma[-2]
        strong_downtrend = self.is_strong_downtrend()

        return price_above_upper and rsi_cross_below and strong_downtrend

    # ------------------------------
    # Dynamic Position Sizing
    # ------------------------------
    def calculate_position_size(self) -> float:
        balance = self.balance
        current_price = self.current_close

        # Early validation checks
        if balance <= 0 or current_price <= 0:
            return 0

        # Minimum balance requirement
        if balance < 50:  # Increased from 10 to 50
            return 0

        # Calculate ATR value
        if len(self.atr) > 0:
            atr_value = self.atr[-1]
        else:
            atr_value = abs(current_price * 0.02)  # Fallback to 2% of price

        # Ensure ATR is reasonable
        if atr_value <= 0 or atr_value > current_price * 0.1:  # ATR shouldn't be more than 10% of price
            atr_value = current_price * 0.02

        # Calculate position size based on risk
        max_risk = balance * 0.01  # 1% risk per trade
        stop_loss_distance = self.hp['stop_loss_atr_multiplier'] * atr_value

        # Ensure stop loss distance is reasonable
        if stop_loss_distance <= 0:
            stop_loss_distance = current_price * 0.02

        position_size = max_risk / stop_loss_distance

        # Convert to quantity
        qty = utils.size_to_qty(position_size, current_price, fee_rate=self.fee_rate)

        # Calculate minimum quantity based on balance and price
        min_position_size = max(balance * 0.001, 20)  # At least 0.1% of balance or $20
        min_qty = utils.size_to_qty(min_position_size, current_price, fee_rate=self.fee_rate)

        # Ensure we have a reasonable minimum
        if min_qty <= 0:
            min_qty = 0.001  # Fallback minimum

        # Return the larger of calculated qty or minimum qty
        final_qty = max(qty, min_qty)

        # Final safety check
        if final_qty <= 0:
            final_qty = 0.001  # Absolute minimum fallback

        return final_qty

    # ------------------------------
    # Position Management (Entry/Exit)
    # ------------------------------
    def go_long(self):
        qty = self.calculate_position_size()
        if qty > 0:  # Additional safety check
            self.buy = qty, self.current_close

    def go_short(self):
        qty = self.calculate_position_size()
        if qty > 0:  # Additional safety check
            self.sell = qty, self.current_close

    # ------------------------------
    # Adaptive Stop Loss & Take Profit
    # ------------------------------
    def on_open_position(self, order) -> None:
        if len(self.atr) == 0:
            return

        current_atr = self.atr[-1]

        if self.is_long:
            stop_loss_price = self.position.entry_price - self.hp['stop_loss_atr_multiplier'] * current_atr
            take_profit_price = self.position.entry_price + self.hp['take_profit_atr_multiplier'] * current_atr
            self.stop_loss = self.position.qty, stop_loss_price
            self.take_profit = self.position.qty, take_profit_price
        elif self.is_short:
            stop_loss_price = self.position.entry_price + self.hp['stop_loss_atr_multiplier'] * current_atr
            take_profit_price = self.position.entry_price - self.hp['take_profit_atr_multiplier'] * current_atr
            self.stop_loss = self.position.qty, stop_loss_price
            self.take_profit = self.position.qty, take_profit_price

    def should_cancel_entry(self) -> bool:
        return False
