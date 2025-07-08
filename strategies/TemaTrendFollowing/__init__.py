import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy


class TemaTrendFollowing(Strategy):
    @property
    def short_term_trend(self):
        # Get short-term trend using TEMA crossover
        tema10 = ta.tema(self.candles, 10)
        tema80 = ta.tema(self.candles, 80)

        if tema10 > tema80:
            return 1  # Uptrend
        else:
            return -1  # Downtrend

    @property
    def long_term_trend(self):
        # Get long-term trend using TEMA crossover on 4h timeframe
        candles_4h = self.get_candles(self.exchange, self.symbol, '4h')
        tema20 = ta.tema(candles_4h, 20)
        tema70 = ta.tema(candles_4h, 70)

        if tema20 > tema70:
            return 1  # Uptrend
        else:
            return -1  # Downtrend

    @property
    def tema10(self):
        return ta.tema(self.candles, 10)

    @property
    def tema80(self):
        return ta.tema(self.candles, 80)

    @property
    def tema20_4h(self):
        candles_4h = self.get_candles(self.exchange, self.symbol, '4h')
        return ta.tema(candles_4h, 20)

    @property
    def tema70_4h(self):
        candles_4h = self.get_candles(self.exchange, self.symbol, '4h')
        return ta.tema(candles_4h, 70)

    @property
    def atr(self):
        return ta.atr(self.candles)

    @property
    def adx(self):
        return ta.adx(self.candles)

    @property
    def cmo(self):
        return ta.cmo(self.candles)

    def should_long(self) -> bool:
        # Check if all conditions for long trade are met
        return (
                self.short_term_trend == 1 and
                self.long_term_trend == 1 and
                self.adx > 40 and
                self.cmo > 40
        )

    def should_short(self) -> bool:
        # Check if all conditions for short trade are met (opposite of long)
        return (
                self.short_term_trend == -1 and
                self.long_term_trend == -1 and
                self.adx > 40 and
                self.cmo < -40
        )

    def go_long(self):
        # Calculate entry, stop and position size
        entry_price = self.price - self.atr  # Limit order 1 ATR below current price
        stop_loss_price = entry_price - (self.atr * 4)  # Stop loss 4 ATR below entry

        # Risk 3% of available margin
        risk_qty = utils.risk_to_qty(self.available_margin, 3, entry_price, stop_loss_price, fee_rate=self.fee_rate)
        max_qty = utils.size_to_qty(0.30 * self.available_margin, entry_price, fee_rate=self.fee_rate)
        qty = min(risk_qty, max_qty)
        # Place the order
        self.buy = qty * 3, entry_price

    def go_short(self):
        # Calculate entry, stop and position size
        entry_price = self.price + self.atr  # Limit order 1 ATR above current price
        stop_loss_price = entry_price + (self.atr * 4)  # Stop loss 4 ATR above entry

        # Risk 3% of available margin
        qty = utils.risk_to_qty(self.available_margin, 3, entry_price, stop_loss_price, fee_rate=self.fee_rate)

        # Place the order
        self.sell = qty * 3, entry_price

    def should_cancel_entry(self) -> bool:
        return True

    def on_open_position(self, order) -> None:
        if self.is_long:
            # Set stop loss and take profit for long position
            self.stop_loss = self.position.qty, self.position.entry_price - (self.atr * 4)
            self.take_profit = self.position.qty, self.position.entry_price + (self.atr * 3)
        elif self.is_short:
            # Set stop loss and take profit for short position
            self.stop_loss = self.position.qty, self.position.entry_price + (self.atr * 4)
            self.take_profit = self.position.qty, self.position.entry_price - (self.atr * 3)

    def hyperparameters(self):
        """
        Returns a list of dicts describing hyperparameters for optimization.
        Each dict contains 'name', 'type', 'min', 'max', and 'default' keys.
        """
        return [
            {'name': 'tema_short_period', 'type': int, 'min': 5, 'max': 20, 'default': 10},
            {'name': 'tema_long_period', 'type': int, 'min': 50, 'max': 120, 'default': 80},
            {'name': 'tema_4h_short_period', 'type': int, 'min': 10, 'max': 30, 'default': 20},
            {'name': 'tema_4h_long_period', 'type': int, 'min': 50, 'max': 100, 'default': 70},
            {'name': 'adx_period', 'type': int, 'min': 10, 'max': 25, 'default': 14},
            {'name': 'adx_threshold', 'type': int, 'min': 25, 'max': 50, 'default': 40},
            {'name': 'cmo_period', 'type': int, 'min': 10, 'max': 25, 'default': 14},
            {'name': 'cmo_threshold', 'type': int, 'min': 30, 'max': 50, 'default': 40},
            {'name': 'atr_period', 'type': int, 'min': 10, 'max': 25, 'default': 14},
            {'name': 'atr_entry_multiplier', 'type': float, 'min': 0.5, 'max': 2.0, 'step': 0.1, 'default': 1.0},
            {'name': 'atr_stop_multiplier', 'type': float, 'min': 2.0, 'max': 6.0, 'step': 0.5, 'default': 4.0},
            {'name': 'atr_tp_multiplier', 'type': float, 'min': 1.5, 'max': 5.0, 'step': 0.5, 'default': 3.0},
            {'name': 'risk_percentage', 'type': float, 'min': 1.0, 'max': 5.0, 'step': 0.5, 'default': 3.0},
        ]

    def dna(self) -> str:
        symbol = self.symbol
        dna_dict = {
            "BTC-USDT": ''
        }
        return dna_dict.get(symbol)
