import math
import logging
from datetime import datetime

from freqtrade.persistence import Trade
from user_data.strategies.TrailingBuyStrat2 import TrailingBuyStrat2
from user_data.strategies.tbedit import tbedit

logger = logging.getLogger(__name__)


class strat_dca(tbedit):
    initial_safety_order_trigger = -0.05
    max_safety_orders = 2
    safety_order_step_scale = 2
    safety_order_volume_scale = 2

    def adjust_trade_position(self, pair: str, trade: Trade,
                              current_time: datetime, current_rate: float, current_profit: float,
                              **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None

        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'buy':
                continue
            if order.status == "closed":
                count_of_buys += 1

        if 1 <= count_of_buys <= self.max_safety_orders:

            safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                    math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / (
                            self.safety_order_step_scale - 1))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(pair, None)
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(
                        f"Initiating safety order buy #{count_of_buys} for {pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {pair}: {str(exception)}')
                    return None

        return None
