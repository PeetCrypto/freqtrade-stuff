# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
# --------------------------------

import logging
import pandas as pd
import numpy as np
import datetime
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class YourStrat(IStrategy):
    # replace this by your strategy
    pass

class TrailingBuyStrat(YourStrat):
    # This class is designed to heritate from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > intial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #
    # if process_only_new_candles = True, then you need to use 1m timeframe (and normal strat timeframe as informative)
    # if process_only_new_candles = False, it will use ticker data and you won't need to change anything

    trailing_buy_order_enabled = True
    trailing_buy_offset = 0.005  # rebound limit before a buy in % of initial price
    # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
    trailing_buy_max = 0.1  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max)

    process_only_new_candles = False

    custom_info = dict()

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None
    }

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            self.custom_info[pair]['trailing_buy'] = self.init_trailing_dict
            logger.info(f'STOP trailing buy for {pair} because of {tag}')
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        if not metadata["pair"] in self.custom_info:
            self.custom_info[metadata["pair"]] = dict()
        if not 'trailing_buy' in self.custom_info[metadata['pair']]:
            self.custom_info[metadata["pair"]]['trailing_buy'] = self.init_trailing_dict
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)
        self.custom_info[pair]['trailing_buy'] = self.init_trailing_dict
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        def get_local_min(x):
            win = dataframe.loc[:, 'barssince_last_buy'].iloc[x.shape[0] - 1].astype('int')
            win = max(win, 0)
            return pd.Series(x).rolling(window=win).min().iloc[-1]

        dataframe = super().populate_buy_trend(dataframe, metadata)
        dataframe = dataframe.rename(columns={"buy": "pre_buy"})

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):  # trailing live dry ticker, 1m
            last_candle = dataframe.iloc[-1].squeeze()
            if not self.process_only_new_candles:
                current_price = self.get_current_price(metadata["pair"])
            else:
                current_price = last_candle['close']
            dataframe['buy'] = 0
            if not self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started'] and last_candle['pre_buy'] == 1:
                self.custom_info[metadata["pair"]]['trailing_buy'] = {
                    'trailing_buy_order_started': True,
                    'trailing_buy_order_uplimit': last_candle['close'],
                    'start_trailing_price': last_candle['close'],
                    'buy_tag': last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                }
                logger.info(f'start trailing buy for {metadata["pair"]} at {last_candle["close"]}')
            elif self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started']:
                if current_price < self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit']:
                    # update uplimit
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + self.trailing_buy_offset), self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'])
                    logger.info(f'update trailing buy for {metadata["pair"]} at {self.custom_info[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                elif current_price < self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price']:
                    # buy ! current price > uplimit but lower thant starting price
                    dataframe.iloc[-1, dataframe.columns.get_loc('buy')] = 1
                    ratio = "%.2f" % ((1 - current_price / self.custom_info[metadata['pair']]['trailing_buy']['start_trailing_price']) * 100)
                    if 'buy_tag' in dataframe.columns:
                        dataframe.iloc[-1, dataframe.columns.get_loc('buy_tag')] = f"{self.custom_info[metadata['pair']]['trailing_buy']['buy_tag']} ({ratio} %)"
                    # stop trailing when buy signal ! prevent from buying much higher price when slot is free
                    self.custom_info[metadata["pair"]]['trailing_buy'] = self.init_trailing_dict
                    logger.info(f'STOP trailing buy for {metadata["pair"]} because I buy it {ratio}')
                elif current_price > (self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price'] * (1 + self.trailing_buy_max)):
                    self.custom_info[metadata["pair"]]['trailing_buy'] = self.init_trailing_dict
                    logger.info(f'STOP trailing buy for {metadata["pair"]} because of the price is higher than starting prix * {1 + self.trailing_buy_max}')
                else:
                    logger.info(f'price to high for {metadata["pair"]} at {current_price} vs {self.custom_info[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
        elif self.trailing_buy_order_enabled:
            # FOR BACKTEST
            # NOT WORKING
            dataframe.loc[
                (dataframe['pre_buy'] == 1) &
                (dataframe['pre_buy'].shift() == 0)
                , 'pre_buy_switch'] = 1
            dataframe['pre_buy_switch'] = dataframe['pre_buy_switch'].fillna(0)

            dataframe['barssince_last_buy'] = dataframe['pre_buy_switch'].groupby(dataframe['pre_buy_switch'].cumsum()).cumcount()

            # Create integer positions of each row
            idx_positions = np.arange(len(dataframe))
            # "shift" those integer positions by the amount in shift col
            shifted_idx_positions = idx_positions - dataframe["barssince_last_buy"]
            # get the label based index from our DatetimeIndex
            shifted_loc_index = dataframe.index[shifted_idx_positions]
            # Retrieve the "shifted" values and assign them as a new column
            dataframe["close_5m_last_buy"] = dataframe.loc[shifted_loc_index, "close_5m"].values

            dataframe.loc[:, 'close_lower'] = dataframe.loc[:, 'close'].expanding().apply(get_local_min)
            dataframe['close_lower'] = np.where(dataframe['close_lower'].isna() == True, dataframe['close'], dataframe['close_lower'])
            dataframe['close_lower_offset'] = dataframe['close_lower'] * (1 + self.trailing_buy_offset)
            dataframe['trailing_buy_order_uplimit'] = np.where(dataframe['barssince_last_buy'] < 20, pd.DataFrame([dataframe['close_5m_last_buy'], dataframe['close_lower_offset']]).min(), np.nan)

            dataframe.loc[
                (dataframe['barssince_last_buy'] < 20) &  # must buy within last 20 candles after signal
                (dataframe['close'] > dataframe['trailing_buy_order_uplimit'])
                , 'trailing_buy'] = 1

            dataframe['trailing_buy_count'] = dataframe['trailing_buy'].rolling(20).sum()

            dataframe.log[
                (dataframe['trailing_buy'] == 1) &
                (dataframe['trailing_buy_count'] == 1)
                , 'buy'] = 1
        else:  # No buy trailing
            dataframe.loc[
                (dataframe['pre_buy'] == 1)
                , 'buy'] = 1
        return dataframe

    def get_current_price(self, pair: str) -> float:
        ticker = self.dp.ticker(pair)
        current_price = ticker['last']
        return current_price
