# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import merge_informative_pair
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from cachetools import TTLCache
import technical.indicators as ftt
import time

class LongShortRangeTradingMachetesV1(IStrategy):

    INTERFACE_VERSION = 2

    TF_STAND_BY_WAITING_FOR_MARKET_CONDITION_STATE = 0
    TF_ENTERED_MARKET_CONDITION_WAITING_FOR_CONFIRMATION_STATE = 1
    TF_MARKET_CONDITION_CONFIRMED_WAITING_FOR_ENTRY_SIGNAL_STATE = 2
    TF_ENTRY_SIGNAL_FOUND_STATE = 3
    TF_ENTRY_SIGNAL_FOUND_WAITING_FOR_EXIT_SIGNAL_STATE = 4

    custom_trade_flow_info = {}
    custom_trade_info = {}
    custom_current_price_cache = TTLCache(maxsize=100, ttl=300)

    # ROI table:
    minimal_roi = {
        "0": 1
    }

    # Stoploss:
    stoploss = -0.1

    # Trailing stop:
    trailing_stop = False
    #trailing_stop_positive = 0.097
    #trailing_stop_positive_offset = 0.161
    #trailing_only_offset_is_reached = True

    timeframe = '1m'
    timeframe_medium = '15m'
    timeframe_long = '5m'
    candels_per_timeframe_medium = 4
    candels_per_timeframe_long = 16
    process_only_new_candles = False
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    startup_candle_count: int = 500

    use_dynamic_roi = True
    use_custom_stoploss = True

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            f'base_open_{timeframe}': {},
            f'base_high_{timeframe}': {},
            f'base_low_{timeframe}': {},
            f'base_close_{timeframe}': {},

            f'base_overbought_price_{timeframe}': {},
            f'base_oversold_price_{timeframe}': {},

            f'base_overbought_price_{timeframe_medium}': {},
            f'base_oversold_price_{timeframe_medium}': {},

            f'base_overbought_price_{timeframe_long}': {},
            f'base_oversold_price_{timeframe_long}': {},
        },
        'subplots': {
            "cci": {
                f'base_cci_{timeframe}': {'color': 'yellow'},
                f'base_cci_{timeframe_medium}': {'color': 'yellow'},
                f'base_cci_{timeframe_long}': {'color': 'yellow'},
                f'base_cci_overbought_value_{timeframe}': {
                    'color': 'rgba(35, 138, 29, 0.75)',
                    'fill_to': f'base_cci_oversold_value_{timeframe}',
                    'fill_label': 'cci',
                    'fill_color': 'rgba(51, 255, 117, 0.2)',
                }
            },
            "signals": {
                'has_entered_market_condition': {'color': 'red'},
                'is_in_market_condition': {'color': 'red'},
                'has_confirmation': {'color': 'yellow'},
                'has_entry_signal': {'color': 'green'},
                'entry_signal': {'color': 'blue'},
                'has_exit_signal': {'color': 'green'},
                'exit_signal': {'color': 'blue'}
            }
        }
    }


#
# Hyperopt params
#

    # Tradeflow

    # Dynamic ROI
    droi_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any'], default='any', space='sell', optimize=True)
    droi_pullback = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    droi_pullback_amount = DecimalParameter(0.005, 0.02, default=0.005, space='sell', optimize=True)
    droi_pullback_respect_table = CategoricalParameter([True, False], default=False, space='sell', optimize=True)

    # Custom Stoploss
    cstp_threshold = DecimalParameter(-0.05, 0, default=-0.03, space='sell', optimize=True)
    cstp_bail_how = CategoricalParameter(['roc', 'time', 'any'], default='roc', space='sell', optimize=True)
    cstp_bail_roc = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', optimize=True)
    cstp_bail_time = IntParameter(720, 1440, default=720, space='sell', optimize=True)
    cstp_trailing_stop_positive_offset = DecimalParameter(0.005, 0.06,default=0.01,space='sell', optimize=True)
    cstp_trailing_stop_profit_devider = IntParameter(2, 4,default=2,space='sell', optimize=True)
    cstp_trailing_max_stoploss = DecimalParameter(0.02, 0.08,default=0.02,space='sell', optimize=True)
    cstp_trailing_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)


#
# Events
#

    def on_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        base_pair_name = self.get_base_pair_name(metadata['pair'])
        dataframe = self.get_base_pair_dataframe(dataframe, base_pair_name, self.timeframe_long)
        #dataframe = self.get_base_pair_dataframe(dataframe, base_pair_name, self.timeframe_medium)
        dataframe = self.get_base_pair_dataframe(dataframe, base_pair_name, self.timeframe)

        dataframe = self.get_indicators_custom_stoploss(dataframe)
        self.setup_custom_trade_info(dataframe, metadata)

        return dataframe


    def on_populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if "UP" in metadata['pair'] or "DOWN" in metadata['pair']:
            if "UP" in metadata['pair']:
                dataframe = self.calc_indicator_signals_long(dataframe, metadata)
            elif "DOWN" in metadata['pair']:
                dataframe = self.calc_indicator_signals_short(dataframe, metadata)

            dataframe = self.calc_trade_flow(dataframe, metadata)

        else:
            #print('This pair (' + metadata['pair'] + '} is not a LT.')
            dataframe['entry_signal'] = 0

        return dataframe


    def on_populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if "UP" in metadata['pair'] or "DOWN" in metadata['pair']:
            if "UP" in metadata['pair']:
                dataframe = self.calc_indicator_signals_long(dataframe, metadata)
            elif "DOWN" in metadata['pair']:
                dataframe = self.calc_indicator_signals_short(dataframe, metadata)

            dataframe = self.calc_trade_flow(dataframe, metadata)
        else:
            #print('This pair (' + metadata['pair'] + '} is not a LT.')
            dataframe['exit_signal'] = 0

        return dataframe


#
# IStrategy
#

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = []
        informative_pairs += [(pair, self.timeframe) for pair in pairs]
        informative_pairs += [(pair, self.timeframe_medium) for pair in pairs]
        informative_pairs += [(pair, self.timeframe_long) for pair in pairs]

        return informative_pairs


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if not self.dp:
            return dataframe

        dataframe = self.on_populate_indicators(dataframe, metadata)

        #dataframe.info(verbose=True)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.on_populate_buy_trend(dataframe, metadata)

        dataframe.loc[
            (
                (dataframe['entry_signal'] == 1)
            ),
            'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.on_populate_sell_trend(dataframe, metadata)

        dataframe.loc[
            (
                (dataframe['exit_signal'] == 1)
            ),
            'sell'] = 1

        return dataframe


    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        #print(pair,trade.open_date,trade.is_open,current_profit)
        return False


#
# Indicators - Data
#

    def get_indicators_custom_stoploss(self, dataframe):

        def RMI(dataframe, *, length=20, mom=5):
            """
            Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
            """
            df = dataframe.copy()

            df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
            df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

            df.fillna(0, inplace=True)

            df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
            df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

            df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

            return df["RMI"]

        def SSLChannels_ATR(dataframe, length=7):
            """
            SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
            Credit to @JimmyNixx for python
            """
            df = dataframe.copy()

            df['ATR'] = ta.ATR(df, timeperiod=14)
            df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
            df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
            df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
            df['hlv'] = df['hlv'].ffill()
            df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
            df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

            return df['sslDown'], df['sslUp']

        def SROC(dataframe, roclen=21, emalen=13, smooth=21):
            df = dataframe.copy()

            roc = ta.ROC(df, timeperiod=roclen)
            ema = ta.EMA(df, timeperiod=emalen)
            sroc = ta.ROC(ema, timeperiod=smooth)

            return sroc

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)
        dataframe['rmi'] = RMI(dataframe, length=24, mom=5)
        ssldown, sslup = SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown,'up','down')
        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(),1,0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3,1,0)
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['close'].shift(),1,0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3,1,0)

        return dataframe


    def get_indicators(self, dataframe):

        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['cci_overbought_value'] = 100
        dataframe['cci_oversold_value'] = -100

        dataframe['overbought_price'] = (
            (qtpylib.crossed_below(dataframe['cci'], 100))
        ).fillna(0).astype('int') * dataframe['high']
        dataframe['overbought_price'] = dataframe['overbought_price'].replace(to_replace=0, method='ffill')

        dataframe['oversold_price'] = (
            (qtpylib.crossed_above(dataframe['cci'], -100))
        ).fillna(0).astype('int') * dataframe['low']
        dataframe['oversold_price'] = dataframe['oversold_price'].replace(to_replace=0, method='ffill')

        return dataframe


    def get_base_pair_name(self, pair_name):

        #extract target pair
        pair_name_parts = pair_name.split('/')
        target_pair_name = pair_name_parts[0]
        #remove up or down
        target_pair_suffix = 'DOWN' if ("DOWN" in pair_name) else 'UP'
        target_base_name = target_pair_name.replace(target_pair_suffix, "")
        #add usd stake
        base_pair_name = target_base_name + '/USDT'

        return base_pair_name


    def get_base_pair_dataframe(self,dataframe, base_pair_name, timeframe_str, smooth_list = None, candels_per_timeframe = 2):

        base_pair_dataframe = self.dp.get_pair_dataframe(base_pair_name, timeframe_str)
        base_pair_dataframe = self.get_indicators(base_pair_dataframe)
        ignore_columns = ['date']
        base_pair_dataframe.rename(columns=lambda s: "base_" + s if (not s in ignore_columns) else s, inplace=True)
        dataframe = merge_informative_pair(dataframe, base_pair_dataframe, self.timeframe, timeframe_str, ffill=True)

        if smooth_list != None:
            for indicator_key in smooth_list:
                dataframe[f'base_{indicator_key}_{timeframe_str}'] = ta.SMA(dataframe[f'base_{indicator_key}_{timeframe_str}'], timeperiod=candels_per_timeframe)

        return dataframe


#
# Indicators - Logic
#

    def calc_indicator_signals_long(self, dataframe, metadata):

        dataframe['has_entered_market_condition'] = (
            (qtpylib.crossed_above(dataframe[f'base_close_{self.timeframe}'], dataframe[f'base_oversold_price_{self.timeframe_long}']))
        ).fillna(0).astype('int')

        dataframe['is_in_market_condition'] = (
            (dataframe[f'base_close_{self.timeframe}'] > dataframe[f'base_oversold_price_{self.timeframe_long}'])
        ).fillna(0).astype('int')

        dataframe['has_confirmation'] = (
            (qtpylib.crossed_above(dataframe[f'base_cci_{self.timeframe_long}'], 0))
        ).fillna(0).astype('int')

        dataframe['has_entry_signal'] = (
            (dataframe[f'base_cci_{self.timeframe_long}'] >= 0)
        ).fillna(0).astype('int')

        dataframe['has_exit_signal'] = (
            (qtpylib.crossed_above(dataframe[f'base_close_{self.timeframe}'], dataframe[f'base_overbought_price_{self.timeframe_long}']))
        ).fillna(0).astype('int')

        return dataframe


    def calc_indicator_signals_short(self, dataframe, metadata):

        dataframe['has_entered_market_condition'] = (
            (qtpylib.crossed_below(dataframe[f'base_close_{self.timeframe}'], dataframe[f'base_overbought_price_{self.timeframe_long}']))
        ).fillna(0).astype('int')

        dataframe['is_in_market_condition'] = (
            (dataframe[f'base_close_{self.timeframe}'] < dataframe[f'base_overbought_price_{self.timeframe_long}'])
        ).fillna(0).astype('int')

        dataframe['has_confirmation'] = (
            (qtpylib.crossed_below(dataframe[f'base_cci_{self.timeframe_long}'], 0))
        ).fillna(0).astype('int')

        dataframe['has_entry_signal'] = (
            (dataframe[f'base_cci_{self.timeframe_long}'] <= 0)
        ).fillna(0).astype('int')

        dataframe['has_exit_signal'] = (
            (qtpylib.crossed_below(dataframe[f'base_close_{self.timeframe}'], dataframe[f'base_oversold_price_{self.timeframe_long}']))
        ).fillna(0).astype('int')

        return dataframe


    def init_trade_flow_info(self, pair):

        if not pair in self.custom_trade_flow_info:
            self.custom_trade_flow_info[pair] = {}
            self.custom_trade_flow_info[pair]['trade_flow'] = None

        self.set_trade_flow_state(self.TF_STAND_BY_WAITING_FOR_MARKET_CONDITION_STATE, pair)


    def is_trade_flow_state(self, trade_flow_state, pair):
        return self.custom_trade_flow_info[pair]['trade_flow'] == trade_flow_state


    def set_trade_flow_state(self, trade_flow_state, pair):
        self.custom_trade_flow_info[pair]['trade_flow'] = trade_flow_state


    def calc_trade_flow(self, dataframe, metadata):

        dataframe['entry_signal'] = 0
        dataframe['exit_signal'] = 0
        pair = metadata['pair']

        start_time = time.time()

        self.init_trade_flow_info(pair)

        for row_df in zip(dataframe['has_entered_market_condition'],dataframe['is_in_market_condition'],dataframe['has_confirmation'],dataframe['has_entry_signal'],dataframe['has_exit_signal'],dataframe['date']):
            if self.is_trade_flow_state(self.TF_STAND_BY_WAITING_FOR_MARKET_CONDITION_STATE, pair):
                if row_df[0] == 1:
                    self.set_trade_flow_state(self.TF_ENTERED_MARKET_CONDITION_WAITING_FOR_CONFIRMATION_STATE, pair)
                else:
                    continue

            if self.is_trade_flow_state(self.TF_ENTERED_MARKET_CONDITION_WAITING_FOR_CONFIRMATION_STATE, pair):
                if row_df[1] == 1:
                    if row_df[2] == 1:
                        self.set_trade_flow_state(self.TF_MARKET_CONDITION_CONFIRMED_WAITING_FOR_ENTRY_SIGNAL_STATE, pair)
                    else:
                        continue
                else:
                    self.set_trade_flow_state(self.TF_STAND_BY_WAITING_FOR_MARKET_CONDITION_STATE, pair)
                    continue

            if self.is_trade_flow_state(self.TF_MARKET_CONDITION_CONFIRMED_WAITING_FOR_ENTRY_SIGNAL_STATE, pair):
                if row_df[1] == 1:
                    if row_df[2] == 1:
                        if row_df[3] == 1:
                            self.set_trade_flow_state(self.TF_ENTRY_SIGNAL_FOUND_STATE, pair)
                        else:
                            continue
                    else:
                        self.set_trade_flow_state(self.TF_ENTERED_MARKET_CONDITION_WAITING_FOR_CONFIRMATION_STATE, pair)
                        continue
                else:
                    self.set_trade_flow_state(self.TF_STAND_BY_WAITING_FOR_MARKET_CONDITION_STATE, pair)
                    continue

            if self.is_trade_flow_state(self.TF_ENTRY_SIGNAL_FOUND_STATE, pair):
                dataframe.at[dataframe['date'] == row_df[5], 'entry_signal'] = 1
                self.set_trade_flow_state(self.TF_ENTRY_SIGNAL_FOUND_WAITING_FOR_EXIT_SIGNAL_STATE, pair)

            if self.is_trade_flow_state(self.TF_ENTRY_SIGNAL_FOUND_WAITING_FOR_EXIT_SIGNAL_STATE, pair):
                if row_df[4] == 1:
                    self.set_trade_flow_state(self.TF_STAND_BY_WAITING_FOR_MARKET_CONDITION_STATE, pair)
                    dataframe.at[dataframe['date'] == row_df[5], 'exit_signal'] = 1
                else:
                    continue

        print(pair, 'calc_trade_flow', time.time() - start_time)

        return dataframe


#
# Custom stoploss
#

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            sroc = dataframe['sroc'].iat[-1]
        # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
        else:
            sroc = self.custom_trade_info[trade.pair]['sroc'].loc[current_time]['sroc']

        new_stoploss = 1

        if current_profit < self.cstp_threshold.value:
            if self.cstp_bail_how.value == 'roc' or self.cstp_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if (sroc/100) <= self.cstp_bail_roc.value:
                    new_stoploss = 0.001
            if self.cstp_bail_how.value == 'time' or self.cstp_bail_how.value == 'any':
                # Dynamic bailout based on time
                if trade_dur > self.cstp_bail_time.value:
                    new_stoploss = 0.001
        else:
            if self.cstp_trailing_enabled.value == True and current_profit >= self.cstp_trailing_stop_positive_offset.value:
                desired_stoploss = current_profit / self.cstp_trailing_stop_profit_devider.value
                new_stoploss = max(min(desired_stoploss, self.cstp_trailing_max_stoploss.value), 0.025)

        return new_stoploss


#
# Dynamic roi
#

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.use_dynamic_roi:
            _, roi = self.min_roi_reached_dynamic(trade, current_profit, current_time, trade_dur)
        else:
            _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi


    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:

        minimal_roi = self.minimal_roi
        _, table_roi = self.min_roi_reached_entry(trade_dur)

        # see if we have the data we need to do this, otherwise fall back to the standard table
        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            if self.config['runmode'].value in ('live', 'dry_run'):
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                rmi_trend = dataframe['rmi-up-trend'].iat[-1]
                candle_trend = dataframe['candle-up-trend'].iat[-1]
                ssl_dir = dataframe['ssl-dir'].iat[-1]
            # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
            else:
                rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']
                candle_trend = self.custom_trade_info[trade.pair]['candle-up-trend'].loc[current_time]['candle-up-trend']
                ssl_dir = self.custom_trade_info[trade.pair]['ssl-dir'].loc[current_time]['ssl-dir']

            min_roi = table_roi
            max_profit = trade.calc_profit_ratio(trade.max_rate)
            pullback_value = (max_profit - self.droi_pullback_amount.value)
            in_trend = False

            if self.droi_trend_type.value == 'rmi' or self.droi_trend_type.value == 'any':
                if rmi_trend == 1:
                    in_trend = True
            if self.droi_trend_type.value == 'ssl' or self.droi_trend_type.value == 'any':
                if ssl_dir == 'up':
                    in_trend = True
            if self.droi_trend_type.value == 'candle' or self.droi_trend_type.value == 'any':
                if candle_trend == 1:
                    in_trend = True

            # Force the ROI value high if in trend
            if (in_trend == True):
                min_roi = 100
                # If pullback is enabled, allow to sell if a pullback from peak has happened regardless of trend
                if self.droi_pullback.value == True and (current_profit < pullback_value):
                    if self.droi_pullback_respect_table.value == True:
                        min_roi = table_roi
                    else:
                        min_roi = current_profit / 2

        else:
            min_roi = table_roi

        return trade_dur, min_roi


    def get_current_price(self, pair: str, refresh: bool) -> float:
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)
            # Check if cache has been invalidated
            if rate:
                return rate

        ask_strategy = self.config.get('ask_strategy', {})
        if ask_strategy.get('use_order_book', False):
            ob = self.dp.orderbook(pair, 1)
            rate = ob[f"{ask_strategy['price_side']}s"][0][0]
        else:
            ticker = self.dp.ticker(pair)
            rate = ticker['last']

        self.custom_current_price_cache[pair] = rate
        return rate


    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {}
        trade_data['active_trade'] = False

        # active trade stuff only works in live and dry, not backtest
        if self.config['runmode'].value in ('live', 'dry_run'):

            # find out if we have an open trade for this pair
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            # if so, get some information
            if active_trade:
                # get current price and update the min/max rate
                current_rate = self.get_current_price(pair, True)
                """
                freqtrade    | Traceback (most recent call last):
                freqtrade    |   File "/freqtrade/freqtrade/strategy/strategy_wrapper.py", line 17, in wrapper
                freqtrade    |     return f(*args, **kwargs)
                freqtrade    |   File "/freqtrade/freqtrade/strategy/interface.py", line 417, in _analyze_ticker_internal
                freqtrade    |     dataframe = self.analyze_ticker(dataframe, metadata)
                freqtrade    |   File "/freqtrade/freqtrade/strategy/interface.py", line 396, in analyze_ticker
                freqtrade    |     dataframe = self.advise_indicators(dataframe, metadata)
                freqtrade    |   File "/freqtrade/freqtrade/strategy/interface.py", line 763, in advise_indicators
                freqtrade    |     return self.populate_indicators(dataframe, metadata)
                freqtrade    |   File "/freqtrade/user_data/strategies/LongAndShortMachetes.py", line 258, in populate_indicators
                freqtrade    |     dataframe = self.on_populate_indicators(dataframe, metadata)
                freqtrade    |   File "/freqtrade/user_data/strategies/LongAndShortMachetes.py", line 142, in on_populate_indicators
                freqtrade    |     self.setup_custom_trade_info(dataframe, metadata)
                freqtrade    |   File "/freqtrade/user_data/strategies/LongAndShortMachetes.py", line 636, in setup_custom_trade_info
                freqtrade    |     self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
                freqtrade    |   File "/freqtrade/user_data/strategies/LongAndShortMachetes.py", line 629, in populate_trades
                freqtrade    |     active_trade[0].adjust_min_max_rates(current_rate)
                freqtrade    | TypeError: adjust_min_max_rates() missing 1 required positional argument: 'current_price_low'
                freqtrade    | 2021-09-18 11:34:11,851 - freqtrade.strategy.interface - WARNING - Unable to analyze candle (OHLCV) data for pair DOTDOWN/USDT: adjust_min_max_rates() missing 1 required positional argument: 'current_price_low'

                from interface:should_sell
                This function evaluates if one of the conditions required to trigger a sell
                has been reached, which can either be a stop-loss, ROI or sell-signal.
                :param low: Only used during backtesting to simulate stoploss
                :param high: Only used during backtesting, to simulate ROI
                :param force_stoploss: Externally provided stoploss
                :return: True if trade should be sold, False otherwise
                """
                active_trade[0].adjust_min_max_rates(current_rate, current_rate)

        return trade_data


    def setup_custom_trade_info(self, dataframe, metadata):

        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])

        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['roc'] = dataframe[['date', 'roc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['atr'] = dataframe[['date', 'atr']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['sroc'] = dataframe[['date', 'sroc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['ssl-dir'] = dataframe[['date', 'ssl-dir']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend'] = dataframe[['date', 'rmi-up-trend']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['candle-up-trend'] = dataframe[['date', 'candle-up-trend']].copy().set_index('date')
