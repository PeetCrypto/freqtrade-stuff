import copy
import logging
import pathlib
import rapidjson
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes, DecimalParameter, IntParameter, CategoricalParameter
from freqtrade.exchange import timeframe_to_prev_date
from pandas import DataFrame, Series, concat
from functools import reduce
import math
from typing import Dict
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from technical.util import resample_to_interval, resampled_merge
from technical.indicators import zema, VIDYA, ichimoku
import time

log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)


try:
    import pandas_ta as pta
except ImportError:
    log.error(
        "IMPORTANT - please install the pandas_ta python module which is needed for this strategy. "
        "If you're running Docker, add RUN pip install pandas_ta to your Dockerfile, otherwise run: "
        "pip install pandas_ta"
    )
else:
    log.info("pandas_ta successfully imported")



###########################################################################################################
##               Pump protection implementation                                                          ##
##                                                                                                       ##
##   This is an exemple of how to add max_pump_detect_price_15m function in your strategies.             ##
##   This is not a functionnal strategy, don't use it live.                                              ##
##   Parameters needs to be tuned, pump_**** values are hyperoptable but I have never seen great         ##
##   results with hyperopt. The best is to plot some coins pumped on specific timeframe and deduced      ##
##   the value you need to prevent your strat to buy.                                                    ##
##   If you find any way to improve it please keep me updated :)                                         ##
##                                                                                                       ##
##   The exemple here use 15m timeframe for pump detection, 1h and 30m show also good results            ##
##                                                                                                       ##
##   There are 2 ways to use it :                                                                        ##
##   - add a line to each of your buy conditions to check if the coins was resently pump and didn't      ##
##     recovered yet                                                                                     ##
##                                                                                                       ##
##   - add a confirm_trade_entry condition (easier for a strat like NFI with 50 buy conditions           ##
###########################################################################################################

def max_pump_detect_price_15m(dataframe, period=14, pause = 288 ):
    df = dataframe.copy()
    df['size'] = df['high'] - df['low']
    cumulativeup = 0
    countup = 0
    cumulativedown = 0
    countdown = 0
    for i in range(period):

        cumulativeup = cumulativeup + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 1, 0)
        cumulativedown = cumulativedown + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 0, 1)
            
    flow_price = cumulativeup - cumulativedown
    flow_price_normalized = flow_price / (df['volume'].rolling(499).mean() * (df['high']-df['low']).rolling(499).mean())
    max_flow_price = flow_price_normalized.rolling(pause).max()
    
    return max_flow_price
    
def flow_price_15m(dataframe, period=14, pause = 288 ):
    df = dataframe.copy()
    df['size'] = df['high'] - df['low']
    cumulativeup = 0
    countup = 0
    cumulativedown = 0
    countdown = 0
    for i in range(period):

        cumulativeup = cumulativeup + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 1, 0)
        cumulativedown = cumulativedown + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 0, 1)
            
    flow_price = cumulativeup - cumulativedown
    flow_price_normalized = flow_price / (df['volume'].rolling(499).mean() * (df['high']-df['low']).rolling(499).mean())
    
    return flow_price_normalized
    
    
class YourStratName(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "pump_limit": 1000,
        "pump_pause_duration": 192,
        "pump_period": 14,
        "pump_recorver_price": 1.1,
    }
    # Pump protection
    pump_period = IntParameter(
        5, 24, default=buy_params['pump_period'], space='buy', optimize=False)
    pump_limit = IntParameter(
        100,10000, default=buy_params['pump_limit'], space='buy', optimize=True)
    pump_recorver_price = DecimalParameter(
        1.0, 1.3, default=buy_params['pump_recorver_price'], space='buy', optimize=True)
    pump_pause_duration = IntParameter(
        6, 500, default=buy_params['pump_pause_duration'], space='buy', optimize=True)


    # Optimal timeframe for the strategy.
    yourtimeframe = '5m'
    inf_15m = '15m' #use for pump detection

    # Number of candles the strategy requires before producing valid signals, needed for normalized the pump detection (using average candle size and volume on large period of time)
    startup_candle_count: int = 499
    
########## Your Strat parameters, has no effect on pump detection ##########

    # ROI table:
    minimal_roi = {
        "0": 10,
    }

    stoploss = -0.50

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    use_custom_stoploss = False    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'trailing_stop_loss': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    #############################################################

    plot_config = {
        'main_plot': {
            'weekly_close_avg_offset' : {'color': 'red'},
        },
        'subplots': {
            "Pump detectors": {
                'max_flow_price_15m': {'color': 'red'},
                'flow_price_15m': {'color': 'blue'}
            }
        }
    }

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_1h)

        #Weekly average close price        
        informative_1h['weekly_close_avg'] = informative_1h['close'].rolling(168).mean()

        return informative_1h

    # Informative indicator for pump detection 

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        
        informative_15m['max_flow_price'] = max_pump_detect_price_15m(informative_15m, period=self.pump_period.value, pause=self.pump_pause_duration.value)
        informative_15m['flow_price'] = flow_price_15m(informative_15m, period=self.pump_period.value, pause=self.pump_pause_duration.value)
        
        return informative_15m

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        informative_pairs.extend([(pair, self.info_timeframe_1h) for pair in pairs])

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.info_timeframe_1h)

        return dataframe
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #Import 1h indicators
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [(s + "_" + self.inf_1h) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        #Import 15m indicators    
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)
        drop_columns = [(s + "_" + self.inf_15m) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)
        
        
        #Pump protection
        dataframe['weekly_close_avg_offset'] = self.pump_recorver_price.value * dataframe['weekly_close_avg_1h']
        dataframe['price_test'] = dataframe['close'] > dataframe['weekly_close_avg_offset']
        dataframe['pump_price_test'] = dataframe['max_flow_price_15m'] > self.pump_limit.value

        # Check if a pump uccured during pump_pause_duration and coin didn't recovered its pre pump value
        dataframe['pump_dump_alert'] = dataframe['price_test'] & dataframe['pump_price_test']
        dataframe['buy_ok'] = np.where(dataframe['pump_dump_alert'], False, True)
      
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (Condition_11 & (Condition_12 | Conditon_13)) & # you first buy condition
                dataframe['buy_ok']
            )
        )

        conditions.append(
            (
                (Condition_21 & (Condition_22 | Conditon_23) & # you second buy condition
                dataframe['buy_ok']
            )
        )
        # etc & etc
        
        conditions.append(
            (
                (Condition_n1 & (Condition_n2 | Conditon_n3) & # you n buy condition
                dataframe['buy_ok']
            )
        )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                #Your sell conditions here
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1


        return dataframe


### Confirm trade entry is to use only if the dataframe['buy_ok'] test was not added in populate_buy_trend (in case of too many buy conditions like for NFI)
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        if current_candle['buy_ok'] :
            return True
        else:
            return False
        
