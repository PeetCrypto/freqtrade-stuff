# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade

# supertrend
# ref: https://www.tradingfuel.com/supertrend-indicator-formula-and-calculation/
# manipulates the existing dataframe and this is almost certainly bad
def supertrend(dataframe: DataFrame, multiplier=3) -> DataFrame:
    average_price = (dataframe['high'] + dataframe['low']) / 2

    basic_upperband = (average_price + (multiplier * dataframe['atr']))
    basic_lowerband = (average_price - (multiplier * dataframe['atr']))

    # Final Upper Band   
    if not 'st_final_upperband' in dataframe.columns:
        dataframe['st_final_upperband'] = 0     
    else:  
        if (basic_upperband < dataframe['st_final_upperband'].shift(1)) or (dataframe['close'].shift(1) > dataframe['st_final_upperband'].shift(1)):
            dataframe['st_final_upperband'] = basic_upperband
        else:
            dataframe['st_final_upperband'] = dataframe['st_final_upperband'].shift(1)

    # Final Lower Band
    if not 'st_final_lowerband' in dataframe.columns:
        dataframe['st_final_lowerband'] = 0     
    else:  
        if (basic_lowerband > dataframe['st_final_lowerband'].shift(1)) or (dataframe['close'].shift(1) < dataframe['st_final_lowerband'].shift(1)):
            dataframe['st_final_lowerband'] = basic_lowerband   
        else:
            dataframe['st_final_lowerband'] = dataframe['st_final_lowerband'].shift(1)

    # Supertrend
    if not 'supertrend' in dataframe.columns:
        dataframe['supertrend'] = 0
    elif (dataframe['supertrend'].shift(1) == dataframe['st_final_upperband'].shift(1)) and (dataframe['close'] <= dataframe['st_final_upperband']):
        dataframe['supertrend'] = dataframe['st_final_upperband']
    
    elif (dataframe['supertrend'].shift(1) == dataframe['st_final_upperband'].shift(1)) and (dataframe['close'] > dataframe['st_final_upperband']):
        dataframe['supertrend'] = dataframe['st_final_lowerband']
    
    elif (dataframe['supertrend'].shift(1) == dataframe['st_final_lowerband'].shift(1)) and (dataframe['close'] >= dataframe['st_final_lowerband']):
        dataframe['supertrend'] = dataframe['st_final_lowerband']
    
    elif (dataframe['supertrend'].shift(1) == dataframe['st_final_lowerband'].shift(1)) and (dataframe['close'] < dataframe['st_final_lowerband']):
        dataframe['supertrend'] = dataframe['st_final_upperband']     

    return dataframe


class Supertrend(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """
    # Buy hyperspace params:
    buy_params = {
        'buy-method': 'both', # sar, supertrend, both
        'buy-price': 'ohlc4' # open, close, hl2, hlc3, ohlc4
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-method': 'both', # sar, supertrend, both
        'sell-price': 'ohlc4' # open, close, hl2, hlc3, ohlc4
    }

    # ROI table:
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0
    }

    stoploss = -1

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.02
    ignore_roi_if_buy_signal = True

    process_only_new_candles = False

    startup_candle_count: int = 10

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ohlc4'] = ta.AVGPRICE(dataframe)
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=10)
        dataframe['sar'] = ta.SAR(dataframe)

        dataframe = supertrend(dataframe, multiplier=1)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        conditions = []

        if params['buy-method'] == 'supertrend' or params['buy-method'] == 'both':
            conditions.append(dataframe[params['buy-price']] > dataframe['supertrend'])
        
        if params['buy-method'] == 'sar' or params['buy-method'] == 'both':
            conditions.append(dataframe[params['buy-price']] > dataframe['sar'])

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []

        if params['sell-method'] == 'supertrend' or params['sell-method'] == 'both':
            conditions.append(dataframe[params['sell-price']] < dataframe['supertrend'])
        
        if params['sell-method'] == 'sar' or params['sell-method'] == 'both':
            conditions.append(dataframe[params['sell-price']] < dataframe['sar'])

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
    
    """
    Additional buy/sell timeout override if price drifts
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > order['price'] * 1.01:
            return True
        return False


    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel sell order if price is more than 1% below the order.
        if current_price < order['price'] * 0.99:
            return True
        return False