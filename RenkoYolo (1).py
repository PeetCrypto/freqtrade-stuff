import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, DatetimeIndex, merge, Series


class RenkoYolo(IStrategy):

    # Buy hyperspace params:
    buy_params = {

    }

    # Sell hyperspace params:
    sell_params = {

    }

    # ROI table:
    minimal_roi = {
        "0": 0.03,
        "7": 0.02,
        "33": 0.01,
        "71": 0.005
    }

    # Stoploss:
    stoploss = -100

    # Trailing stop:
    trailing_stop = False
    
    timeframe = '15m'

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=90) #90
        aroon = ta.AROON(dataframe, timeperiod=60) #60

        dataframe['aroon-down'] = aroon['aroondown'] 
        dataframe['aroon-up'] = aroon['aroonup']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            ( 
                (dataframe['adx'] > params['adx']) &
                (dataframe['aroon-up'] > params['aroon-up']) &
                (dataframe['aroon-down'] < params['aroon-down']) &
                (dataframe['volume'] > 0)
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe['sell'] = 0
        return dataframe