import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from technical.indicators import PMAX, zema
from typing import Dict, List
from functools import reduce
from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Pmax(IStrategy):
    stoploss = -1
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        atrperiod = 10
        multiplier = 3
        matype = 9
        malength = 10
        source = 2

        dataframe['ZLEMA'] = zema(dataframe, period=malength)
        dataframe = PMAX(dataframe, atrperiod=atrperiod, multiplier=multiplier, malength=malength, matype=matype, source=source)
        # print(dataframe.keys)
        return dataframe
    
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ZLEMA'], dataframe['pm_40_3_40_9']))
            ),
            'buy'] = 1
    
        return dataframe
    
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ZLEMA'], dataframe['pm_40_3_40_9']))
            ),
            'sell'] = 1
    
        return dataframe
