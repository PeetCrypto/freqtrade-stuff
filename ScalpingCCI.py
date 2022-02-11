# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa
from technical.util import resample_to_interval, resampled_merge
import pandas as pd


class ScalpingCCI(IStrategy):
    """
        this strategy is based around the idea of generating a lot of potentatils buys and make tiny profits on each trade

        we recommend to have at least 60 parallel trades at any time to cover non avoidable losses.

        Recommended is to only sell based on ROI for this strategy
    """



    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.02,
        "10": 0.05,
        "20": 0.04,
        "60": 0.3,
        "120": 0.2
    }
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # should not be below 3% loss

    stoploss = -0.04
    # Optimal ticker interval for the strategy
    # the shorter the better
    ticker_interval = '15'

    def get_ticker_indicator(self):
        if 'm' in self.ticker_interval:
            return [int(s) for s in self.ticker_interval.split('m') if s.isdigit()][0]
        elif 'h' in self.ticker_interval:
            return [int(s) for s in self.ticker_interval.split('h') if s.isdigit()][0]*60
        return int(self.ticker_interval)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma'] = ta.EMA(dataframe, timeperiod=20, price='close')

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe_daily = resample_to_interval(dataframe, self.get_ticker_indicator() * 96)

        #TC = (Pivotr - BC) + Pivot
        #Pivot = (high + low + Close) /3
        #BC = (high + low) / 2
        dataframe_daily['high_daily'] = dataframe_daily['high']
        dataframe_daily['low_daily'] = dataframe_daily['low']
        dataframe_daily['pivot'] = (dataframe_daily['high'] + dataframe_daily['low'] + dataframe_daily['close']) / 3
        dataframe_daily['bc'] = (dataframe_daily['high'] + dataframe_daily['low'] ) / 2
        dataframe_daily['tc'] = (dataframe_daily['pivot'] - dataframe_daily['bc']) / 2

        dataframe = resampled_merge(dataframe, dataframe_daily)

        dataframe.fillna(method='ffill', inplace=True)

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        # print("dataframe daily pivot", dataframe['pivot'])
        # print("dataframe daily bc", dataframe['bc'])
        # print("dataframe daily tc", dataframe['tc'])

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['sma']) &
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
                (dataframe['close'] > dataframe['pivot']) &
                (dataframe['close'] > dataframe['bc']) &
                (dataframe['close'] > dataframe['tc']) &
                (
                   (dataframe['pivot'] > dataframe['pivot'].shift(97)) &
                    (dataframe['bc'] > dataframe['bc'].shift(97)) &
                    (dataframe['close'] > dataframe['tc'].shift(97))
                )
            ),
            'buy'] = 1
        return dataframe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['open'] >= (0.98 * dataframe['high_daily']))
            ) |
            (
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))
            ),
            'sell'] = 1
        return dataframe
