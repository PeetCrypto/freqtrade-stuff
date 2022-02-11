# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List
from freqtrade.strategy import merge_informative_pair

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib

informative_timeframe = '1h'

bb_arr_bin = [i for i in range(20, 80 + 1, 5)]
bb_arr_cluc = [i for i in range(10, 50 + 1, 5)]
ema_slow_arr = [i for i in range(30, 80 + 1, 5)]
volume_mean_slow_arr = [i for i in range(10, 50 + 1, 5)]

volume_mean_multiplier_arr = [i for i in range(10, 30 + 1, 5)]


class BinClucHyperOpt(IHyperOpt):
    """
    Hyperopt file for optimizing BinHV45Strategy.
    Uses ranges to find best parameter combination for bbdelta, closedelta and tail
    of the buy strategy.

    Sell strategy is ignored, because it's ignored in BinHV45Strategy as well.
    This strategy therefor works without explicit sell signal therefor hyperopting
    for 'roi' is recommend as well

    Also, this is just ONE way to optimize this strategy - others might also include
    disabling certain conditions completely. This file is just a starting point, feel free
    to improve and PR.
    """
    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:

        typical_price = qtpylib.typical_price(dataframe)

        for i in bb_arr_bin:
            bollinger = qtpylib.bollinger_bands(typical_price, window=i, stds=2)
            mid = bollinger['mid']
            lower = bollinger['lower']
            dataframe[f'mid_{i}'] = np.nan_to_num(mid)
            dataframe[f'lower_{i}'] = np.nan_to_num(lower)
            dataframe[f'bbdelta_{i}'] = (dataframe[f'mid_{i}'] - dataframe[f'lower_{i}']).abs()

        for i in bb_arr_cluc:
            # strategy ClucMay72018
            bollinger = qtpylib.bollinger_bands(typical_price, window=i, stds=2)
            dataframe[f'bb_lowerband_{i}'] = bollinger['lower']
            dataframe[f'bb_middleband_{i}'] = bollinger['mid']

        for i in ema_slow_arr:
            dataframe[f'ema_slow_{i}'] = ta.EMA(dataframe, timeperiod=i)

        for i in volume_mean_slow_arr:
            dataframe[f'volume_mean_slow_{i}'] = dataframe['volume'].rolling(window=i).mean()

        dataframe['pricedelta'] = (dataframe['open'] - dataframe['close']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use.
            """

            dataframe.loc[
                (
                    (
                        dataframe[f'lower_{params["bband_size_bin"]}'].shift().gt(0) &
                        dataframe[f'bbdelta_{params["bband_size_bin"]}'].gt(dataframe['close'] * params['bbdelta_multiplier']) &
                        dataframe['closedelta'].gt(dataframe['close'] * params['closedelta_multiplier']) &
                        dataframe['tail'].lt(dataframe[f'bbdelta_{params["bband_size_bin"]}'] * params['tail_multiplier']) &
                        dataframe['close'].lt(dataframe[f'lower_{params["bband_size_bin"]}'].shift()) &
                        dataframe['close'].le(dataframe['close'].shift())
                    )
                    |
                    (  # strategy ClucMay72018
                        (dataframe['close'] < dataframe[f'ema_slow_{params["ema_slow_size"]}']) &
                        (dataframe['close'] < params['bb_lowerband_multiplier'] * dataframe[f'bb_lowerband_{params["bband_size_cluc_buy"]}']) &
                        (dataframe['volume'] < (dataframe[f'volume_mean_slow_{params["volume_mean_slow_size"]}'].shift(1) * params['volume_mean_multiplier_size']))
                    )
                 )

                ,
                'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching buy strategy parameters.
        """
        return [
            Real(0.005, 0.013, name='bbdelta_multiplier'),
            Real(0.0125, 0.0225, name='closedelta_multiplier'),
            Real(0.19, 0.31, name='tail_multiplier'),
            Categorical(bb_arr_bin, name='bband_size_bin'),
            Categorical(bb_arr_cluc, name='bband_size_cluc_buy'),
            Categorical(ema_slow_arr, name='ema_slow_size'),
            Categorical(volume_mean_slow_arr, name='volume_mean_slow_size'),
            Categorical(volume_mean_multiplier_arr, name='volume_mean_multiplier_size'),
            Real(0.965, 0.995, name='bb_lowerband_multiplier'),
        ]

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by Hyperopt.
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            no sell signal
            """
            dataframe.loc[
                (dataframe['close'] > dataframe[f'bb_middleband_{params["bband_size_cluc_sell"]}']),
                'sell'
            ] = 1
            return dataframe

        return populate_sell_trend

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters.
        """
        return [
            Categorical(bb_arr_cluc, name='bband_size_cluc_sell'),
        ]
