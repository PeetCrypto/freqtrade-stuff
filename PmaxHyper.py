# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List
from technical.indicators import PMAX, zema
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa
import multiprocessing
import time
import sys
import threading
from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib


class PmaxHyper(IHyperOpt):
    """
    This is a sample hyperopt to inspire you.
    Feel free to customize it.

    More information in the documentation: https://www.freqtrade.io/en/latest/hyperopt/

    You should:
    - Rename the class name to some unique name.
    - Add any methods you want to build your hyperopt.
    - Add any lib you need to build your hyperopt.

    You must keep:
    - The prototypes for the methods: populate_indicators, indicator_space, buy_strategy_generator.

    The methods roi_space, generate_roi_table and stoploss_space are not required
    and are provided by default.
    However, you may override them if you need the
    'roi' and the 'stoploss' spaces that differ from the defaults offered by Freqtrade.

    This sample illustrates how to override these methods.
    """

    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        This method can also be loaded from the strategy, if it doesn't exist in the hyperopt class.
        """
        n=0
        length = 10
        dframe = dataframe.copy()
        default = dataframe.copy()
        jobs = []
        tuples = []

        def thr(default, length, MAtype, multiplier, src_val, ns):
            DAT = PMAX(default, period=length, multiplier=multiplier, length=length, MAtype=MAtype,
                       src=src_val)
            xdata = ns.df
            xdata[list(DAT.keys())[-1]] = DAT[list(DAT.keys())[-1]]
            ns.df = xdata

        mgr = multiprocessing.Manager()
        ns = mgr.Namespace()
        ns.df = dataframe
        print(dataframe.head())
        def wrapper(mpobject):
            mpobject.start()
            mpobject.join()
        for length in range(2, 200):
            for MAtype in range(1, 8):
                for multiplier in range(1, 30):
                    for src_val in range(1, 3):
                        p = multiprocessing.Process(target=thr, args=(default, length, MAtype, multiplier, src_val, ns))
                        #                        p.start()
                        #                        print(p.exitcode)
                        #                        thr(default, length, MAtype, multiplier, src_val, ns)
                        #                        jobs.append(p)
                        #                        print(ns.df)
                        #                        print(dataframe)
                        jobs.append(p)

        #                        tuples.append(tuple([length, MAtype, multiplier, src_val, default]))
#        pool = multiprocessing.Pool(processes=16)
#        pool.map_async(wrapper, jobs)
#        pool.close()
#        pool.join()
        for process in jobs:
            th = threading.Thread(target=wrapper, args=(process,))
            th.start()
            n=n+1
            if n==512:
                th.join()
                n=0
                print(sys.getsizeof(ns.df))
        dataframe = ns.df
        for i in range(2, 200):
            EMA = ''
            EMA = 'EMA' + str(i)
            dataframe[EMA] = zema(dataframe, period=length)
        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by hyperopt
        """

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            EMA = "EMA" + str(params['length'])
            pmax = "pm_" + str(params['length']) + "_" + str(params['multiplier']) + "_" + str(
                params['length']) + "_" + str(params['MAtype'])
            dataframe.loc[
                (
                    (qtpylib.crossed_above(dataframe[EMA], dataframe[pmax]))
                ),
                'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching strategy parameters
        """
        return [
            Integer(1, 30, name='multiplier'),
            Integer(1, 200, name='length'),
            Integer(1, 3, name='src_val'),
            Integer(1, 8, name='MAtype')
        ]

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by hyperopt
        """

        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            EMA = "EMA" + str(params['length'])
            pmax = "pm_" + str(params['length']) + "_" + str(params['multiplier']) + "_" + str(
                params['length']) + "_" + str(params['MAtype'])
            dataframe.loc[
                (
                    (qtpylib.crossed_below(dataframe[EMA], dataframe[pmax]))
                ),
                'sell'] = 1
            return dataframe

        return populate_sell_trend

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters
        """
        return [
            Integer(1, 30, name='multiplier'),
            Integer(1, 200, name='length'),
            Integer(1, 3, name='src_val'),
            Integer(1, 8, name='MAtype')
        ]
