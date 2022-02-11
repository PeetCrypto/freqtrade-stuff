
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import logging
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --------------------------------
import pandas as pd
import numpy as np
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes
from technical.util import resample_to_interval, resampled_merge

logger = logging.getLogger(__name__)

def pivots_points(dataframe: pd.DataFrame, timeperiod=1, levels=4) -> pd.DataFrame:
    """
    Pivots Points
    https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/
    Formula:
    Pivot = (Previous High + Previous Low + Previous Close)/3
    Resistance #1 = (2 x Pivot) - Previous Low
    Support #1 = (2 x Pivot) - Previous High
    Resistance #2 = (Pivot - Support #1) + Resistance #1
    Support #2 = Pivot - (Resistance #1 - Support #1)
    Resistance #3 = (Pivot - Support #2) + Resistance #2
    Support #3 = Pivot - (Resistance #2 - Support #2)
    ...
    :param dataframe:
    :param timeperiod: Period to compare (in ticker)
    :param levels: Num of support/resistance desired
    :return: dataframe
    """

    data = {}

    low = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index, data=dataframe["low"]), window=timeperiod
    )

    high = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index, data=dataframe["high"]), window=timeperiod
    )

    # Pivot
    data["pivot"] = qtpylib.rolling_mean(series=qtpylib.typical_price(dataframe), window=timeperiod)

    # Resistance #1
    # data["r1"] = (2 * data["pivot"]) - low ... Standard
    # R1 = PP + 0.382 * (HIGHprev - LOWprev) ... fibonacci
    data["r1"] = data['pivot'] + 0.382 * (high - low)

    data["rS1"] = data['pivot'] + 0.0955 * (high - low)


    # Resistance #2
    # data["s1"] = (2 * data["pivot"]) - high ... Standard
    # S1 = PP - 0.382 * (HIGHprev - LOWprev) ... fibonacci
    data["s1"] = data["pivot"] - 0.382 * (high - low)

    # Calculate Resistances and Supports >1
    for i in range(2, levels + 1):
        prev_support = data["s" + str(i - 1)]
        prev_resistance = data["r" + str(i - 1)]

        # Resitance
        data["r" + str(i)] = (data["pivot"] - prev_support) + prev_resistance

        # Support
        data["s" + str(i)] = data["pivot"] - (prev_resistance - prev_support)

    return pd.DataFrame(index=dataframe.index, data=data)


def create_ichimoku(dataframe, conversion_line_period, displacement, base_line_periods, laggin_span):
    ichimoku = ftt.ichimoku(dataframe,
                            conversion_line_period=conversion_line_period,
                            base_line_periods=base_line_periods,
                            laggin_span=laggin_span,
                            displacement=displacement
                            )
    dataframe[f'tenkan_sen_{conversion_line_period}'] = ichimoku['tenkan_sen']
    dataframe[f'kijun_sen_{conversion_line_period}'] = ichimoku['kijun_sen']
    dataframe[f'senkou_a_{conversion_line_period}'] = ichimoku['senkou_span_a']
    dataframe[f'senkou_b_{conversion_line_period}'] = ichimoku['senkou_span_b']


class Miku_PP_v3(IStrategy):
    """
     Miku_PP_v3
     La base de la Estrategia es: Miku_PP_v2 y Miku_1m_5m_CSen44_1_5m
    
    Provando en:
     Miku_1m_5m_CSen444v2_N_1_5
     SymphonIK
    """

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # generate signals from the 1h timeframe
    informative_timeframe = '1d'

    # WARNING: ichimoku is a long indicator, if you remove or use a
    # shorter startup_candle_count your results will be unstable/invalid
    # for up to a week from the start of your backtest or dry/live run
    # (180 candles = 7.5 days)
    startup_candle_count = 444  # MAXIMUM ICHIMOKU

    # NOTE: this strat only uses candle information, so processing between
    # new candles is a waste of resources as nothing will change
    process_only_new_candles = True

    minimal_roi = {
        "0": 10,
    }
    
    plot_config = {
        'main_plot': {
            'pivot_1d': {},
            'rS1_1d': {},
            'r1_1d': {},
            's1_1d': {},
            'senkou_b_88': {},
        },
        'subplots': {
            'MACD': {
                'macd_1h': {'color': 'blue'},
                'macdsignal_1h': {'color': 'orange'},
            },
        }
    }

    # WARNING setting a stoploss for this strategy doesn't make much sense, as it will buy
    # back into the trend at the next available opportunity, unless the trend has ended,
    # in which case it would sell anyway.

    # Stoploss:
    stoploss = -0.10

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe)
                             for pair in pairs]
        if self.dp:
            for pair in pairs:
                informative_pairs += [(pair, "1d")]

        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        """
        # dataframe "1d"
        """

        dataframe1d = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1d")

        # Pivots Points
        pp = pivots_points(dataframe1d)
        dataframe1d['pivot'] = pp['pivot']
        dataframe1d['r1'] = pp['r1']
        dataframe1d['s1'] = pp['s1']
        dataframe1d['rS1'] = pp['rS1']
        # Pivots Points

        dataframe = merge_informative_pair(
            dataframe, dataframe1d, self.timeframe, "1d", ffill=True)

        """
        # dataframe normal
        """
        """
        create_ichimoku(dataframe, conversion_line_period=9, 
                        displacement=26, base_line_periods=26, laggin_span=52)
        """
        create_ichimoku(dataframe, conversion_line_period=20, 
                        displacement=88, base_line_periods=88, laggin_span=88)

        create_ichimoku(dataframe, conversion_line_period=88, 
                        displacement=444, base_line_periods=88, laggin_span=88)

        create_ichimoku(dataframe, conversion_line_period=355,
                        displacement=880, base_line_periods=175, laggin_span=175)


        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)


        """
        Notes: Start Trading

        * En 1m
        dataframe['ichimoku_ok'] = (
            (dataframe['kijun_sen_355_5m'] >= dataframe['tenkan_sen_355_5m']) &
            (dataframe['senkou_a_100'] > dataframe['senkou_b_100']) &
            (dataframe['senkou_a_20'] > dataframe['senkou_b_20']) &
            (dataframe['kijun_sen_20'] > dataframe['tenkan_sen_444']) &
            (dataframe['senkou_a_9'] > dataframe['senkou_a_20']) &
            (dataframe['tenkan_sen_20'] >= dataframe['kijun_sen_20']) &
            (dataframe['tenkan_sen_9'] >= dataframe['tenkan_sen_20']) &
            (dataframe['tenkan_sen_9'] >= dataframe['kijun_sen_9'])
        ).astype('int')

        * En 5m
        dataframe['ichimoku_ok'] = (
            (dataframe['close'] > dataframe['pivot_1d']) &
            (dataframe['r1_1d'] > dataframe['close']) &
            (dataframe['kijun_sen_355'] >= dataframe['tenkan_sen_355']) &
            (dataframe['senkou_a_20'] > dataframe['senkou_b_20']) &
            (dataframe['kijun_sen_20'] > dataframe['tenkan_sen_88']) &
            (dataframe['senkou_a_9'] > dataframe['senkou_a_20']) &
            (dataframe['tenkan_sen_20'] >= dataframe['kijun_sen_20']) &
            (dataframe['tenkan_sen_9'] >= dataframe['tenkan_sen_20']) &
            (dataframe['tenkan_sen_9'] >= dataframe['kijun_sen_9'])
        ).astype('int')


            (dataframe['pivot_1d'] > dataframe['ema20_5m']) anulo ema20_5m para ver si hace entradas en Dry Run

        dataframe['trending_over'] = (
            (
            (dataframe['senkou_b_444'] > dataframe['close'])
            )
            |
            (
            (dataframe['pivot_1d'] > dataframe['close'])
            )
            
        ).astype('int')

        return dataframe
        """

        # Start Trading

        dataframe['pivots_ok'] = (
            (dataframe['close'] > dataframe['pivot_1d']) &
            (dataframe['rS1_1d'] > dataframe['close']) &
            (dataframe['kijun_sen_355'] >= dataframe['tenkan_sen_355']) &
            (dataframe['senkou_a_20'] > dataframe['senkou_b_20'])
        ).astype('int')        

        
        dataframe['trending_over'] = (
            
            (dataframe['senkou_b_88'] > dataframe['close'])
            
        ).astype('int')

        return dataframe
        

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.slow_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['pivots_ok'] > 0)
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trending_over'] > 0)
            ), 'sell'] = 1
        return dataframe
