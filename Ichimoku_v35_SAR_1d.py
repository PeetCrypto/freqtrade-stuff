from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
#from technical.indicators import accumulation_distribution
from technical.util import resample_to_interval, resampled_merge
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy
from technical.indicators import ichimoku

class Ichimoku_v35(IStrategy):
    """

    """

    minimal_roi = {
        "0": 100
    }

    stoploss = -1 #-0.35

    ticker_interval = '4h' #3m

    # startup_candle_count: int = 2

    # trailing stoploss
    #trailing_stop = True
    #trailing_stop_positive = 0.40 #0.35
    #trailing_stop_positive_offset = 0.50
    #trailing_only_offset_is_reached = False

    def informative_pairs(self):
        # Optionally Add additional "static" pairs
        informative_pairs += [(pair, '1d') for pair in pairs] # [("BTC/USDT", "1d")]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        inf_tf = '1d'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Get the 14 day rsi
        # informative['rsi'] = ta.RSI(informative, timeperiod=14)

        # Get the 14 day Stochastic
        # stochastic = stoch(informative, window=14, d=3, k=3, fast=False)
        # informative['slowd'] = stochastic['slow_d']
        # informative['slowk'] = stochastic['slow_k']

        # Get the Ichimoku
        ichi = ichimoku(informative, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        informative['tenkan'] = ichi['tenkan_sen']
        informative['kijun'] = ichi['kijun_sen']
        informative['senkou_a'] = ichi['senkou_span_a']
        informative['senkou_b'] = ichi['senkou_span_b']
        # Calculate rsi of the original dataframe (5m timeframe)
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Pattern Recognition - Bearish candlestick patterns
        # # Evening Doji Star: values [0, 100]
        informative['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)

        # Parabolic SAR
        informative['sar'] = ta.SAR(dataframe)

        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # Calculate Stoch of the original dataframe (4h timeframe)
        # stochastic = stoch(dataframe, window=14, d=3, k=3, fast=False)
        # dataframe['slowd'] = stochastic['slow_d']
        # dataframe['slowk'] = stochastic['slow_k']

        ichi = ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        # dataframe['chikou_span'] = ichi['chikou_span']
        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'].shift(2), dataframe['senkou_a'])) &
                (dataframe['close'].shift(2) > dataframe['senkou_a']) &
                (dataframe['close'].shift(2) > dataframe['senkou_b'])
            ),
            'buy'] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'].shift(2), dataframe['senkou_b'])) &
                (dataframe['close'].shift(2) > dataframe['senkou_a']) &
                (dataframe['close'].shift(2 ) > dataframe['senkou_b'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close_1d'] < dataframe['sar_1d'])
            ),
            'sell'] = 1

        return dataframe
