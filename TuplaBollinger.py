# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------


class TuplaBollinger(IStrategy):
   
    EMA_LONG_TERM = 200
    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.9,
        "1": 0.05,
        "10": 0.04,
        "15": 0.5
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.25

    # Optimal timeframe for the strategy
    timeframe = '5h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger bands inner
        bollinger_inner = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['inner_lowerband'] = bollinger_inner['lower']
        dataframe['bb_middleband'] = bollinger_inner['mid']
        dataframe['inner_upperband'] = bollinger_inner['upper']
        
        # Bollinger bands outer
        bollinger_outer = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['outer_lowerband'] = bollinger_outer['lower']
        #dataframe['bb_middleband'] = bollinger_outer['mid']
        dataframe['outer_upperband'] = bollinger_outer['upper']

        # EMA 200 for trend indicator
        dataframe['ema_{}'.format(self.EMA_LONG_TERM)] = ta.EMA(
            dataframe, timeperiod=self.EMA_LONG_TERM
        )


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['inner_lowerband']) &
                    (dataframe['close'].shift(1) < dataframe['close'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['inner_upperband']) &
                    (dataframe['close'].shift(1) > dataframe['close'])

            ),
            'sell'] = 1
        return dataframe
