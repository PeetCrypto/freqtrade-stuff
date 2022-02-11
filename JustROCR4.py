from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class JustROCR4(IStrategy):
    minimal_roi = {
        "0": 0.15
    }

    stoploss = -0.15
    trailing_stop = False
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rocr'] = ta.ROCR(dataframe, period=499)
        dataframe['rocr_200'] = ta.ROCR(dataframe, period=200)
        dataframe['rocr_100'] = ta.ROCR(dataframe, period=100)
        dataframe['rocr_20'] = ta.ROCR(dataframe, period=20)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rocr'] > 1.20) &
                (dataframe['rocr_200'] > 1.15) &
                (dataframe['rocr_100'] > 1.10) &
                (dataframe['rocr_20'] > 1.05)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
