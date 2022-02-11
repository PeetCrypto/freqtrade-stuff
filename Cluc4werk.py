import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

class Cluc4werk(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    #    943/1000:   2423 trades. 1754/0/669 Wins/Draws/Losses. Avg profit   0.65%. Median profit   0.86%. Total profit  0.15692605 ETH ( 1566.74Σ%). Avg duration  58.4 min. Objective: -213.61288
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.0085,
        'bbdelta-tail': 0.76175,
        'close-bblower': 0.01517,
        'closedelta-close': 0.01514,
        'rocr-1h': 0.58912,
        'volume': 21
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-bbmiddle-close': 0.99955
    }

    # ROI table:
    minimal_roi = {
        "0": 0.01497,
        "77": 0.01321,
        "130": 0.00976,
        "356": 0.00709,
        "464": 0.0027,
        "564": 0.0016,
        "697": 0
    }

    # Stoploss:
    stoploss = -0.02055

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.17429
    trailing_stop_positive_offset = 0.2716
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """
    
    timeframe = '1m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']

        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe, timeperiod=28)
        
        inf_tf = '1h'
        
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['rocr'] = ta.ROCR(informative, timeperiod=168)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr-1h'])
            ) &
            ((      
                    dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['close'] * params['bbdelta-close']) &
                    dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail']) &
                    dataframe['close'].lt(dataframe['lower'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
            ) |
            (       
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < params['close-bblower'] * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (dataframe['high'].le(dataframe['high'].shift(1))) &
            (dataframe['high'].shift(1).le(dataframe['high'].shift(2))) &
            (dataframe['close'].le(dataframe['close'].shift(1))) &
            ((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        """
        dataframe.loc[
            #(dataframe['high'].le(dataframe['high'].shift(1))) &
            #(dataframe['close'] > dataframe['bb_middleband']) &
            (qtpylib.crossed_above((dataframe['close'] * params['sell-bbmiddle-close']),dataframe['bb_middleband'])) &
            #(qtpylib.crossed_above(dataframe['close'],dataframe['bb_middleband'])) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1
        """

        return dataframe

class Cluc4werk_ETH(Cluc4werk):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    #    943/1000:   2423 trades. 1754/0/669 Wins/Draws/Losses. Avg profit   0.65%. Median profit   0.86%. Total profit  0.15692605 ETH ( 1566.74Σ%). Avg duration  58.4 min. Objective: -213.61288
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.0085,
        'bbdelta-tail': 0.76175,
        'close-bblower': 0.01517,
        'closedelta-close': 0.01514,
        'rocr-1h': 0.58912,
        'volume': 21
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-bbmiddle-close': 0.99955
    }

    # ROI table:
    minimal_roi = {
        "0": 0.01497,
        "77": 0.01321,
        "130": 0.00976,
        "356": 0.00709,
        "464": 0.0027,
        "564": 0.0016,
        "697": 0
    }

    # Stoploss:
    stoploss = -0.02055

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.17429
    trailing_stop_positive_offset = 0.2716
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """