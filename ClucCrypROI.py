import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from functools import reduce
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from datetime import datetime
from freqtrade.persistence import Trade
from pandas import DataFrame, Series

class ClucCrypROI(IStrategy):

    # Used for "informative pairs"
    fiat  = 'USD'
    
    startup_candle_count: int = 48

    def informative_pairs(self):
        """
        Add informative pairs as follows
        coin to fiat @ same candle as base strategy
        stake to fiat @ same candle as base strategy
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []

        for pair in pairs:
            coin, stake = pair.split('/')
            coin_fiat = f"{coin}/{self.fiat}"
            informative_pairs += [(coin_fiat, self.timeframe)]
        
        stake_fiat = f"{stake}/{self.fiat}"

        informative_pairs += [(stake_fiat, self.timeframe)]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Set Up Bollinger Bands
        upper_bb1, mid_bb1, lower_bb1 = ta.BBANDS(dataframe['close'], timeperiod=36)
        upper_bb2, mid_bb2, lower_bb2 = ta.BBANDS(qtpylib.typical_price(dataframe), timeperiod=12)

        # Only putting some bands into dataframe as the others are not used elsewhere in the strategy
        dataframe['lower-bb1'] = lower_bb1
        dataframe['lower-bb2'] = lower_bb2
        dataframe['mid-bb2'] = mid_bb2
       
        dataframe['bb1-delta'] = (mid_bb1 - dataframe['lower-bb1']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        # Additional indicators
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=24).mean()

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher-rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        
        # Informative Pair Indicators
        coin, stake = metadata['pair'].split('/')
        fiat = self.fiat
        stake_fiat = f"{stake}/{self.fiat}"
        coin_fiat = f"{coin}/{self.fiat}"

        coin_fiat_inf  = self.dp.get_pair_dataframe(pair=f"{coin}/{fiat}", timeframe=self.timeframe)
        dataframe['coin-fiat-adx'] = ta.ADX(coin_fiat_inf, timeperiod=21)
        coin_aroon = ta.AROON(coin_fiat_inf, timeperiod=25)
        dataframe['coin-fiat-aroon-down'] = coin_aroon['aroondown'] 
        dataframe['coin-fiat-aroon-up'] = coin_aroon['aroonup']

        stake_fiat_inf = self.dp.get_pair_dataframe(pair=f"{stake}/{fiat}", timeframe=self.timeframe)
        dataframe['stake-fiat-adx'] = ta.ADX(stake_fiat_inf, timeperiod=21)
        stake_aroon = ta.AROON(stake_fiat_inf, timeperiod=25)
        dataframe['stake-fiat-aroon-down'] = stake_aroon['aroondown'] 
        dataframe['stake-fiat-aroon-up'] = stake_aroon['aroonup']

        # These indicators are used to persist a buy signal in live trading only
        # They dramatically slow backtesting down
        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        active_trade = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()

        conditions = []

        """
        If this is a fresh buy, apple additional conditions.
        Idea is to leverage "ignore_roi_if_buy_signal = True" functionality by using certain
        indicators for active trades while applying additional protections to new trades.
        """
        if not active_trade:
            if 'stake-fiat-adx' in dataframe.columns and 'coin-fiat-adx' in dataframe.columns:
                conditions.append(
                    ((
                        (dataframe['stake-fiat-adx'] > params['adx']) & 
                        (dataframe['stake-fiat-aroon-down'] > params['aroon'])
                    ) | (
                        (dataframe['stake-fiat-adx'] < params['adx']) 
                    )) & ((
                        (dataframe['coin-fiat-adx'] > params['adx']) & 
                        (dataframe['coin-fiat-aroon-up'] > params['aroon'])
                    ) | (
                        (dataframe['coin-fiat-adx'] < params['adx']) 
                    ))
                )

            conditions.append(
                (      
                    dataframe['bb1-delta'].gt(dataframe['close'] * params['bbdelta-close']) &
                    dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                    dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail']) &
                    dataframe['close'].lt(dataframe['lower-bb1'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
                ) |
                (       
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < params['close-bblower'] * dataframe['lower-bb2']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
                )
            )

        else:
            conditions.append(dataframe['close'] > dataframe['close'].shift())
            conditions.append(dataframe['close'] > dataframe['sar'])

        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []

        if 'stake-fiat-adx' in dataframe.columns and 'coin-fiat-adx' in dataframe.columns:
            conditions.append(
                ((
                    (dataframe['stake-fiat-adx'] > params['sell-adx']) & 
                    (dataframe['stake-fiat-aroon-up'] > params['sell-aroon'])
                ) | (
                    (dataframe['stake-fiat-adx'] < params['sell-adx']) 
                )) & ((
                    (dataframe['coin-fiat-adx'] > params['sell-adx']) & 
                    (dataframe['coin-fiat-aroon-down'] > params['sell-aroon'])
                ) | (
                    (dataframe['coin-fiat-adx'] < params['sell-adx']) 
                ))
            )

        conditions.append((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['mid-bb2'])
        conditions.append(dataframe['ema_fast'].gt(dataframe['close']))
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

    """
    https://www.freqtrade.io/en/latest/strategy-advanced/

    Custom Order Timeouts
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > order['price'] * 1.01:
            return True
        return False


    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel sell order if price is more than 1% below the order.
        if current_price < order['price'] * 0.99:
            return True
        return False

# Overriding strategy with optimized values when ETH is the stake currency
class ClucCrypROI_ETH(ClucCrypROI):

    timeframe = '15m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Buy hyperspace params:
    buy_params = {
        'adx': 46,
        'aroon': 51,
        'bbdelta-close': 0.01775,
        'bbdelta-tail': 0.84013,
        'close-bblower': 0.01096,
        'closedelta-close': 0.01068,
        'volume': 15
    }
	
    # Sell hyperspace params:
    sell_params = {
		'sell-adx': 39, 
		'sell-aroon': 65, 
		'sell-bbmiddle-close': 0.98656
    }
	
    # ROI table:
    minimal_roi = {
        "0": 0.11487,
        "4": 0.10783,
        "5": 0.07209,
        "15": 0.03179,
        "60": 0.01044,
        "112": 0.00541,
        "270": 0
    }
	
    # Stoploss:
    stoploss = -0.32238
	
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0102
    trailing_stop_positive_offset = 0.02872
    trailing_only_offset_is_reached = False

# Overriding strategy with optimized values when BTC is the stake currency
class ClucCrypROI_BTC(ClucCrypROI):

    timeframe = '15m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Buy hyperspace params:
    buy_params = {
        'adx': 47,
        'aroon': 34,
        'bbdelta-close': 0.01957,
        'bbdelta-tail': 0.86961,
        'close-bblower': 0.00257,
        'closedelta-close': 0.01381,
        'volume': 27
    }

    # Sell hyperspace params:
    sell_params = {
		'sell-adx': 20, 
		'sell-aroon': 63, 
		'sell-bbmiddle-close': 0.95563
    }
	
    # ROI table:
    minimal_roi = {
        "0": 0.08449,
        "7": 0.04432,
        "28": 0.0387,
        "30": 0.0137,
        "145": 0.0086,
        "318": 0.00344,
        "600": 0
    }

    # Stoploss:
    stoploss = -0.33807
	
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01001
    trailing_stop_positive_offset = 0.02495
    trailing_only_offset_is_reached = False