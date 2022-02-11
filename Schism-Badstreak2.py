import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.persistence import Trade
from technical.indicators import RMI
from technical.util import resample_to_interval, resampled_merge

from typing import Dict, List, Optional, Tuple
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from statistics import mean
from cachetools import TTLCache
from collections import namedtuple

"""
Loosely based on:
https://github.com/nicolay-zlobin/jesse-indicators/blob/main/strategies/BadStreak/__init__.py
"""

class Schism(IStrategy):
    """
    Strategy Configuration Items
    """
    timeframe = '15m'
    inf_timeframe = '4h'

    buy_params = {
        'inf-rsi-1': 51,
        'inf-rsi-2': 52,
        'mp': 42,
        'rmi-above': 17,
        'rmi-below': 60,
        'xinf-stake-rmi': 59,
        'xtf-fiat-rsi': 48,
        'xtf-stake-rsi': 66
    }

    sell_params = {}

    minimal_roi = {
        "0": 0.05,
        "60": 0.025,
        "180": 0.01,
        "1440": 0
    }

    stoploss = -0.30

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72

    # Custom Variables
    custom_trade_info = {}
    custom_fiat = "USD"
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes
    
    """
    Informative Pair Definitions
    """
    def informative_pairs(self):
        # add existing pairs from whitelist on the inf_timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        # add additional informative pairs based on certain stakes
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            for pair in pairs:
                # add in the COIN/FIAT pairs (e.g. XLM/USD) on base timeframe
                coin, stake = pair.split('/')
                coin_fiat = f"{coin}/{self.custom_fiat}"
                informative_pairs += [(coin_fiat, self.timeframe)]

            # add in the STAKE/FIAT pair (e.g. BTC/USD) on base and inf timeframes
            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"
            informative_pairs += [(stake_fiat, self.timeframe)]
            informative_pairs += [(stake_fiat, self.inf_timeframe)]

        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Populate/update the trade data if there is any, set trades to false if not live/dry
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
    
        # Momentum Pinball
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['mp']  = ta.RSI(dataframe['roc'], timeperiod=3)

        # MA Streak
        dataframe['mac'] = ci_mac(dataframe, 20, 50)
        dataframe['streak'] = ci_mastreak(dataframe, period=4)

        streak = abs(int(dataframe['streak'].iloc[-1]))
        streak_back_close = dataframe['close'].shift(streak + 1)

        dataframe['streak-roc'] = 100 * (dataframe['close'] - streak_back_close) / streak_back_close

        # Percent Change Channel
        pcc = ci_pcc(dataframe, period=20, mult=2)
        dataframe['pcc-lowerband'] = pcc.lowerband
        dataframe['pcc-upperband'] = pcc.upperband

        # RMI Trend Strength
        dataframe['rmi'] = RMI(dataframe, length=21, mom=5)

        # RMI Trend Calculations    
        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        # Informative for STAKE/FIAT and COIN/FIAT on default timeframe, only relevant if stake currency is BTC or ETH
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            fiat = self.custom_fiat
            coin_fiat = f"{coin}/{fiat}"
            stake_fiat = f"{stake}/{fiat}"

            # COIN/FIAT (e.g. XLM/USD) - timeframe
            coin_fiat_tf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            dataframe[f"{fiat}_rsi"] = ta.RSI(coin_fiat_tf, timeperiod=14)

            # STAKE/FIAT (e.g. BTC/USD) - inf_timeframe
            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            stake_fiat_inf_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.inf_timeframe)

            dataframe[f"{stake}_rsi"] = ta.RSI(stake_fiat_tf, timeperiod=14)
            dataframe[f"{stake}_rmi_{self.inf_timeframe}"] = RMI(stake_fiat_inf_tf, length=21, mom=5)

        # Informative indicators for current pair on inf_timeframe
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    """
    Buy Trigger Signals
    """
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # Persist a buy signal for existing trades to make use of ignore_roi_if_buy_signal = True
        # when this buy signal is not present a sell can happen according to the defined ROI table
        if trade_data['active_trade']:
            # peak_profit factor f(x)=1-x/400, rmi 30 -> 0.925, rmi 80 -> 0.80
            profit_factor = (1 - (dataframe['rmi'].iloc[-1] / 400))
            # grow from 30 -> 70 after 720 minutes starting after 180 minutes
            rmi_grow = linear_growth(30, 70, 180, 720, trade_data['open_minutes'])

            conditions.append(dataframe['rmi-up-trend'] == 1)
            conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * profit_factor))
            conditions.append(dataframe['rmi'] >= rmi_grow)

        # Normal buy triggers that apply to new trades we want to enter
        else:
            # Primary buy triggers
            conditions.append(
                (
                    # core "buy the dip" badstreak
                    (dataframe[f"rsi_{self.inf_timeframe}"] < params['inf-rsi-1']) &
                    (dataframe['mp'] < params['mp']) &
                    (dataframe['streak-roc'] > dataframe['pcc-lowerband']) &
                    (dataframe['mac'] == 1)
                ) | 
                (
                    # additional set of conditions to buy in strong upward trends
                    (dataframe[f"rsi_{self.inf_timeframe}"] < params['inf-rsi-2']) &
                    (dataframe['rmi-up-trend'] == 1) &
                    (dataframe['rmi'] < params['rmi-below']) &
                    (dataframe['rmi'] > params['rmi-above'])
                )
            )

            # If the stake is BTC or ETH apply additional conditions
            if self.config['stake_currency'] in ('BTC', 'ETH'):
                # default timeframe conditions
                conditions.append(
                    (dataframe[f"{self.config['stake_currency']}_rsi"] > params['xtf-stake-rsi']) | 
                    (dataframe[f"{self.custom_fiat}_rsi"] < params['xtf-fiat-rsi'])
                )
                # informative timeframe conditions
                conditions.append(dataframe[f"{self.config['stake_currency']}_rmi_{self.inf_timeframe}"] < params['xinf-stake-rmi'])

        # Anything below here applies to persisting and new buy signal
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    """
    Sell Trigger Signals
    """
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # In this strategy all sells for profit happen according to ROI
        # This sell signal is designed only as a "dynamic stoploss"

        # if we are in an active trade for this pair
        if trade_data['active_trade']:     
            # grow from -0.03 -> 0 after 300 minutes starting immediately
            loss_cutoff = linear_growth(-0.03, 0, 0, 300, trade_data['open_minutes'])

            # if we are at a loss, consider what the trend looks and preempt the stoploss
            conditions.append(
                (trade_data['current_profit'] < loss_cutoff) & 
                (trade_data['current_profit'] > self.stoploss) &  
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['volume'].gt(0))
            )
            # if the peak profit was positive at some point but never reached ROI, set a higher cross point for exit
            if trade_data['peak_profit'] > 0:
                conditions.append(qtpylib.crossed_below(dataframe['rmi'], 50))
            # if the trade was always negative, the bounce we expected didn't happen
            else:
                conditions.append(qtpylib.crossed_below(dataframe['rmi'], 10))

            # if there are other open trades in addition to this one, consider the average profit 
            # across them all and how many free slots we have in our sell decision
            if trade_data['other_trades']:
                if trade_data['free_slots'] > 0:
                    # more free slots, the higher this threshold gets
                    hold_pct = (trade_data['free_slots'] / 100) * -1
                    conditions.append(trade_data['avg_other_profit'] >= hold_pct)
                else:
                    # if were out of slots, allow the biggest losing trade to sell regardless of avg profit
                    conditions.append(trade_data['biggest_loser'] == True)

        # the bot comes through this loop even when there isn't an open trade to sell
        # so we pass an impossible condiiton here
        else:
            conditions.append(dataframe['volume'].lt(0))
                           
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        
        return dataframe

    """
    Custom Methods for Live Trade Data and Realtime Price
    """
    # Populate trades_data from the database
    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = trade_data['biggest_loser'] = False
        self.custom_trade_info['meta'] = {}

        # active trade stuff only works in live and dry, not backtest
        if self.config['runmode'].value in ('live', 'dry_run'):
            
            # find out if we have an open trade for this pair
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            # if so, get some information
            if active_trade:
                # get current price and update the min/max rate
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

                # get how long the trade has been open in minutes and candles
                present = arrow.utcnow()
                trade_start  = arrow.get(active_trade[0].open_date)
                open_minutes = (present - trade_start).total_seconds() // 60  # floor

                # set up the things we use in the strategy
                trade_data['active_trade']   = True
                trade_data['current_profit'] = active_trade[0].calc_profit_ratio(current_rate)
                trade_data['peak_profit']    = max(0, active_trade[0].calc_profit_ratio(active_trade[0].max_rate))
                trade_data['open_minutes']   : int = open_minutes
                trade_data['open_candles']   : int = (open_minutes // active_trade[0].timeframe) # floor
            else: 
                trade_data['current_profit'] = trade_data['peak_profit']  = 0.0
                trade_data['open_minutes']   = trade_data['open_candles'] = 0

            # if there are open trades not including the current pair, get some information
            # future reference, for *all* open trades: open_trades = Trade.get_open_trades()
            other_trades = Trade.get_trades([Trade.pair != pair, Trade.is_open.is_(True),]).all()

            if other_trades:
                trade_data['other_trades'] = True
                other_profit = tuple(trade.calc_profit_ratio(self.get_current_price(trade.pair, False)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(other_profit) 
                # find which of our trades is the biggest loser
                if trade_data['current_profit'] < min(other_profit):
                    trade_data['biggest_loser'] = True
            else:
                trade_data['avg_other_profit'] = 0

            # get the number of free trade slots, storing in every pairs dict due to laziness
            open_trades = len(Trade.get_open_trades())
            trade_data['free_slots'] = max(0, self.config['max_open_trades'] - open_trades)

        return trade_data

    # Get the current price from the exchange (or cache)
    def get_current_price(self, pair: str, refresh: bool) -> float:
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)
            # Check if cache has been invalidated
            if rate:
                return rate

        ask_strategy = self.config.get('ask_strategy', {})
        if ask_strategy.get('use_order_book', False):
            ob = self.dp.orderbook(pair, 1)
            rate = ob[f"{ask_strategy['price_side']}s"][0][0]
        else:
            ticker = self.dp.ticker(pair)
            rate = ticker['last']

        self.custom_current_price_cache[pair] = rate
        return rate

    """
    Price protection on trade entry and timeouts, built-in Freqtrade functionality
    https://www.freqtrade.io/en/latest/strategy-advanced/
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > order['price'] * 1.01:
            return True
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ask_strategy = self.config.get('ask_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{ask_strategy['price_side']}s"][0][0]
        # Cancel sell order if price is more than 1% below the order.
        if current_price < order['price'] * 0.99:
            return True
        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > rate * 1.01:
            return False
        return True

"""
Sub-strategy overrides
Anything not explicity defined here will follow the settings in the base strategy
"""
# Sub-strategy with parameters specific to BTC stake
class Schism_BTC(Schism):

    timeframe = '1h'
    inf_timeframe = '4h'

    buy_params = {
        'inf-rsi': 14,
        'inf-stake-rmi': 51,
        'mp': 66,
        'rmi-fast': 34,
        'rmi-slow': 19,
        'tf-fiat-rsi': 38,
        'tf-stake-rsi': 60
    }

    minimal_roi = {
        "0": 0.05,
        "360": 0.025,
        "1440": 0.01,
        "4320": 0
    }

    use_sell_signal = False

# Sub-strategy with parameters specific to ETH stake
class Schism_ETH(Schism):

    timeframe = '1h'
    inf_timeframe = '4h'

    buy_params = {
        'inf-rsi': 13,
        'inf-stake-rmi': 69,
        'mp': 40,
        'rmi-fast': 42,
        'rmi-slow': 17,
        'tf-fiat-rsi': 15,
        'tf-stake-rsi': 92
    }

    minimal_roi = {
        "0": 0.05,
        "360": 0.025,
        "1440": 0.01,
        "4320": 0
    }

    use_sell_signal = False

"""
Custom Indicators
"""

"""
linear growth, starts at X and grows to Y after A minutes (starting after B miniutes)
f(t) = X + (rate * t), where rate = (Y - X) / (A - B)
"""
def linear_growth(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)
    return min(end, start + (rate * trade_time))

"""
Moving Average Cross
Port of: https://www.tradingview.com/script/PcWAuplI-Moving-Average-Cross/
"""
def ci_mac(dataframe: DataFrame, fast: int = 20, slow: int = 50) -> Series:

    dataframe = dataframe.copy()

    # Fast MAs
    upper_fast = ta.EMA(dataframe['high'], timeperiod=fast)
    lower_fast = ta.EMA(dataframe['low'], timeperiod=fast)

    # Slow MAs
    upper_slow = ta.EMA(dataframe['high'], timeperiod=slow)
    lower_slow = ta.EMA(dataframe['low'], timeperiod=slow)

    # Crosses
    crosses_lf_us = qtpylib.crossed_above(lower_fast, upper_slow) | qtpylib.crossed_below(lower_fast, upper_slow)
    crosses_uf_ls = qtpylib.crossed_above(upper_fast, lower_slow) | qtpylib.crossed_below(upper_fast, lower_slow)

    dir_1 = np.where(crosses_lf_us, 1, np.nan)
    dir_2 = np.where(crosses_uf_ls, -1, np.nan)

    dir = np.where(dir_1 == 1, dir_1, np.nan)
    dir = np.where(dir_2 == -1, dir_2, dir_1)

    res = Series(dir).fillna(method='ffill').to_numpy()

    return res

"""
MA Streak
Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
"""
def ci_mastreak(dataframe: DataFrame, period: int = 4, source_type='close') -> Series:
    
    dataframe = dataframe.copy()

    avgval = zlema(dataframe[source_type], period)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                    -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    res = np.concatenate((np.full((dataframe.shape[0] - streak.shape[0]), np.nan), streak))

    return res

"""
Percent Change Channel
PCC is like KC unless it uses percentage changes in price to set channel distance.
https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
"""
def ci_pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):

    PercentChangeChannel = namedtuple('PercentChangeChannel', ['upperband', 'middleband', 'lowerband'])

    dataframe = dataframe.copy()

    close = dataframe['close']
    previous_close = close.shift()
    low = dataframe['low']
    high = dataframe['high']

    close_change = (close - previous_close) / previous_close * 100
    high_change = (high - close) / close * 100
    low_change = (low - close) / close * 100

    mid = zlema(close_change, period)
    rangema = zlema(high_change - low_change, period)

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return PercentChangeChannel(upper, rangema, lower)

"""
Zero Lag EMA
"""
def zlema(series: Series, period):
    ema1 = ta.EMA(series, period)
    ema2 = ta.EMA(ema1, period)
    d = ema1 - ema2
    zlema = ema1 + d
    return zlema