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
TODO: 
    - Continue to hunt for a better all around buy signal.
        - Existing signal to "buy the dip" seems to work well but would it be nice to have an additional
          buy signal for strong upward trends without dips?
        - Prevent buys when potential for strong downward trend and not just a dip?
            - Initial 24h range thing seems to be step 1 of this process, anything else we can do?
            - Fibonacci Retracement for resistance/support levels?
    - Tweak ROI Ride
        - Maybe use free_slots as a factor in how eager we are to sell?
    - Tweak sell signal
        - Continue to evaluate good circumstances to sell vs hold

Loosely based on:
https://github.com/nicolay-zlobin/jesse-indicators/blob/main/strategies/BadStreak/__init__.py
"""

class Schism(IStrategy):
    """
    Strategy Configuration Items
    """
    timeframe = '5m'

    buy_params = {
        'mp': 65
    }

    sell_params = {}

    minimal_roi = {
        "0": 0.02,
        "60": 0.01,
        "1440": 0
    }

    use_custom_stoploss = True
    stoploss = -0.20

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72

    # Custom Variables
    custom_trade_info = {}
    custom_fiat = "USD"
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes

    # at what % profit to start trailing
    custom_stoploss_trail_start = 0.01
    # how far behind it should trail (positive here means 1% below)
    custom_stoploss_trail_dist = 0.01
    
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
        dataframe['mac'] = cta.macross(dataframe, 21, 55)
        dataframe['streak'] = cta.mastreak(dataframe, period=4)

        lookup_idxs = dataframe.index.values - (abs(dataframe['streak'].values) + 1)
        valid_lookups = lookup_idxs >= 0
        dataframe['sbc'] = np.nan
        dataframe.loc[valid_lookups, 'sbc'] = dataframe['close'].to_numpy()[lookup_idxs[valid_lookups].astype(int)]

        dataframe['streak-roc'] = 100 * (dataframe['close'] - dataframe['sbc']) / dataframe['sbc']

        # Percent Change Channel
        upper, mid, lower = cta.pcc(dataframe, period=20, mult=2)
        dataframe['pcc-lowerband'] = lower
        dataframe['pcc-upperband'] = upper

        # RMI Trend Strength
        dataframe['rmi'] = RMI(dataframe, length=21, mom=5)

        # RMI Trend Calculations    
        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

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
                (dataframe['mp'] < params['mp']) &
                (dataframe['streak-roc'] > dataframe['pcc-lowerband']) &
                (dataframe['mac'] == 1)
            )

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
    Price protection on trade entry and timeouts, built-in Freqtrade functionality
    https://www.freqtrade.io/en/latest/strategy-advanced/#custom-stoploss
    """
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit < self.custom_stoploss_trail_start:
            return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # After reaching the desired offset, allow the stoploss to trail the defined value
        desired_stoploss = current_profit - (current_profit * self.custom_stoploss_trail_dist)

        # Use a minimum of 0.5%
        return max(desired_stoploss, 0.005)

"""
Sub-strategy overrides
Anything not explicity defined here will follow the settings in the base strategy
"""
# Sub-strategy with parameters specific to BTC stake
class Schism_BTC(Schism):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'mp': 66,
        'tf-fiat-rsi': 38,
        'tf-stake-rsi': 60,
        'inf-rsi': 14,
        'inf-stake-rmi': 51
    }

    use_sell_signal = False

# Sub-strategy with parameters specific to ETH stake
class Schism_ETH(Schism):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'mp': 40,
        'tf-fiat-rsi': 15,
        'tf-stake-rsi': 92,
        'inf-rsi': 13,
        'inf-stake-rmi': 69,
    }

    trailing_stop = True
    trailing_stop_positive = 0.014
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = False

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