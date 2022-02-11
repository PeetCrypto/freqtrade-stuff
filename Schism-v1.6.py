import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from technical.indicators import RMI, PMAX
from statistics import mean
from cachetools import TTLCache

import pandas as pd  
pd.options.mode.chained_assignment = None  # default='warn'

"""
TODO: 
    - Continue to hunt for a better all around buy signal.
        - Existing signal to "buy the dip" seems to work well but would it be nice to have an additional
          buy signal for strong upward trends without dips?
        - Prevent buys when potential for strong downward trend and not just a dip?
            - Fibonacci Retracement for resistance/support levels?
            - TD Sequential?
    - Tweak ROI Trend Ride
        - Maybe use free_slots as a factor in how eager we are to sell?
        - Maybe further exploit the ROI table itself using trade data?
    - Tweak sell signal
        - Continue to evaluate good circumstances to bail and sell vs hold
    - Optimize Custom Stop Loss
    - Figure out a way to directly feed a daily hyperopt output into a running strategy and reload it?
    - Per-pair automatic/dynamic ROI based on ADR/ATR or something similar?

NOTES:
    - It is recommended to configure protections as you will use in live and run hyperopt/backtest with
      --enable-protections as this strategy will hit a lot of stoplosses so the stoploss protection is helpful
      to test.
    - Keep in mind the sell signal (dynamic bailout) does not function in backtest and this strategy should be
      validated and tested in dry-run before live. If you do not want to use the sell and only rely on the bits
      of the strategy that can be backtested be sure to turn use_sell_signal = False.
    - Keep in mind that due to the dynamic ROI trend ride this strategy implements that most sells for ROI will
      actually sell for more profit than the ROI table dictates and you can assume that your average profit from 
      ROI based sells will be higher than the backtest shows.
"""

class Schism6(IStrategy):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'inf-guard': 'upper',
        'inf-pct-adr-bot': 0.08848,
        'inf-pct-adr-top': 0.82507,
        'base-mp': 68,
        'base-rmi-fast': 47,
        'base-rmi-slow': 16,
        'xtra-inf-stake-rmi': 54,
        'xtra-base-fiat-rsi': 69,
        'xtra-base-stake-rsi': 61
    }

    sell_params = {
        'sell-rmi-drop': 0.50
    }

    minimal_roi = {
        "0": 0.05,
        "30": 0.025,
        "120": 0.01,
        "360": 0.01,
        "1440": 0
    }

    dynamic_roi = {
        'dynamic_roi_enabled': True,
        'dynamic_roi_type': 'connect'
    }

    stoploss = -0.30
    use_custom_stoploss = True

    custom_stop = {
        'decay-time': 1080,   # minutes to reach end
        'decay-delay': 0,     # minutes to wait before decay starts
        'decay-start': -0.30, # should be the same as initial stoploss
        'decay-end': -0.03,   # ending value
    }

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USD"
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
    
    """
    Informative Pair Definitions
    """
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        # add extra informative pairs if the stake is BTC or ETH
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            for pair in pairs:
                coin, stake = pair.split('/')
                coin_fiat = f"{coin}/{self.custom_fiat}"
                informative_pairs += [(coin_fiat, self.timeframe)]

            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"
            informative_pairs += [(stake_fiat, self.timeframe)]
            informative_pairs += [(stake_fiat, self.inf_timeframe)]

        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
    
        # Base timeframe indicators
        dataframe['rmi-slow'] = RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=8, mom=4)

        # Momentum Pinball: 
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['mp']  = ta.RSI(dataframe['roc'], timeperiod=6)
  
        # RMI Trends and Peaks
        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)
        dataframe['rmi-max'] = dataframe['rmi-slow'].rolling(10, min_periods=1).max()

        # See: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L1059
        # matype = vidya, src = hl2
        dataframe = PMAX(dataframe, period=10, multiplier=3, length=10, MAtype=5, src=2)

        # Other stake specific informative indicators
        # TODO: Re-evaluate the value of these informative indicators
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            fiat = self.custom_fiat
            coin_fiat = f"{coin}/{fiat}"
            stake_fiat = f"{stake}/{fiat}"

            coin_fiat_tf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            dataframe[f"{fiat}_rsi"] = ta.RSI(coin_fiat_tf, timeperiod=14)

            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            stake_fiat_inf_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.inf_timeframe)

            dataframe[f"{stake}_rsi"] = ta.RSI(stake_fiat_tf, timeperiod=14)
            dataframe[f"{stake}_rmi_{self.inf_timeframe}"] = RMI(stake_fiat_inf_tf, length=21, mom=5)

        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['rsi'] = ta.RSI(informative, timeperiod=14)
        
        # Get the "average day range" between the 1d high and 3d low to set up guards
        informative['1d_high'] = informative['close'].rolling(24).max()
        informative['3d_low'] = informative['close'].rolling(72).min()
        informative['adr'] = informative['1d_high'] - informative['3d_low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        dataframe.to_csv('user_data/foo.csv')

        return dataframe

    """
    Buy Signal
    """ 
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.get_pair_params(metadata['pair'], 'buy')
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        """
        The primary "secret sauce" of Schism is to take advantage of the ignore_roi_if_buy_signal setting.
        Ideally, the ROI table is very tight and aggressive allowing for quick exits on ROI without reliance 
        on a sell signal. However, if certain criteria are met for an open trade, we stimulate a sticking buy
        signal on purpose to prevent the bot from selling to the ROI in the midst of an upward trend.

        This is not currently able to be backtested or hyperopted so all settings here must be tested and validated
        in dry-run or live. It is safe to assume that when backtesting, all sells for reason roi will have likely been 
        for higher profits in live trading than in backtest.
        """

        # If active trade, look at trend to persist a buy signal for ignore_roi_if_buy_signal
        if trade_data['active_trade']:
            profit_factor = (1 - (dataframe['rmi-slow'].iloc[-1] / 400))
            rmi_grow = self.linear_growth(30, 70, 180, 720, trade_data['open_minutes'])

            conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * profit_factor))
            # TODO: Experiment with replacing or augmenting this with PMAX 
            conditions.append(dataframe['rmi-up-trend'] == 1)
            conditions.append(dataframe['rmi-slow'] >= rmi_grow)
        
        # Standard signals for entering new trades
        else:

            # Guards on informative timeframe
            if params['inf-guard'] == 'upper' or params['inf-guard'] == 'both':
                conditions.append(
                    (dataframe['close'] <= dataframe[f"3d_low_{self.inf_timeframe}"] + 
                    (params['inf-pct-adr-top'] * dataframe[f"adr_{self.inf_timeframe}"]))
                )

            if params['inf-guard'] == 'lower' or params['inf-guard'] == 'both':
                conditions.append(
                    (dataframe['close'] >= dataframe[f"3d_low_{self.inf_timeframe}"] + 
                    (params['inf-pct-adr-bot'] * dataframe[f"adr_{self.inf_timeframe}"]))
                )
   
            # Primary buy signal
            conditions.append(
                (dataframe[f"rsi_{self.inf_timeframe}"] >= params['inf-rsi']) &
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['base-rmi-slow']) &
                (dataframe['rmi-fast'] <= params['base-rmi-fast']) &
                (dataframe['mp'] <= params['base-mp'])
            )
            
            # Extra conditions for BTC/ETH stakes on additional informative pairs
            if self.config['stake_currency'] in ('BTC', 'ETH'):
                conditions.append(
                    (dataframe[f"{self.config['stake_currency']}_rsi"] < params['xtra-base-stake-rsi']) | 
                    (dataframe[f"{self.custom_fiat}_rsi"] > params['xtra-base-fiat-rsi'])
                )
                conditions.append(dataframe[f"{self.config['stake_currency']}_rmi_{self.inf_timeframe}"] < params['xtra-inf-stake-rmi'])

        # Applies to active or new trades
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    """
    Sell Signal
    """
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.get_pair_params(metadata['pair'], 'sell')
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []
        
        """
        Previous Schism sell implementation has been *partially* replaced by custom_stoploss.
        Doing it there makes it able to be backtested more realistically but some of the functionality
        such as the ability to use other trades or the # of slots in our decision need to remain here.

        The current sell is meant to be a pre-stoploss bailout but most sells at a loss
        should happen due to the stoploss in live/dry.

        ** Only part of this signal functions in backtest/hyperopt. Can only be optimized in live/dry.
        ** Results in backtest/hyperopts will sell far more readily than in live due to the profit guards.
        """

        # If we are in an active trade (which is the only way a sell can occur...)
        # This is needed to block backtest/hyperopt from hitting the profit stuff and erroring out.
        if trade_data['active_trade']:  
            # Decay a loss cutoff point where we allow a sell to occur idea is this allows
            # a trade to go into the negative a little bit before we react to sell.
            loss_cutoff = self.linear_growth(-0.03, 0, 0, 240, trade_data['open_minutes'])
            conditions.append((trade_data['current_profit'] < loss_cutoff))

            # Primary sell trigger
            # TODO: Experiment with replacing or augmenting this with PMAX 
            rmi_drop = dataframe['rmi-max'] - (dataframe['rmi-max'] * 0.50)
            conditions.append(
                (dataframe['rmi-dn-trend'] == 1) &
                (qtpylib.crossed_below(dataframe['rmi-slow'], rmi_drop)) &
                (dataframe['volume'].gt(0))
            )
            
            # Examine the state of our other trades and free_slots to inform the decision
            if trade_data['other_trades']:
                if trade_data['free_slots'] > 0:
                    # If the average of all our other trades is below a certain threshold based
                    # on free slots available, hold and wait for market recovery.
                    max_market_down = -0.04 
                    hold_pct = (1/trade_data['free_slots']) * max_market_down
                    conditions.append(trade_data['avg_other_profit'] >= hold_pct)
                else:
                    # If we are out of free slots disregard the above and allow the biggest loser to sell.
                    conditions.append(trade_data['biggest_loser'] == True)
        else:
            # This is an impossible condition but needs to be here for some reason?
            conditions.append(dataframe['volume'].lt(0))

        """
        rmi_drop = dataframe['rmi-max'] - (dataframe['rmi-max'] * params['sell-rmi-drop'])

        conditions.append(
            (dataframe['rmi-dn-trend'] == 1) &
            (qtpylib.crossed_below(dataframe['rmi-slow'], rmi_drop)) &
            (dataframe['volume'].gt(0))
        )
        """
                           
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

    """
    Custom Stoploss
    """ 
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        start = self.custom_stop['decay-start']
        end = self.custom_stop['decay-end']
        time = self.custom_stop['decay-time']
        delay = self.custom_stop['decay-delay']

        open_minutes: int = (current_time - trade.open_date).total_seconds() // 60

        decay_stoploss = self.linear_growth(start, end, delay, time, open_minutes)

        min_profit = trade.calc_profit_ratio(trade.min_rate)
        max_profit = trade.calc_profit_ratio(trade.max_rate)
        profit_diff = current_profit - min_profit

        # if trade is positive, immediately move the stoploss up to 1% behind it
        if current_profit > 0:
            return current_profit - 0.01

        # if we might be on a rebound, move the stoploss to the low point or keep it where it was
        if current_profit > min_profit:
            if profit_diff > 0.015:
                return min_profit
            return -1

        # otherwise return the time decaying stoploss
        return decay_stoploss

    """
    Trade Timeout Overrides
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        if current_price > order['price'] * 1.01:
            return True
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ask_strategy = self.config.get('ask_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{ask_strategy['price_side']}s"][0][0]
        if current_price < order['price'] * 0.99:
            return True
        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        if current_price > rate * 1.01:
            return False
        return True

    """
    Freqtrade ROI Override for dynamic ROI functionality
    """
    def min_roi_reached_dynamic(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:

        dynamic_roi = self.dynamic_roi
        minimal_roi = self.minimal_roi

        if not dynamic_roi:
            return None, None

        if 'dynamic_roi_type' in dynamic_roi and dynamic_roi['dynamic_roi_type'] \
           in ['linear', 'exponential', 'connect']:
            roi_type = dynamic_roi['dynamic_roi_type']
            # linear decay: f(t) = start - (rate * t)
            if roi_type == 'linear':
                if 'dynamic_roi_start' in dynamic_roi and 'dynamic_roi_end' in dynamic_roi and \
                   'dynamic_roi_time' in dynamic_roi:
                    start = dynamic_roi['dynamic_roi_start']
                    end = dynamic_roi['dynamic_roi_end']
                    time = dynamic_roi['dynamic_roi_time']
                    rate = (start - end) / time
                    min_roi = max(end, start - (rate * trade_dur))
                else:
                    return None, None
            # exponential decay: f(t) = start * e^(-rate*t)
            elif roi_type == 'exponential':
                if 'dynamic_roi_start' in dynamic_roi and 'dynamic_roi_end' in dynamic_roi and \
                   'dynamic_roi_rate' in dynamic_roi:
                    start = dynamic_roi['dynamic_roi_start']
                    end = dynamic_roi['dynamic_roi_end']
                    rate = dynamic_roi['dynamic_roi_rate']
                    min_roi = max(end, start * np.exp(-rate*trade_dur))
                else:
                    return None, None
            # "connect the dots" between the points on the minima_roi table
            elif roi_type == 'connect':
                if not minimal_roi:
                    return None, None
                # figure out where we are in the defined roi table
                past_roi = list(filter(lambda x: x <= trade_dur, minimal_roi.keys()))
                next_roi = list(filter(lambda x: x > trade_dur, minimal_roi.keys()))
                # if we are past the final point in the table, use that key/vaule pair
                if not past_roi:
                    return None, None
                current_entry = max(past_roi)
                if not next_roi:
                    return current_entry, minimal_roi[current_entry]
                next_entry = min(next_roi)
                # use the slope-intercept formula between the two points
                # y = mx + b
                x1, y1 = current_entry, minimal_roi[current_entry]
                x2, y2 = next_entry, minimal_roi[next_entry]
                m = (y1-y2)/(x1-x2)
                b = (x1*y2 - x2*y1)/(x1-x2)
                min_roi = (m * trade_dur) + b
        else:
            return None, None

        return trade_dur, min_roi

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.dynamic_roi and 'dynamic_roi_enabled' in self.dynamic_roi \
           and self.dynamic_roi['dynamic_roi_enabled']:
            _, roi = self.min_roi_reached_dynamic(trade_dur)
        else:
            _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

    """
    Schism Custom Methods
    """
    def populate_trades(self, pair: str) -> dict:
        """
        Query the database and populate the custom_trade_info dict.
        """
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = trade_data['biggest_loser'] = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            if active_trade:
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

                present = arrow.utcnow()
                trade_start  = arrow.get(active_trade[0].open_date)
                open_minutes = (present - trade_start).total_seconds() // 60

                trade_data['active_trade']   = True
                trade_data['current_profit'] = active_trade[0].calc_profit_ratio(current_rate)
                trade_data['peak_profit']    = max(0, active_trade[0].calc_profit_ratio(active_trade[0].max_rate))
                trade_data['open_minutes']   : int = open_minutes
                trade_data['open_candles']   : int = (open_minutes // active_trade[0].timeframe) # floor
            else: 
                trade_data['current_profit'] = trade_data['peak_profit']  = 0.0
                trade_data['open_minutes']   = trade_data['open_candles'] = 0

            other_trades = Trade.get_trades([Trade.pair != pair, Trade.is_open.is_(True),]).all()

            if other_trades:
                trade_data['other_trades'] = True
                other_profit = tuple(trade.calc_profit_ratio(self.get_current_price(trade.pair, False)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(other_profit) 
                if trade_data['current_profit'] < min(other_profit):
                    trade_data['biggest_loser'] = True
            else:
                trade_data['avg_other_profit'] = 0

            open_trades = len(Trade.get_open_trades())
            # this was lazy.
            trade_data['free_slots'] = max(0, self.config['max_open_trades'] - open_trades)

        return trade_data


    def get_current_price(self, pair: str, refresh: bool) -> float:
        """
        Query the exchange for current price information, used in profit calculations.
        Implements a cache to prevent excessive api calls.
        """
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)
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

    def linear_growth(self, start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
        """
        Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
        """
        time = max(0, trade_time - start_time)
        rate = (end - start) / (end_time - start_time)
        return min(end, start + (rate * time))

    def get_pair_params(self, pair: str, side: str) -> Dict:
        """
        Returns buy/sell params that are specific to pairs or groups of pairs if desired.
        TODO:
            - Make this function more robust so that one does not need to ever edit it.
            - Consider adding stoploss and/or ROI tables to the pair/group specific options
        """
        buy_params = self.buy_params
        sell_params = self.sell_params
  
        ### Stake: USD
        if pair in ('ABC/XYZ', 'DEF/XYZ'):
            buy_params = self.buy_params_GROUP1
            sell_params = self.sell_params_GROUP1
        elif pair in ('QRD/WTF'):
            buy_params = self.buy_params_QRD
            sell_params = self.sell_params_QRD

        if side == 'sell':
            return sell_params

        return buy_params

# Sub-strategy with parameters specific to BTC stake
class Schism6_BTC(Schism6):

    timeframe = '1h'
    inf_timeframe = '4h'

    buy_params = {
        'inf-rsi': 64,
        'mp': 55,
        'rmi-fast': 31,
        'rmi-slow': 16,
        'xinf-stake-rmi': 67,
        'xtf-fiat-rsi': 17,
        'xtf-stake-rsi': 57
    }

    minimal_roi = {
        "0": 0.05,
        "240": 0.025,
        "1440": 0.01,
        "4320": 0
    }

    stoploss = -0.30
    use_custom_stoploss = False

# Sub-strategy with parameters specific to ETH stake
class Schism6_ETH(Schism6):

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
        "240": 0.025,
        "1440": 0.01,
        "4320": 0
    }

    stoploss = -0.30
    use_custom_stoploss = False