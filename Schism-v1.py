import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI, VIDYA
from statistics import mean
from cachetools import TTLCache


"""
TODO: 
    - Better buy signal.
    - Potentially leverage an external data source?
    - Make stop-loss bailout a bit more intelligent (based on free slots, time passed, etc.)
        - We're getting stuck in too many losing trades and it creates a domino effect.
"""

class Schism(IStrategy):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params  = {}
    sell_params = {}

    # ROI table:
    minimal_roi = {
        "0": 0.05,
        "10": 0.025,
        "20": 0.015,
        "30": 0.01
    }

    # Stoploss:
    stoploss = -0.50

    # Probably don't change these
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 21

    # Custom stored data, these are required.
    custom_trade_info = {}
    custom_fiat = "USD"
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        # add additional informative pairs based on certain stakes
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            for pair in pairs:
                # add in the COIN/FIAT pairs (e.g. XLM/USD)
                coin, stake = pair.split('/')
                coin_fiat = f"{coin}/{self.custom_fiat}"
                informative_pairs += [(coin_fiat, self.timeframe)]

            # add in the STAKE/FIAT pair (e.g. BTC/USD)
            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"
            informative_pairs += [(stake_fiat, self.timeframe)]

        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Populate/update the trade data if there is any, set trades to false if not live/dry
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
    
        # Set up primary indicators
        dataframe['rmi-slow'] = RMI(dataframe, length=20, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=9, mom=3)
        dataframe['vidya'] = VIDYA(dataframe)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['kama-fast'] = ta.KAMA(dataframe, timeperiod=5)
        dataframe['kama-slow'] = ta.KAMA(dataframe, timeperiod=13)

        # Trend Calculations    
        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        dataframe['vdy-up'] = np.where(dataframe['vidya'] >= dataframe['vidya'].shift(),1,0)      
        dataframe['vdy-dn'] = np.where(dataframe['vidya'] <= dataframe['vidya'].shift(),1,0)      
        dataframe['vdy-up-trend'] = np.where(dataframe['vdy-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['vdy-dn-trend'] = np.where(dataframe['vdy-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        # Informative for STAKE/FIAT and COIN/FIAT on default timeframe, only relevant if stake currency is BTC or ETH
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            coin_fiat = f"{coin}/{self.custom_fiat}"
            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"

            coin_fiat_inf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            stake_fiat_inf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)

            dataframe[f"{self.custom_fiat}_rmi-slow"] = RMI(coin_fiat_inf, length=20, mom=5)
            dataframe[f"{self.config['stake_currency']}_rmi-slow"] = RMI(stake_fiat_inf, length=20, mom=5)

        # Informative for current pair on inf_timeframe
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        inf_macd = ta.MACD(informative, fastperiod=12, slowperiod=26, signalperiod=9)
        informative['macd'] = inf_macd['macd']
        informative['macdsignal'] = inf_macd['macdsignal']
        informative['macdhist'] = inf_macd['macdhist']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # Persist a buy signal for existing trades to make use of ignore_roi_if_buy_signal = True
        # when this buy signal is not present a sell can happen according to the defined ROI table
        if trade_data['active_trade']:
            conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * (dataframe['rmi-slow'] / 105)))
            conditions.append(dataframe['rmi-slow'] >= 50)

        # Normal buy triggers that apply to new trades we want to enter
        else:
            # If the stake is BTC or ETH apply an additional condition
            if self.config['stake_currency'] in ('BTC', 'ETH'):
                conditions.append(
                    (dataframe[f"{self.config['stake_currency']}_rmi-slow"] < 50) |
                    (dataframe[f"{self.custom_fiat}_rmi-slow"] > 40)
                )

            conditions.append(
                # informative timeframe conditions
                (dataframe[f"macd_{self.inf_timeframe}"] > dataframe[f"macdsignal_{self.inf_timeframe}"]) &

                # default timeframe conditions
                (dataframe['rmi-up-trend'] == 1) &
                (dataframe['vdy-up-trend'] == 1) &
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['open'] > dataframe['kama-fast']) &
                (dataframe['close'] > dataframe['kama-slow'])
            )

        # applies to both new buys and persisting buy signal
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []
        
        # In this strategy all sells for profit happen according to ROI
        # This sell signal is designed as a "dynamic stoploss"

        # if we are in an active trade for this pair
        if trade_data['active_trade']:
            # if we are at a loss, consider what the trend looks and preempt the stoploss
            conditions.append(
                (trade_data['current_profit'] < 0) &
                (trade_data['current_profit'] > self.stoploss) &  
                (dataframe['rmi-dn-trend'] == 1) &
                (qtpylib.crossed_below(dataframe['rmi-fast'], 50)) &
                (dataframe['volume'].gt(0))
            )

            # if there are other open trades in addition to this one, consider the average profit 
            # across them all (not including this one), don't sell if entire market is down big and wait for recovery
            if trade_data['other_trades']:
                if trade_data['free_slots'] == 0:
                    conditions.append(trade_data['avg_other_profit'] >= -0.03)
                else:
                    conditions.append(trade_data['avg_other_profit'] >= -0.01)

        # the bot comes through this loop even when there isn't an open trade to sell
        # so we pass an impossible condiiton here because we don't want a sell signal 
        # clogging up the charts and not having one leads the bot to crash
        else:
            conditions.append(dataframe['volume'].lt(0))
                           
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        
        return dataframe

    """
    Super Legit Custom Methods
    """
    # Populate trades_data from the database
    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = False

        # active trade stuff only really works in live and dry, not backtest
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
            # future reference, for all open trades: open_trades = Trade.get_open_trades()
            other_trades = Trade.get_trades([Trade.pair != pair, Trade.is_open.is_(True),]).all()

            if other_trades:
                trade_data['other_trades'] = True
                total_other_profit = tuple(trade.calc_profit_ratio(self.get_current_price(trade.pair, False)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(total_other_profit) 
            else:
                trade_data['avg_other_profit'] = 0

            # get the number of free trade slots, storing in every pairs dict due to laziness
            trade_data['free_slots'] = max(0, self.config['max_open_trades'] - (len(active_trade) + len(other_trades)))

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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > rate * 1.01:
            return False
        return True


"""
Sub-strategy overrides
"""
# Sub-strategy with parameters specific to BTC stake
class Schism_BTC(Schism):

    timeframe = '5m'
    inf_timeframe = '1h'

    # ROI table:
    minimal_roi = {
        "0": 0.05,
        "10": 0.025,
        "20": 0.015,
        "30": 0.01
    }

# Sub-strategy with parameters specific to ETH stake
class Schism_ETH(Schism):

    timeframe = '5m'
    inf_timeframe = '1h'

    # ROI table:
    minimal_roi = {
        "0": 0.05,
        "10": 0.025,
        "20": 0.015,
        "30": 0.01
    }
