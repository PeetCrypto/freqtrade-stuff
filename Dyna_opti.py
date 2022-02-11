import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, stoploss_from_open, IntParameter, DecimalParameter, CategoricalParameter
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

# Get rid of pandas warnings during backtesting
import pandas as pd  
pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import custom_indicators as cta

"""
Solipsis - By @werkkrew

Credits - 
@JimmyNixx for many of the ideas used throughout as well as helping me stay motivated throughout development! 
@JoeSchr for documenting and doing the legwork of getting indicators to be available in the custom_stoploss

We ask for nothing in return except that if you make changes which bring you greater success than what has been provided, you share those ideas back to us
and the rest of the community. Also, please don't nag us with a million questions and especially don't blame us if you lose a ton of money using this.

We take no responsibility for any success or failure you have using this strategy.
"""

class Dyna_opti(IStrategy):

    ## Buy Space Hyperopt Variables

    # Base Pair Params
    bbdelta_close = DecimalParameter(0.0, 0.1, default=0.025, space='buy')
    closedelta_close = DecimalParameter(0.0, 0.5, default=0.018, space='buy')
    tail_bbdelta = DecimalParameter(0.0, 1, default=0.945, space='buy')
    inf_guard = CategoricalParameter(['lower', 'upper', 'both', 'none'], default='lower', space='buy', optimize=True)
    inf_pct_adr_top = DecimalParameter(0.70, 0.99, default=0.792, space='buy')
    inf_pct_adr_bot = DecimalParameter(0.01, 0.20, default=0.172, space='buy')

    ## Sell Space Params are being "hijacked" for custom_stoploss and dynamic_roi

    # Dynamic ROI
    droi_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any'], default='any', space='sell', optimize=True)
    droi_pullback = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    droi_pullback_amount = DecimalParameter(0.005, 0.02, default=0.015, space='sell')
    droi_pullback_respect_table = CategoricalParameter([True, False], default=True, space='sell', optimize=True)

    # Custom Stoploss
    cstp_threshold = DecimalParameter(-0.15, 0, default=-0.05, space='sell')
    cstp_bail_how = CategoricalParameter(['roc', 'time', 'any'], default='time', space='sell', optimize=True)
    cstp_bail_roc = DecimalParameter(-0.05, -0.01, default=--0.018, space='sell')
    cstp_bail_time = IntParameter(720, 1440, default=961, space='sell')
    
    timeframe = '5m'
    inf_timeframe = '1h'
    
    # Custom buy/sell parameters per pair
    custom_pair_params = []

    # Buy hyperspace params:
    buy_params = {
        'bbdelta_close': 0.025,
        'closedelta_close': 0.018,
        'inf_guard': 'lower',
        'inf_pct_adr': 0.856,
        'inf_pct_adr_bot': 0.172,
        'inf_pct_adr_top': 0.792,
        'tail_bbdelta': 0.945
    }

    # Sell hyperspace params:
    sell_params = {
        'cstp_bail_how': 'time',
        'cstp_bail_roc': -0.018,
        'cstp_bail_time': 961,
        'cstp_threshold': -0.05,
        'droi_pullback': False,
        'droi_pullback_amount': 0.015,
        'droi_pullback_respect_table': True,
        'droi_trend_type': 'any'
    }

    # ROI table:
    minimal_roi = {
        "0": 0.18724,
        "20": 0.04751,
        "30": 0.02393,
        "72": 0
    }

    # Stoploss:
    stoploss = -0.28819

    # Enable or disable these as desired
    # Must be enabled when hyperopting the respective spaces
    use_dynamic_roi = True
    use_custom_stoploss = True

    # If custom_stoploss disabled
    #stoploss = -0.234

    # Recommended
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 233
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
#    custom_fiat = "USDT" # Only relevant if stake is BTC or ETH
#    custom_btc_inf = False # Don't change this.
    
    """
    Informative Pair Definitions
    """
    def informative_pairs(self):
        # add all whitelisted pairs on informative timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
               
        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}

        ## Base Timeframe / Pair

        # strategy BinHV45
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=12, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bbdelta'] = (dataframe['bb_middleband'] - dataframe['bb_lowerband']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        # Indicators for ROI and Custom Stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=24)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)
    
        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
        dataframe['rmi-slow'] = cta.RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = cta.RMI(dataframe, length=8, mom=4)
        # Momentum Pinball: https://www.tradingview.com/script/fBpVB1ez-Momentum-Pinball-Indicator/
        dataframe['roc-mp'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['mp']  = ta.RSI(dataframe['roc-mp'], timeperiod=3)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

        # Percent Change Channel: https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
        upper, mid, lower = cta.pcc(dataframe, period=40, mult=3)
        dataframe['pcc-lowerband'] = lower
        dataframe['pcc-upperband'] = upper

        lookup_idxs = dataframe.index.values - (abs(dataframe['mastreak'].values) + 1)
        valid_lookups = lookup_idxs >= 0
        dataframe['sbc'] = np.nan
        dataframe.loc[valid_lookups, 'sbc'] = dataframe['close'].to_numpy()[lookup_idxs[valid_lookups].astype(int)]

        dataframe['streak-roc'] = 100 * (dataframe['close'] - dataframe['sbc']) / dataframe['sbc']

        # Trends, Peaks and Crosses
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['close'].shift(),1,0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3,1,0)

        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3,1,0)      

        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(),1,0)
        dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()

        dataframe['streak-bo'] = np.where(dataframe['streak-roc'] < dataframe['pcc-lowerband'],1,0)
        dataframe['streak-bo-count'] = dataframe['streak-bo'].rolling(8).sum()

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown,'up','down')

        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        
        # Get the "average day range" between the 1d high and 1d low to set up guards
        informative['1d-high'] = informative['close'].rolling(24).max()
        informative['1d-low'] = informative['close'].rolling(24).min()
        informative['3d_low'] = informative['close'].rolling(72).min()
        informative['adr'] = informative['1d-high'] - informative['1d-low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)


        # Slam some indicators into the trade_info dict so we can dynamic roi and custom stoploss in backtest
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['sroc'] = dataframe[['date', 'sroc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['ssl-dir'] = dataframe[['date', 'ssl-dir']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend'] = dataframe[['date', 'rmi-up-trend']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['candle-up-trend'] = dataframe[['date', 'candle-up-trend']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['roc'] = dataframe[['date', 'roc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['atr'] = dataframe[['date', 'atr']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-slow'] = dataframe[['date', 'rmi-slow']].copy().set_index('date')

        return dataframe

    """
    Buy Signal
    """ 
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        if self.inf_guard.value == 'upper' or self.inf_guard.value == 'both':
            conditions.append(
                (dataframe['close'] <= dataframe[f"3d_low_{self.inf_timeframe}"] + 
                (self.inf_pct_adr_top.value * dataframe[f"adr_{self.inf_timeframe}"]))
            )

        if self.inf_guard.value == 'lower' or self.inf_guard.value == 'both':
            conditions.append(
                (dataframe['close'] >= dataframe[f"3d_low_{self.inf_timeframe}"] + 
                (self.inf_pct_adr_bot.value * dataframe[f"adr_{self.inf_timeframe}"]))
            )

        # strategy BinHV45
        conditions.append(
            dataframe['bb_lowerband'].shift().gt(0) &
            dataframe['bbdelta'].gt(dataframe['close'] * self.bbdelta_close.value) &
            dataframe['closedelta'].gt(dataframe['close'] * self.closedelta_close.value) &
            dataframe['tail'].lt(dataframe['bbdelta'] * self.tail_bbdelta.value) &
            dataframe['close'].lt(dataframe['bb_lowerband'].shift()) &
            dataframe['close'].le(dataframe['close'].shift())
        )

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
  
        dataframe['sell'] = 0

        return dataframe

    """
    Custom Stoploss
    """ 
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        params = self.sell_params
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            sroc = dataframe['sroc'].iat[-1]
        # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
        else:
            sroc = self.custom_trade_info[trade.pair]['sroc'].loc[current_time]['sroc']

        if current_profit < self.cstp_threshold.value:
            if self.cstp_bail_how.value == 'roc' or self.cstp_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if (sroc/100) <= self.cstp_bail_roc.value:
                    return 0.001
            if self.cstp_bail_how.value == 'time' or self.cstp_bail_how.value == 'any':
                # Dynamic bailout based on time
                if trade_dur > self.cstp_bail_time.value:
                    return 0.001
                   
        return 1

    """
    Freqtrade ROI Overload for dynamic ROI functionality
    """
    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        
        params = self.sell_params
        minimal_roi = self.minimal_roi
        _, table_roi = self.min_roi_reached_entry(trade_dur)

        # see if we have the data we need to do this, otherwise fall back to the standard table
        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            if self.config['runmode'].value in ('live', 'dry_run'):
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                rmi_trend = dataframe['rmi-up-trend'].iat[-1]
                candle_trend = dataframe['candle-up-trend'].iat[-1]
                ssl_dir = dataframe['ssl-dir'].iat[-1]
            # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
            else:
                rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']
                candle_trend = self.custom_trade_info[trade.pair]['candle-up-trend'].loc[current_time]['candle-up-trend']
                ssl_dir = self.custom_trade_info[trade.pair]['ssl-dir'].loc[current_time]['ssl-dir']

            min_roi = table_roi
            max_profit = trade.calc_profit_ratio(trade.max_rate)
            pullback_value = (max_profit - self.droi_pullback_amount.value)
            in_trend = False

            if self.droi_trend_type.value == 'rmi' or self.droi_trend_type.value == 'any':
                if rmi_trend == 1:
                    in_trend = True
            if self.droi_trend_type.value == 'ssl' or self.droi_trend_type.value == 'any':
                if ssl_dir == 'up':
                    in_trend = True
            if self.droi_trend_type.value == 'candle' or self.droi_trend_type.value == 'any':
                if candle_trend == 1:
                    in_trend = True

            # Force the ROI value high if in trend
            if (in_trend == True):
                min_roi = 100
                # If pullback is enabled, allow to sell if a pullback from peak has happened regardless of trend
                if self.droi_pullback.value == True and (current_profit < pullback_value):
                    if self.droi_pullback_respect_table.value == True:
                        min_roi = table_roi
                    else:
                        min_roi = current_profit / 2

        else:
            min_roi = table_roi

        return trade_dur, min_roi

    # Change here to allow loading of the dynamic_roi settings
    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:  
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.use_dynamic_roi:
            _, roi = self.min_roi_reached_dynamic(trade, current_profit, current_time, trade_dur)
        else:
            _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

#
    """
    Trade Timeout Overloads
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

    """
    Custom Methods
    """
    # Get parameters for various settings on a per pair or group of pairs basis
    # This function can probably be simplified dramatically
    def get_pair_params(self, pair: str, params: str) -> Dict:
        buy_params, sell_params = self.buy_params, self.sell_params
        minimal_roi, dynamic_roi = self.minimal_roi, self.sell_params
        custom_stop = self.sell_params
  
        if self.custom_pair_params:
            # custom_params = next(item for item in self.custom_pair_params if pair in item['pairs'])
            for item in self.custom_pair_params:
                if 'pairs' in item and pair in item['pairs']:
                    custom_params = item 
                    if 'buy_params' in custom_params:
                        buy_params = custom_params['buy_params']
                    if 'sell_params' in custom_params:
                        sell_params = custom_params['sell_params']
                    if 'minimal_roi' in custom_params:
                        minimal_roi = custom_params['minimal_roi']
                    if 'custom_stop' in custom_params:
                        custom_stop = custom_params['custom_stop']
                    if 'dynamic_roi' in custom_params:
                        dynamic_roi = custom_params['dynamic_roi']
                    break
            
        if params == 'buy':
            return buy_params
        if params == 'sell':
            return sell_params
        if params == 'minimal_roi':
            return minimal_roi
        if params == 'custom_stop':
            return custom_stop
        if params == 'dynamic_roi':
            return dynamic_roi

        return False

    # Get the current price from the exchange (or local cache)
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
    Stripped down version from Schism, meant only to update the price data a bit
    more frequently than the default instead of getting all sorts of trade information
    """
    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {}
        trade_data['active_trade'] = False

        # active trade stuff only works in live and dry, not backtest
        if self.config['runmode'].value in ('live', 'dry_run'):
            
            # find out if we have an open trade for this pair
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            # if so, get some information
            if active_trade:
                # get current price and update the min/max rate
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

        return trade_data