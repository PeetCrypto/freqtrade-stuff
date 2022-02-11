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

class Solipsis4(IStrategy):

    ## Buy Space Hyperopt Variables

    # Base Pair Params
    base_mp = IntParameter(10, 50, default=30, space='buy')
    base_rmi_max = IntParameter(30, 60, default=50, space='buy')
    base_rmi_min = IntParameter(0, 30, default=20, space='buy')
    base_ma_streak = IntParameter(1, 4, default=1, space='buy')
    base_rmi_streak = IntParameter(3, 8, default=3, space='buy')
    base_trigger = CategoricalParameter(['pcc', 'rmi', 'none'], default='rmi', space='buy', optimize=False)
    inf_pct_adr = DecimalParameter(0.70, 0.99, default=0.80, space='buy')
    # BTC Informative
    xbtc_guard = CategoricalParameter(['strict', 'lazy', 'none'], default='lazy', space='buy', optimize=True)
    xbtc_base_rmi = IntParameter(20, 70, default=40, space='buy')
    # BTC / ETH Stake Parameters
    xtra_base_stake_rmi = IntParameter(10, 50, default=50, space='buy')
    xtra_base_fiat_rmi = IntParameter(30, 70, default=50, space='buy')

    ## Sell Space Params are being "hijacked" for custom_stoploss and dynamic_roi

    # Dynamic ROI
    droi_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any'], default='any', space='sell', optimize=True)
    droi_pullback = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    droi_pullback_amount = DecimalParameter(0.005, 0.02, default=0.005, space='sell')
    droi_pullback_respect_table = CategoricalParameter([True, False], default=False, space='sell', optimize=True)

    # Custom Stoploss
    cstp_threshold = DecimalParameter(-0.05, 0, default=-0.03, space='sell')
    cstp_bail_how = CategoricalParameter(['roc', 'time', 'any'], default='roc', space='sell', optimize=True)
    cstp_bail_roc = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell')
    cstp_bail_time = IntParameter(720, 1440, default=720, space='sell')
    
    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'base_ma_streak': 1,
        'base_mp': 12,
        'base_rmi_max': 50,
        'base_rmi_min': 20,
        'base_rmi_streak': 3,
        'inf_pct_adr': 950,
        'xbtc_base_rmi': 20,
        'xbtc_guard': 'none',
        'xtra_base_fiat_rmi': 45,
        'xtra_base_stake_rmi': 13
    }

    sell_params = {
        'droi_pullback': True,
        'droi_pullback_amount': 0.006,
        'droi_pullback_respect_table': False,
        'droi_trend_type': 'any'
    }

    minimal_roi = {
        "0": 0.01,
        "1440": 0
    }

    # Enable or disable these as desired
    # Must be enabled when hyperopting the respective spaces
    use_dynamic_roi = True
    use_custom_stoploss = True

    stoploss = -0.10

    # Recommended
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Required
    startup_candle_count: int = 233
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USD" # Only relevant if stake is BTC or ETH
    custom_btc_inf = False # Don't change this.
    
    """
    Informative Pair Definitions
    """
    def informative_pairs(self):
        # add all whitelisted pairs on informative timeframe
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
        # if BTC/STAKE is not in whitelist, add it as an informative pair on both timeframes
        else:
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                informative_pairs += [(btc_stake, self.timeframe)]
        
        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}

        ## Base Timeframe / Pair

        dataframe['kama'] = ta.KAMA(dataframe, length=233)
    
        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

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
        informative['adr'] = informative['1d-high'] - informative['1d-low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # Other stake specific informative indicators
        # e.g if stake is BTC and current coin is XLM (pair: XLM/BTC)
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            fiat = self.custom_fiat
            coin_fiat = f"{coin}/{fiat}"
            stake_fiat = f"{stake}/{fiat}"

            # Informative COIN/FIAT e.g. XLM/USD - Base Timeframe
            coin_fiat_tf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            dataframe[f"{fiat}_rmi"] = cta.RMI(coin_fiat_tf, length=55, mom=5)

            # Informative STAKE/FIAT e.g. BTC/USD - Base Timeframe
            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            dataframe[f"{stake}_rmi"] = cta.RMI(stake_fiat_tf, length=55, mom=5)

        # Informatives for BTC/STAKE if not in whitelist
        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                self.custom_btc_inf = True
                # BTC/STAKE - Base Timeframe
                btc_stake_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.timeframe)
                dataframe['BTC_rmi'] = cta.RMI(btc_stake_tf, length=55, mom=5)
                dataframe['BTC_close'] = btc_stake_tf['close']
                dataframe['BTC_kama'] = ta.KAMA(btc_stake_tf, length=144)

        # Slam some indicators into the trade_info dict so we can dynamic roi and custom stoploss in backtest
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['sroc'] = dataframe[['date', 'sroc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['ssl-dir'] = dataframe[['date', 'ssl-dir']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend'] = dataframe[['date', 'rmi-up-trend']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['candle-up-trend'] = dataframe[['date', 'candle-up-trend']].copy().set_index('date')

        return dataframe

    """
    Buy Signal
    """ 
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Informative Timeframe Guards
        conditions.append(
            (dataframe['close'] <= dataframe[f"1d-low_{self.inf_timeframe}"] + 
            (self.inf_pct_adr.value * dataframe[f"adr_{self.inf_timeframe}"]))
        )

        # Base Timeframe Guards
        conditions.append(
            (dataframe['rmi-dn-count'] >= self.base_rmi_streak.value) &
            (dataframe['streak-bo-count'] >= self.base_ma_streak.value) &
            (dataframe['rmi'] <= self.base_rmi_max.value) &
            (dataframe['rmi'] >= self.base_rmi_min.value) &
            (dataframe['mp'] <= self.base_mp.value)        
        )

        # Base Timeframe Trigger
        if self.base_trigger.value == 'pcc':
            conditions.append(qtpylib.crossed_above(dataframe['streak-roc'], dataframe['pcc-lowerband']))

        if self.base_trigger.value == 'rmi':
            conditions.append(dataframe['rmi-up-trend'] == 1)

        # Extra conditions for */BTC and */ETH stakes on additional informative pairs
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            conditions.append(
                (dataframe[f"{self.custom_fiat}_rmi"] > self.xtra_base_fiat_rmi.value) |
                (dataframe[f"{self.config['stake_currency']}_rmi"] < self.xtra_base_stake_rmi.value)
            )
        # Extra conditions for BTC/STAKE if not in whitelist
        else:
            if self.custom_btc_inf:
                if self.xbtc_guard.value == 'strict':
                    conditions.append(
                        (
                            (dataframe['BTC_rmi'] > self.xbtc_base_rmi.value) &
                            (dataframe['BTC_close'] > dataframe['BTC_kama'])
                        )
                    )
                if self.xbtc_guard.value == 'lazy':
                    conditions.append(
                        (dataframe['close'] > dataframe['kama']) |
                        (
                            (dataframe['BTC_rmi'] > self.xbtc_base_rmi.value) &
                            (dataframe['BTC_close'] > dataframe['BTC_kama'])
                        )
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

# Sub-strategy with parameters specific to BTC stake
class Solipsis4_BTC(Solipsis4):

    timeframe = '5m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "1440": 0
    }

# Sub-strategy with parameters specific to ETH stake
class Solipsis4_ETH(Solipsis4):

    timeframe = '5m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "1440": 0
    }