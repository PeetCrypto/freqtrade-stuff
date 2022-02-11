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
from statistics import mean
from cachetools import TTLCache

# Get rid of pandas warnings during backtesting
import pandas as pd  
pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import custom_indicators as cta

"""
FEATURES:
    - Dynamic ROI which looks at an upward trend and positive profit cutoff to extend roi table and hold on longer during trends.
        - Now fully working in backtest!
    - Custom Stoploss
    - Dynamic informative indicators based on certain stake currences and whitelist contents.
    - Ability to provide custom buy/sell parameters on a per-pair or group of pairs basis (may extend this to ROI and/or stoploss settings)
    - Custom indicator file to keep primary strategy clean(ish).
    - Child strategies for stake specific settings.

STRATEGY NOTES:
    - If trading on a stablecoin or fiat stake (such as USD, EUR, USDT, etc.) is *highly recommended* that you remove BTC/STAKE
      from your whitelist as this strategy performs much better on alts when using BTC as an informative but does not buy any BTC
      itself.
    - It is recommended to configure protections *if/as* you will use them in live and run *some* hyperopt/backtest with
      "--enable-protections" as this strategy will hit a lot of stoplosses so the stoploss protection is helpful
      to test. *However* - this option makes hyperopt very slow, so run your initial backtest/hyperopts without this
      option. Once you settle on a baseline set of options, do some final optimizations with protections on.
    - It *might be* worthwhile to hyperopt the stoploss but we can't hyperopt any of the parameters in the custom stoploss
      so hyperopting the stoploss is only changing the initial position relative to the other settings.
    - It is *not* recommended to use freqtrades built-in trailing stop, nor to hyperopt for that.
    - It is *highly* recommended to hyperopt this with '--spaces buy' only and at least 1000 total epochs several times. There are
      a lot of variables being hyperopted and it may take a lot of epochs to find the right settings.
        
    - Example of unique buy/sell params per pair/group of pairs:

    custom_pair_params = [
        {
            'pairs': ('ABC/XYZ', 'DEF/XYZ'),
            'buy_params': {},
            'sell_params': {}
        }
    ]

TODO: 
    - Continue to hunt for a better all around buy signal.
        - Completely eliminate any trades from hitting the initial stoploss (default: -30%)
        - Prevent buys when potential for strong downward trend and not just a dip?
            - Need to reduce drawdown.
    - Tweak ROI Trend Ride
    - Further enchance and optimize custom stop loss
        - Continue to evaluate good circumstances to bail and sell vs hold on for recovery
    - Develop a PR to fully support trades database so we can go back to previous Schism methodology for buy/sell and trade data?
    - Develop a PR to fully support hyperopting the custom_stoploss space?
"""

class Solipsis3(IStrategy):

    # Recommended for USD/USDT/etc.
    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'base-mp': 48,
        'base-rmi-fast': 44,
        'base-rmi-slow': 21,
        'inf-guard': 'both',
        'inf-pct-adr-bot': 0.17436,
        'inf-pct-adr-top': 0.84712,
        'xbtc-base-rmi': 70,
        'xbtc-inf-rmi': 14,
        'xtra-base-fiat-rmi': 14,
        'xtra-base-stake-rmi': 50,
        'xtra-inf-stake-rmi': 31
    }

    sell_params = {}

    # Custom buy/sell parameters per pair
    custom_pair_params = []

    # Recommended on 5m timeframe
    minimal_roi = {
        "0": 0.05,
        "30": 0.025,
        "120": 0.01,
        "360": 0.01,
        "720": 0.005,
        "1440": 0
    }

    dynamic_roi = {
        'enabled': False,
        'profit-factor': 400,
        'rmi-start': 30,
        'rmi-end': 70,
        'grow-delay': 180,
        'grow-time': 720,
        'fallback': 'table'
    }

    stoploss = -0.30

    use_custom_stoploss = False
    custom_stop = {
        # Linear Decay Parameters
        'decay-time': 1080,      # minutes to reach end, I find it works well to match this to the final ROI value
        'decay-delay': 0,        # minutes to wait before decay starts
        'decay-start': -0.30,    # starting value: should be the same as initial stoploss
        'decay-end': -0.03,      # ending value
        # Profit and TA  
        'cur-min-diff': 0.02,    # diff between current and minimum profit to move stoploss up to min profit point
        'cur-threshold': 0,      # how far negative should current profit be before we consider moving it up based on cur/min or roc
        'cur-roc': -0.04,        # value for roc to use for dynamic bailout
        # Positive Trailing
        'pos-trail': False,      # enable trailing once positive  
        'pos-threshold': 0.005,  # trail after how far positive
        'pos-trail-dist': 0.015  # how far behind to place the trail
    }

    # Recommended
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Required
    startup_candle_count: int = 72
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USD" # Only relevant if stake is BTC or ETH
    
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
            informative_pairs += [(stake_fiat, self.inf_timeframe)]
        # if BTC/STAKE is not in whitelist, add it as an informative pair on both timeframes
        else:
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                informative_pairs += [(btc_stake, self.timeframe)]
                informative_pairs += [(btc_stake, self.inf_timeframe)]

        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}

        # Base timeframe indicators
        dataframe['rmi-slow'] = cta.RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = cta.RMI(dataframe, length=8, mom=4)

        # Indicators for ROI and Custom Stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=24)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)

        # Momentum Pinball: https://www.tradingview.com/script/fBpVB1ez-Momentum-Pinball-Indicator/
        dataframe['roc-mp'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['mp']  = ta.RSI(dataframe['roc-mp'], timeperiod=6)

        # Trends, Peaks and Crosses
        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)
        dataframe['rmi-max'] = dataframe['rmi-slow'].rolling(10, min_periods=1).max()

        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        
        # Get the "average day range" between the 1d high and 3d low to set up guards
        informative['1d_high'] = informative['close'].rolling(24).max()
        informative['3d_low'] = informative['close'].rolling(72).min()
        informative['adr'] = informative['1d_high'] - informative['3d_low']

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
            dataframe[f"{fiat}_rmi"] = cta.RMI(coin_fiat_tf, length=21, mom=5)

            # Informative STAKE/FIAT e.g. BTC/USD - Base Timeframe
            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            dataframe[f"{stake}_rmi"] = cta.RMI(stake_fiat_tf, length=21, mom=5)

            # Informative STAKE/FIAT e.g. BTC/USD - Informative Timeframe
            stake_fiat_inf_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.inf_timeframe)
            stake_fiat_inf_tf[f"{stake}_rmi"] = cta.RMI(stake_fiat_inf_tf, length=48, mom=5)
            dataframe = merge_informative_pair(dataframe, stake_fiat_inf_tf, self.timeframe, self.inf_timeframe, ffill=True)

        # Informatives for BTC/STAKE if not in whitelist
        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                # BTC/STAKE - Base Timeframe
                btc_stake_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.timeframe)
                dataframe['BTC_rmi'] = cta.RMI(btc_stake_tf, length=14, mom=3)

                # BTC/STAKE - Informative Timeframe
                btc_stake_inf_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.inf_timeframe)
                btc_stake_inf_tf['BTC_rmi'] = cta.RMI(btc_stake_inf_tf, length=48, mom=5)
                dataframe = merge_informative_pair(dataframe, btc_stake_inf_tf, self.timeframe, self.inf_timeframe, ffill=True)

        # Slam some indicators into the trade_info dict so we can dynamic roi and custom stoploss in backtest
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['roc'] = dataframe[['date', 'roc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['atr'] = dataframe[['date', 'atr']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-slow'] = dataframe[['date', 'rmi-slow']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend'] = dataframe[['date', 'rmi-up-trend']].copy().set_index('date')

        return dataframe

    """
    Buy Signal
    """ 
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.get_pair_params(metadata['pair'], 'buy')
        conditions = []

        # Primary guards on informative timeframe to make sure we don't trade when market is peaked or bottomed out
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

        # Base Timeframe
        conditions.append(
            (dataframe['rmi-dn-trend'] == 1) &
            (dataframe['rmi-slow'] >= params['base-rmi-slow']) &
            (dataframe['rmi-fast'] <= params['base-rmi-fast']) &
            (dataframe['mp'] <= params['base-mp'])
        )

        # Extra conditions for */BTC and */ETH stakes on additional informative pairs
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            conditions.append(
                (dataframe[f"{self.config['stake_currency']}_rmi"] < params['xtra-base-stake-rmi']) | 
                (dataframe[f"{self.custom_fiat}_rmi"] > params['xtra-base-fiat-rmi'])
            )
            conditions.append(dataframe[f"{self.config['stake_currency']}_rmi_{self.inf_timeframe}"] > params['xtra-inf-stake-rmi'])
        # Extra conditions for BTC/STAKE if not in whitelist
        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                conditions.append(
                    (dataframe['BTC_rmi'] < params['xbtc-base-rmi']) &
                    (dataframe[f"BTC_rmi_{self.inf_timeframe}"] > params['xbtc-inf-rmi'])
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
        cs = self.custom_stop
        
        trade_dur: int = (current_time - trade.open_date).total_seconds() // 60
        min_profit = trade.calc_profit_ratio(trade.min_rate)
        max_profit = trade.calc_profit_ratio(trade.max_rate)
        profit_diff = current_profit - min_profit

        decay_stoploss = cta.linear_growth(cs['decay-start'], cs['decay-end'], cs['decay-delay'], cs['decay-time'], trade_dur)

        # enable stoploss in positive profits after threshold to trail as specifed distance
        if cs['pos-trail'] == True:
            if current_profit > cs['pos-threshold']:
                return current_profit - cs['pos-trail-dist']

        # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
        if self.dp and self.dp.runmode.value in ('backtest', 'hyperopt'):
            roc = self.custom_trade_info[trade.pair]['roc'].loc[current_time]['roc']
            atr = self.custom_trade_info[trade.pair]['atr'].loc[current_time]['atr']
            rmi_slow = self.custom_trade_info[trade.pair]['rmi-slow'].loc[current_time]['rmi-slow']
            rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']
        # Otherwise get it out of the dataframe
        else:
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            roc = dataframe['roc'].iat[-1]
            atr = dataframe['atr'].iat[-1]
            rmi_slow = dataframe['rmi-slow'].iat[-1]
            rmi_trend = dataframe['rmi-up-trend'].iat[-1]

        if current_profit < cs['cur-threshold']:
            # Dynamic bailout based on rate of change
            if (roc/100) < cs['cur-roc']:
                stoploss = (current_rate - atr)/current_rate
                return stoploss - 1

            # if we might be on a rebound, move the stoploss to the low point or keep it where it was
            if (current_profit > min_profit) or roc > 0:
                if profit_diff > cs['cur-min-diff']:
                    return min_profit
                return -1
        
        return decay_stoploss


    """
    Freqtrade ROI Overload for dynamic trend based ROI functionality
    """
    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:

        dynamic_roi = self.dynamic_roi
        minimal_roi = self.minimal_roi

        print(f"In the ROI place for pair: {trade.pair}")

        if not dynamic_roi or not minimal_roi:
            return None, None

        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
            if self.dp and self.dp.runmode.value in ('backtest', 'hyperopt'):
                roc = self.custom_trade_info[trade.pair]['roc'].loc[current_time]['roc']
                atr = self.custom_trade_info[trade.pair]['atr'].loc[current_time]['atr']
                rmi_slow = self.custom_trade_info[trade.pair]['rmi-slow'].loc[current_time]['rmi-slow']
                rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']
            # Otherwise get it out of the dataframe
            else:
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                roc = dataframe['roc'].iat[-1]
                atr = dataframe['atr'].iat[-1]
                rmi_slow = dataframe['rmi-slow'].iat[-1]
                rmi_trend = dataframe['rmi-up-trend'].iat[-1]

            d = dynamic_roi
            profit_factor = (1 - (rmi_slow / d['profit-factor']))
            rmi_grow = cta.linear_growth(d['rmi-start'], d['rmi-end'], d['grow-delay'], d['grow-time'], trade_dur)
            max_profit = trade.calc_profit_ratio(trade.max_rate)
            open_rate = trade.open_rate

            atr_roi = max(0, ((open_rate + atr) / open_rate) - 1)
            roc_roi = max(0, (roc/100))

            # If we observe a strong upward trend and our current profit is above the max (multiplied by a factor), hold
            if (current_profit > (max_profit * profit_factor)) and (rmi_trend == 1) and (rmi_slow > rmi_grow):
                min_roi = 100
            # Otherwise fallback something else as specified
            else:
                if d['fallback'] == 'atr':
                    min_roi = atr_roi
                if d['fallback'] == 'roc':
                    min_roi = roc_roi
                else:
                    _, min_roi = self.min_roi_reached_entry(trade_dur)
        else:
            return self.min_roi_reached_entry(trade_dur)

        return trade_dur, min_roi

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:  
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.dynamic_roi and 'enabled' in self.dynamic_roi and self.dynamic_roi['enabled']:
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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        if current_price > rate * 1.01:
            return False
        return True


    """
    Custom Methods
    Stripped the active_trade and current_price stuff out of here when we moved to dynamic_roi and custom_stoploss
    """
    def get_pair_params(self, pair: str, side: str) -> Dict:
        buy_params = self.buy_params
        sell_params = self.sell_params
  
        if self.custom_pair_params:
            custom_params = next(item for item in self.custom_pair_params if pair in item['pairs'])
            if custom_params['buy_params']:
                buy_params = custom_params['buy_params']
            if custom_params['sell_params']:
                sell_params = custom_params['sell_params']

        if side == 'sell':
            return sell_params

        return buy_params

# Sub-strategy with parameters specific to BTC stake
class Solipsis_BTC(Solipsis3):

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
class Solipsis_ETH(Solipsis3):

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