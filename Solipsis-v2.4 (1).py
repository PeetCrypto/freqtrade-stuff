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

# Get rid of pandas warnings during backtesting
import pandas as pd  
pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import custom_indicators as cta
from technical.consensus import Consensus

class Solipsis(IStrategy):

    # Recommended for USD/USDT/etc.
    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'consensus-buy': 0,
        'inf-guard': 'upper',
        'inf-pct-adr-bot': 0.09086,
        'inf-pct-adr-top': 0.78614,
        'xbtc-consensus-buy': 31,
        'xtra-base-fiat-rmi': 15,
        'xtra-base-stake-rmi': 23,
        'xtra-inf-stake-rmi': 33
    }

    sell_params = {}

    # Custom buy/sell parameters per pair
    custom_pair_params = []

    # Recommended on 5m timeframe
    minimal_roi = {
        "0": 0.01,
        "360": 0.005,
        "720": 0
    }

    dynamic_roi = {
        'enabled': True,        # enable dynamic roi which uses trennds and indicators to dynamically manipulate the roi table
        'profit-factor': 400,   # factor for forumla of how far below peak profit to trigger sell
        'rmi-start': 30,        # starting value for rmi-slow to be considered a positive trend
        'rmi-end': 70,          # ending value
        'grow-delay': 180,      # delay on growth
        'grow-time': 720,       # finish time of growth
        'fallback': 'table',    # if no trend, do what? (table, roc, atr, roc-table, atr-table)
        'min-roc-atr': 0        # minimum roi value to return in roc or atr mode
    }

    stoploss = -0.30

    use_custom_stoploss = True
    custom_stop = {
        # Linear Decay Parameters
        'decay-time': 1080,      # minutes to reach end, I find it works well to match this to the final ROI value
        'decay-delay': 0,        # minutes to wait before decay starts
        'decay-start': -0.30,    # starting value: should be the same as initial stoploss
        'decay-end': -0.01,      # ending value
        # Profit and TA  
        'cur-min-diff': 0.02,    # diff between current and minimum profit to move stoploss up to min profit point
        'cur-threshold': -0.01,  # how far negative should current profit be before we consider moving it up based on cur/min or roc
        'roc-bail': -0.04,       # value for roc to use for dynamic bailout
        'con-bail': 50,          # value for consensus-sell to use for dynamic bailout
        'rmi-trend': 50,         # rmi-slow value to pause stoploss decay
        'con-trend': 50,         # consensus-buy value to pause stoploss decay
        'bail-how': 'atr',       # set the stoploss to the atr offset below current price, or immediate
        # Positive Trailing
        'pos-trail': True,       # enable trailing once positive  
        'pos-threshold': 0.005,  # trail after how far positive
        'pos-trail-dist': 0.015  # how far behind to place the trail
    }

    # Recommended
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

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
    
        c = Consensus(dataframe)

        # Overlap / MA's
        c.evaluate_tema(period=12)
        c.evaluate_ema(period=6)
        c.evaluate_ema(period=12)
        c.evaluate_sma(period=12)
        c.evaluate_ichimoku()                 # slightly slow
        #c.evaluate_hull()                    # very slow
        c.evaluate_vwma(period=20)

        # Oscillators
        c.evaluate_rsi(period=14)
        c.evaluate_stoch()
        #c.evaluate_cci()                     # slightly slow
        #c.evaluate_adx()                
        c.evaluate_macd()
        c.evaluate_momentum() 
        c.evaluate_williams()
        # c.evaluate_ultimate_oscilator()     # extremely slow
        # missing: awesome osc
        # missing: bull bear
        # missing: stoch rsi

        #c.evaluate_macd_cross_over()
        c.evaluate_osc()
        c.evaluate_cmf()
        c.evaluate_cmo()                      # slightly slow
        #c.evaluate_laguerre()                # slow

        dataframe['consensus-buy'] = c.score()['buy']
        dataframe['consensus-sell'] = c.score()['sell']

        # Indicators for ROI and Custom Stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=24)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)

        # Trend Indicators for ROI
        dataframe['rmi-slow'] = cta.RMI(dataframe, length=21, mom=5)

        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)        
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)

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
                cbtc = Consensus(btc_stake_tf)

                cbtc.evaluate_tema(period=12)
                cbtc.evaluate_ema(period=12)
                cbtc.evaluate_ema(period=24)
                cbtc.evaluate_sma(period=12)
                cbtc.evaluate_ichimoku()
                cbtc.evaluate_vwma(period=20)
                cbtc.evaluate_rsi(period=14)
                cbtc.evaluate_stoch()
                cbtc.evaluate_macd()
                cbtc.evaluate_momentum()
                cbtc.evaluate_williams()
                cbtc.evaluate_osc()
                cbtc.evaluate_cmf()
                cbtc.evaluate_cmo()
                cbtc.evaluate_laguerre()        

                dataframe['BTC_consensus-buy'] = cbtc.score()['buy']

                # BTC/STAKE - Informative Timeframe
                # btc_stake_inf_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.inf_timeframe)
                # dataframe = merge_informative_pair(dataframe, btc_stake_inf_tf, self.timeframe, self.inf_timeframe, ffill=True)

        # Slam some indicators into the trade_info dict so we can dynamic roi and custom stoploss in backtest
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            # Attempting to use a temporary holding place for dynamic roi backtest return value...
            if not 'backtest' in self.custom_trade_info:
                self.custom_trade_info['backtest'] = {}
            
            self.custom_trade_info[metadata['pair']]['roc'] = dataframe[['date', 'roc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['atr'] = dataframe[['date', 'atr']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-slow'] = dataframe[['date', 'rmi-slow']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend'] = dataframe[['date', 'rmi-up-trend']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['consensus-buy'] = dataframe[['date', 'consensus-buy']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['consensus-sell'] = dataframe[['date', 'consensus-sell']].copy().set_index('date')

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

        conditions.append(dataframe['consensus-buy'] > params['consensus-buy'])            

        # Extra conditions for */BTC and */ETH stakes on additional informative pairs
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            conditions.append(
                (dataframe[f"{self.config['stake_currency']}_rmi"] < params['xtra-base-stake-rmi']) | 
                (dataframe[f"{self.custom_fiat}_rmi"] > params['xtra-base-fiat-rmi'])
            )
            conditions.append(dataframe[f"{self.config['stake_currency']}_rmi_{self.inf_timeframe}"] < params['xtra-inf-stake-rmi'])
        # Extra conditions for BTC/STAKE if not in whitelist
        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                conditions.append(dataframe['BTC_consensus-buy'] > params['xbtc-consensus-buy'])

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
        # params = self.get_pair_params(metadata['pair'], 'sell')

        dataframe['sell'] = 0

        return dataframe

    """
    Custom Stoploss
    """ 
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        cs = self.get_pair_params(pair, 'custom_stop')
        
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        min_profit = trade.calc_profit_ratio(trade.min_rate)
        max_profit = trade.calc_profit_ratio(trade.max_rate)
        profit_diff = current_profit - min_profit

        decay_stoploss = cta.linear_growth(cs['decay-start'], cs['decay-end'], cs['decay-delay'], cs['decay-time'], trade_dur)

        # enable stoploss in positive profits after threshold to trail as specifed distance
        if cs['pos-trail'] == True:
            if current_profit > cs['pos-threshold']:
                return current_profit - cs['pos-trail-dist']

        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            roc = dataframe['roc'].iat[-1]
            atr = dataframe['atr'].iat[-1]
            rmi_slow = dataframe['rmi-slow'].iat[-1]
            consensus_buy = dataframe['consensus-buy'].iat[-1]
            consensus_sell = dataframe['consensus-sell'].iat[-1]
        # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
        else:
            roc = self.custom_trade_info[trade.pair]['roc'].loc[current_time]['roc']
            atr = self.custom_trade_info[trade.pair]['atr'].loc[current_time]['atr']
            rmi_slow = self.custom_trade_info[trade.pair]['rmi-slow'].loc[current_time]['rmi-slow']
            consensus_buy = self.custom_trade_info[trade.pair]['consensus-buy'].loc[current_time]['consensus-buy']
            consensus_sell = self.custom_trade_info[trade.pair]['consensus-sell'].loc[current_time]['consensus-sell']

        if current_profit < cs['cur-threshold']:
            # Dynamic bailout based on rate of change
            if (roc/100) <= cs['roc-bail'] or cs['con-bail'] <= consensus_sell:
                if cs['bail-how'] == 'atr':
                    return ((current_rate - atr)/current_rate) - 1
                elif cs['bail-how'] == 'immediate':
                    return current_rate
                else:
                    return decay_stoploss

        # if we might be on a rebound, move the stoploss to the low point or keep it where it was
        if (current_profit > min_profit) or roc > 0 or rmi_slow >= cs['rmi-trend'] or consensus_buy >= cs['con-trend']:
            if profit_diff > cs['cur-min-diff']:
                return min_profit
            return -1
        
        return decay_stoploss

    """
    Freqtrade ROI Overload for dynamic ROI functionality
    """
    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:

        dynamic_roi = self.get_pair_params(trade.pair, 'dynamic_roi')
        minimal_roi = self.get_pair_params(trade.pair, 'minimal_roi')

        if not dynamic_roi or not minimal_roi:
            return None, None

        _, table_roi = self.min_roi_reached_entry(trade_dur, trade.pair)

        # see if we have the data we need to do this, otherwise fall back to the standard table
        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            if self.config['runmode'].value in ('live', 'dry_run'):
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                roc = dataframe['roc'].iat[-1]
                atr = dataframe['atr'].iat[-1]
                rmi_slow = dataframe['rmi-slow'].iat[-1]
                rmi_trend = dataframe['rmi-up-trend'].iat[-1]
            # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
            else:
                roc = self.custom_trade_info[trade.pair]['roc'].loc[current_time]['roc']
                atr = self.custom_trade_info[trade.pair]['atr'].loc[current_time]['atr']
                rmi_slow = self.custom_trade_info[trade.pair]['rmi-slow'].loc[current_time]['rmi-slow']
                rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']

            d = dynamic_roi
            profit_factor = (1 - (rmi_slow / d['profit-factor']))
            rmi_grow = cta.linear_growth(d['rmi-start'], d['rmi-end'], d['grow-delay'], d['grow-time'], trade_dur)
            max_profit = trade.calc_profit_ratio(trade.max_rate)
            open_rate = trade.open_rate

            atr_roi = max(d['min-roc-atr'], ((open_rate + atr) / open_rate) - 1)
            roc_roi = max(d['min-roc-atr'], (roc/100))
            
            # atr as the fallback (if > min-roc-atr)
            if d['fallback'] == 'atr':
                min_roi = atr_roi
            # roc as the fallback (if > min-roc-atr)
            elif d['fallback'] == 'roc':
                min_roi = roc_roi
            # atr or table as the fallback (whichever is larger)
            elif d['fallback'] == 'atr-table':
                min_roi = max(table_roi, atr_roi)
            # roc or table as the fallback (whichever is larger)
            elif d['fallback'] == 'roc-table': 
                min_roi = max(table_roi, roc_roi)
            # default to table
            else:
                min_roi = table_roi

            # If we observe a strong upward trend and our current profit has not retreated from the peak by much, hold
            if (rmi_trend == 1) and (rmi_slow > rmi_grow):
                if current_profit > min_roi and (current_profit < (max_profit * profit_factor)):
                    min_roi = min_roi
                else:
                    min_roi = 100

            """
            # If we observe a strong upward trend and our current profit has not retreated from the peak by much, hold
            if (current_profit > (max_profit * profit_factor)) and (rmi_trend == 1) and (rmi_slow > rmi_grow):
                min_roi = 100
            """

        else:
            min_roi = table_roi

        # Attempting to wedge the dynamic roi value into a thing so we can trick backtesting...
        if self.config['runmode'].value not in ('live', 'dry_run'):
            # Theoretically, if backtesting uses this value, ROI was triggered so we need to trick it with a sell
            # rate other than what is on the standard ROI table...
            self.custom_trade_info['backtest']['roi'] = max(min_roi, current_profit)

        return trade_dur, min_roi

    def min_roi_reached_entry(self, trade_dur: int, pair: str = 'backtest') -> Tuple[Optional[int], Optional[float]]:
        minimal_roi = self.get_pair_params(pair, 'minimal_roi')

        roi_list = list(filter(lambda x: x <= trade_dur, minimal_roi.keys()))
        if not roi_list:
            return None, None
        roi_entry = max(roi_list)
        min_roi = minimal_roi[roi_entry]

        # Attempting to take the dynamic roi value out of a thing so we can trick backtesting...
        if self.dynamic_roi and 'enabled' in self.dynamic_roi and self.dynamic_roi['enabled']:
            if pair == 'backtest':
                min_roi = self.custom_trade_info['backtest']['roi']

        return roi_entry, min_roi

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:  
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.dynamic_roi and 'enabled' in self.dynamic_roi and self.dynamic_roi['enabled']:
            _, roi = self.min_roi_reached_dynamic(trade, current_profit, current_time, trade_dur)
        else:
            _, roi = self.min_roi_reached_entry(trade_dur, trade.pair)
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
    """
    def get_pair_params(self, pair: str, params: str) -> Dict:
        buy_params = self.buy_params
        sell_params = self.sell_params
        minimal_roi = self.minimal_roi
        custom_stop = self.custom_stop
        dynamic_roi = self.dynamic_roi
  
        if self.custom_pair_params:
            custom_params = next(item for item in self.custom_pair_params if pair in item['pairs'])
            if custom_params['buy_params']:
                buy_params = custom_params['buy_params']
            if custom_params['sell_params']:
                sell_params = custom_params['sell_params']
            if custom_params['minimal_roi']:
                custom_stop = custom_params['minimal_roi']
            if custom_params['custom_stop']:
                custom_stop = custom_params['custom_stop']
            if custom_params['dynamic_roi']:
                dynamic_roi = custom_params['dynamic_roi']
            
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

# Sub-strategy with parameters specific to BTC stake
class Solipsis_BTC(Solipsis):

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
class Solipsis_ETH(Solipsis):

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