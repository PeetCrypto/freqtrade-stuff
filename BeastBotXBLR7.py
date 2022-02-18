import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from datetime import datetime, timedelta
from pandas import DataFrame, Series, concat
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce
from technical.indicators import RMI,vwmacd
import logging
import pandas_ta as pta
from numpy import where
import time
import datetime

##################################################### BeastBot7 Final Rev 5 ####################################################################
# don't need hyperopt, thanks free comunity, it have parts other strategies, without freqtrade comunity this strategy is not possible              #
# if you earn money and consider rewarding the author ETH: 0xda884c4dbe47421ba63f033db3cc2ec49d552365    BTC: 1hxiKHLaPDKTWuXhVgTZvBdPZtjC9msxi    #
####################################################################################################################################################
# hyperopt for each conditions and finally all conditions true
#  backtesting: freqtrade backtesting -c config_test.json -s BeastBotXBLR5x --timerange 20210920-20220127 --breakdown day -v --enable-protections
#  119 days
# for each conditions
#           Trades |    Win Draw Loss |   Avg profit |      Profit |    Avg duration |    Max Drawdown
#   Con1        12 |     11    0    1 |        2.63% |    (15.82%) | 0 days 00:15:00 |        (8.27%) |
#   Con2        16 |     15    0    1 |        2.47% |    (19.77%) | 0 days 01:19:00 |        (5.15%) |
#   Con3        12 |     12    0    0 |        2.85% |    (17.11%) | 0 days 00:31:00 |             -- |
#   Con4        18 |     16    0    2 |        2.29% |    (20.61%) | 0 days 03:11:00 |        (5.23%) |
#   Con6        11 |     10    0    1 |        2.73% |    (15.05%) | 0 days 03:15:00 |        (0.75%) |
#   Con7         1 |      1    0    0 |        9.37% |     (4.69%) | 0 days 00:30:00 |             -- |
#   Con8        13 |     12    0    1 |        2.31% |    (15.06%) | 0 days 05:19:00 |        (5.09%) |
#   Con9         7 |      7    0    0 |        2.62% |     (9.17%) | 0 days 00:19:00 |             -- |
#   con10       26 |     19    0    7 |        1.03% |    (13.41%) | 0 days 02:39:00 |        (3.76%)
# all conditions true 
logger = logging.getLogger(__name__)

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc
def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

class BeastBotXBLR7(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '5m'
    inf_1h = '1h'
    info_timeframe_1d = "1d"
    has_BTC_info_tf = True

    # Buy hyperspace params:
    buy_params = {
        "buy_bb_delta": 0.025,
        "buy_bb_factor": 0.996,
        "buy_bb_width": 0.115,
        "buy_c10_1": -96.1,
        "buy_c10_2": -0.95,
        "buy_c6_1": 0.2,
        "buy_c6_2": 0.05,
        "buy_c6_3": 0.007,
        "buy_c6_4": 0.017,
        "buy_c6_5": 0.313,
        "buy_c7_1": 1.05,
        "buy_c7_2": 0.96,
        "buy_c7_3": -85,
        "buy_c7_4": -84,
        "buy_c7_5": 75.5,
        "buy_cci": -134,
        "buy_cci_length": 38,
        "buy_closedelta": 14.098,
        "buy_rmi": 49,
        "buy_rmi_length": 18,
        "buy_srsi_fk": 45,
        "buy_c2_1": 0.02,  # value loaded from strategy
        "buy_c2_2": 0.991,  # value loaded from strategy
        "buy_c2_3": -0.7,  # value loaded from strategy
        "buy_c9_1": 40.0,  # value loaded from strategy
        "buy_c9_2": -69.0,  # value loaded from strategy
        "buy_c9_3": -67.9,  # value loaded from strategy
        "buy_c9_4": 42.3,  # value loaded from strategy
        "buy_c9_5": 32.0,  # value loaded from strategy
        "buy_c9_6": 85.7,  # value loaded from strategy
        "buy_c9_7": -81.9,  # value loaded from strategy
        "buy_con1_enable": True,  # value loaded from strategy
        "buy_con2_enable": True,  # value loaded from strategy
        "buy_con3_1": 0.021,  # value loaded from strategy
        "buy_con3_2": 0.981,  # value loaded from strategy
        "buy_con3_3": 0.973,  # value loaded from strategy
        "buy_con3_4": -0.88,  # value loaded from strategy
        "buy_con3_enable": True,  # value loaded from strategy
        "buy_con4_enable": True,  # value loaded from strategy
        "buy_con6_enable": True,  # value loaded from strategy
        "buy_condition_10_enable": True,  # value loaded from strategy
        "buy_condition_7_enable": True,  # value loaded from strategy
        "buy_condition_8_enable": True,  # value loaded from strategy
        "buy_condition_9_enable": True,  # value loaded from strategy
        "buy_dip_threshold_5": 0.05,  # value loaded from strategy
        "buy_dip_threshold_6": 0.2,  # value loaded from strategy
        "buy_dip_threshold_7": 0.4,  # value loaded from strategy
        "buy_dip_threshold_8": 0.5,  # value loaded from strategy
        "buy_macd_41": 0.09,  # value loaded from strategy
        "buy_mfi_1": 29.8,  # value loaded from strategy
        "buy_min_inc_1": 0.025,  # value loaded from strategy
        "buy_pump_pull_threshold_1": 1.75,  # value loaded from strategy
        "buy_pump_threshold_1": 0.5,  # value loaded from strategy
        "buy_rsi_1": 39.8,  # value loaded from strategy
        "buy_rsi_1h_42": 31.1,  # value loaded from strategy
        "buy_rsi_1h_max_1": 73.8,  # value loaded from strategy
        "buy_rsi_1h_min_1": 36.2,  # value loaded from strategy
        "buy_volume_drop_41": 1.7,  # value loaded from strategy
        "buy_volume_pump_41": 0.2,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_bb_relative_8": 1.1,  # value loaded from strategy
        "sell_condition_1_enable": True,  # value loaded from strategy
        "sell_condition_2_enable": True,  # value loaded from strategy
        "sell_condition_3_enable": True,  # value loaded from strategy
        "sell_condition_4_enable": True,  # value loaded from strategy
        "sell_condition_5_enable": True,  # value loaded from strategy
        "sell_condition_6_enable": True,  # value loaded from strategy
        "sell_condition_7_enable": True,  # value loaded from strategy
        "sell_condition_8_enable": True,  # value loaded from strategy
        "sell_custom_dec_profit_1": 0.05,  # value loaded from strategy
        "sell_custom_dec_profit_2": 0.07,  # value loaded from strategy
        "sell_custom_profit_0": 0.01,  # value loaded from strategy
        "sell_custom_profit_1": 0.03,  # value loaded from strategy
        "sell_custom_profit_2": 0.05,  # value loaded from strategy
        "sell_custom_profit_3": 0.08,  # value loaded from strategy
        "sell_custom_profit_4": 0.25,  # value loaded from strategy
        "sell_custom_profit_under_rel_1": 0.024,  # value loaded from strategy
        "sell_custom_profit_under_rsi_diff_1": 4.4,  # value loaded from strategy
        "sell_custom_rsi_0": 33.0,  # value loaded from strategy
        "sell_custom_rsi_1": 38.0,  # value loaded from strategy
        "sell_custom_rsi_2": 43.0,  # value loaded from strategy
        "sell_custom_rsi_3": 48.0,  # value loaded from strategy
        "sell_custom_rsi_4": 50.0,  # value loaded from strategy
        "sell_custom_stoploss_under_rel_1": 0.004,  # value loaded from strategy
        "sell_custom_stoploss_under_rsi_diff_1": 8.0,  # value loaded from strategy
        "sell_custom_under_profit_1": 0.02,  # value loaded from strategy
        "sell_custom_under_profit_2": 0.04,  # value loaded from strategy
        "sell_custom_under_profit_3": 0.6,  # value loaded from strategy
        "sell_custom_under_rsi_1": 56.0,  # value loaded from strategy
        "sell_custom_under_rsi_2": 60.0,  # value loaded from strategy
        "sell_custom_under_rsi_3": 62.0,  # value loaded from strategy
        "sell_dual_rsi_rsi_1h_4": 79.6,  # value loaded from strategy
        "sell_dual_rsi_rsi_4": 73.4,  # value loaded from strategy
        "sell_ema_relative_5": 0.024,  # value loaded from strategy
        "sell_profit_trendstop": 0.02,  # value loaded from strategy
        "sell_rsi_1h_7": 81.7,  # value loaded from strategy
        "sell_rsi_bb_1": 79.5,  # value loaded from strategy
        "sell_rsi_bb_2": 81,  # value loaded from strategy
        "sell_rsi_diff_5": 4.4,  # value loaded from strategy
        "sell_rsi_main_3": 82,  # value loaded from strategy
        "sell_rsi_under_6": 79.0,  # value loaded from strategy
        "sell_time_stoploss": 114,  # value loaded from strategy
        "sell_time_trendstop": 113,  # value loaded from strategy
        "sell_trail_down_1": 0.18,  # value loaded from strategy
        "sell_trail_down_2": 0.14,  # value loaded from strategy
        "sell_trail_down_3": 0.01,  # value loaded from strategy
        "sell_trail_profit_max_1": 0.46,  # value loaded from strategy
        "sell_trail_profit_max_2": 0.12,  # value loaded from strategy
        "sell_trail_profit_max_3": 0.1,  # value loaded from strategy
        "sell_trail_profit_min_1": 0.15,  # value loaded from strategy
        "sell_trail_profit_min_2": 0.01,  # value loaded from strategy
        "sell_trail_profit_min_3": 0.05,  # value loaded from strategy
    }
    minimal_roi = {
        "0": 100
    }

    # new sell
    stoploss = -0.99
    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 300
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USD" # Only relevant if stake is BTC or ETH
    

    plot_config = {
          "main_plot": {
            "ema_50_1h": {"color": "rgba(255,250,200,2.4)"},
            "bb_lowerband": {"color": "#792bbb","type": "line"},
            "bb_upperband": {"color": "#bc281d","type": "line"}
          },
          "subplots": {
            "RSI/BTC": {
              "mfi": {"color": "#e12a7c","type": "line"},
              "cci": {"color": "#794491","type": "line"},
              "ssl-dir_1h": {"color": "#2773a7","type": "line"},
              "ssl-dir": {"color": "#5379a2","type": "line"}
            }
          }
        }


    custom_trendBTC_info = {}

    if not 'trend' in custom_trendBTC_info:
        custom_trendBTC_info['trend'] = {}
    if not 'not_downtrend' in custom_trendBTC_info['trend']:
        custom_trendBTC_info['trend']['not_downtrend'] = 0
    if not 'st' in custom_trendBTC_info['trend']:
        custom_trendBTC_info['trend']['st'] = 0
    if not 'stx' in custom_trendBTC_info['trend']:
        custom_trendBTC_info['trend']['stx'] = 0


    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration": 120
            },
            {
                "method": "StoplossGuard",
                "lookback_period": 90,
                "trade_limit": 2,
                "stop_duration": 120,
                "only_per_pair": False
            },
            {
                "method": "StoplossGuard",
                "lookback_period": 90,
                "trade_limit": 1,
                "stop_duration": 120,
                "only_per_pair": True
            },
        ]


    ###########################################################################
    # Buy
    Optimize_condition = False
    buy_con1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_con2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_con3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_con4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_con6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=Optimize_condition, load=True)   

    optc1 = True
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = optc1, load=True)
    buy_rmi = IntParameter(30, 50, default=35, optimize= optc1, load=True)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = optc1, load=True)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= optc1, load=True)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= optc1, load=True)
    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.095, optimize = optc1, load=True)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.025, optimize = optc1, load=True)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = optc1, load=True)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = optc1, load=True)
 
    optc2 = False
    buy_c2_1 = DecimalParameter(0.010, 0.025, default=0.018, space='buy', decimals=3, optimize=optc2, load=True)
    buy_c2_2 = DecimalParameter(0.980, 0.995, default=0.982, space='buy', decimals=3, optimize=optc2, load=True)
    buy_c2_3 = DecimalParameter(-0.8, -0.3, default=-0.5, space='buy', decimals=1, optimize=optc2, load=True)
    
    optc3 = False
    buy_con3_1 = DecimalParameter(0.010, 0.025, default=0.017, space='buy', decimals=3, optimize=optc3, load=True)
    buy_con3_2 = DecimalParameter(0.980, 0.995, default=0.984, space='buy', decimals=3, optimize=optc3, load=True)
    buy_con3_3 = DecimalParameter(0.955, 0.975, default=0.965, space='buy', decimals=3, optimize=optc3, load=True)
    buy_con3_4 = DecimalParameter(-0.95, -0.70, default=-0.85, space='buy', decimals=2, optimize=optc3, load=True)

    optc4 = False
    buy_rsi_1h_42 = DecimalParameter(10.0, 50.0, default=15.0, space='buy', decimals=1, optimize=optc4, load=True)
    buy_macd_41 = DecimalParameter(0.01, 0.09, default=0.02, space='buy', decimals=2, optimize=optc4, load=True)
    buy_volume_pump_41 = DecimalParameter(0.1, 0.9, default=0.4, space='buy', decimals=1, optimize=optc4, load=True)
    buy_volume_drop_41 = DecimalParameter(1, 10, default=3.8, space='buy', decimals=1, optimize=optc4, load=True)

    optc6 = True
    buy_c6_2 = DecimalParameter(0.980, 0.999, default=0.985, space='buy', decimals=3, optimize=optc6, load=True)
    buy_c6_1 = DecimalParameter(0.08, 0.2, default=0.12, space='buy', decimals=2, optimize=optc6, load=True) 
    buy_c6_2 = DecimalParameter(0.02, 0.4, default=0.28, space='buy', decimals=2, optimize=optc6, load=True)
    buy_c6_3 = DecimalParameter(0.005, 0.04, default=0.031, space='buy', decimals=3, optimize=optc6, load=True) 
    buy_c6_4 = DecimalParameter(0.01, 0.03, default=0.021, space='buy', decimals=3, optimize=optc6, load=True)
    buy_c6_5 = DecimalParameter(0.2, 0.4, default=0.264, space='buy', decimals=3, optimize=optc6, load=True)
   
    optc7 = True
    buy_c7_1 = DecimalParameter(0.95, 1.10, default=1.01, space='buy', decimals=2, optimize=optc7, load=True)
    buy_c7_2 = DecimalParameter(0.95, 1.10, default=0.99, space='buy', decimals=2, optimize=optc7, load=True)
    buy_c7_3 = IntParameter(-100, -80, default=-94, space='buy', optimize= optc7, load=True)
    buy_c7_4 = IntParameter(-90, -60, default=-75, space='buy', optimize= optc7, load=True)
    buy_c7_5 = DecimalParameter(75.1, 90.1, default=80.0, space='buy',decimals=1, optimize= optc7, load=True)

    optc8 = False
    buy_min_inc_1 = DecimalParameter(0.01, 0.05, default=0.022, space='buy', decimals=3, optimize=optc8, load=True)
    buy_rsi_1h_min_1 = DecimalParameter(25.0, 40.0, default=30.0, space='buy', decimals=1, optimize=optc8, load=True)
    buy_rsi_1h_max_1 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=optc8, load=True)
    buy_rsi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=optc8, load=True)
    buy_mfi_1 = DecimalParameter(20.0, 40.0, default=26.0, space='buy', decimals=1, optimize=optc8, load=True)

    optc9 = False
    buy_c9_1 = DecimalParameter(25.0, 44.0, default=36.0, space='buy', decimals=1, optimize=optc9, load=True)
    buy_c9_2 = DecimalParameter(-80.0, -67.0, default=-75.0, space='buy', decimals=1, optimize=optc9, load=True)
    buy_c9_3 = DecimalParameter(-80.0, -67.0, default=-75.0, space='buy', decimals=1, optimize=optc9, load=True)
    buy_c9_4 = DecimalParameter(35.0, 54.0, default=46.0, space='buy', decimals=1, optimize=optc9, load=True)
    buy_c9_5 = DecimalParameter(20.0, 44.0, default=30.0, space='buy', decimals=1, optimize=optc9, load=True)
    buy_c9_6 = DecimalParameter(65.0, 94.0, default=84.0, space='buy', decimals=1, optimize=optc9, load=True)
    buy_c9_7 = DecimalParameter(-110.0, -80.0, default=-99.0, space='buy', decimals=1, optimize=optc9, load=True)
 
    optc10 = True
    buy_c10_1 = DecimalParameter(-110.0, -80.0, default=-99.0, space='buy', decimals=1, optimize=optc10, load=True)
    buy_c10_2 = DecimalParameter(-1, -0.5, default=-0.78, space='buy', decimals=2, optimize=optc10, load=True)

    buy_dip_threshold_5 = DecimalParameter(0.001, 0.05, default=0.015, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_6 = DecimalParameter(0.01, 0.2, default=0.06, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_7 = DecimalParameter(0.05, 0.4, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_8 = DecimalParameter(0.2, 0.5, default=0.4, space='buy', decimals=3, optimize=False, load=True)
   # 24 hours
    buy_pump_pull_threshold_1 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_1 = DecimalParameter(0.4, 1.0, default=0.5, space='buy', decimals=3, optimize=False, load=True)


    # Sell··································································
    sell_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)

    sell_rsi_bb_1 = DecimalParameter(60.0, 80.0, default=79.5, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_bb_2 = DecimalParameter(72.0, 90.0, default=81, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_main_3 = DecimalParameter(77.0, 90.0, default=82, space='sell', decimals=1, optimize=False, load=True)
    sell_dual_rsi_rsi_4 = DecimalParameter(72.0, 84.0, default=73.4, space='sell', decimals=1, optimize=False, load=True)
    sell_dual_rsi_rsi_1h_4 = DecimalParameter(78.0, 92.0, default=79.6, space='sell', decimals=1, optimize=False, load=True)

    sell_ema_relative_5 = DecimalParameter(0.005, 0.05, default=0.024, space='sell', optimize=False, load=True)
    sell_rsi_diff_5 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    sell_rsi_under_6 = DecimalParameter(72.0, 90.0, default=79.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1h_7 = DecimalParameter(80.0, 95.0, default=81.7, space='sell', decimals=1, optimize=False, load=True)
    sell_bb_relative_8 = DecimalParameter(1.05, 1.3, default=1.1, space='sell', decimals=3, optimize=False, load=True)
 
    optimize_sell = False
    sell_custom_profit_0 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=optimize_sell, load=True)
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=33.0, space='sell', decimals=3, optimize=optimize_sell, load=True)
    sell_custom_profit_1 = DecimalParameter(0.01, 0.1, default=0.03, space='sell', decimals=3, optimize=optimize_sell, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=38.0, space='sell', decimals=2, optimize=optimize_sell, load=True)
    sell_custom_profit_2 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=optimize_sell, load=True)
    sell_custom_rsi_2 = DecimalParameter(34.0, 50.0, default=43.0, space='sell', decimals=2, optimize=optimize_sell, load=True)
    sell_custom_profit_3 = DecimalParameter(0.06, 0.30, default=0.08, space='sell', decimals=3, optimize=optimize_sell, load=True)
    sell_custom_rsi_3 = DecimalParameter(38.0, 55.0, default=48.0, space='sell', decimals=2, optimize=optimize_sell, load=True)
    sell_custom_profit_4 = DecimalParameter(0.3, 0.6, default=0.25, space='sell', decimals=3, optimize=optimize_sell, load=True)
    sell_custom_rsi_4 = DecimalParameter(40.0, 58.0, default=50.0, space='sell', decimals=2, optimize=optimize_sell, load=True)

    optimize_sell_u = False
    sell_custom_under_profit_1 = DecimalParameter(0.01, 0.10, default=0.02, space='sell', decimals=3, optimize=optimize_sell_u, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=56.0, space='sell', decimals=1, optimize=optimize_sell_u, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.01, 0.10, default=0.04, space='sell', decimals=3, optimize=optimize_sell_u, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=60.0, space='sell', decimals=1, optimize=optimize_sell_u, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.01, 0.10, default=0.6, space='sell', decimals=3, optimize=optimize_sell_u, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=62.0, space='sell', decimals=1, optimize=optimize_sell_u, load=True)

    sell_custom_dec_profit_1 = DecimalParameter(0.01, 0.10, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_dec_profit_2 = DecimalParameter(0.05, 0.2, default=0.07, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.15, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.46, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.2, default=0.18, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.12, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.14, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_3 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.08, 0.16, default=0.1, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_3 = DecimalParameter(0.01, 0.04, default=0.01, space='sell', decimals=3, optimize=False, load=True)

    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=0.024, space='sell', optimize=False, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=0.004, space='sell', optimize=False, load=True)
    sell_custom_stoploss_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=8.0, space='sell', optimize=False, load=True)

    sell_time_stoploss = IntParameter(70, 120, default=90, space='sell', optimize=True, load=True)
    sell_time_trendstop = IntParameter(70, 120, default=90, space='sell', optimize=True, load=True)
    sell_profit_trendstop = DecimalParameter(0.009, 0.02, default=0.015, space='sell', optimize=True, load=True)
    #############################################################

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])



    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)

        if (last_candle is not None):
            if (current_profit > self.sell_custom_profit_4.value) & (last_candle['rsi'] < self.sell_custom_rsi_4.value):
                return f'sf_4( {buy_tag})'
            elif (current_profit > self.sell_custom_profit_3.value) & (last_candle['rsi'] < self.sell_custom_rsi_3.value):
                return f'sf_3( {buy_tag})'
            elif (current_profit > self.sell_custom_profit_2.value) & (last_candle['rsi'] < self.sell_custom_rsi_2.value):
                return f'sf_2( {buy_tag})'
            elif (current_profit > self.sell_custom_profit_1.value) & (last_candle['rsi'] < self.sell_custom_rsi_1.value):
                return f'sf_1( {buy_tag})'
            elif (current_profit > self.sell_custom_profit_0.value) & (last_candle['rsi'] < self.sell_custom_rsi_0.value):
                return f'sf_0( {buy_tag})'

            elif (current_profit > self.sell_custom_under_profit_1.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_1.value) & (last_candle['close'] < last_candle['ema_200']):
                return f'sf_u_1( {buy_tag})'
            elif (current_profit > self.sell_custom_under_profit_2.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_2.value) & (last_candle['close'] < last_candle['ema_200']):
                return f'sf_u_2( {buy_tag})'
            elif (current_profit > self.sell_custom_under_profit_3.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_3.value) & (last_candle['close'] < last_candle['ema_200']):
                return f'sf_u_3( {buy_tag})'

            elif (current_profit > self.sell_custom_dec_profit_1.value) & (last_candle['sma_200_dec']):
                return f'sf_d_1( {buy_tag})'
            elif (current_profit > self.sell_custom_dec_profit_2.value) & (last_candle['close'] < last_candle['ema_100']):
                return f'sf_d_2( {buy_tag})'

            elif (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (max_profit > (current_profit + self.sell_trail_down_1.value)):
                return f'sf_t_1( {buy_tag})'
            elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (max_profit > (current_profit + self.sell_trail_down_2.value)):
                return f'sf_t_2( {buy_tag})'

            elif (last_candle['close'] < last_candle['ema_200']) & (current_profit > self.sell_trail_profit_min_3.value) & (current_profit < self.sell_trail_profit_max_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)):
                return f'sf_u_t_1( {buy_tag})'

            elif (current_profit > 0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_profit_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_profit_under_rsi_diff_1.value):
                return f'sf_u_e_1( {buy_tag})'

            elif (current_profit < -0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_stoploss_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_under_rsi_diff_1.value):
                return f'stoploss ( {buy_tag})'

            elif (current_profit < -0.05) & (trade_dur > self.sell_time_stoploss.value) & (last_candle['ssl-dir'] == 'down'):
                return f'stoploss5 ( {buy_tag})'

            elif ((buy_tag in [' trend ']) & (trade_dur > self.sell_time_trendstop.value) & ((last_candle['ssl-dir'] == 'down') & (current_profit < self.sell_profit_trendstop.value))):
                return f'trend_stop'

            elif (current_profit < -0.08):
                return f'stoploss8 ( {buy_tag})'

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        return True

        
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_1h) for pair in pairs]
       # informative_pairs.extend([(pair, self.info_timeframe_1d) for pair in pairs])

        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1h))
        informative_pairs.append((btc_info_pair, self.info_timeframe_1d))

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        #informative_1h['ema_15'] = ta.EMA(informative_1h, timeperiod=15)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        #informative_1h['not_downtrend'] = ((informative_1h['close'] > informative_1h['close'].shift(2)) | (informative_1h['rsi'] > 50))
        informative_1h['r_480'] = williams_r(dataframe, period=480)
        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_1.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            self.buy_pump_pull_threshold_1.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
               
        informative_1h['cti'] = pta.cti(informative_1h["close"], length=20) 
        
        ssldown, sslup = SSLChannels_ATR(informative_1h, 14)
        informative_1h['ssl-dir'] = np.where(sslup > ssldown,'up','down')


#        informative_1h['cti'] = pta.cti(informative_1h["close"], length=20)
           
        return informative_1h

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -----------------------------------------------------------------------------------------
        if not 'trend' in self.custom_trendBTC_info:
            self.custom_trendBTC_info['trend'] = {}
        if not 'not_downtrend' in self.custom_trendBTC_info['trend']:
            self.custom_trendBTC_info['trend']['not_downtrend'] = 0
  

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['not_downtrend'] = ((dataframe['close'] > dataframe['close'].shift(2)) | (dataframe['rsi'] > 50))
        self.custom_trendBTC_info["trend"]['not_downtrend'] = {}
        self.custom_trendBTC_info["trend"]['not_downtrend'] = dataframe['not_downtrend']

      # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        # nuevo #
        
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband'])
        dataframe['bb_bottom_cross'] = qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband3']).astype('int')
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        # CCI hyperopt
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)
        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20) 

        # RMI hyperopt
        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)
        #dataframe['rmi'] = RMI(dataframe, length=8, mom=4)

        # SRSI hyperopt ?
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

         # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()

        #cols_to_norm = ['vwmacd','signal','hist'] normalize
        #dataframe[cols_to_norm] = dataframe[cols_to_norm].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_32'] = williams_r(dataframe, period=32)
        dataframe['r_64'] = williams_r(dataframe, period=64)
#        dataframe['r_480'] = williams_r(dataframe, period=480)

               # EMA 200
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['safe_dips_strict'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_5.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_6.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_7.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_8.value))

        return dataframe



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()

        """
        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        if metadata['pair'] in btc_info_pair:
            btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
            btc_info_tfx = self.info_tf_btc_indicators(btc_info_tf, metadata)
            dataframe = merge_informative_pair(dataframe, btc_info_tfx, self.timeframe, self.inf_1h, ffill=True)
            drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)
        """
        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)
        
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        ssldown, sslup = SSLChannels_ATR(dataframe, 64)
        dataframe['ssl-up'] = sslup
        dataframe['ssl-down'] = ssldown
        dataframe['ssl-dir'] = np.where(sslup > ssldown,'up','down')
        dataframe['rmi'] =  RMI(dataframe, length=24, mom=5)

        tok = time.perf_counter()
        logger.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        con1 = ( 
                self.buy_con1_enable.value &
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value) &
                ((dataframe['bb_delta'] > self.buy_bb_delta.value) & (dataframe['bb_width'] > self.buy_bb_width.value)) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)
             )

        con2= (
                self.buy_con2_enable.value &
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_c2_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_c2_2.value)) &
                (dataframe['cti_1h'] > self.buy_c2_3.value)
            ) 


        con3 = (
                self.buy_con3_enable.value & 

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_con3_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_con3_2.value)) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_con3_3.value) &
                (dataframe['cti'] < self.buy_con3_4.value)
            )

        con4 = (
                self.buy_con4_enable.value &

                (dataframe['rsi_1h'] < self.buy_rsi_1h_42.value) &
                
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_41.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                
                (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_41.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_41.value) &
                (dataframe['volume_mean_slow'] * self.buy_volume_pump_41.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            )

        con6  = (
                self.buy_con6_enable.value &  
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_c6_1.value) &
                (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_c6_2.value) &
                dataframe['bb_lowerband'].shift().gt(0) &
                dataframe['bb_delta'].gt(dataframe['close'] * self.buy_c6_3.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_c6_4.value) &
                dataframe['tail'].lt(dataframe['bb_delta'] * self.buy_c6_5.value) &
                dataframe['close'].lt(dataframe['bb_lowerband'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (dataframe['volume'] > 0) 
            )

        con7 = (
                self.buy_condition_7_enable.value &
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * self.buy_c7_1.value)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_c7_2.value)) &
                (dataframe['r_14'] < self.buy_c7_3.value) &
                (dataframe['r_64'] < self.buy_c7_4.value) &
                (dataframe['rsi_1h'] < self.buy_c7_5.value) 
            )

        con8 = (
                self.buy_condition_8_enable.value &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['sma_200'] > dataframe['sma_200'].shift(50)) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_1h']) &

                (((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_1.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_1.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_1.value) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &

                (dataframe['volume'] > 0)
            )
            
        con9 = (
                self.buy_condition_9_enable.value &
                (((dataframe['close'] - dataframe['open'].rolling(12).min()) / dataframe['open'].rolling(12).min()) > 0.032) &
                (dataframe['rsi'] < self.buy_c9_1.value) &
                (dataframe['r_14'] < self.buy_c9_2.value) &
                (dataframe['r_32'] < self.buy_c9_3.value) &
                (dataframe['mfi'] < self.buy_c9_4.value) &
                (dataframe['rsi_1h'] > self.buy_c9_5.value) &
                (dataframe['rsi_1h'] < self.buy_c9_6.value) &
                (dataframe['r_480_1h'] > self.buy_c9_7.value) 
            )

        co10 = (
                self.buy_condition_10_enable.value &
                (dataframe['close'].shift(4) < (dataframe['close'].shift(3))) &
                (dataframe['close'].shift(3) < (dataframe['close'].shift(2))) &
                (dataframe['close'].shift(2) < (dataframe['close'].shift())) &
                (dataframe['close'].shift(1) < (dataframe['close'])) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['close'] > (dataframe['open'])) &
                (dataframe['cci'].shift() < dataframe['cci']) &
                (dataframe['ssl-dir_1h'] == 'up') &
                (dataframe['cci'] < self.buy_c10_1.value) &
                (dataframe['cti'] < self.buy_c10_2.value) &
                (dataframe['volume'] > 0) 
            )


        conditions.append(con1)
        conditions.append(con2)
        conditions.append(con3)
        conditions.append(con4)
        conditions.append(con6)
        conditions.append(con7)
        conditions.append(con8)
        conditions.append(con9)
        conditions.append(co10)

        dataframe.loc[con1, 'buy_tag'] = " con1 "
        dataframe.loc[con2, 'buy_tag'] = " Andalusian  "
        dataframe.loc[con3, 'buy_tag'] = " con3 "
        dataframe.loc[con4, 'buy_tag'] = " con4 "
        dataframe.loc[con6, 'buy_tag'] = " con6 "
        dataframe.loc[con7, 'buy_tag'] = " con7 "
        dataframe.loc[con8, 'buy_tag'] = " con8 "
        dataframe.loc[con9, 'buy_tag'] = " con9 "
        dataframe.loc[co10, 'buy_tag'] = " trend "

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.sell_condition_1_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_1.value) &
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb_upperband'].shift(2)) &
                (dataframe['close'].shift(3) > dataframe['bb_upperband'].shift(3)) &
                (dataframe['close'].shift(4) > dataframe['bb_upperband'].shift(4)) &
                (dataframe['close'].shift(5) > dataframe['bb_upperband'].shift(5)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_2_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_2.value) &
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb_upperband'].shift(2)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_3_enable.value &

                (dataframe['rsi'] > self.sell_rsi_main_3.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_4_enable.value &

                (dataframe['rsi'] > self.sell_dual_rsi_rsi_4.value) &
                (dataframe['rsi_1h'] > self.sell_dual_rsi_rsi_1h_4.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_6_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['rsi'] > self.sell_rsi_under_6.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_7_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_7.value) &
                qtpylib.crossed_below(dataframe['ema_12'], dataframe['ema_26']) &
                (dataframe['volume'] > 0)
            )
        )


        """
    for i in self.ma_types:
            conditions.append(
                (
                    (dataframe['close'] > dataframe[f'{i}_offset_sell']) &
                    (dataframe['volume'] > 0)
                )
        )
    """

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe

