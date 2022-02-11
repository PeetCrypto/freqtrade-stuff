# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import math
import numpy as np
import pandas as pd
from typing import Dict, List
from functools import reduce
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt


def nz(dataframe, src) :
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index 
    
    copy['nz'] = copy[f'{src}'].fillna(0)
    return copy['nz']

    
def hilbertTransform(dataframe, src):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    
    copy['hilbert'] = 0.0962 * copy[f'{src}'] + 0.5769 * copy[f'{src}'].shift(2) - 0.5769 * copy[f'{src}'].shift(4) - 0.0962 * copy[f'{src}'].shift(6)
    return copy['hilbert']

def computeComponent(dataframe, src, mesaPeriodMult):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    
    copy['computeComponent'] = hilbertTransform(dataframe, src) * mesaPeriodMult
    return copy['computeComponent']

def smoothComponent(dataframe, src):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    
    copy['smoothed'] = 0.2 * copy[f'{src}'] + 0.8 * copy[f'{src}'].shift(1)
    
    return copy['smoothed']
    

def computeAlpha(dataframe, src, fastLimit, slowLimit):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    
    copy['mesaPeriod'] = 0.0
    mesaPeriodMult = 0.075 * nz(copy, 'mesaPeriod').shift(1) + 0.54
    
    copy['smooth'] = 0.0
    copy['smooth'] = (4 * copy[f'{src}'] + 3 * nz(dataframe, src).shift(1) + 2 * nz(dataframe, src).shift(2) + nz(dataframe, src).shift(3)) / 10

    copy['detrender'] = computeComponent(copy, 'smooth', mesaPeriodMult)
    
    # Compute InPhase and Quadrature components
    copy['I1'] = nz(copy, 'detrender').shift(3)
    copy['Q1'] = computeComponent(copy, 'detrender', mesaPeriodMult)
    
    # Advance the phase of I1 and Q1 by 90 degrees
    jI = computeComponent(copy, 'I1', mesaPeriodMult)
    jQ = computeComponent(copy, 'Q1', mesaPeriodMult)
       
    # Phasor addition for 3 bar averaging
    copy['I2'] = copy['I1'] - jQ
    copy['Q2'] = copy['Q1'] + jI
    
    # Smooth the I and Q components before applying the discriminator
    copy['I22'] = smoothComponent(copy, 'I2')
    copy['Q22'] = smoothComponent(copy, 'Q2')
    
    # Homodyne Discriminator
    copy['Re'] = copy['I22'] * copy['I22'].shift(1) + copy['Q22'] * copy['Q22'].shift(1)
    copy['Im'] = copy['I22'] * copy['Q22'].shift(1) - copy['Q22'] * copy['I22'].shift(1)
    
    copy['Re_use'] = smoothComponent(copy, 'Re')
    copy['Im_use'] = smoothComponent(copy, 'Im')

    copy['mesaPeriod_u1'] = np.where((copy['Re_use'] != 0) & (copy['Im_use'] != 0), 2 * math.pi / np.arctan(copy['Im_use'] /  copy['Re_use']), np.nan)
    
    copy['mesaPeriod_u2'] = np.where(copy['mesaPeriod_u1'] > 1.5 * np.where(copy['mesaPeriod_u1'].shift(1) == np.nan, copy['mesaPeriod_u1'], copy['mesaPeriod_u1'].shift(1)), 1.5 * np.where(copy['mesaPeriod_u1'].shift(1) == np.nan, copy['mesaPeriod_u1'], copy['mesaPeriod_u1'].shift(1)), copy['mesaPeriod_u1'])
    copy['mesaPeriod_u3'] = np.where(copy['mesaPeriod_u2'] < 0.67 * np.where(copy['mesaPeriod_u2'].shift(1) == np.nan, copy['mesaPeriod_u2'], copy['mesaPeriod_u2'].shift(1)), 0.67 * np.where(copy['mesaPeriod_u2'].shift(1) == np.nan, copy['mesaPeriod_u2'], copy['mesaPeriod_u2'].shift(1)), copy['mesaPeriod_u2'])
    copy['mesaPeriod_u4'] = np.where(copy['mesaPeriod_u3'] < 6, 6, copy['mesaPeriod_u3'])
    copy['mesaPeriod_u5'] = np.where(copy['mesaPeriod_u4'] > 50, 50, copy['mesaPeriod_u4'])

    copy['mesaPeriod'] = smoothComponent(copy, 'mesaPeriod_u5')
    
    copy['phase'] = np.where (copy['I1'] != 0, (180 / math.pi) * np.arctan(copy['Q1'] / copy['I1']), 0.0)
    copy['deltaPhase'] = nz(copy, 'phase').shift(1) - copy['phase']
    
    deltaphase_use = np.where(copy['deltaPhase'] < 1, 1, copy['deltaPhase'])

    copy['alpha'] = fastLimit / copy['deltaPhase']
    
    copy['alpha_use'] = np.where(copy['alpha'] < slowLimit, slowLimit, copy['alpha'])
    

    return copy['alpha_use'] , copy['alpha_use']/2

def abs_change(dataframe, src, length):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
     
    copy['abs_change'] = (copy[f'{src}'] - copy[f'{src}'].shift(length)).abs()
    return copy['abs_change']
    
def calcul_er(dataframe, src, length):
    er = abs_change(dataframe, src, length) / (abs_change(dataframe, src, 1).rolling(length).sum())
    
    return er

def mama_fama_kama(dataframe, src, length):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    
    copy['er'] =  calcul_er(dataframe, src, length)
    copy['erb'] = 0.1 * copy['er']
    
    copy['a'], copy['b'],  = computeAlpha(dataframe, src, copy['er'], copy['erb'])
    
    copy['mama'] = 0.0
    copy['mama'] = copy['a'] * copy[f'{src}'] + (1 - copy['a']) * copy['mama'].shift(1)

    copy['fama'] = 0.0
    copy['fama'] = copy['b'] * copy['mama'] + (1 - copy['b']) * copy['fama'].shift(1)

    alpha = ((copy['er'] * (copy['b'] - copy['a'])) + copy['a'])**2
    copy['kama'] = 0.0
    copy['kama'] = alpha * copy[f'{src}'] + (1 - alpha) * copy['kama'].shift(1)

    return copy['mama'], copy['fama'], copy['kama'], alpha, copy['er']

def debug_mama_fama_kama(dataframe, src, length):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    
    copy['er'] =  calcul_er(dataframe, src, length)
    copy['erb'] = 0.1 * copy['er']

    
    copy['a'], copy['b'] = computeAlpha(dataframe, src, copy['er'], copy['erb'])
    copy['phase'], copy['I1'], copy['Q1'], copy['mesaPeriod'], copy['detrender'], copy['smooth']  = debug_computeAlpha(dataframe, src, copy['er'], copy['erb'])
    
    copy['mama'] = 0.0
    copy['mama'] = copy['a'] * copy[f'{src}'] + (1 - copy['a']) * nz(copy, 'mama').shift(1)

    copy['fama'] = 0.0
    copy['fama'] = copy['b'] * copy['mama'] + (1 - copy['b']) * nz(copy, 'fama').shift(1)

    alpha = ((copy['er'] * (copy['b'] - copy['a'])) + copy['a'])**2
    copy['kama'] = 0.0
    copy['kama'] = alpha * copy[f'{src}'] + (1 - alpha) * nz(copy, 'kama').shift(1)

    return copy['a'], copy['b'], copy['phase'], copy['I1'], copy['Q1'], copy['mesaPeriod'], copy['detrender'], copy['smooth']

def debug_computeAlpha(dataframe, src, fastLimit, slowLimit):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    
    copy['mesaPeriod'] = 0.0
    mesaPeriodMult = 0.075 * nz(copy, 'mesaPeriod').shift(1) + 0.54
    
    copy['smooth'] = 0.0
    copy['smooth'] = (4 * copy[f'{src}'] + 3 * copy[f'{src}'].shift(1) + 2 * copy[f'{src}'].shift(2) + copy[f'{src}'].shift(3)) / 10

    copy['detrender'] = 0.0
    copy['detrender'] = computeComponent(copy, 'smooth', mesaPeriodMult)
    
    # Compute InPhase and Quadrature components
    copy['I1'] = nz(copy, 'detrender').shift(3)
    copy['Q1'] = computeComponent(copy, 'detrender', mesaPeriodMult)
    
    # Advance the phase of I1 and Q1 by 90 degrees
    jI = computeComponent(copy, 'I1', mesaPeriodMult)
    jQ = computeComponent(copy, 'Q1', mesaPeriodMult)
    
    copy['I2'] = 0.0
    copy['Q2'] = 0.0
    
    # Phasor addition for 3 bar averaging
    copy['I2'] = copy['I1'] - jQ
    copy['Q2'] = copy['Q1'] + jI
    
    # Smooth the I and Q components before applying the discriminator
    copy['I22'] = 0.2 * copy['I2'] + 0.8 * copy['I2'].shift(1)
    copy['Q22'] = 0.2 * copy['Q2'] + 0.8 * copy['Q2'].shift(1)
    
    # Homodyne Discriminator
    copy['Re'] = copy['I22'] * copy['I22'].shift(1) + copy['Q22'] * copy['Q22'].shift(1)
    copy['Im'] = copy['I22'] * copy['Q22'].shift(1) - copy['Q22'] * copy['I22'].shift(1)
    
    copy['Re_use'] = 0.2 * copy['Re'] + 0.8 * copy['Re'].shift(1)
    copy['Im_use'] = 0.2 * copy['Im'] + 0.8 * copy['Im'].shift(1)

    copy['mesaPeriod_u1'] = np.where((copy['Re_use'] != 0) & (copy['Im_use'] != 0), 2 * math.pi / np.arctan(copy['Im_use'] /  copy['Re_use']), 
                            np.where(copy['mesaPeriod'] > 1.5 * nz(copy, 'mesaPeriod').shift(1), 1.5 * nz(copy, 'mesaPeriod').shift(1), 
                                np.where(copy['mesaPeriod'] < 0.67 * nz(copy, 'mesaPeriod').shift(1), 0.67 * nz(copy, 'mesaPeriod').shift(1),
                                    np.where(copy['mesaPeriod'] < 6, 6,
                                        np.where(copy['mesaPeriod'] > 50, 50, np.nan)
                                    )
                                )
                            )
                        )
    copy['mesaPeriod'] = 0.2 * copy['mesaPeriod_u1'] + 0.8 * copy['mesaPeriod_u1'].shift(1)
    
    copy['phase'] = np.where (copy['I1'] != 0, (180 / math.pi) * np.arctan(copy['Q1'] / copy['I1']), 0.0)
    copy['deltaPhase'] = nz(copy, 'phase').shift(1) - copy['phase']
    
    deltaphase_use = np.where(copy['deltaPhase'] < 1, 1, copy['deltaPhase'])

    copy['alpha'] = fastLimit / copy['deltaPhase']
    
    copy['alpha_use'] = np.where(copy['alpha'] < slowLimit, slowLimit, copy['alpha'])

    return copy['phase'], copy['I1'], copy['Q1'], copy['mesaPeriod'], copy['detrender'], copy['smooth']
    
class Test_MAMA4(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
    "length": 20,
    "fastLimit" : 0.5,
    "slowLimit" : 0.05,
    }

    sell_params = {}

    protection_params = {}
    
    minimal_roi = {
        "0": 20
    }

    stoploss = -0.15
    
    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = False
    
    timeframe = '5m'

    
    process_only_new_candles = True
    startup_candle_count = 100
    use_custom_stoploss = False

    # Protections
    lenght = IntParameter(5, 40, default=buy_params['length'], space='buy', optimize=True)
    fastLimit = DecimalParameter(0.01, 1, default=buy_params['fastLimit'], space='buy', optimize=True)
    slowLimit = DecimalParameter(0.01, 0.1, default=buy_params['slowLimit'], space='buy', optimize=True)
    
    # Hyperopt parameters

    plot_config = {
        'main_plot':{
#            'mama':{'color': 'blue'},
#            'kama':{'color': 'purple'},
#            'I1':{'color': 'orange'},
#            'Q1':{'color': 'blue'},
#            'mesaPeriod':{'color': 'orange'},
            'detrender':{'color': 'blue'},
#            'smooth':{'color': 'grey'},
            },
#        'subplots':{
#            "MAFAKAMA": {
#                'a':{'color': 'orange'},
#                'b':{'color': 'blue'},
#                'er':{'color': 'grey'},
#                }
#        }
    }
    
    init_trailing_dict = {
    'trailing_buy_order_started': False,
    'trailing_buy_order_uplimit': 0,
    'start_trailing_price': 0,
    'buy_tag': None,
    }
  
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['mama'], dataframe['fama'], dataframe['kama'], dataframe['alpha'], dataframe['er'] = mama_fama_kama(dataframe, 'close', self.lenght.value)
        dataframe['a'], dataframe['b'], dataframe['phase'], dataframe['I1'], dataframe['Q1'], dataframe['mesaPeriod'], dataframe['detrender'], dataframe['smooth'] = debug_mama_fama_kama(dataframe, 'close', self.lenght.value)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
            
        dataframe.loc[:, 'buy_tag'] = ''
        
        buy_4_conditions = (
            qtpylib.crossed_above(dataframe['mama'], dataframe['fama'])
            )
        dataframe.loc[buy_4_conditions, 'buy_tag'] += 'MAMA > FAMA ' # + tag_gain
        conditions.append(buy_4_conditions)
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                ['buy']
            ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        
        sell_4_conditions = (
            qtpylib.crossed_above(dataframe['fama'], dataframe['mama'])
        )
        dataframe.loc[sell_4_conditions, 'exit_tag'] += 'FAMA > MAMA'
        conditions.append(sell_4_conditions)
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                ['sell']
            ] = 1
        return dataframe

