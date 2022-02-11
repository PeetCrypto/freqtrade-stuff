
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class new_turtle(IStrategy):

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 100
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -30

    # Optimal timeframe for the strategy
    timeframe = '1d'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    

    # Optional order type mapping
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    Length = 28
    Multiplier = 3.11
    bardelay = 2
    trailingmenu = "Re-entries"#options=["Normal","Re-entries","None"]
    trailinmode = "Custom"#options=["Auto","Custom"]
    usetrail = True if trailingmenu!="None" else False
    longTrailPerc = 6.58*0.01
    shortTrailPerc = 5.76 * 0.01

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        dataframe['_avgTR'] = ta.ATR(dataframe, 1)
        dataframe.loc[0, 'avgTR'] = dataframe.loc[0, '_avgTR']
        for i in range(0, len(dataframe)):
            if(i>self.Length):
                _norm = 0.0
                _sum = 0.0
                for j in range(0,self.Length-1):
                    weight = (j - i) * j
                    _norm = _norm + weight
                    _sum = _sum + dataframe.loc[i-self.Length+1+j,'_avgTR'] * weight

                dataframe.loc[i,'avgTR'] = _sum / _norm

        dataframe['highestC'] = dataframe['high'].rolling(self.Length).max()
        dataframe['lowestC'] = dataframe['low'].rolling(self.Length).min()

        dataframe['hiLimit'] = dataframe['highestC']-dataframe['avgTR']*self.Multiplier
        dataframe['loLimit'] = dataframe['lowestC']+dataframe['avgTR']*self.Multiplier
        
        dataframe.loc[0, 'ret'] = dataframe.loc[0, 'loLimit']
        for i in range(0, len(dataframe)):
            if dataframe.loc[i,'close']>dataframe.loc[i,'hiLimit'] and \
               dataframe.loc[i,'close']>dataframe.loc[i,'loLimit'] :
               dataframe.loc[i,'ret'] = dataframe.loc[i,'hiLimit']
            else:
                if dataframe.loc[i,'close']<dataframe.loc[i,'hiLimit'] and \
                   dataframe.loc[i,'close']<dataframe.loc[i,'loLimit'] :
                   dataframe.loc[i,'ret'] = dataframe.loc[i,'loLimit']
                else:
                    if i<=1 :
                     dataframe.loc[i,'ret'] = dataframe.loc[i,'close']
                    else:
                     dataframe.loc[i,'ret'] = dataframe.loc[i-1,'ret']

        dataframe.loc[0, 'pos'] = dataframe.loc[0, 'ret']
        for i in range(0, len(dataframe)):
            if (dataframe.loc[i,'avgTR']>0):
                if dataframe.loc[i,'close']>dataframe.loc[i,'ret']:
                   dataframe.loc[i,'pos'] = 1
                else:
                    if dataframe.loc[i,'close']<dataframe.loc[i,'ret']:
                       dataframe.loc[i,'pos'] = -1
                    else:
                        if i==1 :
                         dataframe.loc[i,'pos'] = 0
                        else:
                         dataframe.loc[i,'pos'] = dataframe.loc[i-1,'pos']
            else:
                dataframe.loc[i,'pos'] = 0

        dataframe.loc[0, 'rising'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'falling'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'enterLong'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'enterShort'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'trade'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'longStopPrice'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'shortStopPrice'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'Long_exit'] = dataframe.loc[0, 'pos']
        dataframe.loc[0, 'Short_exit'] = dataframe.loc[0, 'pos']

        for i in range(1, len(dataframe)):
            dataframe.loc[i,'enterLong']=False        
            dataframe.loc[i,'enterShort']=False
            dataframe.loc[i,'Long_exit']=False        
            dataframe.loc[i,'Short_exit']=False
            if(i<self.bardelay):
                continue
            rising = True
            falling = True
            for j in range(1,self.bardelay+1):
                if(dataframe.loc[i-j+1,'close']<dataframe.loc[i-j,'close']):
                    rising = False
                if(dataframe.loc[i-j+1,'close']>dataframe.loc[i-j,'close']):
                    falling = False
            dataframe.loc[i, 'rising'] = rising
            dataframe.loc[i, 'falling'] = falling

            if dataframe.loc[i,'pos'] ==  1 and \
               (self.trailingmenu != "Normal" or (i>1 and dataframe.loc[i,'pos']!=dataframe.loc[i-1,'pos']) ) and \
               dataframe.loc[i, 'rising'] and \
               i>1+self.bardelay and dataframe.loc[i-1, 'trade']!=1 :
               dataframe.loc[i,'enterLong']=True        
            
            if dataframe.loc[i,'pos'] ==  -1 and \
               (self.trailingmenu != "Normal" or (i>1 and dataframe.loc[i,'pos']!=dataframe.loc[i-1,'pos']) ) and \
               dataframe.loc[i, 'falling'] and \
               i>1+self.bardelay and dataframe.loc[i-1, 'trade']!=-1 :
               dataframe.loc[i,'enterShort']=True      

            if dataframe.loc[i,'enterLong'] :
                dataframe.loc[i,'trade'] =1
            else:
                if dataframe.loc[i,'enterShort']:
                    dataframe.loc[i,'trade'] =-1
                else:
                    if i<=1 :
                     dataframe.loc[i,'trade'] = 0
                    else:
                     dataframe.loc[i,'trade'] = dataframe.loc[i-1,'trade']

            if(dataframe.loc[i,'trade']==1):
                stopValue = 0.0
                if(self.trailinmode== "Auto"):
                    stopValue = (dataframe.loc[i, 'high']+dataframe.loc[i, 'low'])/2 - 1 * dataframe.loc[i, 'avgTR']  
                else:
                    stopValue = dataframe.loc[i, 'close'] * (1 - self.longTrailPerc) 
                dataframe.loc[i, 'longStopPrice'] = stopValue
                if i>1:
                    if stopValue>dataframe.loc[i-1, 'longStopPrice']:
                        dataframe.loc[i, 'longStopPrice'] = stopValue
                    else:
                        dataframe.loc[i, 'longStopPrice'] = dataframe.loc[i-1, 'longStopPrice']
            else:
                dataframe.loc[i, 'longStopPrice'] = 0

            if(dataframe.loc[i,'trade']==-1):
                stopValue = 0.0
                if(self.trailinmode== "Auto"):
                    stopValue = (dataframe.loc[i, 'high']+dataframe.loc[i, 'low'])/2 + 1 * dataframe.loc[i, 'avgTR']  
                else:
                    stopValue = dataframe.loc[i, 'close'] * (1 + self.shortTrailPerc) 
                dataframe.loc[i, 'shortStopPrice'] = stopValue
                if i>1:
                    if stopValue<dataframe.loc[i-1, 'shortStopPrice']:
                        dataframe.loc[i, 'shortStopPrice'] = stopValue
                    else:
                        dataframe.loc[i, 'shortStopPrice'] = dataframe.loc[i-1, 'shortStopPrice']
            else:
                dataframe.loc[i, 'shortStopPrice'] = 999999

            dataframe.loc[i, 'Long_exit'] = False
            if(dataframe.loc[i, 'enterLong']!=True and dataframe.loc[i, 'trade']==1 and self.usetrail and \
                dataframe.loc[i, 'low']<dataframe.loc[i, 'longStopPrice'] and \
                dataframe.loc[i, 'open']>dataframe.loc[i, 'longStopPrice'] 
                ):
                dataframe.loc[i, 'Long_exit'] = True
            dataframe.loc[i, 'Short_exit'] = False
            if(dataframe.loc[i, 'enterShort']!=True and dataframe.loc[i, 'trade']==-1 and self.usetrail and \
                dataframe.loc[i, 'high']>dataframe.loc[i, 'shortStopPrice'] and \
                dataframe.loc[i, 'open']<dataframe.loc[i, 'shortStopPrice'] 
                ):
                dataframe.loc[i, 'Short_exit'] = True
            if dataframe.loc[i, 'Long_exit'] or dataframe.loc[i, 'Short_exit']:
                dataframe.loc[i, 'trade'] = 0

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with entry column , 1: long, -1:short
        """
        dataframe.loc[
            (
                dataframe['enterLong']==True
            ),
            'entry'] = 1#-1.1
        dataframe.loc[
            (
                dataframe['enterShort']==True
            ),
            'entry'] = -1#-1.1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with exit column
        """
        dataframe.loc[
            (
                ((dataframe['enterLong']==True ) |  \
                 (dataframe['enterShort']==True) |  \
                 (dataframe['Long_exit']==True) |  \
                 (dataframe['Short_exit']==True) \
                )
            ),
            'exit'] = 1#-1.
        return dataframe
