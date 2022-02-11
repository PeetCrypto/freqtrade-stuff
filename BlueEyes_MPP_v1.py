
class BlueEyes_MPP_v1(IStrategy):


    # Optimal timeframe for the strategy
    timeframe = '5m'

    # generate signals from the 1h timeframe
    informative_timeframe = '1d'

    minimal_roi = {
        "0": 10,
    }
    

    # Stoploss:
    stoploss = -0.10

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe)
                             for pair in pairs]
        if self.dp:
            for pair in pairs:
                informative_pairs += [(pair, "1d")]

        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        """
        # dataframe "1d"
        """

        dataframe1d = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1d")

        # Pivots Points
        pp = pivots_points(dataframe1d)
        dataframe1d['pivot'] = pp['pivot']
        dataframe1d['r1'] = pp['r1']
        dataframe1d['s1'] = pp['s1']
        dataframe1d['rS1'] = pp['rS1']
        # Pivots Points

        dataframe = merge_informative_pair(
            dataframe, dataframe1d, self.timeframe, "1d", ffill=True)

        """
        # dataframe normal
        """
 
        create_ichimoku(dataframe, conversion_line_period=20, 
                        displacement=88, base_line_periods=88, laggin_span=88)

        create_ichimoku(dataframe, conversion_line_period=88, 
                        displacement=444, base_line_periods=88, laggin_span=88)

        create_ichimoku(dataframe, conversion_line_period=355,
                        displacement=880, base_line_periods=175, laggin_span=175)


        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema88'] = ta.EMA(dataframe, timeperiod=88)
        dataframe['ema440'] = ta.EMA(dataframe, timeperiod=440)

        # Start Trading

        dataframe['pivots_ok'] = (
            (dataframe['close'] > dataframe['pivot_1d']) &
            (dataframe['r1_1d'] > dataframe['close']) &
            (dataframe['close'] > dataframe['ema440']) &
            (dataframe['ema88'] > dataframe['ema440']) &
            (dataframe['kijun_sen_355'] >= dataframe['tenkan_sen_355']) &
            (dataframe['close'] > dataframe['senkou_b_88'])

        ).astype('int')        

        
        dataframe['trending_over'] = (
            
            (dataframe['ema88'] > dataframe['close'])
            
        ).astype('int')

        return dataframe
        

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.slow_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['pivots_ok'] > 0)
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trending_over'] > 0)
            ), 'sell'] = 1
        return dataframe
