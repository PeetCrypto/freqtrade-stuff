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
            ha_trend = dataframe['ha_trend'].iat[-1]
            ssl_dir = dataframe['ssl-dir'].iat[-1]
        # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
        else:
            rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']
            candle_trend = self.custom_trade_info[trade.pair]['candle-up-trend'].loc[current_time]['candle-up-trend']
            ha_trend =self.custom_trade_info[trade.pair]['ha_trend'].loc[current_time]['ha_trend']
            ssl_dir = self.custom_trade_info[trade.pair]['ssl-dir'].loc[current_time]['ssl-dir']

        min_roi = table_roi
        max_profit = trade.calc_profit_ratio(trade.max_rate)
        pullback_value = (max_profit - params['droi_pullback_amount'])
        in_trend = False

        if params['droi_trend_type'] == 'rmi' or params['droi_trend_type'] =='any':
            if rmi_trend == 1:
                in_trend = True
        if params['droi_trend_type'] == 'ssl' or params['droi_trend_type'] =='any':
            if ssl_dir == 'up':
                in_trend = True
        if params['droi_trend_type'] == 'candle' or params['droi_trend_type'] =='any':
            if candle_trend == 1:
                in_trend = True

                
        # Force the ROI value high if in strong rentability
        # A strong rentability is defined by a ratio between trade_dur and current profit.
        
        #This function will be activated after few hour of trade
        trade_dur_hours = int(trade_dur // 60)
        
        #using trade_dur in min to have a smoother limit
        
        #Use this line for hyperopting profit_slope_value  // activation_profit_slope_hour can also be hyperopt with few modification aka self.activation_profit_slope_hour.value   
        #profit_limit = trade_dur * (self.profit_slope.value/1000)/60
        
        #Here we want to kick any trades with a rentability under 0.5% per hour // 12% per day, pretty high but needed to see if it works 
        profit_limit = trade_dur * (5/1000)/60
        
        #We look trade rentability only for trade that last more than 6h
        activation_profit_slope_hour = 6
        
        rentability = False
        
        if trade_dur_hours < activation_profit_slope_hour:
            rentability = True
        else :
            if current_profit > profit_limit:
                rentability = True
                
        if (in_trend == True) & (rentability == True):
            min_roi = 100
            # If pullback is enabled, allow to sell if a pullback from peak has happened regardless of trend
            if params['droi_pullback'] == True and (current_profit < pullback_value):
                if params['droi_pullback_respect_table'] == True:
                    min_roi = table_roi
                else:
                    min_roi = current_profit * 0.90
    else:
        min_roi = table_roi

    return  min_roi, trade_dur